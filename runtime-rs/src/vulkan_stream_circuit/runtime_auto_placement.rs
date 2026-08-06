#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCandidate {
    pub device_id: String,
    pub safe_capacity_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeAutoPlacement {
    pub runtime_model: VulkanResidentRuntimeModel,
    pub residency_plan: VulkanRuntimeResidencyPlan,
    pub selected_device_ids: Vec<String>,
}

fn runtime_model_placement_signature(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Vec<(String, String)> {
    let mut signature = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.clone(), instance.device_id.clone()))
        .collect::<Vec<_>>();
    signature.sort();
    signature
}

fn hardware_profiles_for_runtime_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    profiles_by_physical_device: &BTreeMap<String, crate::HardwareProcessProfile>,
) -> Result<BTreeMap<String, crate::HardwareProcessProfile>, VulkanRuntimeResidencyPlanError> {
    runtime_model
        .placement_device_ids()
        .into_iter()
        .map(|logical_device_id| {
            profiles_by_physical_device
                .get(&logical_device_id)
                .cloned()
                .map(|profile| (logical_device_id.clone(), profile))
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "runtime placement device {logical_device_id:?} has no hardware profile",
                    ))
                })
        })
        .collect()
}

/// Solves placement and implementation selection together. Exact compiled
/// implementations establish the first capacity-safe placement. Alternatives
/// are then selected against the physical profile that will execute each
/// component, followed by an exact residency re-plan. If representation sizes
/// move a placement boundary, selection is repeated from the untouched exact
/// model at the new boundary until both decisions are stable.
#[allow(clippy::too_many_arguments)]
pub fn capacity_pack_and_select_vulkan_runtime_model(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    candidates: &[VulkanRuntimePlacementCandidate],
    profiles_by_physical_device: &BTreeMap<String, crate::HardwareProcessProfile>,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
    execution: crate::RuntimeExecutionEnvelope,
) -> Result<VulkanRuntimeAutoPlacement, VulkanRuntimeResidencyPlanError> {
    let manifest_dir = manifest_dir.as_ref();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(manifest_dir)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let initial = capacity_pack_vulkan_runtime_model(
        manifest_dir,
        runtime_model,
        &tensor_index,
        candidates,
        context_capacity_activations,
        speculative_draft_tokens,
        residency_policy,
    )?;
    let exact_model = runtime_model.clone();
    let mut exact_placed_model = initial.runtime_model;
    let maximum_iterations = exact_model
        .runtime_graph
        .instances
        .len()
        .saturating_add(candidates.len())
        .max(1);
    let mut observed_placements = BTreeSet::new();

    for _ in 0..maximum_iterations {
        let placement_signature = runtime_model_placement_signature(&exact_placed_model);
        if !observed_placements.insert(placement_signature.clone()) {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime representation and placement selection entered a placement cycle"
                    .to_string(),
            ));
        }
        let profiles = hardware_profiles_for_runtime_placement(
            &exact_placed_model,
            profiles_by_physical_device,
        )?;
        let (selected_model, _) = exact_placed_model
            .clone()
            .select_and_apply_runtime_implementations(
                manifest_dir,
                &profiles,
                execution.clone(),
            )
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        let selected_tensor_index = selected_model
            .load_runtime_tensor_index(manifest_dir)
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        let selected = capacity_pack_vulkan_runtime_model(
            manifest_dir,
            &selected_model,
            &selected_tensor_index,
            candidates,
            context_capacity_activations,
            speculative_draft_tokens,
            residency_policy,
        )?;
        let selected_signature = runtime_model_placement_signature(&selected.runtime_model);
        if selected_signature == placement_signature {
            return Ok(selected);
        }

        let selected_placement = selected_signature.into_iter().collect::<BTreeMap<_, _>>();
        let default_device_id = selected
            .selected_device_ids
            .first()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "selected runtime placement has no physical devices".to_string(),
                )
            })?;
        exact_placed_model = vulkan_runtime_model_with_component_placement(
            &exact_model,
            default_device_id,
            &selected_placement,
        )?;
    }

    Err(VulkanRuntimeResidencyPlanError(
        "runtime representation and placement selection did not converge".to_string(),
    ))
}

pub fn vulkan_runtime_maximum_device_resident_bytes(
    plan: &VulkanRuntimeDeviceResidencyPlan,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    [
        plan.parameter_residency.maximum_addressable_bytes,
        plan.resource_store.maximum_extra_device_bytes()?,
        plan.working_set.transient_state_bytes,
        plan.working_set.activation_headroom_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "maximum retained device residency")
    })
}

/// Returns the physical capacity a residency policy must prove before mount.
/// A paged store admits its fixed runtime state and one complete selector load
/// wave; its much larger virtual resource address space remains bounded by the
/// store's measured cache capacity and eviction policy. Retained and eager
/// stores must still prove their complete eventual residency.
pub fn vulkan_runtime_device_capacity_admission_bytes(
    plan: &VulkanRuntimeDeviceResidencyPlan,
    residency_policy: ResourceResidencyPolicy,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    if residency_policy != ResourceResidencyPolicy::DemandPaged {
        return vulkan_runtime_maximum_device_resident_bytes(plan);
    }
    [
        plan.initial_device_resident_bytes,
        plan.resource_store.maximum_load_wave_payload_bytes,
        plan.resource_store
            .maximum_dynamic_allocation_padding_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "demand-paged device capacity admission")
    })
}

/// Finds the smallest prefix of caller-ranked devices that can satisfy the
/// selected residency policy. Within that prefix, components remain in graph
/// order and each device receives the longest capacity-safe contiguous segment.
///
/// The tensor weights establish candidate boundaries. Every candidate is then
/// corrected and admitted using the runtime's exact residency plan, including
/// fixed adapters, transient state, boundary transport, staging headroom,
/// shared resources, and either the complete retained address space or a
/// bounded demand-paged cache with one complete atomic load wave.
#[allow(clippy::too_many_arguments)]
pub fn capacity_pack_vulkan_runtime_model(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    candidates: &[VulkanRuntimePlacementCandidate],
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeAutoPlacement, VulkanRuntimeResidencyPlanError> {
    if candidates.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime auto-placement requires candidate devices".to_string(),
        ));
    }
    let components = capacity_packed_runtime_components(
        runtime_model,
        tensor_index,
        speculative_draft_tokens > 0,
    )?;
    if components.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime auto-placement found no independently placeable signal processors"
                .to_string(),
        ));
    }
    let maximum_device_count = candidates.len().min(components.len());
    let manifest_dir = manifest_dir.as_ref();
    let mut failures = Vec::new();
    for device_count in 1..=maximum_device_count {
        let selected = &candidates[..device_count];
        match capacity_pack_vulkan_runtime_model_on_devices(
            manifest_dir,
            runtime_model,
            tensor_index,
            &components,
            selected,
            context_capacity_activations,
            speculative_draft_tokens,
            residency_policy,
        ) {
            Ok(placed) => return Ok(placed),
            Err(error) => failures.push(format!("{device_count} device(s): {error}")),
        }
    }
    if residency_policy == ResourceResidencyPolicy::DemandPaged {
        let selected = &candidates[..maximum_device_count];
        match proportional_paged_component_placement(&components, selected).and_then(
            |placement| {
                admit_fixed_vulkan_runtime_placement(
                    manifest_dir,
                    runtime_model,
                    tensor_index,
                    &placement,
                    selected,
                    context_capacity_activations,
                    speculative_draft_tokens,
                    residency_policy,
                )
            },
        ) {
            Ok(placed) => return Ok(placed),
            Err(error) => failures.push(format!(
                "{maximum_device_count} device(s) with paged virtual overcommit: {error}"
            )),
        }
    }
    Err(VulkanRuntimeResidencyPlanError(format!(
        "no capacity-packed contiguous placement can admit the model working set: {}",
        failures.join("; "),
    )))
}

/// Partitions a virtual resource set across every selected paged cache. This
/// path is used only when the complete addressable set exceeds their aggregate
/// physical capacity. It preserves graph order, gives every device a nonempty
/// contiguous segment, and chooses boundaries proportional to measured cache
/// capacity. Physical fixed-state and load-wave admission is proven separately.
fn proportional_paged_component_placement(
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
) -> Result<BTreeMap<String, String>, VulkanRuntimeResidencyPlanError> {
    if components.is_empty()
        || candidates.is_empty()
        || candidates.len() > components.len()
        || candidates
            .iter()
            .any(|candidate| candidate.device_id.is_empty() || candidate.safe_capacity_bytes == 0)
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged proportional placement requires components and no more positive-capacity devices than components"
                .to_string(),
        ));
    }
    let unique_device_count = candidates
        .iter()
        .map(|candidate| candidate.device_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let unique_component_count = components
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if unique_device_count != candidates.len()
        || unique_component_count != components.len()
        || components
            .iter()
            .any(|component| component.component_id.is_empty())
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged proportional placement requires unique nonempty component and device ids"
                .to_string(),
        ));
    }

    let total_capacity = candidates.iter().try_fold(0u128, |total, candidate| {
        total
            .checked_add(candidate.safe_capacity_bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "paged proportional device capacity overflowed".to_string(),
                )
            })
    })?;
    let mut prefix_weights = Vec::with_capacity(components.len() + 1);
    prefix_weights.push(0u128);
    for component in components {
        let next = prefix_weights
            .last()
            .copied()
            .expect("weight prefix has an origin")
            .checked_add(component.resident_weight_bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "paged proportional component weight overflowed".to_string(),
                )
            })?;
        prefix_weights.push(next);
    }
    let total_weight = *prefix_weights
        .last()
        .expect("component weight prefix is nonempty");
    let mut placement = BTreeMap::new();
    let mut cursor = 0usize;
    let mut cumulative_capacity = 0u128;
    for (device_index, candidate) in candidates.iter().enumerate() {
        if device_index + 1 == candidates.len() {
            for component in &components[cursor..] {
                placement.insert(component.component_id.clone(), candidate.device_id.clone());
            }
            break;
        }
        cumulative_capacity = cumulative_capacity
            .checked_add(candidate.safe_capacity_bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "paged proportional cumulative capacity overflowed".to_string(),
                )
            })?;
        let remaining_devices = candidates.len() - device_index - 1;
        let minimum_cut = cursor + 1;
        let maximum_cut = components.len() - remaining_devices;
        let target = if total_weight == 0 {
            (components.len() as u128)
                .checked_mul(cumulative_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "paged proportional boundary target overflowed".to_string(),
                    )
                })?
                / total_capacity
        } else {
            total_weight
                .checked_mul(cumulative_capacity)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "paged proportional boundary target overflowed".to_string(),
                    )
                })?
                / total_capacity
        };
        let cut = (minimum_cut..=maximum_cut)
            .min_by_key(|candidate_cut| {
                let position = if total_weight == 0 {
                    *candidate_cut as u128
                } else {
                    prefix_weights[*candidate_cut]
                };
                (position.abs_diff(target), std::cmp::Reverse(*candidate_cut))
            })
            .expect("a nonempty component segment remains for every device");
        for component in &components[cursor..cut] {
            placement.insert(component.component_id.clone(), candidate.device_id.clone());
        }
        cursor = cut;
    }
    Ok(placement)
}

#[allow(clippy::too_many_arguments)]
fn admit_fixed_vulkan_runtime_placement(
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    placement: &BTreeMap<String, String>,
    candidates: &[VulkanRuntimePlacementCandidate],
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeAutoPlacement, VulkanRuntimeResidencyPlanError> {
    let placed_model = vulkan_runtime_model_with_component_placement(
        runtime_model,
        &candidates[0].device_id,
        placement,
    )?;
    let residency_plan = plan_vulkan_runtime_residency(
        manifest_dir,
        &placed_model,
        tensor_index,
        context_capacity_activations,
        speculative_draft_tokens,
        residency_policy,
    )?;
    for candidate in candidates {
        let device_plan = residency_plan
            .device_plans
            .iter()
            .find(|plan| plan.device_id == candidate.device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "exact paged residency plan omitted selected device {:?}",
                    candidate.device_id,
                ))
            })?;
        let required =
            vulkan_runtime_device_capacity_admission_bytes(device_plan, residency_policy)?;
        if required > candidate.safe_capacity_bytes {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "paged segment on device {:?} needs {required} physical admission bytes but only {} are safely available",
                candidate.device_id, candidate.safe_capacity_bytes,
            )));
        }
    }
    Ok(VulkanRuntimeAutoPlacement {
        runtime_model: placed_model,
        residency_plan,
        selected_device_ids: candidates
            .iter()
            .map(|candidate| candidate.device_id.clone())
            .collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn capacity_pack_vulkan_runtime_model_on_devices(
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeAutoPlacement, VulkanRuntimeResidencyPlanError> {
    let actual_capacities = candidates
        .iter()
        .map(|candidate| (candidate.device_id.clone(), candidate.safe_capacity_bytes))
        .collect::<BTreeMap<_, _>>();
    if actual_capacities.len() != candidates.len()
        || candidates
            .iter()
            .any(|candidate| candidate.device_id.is_empty() || candidate.safe_capacity_bytes == 0)
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime auto-placement candidates require unique nonempty ids and positive capacities"
                .to_string(),
        ));
    }
    let mut effective_capacities = candidates
        .iter()
        .map(|candidate| CapacityPackedPlacementDevice {
            device_id: candidate.device_id.clone(),
            capacity_bytes: candidate.safe_capacity_bytes,
        })
        .collect::<Vec<_>>();
    let mut previous_placement = None;
    let maximum_refinements = components.len().saturating_add(candidates.len()).max(1);
    for _ in 0..maximum_refinements {
        let placement = capacity_packed_component_placement(components, &effective_capacities)
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        if previous_placement.as_ref() == Some(&placement) {
            return Err(VulkanRuntimeResidencyPlanError(
                "exact residency correction converged to an over-capacity placement".to_string(),
            ));
        }
        let placed_model = vulkan_runtime_model_with_component_placement(
            runtime_model,
            &candidates[0].device_id,
            &placement,
        )?;
        let residency_plan = plan_vulkan_runtime_residency(
            manifest_dir,
            &placed_model,
            tensor_index,
            context_capacity_activations,
            speculative_draft_tokens,
            residency_policy,
        )?;
        let mut fits = true;
        let component_weight_by_device = components.iter().try_fold(
            BTreeMap::<String, usize>::new(),
            |mut totals, component| {
                let device_id = placement.get(&component.component_id).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "capacity-packed placement omitted component {:?}",
                        component.component_id,
                    ))
                })?;
                let total = totals.entry(device_id.clone()).or_default();
                *total = checked_residency_add(
                    *total,
                    component.resident_weight_bytes,
                    "component placement weight",
                )?;
                Ok(totals)
            },
        )?;
        let mut corrected = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let device_plan = residency_plan
                .device_plans
                .iter()
                .find(|plan| plan.device_id == candidate.device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "exact residency plan omitted selected device {:?}",
                        candidate.device_id,
                    ))
                })?;
            let required = vulkan_runtime_device_capacity_admission_bytes(
                device_plan,
                residency_policy,
            )?;
            fits &= required <= candidate.safe_capacity_bytes;
            let weighted = component_weight_by_device
                .get(&candidate.device_id)
                .copied()
                .unwrap_or(0);
            let non_component_bytes = required.saturating_sub(weighted);
            corrected.push(CapacityPackedPlacementDevice {
                device_id: candidate.device_id.clone(),
                capacity_bytes: candidate
                    .safe_capacity_bytes
                    .saturating_sub(non_component_bytes),
            });
        }
        if fits {
            return Ok(VulkanRuntimeAutoPlacement {
                runtime_model: placed_model,
                residency_plan,
                selected_device_ids: candidates
                    .iter()
                    .map(|candidate| candidate.device_id.clone())
                    .collect(),
            });
        }
        if corrected.iter().any(|device| device.capacity_bytes == 0) {
            return Err(VulkanRuntimeResidencyPlanError(
                "fixed runtime residency exhausts a selected device capacity".to_string(),
            ));
        }
        previous_placement = Some(placement);
        effective_capacities = corrected;
    }
    Err(VulkanRuntimeResidencyPlanError(
        "exact residency correction did not converge".to_string(),
    ))
}

fn capacity_packed_runtime_components(
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    mount_speculative_decoders: bool,
) -> Result<Vec<CapacityPackedPlacementComponent>, VulkanRuntimeResidencyPlanError> {
    let component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.clone())
        .collect::<Vec<_>>();
    let first_component_id = component_ids.first().ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(
            "capacity-packed runtime has no signal processor".to_string(),
        )
    })?;
    let last_component_id = component_ids
        .last()
        .expect("a first signal processor implies a last signal processor");
    let mut tensors_by_component = component_ids
        .iter()
        .map(|component_id| (component_id.clone(), BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    let mut charged_tensors = BTreeSet::new();
    let mut charge_graph = |graph: &VulkanResidentPackageCircuitGraph,
                            fixed_anchor: Option<&str>|
     -> Result<(), VulkanRuntimeResidencyPlanError> {
        for component in &graph.components {
            let anchor = fixed_anchor.unwrap_or(match component.runtime_role {
                CircuitRuntimeRole::InputTransducer => first_component_id,
                CircuitRuntimeRole::SignalProcessor => &component.component_id,
                CircuitRuntimeRole::OutputTransducer
                | CircuitRuntimeRole::Sampler
                | CircuitRuntimeRole::DraftProcessor
                | CircuitRuntimeRole::DraftInputAdapter
                | CircuitRuntimeRole::DraftOutputTransducer => last_component_id,
            });
            let target = tensors_by_component.get_mut(anchor).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "capacity-packed auxiliary component {:?} resolves to unknown signal processor {anchor:?}",
                    component.component_id,
                ))
            })?;
            for tensor in component
                .params
                .refs
                .values()
                .filter_map(|parameter| parameter.tensor.as_deref())
                .collect::<BTreeSet<_>>()
            {
                if !charged_tensors.insert(tensor.to_string()) {
                    continue;
                }
                target.insert(tensor.to_string());
            }
        }
        Ok(())
    };
    charge_graph(&runtime_model.circuit_graph, None)?;
    if mount_speculative_decoders {
        for decoder in &runtime_model.package.speculative_decoders {
            charge_graph(&decoder.circuit_graph, Some(last_component_id))?;
        }
    }

    component_ids
        .into_iter()
        .map(|component_id| {
            let bytes = tensors_by_component
                .remove(&component_id)
                .expect("every signal processor was indexed")
                .into_iter()
                .try_fold(0usize, |bytes, tensor| {
                    let metadata = tensor_index.tensors.get(&tensor).ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "component {component_id:?} references tensor {tensor:?} absent from the runtime tensor index",
                        ))
                    })?;
                    let tensor_bytes = metadata.byte_count.ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "tensor {tensor:?} has no byte_count for capacity-packed placement",
                        ))
                    })?;
                    checked_residency_add(bytes, tensor_bytes, "component tensor weight")
                })?;
            Ok(CapacityPackedPlacementComponent {
                component_id,
                resident_weight_bytes: bytes,
            })
        })
        .collect()
}

fn vulkan_runtime_model_with_component_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    default_device_id: &str,
    placement: &BTreeMap<String, String>,
) -> Result<VulkanResidentRuntimeModel, VulkanRuntimeResidencyPlanError> {
    let mut placed_model = runtime_model.clone();
    let mut runtime_graph = runtime_model.runtime_graph.clone();
    runtime_graph.default_device_id = default_device_id.to_string();
    for instance in &mut runtime_graph.instances {
        instance.device_id = placement
            .get(&instance.instance_id)
            .cloned()
            .unwrap_or_else(|| default_device_id.to_string());
    }
    let source_graph = runtime_model
        .package
        .clone()
        .resolved_source_graph(PathBuf::from("."))
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let runtime_graph = attach_generation_node_devices_for_vulkan(runtime_graph, &source_graph)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let mut placed = StreamCircuitPlacementSpec::new(default_device_id.to_string());
    for instance in &runtime_graph.instances {
        if instance.device_id != default_device_id {
            placed = placed.with_component_device(&instance.instance_id, &instance.device_id);
        }
    }
    placed_model.runtime_graph = runtime_graph;
    placed_model.placement = placed;
    placed_model
        .resolved_graph(PathBuf::from("."))
        .and_then(|graph| {
            graph
                .placement_plan(&placed_model.placement)
                .map(|_| ())
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))
        })
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    Ok(placed_model)
}
