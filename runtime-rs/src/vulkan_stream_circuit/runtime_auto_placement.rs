#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCandidate {
    pub device_id: String,
    pub safe_capacity_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimePlacementCostModel {
    component_execution: BTreeMap<(String, String), (String, u64)>,
    boundary_transfer_ns: BTreeMap<(String, String, usize), u64>,
    default_graph_compatible_devices: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimePlacementBoundaryTransfer {
    source_in_prefix: bool,
    byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimePlacementBoundary {
    transfers: Vec<VulkanRuntimePlacementBoundaryTransfer>,
}

impl VulkanRuntimePlacementCostModel {
    pub fn record_default_graph_compatibility(
        &mut self,
        device_id: &str,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if device_id.is_empty()
            || !self
                .default_graph_compatible_devices
                .insert(device_id.to_string())
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime placement default-graph compatibility requires a unique nonempty device"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn record_calibration(
        &mut self,
        device_id: &str,
        target: &VulkanRuntimePlacementCalibrationTarget,
        measured_ns_per_activation: u64,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if device_id.is_empty() || target.signature_id.is_empty() || measured_ns_per_activation == 0
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime placement cost requires a device, signature, and positive execution cost"
                    .to_string(),
            ));
        }
        for component_id in &target.component_ids {
            if component_id.is_empty()
                || self
                    .component_execution
                    .insert(
                        (device_id.to_string(), component_id.clone()),
                        (target.signature_id.clone(), measured_ns_per_activation),
                    )
                    .is_some()
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement cost contains a duplicate or empty component for device {device_id:?}",
                )));
            }
        }
        Ok(())
    }

    pub fn record_boundary_transfer_cost(
        &mut self,
        source_device_id: &str,
        target_device_id: &str,
        byte_count: usize,
        measured_ns: u64,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        if source_device_id.is_empty()
            || target_device_id.is_empty()
            || source_device_id == target_device_id
            || byte_count == 0
            || measured_ns == 0
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime placement boundary cost requires two distinct nonempty devices, positive bytes, and positive measured time"
                    .to_string(),
            ));
        }
        if self
            .boundary_transfer_ns
            .insert(
                (
                    source_device_id.to_string(),
                    target_device_id.to_string(),
                    byte_count,
                ),
                measured_ns,
            )
            .is_some()
        {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "runtime placement contains a duplicate {byte_count}-byte boundary cost from {source_device_id:?} to {target_device_id:?}",
            )));
        }
        Ok(())
    }

    pub fn validate_runtime_model(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        candidates: &[VulkanRuntimePlacementCandidate],
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        let targets = vulkan_runtime_placement_calibration_targets(runtime_model)?;
        let expected_signatures = targets
            .iter()
            .flat_map(|target| {
                target
                    .component_ids
                    .iter()
                    .map(|component_id| (component_id.as_str(), target.signature_id.as_str()))
            })
            .collect::<BTreeMap<_, _>>();
        let mut covered_components = BTreeSet::new();
        for candidate in candidates {
            let mut candidate_component_count = 0usize;
            for ((device_id, component_id), (signature_id, cost)) in &self.component_execution {
                if device_id != &candidate.device_id {
                    continue;
                }
                let Some(expected_signature) = expected_signatures.get(component_id.as_str())
                else {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "runtime placement cost for device {:?} references unknown component {component_id:?}",
                        candidate.device_id,
                    )));
                };
                if signature_id != expected_signature || *cost == 0 {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "runtime placement cost for component {component_id:?} on device {:?} was measured for a different compiled execution signature",
                        candidate.device_id,
                    )));
                }
                candidate_component_count += 1;
                covered_components.insert(component_id.as_str());
            }
            if candidate_component_count == 0 {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement has no measured compatible component on device {:?}",
                    candidate.device_id,
                )));
            }
        }
        let uncovered_components = expected_signatures
            .keys()
            .copied()
            .filter(|component_id| !covered_components.contains(component_id))
            .collect::<Vec<_>>();
        if !uncovered_components.is_empty() {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "runtime placement candidates cannot execute components {}",
                uncovered_components
                    .iter()
                    .map(|component_id| format!("{component_id:?}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            )));
        }
        if !candidates.iter().any(|candidate| {
            self.default_graph_compatible_devices
                .contains(&candidate.device_id)
        }) {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime placement has no candidate compatible with the default input/output graph"
                    .to_string(),
            ));
        }
        if candidates.len() > 1 {
            let boundary_bytes = vulkan_runtime_placement_boundary_byte_counts(runtime_model)?;
            for source in candidates {
                for target in candidates {
                    if source.device_id == target.device_id {
                        continue;
                    }
                    for byte_count in &boundary_bytes {
                        self.boundary_transfer_ns(
                            &source.device_id,
                            &target.device_id,
                            *byte_count,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn normalized_device_execution_ns(
        &self,
        device_id: &str,
        total_component_count: usize,
    ) -> Result<u128, VulkanRuntimeResidencyPlanError> {
        if total_component_count == 0 {
            return Err(VulkanRuntimeResidencyPlanError(
                "runtime placement normalization requires a positive component count".to_string(),
            ));
        }
        let costs = self
            .component_execution
            .iter()
            .filter(|((candidate_device_id, _), _)| candidate_device_id == device_id)
            .map(|(_, (_, cost))| u128::from(*cost))
            .collect::<Vec<_>>();
        if costs.is_empty() || costs.len() > total_component_count {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "runtime placement has an invalid compatible-component count for device {device_id:?}",
            )));
        }
        let measured_component_count = costs.len() as u128;
        costs
            .into_iter()
            .sum::<u128>()
            .checked_mul(total_component_count as u128)
            .map(|scaled| scaled / measured_component_count)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "runtime placement normalized execution cost overflowed".to_string(),
                )
            })
    }

    fn can_host_default_graph(&self, device_id: &str) -> bool {
        self.default_graph_compatible_devices.contains(device_id)
    }

    fn try_component_execution_ns(&self, device_id: &str, component_id: &str) -> Option<u64> {
        self.component_execution
            .get(&(device_id.to_string(), component_id.to_string()))
            .map(|(_, cost)| *cost)
    }

    fn component_execution_ns(
        &self,
        device_id: &str,
        component_id: &str,
    ) -> Result<u64, VulkanRuntimeResidencyPlanError> {
        self.try_component_execution_ns(device_id, component_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement has no execution cost for component {component_id:?} on device {device_id:?}",
                ))
            })
    }

    fn boundary_transfer_ns(
        &self,
        source_device_id: &str,
        target_device_id: &str,
        byte_count: usize,
    ) -> Result<u64, VulkanRuntimeResidencyPlanError> {
        self.boundary_transfer_ns
            .get(&(
                source_device_id.to_string(),
                target_device_id.to_string(),
                byte_count,
            ))
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement has no measured {byte_count}-byte boundary cost from {source_device_id:?} to {target_device_id:?}",
                ))
            })
    }
}

/// The resident execution slices contain only instantaneous edges between
/// signal processors. Their `edge_index` is therefore the ordinal in this
/// filtered graph, not the ordinal in the package's full graph (which also
/// contains transducer and state edges). Keep every physical-route identity
/// on this exact mounted edge space.
fn vulkan_runtime_mounted_signal_processor_edges(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Vec<&crate::stream_circuit::StreamCircuitGraphEdge> {
    let processor_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    runtime_model
        .circuit_graph
        .edges
        .iter()
        .filter(|edge| {
            edge.connection.is_instantaneous()
                && processor_ids.contains(edge.source.component_id.as_str())
                && processor_ids.contains(edge.destination.component_id.as_str())
        })
        .collect()
}

fn vulkan_runtime_placement_boundaries(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<Vec<VulkanRuntimePlacementBoundary>, VulkanRuntimeResidencyPlanError> {
    let signal_processors = runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .collect::<Vec<_>>();
    if signal_processors.len() < 2 {
        return Ok(Vec::new());
    }
    let component_index = signal_processors
        .iter()
        .enumerate()
        .map(|(index, component)| (component.component_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let bytes_per_element = runtime_model.package.activation_element_bytes.ok_or_else(|| {
        VulkanRuntimeResidencyPlanError(
            "multi-device runtime placement requires a compiled activation element width"
                .to_string(),
        )
    })?;
    if bytes_per_element == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled activation element width must be positive".to_string(),
        ));
    }
    let component_by_id = runtime_model
        .circuit_graph
        .components
        .iter()
        .map(|component| (component.component_id.as_str(), component))
        .collect::<BTreeMap<_, _>>();
    let mut grouped = vec![BTreeSet::<(bool, String, String, usize)>::new(); signal_processors.len() - 1];
    for edge in vulkan_runtime_mounted_signal_processor_edges(runtime_model) {
        let (Some(&source_index), Some(&destination_index)) = (
            component_index.get(edge.source.component_id.as_str()),
            component_index.get(edge.destination.component_id.as_str()),
        ) else {
            continue;
        };
        if source_index.abs_diff(destination_index) != 1 {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "automatic cost-based placement requires a nearest-neighbor signal-processor chain, but edge {:?} connects component positions {source_index} and {destination_index}; use explicit wiring for non-chain graphs",
                edge.id,
            )));
        }
        let source = component_by_id[edge.source.component_id.as_str()];
        let destination = component_by_id[edge.destination.component_id.as_str()];
        let output = source
            .circuit
            .boundary
            .outputs
            .iter()
            .find(|port| port.id == edge.source.port_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement edge {:?} references missing output {}.{}",
                    edge.id, edge.source.component_id, edge.source.port_id,
                ))
            })?;
        let input = destination
            .circuit
            .boundary
            .inputs
            .iter()
            .find(|port| port.id == edge.destination.port_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement edge {:?} references missing input {}.{}",
                    edge.id, edge.destination.component_id, edge.destination.port_id,
                ))
            })?;
        let physical_shape = edge.connection.physical_shape(&output.shape, &input.shape);
        let element_count = physical_shape.iter().try_fold(1usize, |total, dimension| {
            total.checked_mul(*dimension).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime placement edge {:?} activation shape overflows",
                    edge.id,
                ))
            })
        })?;
        let byte_count = element_count.checked_mul(bytes_per_element).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "runtime placement edge {:?} activation byte count overflows",
                edge.id,
            ))
        })?;
        if byte_count == 0 {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "runtime placement edge {:?} has an empty activation payload",
                edge.id,
            )));
        }
        let cut = source_index.min(destination_index);
        grouped[cut].insert((
            source_index < destination_index,
            edge.source.component_id.clone(),
            edge.source.port_id.clone(),
            byte_count,
        ));
    }
    Ok(grouped
        .into_iter()
        .map(|transfers| VulkanRuntimePlacementBoundary {
            transfers: transfers
                .into_iter()
                .map(
                    |(source_in_prefix, _source_component_id, _source_port_id, byte_count)| {
                        VulkanRuntimePlacementBoundaryTransfer {
                            source_in_prefix,
                            byte_count,
                        }
                    },
                )
                .collect(),
        })
        .collect())
}

fn vulkan_runtime_placement_boundary_byte_counts(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<BTreeSet<usize>, VulkanRuntimeResidencyPlanError> {
    Ok(vulkan_runtime_placement_boundaries(runtime_model)?
        .into_iter()
        .flat_map(|boundary| boundary.transfers)
        .map(|transfer| transfer.byte_count)
        .collect())
}

fn runtime_placement_boundary_cost_ns(
    boundary: &VulkanRuntimePlacementBoundary,
    prefix_device_id: &str,
    suffix_device_id: &str,
    costs: &VulkanRuntimePlacementCostModel,
) -> Result<u64, VulkanRuntimeResidencyPlanError> {
    boundary.transfers.iter().try_fold(0u64, |total, transfer| {
        let (source, target) = if transfer.source_in_prefix {
            (prefix_device_id, suffix_device_id)
        } else {
            (suffix_device_id, prefix_device_id)
        };
        total
            .checked_add(costs.boundary_transfer_ns(source, target, transfer.byte_count)?)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "runtime placement boundary transfer time overflowed".to_string(),
                )
            })
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeCostAwarePlacement {
    placement: BTreeMap<String, String>,
    ordered_device_ids: Vec<String>,
    predicted_execution_ns: u128,
}

/// Separates independently routed signal-processor weight from endpoint-owned
/// auxiliary graphs. In a paged model, charging an entire speculative decoder
/// to the final target layer makes that layer look artificially enormous and
/// permits the optimizer to strand most target-cache capacity elsewhere. The
/// auxiliary bytes still reserve capacity on the endpoint that owns them; they
/// simply do not stand in for target-layer working set.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimePagedPlacementBalance {
    component_weights: Vec<u128>,
    input_auxiliary_weight_bytes: u128,
    output_auxiliary_weight_bytes: u128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VulkanRuntimeAutoPlacement {
    /// Exact compiled model at the converged logical placement. Representation
    /// selection may produce a smaller/faster `runtime_model`, but later
    /// measured physical planning must start here so it can choose a different
    /// validated representation without layering overlays on overlays.
    pub exact_runtime_model: VulkanResidentRuntimeModel,
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
    placement_costs: Option<&VulkanRuntimePlacementCostModel>,
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
    let initial = capacity_pack_vulkan_runtime_model_with_costs(
        manifest_dir,
        runtime_model,
        &tensor_index,
        candidates,
        placement_costs,
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
            .select_and_apply_runtime_implementations(manifest_dir, &profiles, execution.clone())
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        let selected_tensor_index = selected_model
            .load_runtime_tensor_index(manifest_dir)
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        let selected = capacity_pack_vulkan_runtime_model_with_costs(
            manifest_dir,
            &selected_model,
            &selected_tensor_index,
            candidates,
            placement_costs,
            context_capacity_activations,
            speculative_draft_tokens,
            residency_policy,
        )?;
        let selected_signature = runtime_model_placement_signature(&selected.runtime_model);
        if selected_signature == placement_signature {
            return Ok(VulkanRuntimeAutoPlacement {
                exact_runtime_model: exact_placed_model,
                ..selected
            });
        }

        let selected_placement = selected_signature.into_iter().collect::<BTreeMap<_, _>>();
        let default_device_id = selected.selected_device_ids.first().ok_or_else(|| {
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
        plan.resource_store.maximum_dynamic_allocation_padding_bytes,
        plan.resource_store.retained_representation_cache_payload_bytes,
        plan.resource_store
            .retained_representation_cache_allocation_padding_bytes,
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
    capacity_pack_vulkan_runtime_model_with_costs(
        manifest_dir,
        runtime_model,
        tensor_index,
        candidates,
        None,
        context_capacity_activations,
        speculative_draft_tokens,
        residency_policy,
    )
}

#[allow(clippy::too_many_arguments)]
fn capacity_pack_vulkan_runtime_model_with_costs(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    candidates: &[VulkanRuntimePlacementCandidate],
    placement_costs: Option<&VulkanRuntimePlacementCostModel>,
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
            "runtime auto-placement found no independently placeable signal processors".to_string(),
        ));
    }
    if let Some(costs) = placement_costs {
        costs.validate_runtime_model(runtime_model, candidates)?;
    }
    let maximum_device_count = candidates.len().min(components.len());
    let manifest_dir = manifest_dir.as_ref();
    let residency_planning_basis = prepare_vulkan_runtime_residency_planning_basis(
        manifest_dir,
        runtime_model,
        tensor_index,
    )?;
    let paged_balance = (residency_policy == ResourceResidencyPolicy::DemandPaged)
        .then(|| {
            runtime_paged_placement_balance(
                runtime_model,
                tensor_index,
                &components,
                speculative_draft_tokens > 0,
            )
        })
        .transpose()?;
    let mut failures = Vec::new();
    // A placement owns the mounted runtime model. Retaining every admissible
    // candidate used to retain one complete model clone per subset (127 for
    // seven targets), which made planning memory proportional to the search
    // space. Keep only the current winners; comparison is a streaming
    // reduction and has constant model residency.
    let mut best_paged_shortfall = None;
    for device_count in 1..=maximum_device_count {
        let selections = if placement_costs.is_some() {
            runtime_placement_candidate_subsets(candidates, device_count)?
        } else {
            vec![candidates[..device_count].to_vec()]
        };
        let mut best_successful = None;
        for selected in selections {
            let selected_ids = selected
                .iter()
                .map(|candidate| candidate.device_id.as_str())
                .collect::<Vec<_>>()
                .join(",");
            let attempt = if let Some(balance) = paged_balance.as_ref() {
                capacity_pack_demand_paged_vulkan_runtime_model_on_devices(
                    runtime_model,
                    tensor_index,
                    &residency_planning_basis,
                    &components,
                    &selected,
                    placement_costs,
                    balance,
                    context_capacity_activations,
                    speculative_draft_tokens,
                )
            } else {
                capacity_pack_vulkan_runtime_model_on_devices(
                    runtime_model,
                    tensor_index,
                    &residency_planning_basis,
                    &components,
                    &selected,
                    placement_costs,
                    context_capacity_activations,
                    speculative_draft_tokens,
                    residency_policy,
                )
            };
            match attempt {
                Ok(placed) => {
                    let predicted_ns = placement_costs.map_or(Ok(0), |costs| {
                        predicted_runtime_placement_ns(
                            &placed.runtime_model,
                            &components,
                            &vulkan_runtime_placement_boundaries(&placed.runtime_model)?,
                            costs,
                        )
                    })?;
                    let shortfall = paged_balance
                        .as_ref()
                        .map(|_| {
                            demand_paged_placement_has_addressable_shortfall(
                                &placed.residency_plan,
                                &selected,
                            )
                        })
                        .transpose()?
                        .unwrap_or(false);
                    let selected_capacity =
                        selected.iter().try_fold(0u128, |total, candidate| {
                            total
                                .checked_add(candidate.safe_capacity_bytes as u128)
                                .ok_or_else(|| {
                                    VulkanRuntimeResidencyPlanError(
                                        "runtime placement selected capacity overflowed"
                                            .to_string(),
                                    )
                                })
                        })?;
                    let outcome = (
                        predicted_ns,
                        placed.selected_device_ids.clone(),
                        selected_capacity,
                        placed,
                    );
                    let best = if shortfall {
                        &mut best_paged_shortfall
                    } else {
                        &mut best_successful
                    };
                    let replace = best.as_ref().is_none_or(|current: &(
                        u128,
                        Vec<String>,
                        u128,
                        VulkanRuntimeAutoPlacement,
                    )| {
                        if shortfall {
                            (std::cmp::Reverse(outcome.2), outcome.0, &outcome.1)
                                < (std::cmp::Reverse(current.2), current.0, &current.1)
                        } else {
                            (outcome.0, &outcome.1) < (current.0, &current.1)
                        }
                    });
                    if replace {
                        *best = Some(outcome);
                    }
                }
                Err(error) => failures.push(format!(
                    "{device_count} device(s) [{selected_ids}]: {error}",
                )),
            }
        }
        if let Some((_, _, _, placed)) = best_successful {
            return Ok(placed);
        }
    }
    if let Some((_, _, _, placed)) = best_paged_shortfall {
        return Ok(placed);
    }
    Err(VulkanRuntimeResidencyPlanError(format!(
        "no capacity-packed contiguous placement can admit the model working set: {}",
        failures.join("; "),
    )))
}

fn runtime_placement_candidate_subsets(
    candidates: &[VulkanRuntimePlacementCandidate],
    subset_size: usize,
) -> Result<Vec<Vec<VulkanRuntimePlacementCandidate>>, VulkanRuntimeResidencyPlanError> {
    if subset_size == 0 || subset_size > candidates.len() {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime placement candidate subset requires a valid positive size".to_string(),
        ));
    }
    let mut indices = (0..subset_size).collect::<Vec<_>>();
    let mut subsets = Vec::new();
    loop {
        subsets.push(
            indices
                .iter()
                .map(|index| candidates[*index].clone())
                .collect(),
        );
        let Some(position) = (0..subset_size)
            .rev()
            .find(|position| indices[*position] < candidates.len() - subset_size + *position)
        else {
            break;
        };
        indices[position] += 1;
        for next in position + 1..subset_size {
            indices[next] = indices[next - 1] + 1;
        }
    }
    Ok(subsets)
}

/// A demand-paged package may execute inside a cache much smaller than its
/// addressable resources, but that does not make the smaller cache the best
/// placement. Use the exact admitted per-device residency plan, including
/// permanent parameters, store metadata/padding, state, and activations. Raw
/// aggregate target capacity can hide an overloaded device behind unused
/// capacity on another target. At the largest legal set paging remains valid
/// and unavoidable.
fn demand_paged_placement_has_addressable_shortfall(
    plan: &VulkanRuntimeResidencyPlan,
    selected: &[VulkanRuntimePlacementCandidate],
) -> Result<bool, VulkanRuntimeResidencyPlanError> {
    if plan.residency_policy != ResourceResidencyPolicy::DemandPaged
        || plan.device_plans.is_empty()
        || selected.is_empty()
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "demand-paged addressable-fit accounting requires a paged residency plan and nonempty device set"
                .to_string(),
        ));
    }
    let capacity_by_device = selected
        .iter()
        .map(|candidate| (candidate.device_id.as_str(), candidate.safe_capacity_bytes))
        .collect::<BTreeMap<_, _>>();
    if capacity_by_device.len() != selected.len()
        || selected.iter().any(|candidate| {
            candidate.device_id.trim().is_empty() || candidate.safe_capacity_bytes == 0
        })
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "demand-paged addressable-fit accounting requires unique nonempty positive-capacity devices"
                .to_string(),
        ));
    }
    let mut planned_devices = BTreeSet::new();
    let mut has_shortfall = false;
    for device_plan in &plan.device_plans {
        if device_plan.device_id.trim().is_empty()
            || !planned_devices.insert(device_plan.device_id.as_str())
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "demand-paged residency plan repeats or empties a logical device".to_string(),
            ));
        }
        let safe_capacity_bytes = capacity_by_device
            .get(device_plan.device_id.as_str())
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "demand-paged residency plan references unselected device {:?}",
                    device_plan.device_id,
                ))
            })?;
        has_shortfall |=
            vulkan_runtime_maximum_device_resident_bytes(device_plan)? > safe_capacity_bytes;
    }
    if planned_devices
        != capacity_by_device.keys().copied().collect::<BTreeSet<_>>()
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "demand-paged residency plan does not cover every selected device".to_string(),
        ));
    }
    Ok(has_shortfall)
}

#[allow(clippy::too_many_arguments)]
fn capacity_pack_demand_paged_vulkan_runtime_model_on_devices(
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    residency_planning_basis: &VulkanRuntimeResidencyPlanningBasis,
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
    placement_costs: Option<&VulkanRuntimePlacementCostModel>,
    paged_balance: &VulkanRuntimePagedPlacementBalance,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
) -> Result<VulkanRuntimeAutoPlacement, VulkanRuntimeResidencyPlanError> {
    let (placement, ordered_candidates) = match placement_costs {
        Some(costs) => {
            let boundaries = vulkan_runtime_placement_boundaries(runtime_model)?;
            let placed = cost_aware_contiguous_component_placement(
                components,
                candidates,
                costs,
                &boundaries,
                Some(paged_balance),
            )?;
            let ordered = placed
                .ordered_device_ids
                .iter()
                .map(|device_id| {
                    candidates
                        .iter()
                        .find(|candidate| &candidate.device_id == device_id)
                        .cloned()
                        .expect("cost-aware placement only returns candidate devices")
                })
                .collect::<Vec<_>>();
            (placed.placement, ordered)
        }
        None => (
            proportional_paged_component_placement(
                components,
                candidates,
                Some(paged_balance),
            )?,
            candidates.to_vec(),
        ),
    };
    admit_fixed_vulkan_runtime_placement(
        runtime_model,
        tensor_index,
        residency_planning_basis,
        &placement,
        &ordered_candidates,
        context_capacity_activations,
        speculative_draft_tokens,
        ResourceResidencyPolicy::DemandPaged,
    )
}

/// Minimizes predicted serial decode latency over every device ordering and
/// every contiguous boundary. Retained placement uses physical byte limits.
/// Paged virtual placement gives each cache a proportional working-set quota
/// plus one component of rounding slack, preventing a fast but small cache from
/// claiming the entire address space and turning steady-state execution into
/// avoidable reload traffic.
fn cost_aware_contiguous_component_placement(
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
    costs: &VulkanRuntimePlacementCostModel,
    boundaries: &[VulkanRuntimePlacementBoundary],
    paged_balance: Option<&VulkanRuntimePagedPlacementBalance>,
) -> Result<VulkanRuntimeCostAwarePlacement, VulkanRuntimeResidencyPlanError> {
    if components.is_empty()
        || candidates.is_empty()
        || candidates.len() > components.len()
        || candidates.len() > u64::BITS as usize
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "cost-aware contiguous placement requires components and no more devices than components"
                .to_string(),
        ));
    }
    let unique_components = components
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let unique_devices = candidates
        .iter()
        .map(|candidate| candidate.device_id.as_str())
        .collect::<BTreeSet<_>>();
    if unique_components.len() != components.len()
        || unique_devices.len() != candidates.len()
        || components
            .iter()
            .any(|component| component.component_id.is_empty())
        || candidates
            .iter()
            .any(|candidate| candidate.device_id.is_empty() || candidate.safe_capacity_bytes == 0)
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "cost-aware contiguous placement requires unique nonempty components and positive-capacity devices"
                .to_string(),
        ));
    }

    if let Some(balance) = paged_balance
        && balance.component_weights.len() != components.len()
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged placement balance does not match the component chain".to_string(),
        ));
    }
    if boundaries.len() != components.len().saturating_sub(1) {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "runtime placement has {} component boundaries for {} components",
            boundaries.len(),
            components.len(),
        )));
    }
    let mut effective_weights = paged_balance.map_or_else(
        || {
            components
                .iter()
                .map(|component| component.resident_weight_bytes as u128)
                .collect::<Vec<_>>()
        },
        |balance| balance.component_weights.clone(),
    );
    if effective_weights.iter().all(|weight| *weight == 0) {
        effective_weights.fill(1);
    }
    let total_weight = effective_weights.iter().try_fold(0u128, |total, weight| {
        total.checked_add(*weight).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError("cost-aware component weights overflowed".to_string())
        })
    })?;
    let maximum_component_weight = effective_weights.iter().copied().max().unwrap_or(0);
    let total_capacity = candidates.iter().try_fold(0u128, |total, candidate| {
        total
            .checked_add(candidate.safe_capacity_bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "cost-aware device capacities overflowed".to_string(),
                )
            })
    })?;
    let (input_auxiliary_weight_bytes, output_auxiliary_weight_bytes) = paged_balance
        .map(|balance| {
            (
                balance.input_auxiliary_weight_bytes,
                balance.output_auxiliary_weight_bytes,
            )
        })
        .unwrap_or_default();
    let effective_total_capacity = total_capacity
        .checked_sub(input_auxiliary_weight_bytes)
        .and_then(|capacity| capacity.checked_sub(output_auxiliary_weight_bytes))
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "paged endpoint auxiliary graphs exhaust aggregate device capacity".to_string(),
            )
        })?;
    if effective_total_capacity == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged endpoint auxiliary graphs leave no signal-processor capacity".to_string(),
        ));
    }

    #[derive(Clone)]
    struct PlacementState {
        predicted_execution_ns: u128,
        segments: Vec<(usize, usize, usize)>,
    }
    let no_device = candidates.len();
    let mut states = BTreeMap::from([(
        (0u64, 0usize, no_device),
        PlacementState {
            predicted_execution_ns: 0,
            segments: Vec::new(),
        },
    )]);
    for _ in 0..candidates.len() {
        let mut next_states = BTreeMap::<(u64, usize, usize), PlacementState>::new();
        for ((mask, cursor, previous_device), state) in states {
            for device_index in 0..candidates.len() {
                let device_bit = 1u64 << device_index;
                if mask & device_bit != 0 {
                    continue;
                }
                if cursor == 0
                    && !costs.can_host_default_graph(&candidates[device_index].device_id)
                {
                    continue;
                }
                let remaining_devices = candidates.len() - (mask.count_ones() as usize) - 1;
                let maximum_cut = components.len().saturating_sub(remaining_devices);
                let mut segment_weight = 0u128;
                let mut segment_execution_ns = 0u128;
                for cut in cursor + 1..=maximum_cut {
                    segment_weight = segment_weight
                        .checked_add(effective_weights[cut - 1])
                        .ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(
                                "cost-aware segment weights overflowed".to_string(),
                            )
                        })?;
                    let is_input_endpoint = cursor == 0;
                    let is_output_endpoint = cut == components.len();
                    let available_capacity = (candidates[device_index].safe_capacity_bytes as u128)
                        .checked_sub(
                            is_input_endpoint
                                .then_some(input_auxiliary_weight_bytes)
                                .unwrap_or_default(),
                        )
                        .and_then(|capacity| {
                            capacity.checked_sub(
                                is_output_endpoint
                                    .then_some(output_auxiliary_weight_bytes)
                                    .unwrap_or_default(),
                            )
                        });
                    let Some(available_capacity) = available_capacity else {
                        continue;
                    };
                    let (minimum_weight, maximum_weight) = if paged_balance.is_some() {
                        let numerator = total_weight
                            .checked_mul(available_capacity)
                            .ok_or_else(|| {
                                VulkanRuntimeResidencyPlanError(
                                    "cost-aware proportional quota overflowed".to_string(),
                                )
                            })?;
                        let proportional = numerator
                            .checked_add(effective_total_capacity.saturating_sub(1))
                            .ok_or_else(|| {
                                VulkanRuntimeResidencyPlanError(
                                    "cost-aware proportional quota rounding overflowed"
                                        .to_string(),
                                )
                            })?
                            / effective_total_capacity;
                        (
                            proportional.saturating_sub(maximum_component_weight),
                            proportional
                                .saturating_add(maximum_component_weight)
                                .min(total_weight),
                        )
                    } else {
                        (0, available_capacity)
                    };
                    if segment_weight > maximum_weight {
                        break;
                    }
                    let Some(component_execution_ns) = costs.try_component_execution_ns(
                        &candidates[device_index].device_id,
                        &components[cut - 1].component_id,
                    ) else {
                        break;
                    };
                    segment_execution_ns = segment_execution_ns
                        .checked_add(u128::from(component_execution_ns))
                        .ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(
                                "cost-aware predicted execution time overflowed".to_string(),
                            )
                        })?;
                    if segment_weight < minimum_weight {
                        continue;
                    }
                    let transfer_ns = if previous_device == no_device {
                        0
                    } else {
                        runtime_placement_boundary_cost_ns(
                            &boundaries[cursor - 1],
                            &candidates[previous_device].device_id,
                            &candidates[device_index].device_id,
                            costs,
                        )?
                    };
                    let predicted_execution_ns = state
                        .predicted_execution_ns
                        .checked_add(segment_execution_ns)
                        .and_then(|total| total.checked_add(u128::from(transfer_ns)))
                        .ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(
                                "cost-aware total execution time overflowed".to_string(),
                            )
                        })?;
                    let mut segments = state.segments.clone();
                    segments.push((device_index, cursor, cut));
                    let key = (mask | device_bit, cut, device_index);
                    let proposed = PlacementState {
                        predicted_execution_ns,
                        segments,
                    };
                    let replace = next_states.get(&key).is_none_or(|current| {
                        (proposed.predicted_execution_ns, &proposed.segments)
                            < (current.predicted_execution_ns, &current.segments)
                    });
                    if replace {
                        next_states.insert(key, proposed);
                    }
                }
            }
        }
        states = next_states;
    }
    let complete_mask = (1u64 << candidates.len()) - 1;
    let best = states
        .into_iter()
        .filter(|((mask, cursor, _), _)| *mask == complete_mask && *cursor == components.len())
        .map(|(_, state)| state)
        .min_by(|left, right| {
            (left.predicted_execution_ns, &left.segments)
                .cmp(&(right.predicted_execution_ns, &right.segments))
        })
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "no cost-aware contiguous placement satisfies every device quota".to_string(),
            )
        })?;
    let mut placement = BTreeMap::new();
    let mut ordered_device_ids = Vec::with_capacity(best.segments.len());
    for (device_index, start, end) in best.segments {
        let device_id = candidates[device_index].device_id.clone();
        ordered_device_ids.push(device_id.clone());
        for component in &components[start..end] {
            placement.insert(component.component_id.clone(), device_id.clone());
        }
    }
    Ok(VulkanRuntimeCostAwarePlacement {
        placement,
        ordered_device_ids,
        predicted_execution_ns: best.predicted_execution_ns,
    })
}

/// Partitions a virtual resource set across every selected paged cache. This
/// path is used only when the complete addressable set exceeds their aggregate
/// physical capacity. It preserves graph order, gives every device a nonempty
/// contiguous segment, and chooses boundaries proportional to measured cache
/// capacity. Physical fixed-state and load-wave admission is proven separately.
fn proportional_paged_component_placement(
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
    paged_balance: Option<&VulkanRuntimePagedPlacementBalance>,
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

    if let Some(balance) = paged_balance
        && balance.component_weights.len() != components.len()
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged proportional balance does not match the component chain".to_string(),
        ));
    }
    let (input_auxiliary_weight_bytes, output_auxiliary_weight_bytes) = paged_balance
        .map(|balance| {
            (
                balance.input_auxiliary_weight_bytes,
                balance.output_auxiliary_weight_bytes,
            )
        })
        .unwrap_or_default();
    let effective_capacities = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            (candidate.safe_capacity_bytes as u128)
                .checked_sub(
                    (index == 0)
                        .then_some(input_auxiliary_weight_bytes)
                        .unwrap_or_default(),
                )
                .and_then(|capacity| {
                    capacity.checked_sub(
                        (index + 1 == candidates.len())
                            .then_some(output_auxiliary_weight_bytes)
                            .unwrap_or_default(),
                    )
                })
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "paged endpoint auxiliary graphs exhaust device {:?}",
                        candidate.device_id,
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_capacity = effective_capacities.iter().try_fold(0u128, |total, capacity| {
        total
            .checked_add(*capacity)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "paged proportional device capacity overflowed".to_string(),
                )
            })
    })?;
    let mut prefix_weights = Vec::with_capacity(components.len() + 1);
    prefix_weights.push(0u128);
    let mut effective_weights = paged_balance.map_or_else(
        || {
            components
                .iter()
                .map(|component| component.resident_weight_bytes as u128)
                .collect::<Vec<_>>()
        },
        |balance| balance.component_weights.clone(),
    );
    if effective_weights.iter().all(|weight| *weight == 0) {
        effective_weights.fill(1);
    }
    for weight in &effective_weights {
        let next = prefix_weights
            .last()
            .copied()
            .expect("weight prefix has an origin")
            .checked_add(*weight)
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
            .checked_add(effective_capacities[device_index])
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
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    residency_planning_basis: &VulkanRuntimeResidencyPlanningBasis,
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
    let residency_plan = plan_vulkan_runtime_residency_with_basis(
        &placed_model,
        tensor_index,
        context_capacity_activations,
        speculative_draft_tokens,
        residency_policy,
        residency_planning_basis,
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
        exact_runtime_model: placed_model.clone(),
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
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    residency_planning_basis: &VulkanRuntimeResidencyPlanningBasis,
    components: &[CapacityPackedPlacementComponent],
    candidates: &[VulkanRuntimePlacementCandidate],
    placement_costs: Option<&VulkanRuntimePlacementCostModel>,
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
        let (placement, ordered_device_ids) = match placement_costs {
            Some(costs) => {
                let boundaries = vulkan_runtime_placement_boundaries(runtime_model)?;
                let effective_candidates = candidates
                    .iter()
                    .map(|candidate| VulkanRuntimePlacementCandidate {
                        device_id: candidate.device_id.clone(),
                        safe_capacity_bytes: effective_capacities
                            .iter()
                            .find(|device| device.device_id == candidate.device_id)
                            .expect("effective capacity exists for every candidate")
                            .capacity_bytes,
                    })
                    .collect::<Vec<_>>();
                let placed = cost_aware_contiguous_component_placement(
                    components,
                    &effective_candidates,
                    costs,
                    &boundaries,
                    None,
                )?;
                (placed.placement, placed.ordered_device_ids)
            }
            None => (
                capacity_packed_component_placement(components, &effective_capacities)
                    .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?,
                candidates
                    .iter()
                    .map(|candidate| candidate.device_id.clone())
                    .collect(),
            ),
        };
        if previous_placement.as_ref() == Some(&placement) {
            return Err(VulkanRuntimeResidencyPlanError(
                "exact residency correction converged to an over-capacity placement".to_string(),
            ));
        }
        let placed_model = vulkan_runtime_model_with_component_placement(
            runtime_model,
            &ordered_device_ids[0],
            &placement,
        )?;
        let residency_plan = plan_vulkan_runtime_residency_with_basis(
            &placed_model,
            tensor_index,
            context_capacity_activations,
            speculative_draft_tokens,
            residency_policy,
            residency_planning_basis,
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
            let required =
                vulkan_runtime_device_capacity_admission_bytes(device_plan, residency_policy)?;
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
                exact_runtime_model: placed_model.clone(),
                runtime_model: placed_model,
                residency_plan,
                selected_device_ids: ordered_device_ids,
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

fn runtime_paged_placement_balance(
    runtime_model: &VulkanResidentRuntimeModel,
    tensor_index: &TensorIndex,
    components: &[CapacityPackedPlacementComponent],
    mount_speculative_decoders: bool,
) -> Result<VulkanRuntimePagedPlacementBalance, VulkanRuntimeResidencyPlanError> {
    let component_index = components
        .iter()
        .enumerate()
        .map(|(index, component)| (component.component_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    if component_index.len() != components.len() {
        return Err(VulkanRuntimeResidencyPlanError(
            "paged placement balance requires unique signal-processor ids".to_string(),
        ));
    }
    let mut component_weights = vec![0u128; components.len()];
    let mut input_auxiliary_weight_bytes = 0u128;
    let mut output_auxiliary_weight_bytes = 0u128;
    let mut charged_tensors = BTreeSet::new();

    let tensor_bytes = |tensor: &str| -> Result<u128, VulkanRuntimeResidencyPlanError> {
        tensor_index
            .tensors
            .get(tensor)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "paged placement tensor {tensor:?} is absent from the runtime tensor index",
                ))
            })?
            .byte_count
            .map(|bytes| bytes as u128)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "paged placement tensor {tensor:?} has no byte_count",
                ))
            })
    };

    for component in &runtime_model.circuit_graph.components {
        let destination = match component.runtime_role {
            CircuitRuntimeRole::InputTransducer => None,
            CircuitRuntimeRole::SignalProcessor => Some(
                *component_index.get(component.component_id.as_str()).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "paged placement balance found unknown signal processor {:?}",
                        component.component_id,
                    ))
                })?,
            ),
            CircuitRuntimeRole::OutputTransducer | CircuitRuntimeRole::Sampler => None,
            CircuitRuntimeRole::DraftProcessor
            | CircuitRuntimeRole::DraftInputAdapter
            | CircuitRuntimeRole::DraftOutputTransducer => {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "target graph contains draft component {:?}",
                    component.component_id,
                )));
            }
        };
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
            let bytes = tensor_bytes(tensor)?;
            if let Some(index) = destination {
                component_weights[index] = component_weights[index]
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "paged signal-processor balance overflowed".to_string(),
                        )
                    })?;
            } else if component.runtime_role == CircuitRuntimeRole::InputTransducer {
                input_auxiliary_weight_bytes = input_auxiliary_weight_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "paged input auxiliary balance overflowed".to_string(),
                        )
                    })?;
            } else {
                output_auxiliary_weight_bytes = output_auxiliary_weight_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(
                            "paged output auxiliary balance overflowed".to_string(),
                        )
                    })?;
            }
        }
    }

    if mount_speculative_decoders {
        for decoder in &runtime_model.package.speculative_decoders {
            for component in &decoder.circuit_graph.components {
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
                    output_auxiliary_weight_bytes = output_auxiliary_weight_bytes
                        .checked_add(tensor_bytes(tensor)?)
                        .ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(
                                "paged speculative auxiliary balance overflowed".to_string(),
                            )
                        })?;
                }
            }
        }
    }

    Ok(VulkanRuntimePagedPlacementBalance {
        component_weights,
        input_auxiliary_weight_bytes,
        output_auxiliary_weight_bytes,
    })
}

/// Returns a copy of a mounted logical model with an explicit physical owner
/// for every named runtime instance. Instances absent from `placement` use the
/// caller's default owner. This is the single placement transformation used by
/// capacity packing, distributed calibration, and representation selection so
/// those paths cannot construct subtly different selection requests.
pub fn vulkan_runtime_model_with_component_placement(
    runtime_model: &VulkanResidentRuntimeModel,
    default_device_id: &str,
    placement: &BTreeMap<String, String>,
) -> Result<VulkanResidentRuntimeModel, VulkanRuntimeResidencyPlanError> {
    vulkan_runtime_model_with_component_placement_owned(
        runtime_model.clone(),
        default_device_id,
        placement,
    )
}

/// Applies explicit physical ownership to an already-owned runtime model.
/// Package calibration and other one-shot transformations use this form so a
/// large compiled package is never deep-cloned merely to rewrite its small
/// runtime graph and placement records.
pub fn vulkan_runtime_model_with_component_placement_owned(
    mut placed_model: VulkanResidentRuntimeModel,
    default_device_id: &str,
    placement: &BTreeMap<String, String>,
) -> Result<VulkanResidentRuntimeModel, VulkanRuntimeResidencyPlanError> {
    if default_device_id.is_empty() || placement.values().any(String::is_empty) {
        return Err(VulkanRuntimeResidencyPlanError(
            "runtime component placement requires nonempty device ids".to_string(),
        ));
    }
    let instance_ids = placed_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| instance.instance_id.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(component_id) = placement
        .keys()
        .find(|component_id| !instance_ids.contains(component_id.as_str()))
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "runtime component placement references unknown instance {component_id:?}",
        )));
    }
    let mut runtime_graph = placed_model.runtime_graph.clone();
    runtime_graph.default_device_id = default_device_id.to_string();
    for instance in &mut runtime_graph.instances {
        instance.device_id = placement
            .get(&instance.instance_id)
            .cloned()
            .unwrap_or_else(|| default_device_id.to_string());
    }
    let runtime_graph = attach_generation_node_devices_for_compiled_package(
        runtime_graph,
        &placed_model.package,
    )
    .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    for instance in &runtime_graph.instances {
        if instance.device_id.is_empty() {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "runtime component placement left instance {:?} without a device",
                instance.instance_id,
            )));
        }
    }
    let mut placed = StreamCircuitPlacementSpec::new(default_device_id.to_string());
    for instance in &runtime_graph.instances {
        if instance.device_id != default_device_id {
            placed = placed.with_component_device(&instance.instance_id, &instance.device_id);
        }
    }
    placed_model.runtime_graph = runtime_graph;
    placed_model.placement = placed;
    Ok(placed_model)
}
