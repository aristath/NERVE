/// One exact compiler-emitted selected-resource transaction. The component
/// target identifies the logical implementation family; the selector and
/// resource index identify one independently resident arithmetic path inside
/// it. The execution-class ID and contract IDs make the planned target
/// immutable: calibration refuses to silently measure a neighboring physical
/// representation if package lowering changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeSelectedResourceExecutionCalibrationTarget {
    pub component: VulkanRuntimePlacementCalibrationTarget,
    pub selector_id: String,
    pub resource_index: usize,
    pub resource_execution_class_id: String,
    pub phase: VulkanTargetedComponentExecutionPhase,
    pub selected_contract_ids: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeSelectedResourceExecutionRequirementPlan {
    pub component_id: String,
    pub selector_id: String,
    pub resource_execution_class_ids: Vec<String>,
    pub requirements: Vec<VulkanPlacementSelectedResourceExecutionClassRequirement>,
}

struct VulkanRuntimeSelectedResourceExecutionBlueprint {
    logical_device_id: String,
    placed_model: VulkanResidentRuntimeModel,
    tensor_index: Arc<TensorIndex>,
    contract: Arc<CompiledResourceResidencyContract>,
    selector: CompiledResourceSelector,
    loaded_manifest: VulkanLoadedKernelArtifactCatalog,
    contract_phase: nerve_execution_contracts::ExecutionPhase,
    full_execution_plan: VulkanDistributedExecutionPlan,
    resource_execution_class_ids: Vec<String>,
}

impl VulkanRuntimeSelectedResourceExecutionBlueprint {
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        device: &VulkanComputeDevice,
        manifest_dir: &Path,
        runtime_model: &VulkanResidentRuntimeModel,
        component: &VulkanRuntimePlacementCalibrationTarget,
        selector_id: &str,
        phase: VulkanTargetedComponentExecutionPhase,
        selected_contract_ids: &BTreeSet<String>,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if component.signature_id.is_empty()
            || component.component_id.is_empty()
            || selector_id.is_empty()
            || selected_contract_ids.is_empty()
            || phase.activation_batch_width() == 0
        {
            return distributed_calibration_error(
                "selected-resource planning requires an exact component, selector, phase, and contract set",
            );
        }
        let logical_device_id = "calibration:selected_resource".to_string();
        let planning_peer_id = "calibration:selected_resource:planning_peer".to_string();
        let planning_device_ids = vec![logical_device_id.clone(), planning_peer_id];
        let mut placed_model = vulkan_runtime_model_with_component_placement(
            runtime_model,
            "calibration:unmounted",
            &BTreeMap::from([(component.component_id.clone(), logical_device_id.clone())]),
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        placed_model = placed_model
            .with_component_shard_devices(&component.component_id, planning_device_ids.clone())?;
        let capacity = placed_model.package.max_context_activations.clamp(
            1,
            VULKAN_RUNTIME_PLACEMENT_CALIBRATION_MAXIMUM_STATE_ACTIVATIONS,
        );
        let tensor_index = Arc::new(placed_model.load_runtime_tensor_index(manifest_dir)?);
        let contract = Arc::new(
            instantiate_runtime_resource_contract(&placed_model)
                .map_err(|error| distributed_calibration_error_value(error.to_string()))?,
        );
        let selector = contract
            .selectors
            .iter()
            .find(|selector| selector.id == selector_id)
            .cloned()
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "selected-resource execution selector {selector_id:?} is absent",
                ))
            })?;
        if selector.execution_scope != placed_model.execution_scope
            || selector.component_id != component.component_id
        {
            return distributed_calibration_error(
                "selected-resource execution selector belongs to a different component or execution scope",
            );
        }
        let residency_plan = plan_vulkan_runtime_residency_with_contract(
            manifest_dir,
            &placed_model,
            &tensor_index,
            capacity,
            0,
            ResourceResidencyPolicy::DemandRetained,
            &contract,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let targeted_plan = VulkanResidentTargetedModelPackageDeviceSlicePlan::prepare(
            device,
            manifest_dir,
            &placed_model,
            &component.component_id,
            &logical_device_id,
            capacity,
            Arc::clone(&tensor_index),
            Arc::clone(&contract),
            residency_plan,
        )?;
        let loaded_manifest = resident_package_loaded_kernel_manifest_for_slice_plans(
            std::slice::from_ref(&targeted_plan.slice_plan),
        )?;
        let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
            loaded_manifest
                .physical_artifacts
                .iter()
                .map(|artifact| artifact.artifact.clone())
                .collect(),
        );
        let graph = placed_model.executable_circuit_graph()?;
        let (_, placement_plan, _) = plan_resident_package_placed_stream_circuit_with_tensor_index(
            &logical_device_id,
            &placed_model.placement,
            &graph,
            manifest_dir,
            &tensor_index,
            placed_model.package.activation_element_bytes,
        )?;
        let (contract_phase, execution_shape) = distributed_contract_phase_and_shape(phase);
        let full_execution_plan = VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_with_resource_contract_and_contracts(
            &[(&logical_device_id, &targeted_plan.slice_plan.prepared_plan)],
            &tensor_index,
            &artifact_manifest,
            &BTreeMap::from([(component.component_id.clone(), planning_device_ids)]),
            &placement_plan.edges,
            device.min_storage_buffer_offset_alignment(),
            contract_phase,
            execution_shape,
            &placed_model.execution_scope,
            &contract,
            selected_contract_ids,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if !full_execution_plan.dispatches.iter().any(|dispatch| {
            dispatch
                .selected_resource_partitions
                .iter()
                .any(|partition| partition.selector_id == selector_id)
        }) {
            return Ok(Self {
                logical_device_id,
                placed_model,
                tensor_index,
                contract,
                selector,
                loaded_manifest,
                contract_phase,
                full_execution_plan,
                resource_execution_class_ids: Vec::new(),
            });
        }
        let classes = full_execution_plan
            .selected_resource_execution_classes(selector_id)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if classes.component_id != component.component_id {
            return distributed_calibration_error(
                "selected-resource execution class belongs to a different component",
            );
        }
        Ok(Self {
            logical_device_id,
            placed_model,
            tensor_index,
            contract,
            selector,
            loaded_manifest,
            contract_phase,
            full_execution_plan,
            resource_execution_class_ids: classes.resource_execution_class_ids,
        })
    }
}

/// Discovers the minimum exact calibration set for one physical contract set:
/// one deterministic resource representative per compiler-emitted execution
/// class. A contract set that does not execute the selector returns no targets;
/// it is not mislabeled as selected-resource work.
#[allow(clippy::too_many_arguments)]
pub fn vulkan_runtime_selected_resource_execution_calibration_targets(
    device: &VulkanComputeDevice,
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    component: &VulkanRuntimePlacementCalibrationTarget,
    selector_id: &str,
    phase: VulkanTargetedComponentExecutionPhase,
    selected_contract_ids: &BTreeSet<String>,
) -> Result<Vec<VulkanRuntimeSelectedResourceExecutionCalibrationTarget>, VulkanResidentTokenModelPackageError>
{
    let blueprint = VulkanRuntimeSelectedResourceExecutionBlueprint::prepare(
        device,
        manifest_dir.as_ref(),
        runtime_model,
        component,
        selector_id,
        phase,
        selected_contract_ids,
    )?;
    selected_resource_execution_calibration_targets_for_classes(
        component,
        selector_id,
        phase,
        selected_contract_ids,
        &blueprint.resource_execution_class_ids,
    )
}

/// Rebuilds the selected-resource class requirements from the exact physical
/// plan that production will mount. This is intentionally downstream of exact
/// case replay: changing a contract, artifact, representation, phase, shape,
/// or execution topology changes the requirement and prevents stale catalog
/// evidence from being consumed.
pub fn vulkan_runtime_selected_resource_execution_requirements(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    execution_plan: &VulkanDistributedExecutionPlan,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<Vec<VulkanRuntimeSelectedResourceExecutionRequirementPlan>, VulkanResidentTokenModelPackageError>
{
    let selector_ids = execution_plan
        .dispatches
        .iter()
        .flat_map(|dispatch| {
            dispatch
                .selected_resource_partitions
                .iter()
                .map(|partition| partition.selector_id.clone())
        })
        .collect::<BTreeSet<_>>();
    let (execution_phase, _) = distributed_contract_phase_and_shape(phase);
    let mut plans = Vec::with_capacity(selector_ids.len());
    for selector_id in selector_ids {
        let selector = resource_contract
            .selectors
            .iter()
            .find(|selector| selector.id == selector_id)
            .ok_or_else(|| {
                distributed_calibration_error_value(format!(
                    "mounted selected-resource plan references unknown selector {selector_id:?}",
                ))
            })?;
        let classes = execution_plan
            .selected_resource_execution_classes(&selector_id)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if classes.component_id != selector.component_id {
            return distributed_calibration_error(
                "mounted selected-resource execution class belongs to a different component",
            );
        }
        let component = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            &classes.component_id,
            phase,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        let owner_device_id = execution_plan
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch
                    .selected_resource_partitions
                    .iter()
                    .any(|partition| partition.selector_id == selector_id)
            })
            .map(|dispatch| dispatch.owner_device_id.as_str())
            .expect("selected-resource classes proved an executable dispatch");
        let mut representative_by_class = BTreeMap::<String, usize>::new();
        for (resource_index, class_id) in classes
            .resource_execution_class_ids
            .iter()
            .enumerate()
        {
            representative_by_class
                .entry(class_id.clone())
                .or_insert(resource_index);
        }
        let requirements = representative_by_class
            .into_iter()
            .map(|(class_id, resource_index)| {
                let isolated = execution_plan
                    .isolated_selected_resource_transaction(
                        &selector_id,
                        resource_index,
                        owner_device_id,
                        execution_phase,
                    )
                    .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
                selected_resource_execution_requirement(
                    &component.signature_id,
                    selector,
                    &class_id,
                    &isolated,
                    loaded_manifest,
                    phase,
                )
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
        plans.push(VulkanRuntimeSelectedResourceExecutionRequirementPlan {
            component_id: classes.component_id,
            selector_id,
            resource_execution_class_ids: classes.resource_execution_class_ids,
            requirements,
        });
    }
    Ok(plans)
}

/// Solves selected-resource ownership only when the mounted plan has complete
/// exact execution and load-wave evidence for every required class on every
/// compiled participant. `None` means the optional optimization is
/// unavailable and the caller must preserve the validated compiler ownership.
#[allow(clippy::too_many_arguments)]
pub fn try_plan_vulkan_runtime_selected_resource_placements(
    execution_plan: &VulkanDistributedExecutionPlan,
    requirement_plans: &[VulkanRuntimeSelectedResourceExecutionRequirementPlan],
    catalog: &VulkanPlacementCalibrationCatalog,
    capacities: &[VulkanPlacementSelectedResourceDeviceCapacity],
    telemetry: Option<&VulkanSelectionTelemetrySnapshot>,
    residency_policy: ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<Option<Vec<VulkanSelectedResourcePlacementPlan>>, VulkanResidentTokenModelPackageError>
{
    let selector_ids = execution_plan
        .dispatches
        .iter()
        .flat_map(|dispatch| {
            dispatch
                .selected_resource_partitions
                .iter()
                .map(|partition| partition.selector_id.as_str())
        })
        .collect::<BTreeSet<_>>();
    if selector_ids.is_empty() {
        return Ok(Some(Vec::new()));
    }
    if requirement_plans
        .iter()
        .map(|plan| plan.selector_id.as_str())
        .collect::<BTreeSet<_>>()
        != selector_ids
    {
        return distributed_calibration_error(
            "mounted selected-resource requirements do not cover every selector exactly once",
        );
    }

    let mut placements = Vec::with_capacity(requirement_plans.len());
    for requirement_plan in requirement_plans {
        let partitions = execution_plan
            .dispatches
            .iter()
            .flat_map(|dispatch| {
                dispatch
                    .selected_resource_partitions
                    .iter()
                    .filter(|partition| partition.selector_id == requirement_plan.selector_id)
                    .map(move |partition| (dispatch, partition))
            })
            .collect::<Vec<_>>();
        let Some((first_dispatch, partition)) = partitions.first().copied() else {
            return distributed_calibration_error(
                "mounted selected-resource requirement has no executable partition",
            );
        };
        if first_dispatch.component_id != requirement_plan.component_id {
            return distributed_calibration_error(
                "mounted selected-resource requirement belongs to a different component",
            );
        }
        let execution_classes = execution_plan
            .selected_resource_execution_classes(&requirement_plan.selector_id)
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?;
        if execution_classes.resource_execution_class_ids
            != requirement_plan.resource_execution_class_ids
        {
            return distributed_calibration_error(
                "mounted selected-resource classes changed after requirement derivation",
            );
        }
        let participant_ids = partitions
            .iter()
            .flat_map(|(dispatch, _)| dispatch.shards.iter().map(|shard| shard.device_id.as_str()))
            .collect::<BTreeSet<_>>();
        if participant_ids.len() < 2 {
            return Ok(None);
        }
        let candidate_capacities = capacities
            .iter()
            .filter(|capacity| participant_ids.contains(capacity.device_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if candidate_capacities.len() != participant_ids.len()
            || candidate_capacities
                .iter()
                .map(|capacity| capacity.device_id.as_str())
                .collect::<BTreeSet<_>>()
                != participant_ids
        {
            return distributed_calibration_error(
                "mounted selected-resource capacity does not cover every compiled participant",
            );
        }
        let Some(devices) = catalog
            .try_selected_resource_placement_devices(
                &requirement_plan.requirements,
                &candidate_capacities,
            )
            .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        else {
            return Ok(None);
        };
        let uniform_prior;
        let domain = if let Some(telemetry) = telemetry {
            let matching = telemetry
                .domains
                .iter()
                .filter(|domain| {
                    domain.execution_scope == partition.execution_scope
                        && domain.component_id == first_dispatch.component_id
                        && domain.node_id == partition.node_id
                        && domain.domain_id == partition.domain_id
                })
                .collect::<Vec<_>>();
            let [domain] = matching.as_slice() else {
                return distributed_calibration_error(format!(
                    "mounted selected-resource selector {:?} has {} exact telemetry domains; expected one",
                    requirement_plan.selector_id,
                    matching.len(),
                ));
            };
            *domain
        } else {
            uniform_prior = uniform_selected_resource_telemetry(
                &first_dispatch.component_id,
                partition,
            )?;
            &uniform_prior
        };
        let Some(placement) = try_plan_selected_resource_placement(
            &first_dispatch.component_id,
            partition,
            &execution_classes,
            domain,
            &devices,
            residency_policy,
            phase,
        )
        .map_err(|error| distributed_calibration_error_value(error.to_string()))?
        else {
            return Ok(None);
        };
        if placement
            .assignments
            .iter()
            .map(|assignment| assignment.device_id.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            < 2
        {
            return Ok(None);
        }
        placements.push(placement);
    }
    Ok(Some(placements))
}

fn uniform_selected_resource_telemetry(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
) -> Result<VulkanSelectionTelemetryDomainSnapshot, VulkanResidentTokenModelPackageError> {
    let pair_count = partition
        .resource_count
        .checked_mul(partition.resource_count.saturating_sub(1))
        .and_then(|count| count.checked_div(2))
        .ok_or_else(|| {
            distributed_calibration_error_value(
                "selected-resource uniform-prior pair count overflowed",
            )
        })?;
    Ok(VulkanSelectionTelemetryDomainSnapshot {
        execution_scope: partition.execution_scope.clone(),
        component_id: component_id.to_string(),
        node_id: partition.node_id.clone(),
        domain_id: partition.domain_id.clone(),
        resource_count: partition.resource_count,
        selection_counts: vec![1; partition.resource_count],
        co_selection_counts: if partition.selection_count_per_activation > 1 {
            vec![1; pair_count]
        } else {
            Vec::new()
        },
    })
}

pub(crate) fn selected_resource_execution_requirement(
    component_signature: &str,
    selector: &CompiledResourceSelector,
    resource_execution_class_id: &str,
    isolated_execution_plan: &VulkanDistributedExecutionPlan,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<VulkanPlacementSelectedResourceExecutionClassRequirement, VulkanResidentTokenModelPackageError>
{
    let first_island = isolated_execution_plan
        .execution_islands
        .first()
        .ok_or_else(|| {
            distributed_calibration_error_value(
                "selected-resource requirement has no physical execution island",
            )
        })?;
    let last_island = isolated_execution_plan
        .execution_islands
        .last()
        .expect("first selected-resource island was checked above");
    Ok(VulkanPlacementSelectedResourceExecutionClassRequirement {
        resource_execution_class_id: resource_execution_class_id.to_string(),
        compiled_execution_signature: selected_resource_compiled_execution_signature(
            component_signature,
            selector,
            resource_execution_class_id,
            isolated_execution_plan,
        )?,
        runtime_implementation_fingerprint: crate::RUNTIME_IMPLEMENTATION_FINGERPRINT.to_string(),
        phase: match phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                nerve_execution_contracts::ExecutionPhase::Decode
            }
            VulkanTargetedComponentExecutionPhase::Prefill { .. } => {
                nerve_execution_contracts::ExecutionPhase::Prefill
            }
        },
        shape: VulkanPlacementShapeClass {
            activation_batch_width: phase.activation_batch_width(),
            input_byte_capacity: first_island.leader().input_byte_capacity,
            output_byte_capacity: last_island.tail().output_byte_capacity,
        },
        artifact_digest: distributed_calibration_artifact_digest(
            loaded_manifest,
            isolated_execution_plan,
        )?,
        execution_graph_digest: selected_resource_execution_graph_digest(
            component_signature,
            selector,
            resource_execution_class_id,
            isolated_execution_plan,
        ),
    })
}

fn selected_resource_execution_calibration_targets_for_classes(
    component: &VulkanRuntimePlacementCalibrationTarget,
    selector_id: &str,
    phase: VulkanTargetedComponentExecutionPhase,
    selected_contract_ids: &BTreeSet<String>,
    resource_execution_class_ids: &[String],
) -> Result<Vec<VulkanRuntimeSelectedResourceExecutionCalibrationTarget>, VulkanResidentTokenModelPackageError>
{
    if component.signature_id.is_empty()
        || component.component_id.is_empty()
        || selector_id.is_empty()
        || selected_contract_ids.is_empty()
        || phase.activation_batch_width() == 0
        || resource_execution_class_ids
            .iter()
            .any(|class_id| !valid_sha256_digest(class_id))
    {
        return distributed_calibration_error(
            "selected-resource representative planning requires exact nonempty identities",
        );
    }
    let mut representative_by_class = BTreeMap::<String, usize>::new();
    for (resource_index, class_id) in resource_execution_class_ids.iter().enumerate() {
        representative_by_class
            .entry(class_id.clone())
            .or_insert(resource_index);
    }
    Ok(representative_by_class
        .into_iter()
        .map(|(resource_execution_class_id, resource_index)| {
            VulkanRuntimeSelectedResourceExecutionCalibrationTarget {
                component: component.clone(),
                selector_id: selector_id.to_string(),
                resource_index,
                resource_execution_class_id,
                phase,
                selected_contract_ids: selected_contract_ids.clone(),
            }
        })
        .collect())
}

#[cfg(test)]
mod runtime_selected_resource_execution_planning_tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn component() -> VulkanRuntimePlacementCalibrationTarget {
        VulkanRuntimePlacementCalibrationTarget {
            signature_id: digest('a'),
            component_id: "representative-layer".to_string(),
            component_ids: vec![
                "representative-layer".to_string(),
                "equivalent-layer".to_string(),
            ],
            terminal_node_id: "down".to_string(),
            implementation: "sparse-ffn".to_string(),
            planned_resident_parameter_bytes: 1024,
        }
    }

    fn partition(selection_count_per_activation: usize) -> VulkanDistributedSelectedResourcePartitionPlan {
        VulkanDistributedSelectedResourcePartitionPlan {
            execution_scope: "model".to_string(),
            selector_id: "experts".to_string(),
            node_id: "router".to_string(),
            domain_id: "routed".to_string(),
            selection_signal: "routes".to_string(),
            address_table_binding: 0,
            parameter_slots_binding: 1,
            resource_count: 4,
            parameters_per_resource: 1,
            parameter_partitions: Vec::new(),
            selection_count_per_activation,
            resource_operation_class_ids: vec![digest('c'); 4],
            atomic_group_ids: (0..4).map(|index| format!("expert-{index}")).collect(),
            atomic_group_byte_counts: vec![16; 4],
            atomic_group_resource_ids: (0..4)
                .map(|index| vec![format!("weight-{index}")])
                .collect(),
            parameter_resource_ids: (0..4)
                .map(|index| vec![format!("weight-{index}")])
                .collect(),
            parameter_resource_byte_counts: vec![vec![16]; 4],
        }
    }

    #[test]
    fn representative_plan_keeps_one_exact_resource_per_execution_class() {
        let class_a = digest('1');
        let class_b = digest('2');
        let targets = selected_resource_execution_calibration_targets_for_classes(
            &component(),
            "experts",
            VulkanTargetedComponentExecutionPhase::Decode,
            &BTreeSet::from(["contract".to_string()]),
            &[class_b.clone(), class_a.clone(), class_b.clone(), class_a.clone()],
        )
        .unwrap();

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].resource_execution_class_id, class_a);
        assert_eq!(targets[0].resource_index, 1);
        assert_eq!(targets[1].resource_execution_class_id, class_b);
        assert_eq!(targets[1].resource_index, 0);
        assert!(targets.iter().all(|target| {
            target.selector_id == "experts"
                && target.selected_contract_ids == BTreeSet::from(["contract".to_string()])
        }));
    }

    #[test]
    fn representative_plan_rejects_an_untyped_class_instead_of_averaging_it() {
        let error = selected_resource_execution_calibration_targets_for_classes(
            &component(),
            "experts",
            VulkanTargetedComponentExecutionPhase::Decode,
            &BTreeSet::from(["contract".to_string()]),
            &[String::new()],
        )
        .unwrap_err();

        assert!(error.to_string().contains("exact nonempty identities"));
    }

    #[test]
    fn unobserved_selector_prior_is_uniform_and_pair_complete() {
        let multi = uniform_selected_resource_telemetry("block", &partition(3)).unwrap();
        assert_eq!(multi.selection_counts, vec![1; 4]);
        assert_eq!(multi.co_selection_counts, vec![1; 6]);
        assert_eq!(multi.component_id, "block");
        assert_eq!(multi.node_id, "router");
        assert_eq!(multi.domain_id, "routed");

        let single = uniform_selected_resource_telemetry("block", &partition(1)).unwrap();
        assert!(single.co_selection_counts.is_empty());
    }
}
