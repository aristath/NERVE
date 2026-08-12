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
}
