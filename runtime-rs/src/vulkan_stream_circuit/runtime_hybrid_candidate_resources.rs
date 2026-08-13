struct VulkanRuntimeHybridExactCandidateResourcePlanner<'a> {
    package_root: &'a Path,
    logical_device_id_by_physical_device: &'a BTreeMap<String, String>,
    planning_devices: &'a [VulkanRuntimePhysicalPlanningDevice],
    context_capacity_activations: usize,
}

impl VulkanRuntimeHybridExactCandidateResourcePlanner<'_> {
    fn validate_bindings(&self) -> Result<(), VulkanRuntimeHybridPlacementError> {
        for device in self.planning_devices {
            let physical_device_id = &device.identity.physical_device_id;
            if self
                .logical_device_id_by_physical_device
                .get(physical_device_id)
                != Some(&device.logical_device_id)
            {
                return runtime_hybrid_error(format!(
                    "exact hybrid planning device {physical_device_id:?} is not bound to logical device {:?}",
                    device.logical_device_id,
                ));
            }
        }
        Ok(())
    }

    fn execution_case_is_eligible(
        &self,
        execution_case: &VulkanPlacementExecutionCaseIdentity,
    ) -> bool {
        let eligible_identities = self
            .planning_devices
            .iter()
            .map(|device| &device.identity)
            .collect::<BTreeSet<_>>();
        execution_case
            .devices
            .iter()
            .all(|device| eligible_identities.contains(device))
    }

    fn parameter_requirements(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase: VulkanTargetedComponentExecutionPhase,
        component_ids: &[String],
        execution_case: &VulkanPlacementExecutionCaseIdentity,
        region_execution: Option<&VulkanPlacementRegionExecutionCalibration>,
        tensor_index: &TensorIndex,
        resource_contract: &CompiledResourceResidencyContract,
    ) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
        let component_cases = if let Some(region) = region_execution {
            if region.component_cases.len() != component_ids.len()
                || region.execution_case != *execution_case
            {
                return runtime_hybrid_error(
                    "exact hybrid candidate region does not match its component range",
                );
            }
            region.component_cases.as_slice()
        } else if component_ids.len() == 1 {
            std::slice::from_ref(execution_case)
        } else {
            return runtime_hybrid_error(
                "multi-component exact hybrid candidate has no region execution",
            );
        };

        let identity_by_logical_device = self
            .planning_devices
            .iter()
            .map(|device| (device.logical_device_id.clone(), device.identity.clone()))
            .collect::<BTreeMap<_, _>>();
        if identity_by_logical_device.len() != self.planning_devices.len() {
            return runtime_hybrid_error(
                "exact hybrid candidate resources repeat a logical planning device",
            );
        }
        let storage_buffer_offset_alignment = self
            .planning_devices
            .iter()
            .map(|device| device.storage_buffer_offset_alignment)
            .max()
            .unwrap_or(1);

        let mut requirements = Vec::new();
        for (component_id, component_case) in component_ids.iter().zip(component_cases) {
            requirements.extend(exact_vulkan_runtime_hybrid_component_parameter_requirements(
                self.package_root,
                runtime_model,
                component_id,
                component_case,
                phase,
                self.logical_device_id_by_physical_device,
                &identity_by_logical_device,
                storage_buffer_offset_alignment,
                self.context_capacity_activations,
                tensor_index,
                resource_contract,
            )?);
        }
        Ok(requirements)
    }
}

#[allow(clippy::too_many_arguments)]
fn exact_vulkan_runtime_hybrid_component_parameter_requirements(
    package_root: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    phase: VulkanTargetedComponentExecutionPhase,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
    storage_buffer_offset_alignment: usize,
    context_capacity_activations: usize,
    tensor_index: &TensorIndex,
    resource_contract: &CompiledResourceResidencyContract,
) -> Result<Vec<VulkanHybridSharedRangeRequirement>, VulkanRuntimeHybridPlacementError> {
    let owner_logical_device_id = logical_device_id_by_physical_device
        .get(&execution_case.owner_physical_device_id)
        .ok_or_else(|| {
            VulkanRuntimeHybridPlacementError(format!(
                "exact hybrid candidate owner {:?} has no logical binding",
                execution_case.owner_physical_device_id,
            ))
        })?;
    let participant_logical_device_ids = runtime_hybrid_candidate_participant_logical_device_ids(
        execution_case,
        logical_device_id_by_physical_device,
    )?;
    let mut placed_model = vulkan_runtime_model_with_component_placement(
        runtime_model,
        "hybrid:unmounted",
        &BTreeMap::from([(component_id.to_string(), owner_logical_device_id.clone())]),
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    if execution_case.strategy != VulkanPlacementExecutionStrategy::SingleDevice {
        placed_model = placed_model
            .with_component_shard_devices(component_id, participant_logical_device_ids.clone())
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    }
    let slice_plan = VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
        package_root,
        &placed_model,
        resource_contract,
        tensor_index,
        owner_logical_device_id,
        context_capacity_activations,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let loaded_manifest =
        resident_package_loaded_kernel_manifest_for_slice_plans(std::slice::from_ref(&slice_plan))
            .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
        loaded_manifest
            .physical_artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let graph = placed_model
        .executable_circuit_graph()
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (_, placement_plan, _) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        owner_logical_device_id,
        &placed_model.placement,
        &graph,
        package_root,
        tensor_index,
        placed_model.package.activation_element_bytes,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let (execution_phase, execution_shape) = distributed_contract_phase_and_shape(phase);
    let component_device_pools =
        if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
            BTreeMap::new()
        } else {
            BTreeMap::from([(component_id.to_string(), participant_logical_device_ids)])
        };
    let mut execution_plan =
        VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_with_resource_contract(
            &[(owner_logical_device_id.as_str(), &slice_plan.prepared_plan)],
            tensor_index,
            &artifact_manifest,
            &component_device_pools,
            &placement_plan.edges,
            storage_buffer_offset_alignment,
            execution_phase,
            execution_shape,
            &placed_model.execution_scope,
            resource_contract,
        )
        .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    replay_exact_execution_cases_to_phase(
        &mut execution_plan,
        &BTreeMap::from([(component_id.to_string(), execution_case.clone())]),
        execution_phase,
        identity_by_logical_device,
        &loaded_manifest,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    let execution_plans = VulkanDistributedExecutionPlanSet {
        decode: execution_plan.clone(),
        decode_batch: execution_plan.clone(),
        prefill: execution_plan,
    };
    let mut requirements = vulkan_hybrid_dispatch_parameter_requirements_by_component(
        &[(owner_logical_device_id.as_str(), &slice_plan.prepared_plan)],
        &execution_plans,
        tensor_index,
        identity_by_logical_device,
        |prepared, parameter_id, actual_tensor| {
            exact_vulkan_hybrid_parameter_resource_identity_for_tensor(
                runtime_model,
                resource_contract,
                tensor_index,
                prepared,
                parameter_id,
                actual_tensor,
            )
        },
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    append_vulkan_hybrid_graph_parameter_requirements(
        &placed_model,
        tensor_index,
        resource_contract,
        identity_by_logical_device,
        Some(&BTreeSet::from([component_id.to_string()])),
        &requirements.prepared_parameter_tensors,
        &mut requirements.requirements_by_component,
    )
    .map_err(|error| VulkanRuntimeHybridPlacementError(error.to_string()))?;
    Ok(requirements
        .requirements_by_component
        .get(component_id)
        .cloned()
        .unwrap_or_default())
}

fn runtime_hybrid_candidate_participant_logical_device_ids(
    execution_case: &VulkanPlacementExecutionCaseIdentity,
    logical_device_id_by_physical_device: &BTreeMap<String, String>,
) -> Result<Vec<String>, VulkanRuntimeHybridPlacementError> {
    if execution_case.strategy == VulkanPlacementExecutionStrategy::SingleDevice {
        return logical_device_id_by_physical_device
            .get(&execution_case.owner_physical_device_id)
            .cloned()
            .map(|device| vec![device])
            .ok_or_else(|| {
                VulkanRuntimeHybridPlacementError(
                    "single-device hybrid candidate has no logical owner binding".to_string(),
                )
            });
    }
    let mut physical_by_ordinal = BTreeMap::new();
    for shard in &execution_case.shards {
        if let Some(existing) =
            physical_by_ordinal.insert(shard.participant_ordinal, shard.physical_device_id.as_str())
            && existing != shard.physical_device_id
        {
            return runtime_hybrid_error(
                "exact hybrid candidate participant ordinal changes physical device",
            );
        }
    }
    if physical_by_ordinal.is_empty()
        || physical_by_ordinal
            .keys()
            .copied()
            .ne(0..physical_by_ordinal.len())
    {
        return runtime_hybrid_error(
            "distributed hybrid candidate has an incomplete participant ordering",
        );
    }
    physical_by_ordinal
        .into_values()
        .map(|physical| {
            logical_device_id_by_physical_device
                .get(physical)
                .cloned()
                .ok_or_else(|| {
                    VulkanRuntimeHybridPlacementError(format!(
                        "exact hybrid candidate participant {physical:?} has no logical binding",
                    ))
                })
        })
        .collect()
}
