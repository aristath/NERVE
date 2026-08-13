#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePhysicalPlanningDevice {
    pub logical_device_id: String,
    pub identity: VulkanPlacementDeviceExecutionIdentity,
    pub safe_capacity_bytes: usize,
    pub storage_buffer_offset_alignment: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePhysicalMountPlan {
    pub physical_execution_residency_plan: VulkanRuntimePhysicalExecutionResidencyPlan,
    pub exact_parameter_resources_by_component: BTreeMap<String, VulkanHybridCandidateResources>,
    pub selected_resource_placements: Vec<VulkanSelectedResourcePlacementPlan>,
    pub selected_resource_cache_quota_bytes_by_logical_device: BTreeMap<String, usize>,
    pub maximum_load_wave_bytes_by_logical_device: BTreeMap<String, usize>,
    pub shared_host_cache_quota_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimePhysicalSelectedResourceSummary {
    maximum_load_wave_bytes_by_logical_device: BTreeMap<String, usize>,
    uses_shared_host_cache: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn plan_vulkan_runtime_physical_mount(
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    physical_execution_plan: &VulkanRuntimePhysicalExecutionPlan,
    placement_calibration_catalog: Option<&VulkanPlacementCalibrationCatalog>,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    resource_residency_policy: ResourceResidencyPolicy,
    devices: &[VulkanRuntimePhysicalPlanningDevice],
    host_safe_capacity_bytes: usize,
) -> Result<Option<VulkanRuntimePhysicalMountPlan>, VulkanResidentTokenModelPackageError> {
    let manifest_dir = manifest_dir.as_ref();
    physical_execution_plan
        .validate(runtime_model)
        .map_err(|error| physical_mount_planning_error("physical execution validation", error))?;
    validate_runtime_physical_planning_devices(runtime_model, physical_execution_plan, devices)?;
    let input_component_id = runtime_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()?
        .0;
    let output_component_id = runtime_model
        .circuit_graph
        .signal_processor_endpoint_component_ids()?
        .1;
    let input_device_id = runtime_model
        .placement
        .device_for_component(&input_component_id)
        .to_string();
    let output_device_id = runtime_model
        .placement
        .device_for_component(&output_component_id)
        .to_string();
    let tensor_index = runtime_model.load_runtime_tensor_index(manifest_dir)?;
    let resource_contract = instantiate_runtime_resource_contract(runtime_model)
        .map_err(|error| physical_mount_planning_error("compiled resource contract", error))?;
    let residency_plan = plan_vulkan_runtime_residency_with_contract(
        manifest_dir,
        runtime_model,
        &tensor_index,
        context_capacity_activations,
        speculative_draft_tokens,
        resource_residency_policy,
        &resource_contract,
    )
    .map_err(|error| physical_mount_planning_error("base residency", error))?;
    let (_, placement_plan, _) = plan_resident_package_placed_stream_circuit_with_tensor_index(
        &input_device_id,
        &runtime_model.placement,
        &runtime_model.circuit_graph,
        manifest_dir,
        &tensor_index,
        runtime_model.package.activation_element_bytes,
    )?;
    let owner_device_ids = runtime_model
        .circuit_graph
        .signal_processor_owner_device_ids(&runtime_model.placement);
    let mut slice_plans = Vec::with_capacity(owner_device_ids.len());
    for device_id in &owner_device_ids {
        slice_plans.push(
            VulkanResidentModelPackageDeviceSlicePlan::prepare_for_physical_planning(
                manifest_dir,
                runtime_model,
                &resource_contract,
                &tensor_index,
                device_id,
                context_capacity_activations,
            )?,
        );
    }
    let prepared_plans = slice_plans
        .iter()
        .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
        .collect::<Vec<_>>();
    let loaded_manifest = resident_package_loaded_kernel_manifest_for_slice_plans(&slice_plans)?;
    let artifact_manifest = VulkanPhysicalKernelArtifactManifest::new(
        loaded_manifest
            .physical_artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let storage_buffer_offset_alignment = devices
        .iter()
        .map(|device| device.storage_buffer_offset_alignment)
        .max()
        .unwrap_or(1);
    let mut execution_plans = VulkanDistributedExecutionPlanSet::from_prepared_plans_with_resource_contract_and_execution_cases(
        &prepared_plans,
        &tensor_index,
        &artifact_manifest,
        &physical_execution_plan.component_device_pools,
        &placement_plan.edges,
        storage_buffer_offset_alignment,
        &runtime_model.execution_scope,
        &resource_contract,
        &physical_execution_plan.decode_execution_cases_by_component,
        &physical_execution_plan.decode_batch_execution_cases_by_component,
        &physical_execution_plan.prefill_execution_cases_by_component,
    )
    .map_err(|error| physical_mount_planning_error("distributed execution planning", error))?;
    let identity_by_logical_device = devices
        .iter()
        .map(|device| (device.logical_device_id.clone(), device.identity.clone()))
        .collect::<BTreeMap<_, _>>();
    physical_execution_plan
        .validate_bound_boundary_device_identities(&identity_by_logical_device)
        .map_err(|error| physical_mount_planning_error("physical boundary validation", error))?;
    execution_plans
        .apply_exact_execution_cases(
            &physical_execution_plan.decode_execution_cases_by_component,
            &physical_execution_plan.decode_batch_execution_cases_by_component,
            &physical_execution_plan.prefill_execution_cases_by_component,
            &identity_by_logical_device,
            &loaded_manifest,
        )
        .map_err(|error| physical_mount_planning_error("exact execution replay", error))?;
    let exact_parameter_resources_by_component =
        vulkan_runtime_hybrid_parameter_resources_by_component(
            runtime_model,
            &prepared_plans,
            &execution_plans,
            &tensor_index,
            &resource_contract,
            &identity_by_logical_device,
        )
        .map_err(|error| physical_mount_planning_error("exact parameter resources", error))?;

    let contract_alignment =
        compiled_resource_contract_minimum_upload_alignment(&resource_contract)
            .map_err(|error| physical_mount_planning_error("resource upload alignment", error))?;
    let mount_devices = devices
        .iter()
        .map(|device| VulkanRuntimeSelectedResourceMountDevice {
            logical_device_id: device.logical_device_id.clone(),
            physical_device_id: device.identity.physical_device_id.clone(),
            execution_identity: device.identity.clone(),
            live_safe_capacity_bytes: device.safe_capacity_bytes,
            upload_alignment: device
                .storage_buffer_offset_alignment
                .max(contract_alignment)
                .max(std::mem::align_of::<u64>()),
        })
        .collect::<Vec<_>>();
    let resolution = resolve_vulkan_runtime_selected_resource_mount(
        runtime_model,
        &resource_contract,
        &loaded_manifest,
        execution_plans,
        &residency_plan,
        &physical_execution_plan.device_ids(runtime_model),
        &prepared_plans,
        &tensor_index,
        &mount_devices,
        &input_device_id,
        &output_device_id,
        speculative_draft_tokens > 0,
        resource_residency_policy,
        placement_calibration_catalog,
        None,
    )?;
    let physical_device_by_logical_device = devices
        .iter()
        .map(|device| {
            (
                device.logical_device_id.clone(),
                device.identity.physical_device_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let safe_capacity_by_physical_device = devices
        .iter()
        .map(|device| {
            (
                device.identity.physical_device_id.clone(),
                device.safe_capacity_bytes,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if !runtime_physical_mount_fits(
        &resolution.plans.physical_execution_residency_plan,
        &physical_device_by_logical_device,
        &safe_capacity_by_physical_device,
        host_safe_capacity_bytes,
    )? {
        return Ok(None);
    }
    let final_capacities = selected_resource_mount_capacities(
        runtime_model,
        &resource_contract,
        &resolution.plans,
        &mount_devices,
        &input_device_id,
        &output_device_id,
        speculative_draft_tokens > 0,
        resource_residency_policy,
    )?
    .ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(
            "selected-resource physical mount lost its final cache capacity",
        )
    })?;
    let selected_resource_cache_quota_bytes_by_logical_device = final_capacities
        .iter()
        .map(|capacity| {
            (
                capacity.device_id.clone(),
                capacity.resident_payload_capacity_bytes,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let selected_resource_summary = summarize_vulkan_runtime_physical_selected_resources(
        &resolution.plans.selected_resource_store_plan,
        &selected_resource_cache_quota_bytes_by_logical_device,
        resource_residency_policy,
    );
    let shared_host_cache_quota_bytes = if selected_resource_summary.uses_shared_host_cache {
        let remaining = remaining_vulkan_runtime_host_cache_bytes(
            host_safe_capacity_bytes,
            resolution
                .plans
                .physical_execution_residency_plan
                .total_stream_shared_host_bytes,
        )?;
        if remaining == 0 {
            return Ok(None);
        }
        remaining
    } else {
        0
    };
    Ok(Some(VulkanRuntimePhysicalMountPlan {
        physical_execution_residency_plan: resolution.plans.physical_execution_residency_plan,
        exact_parameter_resources_by_component,
        selected_resource_placements: resolution.placements,
        selected_resource_cache_quota_bytes_by_logical_device,
        maximum_load_wave_bytes_by_logical_device: selected_resource_summary
            .maximum_load_wave_bytes_by_logical_device,
        shared_host_cache_quota_bytes,
    }))
}

fn remaining_vulkan_runtime_host_cache_bytes(
    safe_host_capacity_bytes: usize,
    stream_shared_host_bytes: usize,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    safe_host_capacity_bytes
        .checked_sub(stream_shared_host_bytes)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime stream needs {stream_shared_host_bytes} shared-host bytes but only {safe_host_capacity_bytes} safe host bytes are available",
            ))
        })
}

fn summarize_vulkan_runtime_physical_selected_resources(
    store_plan: &VulkanDistributedSelectedResourceStorePlan,
    cache_quota_bytes_by_logical_device: &BTreeMap<String, usize>,
    residency_policy: ResourceResidencyPolicy,
) -> VulkanRuntimePhysicalSelectedResourceSummary {
    let maximum_load_wave_bytes_by_logical_device = store_plan
        .devices
        .iter()
        .filter(|device| device.maximum_load_wave_bytes > 0)
        .map(|device| (device.device_id.clone(), device.maximum_load_wave_bytes))
        .collect::<BTreeMap<_, _>>();
    let uses_shared_host_cache = residency_policy == ResourceResidencyPolicy::DemandPaged
        && store_plan.devices.iter().any(|device| {
            cache_quota_bytes_by_logical_device
                .get(&device.device_id)
                .is_some_and(|quota| device.total_addressable_bytes > *quota)
        });
    VulkanRuntimePhysicalSelectedResourceSummary {
        maximum_load_wave_bytes_by_logical_device,
        uses_shared_host_cache,
    }
}

fn runtime_physical_mount_fits(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
    host_safe_capacity_bytes: usize,
) -> Result<bool, VulkanResidentTokenModelPackageError> {
    let mut required_by_physical_device = BTreeMap::<String, usize>::new();
    for device in &plan.device_plans {
        let physical_device_id = physical_device_by_logical_device
            .get(&device.device_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "physical mount residency has no binding for logical device {:?}",
                    device.device_id,
                ))
            })?;
        let required = device
            .mount_device_local_bytes
            .checked_add(device.stream_device_local_bytes)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "physical mount device capacity overflowed",
                )
            })?;
        let total = required_by_physical_device
            .entry(physical_device_id.clone())
            .or_default();
        *total = total.checked_add(required).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new("physical mount device capacity overflowed")
        })?;
    }
    if required_by_physical_device
        .iter()
        .any(|(device, required)| {
            safe_capacity_by_physical_device
                .get(device)
                .is_none_or(|capacity| required > capacity)
        })
    {
        return Ok(false);
    }
    Ok(plan.total_stream_shared_host_bytes <= host_safe_capacity_bytes)
}

fn validate_runtime_physical_planning_devices(
    runtime_model: &VulkanResidentRuntimeModel,
    physical_execution_plan: &VulkanRuntimePhysicalExecutionPlan,
    devices: &[VulkanRuntimePhysicalPlanningDevice],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let required = physical_execution_plan
        .device_ids(runtime_model)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let supplied = devices
        .iter()
        .map(|device| device.logical_device_id.clone())
        .collect::<BTreeSet<_>>();
    let physical = devices
        .iter()
        .map(|device| device.identity.physical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    if required != supplied
        || supplied.len() != devices.len()
        || physical.len() != devices.len()
        || devices.iter().any(|device| {
            device.logical_device_id.trim().is_empty()
                || device.identity.physical_device_id.trim().is_empty()
                || device.safe_capacity_bytes == 0
                || !device.storage_buffer_offset_alignment.is_power_of_two()
        })
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "physical mount planning requires one exact positive-capacity device record per mounted logical device",
        ));
    }
    Ok(())
}

fn physical_mount_planning_error(
    stage: &str,
    error: impl Display,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(format!("failed workload-free {stage}: {error}",))
}
