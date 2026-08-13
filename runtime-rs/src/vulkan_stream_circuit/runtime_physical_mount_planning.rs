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
    pub normal_prefill_lane_capacity: usize,
}

struct VulkanRuntimeSelectedResourceTransientResolution {
    resolution: VulkanRuntimeSelectedResourceMountResolution,
    normal_prefill_lane_capacity: usize,
}

#[allow(clippy::too_many_arguments)]
fn resolve_vulkan_runtime_selected_resources_for_prefill_lane_capacity(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    baseline_execution_plans: &VulkanDistributedExecutionPlanSet,
    residency_plan: &VulkanRuntimeResidencyPlan,
    logical_device_ids: &[String],
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    speculative_decoder_slice_plans:
        &BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    tensor_index: &TensorIndex,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    input_device_id: &str,
    output_device_id: &str,
    normal_prefill_lane_capacity: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
    catalog: Option<&VulkanPlacementCalibrationCatalog>,
    telemetry: Option<&VulkanSelectionTelemetrySnapshot>,
) -> Result<
    (
        VulkanRuntimeSelectedResourceMountResolution,
        VulkanRuntimeHybridExecutionTransientPlan,
    ),
    VulkanResidentTokenModelPackageError,
> {
    let prepared_plans = slice_plans
        .iter()
        .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
        .collect::<Vec<_>>();
    let speculative_catch_up_transient =
        exact_vulkan_runtime_speculative_catch_up_transient_plan(
            runtime_model,
            speculative_decoder_slice_plans,
            normal_prefill_lane_capacity,
            speculative_draft_tokens,
            residency_policy,
        )?;
    let mut execution_transient = exact_vulkan_runtime_mounted_prefill_transient_plan(
        runtime_model,
        slice_plans,
        &baseline_execution_plans.prefill,
        normal_prefill_lane_capacity,
        resource_contract,
        residency_policy,
        speculative_draft_tokens,
    )?;
    execution_transient
        .extend(speculative_catch_up_transient.clone())
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to attach speculative catch-up transients: {error}",
            ))
        })?;
    let mut seen_transients = Vec::new();
    for _ in 0..VULKAN_SELECTED_RESOURCE_MOUNT_PLACEMENT_MAXIMUM_ITERATIONS {
        if seen_transients.contains(&execution_transient) {
            return Err(VulkanResidentTokenModelPackageError::new(
                "selected-resource placement and execution-transient planning entered a cycle",
            ));
        }
        seen_transients.push(execution_transient.clone());
        let resolution = resolve_vulkan_runtime_selected_resource_mount(
            runtime_model,
            resource_contract,
            loaded_manifest,
            baseline_execution_plans.clone(),
            residency_plan,
            logical_device_ids,
            &prepared_plans,
            tensor_index,
            devices,
            input_device_id,
            output_device_id,
            speculative_draft_tokens > 0,
            residency_policy,
            catalog,
            telemetry,
            &execution_transient.device_bytes_by_logical_device,
        )?;
        let mut resolved_transient = exact_vulkan_runtime_mounted_prefill_transient_plan(
            runtime_model,
            slice_plans,
            &resolution.plans.execution_plans.prefill,
            normal_prefill_lane_capacity,
            resource_contract,
            residency_policy,
            speculative_draft_tokens,
        )?;
        resolved_transient
            .extend(speculative_catch_up_transient.clone())
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to attach resolved speculative catch-up transients: {error}",
                ))
            })?;
        if resolved_transient == execution_transient {
            return Ok((resolution, resolved_transient));
        }
        execution_transient = resolved_transient;
    }
    Err(VulkanResidentTokenModelPackageError::new(
        "selected-resource placement and execution-transient planning did not converge",
    ))
}

#[allow(clippy::too_many_arguments)]
fn try_resolve_vulkan_runtime_selected_resources_with_exact_execution_transients(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    baseline_execution_plans: &VulkanDistributedExecutionPlanSet,
    residency_plan: &VulkanRuntimeResidencyPlan,
    logical_device_ids: &[String],
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    speculative_decoder_slice_plans:
        &BTreeMap<String, VulkanResidentModelPackageDeviceSlicePlan>,
    edge_plans: &[VulkanPlacedEdgeIoPlan],
    selected_boundary_routes: &BTreeMap<usize, VulkanRuntimeMountedBoundaryRoute>,
    tensor_index: &TensorIndex,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    input_device_id: &str,
    output_device_id: &str,
    exact_prefill_lane_capacity: Option<usize>,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
    catalog: Option<&VulkanPlacementCalibrationCatalog>,
    telemetry: Option<&VulkanSelectionTelemetrySnapshot>,
    host_safe_capacity_bytes: usize,
) -> Result<Option<VulkanRuntimeSelectedResourceTransientResolution>, VulkanResidentTokenModelPackageError>
{
    let physical_device_by_logical_device = devices
        .iter()
        .map(|device| {
            (
                device.logical_device_id.clone(),
                device.physical_device_id.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let safe_capacity_by_physical_device = devices.iter().fold(
        BTreeMap::<String, usize>::new(),
        |mut capacities, device| {
            capacities
                .entry(device.physical_device_id.clone())
                .and_modify(|capacity| {
                    *capacity = (*capacity).min(device.live_safe_capacity_bytes)
                })
                .or_insert(device.live_safe_capacity_bytes);
            capacities
        },
    );
    for normal_prefill_lane_capacity in vulkan_runtime_normal_prefill_lane_capacity_candidates(
        slice_plans,
        exact_prefill_lane_capacity,
    )? {
        let (mut resolution, execution_transient) =
            resolve_vulkan_runtime_selected_resources_for_prefill_lane_capacity(
                runtime_model,
                resource_contract,
                loaded_manifest,
                baseline_execution_plans,
                residency_plan,
                logical_device_ids,
                slice_plans,
                speculative_decoder_slice_plans,
                tensor_index,
                devices,
                input_device_id,
                output_device_id,
                normal_prefill_lane_capacity,
                speculative_draft_tokens,
                residency_policy,
                catalog,
                telemetry,
            )?;
        let feedback_control = plan_vulkan_runtime_feedback_control_residency(
            runtime_model,
            resource_contract,
            slice_plans,
            &resolution.plans.execution_plans.decode,
            &resolution.plans.selected_resource_store_plan,
            devices,
            input_device_id,
            output_device_id,
            speculative_draft_tokens > 0,
            residency_policy,
        )?;
        resolution
            .plans
            .physical_execution_residency_plan
            .resize_feedback_control_residency(feedback_control.byte_capacity)
            .map_err(|error| {
                physical_mount_planning_error("exact feedback-control residency", error)
            })?;
        resolution
            .plans
            .physical_execution_residency_plan
            .add_execution_transient_reservation(
                &execution_transient.device_allocations,
                &execution_transient.shared_host_allocations,
            )
            .map_err(|error| {
                physical_mount_planning_error("execution transient residency", error)
            })?;
        let activation_plan = resolution.plans.activation_plan.clone();
        resolution
            .plans
            .physical_execution_residency_plan
            .bind_graph_edge_memory_domains(
                edge_plans,
                &activation_plan,
                selected_boundary_routes,
                &physical_device_by_logical_device,
            )
            .map_err(|error| {
                physical_mount_planning_error("graph-edge memory-domain binding", error)
            })?;
        resolution
            .plans
            .physical_execution_residency_plan
            .bind_feedback_control_memory_domain(&physical_device_by_logical_device)
            .map_err(|error| {
                physical_mount_planning_error("feedback-control memory-domain binding", error)
            })?;
        resolution
            .plans
            .physical_execution_residency_plan
            .bind_stream_control_memory_domain(&physical_device_by_logical_device)
            .map_err(|error| {
                physical_mount_planning_error("stream-control memory-domain binding", error)
            })?;
        if runtime_physical_mount_fits(
            &resolution.plans.physical_execution_residency_plan,
            &physical_device_by_logical_device,
            &safe_capacity_by_physical_device,
            host_safe_capacity_bytes,
        )? {
            return Ok(Some(VulkanRuntimeSelectedResourceTransientResolution {
                resolution,
                normal_prefill_lane_capacity,
            }));
        }
    }
    Ok(None)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimePhysicalSelectedResourceSummary {
    maximum_load_wave_bytes_by_logical_device: BTreeMap<String, usize>,
    uses_shared_host_cache: bool,
}

#[allow(clippy::too_many_arguments)]
fn plan_vulkan_runtime_physical_mount(
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
    let mut speculative_decoder_slice_plans = BTreeMap::new();
    if speculative_draft_tokens > 0 {
        for decoder in &runtime_model.package.speculative_decoders {
            let plan = VulkanResidentSpeculativeDecoderModelPackage::prepare_device_slice_for_physical_planning(
                manifest_dir,
                runtime_model,
                &tensor_index,
                decoder,
                &output_device_id,
                context_capacity_activations,
            )?;
            if speculative_decoder_slice_plans
                .insert(decoder.id.clone(), plan)
                .is_some()
            {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "physical mount repeats speculative decoder {:?}",
                    decoder.id,
                )));
            }
        }
    }
    let prepared_plans = slice_plans
        .iter()
        .map(|slice| (slice.device_id.as_str(), &slice.prepared_plan))
        .collect::<Vec<_>>();
    let loaded_manifest = resident_package_loaded_kernel_manifest_for_slice_plans(&slice_plans)?;
    let edge_plans = slice_plans
        .iter()
        .map(|slice| {
            VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
                &slice.placed_plan.placed_resident_plan,
            )
            .map_err(|error| physical_mount_planning_error("graph-edge planning", error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mounted_boundary_routes = physical_execution_plan
        .mounted_boundary_routes()
        .map_err(|error| physical_mount_planning_error("physical boundary routing", error))?;
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
    let Some(transient_resolution) =
        try_resolve_vulkan_runtime_selected_resources_with_exact_execution_transients(
        runtime_model,
        &resource_contract,
        &loaded_manifest,
        &execution_plans,
        &residency_plan,
        &physical_execution_plan.device_ids(runtime_model),
        &slice_plans,
        &speculative_decoder_slice_plans,
        &edge_plans,
        &mounted_boundary_routes,
        &tensor_index,
        &mount_devices,
        &input_device_id,
        &output_device_id,
        physical_execution_plan.prefill_activation_batch_width,
        speculative_draft_tokens,
        resource_residency_policy,
        placement_calibration_catalog,
        None,
        host_safe_capacity_bytes,
    )?
    else {
        return Ok(None);
    };
    let resolution = transient_resolution.resolution;
    let final_capacities = selected_resource_mount_capacities(
        runtime_model,
        &resource_contract,
        &resolution.plans,
        &mount_devices,
        &input_device_id,
        &output_device_id,
        speculative_draft_tokens > 0,
        resource_residency_policy,
        &BTreeMap::new(),
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
        normal_prefill_lane_capacity: transient_resolution.normal_prefill_lane_capacity,
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
