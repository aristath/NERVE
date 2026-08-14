#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanRuntimeFeedbackControlResidencyPlan {
    local_model_dispatch_count: usize,
    local_residency_gate_dispatch_count: usize,
    distributed_model_dispatch_count: usize,
    distributed_residency_gate_dispatch_count: usize,
    input_transducer_dispatch_count: usize,
    output_transducer_dispatch_count: usize,
    sampler_dispatch_count: usize,
    dispatch_capacity: usize,
    byte_capacity: usize,
}

#[allow(clippy::too_many_arguments)]
fn plan_vulkan_runtime_feedback_control_residency(
    runtime_model: &VulkanResidentRuntimeModel,
    resource_contract: &CompiledResourceResidencyContract,
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
    decode_execution_plan: &VulkanDistributedExecutionPlan,
    selected_resource_store_plan: &VulkanDistributedSelectedResourceStorePlan,
    devices: &[VulkanRuntimeSelectedResourceMountDevice],
    input_device_id: &str,
    output_device_id: &str,
    mount_speculative_decoders: bool,
    residency_policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeFeedbackControlResidencyPlan, VulkanResidentTokenModelPackageError> {
    let planned_devices = slice_plans
        .iter()
        .map(|slice| slice.device_id.as_str())
        .collect::<BTreeSet<_>>();
    let supplied_devices = devices
        .iter()
        .map(|device| device.logical_device_id.as_str())
        .collect::<BTreeSet<_>>();
    let owner_devices = runtime_model
        .circuit_graph
        .signal_processor_owner_device_ids(&runtime_model.placement)
        .into_iter()
        .collect::<BTreeSet<_>>();
    if planned_devices.is_empty()
        || planned_devices.len() != slice_plans.len()
        || planned_devices
            != owner_devices
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        || planned_devices
            .iter()
            .any(|device_id| !supplied_devices.contains(device_id))
        || supplied_devices.len() != devices.len()
        || !supplied_devices.contains(input_device_id)
        || !supplied_devices.contains(output_device_id)
        || decode_execution_plan
            .device_ids
            .iter()
            .any(|device_id| !supplied_devices.contains(device_id.as_str()))
    {
        return Err(runtime_feedback_control_residency_error(
            "feedback-control planning requires one prepared slice per signal-processor owner and one device record per decode participant",
        ));
    }

    let mut distributed_indices_by_owner = BTreeMap::<&str, BTreeSet<usize>>::new();
    for island in &decode_execution_plan.execution_islands {
        distributed_indices_by_owner
            .entry(island.owner_device_id.as_str())
            .or_default()
            .extend(island.dispatch_indices());
    }
    let local_model_dispatch_count = slice_plans.iter().try_fold(0usize, |total, slice| {
        let distributed = distributed_indices_by_owner.get(slice.device_id.as_str());
        let local = slice
            .prepared_plan
            .dispatches
            .iter()
            .filter(|dispatch| {
                distributed.is_none_or(|indices| !indices.contains(&dispatch.dispatch_index))
            })
            .count();
        checked_feedback_dispatch_add(total, local, "local model")
    })?;
    let local_residency_gate_dispatch_count = slice_plans.iter().try_fold(
        0usize,
        |total, slice| {
            checked_feedback_dispatch_add(
                total,
                slice
                    .physical_residency_schedule
                    .demand_gate_count(residency_policy),
                "local residency gate",
            )
        },
    )?;

    let mut logical_devices_by_physical = BTreeMap::<&str, BTreeSet<String>>::new();
    for device in devices {
        logical_devices_by_physical
            .entry(device.physical_device_id.as_str())
            .or_default()
            .insert(device.logical_device_id.clone());
    }
    let mut demand_store_devices = BTreeSet::new();
    for logical_devices in logical_devices_by_physical.values() {
        if compiled_resource_selector_ownership_for_device_set(
            runtime_model,
            resource_contract,
            input_device_id,
            output_device_id,
            logical_devices,
            mount_speculative_decoders,
            selected_resource_store_plan,
        )
        .map_err(runtime_feedback_control_residency_error)?
        .is_some()
        {
            demand_store_devices.extend(logical_devices.iter().cloned());
        }
    }
    let mut distributed_model_dispatch_count = 0usize;
    let mut distributed_residency_gate_dispatch_count = 0usize;
    for island in &decode_execution_plan.execution_islands {
        let leader = island.leader();
        if leader.shards.is_empty() || island.dispatches.is_empty() {
            return Err(runtime_feedback_control_residency_error(
                "feedback-control planning found an empty distributed execution island",
            ));
        }
        for shard in &leader.shards {
            distributed_model_dispatch_count = checked_feedback_dispatch_add(
                distributed_model_dispatch_count,
                island.dispatches.len(),
                "distributed model",
            )?;
            if residency_policy.is_demand_loaded()
                && demand_store_devices.contains(&shard.device_id)
            {
                distributed_residency_gate_dispatch_count = checked_feedback_dispatch_add(
                    distributed_residency_gate_dispatch_count,
                    leader.selected_resource_partitions.len(),
                    "distributed residency gate",
                )?;
            }
        }
    }

    let input_transducer_dispatch_count = 1;
    let output_transducer_dispatch_count = 2;
    let sampler_dispatch_count =
        VulkanResidentSamplerRunner::feedback_dispatch_count_for_kernel_roles(
            runtime_model
                .package
                .sampler
                .kernels
                .iter()
                .map(|kernel| kernel.role.as_str()),
            &runtime_model.package.sampler.spec,
        );
    let dispatch_capacity = [
        local_model_dispatch_count,
        local_residency_gate_dispatch_count,
        distributed_model_dispatch_count,
        distributed_residency_gate_dispatch_count,
        input_transducer_dispatch_count,
        output_transducer_dispatch_count,
        sampler_dispatch_count,
    ]
    .into_iter()
    .try_fold(0usize, |total, count| {
        checked_feedback_dispatch_add(total, count, "complete")
    })?;
    let vocabulary_size = runtime_model.package.sampler.spec.logits_byte_capacity
        / std::mem::size_of::<f32>();
    let byte_capacity = resident_feedback_control_byte_capacity(
        vocabulary_size,
        dispatch_capacity,
    )
    .map_err(runtime_feedback_control_residency_error)?;
    Ok(VulkanRuntimeFeedbackControlResidencyPlan {
        local_model_dispatch_count,
        local_residency_gate_dispatch_count,
        distributed_model_dispatch_count,
        distributed_residency_gate_dispatch_count,
        input_transducer_dispatch_count,
        output_transducer_dispatch_count,
        sampler_dispatch_count,
        dispatch_capacity,
        byte_capacity,
    })
}

fn checked_feedback_dispatch_add(
    total: usize,
    count: usize,
    class: &str,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    total.checked_add(count).ok_or_else(|| {
        runtime_feedback_control_residency_error(format!(
            "{class} feedback dispatch capacity overflowed"
        ))
    })
}

fn runtime_feedback_control_residency_error(
    error: impl Display,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(format!(
        "failed workload-free feedback-control residency planning: {error}",
    ))
}
