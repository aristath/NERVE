pub const VULKAN_RUNTIME_PHYSICAL_EXECUTION_RESIDENCY_PLAN_SCHEMA: &str =
    "nerve.vulkan_runtime_physical_execution_residency_plan.v3";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionResidencyBreakdown {
    pub owner_parameter_bytes_before_distributed_replacement: usize,
    pub excluded_owner_parameter_bytes: usize,
    pub independently_admitted_resource_store_bytes: usize,
    pub owner_stream_device_bytes: usize,
    pub owner_stream_control_device_bytes_per_stream: usize,
    pub distributed_parameter_bytes: usize,
    pub distributed_shared_activation_device_bytes_per_stream: usize,
    pub distributed_private_activation_device_bytes_per_stream: usize,
    pub distributed_shared_host_bytes_per_stream: usize,
    pub execution_transient_device_bytes_per_stream: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionDeviceResidencyPlan {
    pub device_id: String,
    pub breakdown: VulkanRuntimePhysicalExecutionResidencyBreakdown,
    pub mount_device_local_bytes: usize,
    pub stream_device_local_bytes: usize,
    pub stream_shared_host_bytes: usize,
    pub execution_transient_device_allocations:
        Vec<VulkanRuntimeDeviceLocalTransientAllocation>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimePhysicalExecutionResidencyPlan {
    pub schema: String,
    pub package_id: String,
    pub device_plans: Vec<VulkanRuntimePhysicalExecutionDeviceResidencyPlan>,
    pub total_mount_device_local_bytes: usize,
    pub total_stream_device_local_bytes: usize,
    pub total_stream_shared_host_bytes: usize,
    pub execution_transient_shared_host_bytes_per_stream: usize,
    pub execution_transient_shared_host_allocations:
        Vec<VulkanRuntimeSharedHostTransientAllocation>,
    pub shared_stream_control_host_bytes_per_stream: usize,
}

impl VulkanRuntimePhysicalExecutionResidencyPlan {
    pub fn plan(
        base: &VulkanRuntimeResidencyPlan,
        logical_device_ids: &[String],
        parameter_allocations: &VulkanDistributedParameterAllocationPlan,
        parameter_exclusions: &VulkanDistributedParameterExclusionPlan,
        activations: &VulkanDistributedActivationBufferPlan,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        let allowed = logical_device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if allowed.is_empty()
            || allowed.len() != logical_device_ids.len()
            || allowed.iter().any(|device_id| device_id.trim().is_empty())
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "physical execution residency requires unique nonempty logical devices".to_string(),
            ));
        }
        validate_physical_execution_residency_inputs(
            parameter_allocations,
            parameter_exclusions,
            activations,
        )?;
        let mut breakdowns = logical_device_ids
            .iter()
            .cloned()
            .map(|device_id| {
                (
                    device_id,
                    VulkanRuntimePhysicalExecutionResidencyBreakdown::default(),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut base_devices = BTreeSet::new();
        for device in &base.device_plans {
            ensure_physical_residency_device(&allowed, &device.device_id, "base residency")?;
            if !base_devices.insert(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency repeats logical device {:?}",
                    device.device_id
                )));
            }
            let store_bytes = match base.residency_policy {
                ResourceResidencyPolicy::Eager => {
                    device.resource_store.maximum_extra_device_bytes()?
                }
                ResourceResidencyPolicy::DemandPaged | ResourceResidencyPolicy::DemandRetained => {
                    device.resource_store.fixed_device_bytes()?
                }
            };
            let owner_stream_device_bytes = checked_residency_add(
                device.working_set.transient_state_bytes,
                device.working_set.activation_headroom_bytes,
                "owner per-stream residency",
            )?;
            let expected_current_parameter_bytes = checked_residency_add(
                device.parameter_residency.always_resident_bytes,
                device.parameter_residency.initial_dynamic_bytes,
                "current parameter residency",
            )?;
            if expected_current_parameter_bytes != device.parameter_residency.current_resident_bytes
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency on {:?} declares {} current parameter bytes but its static and initial dynamic parameters require {expected_current_parameter_bytes}",
                    device.device_id, device.parameter_residency.current_resident_bytes
                )));
            }
            let independently_admitted_resource_store_bytes = checked_residency_add(
                store_bytes,
                device.parameter_residency.initial_dynamic_bytes,
                "independently admitted compiled-resource residency",
            )?;
            let expected_initial_bytes = [
                device.parameter_residency.always_resident_bytes,
                independently_admitted_resource_store_bytes,
                owner_stream_device_bytes,
            ]
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                checked_residency_add(total, bytes, "physical execution base residency")
            })?;
            if expected_initial_bytes != device.initial_device_resident_bytes {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "base residency on {:?} declares {} initial bytes but its parameters, independent store, and stream working set require {expected_initial_bytes}",
                    device.device_id, device.initial_device_resident_bytes
                )));
            }
            let breakdown = breakdowns
                .get_mut(&device.device_id)
                .expect("base device was validated against the physical execution set");
            breakdown.owner_parameter_bytes_before_distributed_replacement =
                device.parameter_residency.always_resident_bytes;
            breakdown.independently_admitted_resource_store_bytes =
                independently_admitted_resource_store_bytes;
            breakdown.owner_stream_device_bytes = owner_stream_device_bytes;
            breakdown.owner_stream_control_device_bytes_per_stream =
                device.breakdown.stream_control_bytes;
        }

        let mut exclusion_devices = BTreeSet::new();
        for device in &parameter_exclusions.devices {
            ensure_physical_residency_device(
                &allowed,
                &device.device_id,
                "distributed parameter exclusion",
            )?;
            if !base_devices.contains(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed parameter exclusion targets helper device {:?} without owner residency",
                    device.device_id
                )));
            }
            if !exclusion_devices.insert(device.device_id.as_str()) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed parameter exclusions repeat logical device {:?}",
                    device.device_id
                )));
            }
            let breakdown = breakdowns
                .get_mut(&device.device_id)
                .expect("exclusion device was validated above");
            if device.total_byte_capacity
                > breakdown.owner_parameter_bytes_before_distributed_replacement
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed replacement excludes {} bytes from {:?}, but its owner parameter residency contains only {} bytes",
                    device.total_byte_capacity,
                    device.device_id,
                    breakdown.owner_parameter_bytes_before_distributed_replacement
                )));
            }
            breakdown.excluded_owner_parameter_bytes = device.total_byte_capacity;
        }

        for allocation in &parameter_allocations.allocations {
            ensure_physical_residency_device(
                &allowed,
                &allocation.device_id,
                "distributed parameter allocation",
            )?;
            let breakdown = breakdowns
                .get_mut(&allocation.device_id)
                .expect("parameter device was validated above");
            breakdown.distributed_parameter_bytes = checked_residency_add(
                breakdown.distributed_parameter_bytes,
                allocation.byte_count,
                "distributed parameter residency",
            )?;
        }

        for allocation in &activations.allocations {
            ensure_physical_residency_activation_devices(
                &allowed,
                &allocation.owner_device_id,
                &allocation.device_ids,
                "distributed activation",
            )?;
            add_physical_residency_shared_activation(
                &mut breakdowns,
                &allocation.owner_device_id,
                allocation.byte_capacity,
                activations.route,
            )?;
        }
        for allocation in &activations.reduction_allocations {
            ensure_physical_residency_activation_devices(
                &allowed,
                &allocation.owner_device_id,
                &allocation.device_ids,
                "distributed reduction",
            )?;
            add_physical_residency_shared_activation(
                &mut breakdowns,
                &allocation.owner_device_id,
                allocation.byte_capacity,
                activations.route,
            )?;
        }
        for allocation in &activations.private_intermediate_allocations {
            if allocation.devices.is_empty() {
                return Err(VulkanRuntimeResidencyPlanError(
                    "distributed private activation has no devices".to_string(),
                ));
            }
            let mut private_devices = BTreeSet::new();
            for device in &allocation.devices {
                ensure_physical_residency_device(
                    &allowed,
                    &device.device_id,
                    "distributed private activation",
                )?;
                if !private_devices.insert(device.device_id.as_str()) || device.byte_capacity == 0 {
                    return Err(VulkanRuntimeResidencyPlanError(
                        "distributed private activation has duplicate devices or empty storage"
                            .to_string(),
                    ));
                }
                let breakdown = breakdowns
                    .get_mut(&device.device_id)
                    .expect("private activation device was validated above");
                breakdown.distributed_private_activation_device_bytes_per_stream =
                    checked_residency_add(
                        breakdown.distributed_private_activation_device_bytes_per_stream,
                        device.byte_capacity,
                        "distributed private activation residency",
                    )?;
            }
        }

        let mut total_mount_device_local_bytes = 0usize;
        let mut total_stream_device_local_bytes = 0usize;
        let mut total_stream_shared_host_bytes = 0usize;
        let mut device_plans = Vec::with_capacity(breakdowns.len());
        for (device_id, breakdown) in breakdowns {
            let retained_owner_bytes = breakdown
                .owner_parameter_bytes_before_distributed_replacement
                .checked_sub(breakdown.excluded_owner_parameter_bytes)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "distributed owner replacement accounting underflowed".to_string(),
                    )
                })?;
            let mount_device_local_bytes = checked_residency_add(
                retained_owner_bytes,
                breakdown.distributed_parameter_bytes,
                "physical execution mount residency",
            )?;
            let stream_device_local_bytes = checked_residency_add(
                breakdown.owner_stream_device_bytes,
                checked_residency_add(
                    breakdown.distributed_shared_activation_device_bytes_per_stream,
                    breakdown.distributed_private_activation_device_bytes_per_stream,
                    "distributed physical execution per-stream residency",
                )?,
                "physical execution per-stream residency",
            )?;
            let stream_shared_host_bytes = breakdown.distributed_shared_host_bytes_per_stream;
            total_mount_device_local_bytes = checked_residency_add(
                total_mount_device_local_bytes,
                mount_device_local_bytes,
                "physical execution total mount residency",
            )?;
            total_stream_device_local_bytes = checked_residency_add(
                total_stream_device_local_bytes,
                stream_device_local_bytes,
                "physical execution total per-stream residency",
            )?;
            total_stream_shared_host_bytes = checked_residency_add(
                total_stream_shared_host_bytes,
                stream_shared_host_bytes,
                "physical execution total shared-host residency",
            )?;
            device_plans.push(VulkanRuntimePhysicalExecutionDeviceResidencyPlan {
                device_id,
                breakdown,
                mount_device_local_bytes,
                stream_device_local_bytes,
                stream_shared_host_bytes,
                execution_transient_device_allocations: Vec::new(),
            });
        }
        Ok(Self {
            schema: VULKAN_RUNTIME_PHYSICAL_EXECUTION_RESIDENCY_PLAN_SCHEMA.to_string(),
            package_id: base.package_id.clone(),
            device_plans,
            total_mount_device_local_bytes,
            total_stream_device_local_bytes,
            total_stream_shared_host_bytes,
            execution_transient_shared_host_bytes_per_stream: 0,
            execution_transient_shared_host_allocations: Vec::new(),
            shared_stream_control_host_bytes_per_stream: 0,
        })
    }

    fn bind_stream_control_memory_domain(
        &mut self,
        physical_device_by_logical_device: &BTreeMap<String, String>,
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        let mut next = self.clone();
        let mut indices_by_physical_device = BTreeMap::<String, Vec<usize>>::new();
        for (index, device) in next.device_plans.iter().enumerate() {
            let physical_device_id = physical_device_by_logical_device
                .get(&device.device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "stream-control binding has no physical device for {:?}",
                        device.device_id,
                    ))
                })?;
            indices_by_physical_device
                .entry(physical_device_id.clone())
                .or_default()
                .push(index);
        }
        if indices_by_physical_device.is_empty()
            || physical_device_by_logical_device.len() != next.device_plans.len()
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "stream-control binding is incomplete or contains extra logical devices"
                    .to_string(),
            ));
        }

        let spans_physical_devices = indices_by_physical_device.len() > 1;
        let mut removed_device_bytes = 0usize;
        for indices in indices_by_physical_device.values() {
            let retained_index = (!spans_physical_devices)
                .then(|| {
                    indices.iter().copied().find(|index| {
                        next.device_plans[*index]
                            .breakdown
                            .owner_stream_control_device_bytes_per_stream
                            > 0
                    })
                })
                .flatten();
            for index in indices {
                if Some(*index) == retained_index {
                    continue;
                }
                let device = &mut next.device_plans[*index];
                let control_bytes = device
                    .breakdown
                    .owner_stream_control_device_bytes_per_stream;
                device.breakdown.owner_stream_control_device_bytes_per_stream = 0;
                device.breakdown.owner_stream_device_bytes = device
                    .breakdown
                    .owner_stream_device_bytes
                    .checked_sub(control_bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "stream-control bytes exceed owner working set on {:?}",
                            device.device_id,
                        ))
                    })?;
                device.stream_device_local_bytes = device
                    .stream_device_local_bytes
                    .checked_sub(control_bytes)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "stream-control bytes exceed stream residency on {:?}",
                            device.device_id,
                        ))
                    })?;
                removed_device_bytes = checked_residency_add(
                    removed_device_bytes,
                    control_bytes,
                    "physically bound stream-control residency",
                )?;
            }
        }
        next.total_stream_device_local_bytes = next
            .total_stream_device_local_bytes
            .checked_sub(removed_device_bytes)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "stream-control total device residency underflowed".to_string(),
                )
            })?;

        if spans_physical_devices {
            if next.shared_stream_control_host_bytes_per_stream != 0 {
                return Err(VulkanRuntimeResidencyPlanError(
                    "stream-control memory domain was bound more than once".to_string(),
                ));
            }
            let representative = next.device_plans.first_mut().ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "stream-control binding has no representative device".to_string(),
                )
            })?;
            representative.stream_shared_host_bytes = checked_residency_add(
                representative.stream_shared_host_bytes,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                "shared stream-control device residency",
            )?;
            next.total_stream_shared_host_bytes = checked_residency_add(
                next.total_stream_shared_host_bytes,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                "shared stream-control total residency",
            )?;
            next.shared_stream_control_host_bytes_per_stream =
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY;
        }
        *self = next;
        Ok(())
    }

    fn add_execution_transient_reservation(
        &mut self,
        device_allocations: &[VulkanRuntimeDeviceLocalTransientAllocation],
        shared_host_allocations: &[VulkanRuntimeSharedHostTransientAllocation],
    ) -> Result<(), VulkanRuntimeResidencyPlanError> {
        // Construct the augmented plan off to the side. A stale logical
        // binding or arithmetic failure must not partially mutate the
        // authoritative admission plan.
        let mut next = self.clone();
        if next.execution_transient_shared_host_bytes_per_stream != 0
            || !next
                .execution_transient_shared_host_allocations
                .is_empty()
            || next.device_plans.iter().any(|device| {
                device
                    .breakdown
                    .execution_transient_device_bytes_per_stream
                    != 0
                    || !device.execution_transient_device_allocations.is_empty()
            })
        {
            return Err(VulkanRuntimeResidencyPlanError(
                "execution transient reservation was already attached".to_string(),
            ));
        }
        for allocation in device_allocations {
            if allocation.logical_device_id.trim().is_empty()
                || allocation.concern.trim().is_empty()
                || allocation.byte_capacity == 0
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "execution transient device allocation {:?} is malformed",
                    allocation.concern,
                )));
            }
            let device = next
                .device_plans
                .iter_mut()
                .find(|device| device.device_id == allocation.logical_device_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "execution transient reservation references absent logical device {:?}",
                        allocation.logical_device_id,
                    ))
                })?;
            device.breakdown.execution_transient_device_bytes_per_stream =
                checked_residency_add(
                    device
                        .breakdown
                        .execution_transient_device_bytes_per_stream,
                    allocation.byte_capacity,
                    "execution transient device residency",
                )?;
            device.stream_device_local_bytes = checked_residency_add(
                device.stream_device_local_bytes,
                allocation.byte_capacity,
                "execution transient stream residency",
            )?;
            next.total_stream_device_local_bytes = checked_residency_add(
                next.total_stream_device_local_bytes,
                allocation.byte_capacity,
                "execution transient total stream residency",
            )?;
            device
                .execution_transient_device_allocations
                .push(allocation.clone());
        }
        let admitted_logical_devices = next
            .device_plans
            .iter()
            .map(|device| device.device_id.as_str())
            .collect::<BTreeSet<_>>();
        let shared_host_bytes = shared_host_allocations.iter().try_fold(
            0usize,
            |total, allocation| {
                if allocation.owner_device_id.trim().is_empty()
                    || allocation.concern.trim().is_empty()
                    || allocation.byte_capacity == 0
                    || !allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| device_id == &allocation.owner_device_id)
                    || allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| device_id.trim().is_empty())
                    || allocation
                        .participant_device_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                    || allocation
                        .participant_device_ids
                        .iter()
                        .any(|device_id| !admitted_logical_devices.contains(device_id.as_str()))
                    || (allocation.mode
                        == VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging
                        && allocation.participant_device_ids.len() != 2)
                {
                    return Err(VulkanRuntimeResidencyPlanError(format!(
                        "execution transient shared-host allocation {:?} is malformed",
                        allocation.concern,
                    )));
                }
                checked_residency_add(
                    total,
                    allocation.byte_capacity,
                    "execution transient shared-host allocation ledger",
                )
            },
        )?;
        next.execution_transient_shared_host_bytes_per_stream = checked_residency_add(
            next.execution_transient_shared_host_bytes_per_stream,
            shared_host_bytes,
            "execution transient shared-host residency",
        )?;
        next.total_stream_shared_host_bytes = checked_residency_add(
            next.total_stream_shared_host_bytes,
            shared_host_bytes,
            "execution transient total shared-host residency",
        )?;
        next.execution_transient_shared_host_allocations = shared_host_allocations.to_vec();
        *self = next;
        Ok(())
    }
}

fn validate_physical_execution_residency_inputs(
    parameter_allocations: &VulkanDistributedParameterAllocationPlan,
    parameter_exclusions: &VulkanDistributedParameterExclusionPlan,
    activations: &VulkanDistributedActivationBufferPlan,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    let parameter_bytes =
        parameter_allocations
            .allocations
            .iter()
            .try_fold(0usize, |total, allocation| {
                if allocation.byte_count == 0 {
                    return Err(VulkanRuntimeResidencyPlanError(
                        "distributed parameter residency contains an empty allocation".to_string(),
                    ));
                }
                checked_residency_add(
                    total,
                    allocation.byte_count,
                    "distributed parameter residency input",
                )
            })?;
    let parameter_tensor_count = parameter_allocations
        .allocations
        .iter()
        .map(|allocation| allocation.tensor.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if parameter_allocations.allocation_count != parameter_allocations.allocations.len()
        || parameter_allocations.tensor_count != parameter_tensor_count
        || parameter_allocations.total_byte_capacity != parameter_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed parameter residency summary disagrees with its allocations".to_string(),
        ));
    }

    let exclusion_bytes =
        parameter_exclusions
            .devices
            .iter()
            .try_fold(0usize, |total, device| {
                checked_residency_add(
                    total,
                    device.total_byte_capacity,
                    "distributed exclusion residency input",
                )
            })?;
    let exclusion_allocation_count =
        parameter_exclusions
            .devices
            .iter()
            .try_fold(0usize, |total, device| {
                total.checked_add(device.tensors.len()).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(
                        "distributed exclusion allocation count overflowed".to_string(),
                    )
                })
            })?;
    let exclusion_tensor_count = parameter_exclusions
        .devices
        .iter()
        .flat_map(|device| &device.tensors)
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .len();
    if parameter_exclusions.device_count != parameter_exclusions.devices.len()
        || parameter_exclusions.unique_tensor_count != exclusion_tensor_count
        || parameter_exclusions.excluded_full_allocation_count != exclusion_allocation_count
        || parameter_exclusions.excluded_full_byte_capacity != exclusion_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed parameter exclusion summary disagrees with its devices".to_string(),
        ));
    }

    let shared_activation_bytes = activations
        .allocations
        .iter()
        .map(|allocation| allocation.byte_capacity)
        .chain(
            activations
                .reduction_allocations
                .iter()
                .map(|allocation| allocation.byte_capacity),
        )
        .try_fold(0usize, |total, bytes| {
            checked_residency_add(total, bytes, "distributed shared activation input")
        })?;
    let private_activation_bytes = activations
        .private_intermediate_allocations
        .iter()
        .flat_map(|allocation| &allocation.devices)
        .try_fold(0usize, |total, device| {
            checked_residency_add(
                total,
                device.byte_capacity,
                "distributed private activation input",
            )
        })?;
    let private_allocation_count = activations
        .private_intermediate_allocations
        .iter()
        .try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.devices.len()).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "distributed private activation count overflowed".to_string(),
                )
            })
        })?;
    let activation_allocation_count = activations
        .allocations
        .len()
        .checked_add(activations.reduction_allocations.len())
        .and_then(|count| count.checked_add(private_allocation_count))
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "distributed activation allocation count overflowed".to_string(),
            )
        })?;
    if activations.allocation_count != activation_allocation_count
        || activations.total_shared_byte_capacity != shared_activation_bytes
        || activations.total_private_byte_capacity != private_activation_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed activation residency summary disagrees with its allocations".to_string(),
        ));
    }
    Ok(())
}

pub fn admit_vulkan_runtime_physical_execution_mount(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError> {
    admit_vulkan_runtime_physical_execution_device_bytes(
        plan,
        physical_device_by_logical_device,
        safe_capacity_by_physical_device,
        |device| device.mount_device_local_bytes,
        "mount",
    )
}

pub fn admit_vulkan_runtime_physical_execution_stream(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
    safe_host_bytes: usize,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError> {
    if plan.total_stream_shared_host_bytes > safe_host_bytes {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "physical execution stream needs {} shared-host bytes but only {safe_host_bytes} safe host bytes are available",
            plan.total_stream_shared_host_bytes
        )));
    }
    admit_vulkan_runtime_physical_execution_device_bytes(
        plan,
        physical_device_by_logical_device,
        safe_capacity_by_physical_device,
        |device| device.stream_device_local_bytes,
        "stream",
    )
}

fn admit_vulkan_runtime_physical_execution_device_bytes<F>(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
    bytes_for: F,
    phase: &str,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError>
where
    F: Fn(&VulkanRuntimePhysicalExecutionDeviceResidencyPlan) -> usize,
{
    let mut required = BTreeMap::<String, usize>::new();
    for device in &plan.device_plans {
        let physical_device_id = physical_device_by_logical_device
            .get(&device.device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "physical execution device {:?} has no physical-device binding",
                    device.device_id
                ))
            })?;
        let total = required.entry(physical_device_id.clone()).or_default();
        *total = checked_residency_add(*total, bytes_for(device), "physical execution admission")?;
    }
    for (physical_device_id, required_bytes) in &required {
        let safe_capacity = safe_capacity_by_physical_device
            .get(physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "physical device {physical_device_id:?} has no live capacity budget"
                ))
            })?;
        if *required_bytes > safe_capacity {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "physical device {physical_device_id:?} needs {required_bytes} {phase} device bytes but its live safe capacity is {safe_capacity}"
            )));
        }
    }
    Ok(required)
}

fn ensure_physical_residency_device(
    allowed: &BTreeSet<&str>,
    device_id: &str,
    label: &str,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if device_id.trim().is_empty() || !allowed.contains(device_id) {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "{label} references logical device {device_id:?} outside the physical execution plan"
        )));
    }
    Ok(())
}

fn ensure_physical_residency_activation_devices(
    allowed: &BTreeSet<&str>,
    owner_device_id: &str,
    device_ids: &[String],
    label: &str,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    ensure_physical_residency_device(allowed, owner_device_id, label)?;
    let unique = device_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if unique.len() != device_ids.len()
        || !unique.contains(owner_device_id)
        || unique
            .iter()
            .any(|device_id| ensure_physical_residency_device(allowed, device_id, label).is_err())
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "{label} has invalid physical execution participants"
        )));
    }
    Ok(())
}

fn add_physical_residency_shared_activation(
    breakdowns: &mut BTreeMap<String, VulkanRuntimePhysicalExecutionResidencyBreakdown>,
    owner_device_id: &str,
    byte_capacity: usize,
    route: VulkanSharedResidentBufferRoute,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if byte_capacity == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed shared activation has empty storage".to_string(),
        ));
    }
    let owner = breakdowns
        .get_mut(owner_device_id)
        .expect("shared activation owner was validated above");
    match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
            owner.distributed_shared_activation_device_bytes_per_stream = checked_residency_add(
                owner.distributed_shared_activation_device_bytes_per_stream,
                byte_capacity,
                "distributed shared device-local activation residency",
            )?;
        }
        VulkanSharedResidentBufferRoute::SharedHost => {
            owner.distributed_shared_host_bytes_per_stream = checked_residency_add(
                owner.distributed_shared_host_bytes_per_stream,
                byte_capacity,
                "distributed shared-host activation residency",
            )?;
        }
    }
    Ok(())
}
