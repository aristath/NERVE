fn shared_host_allocation_requirement_for_logical_devices<'a, F>(
    owner_device_id: &str,
    participant_device_ids: &[String],
    byte_capacity: usize,
    device_for: &F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    let owner = device_for(owner_device_id)?;
    let mut unique_peers = BTreeMap::<String, &VulkanComputeDevice>::new();
    for device_id in participant_device_ids {
        let participant = device_for(device_id)?;
        if !participant.shares_logical_device_with(owner) {
            unique_peers
                .entry(participant.physical_device_id().to_string())
                .or_insert(participant);
        }
    }
    owner
        .shared_host_allocation_requirement_bytes(
            &unique_peers.values().copied().collect::<Vec<_>>(),
            byte_capacity,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
}

fn distributed_shared_host_requirement_bytes<'a, F>(
    plan: &VulkanDistributedActivationBufferPlan,
    device_for: &F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    if plan.route != VulkanSharedResidentBufferRoute::SharedHost {
        return Ok(0);
    }
    plan.allocations
        .iter()
        .map(|allocation| {
            shared_host_allocation_requirement_for_logical_devices(
                &allocation.owner_device_id,
                &allocation.device_ids,
                allocation.byte_capacity,
                device_for,
            )
        })
        .chain(plan.reduction_allocations.iter().map(|allocation| {
            shared_host_allocation_requirement_for_logical_devices(
                &allocation.owner_device_id,
                &allocation.device_ids,
                allocation.byte_capacity,
                device_for,
            )
        }))
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes?).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(
                        "distributed shared-host allocation requirements overflowed",
                    ),
                )
            })
        })
}

fn shared_stream_control_requirement_bytes(
    physical_devices: &BTreeMap<String, &VulkanComputeDevice>,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    if physical_devices.len() <= 1 {
        return Ok(0);
    }
    let mut participants = physical_devices.values().copied();
    let owner = participants
        .next()
        .expect("multiple physical stream devices exist");
    owner
        .shared_host_allocation_requirement_bytes(
            &participants.collect::<Vec<_>>(),
            VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
}

fn execution_transient_shared_host_requirement_bytes<'a, F>(
    allocations: &[VulkanRuntimeSharedHostTransientAllocation],
    device_for: &F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    execution_transient_shared_host_requirement_bytes_with(allocations, |allocation| {
        if allocation.mode
            == VulkanRuntimeSharedHostTransientAllocationMode::CrossDeviceTimelineStaging
        {
            let owner = device_for(&allocation.owner_device_id)?;
            let participants = allocation
                .participant_device_ids
                .iter()
                .map(|device_id| device_for(device_id))
                .collect::<Result<Vec<_>, _>>()?;
            let distinct_peers = participants
                .iter()
                .copied()
                .filter(|device| !device.shares_logical_device_with(owner))
                .collect::<Vec<_>>();
            if distinct_peers.is_empty()
                || !owner.supports_shared_host_memory()
                || !owner.supports_opaque_fd_timeline_semaphores()
                || distinct_peers.iter().any(|device| {
                    !device.supports_shared_host_memory()
                        || !device.supports_opaque_fd_timeline_semaphores()
                })
            {
                return Ok(0);
            }
        }
        shared_host_allocation_requirement_for_logical_devices(
            &allocation.owner_device_id,
            &allocation.participant_device_ids,
            allocation.byte_capacity,
            device_for,
        )
    })
}

fn execution_transient_shared_host_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimeSharedHostTransientAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimeSharedHostTransientAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(requirement_for(allocation)?).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "execution transient shared-host allocation requirements overflowed",
                ),
            )
        })
    })
}

fn execution_transient_device_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimeDeviceLocalTransientAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimeDeviceLocalTransientAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(requirement_for(allocation)?).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "execution transient device allocation requirements overflowed",
                ),
            )
        })
    })
}

fn resident_stream_device_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimeResidentStreamAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimeResidentStreamAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(requirement_for(allocation)?).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident stream device allocation requirements overflowed",
                ),
            )
        })
    })
}

fn external_device_local_resident_requirement_bytes<'a, F>(
    allocations: &[VulkanRuntimeExternalDeviceLocalResidentAllocation],
    device_for: &F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    external_device_local_resident_requirement_bytes_with(allocations, |allocation| {
        let owner = device_for(&allocation.owner_device_id)?;
        let peers = allocation
            .participant_device_ids
            .iter()
            .map(|device_id| device_for(device_id))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|device| !device.shares_logical_device_with(owner))
            .collect::<Vec<_>>();
        owner
            .shared_device_resident_buffer_memory_requirement_bytes(
                &peers,
                allocation.byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    })
}

fn external_device_local_resident_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimeExternalDeviceLocalResidentAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimeExternalDeviceLocalResidentAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total
            .checked_add(requirement_for(allocation)?)
            .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "external device-local resident requirements overflowed",
                ),
            )
        })
    })
}

fn resident_shared_host_requirement_bytes<'a, F>(
    allocations: &[VulkanRuntimeSharedHostResidentAllocation],
    device_for: &F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    resident_shared_host_requirement_bytes_with(allocations, |allocation| {
        shared_host_allocation_requirement_for_logical_devices(
            &allocation.owner_device_id,
            &allocation.participant_device_ids,
            allocation.byte_capacity,
            device_for,
        )
    })
}

fn resident_shared_host_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimeSharedHostResidentAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimeSharedHostResidentAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total
            .checked_add(requirement_for(allocation)?)
            .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "resident shared-host allocation requirements overflowed",
                ),
            )
        })
    })
}

fn reserve_vulkan_runtime_physical_execution_stream_memory<'a, F>(
    package: &VulkanResidentInProcessPlacedModelPackage,
    device_for: &F,
) -> Result<Arc<VulkanMemoryAdmission>, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    let plan = &package.physical_execution_residency_plan;
    let mut physical_device_by_logical_device = BTreeMap::new();
    let mut safe_capacity_by_physical_device = BTreeMap::new();
    let mut physical_devices = BTreeMap::<String, &VulkanComputeDevice>::new();
    let stream_device_requirements = plan
        .device_plans
        .iter()
        .map(|device_plan| {
            let device = device_for(&device_plan.device_id)?;
            let physical_device_id = device.physical_device_id().to_string();
            physical_device_by_logical_device
                .insert(device_plan.device_id.clone(), physical_device_id.clone());
            let safe_capacity =
                usize::try_from(device.device_local_memory_budget().reservable_bytes)
                    .unwrap_or(usize::MAX);
            safe_capacity_by_physical_device
                .entry(physical_device_id.clone())
                .and_modify(|capacity: &mut usize| *capacity = (*capacity).min(safe_capacity))
                .or_insert(safe_capacity);
            physical_devices.entry(physical_device_id).or_insert(device);
            let logical_without_execution_transients = device_plan
                .stream_device_local_bytes
                .checked_sub(
                    device_plan
                        .breakdown
                        .execution_transient_device_bytes_per_stream,
                )
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical execution device {:?} stream residency omits its execution transients",
                            device_plan.device_id,
                        )),
                    )
                })?;
            let transient_ledger_logical_bytes = device_plan
                .execution_transient_device_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} transient allocation ledger overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })
                })?;
            if transient_ledger_logical_bytes
                != device_plan
                    .breakdown
                    .execution_transient_device_bytes_per_stream
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "physical execution device {:?} transient allocation ledger declares {transient_ledger_logical_bytes} bytes but its residency breakdown declares {}",
                        device_plan.device_id,
                        device_plan
                            .breakdown
                            .execution_transient_device_bytes_per_stream,
                    )),
                ));
            }
            let resident_ledger_logical_bytes = device_plan
                .resident_stream_device_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} resident allocation ledger overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })
                })?;
            let external_ledger_logical_bytes = device_plan
                .external_device_local_resident_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} external allocation ledger overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })
                })?;
            let residual_logical_bytes = logical_without_execution_transients
                .checked_sub(resident_ledger_logical_bytes)
                .and_then(|bytes| bytes.checked_sub(external_ledger_logical_bytes))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical execution device {:?} resident allocation ledger declares {resident_ledger_logical_bytes} bytes outside its {}-byte non-transient stream residency",
                            device_plan.device_id, logical_without_execution_transients,
                        )),
                    )
                })?;
            let exact_resident_bytes = resident_stream_device_requirement_bytes_with(
                &device_plan.resident_stream_device_allocations,
                |allocation| {
                    device
                        .resident_buffer_memory_requirement_bytes(allocation.byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                },
            )?;
            let exact_external_bytes = external_device_local_resident_requirement_bytes(
                &device_plan.external_device_local_resident_allocations,
                device_for,
            )?;
            let exact_transient_bytes = execution_transient_device_requirement_bytes_with(
                &device_plan.execution_transient_device_allocations,
                |allocation| {
                    device
                        .resident_buffer_memory_requirement_bytes(allocation.byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                },
            )?;
            let exact_stream_bytes = residual_logical_bytes
                .checked_add(exact_resident_bytes)
                .and_then(|bytes| bytes.checked_add(exact_external_bytes))
                .and_then(|bytes| bytes.checked_add(exact_transient_bytes))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical execution device {:?} exact stream requirement overflowed",
                            device_plan.device_id,
                        )),
                    )
                })?;
            Ok((device, exact_stream_bytes))
        })
        .collect::<Result<Vec<_>, VulkanResidentInProcessPlacedRuntimeError>>()?;

    let safe_host_bytes = if plan.total_stream_shared_host_bytes == 0
        && physical_devices.len() <= 1
    {
        usize::MAX
    } else {
        vulkan_safe_host_available_bytes()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?
    };
    admit_vulkan_runtime_physical_execution_stream(
        plan,
        &physical_device_by_logical_device,
        &safe_capacity_by_physical_device,
        safe_host_bytes,
    )
    .map_err(|error| {
        VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(format!(
                "failed exact physical execution stream admission: {error}"
            )),
        )
    })?;

    let expected_shared_stream_control_bytes = if physical_devices.len() > 1 {
        VULKAN_STREAM_CONTROL_BYTE_CAPACITY
    } else {
        0
    };
    if plan.shared_stream_control_host_bytes_per_stream != expected_shared_stream_control_bytes {
        return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(format!(
                "physical execution stream-control memory domain declares {} shared-host bytes for {} physical device(s), expected {expected_shared_stream_control_bytes}",
                plan.shared_stream_control_host_bytes_per_stream,
                physical_devices.len(),
            )),
        ));
    }

    let distributed_shared_host_logical_bytes = if package.distributed_activation_plan.route
        == VulkanSharedResidentBufferRoute::SharedHost
    {
        package
            .distributed_activation_plan
            .allocations
            .iter()
            .filter(|allocation| {
                !matches!(allocation.storage, VulkanDistributedActivationStorage::Edge { .. })
            })
            .map(|allocation| allocation.byte_capacity)
            .chain(
                package
                    .distributed_activation_plan
                    .reduction_allocations
                    .iter()
                    .map(|allocation| allocation.byte_capacity),
            )
            .try_fold(0usize, |total, bytes| {
                total.checked_add(bytes).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(
                            "distributed shared-host logical residency overflowed",
                        ),
                    )
                })
            })?
        } else {
            0
        };
    let resident_shared_host_logical_bytes = plan
        .resident_shared_host_allocations
        .iter()
        .try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(
                        "resident shared-host logical residency overflowed",
                    ),
                )
            })
        })?;
    let non_distributed_shared_host_bytes = plan
        .total_stream_shared_host_bytes
        .checked_sub(distributed_shared_host_logical_bytes)
        .and_then(|bytes| bytes.checked_sub(resident_shared_host_logical_bytes))
        .and_then(|bytes| {
            bytes.checked_sub(plan.execution_transient_shared_host_bytes_per_stream)
        })
        .and_then(|bytes| {
            bytes.checked_sub(plan.shared_stream_control_host_bytes_per_stream)
        })
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "physical execution shared-host residency omits distributed allocations",
                ),
            )
        })?;
    let shared_stream_control_requirement =
        shared_stream_control_requirement_bytes(&physical_devices)?;
    let execution_transient_shared_host_requirement =
        execution_transient_shared_host_requirement_bytes(
            &plan.execution_transient_shared_host_allocations,
            device_for,
        )?;
    let resident_shared_host_requirement = resident_shared_host_requirement_bytes(
        &plan.resident_shared_host_allocations,
        device_for,
    )?;
    let stream_host_requirement_bytes =
        distributed_shared_host_requirement_bytes(&package.distributed_activation_plan, device_for)?
            .checked_add(non_distributed_shared_host_bytes)
            .and_then(|bytes| bytes.checked_add(resident_shared_host_requirement))
            .and_then(|bytes| bytes.checked_add(execution_transient_shared_host_requirement))
            .and_then(|bytes| bytes.checked_add(shared_stream_control_requirement))
            .ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(
                "physical execution shared-host reservation overflowed",
            ),
        )
    })?;
    let stream_host_requirement = if stream_host_requirement_bytes == 0 {
        None
    } else {
        let representative = physical_devices.values().next().copied().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "physical execution stream admission has no device",
                ),
            )
        })?;
        Some((
            representative,
            safe_host_bytes,
            stream_host_requirement_bytes,
        ))
    };
    VulkanMemoryAdmission::reserve(&stream_device_requirements, stream_host_requirement)
        .map(Arc::new)
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed atomic physical execution stream reservation: {error}",
                )),
            )
        })
}
