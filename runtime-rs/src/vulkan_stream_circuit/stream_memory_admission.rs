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
        // Graph-edge allocations are deferred and rebound through the exact
        // mounted-route ledger. Charging them here as generic distributed
        // buffers would reserve the same physical staging allocation twice.
        .filter(|allocation| {
            !matches!(allocation.storage, VulkanDistributedActivationStorage::Edge { .. })
        })
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

fn physical_execution_unclassified_shared_host_logical_bytes(
    plan: &VulkanRuntimePhysicalExecutionResidencyPlan,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    let distributed = plan.device_plans.iter().try_fold(0usize, |total, device| {
        total
            .checked_add(
                device
                    .breakdown
                    .distributed_shared_host_bytes_per_stream,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(
                        "physical distributed shared-host residency overflowed",
                    ),
                )
            })
    })?;
    let resident = plan
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
    let classified = [
        distributed,
        resident,
        plan.execution_transient_shared_host_bytes_per_stream,
        plan.execution_transient_host_visible_allocations
            .iter()
            .map(|allocation| allocation.byte_capacity)
            .sum::<usize>(),
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(
                "classified physical shared-host residency overflowed",
            ),
        )
    })?;
    plan.total_stream_shared_host_bytes
        .checked_sub(classified)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "physical shared-host residency declares {} total bytes but its bound ledgers require {classified}: distributed={distributed}, resident={resident}, execution_transient={}",
                    plan.total_stream_shared_host_bytes,
                    plan.execution_transient_shared_host_bytes_per_stream,
                )),
            )
        })
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

fn resident_stream_allocation_class(
    allocation: &VulkanRuntimeResidentStreamAllocation,
) -> VulkanMemoryAdmissionAllocationClass {
    if allocation.scope == VulkanRuntimeResidentStreamAllocationScope::Target
        && matches!(
            allocation.kind,
            VulkanRuntimeResidentStreamAllocationKind::StateTransaction { .. }
                | VulkanRuntimeResidentStreamAllocationKind::CausalVerificationSnapshot { .. }
        )
    {
        VulkanMemoryAdmissionAllocationClass::VerificationRunner
    } else {
        VulkanMemoryAdmissionAllocationClass::Permanent
    }
}

fn stream_transient_allocation_class(
    allocation_class: VulkanRuntimeStreamAllocationClass,
) -> VulkanMemoryAdmissionAllocationClass {
    match allocation_class {
        VulkanRuntimeStreamAllocationClass::Permanent => {
            VulkanMemoryAdmissionAllocationClass::Permanent
        }
        VulkanRuntimeStreamAllocationClass::PromptRunner => {
            VulkanMemoryAdmissionAllocationClass::PromptRunner
        }
        VulkanRuntimeStreamAllocationClass::VerificationRunner => {
            VulkanMemoryAdmissionAllocationClass::VerificationRunner
        }
        VulkanRuntimeStreamAllocationClass::CatchUpRunner => {
            VulkanMemoryAdmissionAllocationClass::CatchUpRunner
        }
    }
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

fn private_activation_resident_requirement_bytes_with<F>(
    allocations: &[VulkanRuntimePrivateActivationResidentAllocation],
    mut requirement_for: F,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>
where
    F: FnMut(
        &VulkanRuntimePrivateActivationResidentAllocation,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError>,
{
    allocations.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(requirement_for(allocation)?).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "private activation resident requirements overflowed",
                ),
            )
        })
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
    let mut stream_device_requirements = plan
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
            let private_activation_ledger_logical_bytes = device_plan
                .private_activation_resident_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} private activation allocation ledger overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })
                })?;
            if private_activation_ledger_logical_bytes
                != device_plan
                    .breakdown
                    .distributed_private_activation_device_bytes_per_stream
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "physical execution device {:?} private activation allocation ledger declares {private_activation_ledger_logical_bytes} bytes but its residency breakdown declares {}",
                        device_plan.device_id,
                        device_plan
                            .breakdown
                            .distributed_private_activation_device_bytes_per_stream,
                    )),
                ));
            }
            let residual_logical_bytes = logical_without_execution_transients
                .checked_sub(resident_ledger_logical_bytes)
                .and_then(|bytes| bytes.checked_sub(external_ledger_logical_bytes))
                .and_then(|bytes| bytes.checked_sub(private_activation_ledger_logical_bytes))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical execution device {:?} resident allocation ledger declares {resident_ledger_logical_bytes} bytes outside its {}-byte non-transient stream residency",
                            device_plan.device_id, logical_without_execution_transients,
                        )),
                    )
                })?;
            let exact_resident_bytes_for =
                |allocation_class: VulkanMemoryAdmissionAllocationClass| {
                    resident_stream_device_requirement_bytes_with(
                        &device_plan
                            .resident_stream_device_allocations
                            .iter()
                            .filter(|allocation| {
                                resident_stream_allocation_class(allocation) == allocation_class
                            })
                            .cloned()
                            .collect::<Vec<_>>(),
                        |allocation| {
                            device
                                .resident_buffer_memory_requirement_bytes(
                                    allocation.byte_capacity,
                                )
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                        },
                    )
                };
            let exact_external_bytes = external_device_local_resident_requirement_bytes(
                &device_plan.external_device_local_resident_allocations,
                device_for,
            )?;
            let exact_private_activation_bytes =
                private_activation_resident_requirement_bytes_with(
                    &device_plan.private_activation_resident_allocations,
                    |allocation| {
                        device
                            .resident_buffer_memory_requirement_bytes(allocation.byte_capacity)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                    },
                )?;
            let exact_transient_bytes_for =
                |allocation_class: VulkanMemoryAdmissionAllocationClass| {
                    execution_transient_device_requirement_bytes_with(
                        &device_plan
                            .execution_transient_device_allocations
                            .iter()
                            .filter(|allocation| {
                                stream_transient_allocation_class(allocation.allocation_class)
                                    == allocation_class
                            })
                            .cloned()
                            .collect::<Vec<_>>(),
                        |allocation| {
                            device
                                .resident_buffer_memory_requirement_bytes(
                                    allocation.byte_capacity,
                                )
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                        },
                    )
                };
            let permanent_base_bytes = residual_logical_bytes
                .checked_add(exact_external_bytes)
                .and_then(|bytes| bytes.checked_add(exact_private_activation_bytes))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "physical execution device {:?} permanent base stream requirement overflowed",
                            device_plan.device_id,
                        )),
                    )
                })?;
            let mut requirements_by_class = BTreeMap::from([(
                VulkanMemoryAdmissionAllocationClass::Permanent,
                permanent_base_bytes,
            )]);
            for allocation_class in VulkanMemoryAdmissionAllocationClass::ALL {
                let exact_class_bytes = exact_resident_bytes_for(allocation_class)?
                    .checked_add(exact_transient_bytes_for(allocation_class)?)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} {allocation_class:?} stream requirement overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })?;
                if exact_class_bytes > 0 {
                    let class_total = requirements_by_class.entry(allocation_class).or_default();
                    *class_total = class_total.checked_add(exact_class_bytes).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "physical execution device {:?} {allocation_class:?} stream requirement overflowed",
                                device_plan.device_id,
                            )),
                        )
                    })?;
                }
            }
            Ok((device, requirements_by_class))
        })
        .collect::<Result<Vec<_>, VulkanResidentInProcessPlacedRuntimeError>>()?;

    let mut host_visible_host_requirements_by_class = BTreeMap::<
        VulkanMemoryAdmissionAllocationClass,
        usize,
    >::new();
    for allocation in &plan.execution_transient_host_visible_allocations {
        let device = device_for(&allocation.logical_device_id)?;
        let requirement = device
            .host_visible_resident_buffer_memory_requirement(allocation.byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let allocation_class = stream_transient_allocation_class(allocation.allocation_class);
        match requirement.domain {
            VulkanResidentBufferMemoryDomain::DeviceLocal => {
                let (_, requirements_by_class) = stream_device_requirements
                    .iter_mut()
                    .find(|(candidate, _)| {
                        candidate.physical_device_id() == device.physical_device_id()
                    })
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "host-visible transient on {:?} has no physical device admission",
                                allocation.logical_device_id,
                            )),
                        )
                    })?;
                let class_total = requirements_by_class.entry(allocation_class).or_default();
                *class_total = class_total
                    .checked_add(requirement.byte_count)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "host-visible transient device requirement overflowed on {:?}",
                                allocation.logical_device_id,
                            )),
                        )
                    })?;
            }
            VulkanResidentBufferMemoryDomain::HostVisible => {
                let class_total = host_visible_host_requirements_by_class
                    .entry(allocation_class)
                    .or_default();
                *class_total = class_total
                    .checked_add(requirement.byte_count)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(
                                "host-visible transient host requirement overflowed",
                            ),
                        )
                    })?;
            }
        }
    }

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

    exact_stream_control_shared_host_allocation(plan, &physical_device_by_logical_device)
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "invalid physical stream-control allocation: {error}",
                )),
            )
        })?;

    let unclassified_shared_host_bytes =
        physical_execution_unclassified_shared_host_logical_bytes(plan)?;
    let execution_transient_shared_host_requirement_for =
        |allocation_class: VulkanMemoryAdmissionAllocationClass| {
            execution_transient_shared_host_requirement_bytes(
                &plan
                    .execution_transient_shared_host_allocations
                    .iter()
                    .filter(|allocation| {
                        stream_transient_allocation_class(allocation.allocation_class)
                            == allocation_class
                    })
                    .cloned()
                    .collect::<Vec<_>>(),
                device_for,
            )
        };
    let resident_shared_host_requirement = resident_shared_host_requirement_bytes(
        &plan.resident_shared_host_allocations,
        device_for,
    )?;
    let permanent_base_host_requirement_bytes =
        distributed_shared_host_requirement_bytes(&package.distributed_activation_plan, device_for)?
            .checked_add(unclassified_shared_host_bytes)
            .and_then(|bytes| bytes.checked_add(resident_shared_host_requirement))
            .ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(
                "physical execution shared-host reservation overflowed",
            ),
        )
    })?;
    let mut host_requirements_by_class = host_visible_host_requirements_by_class;
    if permanent_base_host_requirement_bytes > 0 {
        host_requirements_by_class.insert(
            VulkanMemoryAdmissionAllocationClass::Permanent,
            permanent_base_host_requirement_bytes,
        );
    }
    for allocation_class in VulkanMemoryAdmissionAllocationClass::ALL {
        let exact_class_bytes =
            execution_transient_shared_host_requirement_for(allocation_class)?;
        if exact_class_bytes > 0 {
            let class_total = host_requirements_by_class
                .entry(allocation_class)
                .or_default();
            *class_total = class_total.checked_add(exact_class_bytes).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "physical execution {allocation_class:?} shared-host reservation overflowed",
                    )),
                )
            })?;
        }
    }
    let representative = if host_requirements_by_class.is_empty() {
        None
    } else {
        let representative = physical_devices.values().next().copied().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(
                    "physical execution stream admission has no device",
                ),
            )
        })?;
        Some(representative)
    };
    let classified_host_requirements = host_requirements_by_class
        .iter()
        .map(|(allocation_class, byte_count)| {
            (
                *allocation_class,
                representative.expect("nonempty classified host requirements have a device"),
                safe_host_bytes,
                *byte_count,
            )
        })
        .collect::<Vec<_>>();
    let classified_device_requirements = stream_device_requirements
        .iter()
        .flat_map(|(device, requirements_by_class)| {
            requirements_by_class.iter().map(|(allocation_class, byte_count)| {
                (*allocation_class, *device, *byte_count)
            })
        })
        .collect::<Vec<_>>();
    VulkanMemoryAdmission::reserve_classified(
        &classified_device_requirements,
        &classified_host_requirements,
    )
        .map(Arc::new)
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed atomic physical execution stream reservation: {error}",
                )),
            )
        })
}
