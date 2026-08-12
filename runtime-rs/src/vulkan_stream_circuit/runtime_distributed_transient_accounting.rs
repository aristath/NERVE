fn distributed_calibration_transient_backing_bytes(
    device_ids: &[String],
    plan: &VulkanDistributedActivationBufferPlan,
) -> Result<(BTreeMap<String, usize>, usize), VulkanResidentTokenModelPackageError> {
    let shared_allocations = plan
        .allocations
        .iter()
        .map(|allocation| {
            (
                allocation.owner_device_id.as_str(),
                allocation.byte_capacity,
            )
        })
        .chain(plan.reduction_allocations.iter().map(|allocation| {
            (
                allocation.owner_device_id.as_str(),
                allocation.byte_capacity,
            )
        }))
        .collect::<Vec<_>>();
    let shared_byte_capacity = shared_allocations
        .iter()
        .try_fold(0usize, |total, (_, bytes)| total.checked_add(*bytes))
        .ok_or_else(|| {
            distributed_calibration_error_value(
                "distributed shared transient byte accounting overflowed",
            )
        })?;
    if shared_byte_capacity != plan.total_shared_byte_capacity {
        return distributed_calibration_error(format!(
            "distributed shared transient plan declares {} bytes but contains {shared_byte_capacity}",
            plan.total_shared_byte_capacity,
        ));
    }
    let (mut device_bytes, host_bytes) = distributed_calibration_activation_backing_bytes(
        device_ids,
        plan.route,
        shared_allocations,
    )?;

    let mut private_byte_capacity = 0usize;
    for allocation in &plan.private_intermediate_allocations {
        for device_allocation in &allocation.devices {
            if device_allocation.byte_capacity == 0 {
                return distributed_calibration_error(
                    "distributed private transient allocation has zero bytes",
                );
            }
            let device_total = device_bytes
                .get_mut(&device_allocation.device_id)
                .ok_or_else(|| {
                    distributed_calibration_error_value(format!(
                        "distributed private transient allocation has unknown device {:?}",
                        device_allocation.device_id,
                    ))
                })?;
            *device_total = device_total
                .checked_add(device_allocation.byte_capacity)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed private transient device-byte accounting overflowed",
                    )
                })?;
            private_byte_capacity = private_byte_capacity
                .checked_add(device_allocation.byte_capacity)
                .ok_or_else(|| {
                    distributed_calibration_error_value(
                        "distributed private transient byte accounting overflowed",
                    )
                })?;
        }
    }
    if private_byte_capacity != plan.total_private_byte_capacity {
        return distributed_calibration_error(format!(
            "distributed private transient plan declares {} bytes but contains {private_byte_capacity}",
            plan.total_private_byte_capacity,
        ));
    }
    Ok((device_bytes, host_bytes))
}

#[cfg(test)]
mod runtime_distributed_transient_accounting_tests {
    use super::*;

    fn plan(
        devices: &[String],
        route: VulkanSharedResidentBufferRoute,
    ) -> VulkanDistributedActivationBufferPlan {
        VulkanDistributedActivationBufferPlan {
            allocations: vec![VulkanDistributedActivationBufferAllocation {
                storage: VulkanDistributedActivationStorage::ActivationSlot,
                owner_device_id: "gpu-a".to_string(),
                component_id: "component".to_string(),
                slot: 0,
                byte_capacity: 64,
                signal_ids: vec!["input".to_string()],
                device_ids: devices.to_vec(),
                input_use_count: 1,
                output_use_count: 0,
            }],
            reduction_allocations: vec![VulkanDistributedReductionBufferAllocation {
                owner_device_id: "gpu-a".to_string(),
                dispatch_index: 1,
                component_id: "component".to_string(),
                node_id: "down".to_string(),
                plane_byte_capacity: 48,
                byte_capacity: 96,
                device_ids: devices.to_vec(),
            }],
            private_intermediate_allocations: vec![
                VulkanDistributedPrivateIntermediateBufferAllocation {
                    producer_dispatch_index: 0,
                    consumer_dispatch_index: 1,
                    component_id: "component".to_string(),
                    signal_id: "activated".to_string(),
                    devices: vec![
                        VulkanDistributedPrivateIntermediateDeviceAllocation {
                            device_id: "gpu-a".to_string(),
                            byte_capacity: 16,
                        },
                        VulkanDistributedPrivateIntermediateDeviceAllocation {
                            device_id: "gpu-b".to_string(),
                            byte_capacity: 24,
                        },
                    ],
                },
            ],
            allocation_count: 4,
            import_count: 6,
            reference_count: 6,
            total_shared_byte_capacity: 160,
            total_private_byte_capacity: 40,
            route,
        }
    }

    #[test]
    fn accounts_reduction_and_private_transient_backing() {
        let devices = vec!["gpu-a".to_string(), "gpu-b".to_string()];
        let (shared_host_devices, shared_host_bytes) =
            distributed_calibration_transient_backing_bytes(
                &devices,
                &plan(&devices, VulkanSharedResidentBufferRoute::SharedHost),
            )
            .unwrap();
        assert_eq!(shared_host_devices["gpu-a"], 16);
        assert_eq!(shared_host_devices["gpu-b"], 24);
        assert_eq!(shared_host_bytes, 160);

        let (device_local, host_bytes) = distributed_calibration_transient_backing_bytes(
            &devices,
            &plan(
                &devices,
                VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
            ),
        )
        .unwrap();
        assert_eq!(device_local["gpu-a"], 176);
        assert_eq!(device_local["gpu-b"], 24);
        assert_eq!(host_bytes, 0);
    }

    #[test]
    fn rejects_incomplete_transient_backing_totals() {
        let devices = vec!["gpu-a".to_string(), "gpu-b".to_string()];
        let mut invalid = plan(&devices, VulkanSharedResidentBufferRoute::SharedHost);
        invalid.total_shared_byte_capacity -= 1;
        assert!(
            distributed_calibration_transient_backing_bytes(&devices, &invalid)
                .unwrap_err()
                .to_string()
                .contains("shared transient plan declares 159 bytes")
        );

        invalid.total_shared_byte_capacity = 160;
        invalid.total_private_byte_capacity -= 1;
        assert!(
            distributed_calibration_transient_backing_bytes(&devices, &invalid)
                .unwrap_err()
                .to_string()
                .contains("private transient plan declares 39 bytes")
        );
    }
}
