fn physical_mount_test_device(logical_device_id: &str) -> VulkanRuntimePhysicalPlanningDevice {
    VulkanRuntimePhysicalPlanningDevice {
        logical_device_id: logical_device_id.to_string(),
        identity: VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: "gpu0".to_string(),
            api_version: 1,
            driver_version: 2,
        },
        safe_capacity_bytes: usize::MAX,
        storage_buffer_offset_alignment: 8,
    }
}

#[test]
fn physical_mount_plan_uses_requested_context_without_opening_vulkan() {
    let model = fixture_model_runtime_model();
    let physical = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
    let [logical_device_id] = physical.device_ids(&model).try_into().unwrap();
    let device = physical_mount_test_device(&logical_device_id);

    let short = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        16,
        0,
        ResourceResidencyPolicy::Eager,
        std::slice::from_ref(&device),
        usize::MAX,
    )
    .unwrap();
    let short = short.unwrap();
    let long = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        64,
        0,
        ResourceResidencyPolicy::Eager,
        &[device.clone()],
        usize::MAX,
    )
    .unwrap();
    let long = long.unwrap();

    assert!(
        long.physical_execution_residency_plan
            .total_stream_device_local_bytes
            > short
                .physical_execution_residency_plan
                .total_stream_device_local_bytes
    );
    assert!(short.selected_resource_placements.is_empty());
    assert!(long.selected_resource_placements.is_empty());

    let parameter_capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([(device.identity.clone(), usize::MAX)]),
        host_available_bytes: usize::MAX,
    };
    let parameter_reservations = short
        .exact_parameter_resources_by_component
        .values()
        .try_fold(
            VulkanHybridResourceReservations::default(),
            |reservations, resources| {
                reservations
                    .reserve(resources, &parameter_capacity)
                    .map(|reservation| {
                        reservation.expect("unbounded exact parameter capacity must admit")
                    })
            },
        )
        .unwrap();
    let physical = &short.physical_execution_residency_plan.device_plans[0].breakdown;
    let exact_parameter_bytes = physical
        .owner_parameter_bytes_before_distributed_replacement
        .checked_sub(physical.excluded_owner_parameter_bytes)
        .unwrap()
        .checked_add(physical.distributed_parameter_bytes)
        .unwrap();
    assert_eq!(
        parameter_reservations.device_bytes[&device.identity].permanent_bytes,
        exact_parameter_bytes
    );
}

#[test]
fn physical_mount_plan_rejects_missing_and_duplicate_device_records() {
    let model = fixture_model_runtime_model();
    let physical = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
    let error = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        16,
        0,
        ResourceResidencyPolicy::Eager,
        &[],
        usize::MAX,
    )
    .unwrap_err();

    assert!(error.to_string().contains("one exact positive-capacity"));

    let [logical_device_id] = physical.device_ids(&model).try_into().unwrap();
    let device = physical_mount_test_device(&logical_device_id);
    let error = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        16,
        0,
        ResourceResidencyPolicy::Eager,
        &[device.clone(), device],
        usize::MAX,
    )
    .unwrap_err();

    assert!(error.to_string().contains("one exact positive-capacity"));
}

#[test]
fn physical_mount_plan_reports_full_context_capacity_infeasibility_without_allocating() {
    let model = fixture_model_runtime_model();
    let physical = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
    let [logical_device_id] = physical.device_ids(&model).try_into().unwrap();
    let mut device = physical_mount_test_device(&logical_device_id);
    let exact = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &model,
        &physical,
        None,
        64,
        0,
        ResourceResidencyPolicy::Eager,
        std::slice::from_ref(&device),
        usize::MAX,
    )
    .unwrap()
    .unwrap();
    let required = exact.physical_execution_residency_plan.device_plans[0].mount_device_local_bytes
        + exact.physical_execution_residency_plan.device_plans[0].stream_device_local_bytes;
    device.safe_capacity_bytes = required - 1;

    assert!(
        plan_vulkan_runtime_physical_mount(
            tiny_model_dir(),
            &model,
            &physical,
            None,
            64,
            0,
            ResourceResidencyPolicy::Eager,
            &[device],
            usize::MAX,
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn physical_mount_resource_summary_uses_the_exact_store_plan() {
    let store_plan = VulkanDistributedSelectedResourceStorePlan {
        devices: vec![crate::VulkanDistributedSelectedResourceDevicePlan {
            device_id: "gpu0".to_string(),
            selectors: Vec::new(),
            unique_atomic_group_count: 1,
            maximum_atomic_group_bytes: 64,
            maximum_load_wave_bytes: 128,
            total_addressable_bytes: 1_024,
        }],
        tensor_sharded_residency_cohorts: Vec::new(),
        device_count: 1,
        selector_count: 1,
        selector_placement_count: 1,
        unique_atomic_group_count: 1,
        total_addressable_bytes: 1_024,
    };
    let quotas = BTreeMap::from([("gpu0".to_string(), 256)]);

    let paged = summarize_vulkan_runtime_physical_selected_resources(
        &store_plan,
        &quotas,
        ResourceResidencyPolicy::DemandPaged,
    );
    assert_eq!(
        paged.maximum_load_wave_bytes_by_logical_device,
        BTreeMap::from([("gpu0".to_string(), 128)])
    );
    assert!(paged.uses_shared_host_cache);

    let retained = summarize_vulkan_runtime_physical_selected_resources(
        &store_plan,
        &quotas,
        ResourceResidencyPolicy::DemandRetained,
    );
    assert!(!retained.uses_shared_host_cache);
}
