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
fn physical_parameter_identity_distinguishes_compiled_layout_storage() {
    let model = fixture_model_runtime_model();
    let tensor_index = model.load_runtime_tensor_index(tiny_model_dir()).unwrap();
    let source = vulkan_hybrid_physical_tensor_resource_identity(
        &tensor_index,
        "model.layers.0.mlp.down_proj.weight",
    )
    .unwrap();
    let optimized = vulkan_hybrid_physical_tensor_resource_identity(
        &tensor_index,
        "model.layers.0.mlp.down_proj.weight.__nerve_input_block_major_b128",
    )
    .unwrap();

    assert_ne!(source, optimized);
    assert_eq!(
        optimized,
        vulkan_hybrid_physical_tensor_resource_identity(
            &tensor_index,
            "model.layers.0.mlp.down_proj.weight.__nerve_input_block_major_b128",
        )
        .unwrap()
    );
}

#[test]
fn graph_parameter_requirements_include_non_dispatch_transducer_parameters() {
    let model = fixture_model_runtime_model();
    let package_root = tiny_model_dir();
    let tensor_index = model.load_runtime_tensor_index(&package_root).unwrap();
    let resource_contract = instantiate_runtime_resource_contract(&model).unwrap();
    let logical_device_id = model
        .placement
        .device_for_component("input_transducer")
        .to_string();
    let device = physical_mount_test_device(&logical_device_id);
    let identities = BTreeMap::from([(logical_device_id, device.identity.clone())]);
    let mut requirements = BTreeMap::new();

    append_vulkan_hybrid_graph_parameter_requirements(
        &model,
        &tensor_index,
        &resource_contract,
        &identities,
        Some(&BTreeSet::from(["input_transducer".to_string()])),
        &BTreeSet::new(),
        &mut requirements,
    )
    .unwrap();

    assert_eq!(requirements.len(), 1);
    let transducer = &requirements["input_transducer"];
    assert!(!transducer.is_empty());
    assert!(transducer.iter().all(|requirement| {
        requirement.class == VulkanHybridResourceClass::Permanent
            && requirement.target == VulkanHybridResourceTarget::Device(device.identity.clone())
            && requirement.byte_count > 0
    }));
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
fn physical_mount_plan_admits_the_exact_selected_prefill_runner_geometry() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu0");
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_hybrid_phase_candidates(
        &model,
        &mut catalog,
        VulkanTargetedComponentExecutionPhase::Decode,
        10,
        20,
    );
    record_hybrid_phase_candidates(
        &model,
        &mut catalog,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 4,
        },
        10,
        20,
    );
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device: BTreeMap::from([
            (hybrid_test_device("gpu0"), usize::MAX),
            (hybrid_test_device("gpu1"), usize::MAX),
        ]),
        host_available_bytes: usize::MAX,
    };
    let phase_set =
        plan_vulkan_runtime_hybrid_phase_set(&model, &catalog, &capacity, Some(4)).unwrap();
    let bindings = BTreeMap::from([
        ("gpu0".to_string(), "logical0".to_string()),
        ("gpu1".to_string(), "logical1".to_string()),
    ]);
    let (runtime_model, physical) =
        lower_vulkan_runtime_hybrid_phase_set(&model, &phase_set, &bindings).unwrap();
    assert_eq!(physical.prefill_activation_batch_width, Some(4));
    let mut devices = physical
        .device_ids(&runtime_model)
        .into_iter()
        .map(|logical_device_id| VulkanRuntimePhysicalPlanningDevice {
            identity: hybrid_test_device(if logical_device_id == "logical0" {
                "gpu0"
            } else {
                "gpu1"
            }),
            logical_device_id,
            safe_capacity_bytes: usize::MAX,
            storage_buffer_offset_alignment: 8,
        })
        .collect::<Vec<_>>();
    let exact = plan_vulkan_runtime_physical_mount(
        tiny_model_dir(),
        &runtime_model,
        &physical,
        Some(&catalog),
        64,
        0,
        ResourceResidencyPolicy::Eager,
        &devices,
        usize::MAX,
    )
    .unwrap()
    .unwrap();
    assert!(
        exact
            .physical_execution_residency_plan
            .device_plans
            .iter()
            .map(|device| {
                device
                    .breakdown
                    .execution_transient_device_bytes_per_stream
            })
            .sum::<usize>()
            > 0
    );
    for device in &mut devices {
        let required = exact
            .physical_execution_residency_plan
            .device_plans
            .iter()
            .find(|plan| plan.device_id == device.logical_device_id)
            .map(|plan| plan.mount_device_local_bytes + plan.stream_device_local_bytes)
            .unwrap();
        device.safe_capacity_bytes = required;
    }
    devices[0].safe_capacity_bytes -= 1;
    assert!(
        plan_vulkan_runtime_physical_mount(
            tiny_model_dir(),
            &runtime_model,
            &physical,
            Some(&catalog),
            64,
            0,
            ResourceResidencyPolicy::Eager,
            &devices,
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

#[test]
fn physical_mount_host_cache_capacity_preserves_the_stream_reservation() {
    assert_eq!(
        remaining_vulkan_runtime_host_cache_bytes(1_024, 256).unwrap(),
        768,
    );
    assert_eq!(
        remaining_vulkan_runtime_host_cache_bytes(1_024, 1_024).unwrap(),
        0,
    );
    let error = remaining_vulkan_runtime_host_cache_bytes(1_023, 1_024).unwrap_err();
    assert!(error.to_string().contains("stream needs 1024"));
}
