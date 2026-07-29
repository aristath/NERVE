#[test]
fn runtime_residency_plan_uses_physical_transient_layout_without_opening_vulkan() {
    let runtime_model = fixture_model_runtime_model();
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let short = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        false,
    )
    .unwrap();
    let long = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        64,
        false,
    )
    .unwrap();

    assert_eq!(short.schema, VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA);
    assert_eq!(short.device_plans.len(), 1);
    assert_eq!(long.device_plans.len(), 1);
    let short_device = &short.device_plans[0];
    let long_device = &long.device_plans[0];
    assert!(short_device.breakdown.component_parameter_bytes > 0);
    assert!(short_device.breakdown.transducer_parameter_bytes > 0);
    assert!(
        long_device.breakdown.stream_state_bytes
            > short_device.breakdown.stream_state_bytes
    );
    assert!(
        long_device.total_device_resident_bytes
            > short_device.total_device_resident_bytes
    );
    assert_eq!(
        short.total_device_resident_bytes,
        short_device.total_device_resident_bytes
    );
    assert_eq!(
        sum_residency_breakdown(&short_device.breakdown).unwrap(),
        short_device.total_device_resident_bytes
    );
}

#[test]
fn runtime_residency_plan_fails_closed_for_unplanned_internal_sharding() {
    let runtime_model = fixture_model_runtime_model()
        .with_component_shard_devices(
            "layer_00",
            vec![
                RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
                "gpu1".to_string(),
            ],
        )
        .unwrap();
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let error = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        false,
    )
    .unwrap_err();

    assert!(error.to_string().contains("refuses internal component sharding"));
}
