fn runtime_compatibility_hardware_profile(
    stable_device_id: &str,
    supports_fixture_shaders: bool,
) -> crate::HardwareProcessProfile {
    let capability_extensions = supports_fixture_shaders.then(|| {
        BTreeMap::from([
            (
                "vulkan_compiler_capabilities".to_string(),
                serde_json::json!({
                    "shader_features": [
                        "buffer_device_address",
                        "cooperative_matrix",
                        "shader_bfloat16_cooperative_matrix",
                        "shader_bfloat16_dot_product",
                        "shader_bfloat16_type",
                        "shader_float16",
                        "shader_int8",
                        "shader_int16",
                        "shader_integer_dot_product",
                        "storage_buffer8_bit_access",
                        "storage_buffer16_bit_access",
                        "uniform_and_storage_buffer8_bit_access",
                        "uniform_and_storage_buffer16_bit_access",
                        "vulkan_memory_model",
                        "vulkan_memory_model_device_scope"
                    ],
                    "subgroup_operations": [
                        "arithmetic", "ballot", "basic", "shuffle", "shuffle_relative", "vote"
                    ],
                    "subgroup_compute_supported": true,
                    "subgroup_size": 64,
                    "max_compute_work_group_invocations": 1024,
                    "max_compute_work_group_size_x": 1024,
                    "cooperative_bfloat16_shapes": [[16, 16, 16]],
                    "cooperative_float8_e4m3_shapes": []
                }),
            ),
            (
                "vulkan_device".to_string(),
                serde_json::json!({
                    "extensions": [
                        "VK_KHR_cooperative_matrix",
                        "VK_KHR_shader_bfloat16"
                    ]
                }),
            ),
        ])
    });
    crate::HardwareProcessProfile::create(crate::HardwareProcessProfileDefinition {
        hardware_identity: crate::HardwareIdentity {
            device_kind: crate::HardwareDeviceKind::Gpu,
            stable_device_id: stable_device_id.to_string(),
            name: format!("{stable_device_id} fixture"),
            vendor_id: "fixture".to_string(),
            device_id: stable_device_id.to_string(),
            architecture: "fixture".to_string(),
            physical_location: stable_device_id.to_string(),
        },
        processes: vec![crate::HardwareProcessCapability::new(
            "compute",
            crate::HardwareProcessCategory::Arithmetic,
            crate::HardwareProcessAvailability::Available,
            crate::HardwareProcessProgrammability::Direct,
            "vulkan",
        )],
        memory_domains: vec![crate::HardwareMemoryDomain {
            name: "device_local".to_string(),
            kind: "device_local".to_string(),
            capacity_bytes: 64 * 1024 * 1024 * 1024,
            host_visible: false,
            device_local: true,
            coherent: false,
            cached: false,
            minimum_alignment_bytes: 4,
            properties: BTreeMap::new(),
        }],
        interconnects: Vec::new(),
        provenance: crate::HardwareProfileProvenance {
            api: "vulkan".to_string(),
            api_version: "1.4".to_string(),
            driver: "fixture".to_string(),
            driver_version: "1".to_string(),
            compiler: "fixture".to_string(),
            operating_system: "linux".to_string(),
            discovery_backend: "fixture".to_string(),
        },
        capability_extensions: capability_extensions.unwrap_or_default(),
        identity_extensions: BTreeMap::new(),
        runtime_bindings: BTreeMap::new(),
    })
    .unwrap()
}

#[test]
fn exact_baseline_compatibility_is_computed_per_runtime_instance() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu1");
    let profiles = BTreeMap::from([
        (
            "gpu0".to_string(),
            runtime_compatibility_hardware_profile("physical0", true),
        ),
        (
            "gpu1".to_string(),
            runtime_compatibility_hardware_profile("physical1", false),
        ),
    ]);

    let incompatible = vulkan_runtime_exact_baseline_incompatible_instance_ids(
        &model,
        tiny_model_dir(),
        &profiles,
    )
    .unwrap();

    assert_eq!(incompatible, BTreeSet::from(["layer_00_remote".to_string()]));
}

#[test]
fn exact_baseline_compatibility_requires_every_placed_device_profile() {
    let model = fixture_model_runtime_model_with_three_layer_series("gpu1");
    let profiles = BTreeMap::from([(
        "gpu0".to_string(),
        runtime_compatibility_hardware_profile("physical0", true),
    )]);

    let error = vulkan_runtime_exact_baseline_incompatible_instance_ids(
        &model,
        tiny_model_dir(),
        &profiles,
    )
    .unwrap_err();

    assert!(error.to_string().contains("without a hardware profile"));
    assert!(error.to_string().contains("layer_00_remote"));
}
