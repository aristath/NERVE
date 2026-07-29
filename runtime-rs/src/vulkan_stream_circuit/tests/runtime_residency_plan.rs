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
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let long = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        64,
        false,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();

    assert_eq!(short.schema, VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA);
    assert_eq!(short.device_plans.len(), 1);
    assert_eq!(long.device_plans.len(), 1);
    let short_device = &short.device_plans[0];
    let long_device = &long.device_plans[0];
    assert!(
        short_device
            .parameter_residency
            .always_resident_bytes
            > 0
    );
    assert_eq!(
        short_device
            .parameter_residency
            .current_resident_bytes,
        short_device
            .parameter_residency
            .maximum_addressable_bytes,
    );
    assert!(
        long_device.breakdown.stream_state_bytes
            > short_device.breakdown.stream_state_bytes
    );
    assert!(
        long_device.initial_device_resident_bytes
            > short_device.initial_device_resident_bytes
    );
    assert_eq!(
        short.total_initial_device_resident_bytes,
        short_device.initial_device_resident_bytes
    );
    assert_eq!(
        short_device
            .parameter_residency
            .current_resident_bytes
            + short_device
                .parameter_residency
                .staging_headroom_bytes
            + short_device.working_set.transient_state_bytes
            + short_device.working_set.activation_headroom_bytes,
        short_device.initial_device_resident_bytes
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
        ResourceResidencyPolicy::Eager,
    )
    .unwrap_err();

    assert!(error.to_string().contains("refuses internal component sharding"));
}

#[test]
fn demand_plan_does_not_allocate_its_maximum_parameter_address_space() {
    let mut runtime_model = fixture_model_runtime_model();
    let contract = &mut runtime_model.package.resource_residency;
    let binding = contract
        .bindings
        .iter_mut()
        .find(|binding| binding.parameter_id == "ffn_down")
        .unwrap();
    let (
        CompiledResourceBindingMapping::AtomicGroup {
            atomic_group_id,
            resource_id,
        },
        template_id,
        member_seed,
    ) = (
        binding.mapping.clone(),
        format!("sha256:{}", "1".repeat(64)),
        format!("sha256:{}", "2".repeat(64)),
    )
    else {
        panic!("fixture ffn_down binding is not concrete");
    };
    contract
        .resources
        .retain(|resource| resource.id != resource_id);
    contract
        .atomic_groups
        .iter_mut()
        .find(|group| group.id == atomic_group_id)
        .unwrap()
        .resource_ids
        .retain(|id| id != &resource_id);
    binding.mapping =
        CompiledResourceBindingMapping::PartitionTemplateMember {
            partition_template_id: template_id.clone(),
            resource_identity_seed: member_seed.clone(),
        };
    contract.partition_templates.push(CompiledPartitionTemplate {
        id: template_id,
        partition_count: 1_000,
        lifetime: CompiledResourceLifetime::Dynamic,
        group_identity_seed: format!("sha256:{}", "3".repeat(64)),
        member_templates: vec![CompiledPartitionMemberTemplate {
            resource_identity_seed: member_seed,
            range_templates: vec![CompiledResourceRangeTemplate {
                artifact_path: "weights/parameter.safetensors".to_string(),
                base_byte_offset: 0,
                stride_bytes: 64,
                byte_count: 64,
                alignment_bytes: 64,
                integrity: CompiledResourceRangeIntegrityTemplate {
                    algorithm: "sha256_table".to_string(),
                    digest_table_path:
                        "integrity/partitions.sha256".to_string(),
                    digest_table_byte_offset: 0,
                    digest_stride_bytes: 32,
                    table_sha256: "0".repeat(64),
                },
            }],
            compatibility: CompiledResourceCompatibility {
                device_api: "vulkan".to_string(),
                storage_class: "storage_buffer".to_string(),
                read_only: true,
                required_features: Vec::new(),
            },
        }],
        dependencies: Vec::new(),
    });
    let package_root = tiny_model_dir();
    let tensor_index = runtime_model
        .load_runtime_tensor_index(&package_root)
        .unwrap();

    let demand = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        false,
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let eager = plan_vulkan_runtime_residency(
        &package_root,
        &runtime_model,
        &tensor_index,
        16,
        false,
        ResourceResidencyPolicy::Eager,
    )
    .unwrap();
    let demand_device = &demand.device_plans[0];
    let eager_device = &eager.device_plans[0];

    assert_eq!(
        demand_device
            .parameter_residency
            .initial_dynamic_bytes,
        0
    );
    assert_eq!(
        demand_device
            .parameter_residency
            .staging_headroom_bytes,
        64
    );
    assert_eq!(
        demand_device
            .parameter_residency
            .maximum_addressable_bytes
            - demand_device
                .parameter_residency
                .current_resident_bytes,
        64_000
    );
    assert_eq!(
        eager_device
            .parameter_residency
            .initial_dynamic_bytes,
        64_000
    );
    assert_eq!(
        eager_device.initial_device_resident_bytes
            - demand_device.initial_device_resident_bytes,
        64_000
    );

    let safe_capacity = demand_device.initial_device_resident_bytes + 64;
    let admission = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        64,
        safe_capacity,
    )
    .unwrap();
    assert_eq!(
        admission.projected_resident_parameter_bytes,
        demand_device.parameter_residency.current_resident_bytes + 64
    );
    assert_eq!(admission.projected_device_resident_bytes, safe_capacity);

    let capacity_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        64,
        safe_capacity - 1,
    )
    .unwrap_err();
    assert!(capacity_error.to_string().contains("safe capacity"));

    let maximum_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device
            .parameter_residency
            .maximum_addressable_bytes,
        64,
        usize::MAX,
    )
    .unwrap_err();
    assert!(maximum_error.to_string().contains("maximum addressable"));

    let staging_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes,
        65,
        usize::MAX,
    )
    .unwrap_err();
    assert!(staging_error.to_string().contains("staging headroom"));

    let current_error = admit_vulkan_runtime_residency_growth(
        demand_device,
        demand_device.parameter_residency.current_resident_bytes - 1,
        64,
        usize::MAX,
    )
    .unwrap_err();
    assert!(current_error.to_string().contains("outside the planned range"));
}
