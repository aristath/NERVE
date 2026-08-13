fn empty_selected_resource_mount_execution_plan() -> VulkanDistributedExecutionPlan {
    VulkanDistributedExecutionPlan {
        device_ids: vec!["gpu0".to_string()],
        storage_buffer_offset_alignment: 1,
        dispatches: Vec::new(),
        execution_islands: Vec::new(),
        shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
        shared_input_byte_capacity: 1,
        shared_output_byte_capacity: 1,
        distributed_parameter_byte_count: 0,
    }
}

fn empty_selected_resource_mount_residency_plan() -> VulkanRuntimeResidencyPlan {
    VulkanRuntimeResidencyPlan {
        schema: VULKAN_RUNTIME_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: "fixture".to_string(),
        residency_policy: ResourceResidencyPolicy::DemandRetained,
        context_capacity_activations: 1,
        speculative_draft_tokens: 0,
        device_plans: vec![VulkanRuntimeDeviceResidencyPlan {
            device_id: "gpu0".to_string(),
            parameter_residency: VulkanRuntimeParameterResidencyBytes {
                always_resident_bytes: 0,
                initial_dynamic_bytes: 0,
                current_resident_bytes: 0,
                maximum_addressable_bytes: 0,
                staging_headroom_bytes: 0,
            },
            resource_store: VulkanCompiledResourceStoreResidencyBytes::default(),
            working_set: VulkanRuntimeWorkingSetBytes::default(),
            breakdown: VulkanRuntimeDeviceResidencyBreakdown::default(),
            resident_stream_device_allocations: Vec::new(),
            initial_device_resident_bytes: 0,
        }],
        total_initial_device_resident_bytes: 0,
        total_current_resident_parameter_bytes: 0,
        total_maximum_addressable_parameter_bytes: 0,
    }
}

fn empty_selected_resource_mount_manifest() -> VulkanLoadedKernelArtifactCatalog {
    VulkanLoadedKernelArtifactCatalog {
        reusable_artifacts: Vec::new(),
        physical_artifacts: Vec::new(),
        reusable_word_count: 0,
        physical_word_count: 0,
    }
}

#[test]
fn selected_resource_mount_without_a_catalog_preserves_the_compiler_plan() {
    let runtime_model = fixture_model_runtime_model();
    let plan = empty_selected_resource_mount_execution_plan();
    let plans = VulkanDistributedExecutionPlanSet {
        decode: plan.clone(),
        decode_batch: plan.clone(),
        prefill: plan,
    };
    let baseline = plans.clone();
    let resolution = resolve_vulkan_runtime_selected_resource_mount(
        &runtime_model,
        &runtime_model.package.resource_residency,
        &empty_selected_resource_mount_manifest(),
        plans,
        &empty_selected_resource_mount_residency_plan(),
        &["gpu0".to_string()],
        &[],
        &TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::new(),
        },
        &[],
        "gpu0",
        "gpu0",
        false,
        ResourceResidencyPolicy::DemandRetained,
        None,
        None,
        &BTreeMap::new(),
    )
    .unwrap();

    assert_eq!(resolution.plans.execution_plans, baseline);
    assert!(resolution.placements.is_empty());
}

#[test]
fn selected_resource_mount_without_selected_work_does_not_require_capacity_evidence() {
    let runtime_model = fixture_model_runtime_model();
    let plan = empty_selected_resource_mount_execution_plan();
    let plans = VulkanDistributedExecutionPlanSet {
        decode: plan.clone(),
        decode_batch: plan.clone(),
        prefill: plan,
    };
    let resolution = resolve_vulkan_runtime_selected_resource_mount(
        &runtime_model,
        &runtime_model.package.resource_residency,
        &empty_selected_resource_mount_manifest(),
        plans,
        &empty_selected_resource_mount_residency_plan(),
        &["gpu0".to_string()],
        &[],
        &TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::new(),
        },
        &[],
        "gpu0",
        "gpu0",
        false,
        ResourceResidencyPolicy::DemandRetained,
        Some(&VulkanPlacementCalibrationCatalog::default()),
        None,
        &BTreeMap::new(),
    )
    .unwrap();

    assert!(resolution.placements.is_empty());
}

#[test]
fn exhausted_live_capacity_is_a_valid_but_infeasible_mount_state() {
    let runtime_model = fixture_model_runtime_model();
    let plan = empty_selected_resource_mount_execution_plan();
    let execution_plans = VulkanDistributedExecutionPlanSet {
        decode: plan.clone(),
        decode_batch: plan.clone(),
        prefill: plan,
    };
    let residency_plan = empty_selected_resource_mount_residency_plan();
    let plans = VulkanRuntimeDistributedMountPlans::derive(
        execution_plans,
        &residency_plan,
        &["gpu0".to_string()],
        &[],
        &TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::new(),
        },
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let capacities = selected_resource_mount_capacities(
        &runtime_model,
        &runtime_model.package.resource_residency,
        &plans,
        &[VulkanRuntimeSelectedResourceMountDevice {
            logical_device_id: "gpu0".to_string(),
            physical_device_id: "physical-gpu0".to_string(),
            execution_identity: VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: "physical-gpu0".to_string(),
                api_version: 1,
                driver_version: 1,
            },
            live_safe_capacity_bytes: 0,
            upload_alignment: 1,
        }],
        "gpu0",
        "gpu0",
        false,
        ResourceResidencyPolicy::DemandRetained,
        &BTreeMap::new(),
    )
    .unwrap();

    assert!(capacities.is_none());
}

#[test]
fn selected_resource_capacity_excludes_exact_execution_transients() {
    let runtime_model = fixture_model_runtime_model();
    let plan = empty_selected_resource_mount_execution_plan();
    let residency_plan = empty_selected_resource_mount_residency_plan();
    let plans = VulkanRuntimeDistributedMountPlans::derive(
        VulkanDistributedExecutionPlanSet {
            decode: plan.clone(),
            decode_batch: plan.clone(),
            prefill: plan,
        },
        &residency_plan,
        &["gpu0".to_string()],
        &[],
        &TensorIndex {
            schema: "nerve.tensor_index.v1".to_string(),
            tensors: BTreeMap::new(),
        },
        ResourceResidencyPolicy::DemandRetained,
    )
    .unwrap();
    let device = VulkanRuntimeSelectedResourceMountDevice {
        logical_device_id: "gpu0".to_string(),
        physical_device_id: "physical-gpu0".to_string(),
        execution_identity: VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: "physical-gpu0".to_string(),
            api_version: 1,
            driver_version: 1,
        },
        live_safe_capacity_bytes: 1_000,
        upload_alignment: 1,
    };
    let baseline = selected_resource_mount_capacities(
        &runtime_model,
        &runtime_model.package.resource_residency,
        &plans,
        std::slice::from_ref(&device),
        "gpu0",
        "gpu0",
        false,
        ResourceResidencyPolicy::DemandRetained,
        &BTreeMap::new(),
    )
    .unwrap()
    .unwrap();
    let exact = selected_resource_mount_capacities(
        &runtime_model,
        &runtime_model.package.resource_residency,
        &plans,
        &[device],
        "gpu0",
        "gpu0",
        false,
        ResourceResidencyPolicy::DemandRetained,
        &BTreeMap::from([("gpu0".to_string(), 128)]),
    )
    .unwrap()
    .unwrap();

    assert_eq!(
        exact[0].resident_payload_capacity_bytes,
        baseline[0].resident_payload_capacity_bytes - 128
    );
}
