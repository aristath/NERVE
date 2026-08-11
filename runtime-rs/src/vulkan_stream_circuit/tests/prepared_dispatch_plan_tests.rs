#[test]
fn prepared_dispatch_plan_links_artifacts_to_descriptor_resources() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );

    let prepared = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &manifest,
    )
    .unwrap();

    assert_eq!(prepared.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(
        prepared.reusable_family_count,
        reusable_plan.total_family_count()
    );
    assert_eq!(prepared.dispatches.len(), 10);
    assert_eq!(
        prepared.total_descriptor_count,
        prepared
            .dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum::<usize>()
    );

    let first = prepared.dispatch("layer_00", "operator_norm").unwrap();
    let first_family = reusable_family_with_kernel(&reusable_plan, "layer_00.operator_norm");
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.reusable_family_id, first_family.family_id);
    assert_eq!(first.artifact_path, artifact_path_for_family(first_family));
    assert_eq!(first.entry_point, DEFAULT_SPIRV_ENTRY_POINT);
    assert_eq!(first.local_size_x, DEFAULT_COMPUTE_LOCAL_SIZE_X);
    assert_eq!(first.descriptors.len(), 3);

    let q_projection = prepared
        .dispatch("layer_00", "q_projection__k_projection__v_projection")
        .unwrap();
    let q_family = reusable_family_with_kernel(
        &reusable_plan,
        "layer_00.q_projection__k_projection__v_projection",
    );
    assert_eq!(q_projection.dispatch_index, 1);
    assert_eq!(q_projection.reusable_family_id, q_family.family_id);
    assert_eq!(q_projection.artifact_path, artifact_path_for_family(q_family));

    let kv_append = prepared
        .dispatch("layer_00", "kv_memory_append__attention_read")
        .unwrap();
    let kv_family = reusable_family_with_kernel(
        &reusable_plan,
        "layer_00.kv_memory_append__attention_read",
    );
    assert_eq!(kv_append.dispatch_index, 5);
    assert_eq!(kv_append.reusable_family_id, kv_family.family_id);
    assert!(kv_append.stream_control_binding.is_some());
    assert_eq!(kv_append.stream_control_binding, Some(7));
    assert_eq!(kv_append.descriptors.len(), 7);
    let state = resident_plan
        .stream_state_buffers
        .iter()
        .find(|state| state.component_id == "layer_00" && state.state_id == "kv_memory")
        .unwrap();
    let state_bytes = descriptor_state_byte_capacity(state, 4).unwrap();
    for descriptor_index in [3, 5, 6] {
        assert!(matches!(
            kv_append.descriptors[descriptor_index].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref component_id,
                ref state_id,
                byte_capacity,
                ..
            } if component_id == "layer_00"
                && state_id == "kv_memory"
                && byte_capacity == state_bytes
        ));
    }
}

fn contract_attachment_test_dispatch(
    dispatch_index: usize,
    component_id: &str,
) -> VulkanPreparedDispatch {
    VulkanPreparedDispatch {
        dispatch_index,
        kernel_id: format!("{component_id}.operator_norm"),
        component_id: component_id.to_string(),
        circuit_id: component_id.to_string(),
        node_index: 0,
        node_id: "operator_norm".to_string(),
        op: "rms_norm".to_string(),
        reusable_family_id: "shared-rms-norm-family".to_string(),
        artifact_path: "shaders/shared-rms-norm.spv".to_string(),
        entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
        local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
        descriptors: Vec::new(),
        push_constants: Vec::new(),
        stream_control_binding: None,
        physical_execution_contracts: Vec::new(),
    }
}

fn contract_attachment_test_plan() -> VulkanPreparedDispatchPlan {
    VulkanPreparedDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        reusable_family_count: 1,
        dispatches: vec![
            contract_attachment_test_dispatch(0, "layer_00"),
            contract_attachment_test_dispatch(1, "layer_01"),
        ],
        total_descriptor_count: 0,
    }
}

fn contract_attachment_test_shader(
    component_id: &str,
    contract_suffix: char,
) -> VulkanResidentComponentKernelShaderRef {
    let mut contract = fixture_model_package_manifest()
        .component_executions
        .into_iter()
        .flat_map(|component| component.kernels)
        .find(|kernel| kernel.node_id == "operator_norm")
        .unwrap()
        .physical_execution_contracts
        .into_iter()
        .next()
        .unwrap();
    contract.contract_id = format!("sha256:{}", contract_suffix.to_string().repeat(64));
    VulkanResidentComponentKernelShaderRef {
        component_id: component_id.to_string(),
        node_id: "operator_norm".to_string(),
        shader_path: "shaders/shared-rms-norm.spv".to_string(),
        local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
        workgroup_count_x: 1,
        physical_execution_contracts: vec![contract],
    }
}

#[test]
fn prepared_dispatches_share_code_without_sharing_instance_contracts() {
    let mut plan = contract_attachment_test_plan();
    let shaders = vec![
        contract_attachment_test_shader("layer_00", 'a'),
        contract_attachment_test_shader("layer_01", 'b'),
    ];

    attach_resident_package_physical_execution_contracts(&mut plan, &shaders).unwrap();

    assert_eq!(
        plan.dispatches[0].reusable_family_id,
        plan.dispatches[1].reusable_family_id
    );
    assert_eq!(
        plan.dispatches[0].physical_execution_contracts[0].contract_id,
        format!("sha256:{}", "a".repeat(64))
    );
    assert_eq!(
        plan.dispatches[1].physical_execution_contracts[0].contract_id,
        format!("sha256:{}", "b".repeat(64))
    );
}

#[test]
fn physical_artifact_metadata_is_owned_by_its_contract() {
    let mut contract = fixture_model_package_manifest()
        .component_executions
        .into_iter()
        .flat_map(|execution| execution.kernels)
        .flat_map(|kernel| kernel.physical_execution_contracts)
        .find(|contract| contract.strategy.is_distributed())
        .unwrap();
    contract.partition_launch = Some(nerve_execution_contracts::PartitionLaunch {
        workgroup_x: nerve_execution_contracts::WorkgroupXMapping::Repeated,
        origin: nerve_execution_contracts::PartitionOrigin::PushConstantU32,
        origin_push_constant: Some("input_start".to_string()),
        count_push_constant: Some("input_count".to_string()),
    });
    let mut family = fixture_model_reusable_kernel_plan()
        .families
        .into_iter()
        .find(|family| family.op == contract.operation_family)
        .unwrap();
    let unused_slot = VulkanKernelDescriptorSlotSignature {
        binding: 99,
        usage: VulkanKernelDescriptorUsage::StateRead,
        resource_class: VulkanKernelDescriptorResourceClass::StateBuffer,
        byte_capacity: Some(16),
        shape: None,
    };
    family.descriptor_signature.push(unused_slot.clone());
    family.stream_control_binding = Some(99);
    let artifact = physical_contract_kernel_artifact(
        &family,
        &contract,
        0,
        &contract.artifacts[0],
    )
    .unwrap();

    assert_eq!(
        artifact.artifact_id,
        physical_execution_artifact_id(&contract.contract_id, 0)
    );
    assert_ne!(artifact.artifact_id, family.family_id);
    assert_eq!(artifact.path, contract.artifacts[0].path);
    assert_eq!(
        artifact.local_size_x,
        u32::try_from(contract.geometry.dimensions["local_size_x"]).unwrap()
    );
    assert_eq!(
        artifact.workgroup_count_x,
        u32::try_from(contract.geometry.dimensions["workgroup_count_x"]).unwrap()
    );
    assert_eq!(
        artifact
            .push_constants
            .iter()
            .map(|binding| binding.name.as_str())
            .collect::<Vec<_>>(),
        ["input_start", "input_count"]
    );
    assert!(!artifact.descriptor_signature.contains(&unused_slot));
    assert_eq!(artifact.stream_control_binding, None);
}

#[test]
fn physical_artifact_rejects_a_contract_binding_absent_from_the_canonical_abi() {
    let mut contract = fixture_model_package_manifest()
        .component_executions
        .into_iter()
        .flat_map(|execution| execution.kernels)
        .flat_map(|kernel| kernel.physical_execution_contracts)
        .find(|contract| contract.strategy.is_distributed())
        .unwrap();
    contract.inputs[0].binding = 99;
    let family = fixture_model_reusable_kernel_plan()
        .families
        .into_iter()
        .find(|family| family.op == contract.operation_family)
        .unwrap();
    let error = physical_contract_kernel_artifact(
        &family,
        &contract,
        0,
        &contract.artifacts[0],
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("binding 99 does not identify exactly one canonical descriptor slot"));
}

#[test]
fn prepared_dispatch_contract_attachment_rejects_a_missing_instance() {
    let mut plan = contract_attachment_test_plan();
    let error = attach_resident_package_physical_execution_contracts(
        &mut plan,
        &[contract_attachment_test_shader("layer_00", 'a')],
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("no physical contracts for prepared dispatch layer_01.operator_norm"));
}

#[test]
fn prepared_dispatch_contract_attachment_rejects_a_duplicate_instance() {
    let mut plan = contract_attachment_test_plan();
    let duplicate = contract_attachment_test_shader("layer_00", 'b');
    let error = attach_resident_package_physical_execution_contracts(
        &mut plan,
        &[
            contract_attachment_test_shader("layer_00", 'a'),
            duplicate,
        ],
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("repeats kernel contract source layer_00.operator_norm"));
}

#[test]
fn prepared_dispatch_contract_attachment_rejects_an_empty_contract_set() {
    let mut plan = contract_attachment_test_plan();
    let mut empty = contract_attachment_test_shader("layer_00", 'a');
    empty.physical_execution_contracts.clear();
    let error = attach_resident_package_physical_execution_contracts(
        &mut plan,
        &[empty, contract_attachment_test_shader("layer_01", 'b')],
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains(
        "empty physical contract set for prepared dispatch layer_00.operator_norm"
    ));
}

#[test]
fn prepared_dispatch_plan_rejects_a_compiled_stream_control_binding_mismatch() {
    let error = validate_stream_control_binding(19, Some(6), 5).unwrap_err();

    assert_eq!(
        error,
        VulkanPreparedDispatchPlanError::StreamControlBindingMismatch {
            dispatch_index: 19,
            compiled_binding: 6,
            runtime_descriptor_count: 5,
        }
    );
}

#[test]
fn prepared_dispatch_plan_rejects_unlinked_reusable_kernels() {
    let reusable_plan = fixture_model_reusable_kernel_plan();
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let selected = reusable_family_with_kernel(&reusable_plan, "layer_00.q_projection__k_projection__v_projection");
    let missing = reusable_family_with_kernel(&reusable_plan, "layer_00.kv_memory_append__attention_read");
    let partial_manifest = VulkanReusableKernelArtifactManifest::empty().with_artifact(
        VulkanReusableKernelArtifact::from_family(selected, artifact_path_for_family(selected)),
    );

    let error = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &partial_manifest,
    )
    .unwrap_err();

    let VulkanPreparedDispatchPlanError::Link(link_plan) = error else {
        panic!("expected reusable kernel link failure");
    };
    assert_eq!(link_plan.linked_family_count, 1);
    assert_eq!(
        link_plan.missing_family_count,
        reusable_plan.total_family_count() - 1
    );
    assert_eq!(
        link_plan.linked_command_count,
        selected.command_refs.len()
    );
    assert_eq!(
        link_plan.missing_command_count,
        reusable_plan.total_command_count - selected.command_refs.len()
    );
    assert!(
        link_plan
            .family(&missing.family_id)
            .is_some_and(|family| family.status == VulkanReusableKernelLinkStatus::Missing)
    );
}

#[test]
fn bound_dispatch_plan_maps_prepared_descriptors_to_mounted_stream_buffers() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let binding_plan =
        VulkanStreamCircuitBindingPlan::from_plans(&execution_plan, &resource_plan, &resident_plan)
            .unwrap();
    let dispatch_plan = VulkanKernelDispatchPlan::from_binding_plan(&binding_plan);
    let reusable_plan = VulkanReusableKernelPlan::from_dispatch_plan(&dispatch_plan);
    let descriptor_plan =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 4).unwrap();
    let manifest = VulkanReusableKernelArtifactManifest::new(
        reusable_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let prepared = VulkanPreparedDispatchPlan::from_plans(
        &dispatch_plan,
        &reusable_plan,
        &descriptor_plan,
        &manifest,
    )
    .unwrap();
    let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

    let bound = VulkanBoundDispatchPlan::from_prepared_plan(&prepared, &buffers).unwrap();

    assert_eq!(bound.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(bound.dispatches.len(), 10);
    assert_eq!(
        bound.boundary_descriptor_count
            + bound.permanent_parameter_descriptor_count
            + bound.stream_state_descriptor_count
            + bound.activation_slot_descriptor_count,
        bound.total_descriptor_count
    );

    let first = bound.dispatch("layer_00", "operator_norm").unwrap();
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(
        first.descriptors[0].target,
        VulkanBoundDescriptorTarget::BoundaryInput {
            signal_id: "input_frame".to_string(),
        }
    );
    assert!(matches!(
        first.descriptors[1].target,
        VulkanBoundDescriptorTarget::ActivationSlot {
            ref component_id,
            ref signal_id,
            byte_capacity: 32,
            signal_byte_capacity: 32,
            ..
        } if component_id == "layer_00" && signal_id == "operator_norm_out"
    ));
    assert_eq!(
        first.descriptors[2].target,
        VulkanBoundDescriptorTarget::PermanentParameter {
            param_id: "operator_norm".to_string(),
            tensor: "model.layers.0.input_layernorm.weight".to_string(),
            byte_count: Some(32),
        }
    );

    let kv_append = bound
        .dispatch("layer_00", "kv_memory_append__attention_read")
        .unwrap();
    let expected_state_bytes = buffers
        .state_buffer("layer_00", "kv_memory")
        .unwrap()
        .byte_capacity;
    for descriptor_index in [3, 5, 6] {
        assert!(matches!(
            kv_append.descriptors[descriptor_index].target,
            VulkanBoundDescriptorTarget::StreamStateBuffer {
                ref component_id,
                ref state_id,
                byte_capacity,
                ..
            } if component_id == "layer_00"
                && state_id == "kv_memory"
                && byte_capacity == expected_state_bytes
        ));
    }
}

#[test]
fn mounts_fixture_model_stream_circuit_resources_without_claiming_execution() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    let mounted = VulkanMountedStreamCircuit::from_plans(
        &device,
        &execution_plan,
        &resource_plan,
        resident_plan,
        4,
    )
    .unwrap();

    assert!(!mounted.can_execute());
    assert_eq!(mounted.resident_plan.permanent_parameters.len(), 9);
    assert_eq!(mounted.binding_plan.total_node_count(), 10);
    assert_eq!(mounted.kernel_interface_plan.total_kernel_count(), 10);
    assert_eq!(mounted.dispatch_plan.total_dispatch_count(), 10);
    assert_eq!(
        mounted.reusable_kernel_plan.total_command_count,
        mounted.dispatch_plan.total_dispatch_count()
    );
    let empty_coverage = mounted.reusable_kernel_coverage_report(std::iter::empty::<&str>());
    assert!(!empty_coverage.all_available());
    assert_eq!(
        empty_coverage.missing_family_count,
        mounted.reusable_kernel_plan.total_family_count()
    );
    assert_eq!(empty_coverage.missing_command_count, 10);
    let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
    assert_eq!(
        descriptor_plan.total_descriptor_count,
        descriptor_plan
            .dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum::<usize>()
    );
    assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
    let manifest = VulkanReusableKernelArtifactManifest::new(
        mounted
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    );
    let prepared = mounted.prepared_dispatch_plan(&manifest).unwrap();
    assert_eq!(prepared.dispatches.len(), 10);
    let bound = mounted.bound_dispatch_plan(&manifest).unwrap();
    assert_eq!(bound.dispatches.len(), 10);
    assert_eq!(mounted.buffers.state_buffers.len(), 1);
    assert_eq!(mounted.buffers.activation_slot_buffers.len(), 4);
    assert_eq!(
        mounted.buffers.total_byte_capacity,
        mounted
            .buffers
            .state_buffers
            .iter()
            .map(|buffer| buffer.byte_capacity)
            .chain(
                mounted
                    .buffers
                    .activation_slot_buffers
                    .iter()
                    .map(|buffer| buffer.byte_capacity)
            )
            .sum::<usize>()
    );

    let attention = mounted
        .binding_plan
        .circuit("layer_00")
        .unwrap()
        .node("kv_memory_append__attention_read__partition_partials")
        .unwrap();
    assert!(matches!(
        attention.input("kv_memory").unwrap().resource,
        VulkanSignalResource::StateBuffer { .. }
    ));
    assert_eq!(
        mounted
            .buffers
            .activation_slot_buffer("layer_00", 0)
            .map(|buffer| buffer.byte_capacity),
        Some(32)
    );
    assert_eq!(
        mounted
            .dispatch_plan
            .command("layer_00", "kv_memory_append__attention_read")
            .map(|command| command.dispatch_index),
        Some(5)
    );
}
