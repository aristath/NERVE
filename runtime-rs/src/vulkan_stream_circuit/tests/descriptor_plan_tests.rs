#[test]
fn descriptor_resource_plan_resolves_fixture_model_dispatch_patch_bay() {
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

    assert_eq!(descriptor_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(descriptor_plan.dynamic_state_capacity_activations, 4);
    assert_eq!(descriptor_plan.dispatches.len(), 10);
    assert_eq!(
        descriptor_plan.total_descriptor_count,
        descriptor_plan
            .dispatches
            .iter()
            .map(|dispatch| dispatch.descriptors.len())
            .sum::<usize>()
    );

    let first = descriptor_plan
        .dispatch("layer_00", "operator_norm")
        .unwrap();
    assert_eq!(first.dispatch_index, 0);
    assert_eq!(first.descriptors.len(), 3);
    assert_eq!(
        first.descriptors[0].resource,
        VulkanDescriptorResourceAddress::BoundaryInput {
            signal_id: "input_frame".to_string(),
        }
    );
    assert_eq!(
        first.descriptors[1].resource,
        VulkanDescriptorResourceAddress::ActivationSlot {
            component_id: "layer_00".to_string(),
            signal_id: "operator_norm_out".to_string(),
            slot: 0,
            byte_capacity: 32,
            signal_byte_capacity: 32,
        }
    );
    assert_eq!(
        first.descriptors[2].resource,
        VulkanDescriptorResourceAddress::PermanentParameter {
            param_id: "operator_norm".to_string(),
            tensor: "model.layers.0.input_layernorm.weight".to_string(),
            byte_count: Some(32),
        }
    );

    let kv_append = descriptor_plan
        .dispatch("layer_00", "kv_memory_append__attention_read")
        .unwrap();
    assert_eq!(kv_append.descriptors.len(), 7);
    let state = resident_plan
        .stream_state_buffers
        .iter()
        .find(|state| state.component_id == "layer_00" && state.state_id == "kv_memory")
        .unwrap();
    let expected_state_bytes = descriptor_state_byte_capacity(state, 4).unwrap();
    for descriptor_index in [3, 5, 6] {
        assert!(matches!(
            kv_append.descriptors[descriptor_index].resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref component_id,
                ref state_id,
                byte_capacity,
                bytes_per_activation: Some(32),
                ..
            } if component_id == "layer_00"
                && state_id == "kv_memory"
                && byte_capacity == expected_state_bytes
        ));
    }
    let last = descriptor_plan
        .dispatch("layer_00", "ffn_down_projection__ffn_residual")
        .unwrap();
    let output = last
        .descriptors
        .iter()
        .find(|descriptor| descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal)
        .unwrap();
    assert_eq!(
        output.resource,
        VulkanDescriptorResourceAddress::BoundaryOutput {
            signal_id: "output_frame".to_string(),
        }
    );
}

#[test]
fn descriptor_resource_plan_requires_dynamic_capacity_for_kv_state() {
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

    let error =
        VulkanDescriptorResourcePlan::from_plans(&dispatch_plan, &resident_plan, 0).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("layer_00.kv_memory requires non-zero dynamic state capacity")
    );
}
