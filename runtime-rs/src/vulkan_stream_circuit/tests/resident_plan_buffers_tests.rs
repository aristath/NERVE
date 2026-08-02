#[test]
fn resident_plan_infers_state_sizes_without_a_tensor_index() {
    let graph = fixture_model_execution_graph();
    let resource_plan = StreamCircuitResourcePlan::from_graph(&graph).unwrap();

    let resident_plan =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&resource_plan, None, None).unwrap();

    assert_eq!(resident_plan.permanent_parameters.len(), 9);
    assert_eq!(resident_plan.permanent_parameter_bytes, None);
    assert_eq!(resident_plan.unresolved_parameter_tensors.len(), 9);
    assert_eq!(resident_plan.per_stream_static_state_elements, 0);
    assert_eq!(
        resident_plan.per_stream_dynamic_state_elements_per_activation,
        16
    );
    assert_eq!(resident_plan.per_stream_static_state_bytes, Some(0));
    assert_eq!(
        resident_plan.per_stream_dynamic_state_bytes_per_activation,
        Some(32)
    );
    assert_eq!(resident_plan.per_stream_activation_slot_elements, None);
    assert_eq!(resident_plan.per_stream_activation_slot_bytes, None);
    assert!(!resident_plan.unresolved_activation_slots.is_empty());
}

#[test]
fn permanent_parameter_plan_excludes_physically_lowered_tensors() {
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_plan = graph
        .placement_plan(&StreamCircuitPlacementSpec::new("gpu0"))
        .unwrap();
    let placed_resident_plan =
        VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
            &resource_plan,
            &placement_plan,
            "gpu0",
            Some(&tensor_index),
            Some(2),
        )
        .unwrap();
    let full = VulkanPermanentParameterBufferPlan::from_placed_resident_plan(&placed_resident_plan)
        .unwrap();
    let removed = full.parameters.iter().take(2).cloned().collect::<Vec<_>>();
    let excluded = removed
        .iter()
        .map(|parameter| parameter.tensor.clone())
        .collect::<BTreeSet<_>>();

    let pruned = VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
        &placed_resident_plan,
        &excluded,
    )
    .unwrap();

    assert_eq!(pruned.parameter_count, full.parameter_count - 2);
    assert_eq!(
        pruned.total_byte_capacity,
        Some(
            full.total_byte_capacity.unwrap()
                - removed
                    .iter()
                    .map(|parameter| parameter.byte_capacity.unwrap())
                    .sum::<usize>()
        )
    );
    assert!(
        pruned
            .parameters
            .iter()
            .all(|parameter| !excluded.contains(&parameter.tensor))
    );
    assert!(
        pruned
            .parameters
            .iter()
            .enumerate()
            .all(|(index, parameter)| parameter.buffer_index == index)
    );

    let error = VulkanPermanentParameterBufferPlan::from_placed_resident_plan_excluding_tensors(
        &placed_resident_plan,
        &BTreeSet::from(["not-a-resident-tensor".to_string()]),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot exclude unavailable permanent parameter tensor")
    );
}

#[test]
fn allocates_fixture_model_per_stream_vulkan_buffers_from_resident_plan() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let graph = fixture_model_execution_graph();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    let buffers = resident_plan.allocate_stream_buffers(&device, 4).unwrap();

    assert_eq!(buffers.dynamic_state_capacity_activations, 4);
    assert_eq!(buffers.state_buffers.len(), 1);
    assert_eq!(buffers.activation_slot_buffers.len(), 4);
    assert_eq!(
        buffers.total_byte_capacity,
        buffers
            .state_buffers
            .iter()
            .map(|buffer| buffer.byte_capacity)
            .chain(
                buffers
                    .activation_slot_buffers
                    .iter()
                    .map(|buffer| buffer.byte_capacity)
            )
            .sum::<usize>()
    );

    let layer_00_state = buffers
        .state_buffers
        .iter()
        .find(|buffer| buffer.component_id == "layer_00")
        .unwrap();
    assert_eq!(layer_00_state.state_id, "kv_memory");
    assert_eq!(
        layer_00_state.buffer.byte_capacity(),
        layer_00_state.byte_capacity
    );

    let layer_00_slot_0 = buffers
        .activation_slot_buffers
        .iter()
        .find(|buffer| buffer.component_id == "layer_00" && buffer.slot == 0)
        .unwrap();
    assert_eq!(layer_00_slot_0.byte_capacity, 64);
    assert!(
        layer_00_slot_0
            .signal_ids
            .contains(&"operator_norm_out".to_string())
    );
    assert_eq!(
        buffers
            .state_buffer("layer_00", "kv_memory")
            .map(|buffer| buffer.byte_capacity),
        Some(layer_00_state.byte_capacity)
    );
    assert_eq!(
        buffers
            .activation_slot_buffer("layer_00", 0)
            .map(|buffer| buffer.byte_capacity),
        Some(64)
    );
}

#[test]
fn binds_fixture_model_nodes_to_vulkan_resident_resources() {
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

    assert_eq!(binding_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(binding_plan.circuits.len(), 1);
    assert_eq!(binding_plan.total_node_count(), 10);

    let layer_00 = binding_plan.circuit("layer_00").unwrap();
    let operator_norm = layer_00.node("operator_norm").unwrap();
    assert_eq!(
        operator_norm.input("input_frame").unwrap().resource,
        VulkanSignalResource::BoundaryInput
    );
    assert_eq!(
        operator_norm.parameter("operator_norm").unwrap().tensor,
        "model.layers.0.input_layernorm.weight"
    );

    let q_projection = layer_00
        .node("q_projection__k_projection__v_projection")
        .unwrap();
    assert_eq!(
        q_projection.parameter("q_projection").unwrap().tensor,
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert!(matches!(
        q_projection.output("q_projected").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(32),
            ..
        } if component_id == "layer_00"
    ));

    let kv_append = layer_00
        .node("kv_memory_append__attention_read")
        .unwrap();
    assert_eq!(
        kv_append.input("kv_memory").unwrap().resource,
        VulkanSignalResource::StateBuffer {
            component_id: "layer_00".to_string(),
            state_id: "kv_memory".to_string(),
            static_bytes: None,
            bytes_per_activation: Some(32),
        }
    );
    assert!(matches!(
        kv_append.output("attention_out").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(32),
            ..
        } if component_id == "layer_00"
    ));

    let attention = layer_00
        .node("kv_memory_append__attention_read__partition_partials")
        .unwrap();
    assert!(matches!(
        attention.input("q_positioned").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(32),
            ..
        }
        if component_id == "layer_00"
    ));
    assert!(matches!(
        attention.output("kv_memory_append__attention_read__attention_partials_f32").unwrap().resource,
        VulkanSignalResource::ActivationSlot {
            ref component_id,
            signal_bytes: Some(bytes),
            ..
        }
        if component_id == "layer_00" && bytes > 0
    ));
}
