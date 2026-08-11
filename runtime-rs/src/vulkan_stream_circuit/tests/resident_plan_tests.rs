#[test]
fn resident_plan_uses_typed_activation_slot_byte_capacity() {
    let resource_plan = StreamCircuitResourcePlan {
        circuit_count: 1,
        node_count: 2,
        parameter_ref_count: 0,
        parameters: Vec::new(),
        transducer_parameter_ref_count: 0,
        transducer_parameters: Vec::new(),
        state_allocations: Vec::new(),
        selection_domains: Vec::new(),
        activation_banks: vec![crate::stream_plan::PlannedActivationSlotBank {
            component_id: "layer_00".to_string(),
            circuit_id: "layer_00".to_string(),
            temporary_signal_count: 2,
            state_view_signal_count: 0,
            slot_count: 1,
            slots: vec![crate::stream_plan::PlannedActivationSlot {
                slot: 0,
                signal_ids: vec!["bf16_signal".to_string(), "f32_signal".to_string()],
                max_elements: Some(8),
                max_bytes: Some(32),
            }],
            assignments: Vec::new(),
        }],
        temporary_signal_count: 2,
        state_view_signal_count: 0,
        layer_local_activation_slot_count: 1,
        unknown_temporary_shape_count: 0,
        unknown_state_view_shape_count: 0,
    };

    let resident_plan =
        VulkanStreamCircuitResidentPlan::from_resource_plan(&resource_plan, None, Some(2))
            .unwrap();

    assert_eq!(resident_plan.per_stream_activation_slot_elements, Some(8));
    assert_eq!(resident_plan.per_stream_activation_slot_bytes, Some(32));
    assert_eq!(
        resident_plan.activation_banks[0].slots[0].bytes,
        Some(32)
    );
    assert!(resident_plan.unresolved_activation_slots.is_empty());
}

#[test]
fn plans_fixture_model_vulkan_resident_allocations_from_stream_circuit_resources() {
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

    assert_eq!(resident_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(resident_plan.circuit_count, graph.circuits.len());
    assert_eq!(resident_plan.permanent_parameters.len(), 9);
    assert_eq!(resident_plan.permanent_parameter_bytes, Some(4_672));
    assert!(resident_plan.unresolved_parameter_tensors.is_empty());
    assert_eq!(resident_plan.stream_state_buffers.len(), 1);
    assert_eq!(resident_plan.state_view_signal_count, 2);
    assert_eq!(resident_plan.activation_banks.len(), 1);
    assert_eq!(resident_plan.per_stream_static_state_elements, 0);
    assert_eq!(
        resident_plan.per_stream_dynamic_state_elements_per_activation,
        16
    );
    assert_eq!(resident_plan.per_stream_dynamic_state_bytes_per_activation, Some(32));
    assert!(resident_plan.per_stream_activation_slot_elements.unwrap() > 0);
    assert_eq!(resident_plan.per_stream_static_state_bytes, Some(0));
    assert!(resident_plan.per_stream_activation_slot_bytes.unwrap() > 0);
    assert!(resident_plan.unresolved_activation_slots.is_empty());

    let q_projection = resident_plan
        .permanent_parameters
        .iter()
        .find(|parameter| parameter.tensor == "model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_projection.dtype.as_deref(), Some("BF16"));
    assert_eq!(q_projection.shape, Some(vec![16, 16]));
    assert_eq!(q_projection.byte_count, Some(512));
    assert_eq!(q_projection.use_count, 1);

    let layer_00_bank = resident_plan.activation_bank("layer_00").unwrap();
    assert_eq!(
        layer_00_bank
            .slots
            .iter()
            .map(|slot| slot.bytes)
            .collect::<Vec<_>>(),
        vec![Some(64), Some(32), Some(64), Some(64)]
    );
}

#[test]
fn bounds_each_dynamic_state_buffer_by_its_own_activation_limit() {
    let mut graph = fixture_model_execution_graph();
    let artifact = graph
        .circuits
        .iter_mut()
        .find(|artifact| artifact.component.id == "layer_00")
        .unwrap();
    artifact
        .state
        .state_ports
        .iter_mut()
        .find(|state| state.id == "kv_memory")
        .unwrap()
        .max_dynamic_activations = Some(2);
    artifact
        .circuit
        .state_ports
        .iter_mut()
        .find(|state| state.id == "kv_memory")
        .unwrap()
        .max_dynamic_activations = Some(2);

    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resident_plan = VulkanStreamCircuitResidentPlan::from_resource_plan(
        &resource_plan,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let state = resident_plan
        .stream_state_buffers
        .iter()
        .find(|state| state.component_id == "layer_00" && state.state_id == "kv_memory")
        .unwrap();

    assert_eq!(state.dtype.as_deref(), Some("BF16"));
    assert_eq!(state.max_dynamic_activations, Some(2));
    let layout = VulkanTransientStateBufferLayout::for_state(state, 4).unwrap();
    assert_eq!(
        layout.dynamic_page_byte_capacity,
        state.bytes_per_activation.unwrap() * 2
    );
    assert!(layout.byte_capacity > layout.dynamic_page_byte_capacity);
    assert_eq!(
        descriptor_state_byte_capacity(state, 4).unwrap(),
        layout.byte_capacity
    );
}

#[test]
fn placed_resident_plan_hosts_only_the_components_assigned_to_a_device() {
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let graph = runtime_model
        .circuit_graph
        .to_signal_processor_graph(tiny_model_dir())
        .unwrap();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let mut resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    resource_plan.selection_domains.extend([
        crate::stream_plan::PlannedSelectionDomain {
            component_id: "layer_00".to_string(),
            circuit_id: "layer_00".to_string(),
            node_id: "selector".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 256,
            selection_count_per_activation: 8,
        },
        crate::stream_plan::PlannedSelectionDomain {
            component_id: "layer_00_remote".to_string(),
            circuit_id: "layer_00_remote".to_string(),
            node_id: "selector".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 128,
            selection_count_per_activation: 1,
        },
    ]);
    let placement_plan = graph.placement_plan(&runtime_model.placement).unwrap();

    let gpu0 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1 = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    assert_eq!(gpu0.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(gpu0.device_id, "gpu0");
    assert_eq!(
        gpu0.hosted_component_ids,
        vec!["layer_00".to_string(), "layer_00_tail".to_string()]
    );
    assert!(!gpu0.hosts_component("layer_00_remote"));
    assert_eq!(gpu0.resident_plan.circuit_count, 2);
    assert_eq!(gpu0.resident_plan.permanent_parameters.len(), 9);
    assert_eq!(gpu0.resident_plan.stream_state_buffers.len(), 2);
    assert_eq!(gpu0.resident_plan.selection_telemetry.len(), 1);
    assert_eq!(
        gpu0.resident_plan.selection_telemetry[0],
        VulkanResidentSelectionTelemetry {
            component_id: "layer_00".to_string(),
            node_id: "selector".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 256,
            co_selection_pair_count: 256 * 255 / 2,
            byte_capacity: (256 + 256 * 255 / 2) * std::mem::size_of::<u32>(),
        }
    );
    assert_eq!(gpu0.resident_plan.activation_banks.len(), 2);
    assert_eq!(gpu0.resident_plan.state_view_signal_count, 0);
    assert_eq!(gpu0.signal_element_bytes, Some(2));
    assert_eq!(gpu0.local_edges.len(), 0);
    assert_eq!(gpu0.incoming_edges.len(), 1);
    assert_eq!(gpu0.outgoing_edges.len(), 1);
    assert_eq!(
        gpu0.incoming_edges[0].source_component_id,
        "layer_00_remote"
    );
    assert_eq!(
        gpu0.incoming_edges[0].destination_component_id,
        "layer_00_tail"
    );
    assert_eq!(gpu0.outgoing_edges[0].source_component_id, "layer_00");
    assert_eq!(
        gpu0.outgoing_edges[0].destination_component_id,
        "layer_00_remote"
    );

    let gpu0_edge_io = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&gpu0).unwrap();
    assert_eq!(gpu0_edge_io.local_edge_count, 0);
    assert_eq!(gpu0_edge_io.total_endpoint_count, 2);
    assert_eq!(gpu0_edge_io.total_buffer_count, 2);
    assert_eq!(
        gpu0_edge_io.total_byte_capacity,
        Some(2 * FIXTURE_MODEL_FRAME_BYTES)
    );

    assert_eq!(
        gpu1.hosted_component_ids,
        vec!["layer_00_remote".to_string()]
    );
    assert_eq!(gpu1.resident_plan.circuit_count, 1);
    assert_eq!(gpu1.resident_plan.permanent_parameters.len(), 9);
    assert_eq!(gpu1.resident_plan.stream_state_buffers.len(), 1);
    assert_eq!(gpu1.resident_plan.selection_telemetry.len(), 1);
    assert_eq!(
        gpu1.resident_plan.selection_telemetry[0],
        VulkanResidentSelectionTelemetry {
            component_id: "layer_00_remote".to_string(),
            node_id: "selector".to_string(),
            domain_id: "resources".to_string(),
            resource_count: 128,
            co_selection_pair_count: 0,
            byte_capacity: 128 * std::mem::size_of::<u32>(),
        }
    );
    assert_eq!(gpu1.resident_plan.state_view_signal_count, 0);
    assert_eq!(gpu1.incoming_edges[0].source_component_id, "layer_00");
    assert_eq!(
        gpu1.outgoing_edges[0].destination_component_id,
        "layer_00_tail"
    );
    let gpu1_edge_io = VulkanPlacedEdgeIoPlan::from_placed_resident_plan(&gpu1).unwrap();
    assert_eq!(gpu1_edge_io.local_edge_count, 0);
    assert_eq!(gpu1_edge_io.total_endpoint_count, 2);
    assert_eq!(gpu1_edge_io.total_buffer_count, 2);
    assert_eq!(
        gpu1_edge_io.total_byte_capacity,
        Some(2 * FIXTURE_MODEL_FRAME_BYTES)
    );
    assert!(gpu1_edge_io.unresolved_byte_edges.is_empty());

    let incoming = gpu1_edge_io
        .endpoints
        .iter()
        .find(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        .unwrap();
    assert_eq!(incoming.shape, vec![FIXTURE_MODEL_HIDDEN_SIZE]);
    assert_eq!(incoming.element_count, FIXTURE_MODEL_HIDDEN_SIZE);
    assert_eq!(incoming.byte_capacity, Some(FIXTURE_MODEL_FRAME_BYTES));
    assert_eq!(incoming.local_device_id, "gpu1");
    assert_eq!(incoming.remote_device_id, "gpu0");
    assert_eq!(incoming.local_component_id, "layer_00_remote");
    assert_eq!(incoming.remote_component_id, "layer_00");

    let outgoing = gpu1_edge_io
        .endpoints
        .iter()
        .find(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Outgoing)
        .unwrap();
    assert_eq!(outgoing.byte_capacity, Some(FIXTURE_MODEL_FRAME_BYTES));
    assert_eq!(outgoing.local_device_id, "gpu1");
    assert_eq!(outgoing.remote_device_id, "gpu0");
    assert_eq!(outgoing.local_component_id, "layer_00_remote");
    assert_eq!(outgoing.remote_component_id, "layer_00_tail");

    let edge_plans = vec![gpu0_edge_io, gpu1_edge_io];
    let edge_pairs = pair_placed_edge_endpoints(&edge_plans).unwrap();
    assert_eq!(edge_pairs.len(), 2);
    assert!(edge_pairs.iter().all(|(outgoing, incoming)| {
        VulkanPlacedEdgePacketKey::from_outgoing_endpoint(outgoing)
            == VulkanPlacedEdgePacketKey::from_incoming_endpoint(incoming)
            && outgoing.byte_capacity == incoming.byte_capacity
    }));

    let mut incomplete_plans = edge_plans;
    let incoming_index = incomplete_plans[0]
        .endpoints
        .iter()
        .position(|endpoint| endpoint.direction == VulkanPlacedEdgeDirection::Incoming)
        .unwrap();
    incomplete_plans[0].endpoints.remove(incoming_index);
    assert!(
        pair_placed_edge_endpoints(&incomplete_plans)
            .unwrap_err()
            .to_string()
            .contains("has no incoming endpoint")
    );
}

#[test]
fn placed_stream_circuit_plan_dispatches_only_hosted_components() {
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let graph = runtime_model
        .circuit_graph
        .to_signal_processor_graph(tiny_model_dir())
        .unwrap();
    let tensor_index = TensorIndex::from_json_file(fixture_model_tensor_index_path()).unwrap();
    let execution_plan =
        StreamCircuitExecutionPlan::from_graph_with_tensor_index(&graph, &tensor_index).unwrap();
    let resource_plan =
        StreamCircuitResourcePlan::from_graph_and_plan(&graph, &execution_plan).unwrap();
    let placement_plan = graph.placement_plan(&runtime_model.placement).unwrap();
    let gpu0_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu0",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();
    let gpu1_resident = VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        "gpu1",
        Some(&tensor_index),
        Some(2),
    )
    .unwrap();

    let gpu0_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu0_resident)
            .unwrap();
    let gpu1_plan =
        VulkanPlacedStreamCircuitPlan::from_plans(&execution_plan, &resource_plan, gpu1_resident)
            .unwrap();

    assert_eq!(gpu0_plan.backend_id, VULKAN_STREAM_CIRCUIT_BACKEND_ID);
    assert_eq!(gpu0_plan.device_id, "gpu0");
    assert_eq!(gpu0_plan.binding_plan.circuits.len(), 2);
    assert_eq!(gpu0_plan.binding_plan.total_node_count(), 18);
    assert_eq!(gpu0_plan.kernel_interface_plan.total_kernel_count(), 18);
    assert_eq!(gpu0_plan.dispatch_plan.total_dispatch_count(), 18);
    assert!(gpu0_plan.binding_plan.circuit("layer_00").is_some());
    assert!(gpu0_plan.binding_plan.circuit("layer_00_tail").is_some());
    assert!(gpu0_plan.binding_plan.circuit("layer_00_remote").is_none());
    assert!(
        gpu0_plan
            .dispatch_plan
            .command(
                "layer_00_remote",
                "kv_memory_append__attention_read"
            )
            .is_none()
    );

    assert_eq!(gpu1_plan.device_id, "gpu1");
    assert_eq!(gpu1_plan.binding_plan.circuits.len(), 1);
    assert_eq!(gpu1_plan.binding_plan.total_node_count(), 9);
    assert_eq!(gpu1_plan.dispatch_plan.total_dispatch_count(), 9);
    assert_eq!(
        gpu1_plan
            .dispatch_plan
            .command("layer_00_remote", "operator_norm")
            .map(|command| command.dispatch_index),
        Some(0)
    );
    assert_eq!(
        gpu1_plan
            .dispatch_plan
            .command(
                "layer_00_remote",
                "kv_memory_append__attention_read"
            )
            .map(|command| command.dispatch_index),
        Some(4)
    );
    assert!(
        gpu1_plan
            .dispatch_plan
            .command("layer_00", "operator_norm")
            .is_none()
    );

    let gpu1_manifest = resident_package_reusable_kernel_manifest(&gpu1_plan);
    let gpu1_prepared = gpu1_plan.prepared_dispatch_plan(&gpu1_manifest, 4).unwrap();
    assert_eq!(gpu1_prepared.dispatches.len(), 9);
    assert!(
        gpu1_prepared
            .dispatch("layer_00_remote", "operator_norm")
            .is_some()
    );
    assert!(
        gpu1_prepared
            .dispatch("layer_00", "operator_norm")
            .is_none()
    );

    let gpu1_descriptors = VulkanDescriptorResourcePlan::from_plans(
        &gpu1_plan.dispatch_plan,
        &gpu1_plan.placed_resident_plan.resident_plan,
        4,
    )
    .unwrap();
    assert_eq!(gpu1_descriptors.dispatches.len(), 9);
    let kv_append = gpu1_descriptors
        .dispatch("layer_00_remote", "kv_memory_append__attention_read")
        .unwrap();
    let state = gpu1_plan
        .placed_resident_plan
        .resident_plan
        .stream_state_buffers
        .iter()
        .find(|state| {
            state.component_id == "layer_00_remote" && state.state_id == "kv_memory"
        })
        .unwrap();
    let expected_state_bytes = descriptor_state_byte_capacity(state, 4).unwrap();
    assert!(kv_append.descriptors.iter().any(|descriptor| {
        matches!(
            descriptor.resource,
            VulkanDescriptorResourceAddress::StateBuffer {
                ref component_id,
                ref state_id,
                byte_capacity,
                ..
            } if component_id == "layer_00_remote"
                && state_id == "kv_memory"
                && byte_capacity == expected_state_bytes
        )
    }));
}
