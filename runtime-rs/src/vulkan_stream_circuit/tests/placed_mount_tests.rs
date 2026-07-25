fn mounted_remote_middle_slices(
    device: &VulkanComputeDevice,
) -> (
    VulkanMountedPlacedStreamCircuit,
    VulkanMountedPlacedStreamCircuit,
) {
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_dir = fixture_model_package_manifest_path()
        .parent()
        .unwrap()
        .to_path_buf();
    let gpu0_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        device,
        &manifest_dir,
        runtime_model.clone(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let gpu1_slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        device,
        &manifest_dir,
        runtime_model,
        "gpu1",
        Some(4),
    )
    .unwrap();
    (
        gpu0_slice.create_mounted_stream_circuit(device).unwrap(),
        gpu1_slice.create_mounted_stream_circuit(device).unwrap(),
    )
}

#[test]
fn mounted_colocated_stream_circuit_binds_local_edges_between_component_instances() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let manifest_dir = fixture_model_package_manifest_path()
        .parent()
        .unwrap()
        .to_path_buf();
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        &manifest_dir,
        runtime_model,
        "gpu0",
        Some(4),
    )
    .unwrap();
    let mounted = slice.create_mounted_stream_circuit(&device).unwrap();

    assert_eq!(mounted.device_id(), "gpu0");
    assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 3);
    assert_eq!(
        mounted
            .placed_plan
            .binding_plan
            .circuits
            .iter()
            .map(|circuit| circuit.component_id.as_str())
            .collect::<Vec<_>>(),
        vec!["layer_00", "layer_00_remote", "layer_00_tail"]
    );
    assert_eq!(mounted.placed_plan.dispatch_plan.total_dispatch_count(), 27);
    assert_eq!(mounted.boundary_io.plan.input_count, 1);
    assert_eq!(mounted.boundary_io.plan.output_count, 1);
    assert_eq!(mounted.edge_io.local_buffers.len(), 2);
    assert!(mounted.edge_io.incoming_buffers.is_empty());
    assert!(mounted.edge_io.outgoing_buffers.is_empty());
    assert_eq!(mounted.edge_io.total_byte_capacity, 2 * FIXTURE_MODEL_FRAME_BYTES);
    assert!(mounted.edge_io.local_buffers.iter().all(|edge| {
        edge.byte_capacity == FIXTURE_MODEL_FRAME_BYTES
            && edge.buffer.byte_capacity() == FIXTURE_MODEL_FRAME_BYTES
    }));

    let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let bound = mounted
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .unwrap();
    assert_eq!(bound.dispatches.len(), 27);
    assert!(bound.local_edge_descriptor_count > 0);
    assert_eq!(bound.edge_endpoint_descriptor_count, 0);

    let tick_plan = mounted.stream_tick_plan(&reusable_manifest).unwrap();
    assert_eq!(tick_plan.stage_count, 27);
    assert_eq!(tick_plan.dispatch_stage_count, 27);
    assert_eq!(tick_plan.receive_stage_count, 0);
    assert_eq!(tick_plan.publish_stage_count, 0);
    assert!(tick_plan.local_edge_read_count > 0);
    assert!(tick_plan.local_edge_write_count > 0);

    let tick_run = mounted.advance_stream_tick(&reusable_manifest, 42).unwrap();
    assert_eq!(
        tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::KernelDispatchUnavailable,
        }
    );
    assert_eq!(tick_run.attempted_stage_count, 1);
    assert_eq!(tick_run.completed_stage_count, 0);
}

#[test]
fn mounted_placed_stream_circuit_binds_only_local_device_slice() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let (_, mounted) = mounted_remote_middle_slices(&device);

    assert_eq!(mounted.device_id(), "gpu1");
    assert_eq!(mounted.placed_plan.binding_plan.circuits.len(), 1);
    assert_eq!(
        mounted.placed_plan.binding_plan.circuits[0].component_id,
        "layer_00_remote"
    );
    assert_eq!(mounted.placed_plan.dispatch_plan.total_dispatch_count(), 9);
    assert_eq!(mounted.parameter_buffers.plan.parameter_count, 9);
    assert!(mounted.parameter_buffers.plan.unresolved_tensors.is_empty());
    assert_eq!(mounted.boundary_io.plan.total_buffer_count, 0);
    assert_eq!(mounted.buffers.state_buffers.len(), 1);
    assert_eq!(mounted.edge_io.incoming_buffers.len(), 1);
    assert_eq!(mounted.edge_io.outgoing_buffers.len(), 1);
    assert_eq!(mounted.edge_io.total_byte_capacity, 2 * FIXTURE_MODEL_FRAME_BYTES);

    let incoming = &mounted.edge_io.incoming_buffers[0];
    let outgoing = &mounted.edge_io.outgoing_buffers[0];
    assert_eq!(
        incoming.endpoint.direction,
        VulkanPlacedEdgeDirection::Incoming
    );
    assert_eq!(incoming.endpoint.local_component_id, "layer_00_remote");
    assert_eq!(incoming.endpoint.remote_component_id, "layer_00");
    assert_eq!(incoming.byte_capacity, FIXTURE_MODEL_FRAME_BYTES);
    assert_eq!(
        outgoing.endpoint.direction,
        VulkanPlacedEdgeDirection::Outgoing
    );
    assert_eq!(outgoing.endpoint.local_component_id, "layer_00_remote");
    assert_eq!(outgoing.endpoint.remote_component_id, "layer_00_tail");
    assert_eq!(outgoing.byte_capacity, FIXTURE_MODEL_FRAME_BYTES);

    let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let descriptor_plan = mounted.descriptor_resource_plan().unwrap();
    assert_eq!(descriptor_plan.dispatches.len(), 9);
    assert!(
        descriptor_plan
            .dispatch("layer_00_remote", "kv_memory_append__attention_read")
            .is_some()
    );
    assert!(descriptor_plan.dispatch("layer_00", "operator_norm").is_none());

    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .unwrap();
    assert_eq!(mounted_bound.dispatches.len(), 9);
    assert!(mounted_bound.incoming_edge_descriptor_count > 0);
    assert_eq!(mounted_bound.outgoing_edge_descriptor_count, 1);
    assert!(mounted_bound.resident_descriptor_count > 0);
    assert!(mounted_bound.total_descriptor_count > mounted_bound.resident_descriptor_count);

    let tick_plan = mounted.stream_tick_plan(&reusable_manifest).unwrap();
    assert_eq!(tick_plan.stage_count, 11);
    assert_eq!(tick_plan.receive_stage_count, 1);
    assert_eq!(tick_plan.dispatch_stage_count, 9);
    assert_eq!(tick_plan.publish_stage_count, 1);
    assert!(matches!(
        &tick_plan.stages[0],
        VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            byte_capacity,
            remote_device_id,
            remote_component_id,
            ..
        } if *byte_capacity == FIXTURE_MODEL_FRAME_BYTES
            && remote_device_id == "gpu0"
            && remote_component_id == "layer_00"
    ));
    assert!(matches!(
        tick_plan.stages.last().unwrap(),
        VulkanMountedPlacedStreamTickStage::PublishEdge {
            byte_capacity,
            remote_device_id,
            remote_component_id,
            ..
        } if *byte_capacity == FIXTURE_MODEL_FRAME_BYTES
            && remote_device_id == "gpu0"
            && remote_component_id == "layer_00_tail"
    ));

    let tick_run = mounted.advance_stream_tick(&reusable_manifest, 7).unwrap();
    assert_eq!(
        tick_run.status,
        VulkanMountedPlacedStreamTickRunStatus::Blocked {
            stage_index: 0,
            reason: VulkanMountedPlacedStreamTickBlockReason::EdgeReceiveTransportUnavailable,
        }
    );
    assert_eq!(tick_run.attempted_stage_count, 1);
    assert_eq!(tick_run.completed_stage_count, 0);
}

#[test]
fn in_process_edge_transport_moves_bytes_between_mounted_device_slices() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let (gpu0, gpu1) = mounted_remote_middle_slices(&device);
    let forward_edge = gpu0.edge_io.outgoing_buffers[0].endpoint.edge_index;
    let return_edge = gpu1.edge_io.outgoing_buffers[0].endpoint.edge_index;
    assert_eq!(
        gpu1.edge_io.incoming_buffers[0].endpoint.edge_index,
        forward_edge
    );
    assert_eq!(
        gpu0.edge_io.incoming_buffers[0].endpoint.edge_index,
        return_edge
    );

    let forward_bytes = (0..FIXTURE_MODEL_FRAME_BYTES)
        .map(|index| u8::try_from(index).unwrap())
        .collect::<Vec<_>>();
    gpu0.edge_io
        .outgoing_buffer(forward_edge)
        .unwrap()
        .buffer
        .write_bytes(&forward_bytes)
        .unwrap();

    let mut transport = VulkanInProcessPlacedEdgeTransport::new();
    let published = transport.publish_outgoing_edge(&gpu0, forward_edge).unwrap();
    assert_eq!(
        published.key,
        VulkanPlacedEdgePacketKey {
            edge_index: forward_edge,
            from_device_id: "gpu0".to_string(),
            to_device_id: "gpu1".to_string(),
        }
    );
    assert_eq!(published.byte_count, FIXTURE_MODEL_FRAME_BYTES);
    assert_eq!(transport.packet_count(), 1);

    let received = transport.receive_available_incoming_edges(&gpu1).unwrap();
    assert_eq!(received.received.len(), 1);
    assert!(received.missing_packets.is_empty());
    assert_eq!(
        gpu1.edge_io
            .incoming_buffer(forward_edge)
            .unwrap()
            .buffer
            .read_bytes(FIXTURE_MODEL_FRAME_BYTES)
            .unwrap(),
        forward_bytes
    );
    assert_eq!(transport.packet_count(), 0);

    let return_bytes = (0..FIXTURE_MODEL_FRAME_BYTES)
        .map(|index| u8::try_from(FIXTURE_MODEL_FRAME_BYTES - index).unwrap())
        .collect::<Vec<_>>();
    gpu1.edge_io
        .outgoing_buffer(return_edge)
        .unwrap()
        .buffer
        .write_bytes(&return_bytes)
        .unwrap();
    let published_back = transport.publish_all_outgoing_edges(&gpu1).unwrap();
    assert_eq!(published_back.len(), 1);
    assert_eq!(
        published_back[0].key,
        VulkanPlacedEdgePacketKey {
            edge_index: return_edge,
            from_device_id: "gpu1".to_string(),
            to_device_id: "gpu0".to_string(),
        }
    );

    let received_back = transport.receive_available_incoming_edges(&gpu0).unwrap();
    assert_eq!(received_back.received.len(), 1);
    assert!(received_back.missing_packets.is_empty());
    assert_eq!(
        gpu0.edge_io
            .incoming_buffer(return_edge)
            .unwrap()
            .buffer
            .read_bytes(FIXTURE_MODEL_FRAME_BYTES)
            .unwrap(),
        return_bytes
    );
    assert_eq!(transport.packet_count(), 0);
}
