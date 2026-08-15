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

fn local_fanout_edge(edge_index: usize, destination_component_id: &str) -> VulkanPlacedLocalEdge {
    VulkanPlacedLocalEdge {
        buffer_index: edge_index,
        edge_id: format!("edge_{edge_index}_local"),
        edge_index,
        connection: StreamCircuitConnection::Forward,
        signal: "shared_context".to_string(),
        shape: vec![4_096],
        element_count: 4_096,
        byte_capacity: Some(8_192),
        device_id: "gpu0".to_string(),
        source_component_id: "input_adapter".to_string(),
        source_port_id: "shared_context".to_string(),
        source_component_port: Some("shared_context".to_string()),
        destination_component_id: destination_component_id.to_string(),
        destination_port_id: "shared_context".to_string(),
        destination_component_port: Some("shared_context".to_string()),
        transport: EdgeTransport::LocalBuffer {
            device_id: "gpu0".to_string(),
        },
    }
}

fn local_fanout_edge_plan() -> VulkanPlacedEdgeIoPlan {
    VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        signal_element_bytes: Some(2),
        local_edges: vec![
            local_fanout_edge(4, "draft_00"),
            local_fanout_edge(5, "draft_01"),
            local_fanout_edge(6, "draft_02"),
        ],
        endpoints: Vec::new(),
        local_edge_count: 3,
        incoming_endpoint_count: 0,
        outgoing_endpoint_count: 0,
        total_buffer_count: 3,
        total_endpoint_count: 0,
        total_byte_capacity: Some(3 * 8_192),
        unresolved_byte_edges: Vec::new(),
    }
}

fn outgoing_fanout_endpoint(
    endpoint_index: usize,
    edge_index: usize,
    destination_device_id: &str,
    destination_component_id: &str,
) -> VulkanPlacedEdgeEndpoint {
    VulkanPlacedEdgeEndpoint {
        endpoint_index,
        endpoint_id: format!("edge_{edge_index}_out"),
        direction: VulkanPlacedEdgeDirection::Outgoing,
        edge_index,
        connection: StreamCircuitConnection::Forward,
        signal: "shared_context".to_string(),
        shape: vec![4_096],
        element_count: 4_096,
        byte_capacity: Some(8_192),
        local_device_id: "gpu0".to_string(),
        remote_device_id: destination_device_id.to_string(),
        local_component_id: "input_adapter".to_string(),
        remote_component_id: destination_component_id.to_string(),
        local_port_id: "shared_context".to_string(),
        remote_port_id: "shared_context".to_string(),
        local_component_port: Some("shared_context".to_string()),
        remote_component_port: Some("shared_context".to_string()),
        transport: EdgeTransport::CrossDevice {
            from_device_id: "gpu0".to_string(),
            to_device_id: destination_device_id.to_string(),
        },
    }
}

fn incoming_fanout_endpoint(outgoing: &VulkanPlacedEdgeEndpoint) -> VulkanPlacedEdgeEndpoint {
    VulkanPlacedEdgeEndpoint {
        endpoint_index: outgoing.endpoint_index,
        endpoint_id: format!("edge_{}_in", outgoing.edge_index),
        direction: VulkanPlacedEdgeDirection::Incoming,
        edge_index: outgoing.edge_index,
        connection: outgoing.connection.clone(),
        signal: outgoing.signal.clone(),
        shape: outgoing.shape.clone(),
        element_count: outgoing.element_count,
        byte_capacity: outgoing.byte_capacity,
        local_device_id: outgoing.remote_device_id.clone(),
        remote_device_id: outgoing.local_device_id.clone(),
        local_component_id: outgoing.remote_component_id.clone(),
        remote_component_id: outgoing.local_component_id.clone(),
        local_port_id: outgoing.remote_port_id.clone(),
        remote_port_id: outgoing.local_port_id.clone(),
        local_component_port: outgoing.remote_component_port.clone(),
        remote_component_port: outgoing.local_component_port.clone(),
        transport: outgoing.transport.clone(),
    }
}

fn mixed_fanout_edge_plan() -> VulkanPlacedEdgeIoPlan {
    VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        signal_element_bytes: Some(2),
        local_edges: vec![local_fanout_edge(4, "draft_00")],
        endpoints: vec![
            outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01"),
            outgoing_fanout_endpoint(1, 6, "gpu2", "draft_02"),
        ],
        local_edge_count: 1,
        incoming_endpoint_count: 0,
        outgoing_endpoint_count: 2,
        total_buffer_count: 3,
        total_endpoint_count: 2,
        total_byte_capacity: Some(3 * 8_192),
        unresolved_byte_edges: Vec::new(),
    }
}

#[test]
fn boundary_output_classification_retains_every_local_and_remote_consumer() {
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let mut plan = fixture_model_placed_resident_plan_for_device(&runtime_model, "gpu0");
    let outgoing = plan.outgoing_edges[0].clone();
    let mut local = outgoing.clone();
    local.edge_index = 100;
    local.destination_component_id = "layer_00_tail".to_string();
    local.destination_device_id = "gpu0".to_string();
    local.transport = EdgeTransport::LocalBuffer {
        device_id: "gpu0".to_string(),
    };
    let mut second_outgoing = outgoing.clone();
    second_outgoing.edge_index = 101;
    second_outgoing.destination_component_id = "layer_00_second_remote".to_string();
    second_outgoing.destination_device_id = "gpu2".to_string();
    second_outgoing.transport = EdgeTransport::CrossDevice {
        from_device_id: "gpu0".to_string(),
        to_device_id: "gpu2".to_string(),
    };
    plan.local_edges.push(local);
    plan.outgoing_edges.push(second_outgoing);

    let target = classify_boundary_output(
        &outgoing.source_component_id,
        &outgoing.source_port_id,
        &plan,
    );
    let VulkanPlacedBoundDescriptorTarget::ProducedPort {
        local_edges,
        outgoing_edges,
    } = target
    else {
        panic!("boundary output must classify as one produced port");
    };
    assert_eq!(
        local_edges
            .iter()
            .map(|edge| edge.edge_index)
            .collect::<Vec<_>>(),
        vec![100]
    );
    assert_eq!(
        outgoing_edges
            .iter()
            .map(|edge| edge.edge_index)
            .collect::<Vec<_>>(),
        vec![outgoing.edge_index, 101]
    );
}

#[test]
fn placed_edge_pairs_group_every_remote_consumer_by_produced_port() {
    let first = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let second = outgoing_fanout_endpoint(1, 6, "gpu2", "draft_02");
    let groups = group_placed_edge_pairs_by_produced_port(vec![
        (first.clone(), incoming_fanout_endpoint(&first)),
        (second.clone(), incoming_fanout_endpoint(&second)),
    ])
    .unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].source_device_id, "gpu0");
    assert_eq!(groups[0].source_component_id, "input_adapter");
    assert_eq!(groups[0].source_port_id, "shared_context");
    assert_eq!(groups[0].byte_capacity, 8_192);
    assert_eq!(
        groups[0]
            .edges
            .iter()
            .map(|(outgoing, _)| outgoing.edge_index)
            .collect::<Vec<_>>(),
        vec![5, 6]
    );
}

#[test]
fn placed_graph_groups_local_only_produced_ports_for_distributed_consumers() {
    let plans = vec![local_fanout_edge_plan()];
    let groups = group_placed_graph_edges_by_produced_port(&plans).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].source_device_id, "gpu0");
    assert_eq!(groups[0].source_component_id, "input_adapter");
    assert_eq!(groups[0].source_port_id, "shared_context");
    assert_eq!(groups[0].byte_capacity, 8_192);
    assert!(groups[0].edges.is_empty());
}

#[test]
fn placed_graph_merges_local_and_remote_consumers_of_one_produced_port() {
    let source = mixed_fanout_edge_plan();
    let plans = vec![
        source.clone(),
        VulkanPlacedEdgeIoPlan {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: "gpu1".to_string(),
            signal_element_bytes: Some(2),
            local_edges: Vec::new(),
            endpoints: vec![incoming_fanout_endpoint(&source.endpoints[0])],
            local_edge_count: 0,
            incoming_endpoint_count: 1,
            outgoing_endpoint_count: 0,
            total_buffer_count: 1,
            total_endpoint_count: 1,
            total_byte_capacity: Some(8_192),
            unresolved_byte_edges: Vec::new(),
        },
        VulkanPlacedEdgeIoPlan {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: "gpu2".to_string(),
            signal_element_bytes: Some(2),
            local_edges: Vec::new(),
            endpoints: vec![incoming_fanout_endpoint(&source.endpoints[1])],
            local_edge_count: 0,
            incoming_endpoint_count: 1,
            outgoing_endpoint_count: 0,
            total_buffer_count: 1,
            total_endpoint_count: 1,
            total_byte_capacity: Some(8_192),
            unresolved_byte_edges: Vec::new(),
        },
    ];
    let groups = group_placed_graph_edges_by_produced_port(&plans).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].byte_capacity, 8_192);
    assert_eq!(groups[0].edges.len(), 2);
}

#[test]
fn selected_boundary_route_must_match_mounted_edge_identity_and_fanout() {
    let first = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let second = outgoing_fanout_endpoint(1, 6, "gpu2", "draft_02");
    let groups = group_placed_edge_pairs_by_produced_port(vec![
        (first.clone(), incoming_fanout_endpoint(&first)),
        (second.clone(), incoming_fanout_endpoint(&second)),
    ])
    .unwrap();
    let group = &groups[0];
    let selected = VulkanRuntimeMountedBoundaryRoute {
        edge_index: first.edge_index,
        source_device_id: "gpu0".to_string(),
        destination_device_id: "gpu1".to_string(),
        frame_byte_count: group.byte_capacity,
        route: VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal,
    };
    assert_eq!(
        required_vulkan_boundary_route_for_edge_group(
            group,
            &BTreeMap::from([(first.edge_index, selected.clone())]),
        )
        .unwrap(),
        Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
    );

    let mut stale = selected.clone();
    stale.frame_byte_count += 1;
    assert!(
        required_vulkan_boundary_route_for_edge_group(
            group,
            &BTreeMap::from([(first.edge_index, stale)]),
        )
        .unwrap_err()
        .0
        .contains("disagrees with its mounted endpoints or frame bytes")
    );

    let conflicting = VulkanRuntimeMountedBoundaryRoute {
        edge_index: second.edge_index,
        source_device_id: "gpu0".to_string(),
        destination_device_id: "gpu2".to_string(),
        frame_byte_count: group.byte_capacity,
        route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
    };
    assert!(
        required_vulkan_boundary_route_for_edge_group(
            group,
            &BTreeMap::from([
                (first.edge_index, selected),
                (second.edge_index, conflicting),
            ]),
        )
        .unwrap_err()
        .0
        .contains("incompatible physical boundary routes")
    );
}

#[test]
fn produced_port_resident_route_resolution_is_exact_and_physical() {
    let outgoing = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let groups = group_placed_edge_pairs_by_produced_port(vec![(
        outgoing.clone(),
        incoming_fanout_endpoint(&outgoing),
    )])
    .unwrap();
    let group = &groups[0];

    assert_eq!(
        resolve_vulkan_produced_port_resident_route(
            group,
            Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
            None,
            2,
        )
        .unwrap(),
        Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
    );
    assert_eq!(
        resolve_vulkan_produced_port_resident_route(
            group,
            None,
            Some(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
            2,
        )
        .unwrap(),
        Some(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
    );
    assert_eq!(
        resolve_vulkan_produced_port_resident_route(group, None, None, 1).unwrap(),
        None,
    );

    let conflict = resolve_vulkan_produced_port_resident_route(
        group,
        Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        Some(VulkanPlacedEdgeTransferRoute::DeviceLocalStaging),
        2,
    )
    .unwrap_err();
    assert!(conflict.0.contains("incompatible selected and distributed"));

    let colocated_selected = resolve_vulkan_produced_port_resident_route(
        group,
        Some(VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal),
        None,
        1,
    )
    .unwrap_err();
    assert!(colocated_selected.0.contains("has no physical peer"));

    let missing = resolve_vulkan_produced_port_resident_route(group, None, None, 2).unwrap_err();
    assert!(missing.0.contains("without an exact mounted route"));

    let empty = resolve_vulkan_produced_port_resident_route(group, None, None, 0).unwrap_err();
    assert!(empty.0.contains("has no physical participant"));
}

#[test]
fn workload_free_graph_edge_routes_are_explicit_only_across_physical_devices() {
    let first = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let second = outgoing_fanout_endpoint(1, 6, "gpu2", "draft_02");
    let plans = [VulkanPlacedEdgeIoPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        signal_element_bytes: Some(2),
        local_edges: Vec::new(),
        endpoints: vec![
            first.clone(),
            second.clone(),
            incoming_fanout_endpoint(&first),
            incoming_fanout_endpoint(&second),
        ],
        local_edge_count: 0,
        incoming_endpoint_count: 2,
        outgoing_endpoint_count: 2,
        total_buffer_count: 4,
        total_endpoint_count: 4,
        total_byte_capacity: Some(4 * 8_192),
        unresolved_byte_edges: Vec::new(),
    }];
    let routes = plan_vulkan_workload_free_graph_edge_routes(
        &plans,
        &BTreeMap::from([
            ("gpu0".to_string(), "physical_0".to_string()),
            ("gpu1".to_string(), "physical_1".to_string()),
            ("gpu2".to_string(), "physical_0".to_string()),
        ]),
    )
    .unwrap();

    assert_eq!(routes.len(), 1);
    assert_eq!(
        routes[&first.edge_index],
        VulkanRuntimeMountedBoundaryRoute {
            edge_index: first.edge_index,
            source_device_id: "gpu0".to_string(),
            destination_device_id: "gpu1".to_string(),
            frame_byte_count: 8_192,
            route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
        }
    );
    assert!(!routes.contains_key(&second.edge_index));

    let error = plan_vulkan_workload_free_graph_edge_routes(
        &plans,
        &BTreeMap::from([("gpu0".to_string(), "physical_0".to_string())]),
    )
    .unwrap_err();
    assert!(error.0.contains("destination logical device \"gpu1\""));
}

#[test]
fn exact_boundary_routes_override_without_erasing_serialized_routes() {
    let route = |edge_index, source: &str, destination: &str, route| {
        VulkanRuntimeMountedBoundaryRoute {
            edge_index,
            source_device_id: source.to_string(),
            destination_device_id: destination.to_string(),
            frame_byte_count: 4096,
            route,
        }
    };
    let workload_free = BTreeMap::from([
        (
            4,
            route(
                4,
                "gpu0",
                "gpu1",
                VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
            ),
        ),
        (
            8,
            route(
                8,
                "gpu1",
                "gpu2",
                VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
            ),
        ),
    ]);
    let selected = route(
        4,
        "gpu0",
        "gpu1",
        VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal,
    );

    let composed = compose_vulkan_runtime_mounted_boundary_routes(
        workload_free,
        BTreeMap::from([(4, selected.clone())]),
    );

    assert_eq!(composed.len(), 2);
    assert_eq!(composed[&4], selected);
    assert_eq!(
        composed[&8].route,
        VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
    );
}

#[test]
fn exact_shared_host_mount_rejects_missing_extra_and_aliased_participants() {
    let Some((owner, helper)) = selected_test_vulkan_device_pair() else {
        eprintln!("skipping exact shared-host participant test without two explicit Vulkan devices");
        return;
    };
    let devices = BTreeMap::from([
        ("owner".to_string(), owner.clone()),
        ("owner_alias".to_string(), owner),
        ("helper".to_string(), helper),
    ]);
    let device_for = |device_id: &str| {
        devices.get(device_id).map(Rc::as_ref).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: device_id.to_string(),
            }
        })
    };
    let allocation = VulkanRuntimeSharedHostResidentAllocation {
        kind: VulkanRuntimeSharedHostResidentAllocationKind::StreamControl,
        owner_device_id: "owner".to_string(),
        participant_device_ids: vec!["owner".to_string(), "helper".to_string()],
        byte_capacity: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    };
    let actual = BTreeSet::from(["owner".to_string(), "helper".to_string()]);
    let mounted = exact_physical_allocation_devices(
        &allocation.owner_device_id,
        &allocation.participant_device_ids,
        allocation.byte_capacity,
        &actual,
        "fixture",
        &device_for,
    )
    .unwrap();
    assert_eq!(mounted.len(), 2);
    assert!(mounted[0]
        .0
        .shares_logical_device_with(device_for("owner").unwrap()));

    let missing = BTreeSet::from(["owner".to_string()]);
    assert!(
        exact_physical_allocation_devices(
            &allocation.owner_device_id,
            &allocation.participant_device_ids,
            allocation.byte_capacity,
            &missing,
            "fixture",
            &device_for,
        )
            .err()
            .unwrap()
            .to_string()
            .contains("omitted a planned")
    );

    let owner_only = VulkanRuntimeSharedHostResidentAllocation {
        participant_device_ids: vec!["owner".to_string()],
        ..allocation.clone()
    };
    let mounted_owner_only = exact_physical_allocation_devices(
        &owner_only.owner_device_id,
        &owner_only.participant_device_ids,
        owner_only.byte_capacity,
        &missing,
        "fixture",
        &device_for,
    )
    .unwrap();
    assert_eq!(mounted_owner_only.len(), 1);
    assert!(
        exact_physical_allocation_devices(
            &owner_only.owner_device_id,
            &owner_only.participant_device_ids,
            owner_only.byte_capacity,
            &actual,
            "fixture",
            &device_for,
        )
            .err()
            .unwrap()
            .to_string()
            .contains("introduced unplanned")
    );

    let aliased = VulkanRuntimeSharedHostResidentAllocation {
        participant_device_ids: vec!["owner".to_string(), "owner_alias".to_string()],
        ..allocation
    };
    assert!(
        exact_physical_allocation_devices(
            &aliased.owner_device_id,
            &aliased.participant_device_ids,
            aliased.byte_capacity,
            &missing,
            "fixture",
            &device_for,
        )
            .err()
            .unwrap()
            .to_string()
            .contains("repeats one physical")
    );
}

#[test]
fn placed_edge_allocation_aliases_mixed_local_and_remote_fanout() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let buffers = mixed_fanout_edge_plan().allocate_buffers(&device).unwrap();

    assert_eq!(buffers.local_buffers.len(), 1);
    assert_eq!(buffers.outgoing_buffers.len(), 2);
    assert!(Arc::ptr_eq(
        &buffers.local_buffers[0].buffer,
        &buffers.outgoing_buffers[0].buffer
    ));
    assert!(Arc::ptr_eq(
        &buffers.outgoing_buffers[0].buffer,
        &buffers.outgoing_buffers[1].buffer
    ));
}

#[test]
fn placed_tick_plan_publishes_every_remote_consumer_of_one_produced_port() {
    let local = local_fanout_edge(4, "draft_00");
    let first = outgoing_fanout_endpoint(0, 5, "gpu1", "draft_01");
    let second = outgoing_fanout_endpoint(1, 6, "gpu2", "draft_02");
    let mounted = VulkanMountedPlacedBoundDispatchPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        dispatches: vec![VulkanMountedPlacedBoundDispatch {
            dispatch_index: 0,
            kernel_id: "producer".to_string(),
            component_id: "input_adapter".to_string(),
            circuit_id: "input_adapter".to_string(),
            node_index: 0,
            node_id: "producer".to_string(),
            op: "identity".to_string(),
            reusable_family_id: "identity".to_string(),
            artifact_path: "shaders/identity.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            descriptors: vec![VulkanMountedPlacedBoundDescriptor {
                binding: 0,
                usage: VulkanKernelDescriptorUsage::OutputSignal,
                name: "shared_context".to_string(),
                target: VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer {
                    port: VulkanPlacedProducedPortBufferBinding {
                        local_edges: vec![VulkanPlacedLocalEdgeBufferBinding {
                            buffer_index: 0,
                            edge: local,
                            byte_capacity: 8_192,
                        }],
                        outgoing_endpoints: vec![
                            VulkanPlacedEdgeEndpointBufferBinding {
                                buffer_index: 0,
                                endpoint: first,
                                byte_capacity: 8_192,
                            },
                            VulkanPlacedEdgeEndpointBufferBinding {
                                buffer_index: 1,
                                endpoint: second,
                                byte_capacity: 8_192,
                            },
                        ],
                        byte_capacity: 8_192,
                    },
                },
            }],
            push_constants: Vec::new(),
            stream_control_binding: None,
        }],
        total_descriptor_count: 1,
        resident_descriptor_count: 0,
        model_boundary_descriptor_count: 0,
        local_edge_descriptor_count: 1,
        edge_endpoint_descriptor_count: 1,
        incoming_edge_descriptor_count: 0,
        outgoing_edge_descriptor_count: 1,
    };

    let tick = VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted);

    assert_eq!(tick.stage_count, 3);
    assert_eq!(tick.dispatch_stage_count, 1);
    assert_eq!(tick.publish_stage_count, 2);
    assert_eq!(tick.local_edge_write_count, 1);
    assert_eq!(tick.outgoing_edge_write_count, 2);
    assert!(matches!(
        &tick.stages[1],
        VulkanMountedPlacedStreamTickStage::PublishEdge { edge_index: 5, .. }
    ));
    assert!(matches!(
        &tick.stages[2],
        VulkanMountedPlacedStreamTickStage::PublishEdge { edge_index: 6, .. }
    ));
}

#[test]
fn mounted_three_device_fanout_uses_one_physical_source_and_publishes_every_edge() {
    let Some((owner, first_peer, second_peer)) = selected_test_vulkan_device_triple() else {
        eprintln!(
            "skipping three-device produced-port test without three explicit Vulkan devices"
        );
        return;
    };
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([
        ("gpu0".to_string(), owner.clone()),
        ("gpu1".to_string(), first_peer.clone()),
        ("gpu2".to_string(), second_peer.clone()),
    ]);
    let mut gpu0 = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &owner,
        manifest_dir,
        runtime_model.clone(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let gpu1 = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &first_peer,
        manifest_dir,
        runtime_model.clone(),
        "gpu1",
        Some(4),
    )
    .unwrap();
    let mut gpu2 = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &second_peer,
        manifest_dir,
        fixture_model_runtime_model_with_three_layer_series("gpu2"),
        "gpu2",
        Some(4),
    )
    .unwrap();
    let first_outgoing = gpu0.placed_plan.placed_resident_plan.outgoing_edges[0].clone();
    let mut local = first_outgoing.clone();
    local.edge_index = 100;
    local.destination_component_id = "fanout_local".to_string();
    local.destination_device_id = "gpu0".to_string();
    local.transport = EdgeTransport::LocalBuffer {
        device_id: "gpu0".to_string(),
    };
    let mut second_outgoing = first_outgoing.clone();
    second_outgoing.edge_index = 101;
    second_outgoing.destination_component_id = "layer_00_remote".to_string();
    second_outgoing.destination_device_id = "gpu2".to_string();
    second_outgoing.transport = EdgeTransport::CrossDevice {
        from_device_id: "gpu0".to_string(),
        to_device_id: "gpu2".to_string(),
    };
    gpu0.placed_plan
        .placed_resident_plan
        .local_edges
        .push(local);
    gpu0.placed_plan
        .placed_resident_plan
        .outgoing_edges
        .push(second_outgoing.clone());
    gpu2.placed_plan.placed_resident_plan.incoming_edges.clear();
    gpu2.placed_plan.placed_resident_plan.outgoing_edges.clear();
    gpu2.placed_plan.placed_resident_plan.local_edges.clear();
    gpu2.placed_plan
        .placed_resident_plan
        .incoming_edges
        .push(second_outgoing);
    let slices = vec![Arc::new(gpu0), Arc::new(gpu1), Arc::new(gpu2)];
    let edge_plans = slices
        .iter()
        .map(|slice| {
            VulkanPlacedEdgeIoPlan::from_placed_resident_plan(
                &slice.placed_plan.placed_resident_plan,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let fanout_groups = group_placed_edge_pairs_by_produced_port(
        pair_placed_edge_endpoints(&edge_plans).unwrap(),
    )
    .unwrap();
    assert!(fanout_groups.iter().any(|group| {
        group
            .edges
            .iter()
            .any(|(outgoing, _)| outgoing.edge_index == first_outgoing.edge_index)
    }));
    let mut selected_boundary_routes = BTreeMap::new();
    let mut edge_staging_allocations = Vec::new();
    let mut staged_edge_host_bytes = 0usize;
    for group in &fanout_groups {
        let (selected_outgoing, selected_incoming) = group
            .edges
            .first()
            .expect("produced-port edge group is nonempty");
        selected_boundary_routes.insert(
            selected_outgoing.edge_index,
            VulkanRuntimeMountedBoundaryRoute {
                edge_index: selected_outgoing.edge_index,
                source_device_id: selected_outgoing.local_device_id.clone(),
                destination_device_id: selected_incoming.local_device_id.clone(),
                frame_byte_count: group.byte_capacity,
                route: VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
            },
        );
        let mut edge_indices = edge_plans
            .iter()
            .find(|plan| plan.device_id == group.source_device_id)
            .into_iter()
            .flat_map(|plan| &plan.local_edges)
            .filter(|edge| {
                edge.source_component_id == group.source_component_id
                    && edge.source_port_id == group.source_port_id
            })
            .map(|edge| edge.edge_index)
            .chain(
                group
                    .edges
                    .iter()
                    .map(|(outgoing, _)| outgoing.edge_index),
            )
            .collect::<Vec<_>>();
        edge_indices.sort_unstable();
        let participant_device_ids = group
            .edges
            .iter()
            .flat_map(|(outgoing, incoming)| {
                [
                    outgoing.local_device_id.clone(),
                    incoming.local_device_id.clone(),
                ]
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        edge_staging_allocations.push(VulkanRuntimeSharedHostResidentAllocation {
            kind: VulkanRuntimeSharedHostResidentAllocationKind::EdgeStaging {
                component_id: group.source_component_id.clone(),
                port_id: group.source_port_id.clone(),
                edge_indices,
            },
            owner_device_id: group.source_device_id.clone(),
            participant_device_ids,
            byte_capacity: group.byte_capacity,
        });
        staged_edge_host_bytes += group.byte_capacity;
    }
    let empty_plan = VulkanDistributedActivationBufferPlan {
        allocations: Vec::new(),
        reduction_allocations: Vec::new(),
        private_intermediate_allocations: Vec::new(),
        allocation_count: 0,
        import_count: 0,
        reference_count: 0,
        total_shared_byte_capacity: 0,
        total_private_byte_capacity: 0,
        route: VulkanSharedResidentBufferRoute::SharedHost,
    };
    let mut distributed = VulkanDistributedActivationBuffers::allocate(&empty_plan, |device_id| {
        devices
            .get(device_id)
            .map(Rc::as_ref)
            .ok_or_else(|| format!("missing fixture device {device_id}"))
    })
    .unwrap();
    let mut stream_control_participants = ["gpu0", "gpu1", "gpu2"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    stream_control_participants.sort_by_key(|device_id| {
        devices
            .get(device_id)
            .expect("fixture device exists")
            .physical_device_id()
            .to_string()
    });
    let stream_control_owner = stream_control_participants[0].clone();
    edge_staging_allocations.push(VulkanRuntimeSharedHostResidentAllocation {
        kind: VulkanRuntimeSharedHostResidentAllocationKind::StreamControl,
        owner_device_id: stream_control_owner,
        participant_device_ids: stream_control_participants,
        byte_capacity: VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
    });
    let physical_execution_residency_plan = VulkanRuntimePhysicalExecutionResidencyPlan {
        schema: VULKAN_RUNTIME_PHYSICAL_EXECUTION_RESIDENCY_PLAN_SCHEMA.to_string(),
        package_id: "placed-mount-fixture".to_string(),
        device_plans: ["gpu0", "gpu1", "gpu2"]
            .into_iter()
            .map(|device_id| VulkanRuntimePhysicalExecutionDeviceResidencyPlan {
                device_id: device_id.to_string(),
                breakdown: VulkanRuntimePhysicalExecutionResidencyBreakdown::default(),
                mount_device_local_bytes: 0,
                stream_device_local_bytes: 0,
                stream_shared_host_bytes: 0,
                resident_stream_device_allocations: Vec::new(),
                external_device_local_resident_allocations: Vec::new(),
                private_activation_resident_allocations: Vec::new(),
                execution_transient_device_allocations: Vec::new(),
            })
            .collect(),
        total_mount_device_local_bytes: 0,
        total_stream_device_local_bytes: 0,
        total_stream_shared_host_bytes: staged_edge_host_bytes
            + VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
        execution_transient_shared_host_bytes_per_stream: 0,
        execution_transient_shared_host_allocations: Vec::new(),
        execution_transient_host_visible_allocations: Vec::new(),
        resident_shared_host_allocations: edge_staging_allocations,
        graph_edge_memory_domains_bound: true,
        feedback_control_memory_domain_bound: true,
        host_visible_runtime_buffer_memory_domains_bound: true,
        stream_control_memory_domain_bound: true,
    };
    let links = create_placed_device_links(
        &slices,
        &mut distributed,
        &selected_boundary_routes,
        &physical_execution_residency_plan,
        &|device_id| {
            devices.get(device_id).map(Rc::as_ref).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: device_id.to_string(),
                }
            })
        },
    )
    .unwrap();
    let local = &links.local_edge_overrides["gpu0"];
    let outgoing = links.endpoint_overrides["gpu0"]
        .iter()
        .filter(|override_| override_.direction == VulkanPlacedEdgeDirection::Outgoing)
        .collect::<Vec<_>>();
    assert_eq!(local.len(), 1);
    assert_eq!(outgoing.len(), 2);
    assert!(outgoing.iter().all(|override_| Arc::ptr_eq(
        &local[0].buffer,
        &override_.buffer,
    )));
    assert_eq!(links.synchronizations.edges.len(), 3);
}

#[test]
fn placed_edge_allocation_aliases_every_local_fanout_consumer() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let buffers = local_fanout_edge_plan().allocate_buffers(&device).unwrap();

    assert_eq!(buffers.local_buffers.len(), 3);
    assert!(Arc::ptr_eq(
        &buffers.local_buffers[0].buffer,
        &buffers.local_buffers[1].buffer
    ));
    assert!(Arc::ptr_eq(
        &buffers.local_buffers[0].buffer,
        &buffers.local_buffers[2].buffer
    ));
}

#[test]
fn placed_edge_allocation_rejects_distinct_overrides_for_one_local_fanout() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let first = Arc::new(device.create_resident_buffer(8_192).unwrap());
    let second = Arc::new(device.create_resident_buffer(8_192).unwrap());
    let error = match local_fanout_edge_plan().allocate_buffers_with_overrides(
            &device,
            &[
                VulkanPlacedLocalEdgeBufferOverride {
                    edge_index: 4,
                    buffer: first,
                },
                VulkanPlacedLocalEdgeBufferOverride {
                    edge_index: 5,
                    buffer: second,
                },
            ],
            &[],
        ) {
        Ok(_) => panic!("distinct local fan-out overrides must be rejected"),
        Err(error) => error,
    };

    assert!(error
        .to_string()
        .contains("incompatible physical buffer overrides"));
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
fn mounted_execution_graph_is_one_sequence_and_matches_component_execution() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let manifest_dir = fixture_model_package_manifest_path()
        .parent()
        .unwrap()
        .to_path_buf();
    let run = |composed: bool| {
        let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
            &device,
            &manifest_dir,
            fixture_model_runtime_model_with_colocated_three_layer_series(),
            "gpu0",
            Some(4),
        )
        .unwrap();
        let mounted = slice.create_mounted_stream_circuit(&device).unwrap();
        mounted.buffers.initialize_state_buffers(&device).unwrap();
        let input = mounted.boundary_io.input_buffer("input_frame").unwrap();
        input
            .buffer
            .write_bytes(&vec![0; FIXTURE_MODEL_FRAME_BYTES])
            .unwrap();
        let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
        let bound = mounted
            .mounted_placed_bound_dispatch_plan(&reusable_manifest)
            .unwrap();
        let component_ids = mounted
            .placed_plan
            .binding_plan
            .circuits
            .iter()
            .map(|circuit| circuit.component_id.clone())
            .collect::<Vec<_>>();
        let control = VulkanMountedPlacedStreamControl {
            stream_tick: 0,
            control_flags: 0,
            dynamic_state_capacity_activations: 4,
        };
        reset_vulkan_resident_execution_counters();
        if composed {
            mounted
                .create_resident_execution_graph_runner(
                    &device,
                    &bound,
                    &component_ids,
                    slice.loaded_manifest(),
                )
                .unwrap()
                .run_with_stream_control(&device, control)
                .unwrap();
        } else {
            for component_id in &component_ids {
                mounted
                    .create_resident_component_runner(
                        &device,
                        &bound,
                        component_id,
                        slice.loaded_manifest(),
                    )
                    .unwrap()
                    .run_with_stream_control(&device, control)
                    .unwrap();
            }
        }
        let output = mounted
            .boundary_io
            .output_buffer("output_frame")
            .unwrap()
            .buffer
            .read_bytes(FIXTURE_MODEL_FRAME_BYTES)
            .unwrap();
        let state = mounted
            .buffers
            .state_buffers
            .iter()
            .flat_map(|allocation| {
                allocation
                    .buffer
                    .read_bytes(allocation.byte_capacity)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        (output, state, vulkan_resident_execution_counters())
    };

    let (component_output, component_state, component_counters) = run(false);
    let (graph_output, graph_state, graph_counters) = run(true);

    assert_eq!(graph_output, component_output);
    assert_eq!(graph_state, component_state);
    assert_eq!(graph_counters.resident_sequence_queue_submits, 1);
    assert!(component_counters.resident_sequence_queue_submits > 1);
}

#[test]
fn state_initialization_fills_only_the_prefix_and_uploads_the_page_table() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let manifest_dir = fixture_model_package_manifest_path()
        .parent()
        .unwrap()
        .to_path_buf();
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        &manifest_dir,
        fixture_model_runtime_model_with_colocated_three_layer_series(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let mounted = slice.create_mounted_stream_circuit(&device).unwrap();
    let expected_initialized = mounted
        .buffers
        .state_buffers
        .iter()
        .map(|state| state.layout.dynamic_data_offset)
        .sum::<usize>();
    for state in &mounted.buffers.state_buffers {
        state
            .buffer
            .write_bytes(&vec![0xabu8; state.byte_capacity])
            .unwrap();
    }

    assert_eq!(
        mounted.buffers.initialize_state_buffers(&device).unwrap(),
        expected_initialized
    );
    for state in &mounted.buffers.state_buffers {
        let bytes = state.buffer.read_bytes(state.byte_capacity).unwrap();
        let page_table = state.layout.initial_page_table_bytes().unwrap();
        assert_eq!(&bytes[..page_table.len()], page_table);
        assert!(bytes[page_table.len()..state.layout.dynamic_data_offset]
            .iter()
            .all(|byte| *byte == 0));
        assert!(bytes[state.layout.dynamic_data_offset..]
            .iter()
            .all(|byte| *byte == 0xab));
    }
}

#[test]
fn distributed_binding_replaces_parameters_only_where_they_exist() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let manifest_dir = fixture_model_package_manifest_path()
        .parent()
        .unwrap()
        .to_path_buf();
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        &device,
        &manifest_dir,
        fixture_model_runtime_model_with_colocated_three_layer_series(),
        "gpu0",
        Some(4),
    )
    .unwrap();
    let mounted = slice.create_mounted_stream_circuit(&device).unwrap();
    let manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let prepared = mounted.prepared_dispatch_plan(&manifest).unwrap();
    let parameterized = prepared
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.descriptors.iter().any(|descriptor| {
                matches!(
                    descriptor.resource,
                    VulkanDescriptorResourceAddress::PermanentParameter { .. }
                )
            })
        })
        .expect("fixture must contain a parameterized dispatch");
    let parameterless = prepared
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.descriptors.iter().all(|descriptor| {
                !matches!(
                    descriptor.resource,
                    VulkanDescriptorResourceAddress::PermanentParameter { .. }
                )
            })
        })
        .expect("fixture must contain a parameterless dispatch");
    let distributed = BTreeSet::from([
        parameterized.dispatch_index,
        parameterless.dispatch_index,
    ]);

    let bound = mounted
        .mounted_placed_bound_dispatch_plan_with_distributed_dispatches(&manifest, &distributed)
        .unwrap();
    let bound_parameterized = bound
        .dispatches
        .iter()
        .find(|dispatch| dispatch.dispatch_index == parameterized.dispatch_index)
        .unwrap();
    let bound_parameterless = bound
        .dispatches
        .iter()
        .find(|dispatch| dispatch.dispatch_index == parameterless.dispatch_index)
        .unwrap();
    assert!(bound_parameterized.descriptors.len() < parameterized.descriptors.len());
    assert_eq!(bound_parameterless.descriptors.len(), parameterless.descriptors.len());

    let missing = BTreeSet::from([usize::MAX]);
    let error = mounted
        .mounted_placed_bound_dispatch_plan_with_distributed_dispatches(&manifest, &missing)
        .unwrap_err();
    assert_eq!(
        error,
        VulkanBoundDispatchPlanError::MissingDistributedDispatch {
            dispatch_index: usize::MAX,
        }
    );
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
