pub(super) fn selected_test_vulkan_device() -> Result<VulkanComputeDevice, VulkanError> {
    if let Some(raw_uuid) = std::env::var_os("NERVE_TEST_VULKAN_DEVICE_UUID") {
        let raw_uuid = raw_uuid.to_string_lossy();
        let uuid = parse_test_vulkan_device_uuid(
            "NERVE_TEST_VULKAN_DEVICE_UUID",
            raw_uuid.as_ref(),
        );
        let physical_device_id = format!("vulkan-uuid:{raw_uuid}");
        let catalog = VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(
            &BTreeSet::from([physical_device_id]),
        )?;
        return catalog.open_device_uuid(uuid).map_err(|error| {
            VulkanError(format!(
                "explicit Vulkan test device UUID {raw_uuid:?} could not be opened: {error}"
            ))
        });
    }
    match std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX") {
        Ok(raw_index) => {
            let index = raw_index.parse::<usize>().unwrap_or_else(|error| {
                panic!("invalid NERVE_TEST_VULKAN_DEVICE_INDEX {raw_index:?}: {error}")
            });
            Ok(
                VulkanComputeDevice::new_for_physical_device_index(index).unwrap_or_else(|error| {
                    panic!(
                        "explicit Vulkan test device index {index} could not be opened: {error}"
                    )
                }),
            )
        }
        Err(std::env::VarError::NotPresent) => Err(VulkanError(
            "NERVE_TEST_VULKAN_DEVICE_UUID or NERVE_TEST_VULKAN_DEVICE_INDEX is required for every Vulkan test"
                .to_string(),
        )),
        Err(error) => Err(VulkanError(format!(
            "could not read NERVE_TEST_VULKAN_DEVICE_INDEX: {error}"
        ))),
    }
}

fn selected_test_vulkan_device_pair() -> Option<(Rc<VulkanComputeDevice>, Rc<VulkanComputeDevice>)>
{
    if let (Ok(raw_owner_uuid), Ok(raw_peer_uuid)) = (
        std::env::var("NERVE_TEST_VULKAN_DEVICE_UUID"),
        std::env::var("NERVE_TEST_VULKAN_PEER_DEVICE_UUID"),
    ) {
        let owner_uuid = parse_test_vulkan_device_uuid(
            "NERVE_TEST_VULKAN_DEVICE_UUID",
            &raw_owner_uuid,
        );
        let peer_uuid = parse_test_vulkan_device_uuid(
            "NERVE_TEST_VULKAN_PEER_DEVICE_UUID",
            &raw_peer_uuid,
        );
        assert_ne!(owner_uuid, peer_uuid);
        let catalog = VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(
            &BTreeSet::from([
                format!("vulkan-uuid:{raw_owner_uuid}"),
                format!("vulkan-uuid:{raw_peer_uuid}"),
            ]),
        )
        .expect("explicit Vulkan test owner and peer devices must be discoverable");
        return Some((
            Rc::new(
                catalog
                    .open_device_uuid(owner_uuid)
                    .expect("explicit Vulkan owner UUID must open"),
            ),
            Rc::new(
                catalog
                    .open_device_uuid(peer_uuid)
                    .expect("explicit Vulkan peer UUID must open"),
            ),
        ));
    }
    let (Ok(raw_owner_index), Ok(raw_peer_index)) = (
        std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX"),
        std::env::var("NERVE_TEST_VULKAN_PEER_DEVICE_INDEX"),
    ) else {
        return None;
    };
    let owner_index = raw_owner_index
        .parse::<usize>()
        .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
    let peer_index = raw_peer_index
        .parse::<usize>()
        .expect("NERVE_TEST_VULKAN_PEER_DEVICE_INDEX must be an integer");
    assert_ne!(owner_index, peer_index);
    let owner = Rc::new(
        VulkanComputeDevice::new_for_physical_device_index(owner_index)
            .expect("explicit Vulkan owner device must open"),
    );
    let peer = Rc::new(
        VulkanComputeDevice::new_for_physical_device_index(peer_index)
            .expect("explicit Vulkan peer device must open"),
    );
    Some((owner, peer))
}

fn parse_test_vulkan_device_uuid(variable: &str, raw_uuid: &str) -> [u8; 16] {
    if raw_uuid.len() != 32 {
        panic!(
            "invalid {variable} {raw_uuid:?}; expected 32 hexadecimal digits"
        );
    }
    let mut uuid = [0u8; 16];
    for (index, byte) in uuid.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&raw_uuid[offset..offset + 2], 16)
            .unwrap_or_else(|error| panic!("invalid {variable} {raw_uuid:?}: {error}"));
    }
    uuid
}

fn selected_test_vulkan_device_triple() -> Option<(
    Rc<VulkanComputeDevice>,
    Rc<VulkanComputeDevice>,
    Rc<VulkanComputeDevice>,
)> {
    let (Ok(raw_owner), Ok(raw_first_peer), Ok(raw_second_peer)) = (
        std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX"),
        std::env::var("NERVE_TEST_VULKAN_PEER_DEVICE_INDEX"),
        std::env::var("NERVE_TEST_VULKAN_THIRD_DEVICE_INDEX"),
    ) else {
        return None;
    };
    let indices = [raw_owner, raw_first_peer, raw_second_peer].map(|raw| {
        raw.parse::<usize>()
            .expect("explicit Vulkan test device index must be an integer")
    });
    assert_eq!(indices.iter().copied().collect::<BTreeSet<_>>().len(), 3);
    let devices = indices.map(|index| {
        Rc::new(
            VulkanComputeDevice::new_for_physical_device_index(index)
                .expect("explicit Vulkan test device must open"),
        )
    });
    Some((
        devices[0].clone(),
        devices[1].clone(),
        devices[2].clone(),
    ))
}

#[test]
fn backend_loop_window_is_device_owned_and_snapshot_memory_bounded() {
    assert_eq!(
        backend_loop_window_for_static_state_bytes(0, 4_096, 64 * 1024 * 1024),
        64
    );
    assert_eq!(
        backend_loop_window_for_static_state_bytes(
            2 * 1024 * 1024,
            4_096,
            64 * 1024 * 1024
        ),
        32
    );
    assert_eq!(
        backend_loop_window_for_static_state_bytes(
            128 * 1024 * 1024,
            4_096,
            64 * 1024 * 1024
        ),
        1
    );
    assert_eq!(
        backend_loop_window_for_static_state_bytes(0, 8, 64 * 1024 * 1024),
        8
    );
}

#[test]
fn placed_feedback_window_accepts_bridged_multi_device_execution_graphs() {
    let eligible = VulkanResidentInProcessPlacedFeedbackLoopEligibility {
        device_slice_count: 3,
        every_slice_has_terminal_segment: true,
        distributed_dispatches_are_bridged: true,
        demand_dispatches_are_pipeline_guarded: true,
        demand_checkpoint_resume_is_unambiguous: true,
        every_edge_is_resident_replayable: true,
        feedback_stream_control_is_resident_replayable: true,
        speculative_state_is_resident_replayable: true,
        has_dynamic_push_constants: false,
        window_width: 64,
        sampler_history_capacity: 4_096,
    };
    assert_eq!(eligible.window_width(), Some(64));
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            window_width: 32,
            ..eligible
        }
        .window_width(),
        Some(32)
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            every_edge_is_resident_replayable: false,
            ..eligible
        }
        .disabled_reason(),
        Some("host_staged_edge")
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            feedback_stream_control_is_resident_replayable: false,
            ..eligible
        }
        .disabled_reason(),
        Some("host_staged_feedback_stream_control")
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            speculative_state_is_resident_replayable: false,
            ..eligible
        }
        .disabled_reason(),
        Some("unreplayable_speculative_state_sync")
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            sampler_history_capacity: 8,
            ..eligible
        }
        .window_width(),
        Some(8)
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            device_slice_count: 0,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            every_slice_has_terminal_segment: false,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            distributed_dispatches_are_bridged: false,
            ..eligible
        }
        .window_width(),
        None
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            demand_dispatches_are_pipeline_guarded: false,
            ..eligible
        }
        .disabled_reason(),
        Some("unguarded_demand_distributed_dispatch")
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            demand_checkpoint_resume_is_unambiguous: false,
            ..eligible
        }
        .disabled_reason(),
        Some("ambiguous_demand_checkpoint_resume")
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            has_dynamic_push_constants: true,
            ..eligible
        }
        .window_width(),
        Some(64)
    );
    assert_eq!(
        VulkanResidentInProcessPlacedFeedbackLoopEligibility {
            sampler_history_capacity: 1,
            window_width: 1,
            ..eligible
        }
        .window_width(),
        None
    );
}

#[test]
fn speculative_feedback_history_tracks_each_decoder_state_contract() {
    assert_eq!(
        resident_speculative_feedback_history_requirements([]),
        VulkanResidentSpeculativeFeedbackHistoryRequirements::default(),
    );
    assert_eq!(
        resident_speculative_feedback_history_requirements([true]),
        VulkanResidentSpeculativeFeedbackHistoryRequirements {
            parallel_state: true,
            normalized_frames: false,
        },
    );
    assert_eq!(
        resident_speculative_feedback_history_requirements([false]),
        VulkanResidentSpeculativeFeedbackHistoryRequirements {
            parallel_state: false,
            normalized_frames: true,
        },
    );
    assert_eq!(
        resident_speculative_feedback_history_requirements([true, false]),
        VulkanResidentSpeculativeFeedbackHistoryRequirements {
            parallel_state: true,
            normalized_frames: true,
        },
    );
}

fn fixture_tick_dispatch_stage(stage_index: usize) -> VulkanMountedPlacedStreamTickStage {
    VulkanMountedPlacedStreamTickStage::Dispatch {
        stage_index,
        dispatch: VulkanMountedPlacedStreamTickDispatch {
            dispatch_index: stage_index,
            kernel_id: format!("kernel_{stage_index}"),
            component_id: format!("component_{stage_index}"),
            node_id: format!("node_{stage_index}"),
            op: "fixture".to_string(),
            descriptor_count: 0,
            resident_descriptor_count: 0,
            reads: Vec::new(),
            writes: Vec::new(),
        },
    }
}

#[test]
fn mounted_tick_dispatch_preserves_resident_activation_dependencies() {
    let activation = |binding, usage, signal: &str, slot| VulkanMountedPlacedBoundDescriptor {
        binding,
        usage,
        name: signal.to_string(),
        target: VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target: VulkanBoundDescriptorTarget::ActivationSlot {
                buffer_index: slot,
                component_id: "component".to_string(),
                signal_id: signal.to_string(),
                circuit_id: "circuit".to_string(),
                slot,
                byte_capacity: 16,
                signal_byte_capacity: 16,
            },
        },
    };
    let dispatch = VulkanMountedPlacedStreamTickDispatch::from_bound_dispatch(
        &VulkanMountedPlacedBoundDispatch {
            dispatch_index: 3,
            kernel_id: "kernel".to_string(),
            component_id: "component".to_string(),
            circuit_id: "circuit".to_string(),
            node_index: 3,
            node_id: "consumer".to_string(),
            op: "fixture".to_string(),
            reusable_family_id: "family".to_string(),
            artifact_path: "kernel.spv".to_string(),
            entry_point: "main".to_string(),
            local_size_x: 64,
            descriptors: vec![
                activation(0, VulkanKernelDescriptorUsage::InputSignal, "input", 1),
                activation(1, VulkanKernelDescriptorUsage::OutputSignal, "output", 2),
                activation(2, VulkanKernelDescriptorUsage::Parameter, "scratch", 3),
            ],
            push_constants: Vec::new(),
            stream_control_binding: None,
        },
    );

    assert_eq!(
        dispatch.reads,
        [VulkanMountedPlacedStreamTickIo::ActivationSlot {
            component_id: "component".to_string(),
            signal_id: "input".to_string(),
            slot: 1,
        }]
    );
    assert_eq!(
        dispatch.writes,
        [VulkanMountedPlacedStreamTickIo::ActivationSlot {
            component_id: "component".to_string(),
            signal_id: "output".to_string(),
            slot: 2,
        }]
    );
    assert_eq!(dispatch.resident_descriptor_count, 3);
}

#[test]
fn resident_dispatch_segments_stop_at_transport_boundaries() {
    let stages = vec![
        fixture_tick_dispatch_stage(0),
        fixture_tick_dispatch_stage(1),
        VulkanMountedPlacedStreamTickStage::PublishEdge {
            stage_index: 2,
            edge_index: 0,
            endpoint_id: "out".to_string(),
            buffer_index: 0,
            byte_capacity: 16,
            remote_device_id: "gpu1".to_string(),
            remote_component_id: "remote".to_string(),
        },
        VulkanMountedPlacedStreamTickStage::ReceiveEdge {
            stage_index: 3,
            edge_index: 1,
            endpoint_id: "in".to_string(),
            buffer_index: 0,
            byte_capacity: 16,
            remote_device_id: "gpu1".to_string(),
            remote_component_id: "remote".to_string(),
        },
        fixture_tick_dispatch_stage(4),
        fixture_tick_dispatch_stage(5),
    ];

    assert_eq!(
        resident_dispatch_segment_stage_ranges(&stages),
        vec![(0, 2), (4, 6)]
    );
}

#[test]
fn distributed_dispatches_split_resident_command_segments() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();

    let ranges = resident_dispatch_segment_stage_ranges_excluding_dispatches(
        &stages,
        &BTreeSet::from([2, 4]),
    );

    assert_eq!(ranges, vec![(0, 2), (3, 4), (5, 6)]);
}

#[test]
fn distributed_dependency_topology_covers_edges_and_adjacent_dispatches() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let distributed_indices = BTreeSet::from([0, 2, 3, 5]);
    let distributed_stages = distributed_dispatch_stages(
        &VulkanMountedPlacedStreamTickPlan {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: "gpu0".to_string(),
            stages: stages.clone(),
            stage_count: stages.len(),
            receive_stage_count: 0,
            dispatch_stage_count: stages.len(),
            publish_stage_count: 0,
            local_edge_read_count: 0,
            local_edge_write_count: 0,
            incoming_edge_read_count: 0,
            outgoing_edge_write_count: 0,
            model_input_read_count: 0,
            model_output_write_count: 0,
            can_execute: false,
        },
        &distributed_indices,
    )
    .unwrap();
    let ranges =
        resident_dispatch_segment_stage_ranges_excluding_dispatches(&stages, &distributed_indices);
    let physical_execution_islands = physical_execution_island_stage_groups(
        &distributed_stages,
        &[vec![0], vec![2], vec![3], vec![5]],
    )
    .unwrap();

    assert_eq!(ranges, vec![(1, 2), (4, 5)]);
    assert_eq!(
        distributed_dispatch_dependency_topologies(&physical_execution_islands, &ranges, &stages),
        BTreeMap::from([
            (
                0,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 0,
                    has_owner_producer: false,
                    has_owner_continuation: true,
                    completion_consumer_stage_index: Some(1),
                },
            ),
            (
                2,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 2,
                    has_owner_producer: true,
                    has_owner_continuation: false,
                    completion_consumer_stage_index: None,
                },
            ),
            (
                3,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 3,
                    has_owner_producer: false,
                    has_owner_continuation: true,
                    completion_consumer_stage_index: Some(4),
                },
            ),
            (
                5,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: 5,
                    has_owner_producer: true,
                    has_owner_continuation: false,
                    completion_consumer_stage_index: None,
                },
            ),
        ])
    );
}

#[test]
fn distributed_dependency_topology_uses_composed_group_boundaries() {
    let stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let distributed_indices = BTreeSet::from([2, 3]);
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages: stages.clone(),
        stage_count: stages.len(),
        receive_stage_count: 0,
        dispatch_stage_count: stages.len(),
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let distributed_stages = distributed_dispatch_stages(&tick_plan, &distributed_indices).unwrap();
    let physical_execution_islands =
        physical_execution_island_stage_groups(&distributed_stages, &[vec![2, 3]]).unwrap();
    let ranges =
        resident_dispatch_segment_stage_ranges_excluding_dispatches(&stages, &distributed_indices);

    assert_eq!(ranges, vec![(0, 2), (4, 6)]);
    assert_eq!(
        distributed_dispatch_dependency_topologies(&physical_execution_islands, &ranges, &stages),
        BTreeMap::from([(
            2,
            VulkanMountedPlacedDistributedDispatchDependencies {
                dispatch_index: 2,
                has_owner_producer: true,
                has_owner_continuation: true,
                completion_consumer_stage_index: Some(4),
            },
        )])
    );
}

#[test]
fn sparse_expert_island_keeps_one_router_and_one_reduction_on_the_coordinator() {
    let activation = |signal: &str, slot| VulkanMountedPlacedStreamTickIo::ActivationSlot {
        component_id: "sparse-layer".to_string(),
        signal_id: signal.to_string(),
        slot,
    };
    let mut stages = (0..5).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let io = [
        (vec![activation("hidden", 0)], vec![activation("router_logits", 1)]),
        (vec![activation("router_logits", 1)], vec![activation("routes", 2)]),
        (
            vec![activation("hidden", 0), activation("routes", 2)],
            vec![activation("expert_intermediates", 3)],
        ),
        (
            vec![activation("expert_intermediates", 3), activation("routes", 2)],
            vec![activation("expert_outputs", 4)],
        ),
        (
            vec![activation("expert_outputs", 4)],
            vec![activation("ffn_output", 5)],
        ),
    ];
    for (stage, (reads, writes)) in stages.iter_mut().zip(io) {
        let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = stage else {
            unreachable!();
        };
        dispatch.reads = reads;
        dispatch.writes = writes;
    }
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stage_count: stages.len(),
        dispatch_stage_count: stages.len(),
        stages: stages.clone(),
        receive_stage_count: 0,
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let distributed_indices = BTreeSet::from([2, 3]);
    let distributed_stages = distributed_dispatch_stages(&tick_plan, &distributed_indices).unwrap();
    let islands =
        physical_execution_island_stage_groups(&distributed_stages, &[vec![2, 3]]).unwrap();
    let ranges = resident_dispatch_segment_stage_ranges_for_physical_islands(
        &stages,
        &distributed_indices,
        &islands,
    );
    let dependencies =
        distributed_dispatch_dependency_topologies(&islands, &ranges, &stages);

    assert_eq!(ranges, vec![(0, 2), (4, 5)]);
    assert_eq!(islands.len(), 1);
    assert_eq!(islands[&2].end_stage_index, 4);
    assert_eq!(
        islands[&2]
            .dispatches
            .iter()
            .map(|dispatch| dispatch.dispatch_index)
            .collect::<Vec<_>>(),
        [2, 3],
    );
    assert_eq!(
        dependencies[&2],
        VulkanMountedPlacedDistributedDispatchDependencies {
            dispatch_index: 2,
            has_owner_producer: true,
            has_owner_continuation: true,
            completion_consumer_stage_index: Some(4),
        }
    );
}

#[test]
fn distributed_completion_wait_moves_to_the_first_true_activation_consumer() {
    let activation = |signal: &str, slot| VulkanMountedPlacedStreamTickIo::ActivationSlot {
        component_id: "component".to_string(),
        signal_id: signal.to_string(),
        slot,
    };
    let mut stages = (0..6).map(fixture_tick_dispatch_stage).collect::<Vec<_>>();
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &mut stages[1] else {
        unreachable!();
    };
    dispatch.writes = vec![activation("routed_output", 7)];
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &mut stages[2] else {
        unreachable!();
    };
    dispatch.reads = vec![activation("normalized_hidden", 3)];
    dispatch.writes = vec![activation("shared_hidden", 4)];
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &mut stages[3] else {
        unreachable!();
    };
    dispatch.reads = vec![activation("shared_hidden", 4)];
    dispatch.writes = vec![activation("shared_output", 5)];
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &mut stages[4] else {
        unreachable!();
    };
    dispatch.reads = vec![
        activation("routed_output", 7),
        activation("shared_output", 5),
    ];
    let distributed_indices = BTreeSet::from([1]);
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages: stages.clone(),
        stage_count: stages.len(),
        receive_stage_count: 0,
        dispatch_stage_count: stages.len(),
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let distributed_stages = distributed_dispatch_stages(&tick_plan, &distributed_indices).unwrap();
    let islands =
        physical_execution_island_stage_groups(&distributed_stages, &[vec![1]]).unwrap();
    let ranges = resident_dispatch_segment_stage_ranges_for_physical_islands(
        &stages,
        &distributed_indices,
        &islands,
    );
    let dependencies =
        distributed_dispatch_dependency_topologies(&islands, &ranges, &stages);

    assert_eq!(ranges, vec![(0, 1), (2, 4), (4, 6)]);
    assert_eq!(
        dependencies[&1],
        VulkanMountedPlacedDistributedDispatchDependencies {
            dispatch_index: 1,
            has_owner_producer: true,
            has_owner_continuation: true,
            completion_consumer_stage_index: Some(4),
        }
    );
    let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan {
        tick_plan: Arc::new(tick_plan),
        dispatch_segment_count: ranges.len(),
        dispatch_count: 4,
        distributed_dispatch_count: 1,
        dispatch_segments: Vec::new(),
        distributed_dispatch_stages: distributed_stages,
        physical_execution_islands: islands,
        distributed_dispatch_dependencies: dependencies,
    };
    assert!(!execution_plan.segment_consumes_distributed_completion(2, 1));
    assert!(execution_plan.segment_consumes_distributed_completion(4, 1));
}

#[test]
fn cursor_completes_an_entire_matching_distributed_group() {
    let stages = vec![
        fixture_tick_dispatch_stage(0),
        fixture_tick_dispatch_stage(1),
    ];
    let VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. } = &stages[0] else {
        unreachable!();
    };
    let distributed_dispatch = dispatch.clone();
    let grouped_dispatch = dispatch.clone();
    let VulkanMountedPlacedStreamTickStage::Dispatch {
        dispatch: second_dispatch,
        ..
    } = &stages[1]
    else {
        unreachable!();
    };
    let second_distributed_dispatch = second_dispatch.clone();
    let second_grouped_dispatch = second_dispatch.clone();
    let tick_plan = VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: "gpu0".to_string(),
        stages,
        stage_count: 2,
        receive_stage_count: 0,
        dispatch_stage_count: 2,
        publish_stage_count: 0,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: 0,
        outgoing_edge_write_count: 0,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: false,
    };
    let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan {
        tick_plan: Arc::new(tick_plan),
        dispatch_segment_count: 0,
        dispatch_count: 0,
        distributed_dispatch_count: 2,
        dispatch_segments: Vec::new(),
        distributed_dispatch_stages: BTreeMap::from([
            (0, distributed_dispatch),
            (1, second_distributed_dispatch),
        ]),
        physical_execution_islands: BTreeMap::from([(
            0,
            VulkanMountedPhysicalExecutionIslandStage {
                dispatches: vec![grouped_dispatch, second_grouped_dispatch],
                end_stage_index: 2,
            },
        )]),
        distributed_dispatch_dependencies: BTreeMap::from([(
            0,
            VulkanMountedPlacedDistributedDispatchDependencies {
                dispatch_index: 0,
                has_owner_producer: false,
                has_owner_continuation: false,
                completion_consumer_stage_index: None,
            },
        )]),
    };
    let mut cursor = execution_plan.resident_stream_tick_cursor(7);

    assert_eq!(
        cursor
            .pending_distributed_dispatch(&execution_plan)
            .map(|dispatch| dispatch.dispatch_index),
        Some(0)
    );
    let error = cursor
        .complete_pending_distributed_dispatch(&execution_plan, 1)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("expected distributed dispatch 0")
    );
    assert!(!cursor.is_completed());

    cursor
        .complete_pending_distributed_dispatch(&execution_plan, 0)
        .unwrap();
    assert!(cursor.is_completed());
    assert_eq!(cursor.completed_stage_count, 2);
}

fn numeric_state_error(actual: &[u8], expected: &[u8], dtype: &str) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let (squared_error, squared_reference, max_absolute_error, count) = match dtype {
        "BF16" => actual.chunks_exact(2).zip(expected.chunks_exact(2)).fold(
            (0.0, 0.0, 0.0_f64, 0usize),
            |totals, (actual, expected)| {
                let actual = f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([actual[0], actual[1]])) << 16,
                ));
                let expected = f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([expected[0], expected[1]])) << 16,
                ));
                let error = (actual - expected).abs();
                (
                    totals.0 + error * error,
                    totals.1 + expected * expected,
                    totals.2.max(error),
                    totals.3 + 1,
                )
            },
        ),
        "F32" => actual.chunks_exact(4).zip(expected.chunks_exact(4)).fold(
            (0.0, 0.0, 0.0_f64, 0usize),
            |totals, (actual, expected)| {
                let actual = f64::from(f32::from_le_bytes(actual.try_into().unwrap()));
                let expected = f64::from(f32::from_le_bytes(expected.try_into().unwrap()));
                let error = (actual - expected).abs();
                (
                    totals.0 + error * error,
                    totals.1 + expected * expected,
                    totals.2.max(error),
                    totals.3 + 1,
                )
            },
        ),
        other => panic!("unsupported state dtype {other:?}"),
    };
    assert!(count > 0);
    (
        (squared_error / squared_reference.max(f64::EPSILON)).sqrt(),
        max_absolute_error,
    )
}

fn assert_f32_bits_close(
    actual: u32,
    expected: u32,
    max_absolute_error: f32,
    max_relative_error: f32,
) {
    let actual = f32::from_bits(actual);
    let expected = f32::from_bits(expected);
    let error = (actual - expected).abs();
    let tolerance = max_absolute_error.max(expected.abs() * max_relative_error);
    assert!(
        error <= tolerance,
        "F32 values differ: actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn compile_temperature_top_k_top_p_sampler_test_kernels(
    vocab_size: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let candidates =
        std::fs::read_to_string(shader_dir.join("temperature_top_k_candidates_f32.comp.template"))
            .ok()?
            .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
            .replace("{{TOP_K}}", &top_k.to_string())
            .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
            .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = std::fs::read_to_string(
        shader_dir.join("temperature_top_k_top_p_sampler_f32.comp.template"),
    )
    .ok()?
    .replace("{{TEMPERATURE}}", &temperature.to_string())
    .replace("{{TOP_K}}", &top_k.to_string())
    .replace("{{TOP_P}}", &top_p.to_string())
    .replace("{{MIN_P}}", "0.0")
    .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
    .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    let mut kernels = vec![
        VulkanResidentSamplerKernelArtifact {
            role: "partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ];
    kernels.push(compile_feedback_control_test_kernel()?);
    Some(kernels)
}

fn compile_temperature_distribution_sampler_test_kernels(
    vocab_size: usize,
    _temperature: f32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let render = |template: &str| std::fs::read_to_string(shader_dir.join(template)).ok();
    let partitions = render("temperature_distribution_partitions_runtime_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = render("temperature_distribution_sampler_runtime_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-distribution-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    let mut kernels = vec![
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_partition_distribution".to_string(),
            spirv_words: compile("partitions", partitions)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_sample_distribution".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ];
    kernels.push(compile_feedback_control_test_kernel()?);
    Some(kernels)
}

fn compile_repetition_temperature_sampler_test_kernels(
    vocab_size: usize,
    repetition_penalty: f32,
    top_k: u32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let render = |template: &str| std::fs::read_to_string(shader_dir.join(template)).ok();
    let tracker = render("record_seen_token.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let batch_tracker = render("record_seen_tokens_batch64.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let candidates = render("temperature_top_k_candidates_repetition_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{REPETITION_PENALTY}}", &repetition_penalty.to_string())
        .replace("{{PRESENCE_PENALTY}}", "0.0")
        .replace("{{TOP_K}}", &top_k.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = render("temperature_top_k_top_p_sampler_f32.comp.template")?
        .replace("{{TEMPERATURE}}", "1.0")
        .replace("{{TOP_K}}", &top_k.to_string())
        .replace("{{TOP_P}}", "1.0")
        .replace("{{MIN_P}}", "0.0")
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-repetition-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    let mut kernels = vec![
        VulkanResidentSamplerKernelArtifact {
            role: "record_current_token".to_string(),
            spirv_words: compile("tracker", tracker)?,
            local_size_x: 1,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "record_token_batch".to_string(),
            spirv_words: compile("batch-tracker", batch_tracker)?,
            local_size_x: VULKAN_BACKEND_LOOP_MAX_WINDOW as u32,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ];
    kernels.push(compile_feedback_control_test_kernel()?);
    Some(kernels)
}

fn compile_runtime_temperature_sampler_test_kernels(
    vocab_size: usize,
    top_k_capacity: u32,
    partition_count: u32,
    local_size_x: u32,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let render = |template: &str| std::fs::read_to_string(shader_dir.join(template)).ok();
    let tracker = render("record_seen_token.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let batch_tracker = render("record_seen_tokens_batch64.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string());
    let candidates = render("temperature_top_k_candidates_runtime_f32.comp.template")?
        .replace("{{VOCAB_SIZE}}", &vocab_size.to_string())
        .replace("{{TOP_K_CAPACITY}}", &top_k_capacity.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let sampler = render("temperature_top_k_top_p_sampler_runtime_f32.comp.template")?
        .replace("{{TOP_K_CAPACITY}}", &top_k_capacity.to_string())
        .replace("{{PARTITION_COUNT}}", &partition_count.to_string())
        .replace("{{LOCAL_SIZE_X}}", &local_size_x.to_string());
    let compile = |suffix: &str, source: String| {
        let path = std::env::temp_dir().join(format!(
            "nerve-runtime-sampling-test-{}-{suffix}.comp",
            std::process::id()
        ));
        std::fs::write(&path, source).ok()?;
        let words = crate::vulkan_compute::compile_shader_words_from_source_path(&path);
        let _ = std::fs::remove_file(path);
        words
    };
    let mut kernels = vec![
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_record_current_token".to_string(),
            spirv_words: compile("tracker", tracker)?,
            local_size_x: 1,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_record_token_batch".to_string(),
            spirv_words: compile("batch-tracker", batch_tracker)?,
            local_size_x: VULKAN_BACKEND_LOOP_MAX_WINDOW as u32,
            workgroup_count_x: 1,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_partition_top_k".to_string(),
            spirv_words: compile("candidates", candidates)?,
            local_size_x,
            workgroup_count_x: partition_count,
        },
        VulkanResidentSamplerKernelArtifact {
            role: "runtime_sample_candidates".to_string(),
            spirv_words: compile("sample", sampler)?,
            local_size_x,
            workgroup_count_x: 1,
        },
    ];
    kernels.push(compile_feedback_control_test_kernel()?);
    Some(kernels)
}

fn greedy_sampler_test_kernels(
    spirv_words: Vec<u32>,
) -> Option<Vec<VulkanResidentSamplerKernelArtifact>> {
    Some(vec![
        VulkanResidentSamplerKernelArtifact {
        role: "sample_logits".to_string(),
        spirv_words,
        local_size_x: 1_024,
        workgroup_count_x: 1,
        },
        compile_feedback_control_test_kernel()?,
    ])
}

fn compile_feedback_control_test_kernel() -> Option<VulkanResidentSamplerKernelArtifact> {
    let shader =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/resident_feedback_control.comp");
    Some(VulkanResidentSamplerKernelArtifact {
        role: "feedback_control".to_string(),
        spirv_words: crate::vulkan_compute::compile_shader_words_from_source_path(&shader)?,
        local_size_x: 1,
        workgroup_count_x: 1,
    })
}

fn sampler_test_hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

fn fixture_model_tensor_index_path() -> PathBuf {
    tiny_model_tensor_index_path()
}

fn fixture_model_package_manifest_path() -> PathBuf {
    tiny_model_package_manifest_path()
}

fn tiny_fixture_model_package_manifest_path() -> PathBuf {
    tiny_model_package_manifest_path()
}

pub(super) fn tiny_fixture_model_runtime_model_with_placement(
    placement: StreamCircuitPlacementSpec,
) -> VulkanResidentRuntimeModel {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(
        tiny_fixture_model_package_manifest_path(),
    )
    .unwrap();
    let source_graph = manifest.resolved_source_graph(tiny_model_dir()).unwrap();
    let runtime_graph = source_graph
        .runtime_graph_from_placement(&placement)
        .unwrap();
    manifest.mount_runtime_graph(&runtime_graph).unwrap()
}

fn fixture_model_package_manifest() -> VulkanResidentModelPackageManifest {
    VulkanResidentModelPackageManifest::from_json_file(fixture_model_package_manifest_path())
        .unwrap()
}

fn fixture_model_runtime_model() -> VulkanResidentRuntimeModel {
    fixture_model_package_manifest()
        .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
        .unwrap()
}

fn fixture_model_runtime_model_with_one_dynamic_group() -> VulkanResidentRuntimeModel {
    let mut runtime_model = fixture_model_runtime_model();
    let contract = &mut runtime_model.package.resource_residency;
    let binding = contract
        .bindings
        .iter()
        .find(|binding| binding.parameter_id == "ffn_down")
        .unwrap();
    let CompiledResourceBindingMapping::AtomicGroup {
        atomic_group_id: original_group_id,
        resource_id: original_resource_id,
    } = &binding.mapping
    else {
        panic!("fixture ffn_down binding is not concrete");
    };
    let original_group_id = original_group_id.clone();
    let original_resource_id = original_resource_id.clone();
    let resource = contract
        .resources
        .iter_mut()
        .find(|resource| resource.id == original_resource_id)
        .unwrap();
    resource.lifetime = CompiledResourceLifetime::Dynamic;
    resource.id = package::compiled_resource_identity(resource).unwrap();
    let dynamic_resource_id = resource.id.clone();
    let original_group = contract
        .atomic_groups
        .iter_mut()
        .find(|group| group.id == original_group_id)
        .unwrap();
    original_group
        .resource_ids
        .retain(|resource_id| resource_id != &original_resource_id);
    original_group.id = package::compiled_atomic_group_identity(original_group).unwrap();
    let remaining_group_id = original_group.id.clone();
    let mut dynamic_group = CompiledAtomicResidencyGroup {
        id: String::new(),
        lifetime: CompiledResourceLifetime::Dynamic,
        resource_ids: vec![dynamic_resource_id.clone()],
        dependencies: Vec::new(),
    };
    dynamic_group.id = package::compiled_atomic_group_identity(&dynamic_group).unwrap();
    let dynamic_group_id = dynamic_group.id.clone();
    contract.atomic_groups.push(dynamic_group);
    contract.atomic_groups.sort_by(|left, right| left.id.cmp(&right.id));
    contract.resources.sort_by(|left, right| left.id.cmp(&right.id));
    for binding in &mut contract.bindings {
        let CompiledResourceBindingMapping::AtomicGroup {
            atomic_group_id,
            resource_id,
        } = &mut binding.mapping
        else {
            continue;
        };
        if *atomic_group_id != original_group_id {
            continue;
        }
        if *resource_id == original_resource_id {
            binding.mapping = CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id: dynamic_group_id.clone(),
                resource_id: dynamic_resource_id.clone(),
                selection_signal: "ffn_hidden".to_string(),
                selector_index: 0,
                parameter_slot: 0,
            };
        } else {
            *atomic_group_id = remaining_group_id.clone();
        }
    }
    let mut selector = CompiledResourceSelector {
        id: String::new(),
        execution_scope: "target".to_string(),
        component_id: "layer_00".to_string(),
        node_id: "ffn_gate_projection__ffn_up_projection__ffn_gate_activation__ffn_gate_multiply"
            .to_string(),
        domain_id: "fixture_dynamic_group".to_string(),
        resource_count: 1,
        selection_signal: "ffn_hidden".to_string(),
        execution_signal: "ffn_hidden".to_string(),
        execution_calibration_word_base: 0,
        encoding: CompiledResourceSelectionEncoding {
            element_type: CompiledResourceSelectionElementType::U32,
            selection_count_per_activation: 1,
            index_shift: 0,
            index_mask: 0,
            calibration_word_base: 0,
        },
        mapping: CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: vec![dynamic_group_id],
        },
    };
    selector.id = package::compiled_selector_identity(&selector).unwrap();
    let mut checkpoint = CompiledResidencyCheckpoint {
        id: String::new(),
        execution_scope: "target".to_string(),
        component_id: "layer_00".to_string(),
        after_node_id:
            "ffn_gate_projection__ffn_up_projection__ffn_gate_activation__ffn_gate_multiply"
                .to_string(),
        resume_node_id: "ffn_down_projection__ffn_residual".to_string(),
        selector_ids: vec![selector.id.clone()],
    };
    checkpoint.id = package::compiled_checkpoint_identity(&checkpoint).unwrap();
    contract.selectors = vec![selector];
    contract.checkpoints = vec![checkpoint];
    runtime_model
}

fn fixture_model_runtime_model_with_dynamic_partition(
    partition_count: usize,
    member_bytes: usize,
) -> VulkanResidentRuntimeModel {
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
    ) else {
        panic!("fixture ffn_down binding is not concrete");
    };
    let selector_component_id = binding.component_id.clone();
    let selector_node_id = binding.node_id.clone();
    contract.resources.retain(|resource| resource.id != resource_id);
    contract
        .atomic_groups
        .iter_mut()
        .find(|group| group.id == atomic_group_id)
        .unwrap()
        .resource_ids
        .retain(|id| id != &resource_id);
    binding.mapping = CompiledResourceBindingMapping::PartitionTemplateMember {
        partition_template_id: template_id.clone(),
        resource_identity_seed: member_seed.clone(),
        selection_signal: "selection".to_string(),
        parameter_slot: 0,
    };
    contract.partition_templates.push(CompiledPartitionTemplate {
        id: template_id.clone(),
        partition_count,
        lifetime: CompiledResourceLifetime::Dynamic,
        group_identity_seed: format!("sha256:{}", "3".repeat(64)),
        member_templates: vec![CompiledPartitionMemberTemplate {
            resource_identity_seed: member_seed,
            range_templates: vec![CompiledResourceRangeTemplate {
                artifact_path: "weights/parameter.safetensors".to_string(),
                base_byte_offset: 0,
                stride_bytes: member_bytes,
                byte_count: member_bytes,
                alignment_bytes: member_bytes,
                integrity: CompiledResourceRangeIntegrityTemplate {
                    algorithm: "sha256_table".to_string(),
                    digest_table_path: "integrity/partitions.sha256".to_string(),
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
            resident_derivation: None,
        }],
        dependencies: Vec::new(),
    });
    contract.selectors.push(CompiledResourceSelector {
        id: format!("sha256:{}", "4".repeat(64)),
        execution_scope: "target".to_string(),
        component_id: selector_component_id,
        node_id: selector_node_id,
        domain_id: "fixture_partitions".to_string(),
        resource_count: partition_count,
        selection_signal: "selection".to_string(),
        execution_signal: "selection".to_string(),
        execution_calibration_word_base: 0,
        encoding: CompiledResourceSelectionEncoding {
            element_type: CompiledResourceSelectionElementType::U32,
            selection_count_per_activation: 1,
            index_shift: 0,
            index_mask: u32::MAX,
            calibration_word_base: 0,
        },
        mapping: CompiledResourceSelectorMapping::PartitionTemplate {
            partition_template_id: template_id,
        },
    });
    contract.selectors.sort_by(|left, right| left.id.cmp(&right.id));
    runtime_model
}

pub(super) fn fixture_model_runtime_model_with_three_layer_series(
    middle_device_id: &str,
) -> VulkanResidentRuntimeModel {
    let manifest = fixture_model_package_manifest();
    let source_graph = manifest.resolved_source_graph(tiny_model_dir()).unwrap();
    let runtime_graph = StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_00", "layer_00_remote")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_00_remote", "layer_00_tail")
        .unwrap()
        .with_instance_device("layer_00_remote", middle_device_id)
        .unwrap();
    manifest.mount_runtime_graph(&runtime_graph).unwrap()
}

fn fixture_model_runtime_model_with_colocated_three_layer_series() -> VulkanResidentRuntimeModel {
    fixture_model_runtime_model_with_three_layer_series("gpu0")
}

fn fixture_model_runtime_model_with_remote_middle() -> VulkanResidentRuntimeModel {
    fixture_model_runtime_model_with_three_layer_series("gpu1")
}

fn fixture_model_placed_resident_plan_for_device(
    runtime_model: &VulkanResidentRuntimeModel,
    device_id: &str,
) -> VulkanPlacedStreamCircuitResidentPlan {
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
    VulkanPlacedStreamCircuitResidentPlan::from_resource_plan_for_device(
        &resource_plan,
        &placement_plan,
        device_id,
        Some(&tensor_index),
        Some(2),
    )
    .unwrap()
}

fn fixture_model_execution_graph() -> ResolvedLoweredExecutionGraph {
    let full = fixture_model_package_manifest()
        .resolved_source_graph(tiny_model_dir())
        .unwrap();
    let processor_ids = full
        .circuits
        .iter()
        .filter(|artifact| artifact.circuit.runtime_role.is_signal_processor())
        .map(|artifact| artifact.component.id.as_str())
        .collect::<BTreeSet<_>>();
    let circuits = full
        .circuits
        .iter()
        .filter(|artifact| processor_ids.contains(artifact.component.id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let mut index = full.index.clone();
    index.graph.circuits = circuits
        .iter()
        .map(|artifact| artifact.component.clone())
        .collect();
    index.graph.edges = full
        .index
        .graph
        .edges
        .iter()
        .filter(|edge| {
            edge.connection.is_instantaneous()
                && processor_ids.contains(edge.source.component_id.as_str())
                && processor_ids.contains(edge.destination.component_id.as_str())
        })
        .cloned()
        .collect();
    index.graph.boundary = StreamCircuitGraphBoundary {
        external_inputs: execution_boundary_inputs(&full, &processor_ids),
        public_outputs: execution_boundary_outputs(&full, &processor_ids),
    };
    let mut operator_counts = BTreeMap::new();
    for artifact in &circuits {
        *operator_counts
            .entry(artifact.component.operator_type.clone())
            .or_insert(0) += 1;
    }
    index.summary = LoweredExecutionGraphSummary {
        circuit_count: circuits.len(),
        operator_counts,
    };
    ResolvedLoweredExecutionGraph {
        artifact_root: full.artifact_root,
        index,
        circuits,
    }
}

fn copy_package_integrity_artifacts(
    source_root: &Path,
    destination_root: &Path,
    manifest: &VulkanResidentModelPackageManifest,
) {
    let residency_artifacts = manifest
        .resource_residency
        .resources
        .iter()
        .flat_map(|resource| {
            resource
                .ranges
                .iter()
                .map(|range| range.artifact_path.as_str())
        })
        .chain(
            manifest
                .resource_residency
                .partition_templates
                .iter()
                .flat_map(|template| template.member_templates.iter())
                .flat_map(|member| member.range_templates.iter())
                .flat_map(|range| {
                    [
                        range.artifact_path.as_str(),
                        range.integrity.digest_table_path.as_str(),
                    ]
                }),
        );
    let paths = manifest
        .artifact_integrity
        .files
        .keys()
        .map(String::as_str)
        .chain(residency_artifacts)
        .collect::<BTreeSet<_>>();
    for relative_path in paths {
        let destination = destination_root.join(relative_path);
        std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
        std::fs::copy(source_root.join(relative_path), destination).unwrap();
    }
}

fn reusable_family_with_kernel<'a>(
    reusable_plan: &'a VulkanReusableKernelPlan,
    kernel_id: &str,
) -> &'a VulkanReusableKernelFamily {
    reusable_plan
        .families
        .iter()
        .find(|family| {
            family
                .command_refs
                .iter()
                .any(|command| command.kernel_id == kernel_id)
        })
        .unwrap()
}

fn artifact_path_for_family(family: &VulkanReusableKernelFamily) -> String {
    format!("kernels/{}.spv", family.family_id)
}
