#[test]
fn demand_feedback_guards_the_entire_initial_pipeline_and_splits_after_gates() {
    let commands = [
        VulkanDemandResidencyCommand::Prefix(0),
        VulkanDemandResidencyCommand::Dispatch(0),
        VulkanDemandResidencyCommand::Gate(0),
        VulkanDemandResidencyCommand::Dispatch(1),
        VulkanDemandResidencyCommand::Gate(1),
        VulkanDemandResidencyCommand::Dispatch(2),
        VulkanDemandResidencyCommand::Suffix(0),
    ];

    assert_eq!(
        demand_dispatch_conditional_regions(&commands, 0, 2, true, false).unwrap(),
        vec![
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            Some(3),
        ]
    );
}

#[test]
fn demand_feedback_does_not_record_unconsumed_critical_path_timestamps() {
    assert!(!VulkanDemandResidencySequencePurpose::Unprofiled.records_critical_path());
    assert!(VulkanDemandResidencySequencePurpose::Profiled.records_critical_path());
}

#[test]
fn standalone_demand_execution_leaves_the_selector_prefix_direct() {
    let commands = [
        VulkanDemandResidencyCommand::Prefix(0),
        VulkanDemandResidencyCommand::Dispatch(0),
        VulkanDemandResidencyCommand::Gate(0),
        VulkanDemandResidencyCommand::Dispatch(1),
        VulkanDemandResidencyCommand::Gate(1),
        VulkanDemandResidencyCommand::Dispatch(2),
    ];

    assert_eq!(
        demand_dispatch_conditional_regions(&commands, 0, 2, false, false).unwrap(),
        vec![None, None, None, Some(1), Some(1), Some(2)]
    );
    assert_eq!(
        demand_dispatch_conditional_regions(&commands, 4, 4, true, true).unwrap(),
        vec![None, None]
    );
}

#[test]
fn demand_checkpoint_resume_runs_the_loaded_span_through_the_next_gate_directly() {
    let commands = [
        VulkanDemandResidencyCommand::Prefix(0),
        VulkanDemandResidencyCommand::Dispatch(0),
        VulkanDemandResidencyCommand::Gate(0),
        VulkanDemandResidencyCommand::Dispatch(1),
        VulkanDemandResidencyCommand::Gate(1),
        VulkanDemandResidencyCommand::Dispatch(2),
        VulkanDemandResidencyCommand::Suffix(0),
    ];

    assert_eq!(
        demand_dispatch_conditional_regions(&commands, 2, 2, true, true).unwrap(),
        vec![None, None, None, Some(1), Some(1)]
    );
    assert_eq!(
        demand_dispatch_conditional_regions(&commands, 4, 4, true, true).unwrap(),
        vec![None, None, None]
    );
}

#[test]
fn demand_checkpoint_resume_preserves_the_full_sequence_indirect_offsets() {
    assert_eq!(
        demand_feedback_indirect_command_range(7, 4)
            .unwrap()
            .collect::<Vec<_>>(),
        vec![4, 5, 6]
    );
    assert!(demand_feedback_indirect_command_range(7, 7).is_err());
    assert!(demand_feedback_indirect_command_range(7, 8).is_err());
}

#[test]
fn demand_checkpoint_resume_places_terminal_snapshots_after_the_exact_suffix() {
    assert_eq!(demand_feedback_sequence_step_count(358, 0).unwrap(), 358);
    assert_eq!(demand_feedback_sequence_step_count(358, 10).unwrap(), 348);
    assert!(demand_feedback_sequence_step_count(358, 358).is_err());
    assert!(demand_feedback_sequence_step_count(358, 359).is_err());
}

#[test]
fn demand_feedback_requires_one_unambiguous_miss_checkpoint() {
    assert_eq!(
        unique_pending_demand_feedback_checkpoint(&[]).unwrap(),
        None
    );
    assert_eq!(
        unique_pending_demand_feedback_checkpoint(&[(3, 1, 4)]).unwrap(),
        Some((3, 1, 4))
    );
    assert!(unique_pending_demand_feedback_checkpoint(&[(3, 1, 4), (3, 2, 0)]).is_err());
}

#[test]
fn demand_feedback_fault_is_explicit_and_never_inferred_from_partial_counters() {
    assert_eq!(
        resident_feedback_fault_reason_from_continuation(VULKAN_FEEDBACK_CONTINUATION_READY)
            .unwrap(),
        VULKAN_FEEDBACK_FAULT_NONE,
    );
    assert_eq!(
        resident_feedback_fault_reason_from_continuation(VULKAN_FEEDBACK_CONTINUATION_FAULTED)
            .unwrap(),
        VULKAN_FEEDBACK_FAULT_RESIDENCY,
    );
    assert!(resident_feedback_fault_reason_from_continuation(2).is_err());

    let complete = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 8,
        sampled_tick_count: 8,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_NONE,
        fault_reason: VULKAN_FEEDBACK_FAULT_NONE,
        template_replayed: false,
    };
    assert_eq!(
        resident_feedback_terminal_state(complete, 8).unwrap(),
        VulkanResidentFeedbackTerminalState::Complete
    );

    let fault = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 3,
        sampled_tick_count: 3,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_EOS,
        fault_reason: VULKAN_FEEDBACK_FAULT_RESIDENCY,
        template_replayed: false,
    };
    assert_eq!(
        resident_feedback_terminal_state(fault, 8).unwrap(),
        VulkanResidentFeedbackTerminalState::ResidencyFault
    );

    let fault_before_any_ordinary_progress = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 0,
        sampled_tick_count: 0,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_CANCELLED,
        fault_reason: VULKAN_FEEDBACK_FAULT_RESIDENCY,
        template_replayed: false,
    };
    assert_eq!(
        resident_feedback_terminal_state(fault_before_any_ordinary_progress, 8).unwrap(),
        VulkanResidentFeedbackTerminalState::ResidencyFault
    );

    let inferred_fault = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 3,
        sampled_tick_count: 3,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_NONE,
        fault_reason: VULKAN_FEEDBACK_FAULT_NONE,
        template_replayed: false,
    };
    assert!(resident_feedback_terminal_state(inferred_fault, 8).is_err());

    let contradictory_fault = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 3,
        sampled_tick_count: 4,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_NONE,
        fault_reason: VULKAN_FEEDBACK_FAULT_RESIDENCY,
        template_replayed: false,
    };
    assert!(resident_feedback_terminal_state(contradictory_fault, 8).is_err());

    let unknown_fault = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 8,
        sampled_tick_count: 8,
        stop_reason: VULKAN_FEEDBACK_STOP_REASON_NONE,
        fault_reason: 2,
        template_replayed: false,
    };
    assert!(resident_feedback_terminal_state(unknown_fault, 8).is_err());

    let unknown_stop_during_fault = VulkanResidentFeedbackControlCompletion {
        executed_tick_count: 0,
        sampled_tick_count: 0,
        stop_reason: 3,
        fault_reason: VULKAN_FEEDBACK_FAULT_RESIDENCY,
        template_replayed: false,
    };
    assert!(resident_feedback_terminal_state(unknown_stop_during_fault, 8).is_err());
}

#[test]
fn demand_feedback_allows_one_checkpoint_to_discover_distinct_resource_sets() {
    let checkpoint = VulkanDemandFeedbackCheckpoint {
        feedback_lane: 1,
        slice_index: 2,
        segment_index: 3,
        gate_index: 4,
    };
    let mut resolved = BTreeMap::new();
    assert_eq!(
        record_demand_feedback_resolution(&mut resolved, checkpoint, &[4, 68]).unwrap(),
        2
    );
    assert_eq!(
        record_demand_feedback_resolution(&mut resolved, checkpoint, &[51, 89, 138]).unwrap(),
        3
    );
    assert_eq!(
        resolved.get(&checkpoint),
        Some(&BTreeSet::from([4, 51, 68, 89, 138]))
    );
}

#[test]
fn demand_feedback_rejects_a_resource_that_faults_twice_at_one_checkpoint() {
    let checkpoint = VulkanDemandFeedbackCheckpoint {
        feedback_lane: 1,
        slice_index: 2,
        segment_index: 3,
        gate_index: 4,
    };
    let mut resolved = BTreeMap::new();
    record_demand_feedback_resolution(&mut resolved, checkpoint, &[4, 68]).unwrap();
    let error =
        record_demand_feedback_resolution(&mut resolved, checkpoint, &[68, 89]).unwrap_err();
    assert!(error.0.contains("missed resources [68] again"));
    assert_eq!(resolved.get(&checkpoint), Some(&BTreeSet::from([4, 68])));
}

#[test]
fn demand_feedback_resolution_bound_covers_every_checkpoint_resource_pair() {
    assert_eq!(
        demand_feedback_resolution_bound(8, [256, 128, 64]).unwrap(),
        3_584
    );
    assert!(demand_feedback_resolution_bound(0, [256]).is_err());
    assert!(demand_feedback_resolution_bound(1, [0]).is_err());
    assert!(demand_feedback_resolution_bound(1, []).is_err());
    assert!(demand_feedback_resolution_bound(2, [usize::MAX]).is_err());
}

#[test]
fn demand_residency_loads_the_immutable_gpu_miss_record_not_mutable_selector_state() {
    let requests = [
        VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 7,
            resource_index: 19,
        },
        VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 7,
            resource_index: 3,
        },
        VulkanGpuResidencyMissingRequest {
            checkpoint_tag: 7,
            resource_index: 19,
        },
    ];
    assert_eq!(
        exact_demand_miss_resource_indices(&requests).unwrap(),
        vec![3, 19]
    );
    assert!(exact_demand_miss_resource_indices(&[]).is_err());
}

#[test]
fn scalar_and_feedback_demand_chains_have_distinct_predicate_ownership() {
    assert!(!VulkanDemandResidencyChainLane::Scalar.uses_shared_pipeline_guard());
    assert!(VulkanDemandResidencyChainLane::Feedback(0).uses_shared_pipeline_guard());
}

#[test]
fn demand_feedback_preallocates_every_window_lane_before_execution() {
    assert!(demand_feedback_chain_keys(3, 0).is_err());
    assert_eq!(
        demand_feedback_chain_keys(3, 4).unwrap(),
        vec![
            (3, VulkanDemandResidencyChainLane::Feedback(0)),
            (3, VulkanDemandResidencyChainLane::Feedback(1)),
            (3, VulkanDemandResidencyChainLane::Feedback(2)),
            (3, VulkanDemandResidencyChainLane::Feedback(3)),
        ]
    );
}

fn demand_feedback_test_dispatch(
    stage_index: usize,
    device_id: &str,
) -> VulkanMountedPlacedStreamTickStage {
    VulkanMountedPlacedStreamTickStage::Dispatch {
        stage_index,
        dispatch: VulkanMountedPlacedStreamTickDispatch {
            dispatch_index: stage_index,
            kernel_id: format!("{device_id}.kernel_{stage_index}"),
            component_id: format!("{device_id}.component"),
            node_id: format!("node_{stage_index}"),
            op: "test".to_string(),
            descriptor_count: 0,
            resident_descriptor_count: 0,
            reads: Vec::new(),
            writes: Vec::new(),
        },
    }
}

fn demand_feedback_test_plan(
    device_id: &str,
    stages: Vec<VulkanMountedPlacedStreamTickStage>,
) -> VulkanMountedPlacedStreamTickPlan {
    let stage_count = stages.len();
    let receive_stage_count = stages
        .iter()
        .filter(|stage| {
            matches!(
                stage,
                VulkanMountedPlacedStreamTickStage::ReceiveEdge { .. }
            )
        })
        .count();
    let dispatch_stage_count = stages
        .iter()
        .filter(|stage| matches!(stage, VulkanMountedPlacedStreamTickStage::Dispatch { .. }))
        .count();
    let publish_stage_count = stages
        .iter()
        .filter(|stage| {
            matches!(
                stage,
                VulkanMountedPlacedStreamTickStage::PublishEdge { .. }
            )
        })
        .count();
    VulkanMountedPlacedStreamTickPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: device_id.to_string(),
        stages,
        stage_count,
        receive_stage_count,
        dispatch_stage_count,
        publish_stage_count,
        local_edge_read_count: 0,
        local_edge_write_count: 0,
        incoming_edge_read_count: receive_stage_count,
        outgoing_edge_write_count: publish_stage_count,
        model_input_read_count: 0,
        model_output_write_count: 0,
        can_execute: true,
    }
}

fn demand_feedback_test_publish(
    stage_index: usize,
    edge_index: usize,
    _from: &str,
    to: &str,
) -> VulkanMountedPlacedStreamTickStage {
    VulkanMountedPlacedStreamTickStage::PublishEdge {
        stage_index,
        edge_index,
        endpoint_id: format!("edge_{edge_index}_out"),
        buffer_index: 0,
        byte_capacity: 16,
        remote_device_id: to.to_string(),
        remote_component_id: format!("{to}.component"),
    }
}

fn demand_feedback_test_receive(
    stage_index: usize,
    edge_index: usize,
    from: &str,
    _to: &str,
) -> VulkanMountedPlacedStreamTickStage {
    VulkanMountedPlacedStreamTickStage::ReceiveEdge {
        stage_index,
        edge_index,
        endpoint_id: format!("edge_{edge_index}_in"),
        buffer_index: 0,
        byte_capacity: 16,
        remote_device_id: from.to_string(),
        remote_component_id: format!("{from}.component"),
    }
}

#[test]
fn demand_feedback_resume_plan_keeps_only_the_causal_prefix() {
    let gpu0 = demand_feedback_test_plan(
        "gpu0",
        vec![
            demand_feedback_test_dispatch(0, "gpu0"),
            demand_feedback_test_publish(1, 0, "gpu0", "gpu1"),
        ],
    );
    let gpu1 = demand_feedback_test_plan(
        "gpu1",
        vec![
            demand_feedback_test_receive(0, 0, "gpu0", "gpu1"),
            demand_feedback_test_dispatch(1, "gpu1"),
            demand_feedback_test_publish(2, 1, "gpu1", "gpu2"),
        ],
    );
    let gpu2 = demand_feedback_test_plan(
        "gpu2",
        vec![
            demand_feedback_test_receive(0, 1, "gpu1", "gpu2"),
            demand_feedback_test_dispatch(1, "gpu2"),
        ],
    );
    let plans = [&gpu0, &gpu1, &gpu2];

    let middle = demand_feedback_resume_plan(&plans, 1, 1).unwrap();
    assert_eq!(middle.schedule_start_turn_index, 0);
    assert_eq!(middle.next_stage_indices, [2, 1, 0]);

    let terminal = demand_feedback_resume_plan(&plans, 2, 1).unwrap();
    assert_eq!(terminal.schedule_start_turn_index, 0);
    assert_eq!(terminal.next_stage_indices, [2, 3, 1]);
}

#[test]
fn demand_feedback_resume_plan_handles_a_causal_device_revisit() {
    let gpu0 = demand_feedback_test_plan(
        "gpu0",
        vec![
            demand_feedback_test_dispatch(0, "gpu0"),
            demand_feedback_test_publish(1, 0, "gpu0", "gpu1"),
            demand_feedback_test_receive(2, 1, "gpu1", "gpu0"),
            demand_feedback_test_dispatch(3, "gpu0"),
        ],
    );
    let gpu1 = demand_feedback_test_plan(
        "gpu1",
        vec![
            demand_feedback_test_receive(0, 0, "gpu0", "gpu1"),
            demand_feedback_test_dispatch(1, "gpu1"),
            demand_feedback_test_publish(2, 1, "gpu1", "gpu0"),
        ],
    );

    let resume = demand_feedback_resume_plan(&[&gpu0, &gpu1], 0, 3).unwrap();
    assert_eq!(resume.schedule_start_turn_index, 1);
    assert_eq!(resume.next_stage_indices, [3, 3]);
}

#[test]
fn demand_feedback_resume_rejects_an_independent_parallel_branch() {
    let gpu0 = demand_feedback_test_plan("gpu0", vec![demand_feedback_test_dispatch(0, "gpu0")]);
    let gpu1 = demand_feedback_test_plan("gpu1", vec![demand_feedback_test_dispatch(0, "gpu1")]);

    let error = demand_feedback_resume_plan(&[&gpu0, &gpu1], 1, 0).unwrap_err();
    assert!(error.0.contains("independent parallel"));
}

#[test]
fn demand_feedback_commits_the_causal_prefix_and_resumes_only_the_uncommitted_suffix() {
    assert_eq!(
        demand_feedback_continuation_lanes(8, 3)
            .unwrap()
            .collect::<Vec<_>>(),
        [3, 4, 5, 6, 7]
    );
    assert_eq!(
        demand_feedback_continuation_lanes(8, 7)
            .unwrap()
            .collect::<Vec<_>>(),
        [7]
    );
    assert!(demand_feedback_continuation_lanes(8, 8).is_err());
    assert!(demand_feedback_continuation_lanes(0, 0).is_err());
}

#[test]
fn disjoint_demand_faults_execute_every_feedback_lane_exactly_once() {
    // Lanes 0..3 completed before the first fault. Its continuation completed
    // lanes 3..5 before a second, disjoint fault. The final continuation must
    // begin at lane 5: replaying either earlier range would duplicate sampler
    // and recurrent-state mutations that are already causally committed.
    let executed = (0..3)
        .chain(3..5)
        .chain(demand_feedback_continuation_lanes(8, 5).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(executed, (0..8).collect::<Vec<_>>());
    assert_eq!(
        executed.iter().copied().collect::<BTreeSet<_>>().len(),
        executed.len()
    );
}

#[test]
fn demand_feedback_resume_rejects_non_dispatch_and_unmatched_edge_targets() {
    let unmatched = demand_feedback_test_plan(
        "gpu0",
        vec![demand_feedback_test_publish(0, 0, "gpu0", "gpu1")],
    );
    assert!(
        demand_feedback_resume_plan(&[&unmatched], 0, 0)
            .unwrap_err()
            .0
            .contains("unmatched")
    );

    let gpu0 = demand_feedback_test_plan(
        "gpu0",
        vec![
            demand_feedback_test_dispatch(0, "gpu0"),
            demand_feedback_test_publish(1, 0, "gpu0", "gpu1"),
        ],
    );
    let gpu1 = demand_feedback_test_plan(
        "gpu1",
        vec![demand_feedback_test_receive(0, 0, "gpu0", "gpu1")],
    );
    assert!(
        demand_feedback_resume_plan(&[&gpu0, &gpu1], 1, 0)
            .unwrap_err()
            .0
            .contains("dispatch segment")
    );
    assert!(demand_feedback_resume_plan(&[&gpu0, &gpu1], 2, 0).is_err());
    assert!(demand_feedback_resume_plan(&[&gpu0, &gpu1], 0, 2).is_err());
}
