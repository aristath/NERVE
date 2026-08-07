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
fn demand_feedback_requires_one_unambiguous_miss_checkpoint() {
    assert_eq!(
        unique_pending_demand_feedback_checkpoint(&[]).unwrap(),
        None
    );
    assert_eq!(
        unique_pending_demand_feedback_checkpoint(&[(3, 1, 4)]).unwrap(),
        Some((3, 1, 4))
    );
    assert!(
        unique_pending_demand_feedback_checkpoint(&[(3, 1, 4), (3, 2, 0)]).is_err()
    );
}

#[test]
fn demand_feedback_never_commits_an_attempt_that_resolved_a_residency_checkpoint() {
    assert_eq!(
        demand_feedback_attempt_completion(false),
        VulkanDemandFeedbackAttemptCompletion::Commit
    );
    assert_eq!(
        demand_feedback_attempt_completion(true),
        VulkanDemandFeedbackAttemptCompletion::RestoreBaselineAndReplay
    );
}

#[test]
fn scalar_and_feedback_demand_chains_have_distinct_predicate_ownership() {
    assert!(!VulkanDemandResidencyChainLane::Scalar.uses_shared_pipeline_guard());
    assert!(
        VulkanDemandResidencyChainLane::Feedback(0).uses_shared_pipeline_guard()
    );
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
        .filter(|stage| matches!(stage, VulkanMountedPlacedStreamTickStage::ReceiveEdge { .. }))
        .count();
    let dispatch_stage_count = stages
        .iter()
        .filter(|stage| matches!(stage, VulkanMountedPlacedStreamTickStage::Dispatch { .. }))
        .count();
    let publish_stage_count = stages
        .iter()
        .filter(|stage| matches!(stage, VulkanMountedPlacedStreamTickStage::PublishEdge { .. }))
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
    let gpu0 = demand_feedback_test_plan(
        "gpu0",
        vec![demand_feedback_test_dispatch(0, "gpu0")],
    );
    let gpu1 = demand_feedback_test_plan(
        "gpu1",
        vec![demand_feedback_test_dispatch(0, "gpu1")],
    );

    let error = demand_feedback_resume_plan(&[&gpu0, &gpu1], 1, 0).unwrap_err();
    assert!(error.0.contains("independent parallel"));
}

#[test]
fn demand_feedback_continuation_never_replays_completed_lanes() {
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
