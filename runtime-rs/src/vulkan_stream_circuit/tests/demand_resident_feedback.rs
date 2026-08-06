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
        vec![None, Some(1)]
    );
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
fn demand_feedback_retries_only_after_a_stable_scalar_feedback_tick() {
    assert!(!demand_feedback_retry_deferred_after_scalar(
        true, true, true, false, false,
    ));
    assert!(demand_feedback_retry_deferred_after_scalar(
        true, true, true, false, true,
    ));
    assert!(demand_feedback_retry_deferred_after_scalar(
        true, false, true, false, false,
    ));
    assert!(demand_feedback_retry_deferred_after_scalar(
        true, true, false, false, false,
    ));
    assert!(demand_feedback_retry_deferred_after_scalar(
        true, true, true, true, false,
    ));
    assert!(!demand_feedback_retry_deferred_after_scalar(
        false, true, true, false, true,
    ));
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
