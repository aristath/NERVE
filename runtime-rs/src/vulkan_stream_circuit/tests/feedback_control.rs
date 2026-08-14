#[test]
fn feedback_control_words_encode_stop_tokens_and_dispatches_without_token_caps() {
    let vocabulary_size = 65usize;
    let stop_mask_word_count = vocabulary_size.div_ceil(u32::BITS as usize);
    let dispatch_word_offset =
        VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + stop_mask_word_count;
    let dimensions = [[11, 2, 1], [7, 1, 1]];

    let words = resident_feedback_control_words(
        vocabulary_size,
        stop_mask_word_count,
        dispatch_word_offset,
        &dimensions,
        1,
        1,
        64,
        &[0, 31, 32, 64],
    )
    .unwrap();

    assert_eq!(
        &words[..VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT],
        &[
            VULKAN_FEEDBACK_CONTROL_ENABLED,
            0,
            VULKAN_FEEDBACK_STOP_REASON_NONE,
            64,
            dispatch_word_offset as u32,
            2,
            1,
            1,
            0,
            0,
            VULKAN_FEEDBACK_CONTINUATION_READY,
            0,
        ]
    );
    assert_eq!(
        &words[VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT..dispatch_word_offset],
        &[0x8000_0001, 1, 1]
    );
    assert_eq!(&words[dispatch_word_offset..], &[11, 2, 1, 7, 1, 1]);

    let error = resident_feedback_control_words(
        vocabulary_size,
        stop_mask_word_count,
        dispatch_word_offset,
        &dimensions,
        1,
        1,
        64,
        &[65],
    )
    .unwrap_err();
    assert_eq!(
        error,
        VulkanError("stop token id 65 exceeds vocabulary size 65".to_string())
    );
}

#[test]
fn feedback_execution_stats_distinguish_committed_work_from_predicated_tail() {
    let mut stats = VulkanResidentFeedbackExecutionStats::default();
    let submission_topology = VulkanResidentFeedbackSubmissionTopology::new(96, 4).unwrap();
    stats
        .record_window(64, 7, 6, false, submission_topology)
        .unwrap();
    stats
        .record_window(64, 64, 64, true, submission_topology)
        .unwrap();

    assert_eq!(
        stats,
        VulkanResidentFeedbackExecutionStats {
            window_count: 2,
            planned_tick_count: 128,
            submitted_tick_count: 128,
            executed_tick_count: 71,
            retained_tick_count: 71,
            sampled_tick_count: 70,
            discarded_tick_count: 57,
            template_record_count: 1,
            template_replay_count: 1,
            queue_submission_count: 192,
            host_queue_submit_count: 8,
            maximum_host_queue_submit_count_per_window: 4,
            asynchronous_submission_count: 0,
            completion_poll_count: 0,
            bounded_wait_count: 0,
            bounded_wait_timeout_count: 0,
        }
    );
}

#[test]
fn feedback_execution_stats_reject_impossible_host_submission_evidence() {
    assert_eq!(
        VulkanResidentFeedbackSubmissionTopology::new(0, 0),
        Err(VulkanError(
            "resident feedback window has no queue submissions".to_string()
        ))
    );
    assert_eq!(
        VulkanResidentFeedbackSubmissionTopology::new(4, 5),
        Err(VulkanError(
            "resident feedback window has 5 host queue submits for 4 queued submissions"
                .to_string()
        ))
    );
}

#[test]
fn feedback_execution_stats_reject_overflow_without_mutation() {
    let topology = VulkanResidentFeedbackSubmissionTopology::new(4, 2).unwrap();
    let mut queue_overflow = VulkanResidentFeedbackExecutionStats {
        queue_submission_count: usize::MAX - 3,
        ..VulkanResidentFeedbackExecutionStats::default()
    };
    let queue_before = queue_overflow;
    assert_eq!(
        queue_overflow.record_window(8, 8, 1, false, topology),
        Err(VulkanError(
            "resident feedback aggregate queue-submission count overflowed".to_string()
        ))
    );
    assert_eq!(queue_overflow, queue_before);

    let mut host_overflow = VulkanResidentFeedbackExecutionStats {
        host_queue_submit_count: usize::MAX - 1,
        ..VulkanResidentFeedbackExecutionStats::default()
    };
    let host_before = host_overflow;
    assert_eq!(
        host_overflow.record_window(8, 8, 1, false, topology),
        Err(VulkanError(
            "resident feedback aggregate host queue-submit count overflowed".to_string()
        ))
    );
    assert_eq!(host_overflow, host_before);
}

#[test]
fn feedback_submission_topology_accumulates_demand_continuations_exactly() {
    let initial = VulkanResidentFeedbackSubmissionTopology::new(96, 4).unwrap();
    let continuation = VulkanResidentFeedbackSubmissionTopology::new(32, 2).unwrap();

    assert_eq!(
        initial.merged(continuation).unwrap(),
        VulkanResidentFeedbackSubmissionTopology {
            queue_submission_count: 128,
            host_queue_submit_count: 6,
        }
    );
    assert_eq!(
        VulkanResidentFeedbackSubmissionTopology {
            queue_submission_count: usize::MAX,
            host_queue_submit_count: 1,
        }
        .merged(continuation),
        Err(VulkanError(
            "resident feedback queue-submission count overflowed".to_string()
        ))
    );
    assert_eq!(
        VulkanResidentFeedbackSubmissionTopology {
            queue_submission_count: usize::MAX,
            host_queue_submit_count: usize::MAX,
        }
        .merged(VulkanResidentFeedbackSubmissionTopology {
            queue_submission_count: 0,
            host_queue_submit_count: 1,
        }),
        Err(VulkanError(
            "resident feedback host queue-submit count overflowed".to_string()
        ))
    );
}

#[test]
fn feedback_cancellation_handle_can_cross_the_runtime_worker_boundary() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VulkanResidentFeedbackCancellationHandle>();
}

#[test]
fn feedback_window_policy_learns_a_responsive_execution_width() {
    let policy = VulkanResidentFeedbackWindowPolicy::new(64);
    assert_eq!(policy.next_tick_count(), 2);

    policy.observe_completed_window(2, 2, 100_000_000, false);
    assert_eq!(policy.next_tick_count(), 5);

    policy.observe_completed_window(5, 5, 500_000_000, false);
    assert_eq!(policy.next_tick_count(), 4);

    policy.observe_completed_window(4, 2, 1, true);
    assert_eq!(policy.next_tick_count(), 4);
}
