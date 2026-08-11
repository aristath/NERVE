#[test]
fn placed_prompt_stream_owns_package_devices_and_session() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device.clone()),
    ]);

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    assert_eq!(stream.package().input_device_id, "gpu0");
    assert_eq!(stream.package().output_device_id, "gpu0");
    assert_eq!(stream.devices().len(), 2);
    assert_eq!(stream.next_stream_tick(), 0);

    let first = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("event_a", vec![1], 1))
        .unwrap();
    assert_eq!(first.session_run.prompt_event_index, 0);
    assert_eq!(first.session_run.start_stream_tick, 0);
    assert_eq!(first.session_run.next_stream_tick, 2);
    assert_eq!(stream.next_stream_tick(), 2);
    assert_eq!(stream.completed_prompt_event_count(), 1);

    let second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_b",
            vec![4],
            1,
        ))
        .unwrap();
    assert_eq!(second.session_run.prompt_event_index, 1);
    assert_eq!(second.session_run.start_stream_tick, 2);
    assert_eq!(second.session_run.next_stream_tick, 4);
    assert_eq!(stream.next_stream_tick(), 4);
    assert_eq!(stream.completed_prompt_event_count(), 2);
    assert_eq!(second.session_run.run.output_source_stream_ticks, vec![2]);
}

#[test]
fn placed_prompt_stream_runs_resident_feedback_across_bridged_slices() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let runtime_model = fixture_model_runtime_model_with_colocated_three_layer_series();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
    )
    .unwrap();
    let maximum_window_tick_count = stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .maximum_tick_count;
    assert!(maximum_window_tick_count >= 3);
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(maximum_window_tick_count);

    let mut first_streamed_output_events = Vec::new();
    stream.enqueue_input_event(
        VulkanResidentTokenInputEvent::new("event", vec![1], 8).with_stop_tokens(vec![16]),
    );
    let resident_first = stream
        .run_next_queued_input_event_with_output(|event| first_streamed_output_events.push(event))
        .unwrap()
        .unwrap();

    assert_eq!(resident_first.generated_token_ids, vec![16]);
    assert_eq!(resident_first.output_events, first_streamed_output_events);
    assert_eq!(
        resident_first
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![0]
    );
    assert_eq!(resident_first.session_run.run.stop_reason, "eos");
    assert_eq!(resident_first.session_run.tick_count, 2);
    assert_eq!(resident_first.session_run.next_stream_tick, 2);
    assert_eq!(
        resident_first.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats::default()
    );

    let resident_second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_after_eos",
            vec![4],
            3,
        ))
        .unwrap();
    assert_eq!(
        resident_second
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![2, 3, 4]
    );
    assert_eq!(
        resident_second.session_run.run.stop_reason,
        "max_new_tokens"
    );
    assert_eq!(resident_second.session_run.tick_count, 4);
    assert_eq!(resident_second.session_run.next_stream_tick, 6);
    assert_eq!(stream.next_stream_tick(), 6);
    assert!(stream.is_idle());
    drop(stream);

    let bridged_runtime_model = fixture_model_runtime_model_with_remote_middle();
    let bridged_devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device),
    ]);
    let mut bridged_stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            bridged_devices,
            manifest_dir,
            bridged_runtime_model,
            Some(8),
            0,
            0,
    )
    .unwrap();
    let maximum_window_tick_count = bridged_stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .maximum_tick_count;
    assert!(maximum_window_tick_count >= 3);
    bridged_stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(maximum_window_tick_count);
    let bridged_first = bridged_stream
        .submit_input_event(
            VulkanResidentTokenInputEvent::new("event", vec![1], 8).with_stop_tokens(vec![16]),
        )
        .unwrap();
    let bridged_second = bridged_stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "event_after_eos",
            vec![4],
            3,
        ))
        .unwrap();

    assert_eq!(
        resident_first.generated_token_ids,
        bridged_first.generated_token_ids
    );
    assert_eq!(resident_first.output_events, bridged_first.output_events);
    assert_eq!(
        resident_first.session_run.run.output_token_ids,
        bridged_first.session_run.run.output_token_ids
    );
    assert_eq!(
        resident_first.session_run.run.stop_reason,
        bridged_first.session_run.run.stop_reason
    );
    assert_eq!(
        resident_first.session_run.run.output_source_stream_ticks,
        bridged_first.session_run.run.output_source_stream_ticks
    );
    assert_eq!(
        resident_second.generated_token_ids,
        bridged_second.generated_token_ids
    );
    assert_eq!(resident_second.output_events, bridged_second.output_events);
    assert_eq!(
        resident_second.session_run.run.output_token_ids,
        bridged_second.session_run.run.output_token_ids
    );
    assert_eq!(
        resident_second.session_run.run.stop_reason,
        bridged_second.session_run.run.stop_reason
    );
    assert_eq!(
        resident_second.session_run.run.output_source_stream_ticks,
        bridged_second.session_run.run.output_source_stream_ticks
    );
    assert_eq!(resident_first.session_run.run.scheduler_turn_count, 2);
    assert_eq!(bridged_first.session_run.run.scheduler_turn_count, 4);
    assert_eq!(resident_second.session_run.run.scheduler_turn_count, 4);
    assert_eq!(bridged_second.session_run.run.scheduler_turn_count, 8);
    assert_eq!(
        resident_first.session_run.run.transport_stats,
        VulkanPlacedEdgeTransportStats::default()
    );
    assert_eq!(
        resident_second.session_run.run.transport_stats,
        VulkanPlacedEdgeTransportStats::default()
    );
    assert_eq!(
        bridged_first
            .session_run
            .run
            .transport_stats
            .direct_copy_count,
        4
    );
    assert_eq!(
        bridged_second
            .session_run
            .run
            .transport_stats
            .direct_copy_count,
        8
    );
    assert!(bridged_stream.is_idle());
}

#[test]
fn cross_physical_resident_edges_match_colocated_feedback_execution() {
    let Some((owner, peer)) = selected_test_vulkan_device_pair() else {
        eprintln!(
            "skipping cross-physical resident edge test without explicit owner and peer devices"
        );
        return;
    };
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let manifest = VulkanResidentModelPackageManifest::from_json_file(&manifest_path).unwrap();
    let runtime_model = manifest
        .mount_runtime_graph_controls(
            Some("gpu0"),
            &BTreeMap::from([("layer_00_repeat".to_string(), "gpu1".to_string())]),
            &[("layer_00".to_string(), "layer_00_repeat".to_string())],
            None,
        )
        .unwrap();
    let input = VulkanResidentTokenInputEvent::new("event", vec![1, 2, 3, 4, 5], 4);

    let mut colocated =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            BTreeMap::from([
                ("gpu0".to_string(), owner.clone()),
                ("gpu1".to_string(), owner.clone()),
            ]),
            manifest_dir,
            runtime_model.clone(),
            Some(8),
            0,
            0,
        )
        .unwrap();
    let colocated_run = colocated.submit_input_event(input.clone()).unwrap();
    drop(colocated);

    let mut split =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            BTreeMap::from([
                ("gpu0".to_string(), owner),
                ("gpu1".to_string(), peer),
            ]),
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();
    let split_gpu0_control = &split
        .processor
        .device("gpu0")
        .expect("split input device must be mounted")
        .mounted
        .stream_control_buffer;
    let split_gpu1_control = &split
        .processor
        .device("gpu1")
        .expect("split output device must be mounted")
        .mounted
        .stream_control_buffer;
    assert!(
        split_gpu0_control.shares_host_allocation_with(split_gpu1_control),
        "cross-device token/tick control is a tiny coherence-critical control plane and must use one shared-host allocation",
    );
    let split_run = split.submit_input_event(input).unwrap();

    assert_eq!(
        split.processor.temporal_block_executions.borrow().len(),
        1,
        "a multi-token split prompt must exercise the physical-device temporal pipeline",
    );

    assert_eq!(
        split_run.generated_token_ids,
        colocated_run.generated_token_ids
    );
    assert_eq!(
        split_run.session_run.run.stop_reason,
        colocated_run.session_run.run.stop_reason
    );
    assert_eq!(
        split_run.session_run.run.output_source_stream_ticks,
        colocated_run.session_run.run.output_source_stream_ticks
    );
    let cross_edge = split_run
        .session_run
        .run
        .transport_stats
        .edges
        .iter()
        .find(|edge| edge.key.from_device_id != edge.key.to_device_id)
        .expect("split execution must report its cross-device edge");
    assert!(matches!(
        cross_edge.route,
        VulkanPlacedEdgeTransferRoute::DeviceLocalStaging
            | VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal
            | VulkanPlacedEdgeTransferRoute::SharedHost
    ));
    assert!(cross_edge.queue_overlap_eligible);
    assert!(cross_edge.overlap_submission_count > 0);
    assert_eq!(cross_edge.host_wait_count, 0);
    assert!(
        split_run
            .session_run
            .run
            .resident_feedback
            .template_record_count
            > 0
    );
}

#[test]
fn explicit_internal_component_sharding_matches_canonical_execution() {
    let Some((owner, peer)) = selected_test_vulkan_device_pair() else {
        eprintln!(
            "skipping internal component shard test without explicit owner and peer devices"
        );
        return;
    };
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let canonical_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let sharded_model = canonical_model
        .clone()
        .with_component_shard_devices(
            "layer_00",
            vec!["gpu0".to_string(), "gpu1".to_string()],
        )
        .unwrap();
    let input = VulkanResidentTokenInputEvent::new("event", vec![1], 4);

    let mut canonical =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            BTreeMap::from([("gpu0".to_string(), owner.clone())]),
            manifest_dir,
            canonical_model,
            Some(8),
            0,
            0,
        )
        .unwrap();
    let canonical_run = canonical.submit_input_event(input.clone()).unwrap();
    drop(canonical);

    let mut sharded =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            BTreeMap::from([
                ("gpu0".to_string(), owner),
                ("gpu1".to_string(), peer),
            ]),
            manifest_dir,
            sharded_model,
            Some(8),
            0,
            0,
        )
        .unwrap();
    assert!(
        !sharded
            .package()
            .decode_distributed_execution_plan()
            .dispatches
            .is_empty()
    );
    let sharded_run = sharded.submit_input_event(input).unwrap();

    assert_eq!(
        sharded_run.generated_token_ids,
        canonical_run.generated_token_ids
    );
    assert_eq!(
        sharded_run.session_run.run.stop_reason,
        canonical_run.session_run.run.stop_reason
    );
    assert_eq!(
        sharded_run.session_run.run.output_source_stream_ticks,
        canonical_run.session_run.run.output_source_stream_ticks
    );
}

#[test]
fn placed_prompt_stream_reuses_every_recorded_feedback_window_shape() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID),
    );
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        Rc::new(device),
    )]);
    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            runtime_model,
            Some(4),
            0,
            0,
        )
        .unwrap();
    assert_eq!(
        stream
            .processor
            .resident_feedback_loop
            .as_ref()
            .unwrap()
            .window_policy
            .maximum_tick_count,
        4
    );
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(2);

    let first = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("first", vec![1], 5))
        .unwrap();
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(3);
    let second = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("second", vec![1], 5))
        .unwrap();
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(2);
    let third = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new("third", vec![1], 5))
        .unwrap();

    assert_eq!(
        first.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats {
            window_count: 2,
            planned_tick_count: 4,
            submitted_tick_count: 4,
            executed_tick_count: 4,
            retained_tick_count: 4,
            sampled_tick_count: 4,
            discarded_tick_count: 0,
            template_record_count: 1,
            template_replay_count: 1,
            asynchronous_submission_count: 0,
            completion_poll_count: 0,
            bounded_wait_count: 0,
            bounded_wait_timeout_count: 0,
        }
    );
    assert_eq!(second.session_run.run.resident_feedback.template_record_count, 1);
    assert_eq!(second.session_run.run.resident_feedback.template_replay_count, 0);
    assert_eq!(third.session_run.run.resident_feedback.template_record_count, 0);
    assert_eq!(third.session_run.run.resident_feedback.template_replay_count, 2);
    assert_eq!(stream.resident_feedback_template_catalog.len(), 2);

    stream.reset_transient_state().unwrap();
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(2);
    let after_reset = stream
        .submit_input_event(VulkanResidentTokenInputEvent::new(
            "after-reset",
            vec![1],
            5,
        ))
        .unwrap();
    assert_eq!(
        after_reset
            .session_run
            .run
            .resident_feedback
            .template_record_count,
        0
    );
    assert_eq!(
        after_reset
            .session_run
            .run
            .resident_feedback
            .template_replay_count,
        2
    );
}

#[test]
fn placed_prompt_stream_device_cancel_commits_one_closing_feedback_tick() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let manifest_path = fixture_model_package_manifest_path();
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        Rc::new(device),
    )]);
    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            fixture_model_runtime_model(),
            Some(8),
            0,
            0,
        )
        .unwrap();
    let maximum_window_tick_count = stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .maximum_tick_count;
    stream
        .processor
        .resident_feedback_loop
        .as_ref()
        .unwrap()
        .window_policy
        .next_tick_count
        .set(maximum_window_tick_count);
    stream.enqueue_input_event(VulkanResidentTokenInputEvent::new(
        "cancelled",
        vec![1],
        8,
    ));
    let initial = stream.run_next_activation().unwrap().unwrap();
    assert!(initial.output_event.is_some());

    stream
        .resident_feedback_cancellation_handle()
        .expect("fixture supports resident feedback")
        .request_cancel();
    let completed = stream.run_next_queued_input_event().unwrap().unwrap();

    assert_eq!(completed.session_run.run.stop_reason, "cancelled");
    assert_eq!(completed.generated_token_ids.len(), 2);
    assert_eq!(completed.session_run.tick_count, 3);
    assert_eq!(
        completed.session_run.run.resident_feedback,
        VulkanResidentFeedbackExecutionStats {
            window_count: 1,
            planned_tick_count: 7,
            submitted_tick_count: 7,
            executed_tick_count: 2,
            retained_tick_count: 2,
            sampled_tick_count: 1,
            discarded_tick_count: 5,
            template_record_count: 1,
            template_replay_count: 0,
            asynchronous_submission_count: 0,
            completion_poll_count: 0,
            bounded_wait_count: 0,
            bounded_wait_timeout_count: 0,
        }
    );
    assert!(stream.is_idle());
}

#[test]
fn placed_prompt_stream_queues_input_events_and_emits_output_events() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device.clone()),
    ]);

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    let queued_a =
        stream.enqueue_input_event(VulkanResidentTokenInputEvent::new("event_a", vec![1], 1));
    assert_eq!(queued_a.pending_input_event_count, 1);
    assert_eq!(queued_a.next_stream_tick, 0);
    let queued_b = stream.enqueue_input_event(VulkanResidentTokenInputEvent::new(
        "event_b",
        vec![4],
        1,
    ));
    assert_eq!(queued_b.pending_input_event_count, 2);
    assert_eq!(stream.pending_input_event_count(), 2);
    assert!(!stream.is_idle());

    let mut streamed_output_events = Vec::new();
    let first = stream
        .run_next_queued_input_event_with_output(|event| streamed_output_events.push(event))
        .unwrap()
        .unwrap();
    assert_eq!(first.input_event.id, "event_a");
    assert_eq!(first.pending_input_event_count, 1);
    assert_eq!(first.generated_token_ids.len(), 1);
    assert_eq!(first.output_events.len(), 1);
    assert_eq!(first.output_events[0].input_event_id, "event_a");
    assert_eq!(first.output_events[0].output_index, 0);
    assert_eq!(first.output_events[0].source_stream_tick, 0);
    assert_eq!(streamed_output_events, first.output_events);
    assert_eq!(stream.next_stream_tick(), 2);

    let second = stream.run_next_queued_input_event().unwrap().unwrap();
    assert_eq!(second.input_event.id, "event_b");
    assert_eq!(second.pending_input_event_count, 0);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(second.output_events.len(), 1);
    assert_eq!(second.output_events[0].input_event_id, "event_b");
    assert_eq!(second.output_events[0].source_stream_tick, 2);
    assert_eq!(stream.next_stream_tick(), 4);
    assert!(stream.is_idle());

    let idle = stream.run_next_queued_input_event().unwrap();
    assert!(idle.is_none());
}

#[test]
fn placed_prompt_stream_drains_queued_input_events_until_idle() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device.clone()),
    ]);

    let mut stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(8),
            0,
            0,
        )
        .unwrap();

    let run = stream
        .submit_input_events_until_idle(vec![
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
            VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
        ])
        .unwrap();

    assert_eq!(run.start_stream_tick, 0);
    assert_eq!(run.next_stream_tick, 4);
    assert_eq!(run.tick_count, 4);
    assert_eq!(run.submitted_runs.len(), 2);
    assert_eq!(run.output_events.len(), 2);
    assert_eq!(run.generated_token_ids.len(), 2);
    assert_eq!(run.pending_input_event_count, 0);
    assert_eq!(run.output_events[0].input_event_id, "event_a");
    assert_eq!(run.output_events[0].source_stream_tick, 0);
    assert_eq!(run.output_events[1].input_event_id, "event_b");
    assert_eq!(run.output_events[1].source_stream_tick, 2);
    assert!(stream.is_idle());
}
