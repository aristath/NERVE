#[derive(Clone, Debug, PartialEq, Eq)]
struct TestResidencyContinuation {
    activation_id: u64,
    stream_sequence: u64,
    recurrent_state_epoch: u64,
    attention_state_epoch: u64,
    random_state: [u32; 4],
    transient_slots: Vec<u64>,
}

fn test_residency_continuation(activation_id: u64) -> TestResidencyContinuation {
    TestResidencyContinuation {
        activation_id,
        stream_sequence: activation_id * 10,
        recurrent_state_epoch: activation_id * 100,
        attention_state_epoch: activation_id * 1_000,
        random_state: [
            activation_id as u32,
            0x1234_5678,
            0x9abc_def0,
            0x0fed_cba9,
        ],
        transient_slots: vec![activation_id * 2, activation_id * 2 + 1],
    }
}

fn test_residency_checkpoint() -> VulkanPhysicalResidencyCheckpoint {
    VulkanPhysicalResidencyCheckpoint {
        id: "checkpoint".to_string(),
        execution_scope: "target".to_string(),
        component_id: "component".to_string(),
        selector_ids: vec!["selector".to_string()],
        selection_dispatch_index: 7,
        selected_computation_dispatch_indices: vec![8, 9],
        selected_result_continuation_dispatch_index: Some(10),
    }
}

fn test_backpressure_limits() -> VulkanResidencyBackpressureLimits {
    VulkanResidencyBackpressureLimits {
        maximum_owned_activations: 8,
        maximum_groups_per_activation: 4,
        maximum_outstanding_loads: 32,
    }
}

fn test_activation_key(
    activation_id: u64,
    stream_id: &str,
) -> VulkanResidencyCheckpointActivationKey {
    VulkanResidencyCheckpointActivationKey::new(
        activation_id,
        stream_id,
        "gpu0",
        3,
    )
    .unwrap()
}

fn preload_test_group(
    manager: &DeviceResourceResidencyManager<TestResidentPayload>,
    descriptor: &DeviceResourceGroupDescriptor,
    drops: &SyncArc<AtomicUsize>,
) {
    let permit = match manager
        .request(descriptor.clone(), owner("model"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("test preload did not own the group load"),
    };
    drop(
        permit
            .publish(resident_test_group(
                descriptor.clone(),
                SyncArc::clone(drops),
            ))
            .unwrap(),
    );
}

#[test]
fn residency_backpressure_interleaves_hits_and_shared_misses_without_replay() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 16_384, 512,
    )
    .unwrap();
    let hit = residency_descriptor('1', '2', 64);
    let miss = residency_descriptor('3', '4', 96);
    let drops = SyncArc::new(AtomicUsize::new(0));
    preload_test_group(&manager, &hit, &drops);
    let checkpoint = test_residency_checkpoint();
    let mut scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();

    let warm_continuation = test_residency_continuation(10);
    assert_eq!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(10, "warm"),
                &checkpoint,
                vec![hit.clone()],
                warm_continuation.clone(),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Ready { activation_id: 10 }
    );
    let first_cold_continuation = test_residency_continuation(12);
    assert_eq!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(12, "cold-a"),
                &checkpoint,
                vec![miss.clone()],
                first_cold_continuation.clone(),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Blocked {
            activation_id: 12,
            missing_group_ids: vec![miss.id.clone()],
            new_load_count: 1,
            joined_load_count: 0,
        }
    );
    let second_cold_continuation = test_residency_continuation(11);
    assert_eq!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(11, "cold-b"),
                &checkpoint,
                vec![miss.clone()],
                second_cold_continuation.clone(),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Blocked {
            activation_id: 11,
            missing_group_ids: vec![miss.id.clone()],
            new_load_count: 0,
            joined_load_count: 1,
        }
    );

    let warm = scheduler.pop_ready().unwrap();
    assert_eq!(warm.key().activation_id, 10);
    assert_eq!(warm.continuation(), &warm_continuation);
    assert_eq!(warm.resident_group_ids(), [hit.id.as_str()]);
    assert_eq!(
        warm.checkpoint_trace()
            .iter()
            .map(|entry| entry.responsibility)
            .collect::<Vec<_>>(),
        [
            VulkanPhysicalResidencyResponsibility::Selection,
            VulkanPhysicalResidencyResponsibility::Availability,
            VulkanPhysicalResidencyResponsibility::SelectedComputation,
            VulkanPhysicalResidencyResponsibility::SelectedResultContinuation,
        ]
    );

    let load = scheduler.pop_load_command().unwrap();
    assert_eq!(load.device_id(), "gpu0");
    assert_eq!(load.group_id(), miss.id);
    assert!(scheduler.pop_load_command().is_none());
    load.publish(resident_test_group(
        miss.clone(),
        SyncArc::clone(&drops),
    ))
    .unwrap();
    assert_eq!(scheduler.poll_load_completions().unwrap(), 1);

    let first_cold = scheduler.pop_ready().unwrap();
    let second_cold = scheduler.pop_ready().unwrap();
    assert_eq!(first_cold.key().activation_id, 12);
    assert_eq!(second_cold.key().activation_id, 11);
    assert_eq!(first_cold.continuation(), &first_cold_continuation);
    assert_eq!(second_cold.continuation(), &second_cold_continuation);
    assert_eq!(
        first_cold.checkpoint_trace()[..2]
            .iter()
            .map(|entry| entry.responsibility)
            .collect::<Vec<_>>(),
        [
            VulkanPhysicalResidencyResponsibility::Selection,
            VulkanPhysicalResidencyResponsibility::Availability,
        ]
    );
    assert_eq!(
        first_cold.checkpoint_trace()[2..]
            .iter()
            .map(|entry| entry.responsibility)
            .collect::<Vec<_>>(),
        [
            VulkanPhysicalResidencyResponsibility::SelectedComputation,
            VulkanPhysicalResidencyResponsibility::SelectedResultContinuation,
        ]
    );

    let repeat = test_residency_continuation(13);
    assert!(matches!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(13, "warm-repeat"),
                &checkpoint,
                vec![miss],
                repeat,
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Ready { activation_id: 13 }
    ));
    assert!(scheduler.pop_load_command().is_none());
    assert_eq!(
        manager.statistics().unwrap().successful_load_count,
        2,
        "the shared miss must produce exactly one new publication"
    );
}

#[test]
fn residency_backpressure_cancellation_preserves_state_and_does_not_cancel_shared_work() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 16_384, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('5', '6', 128);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let checkpoint = test_residency_checkpoint();
    let mut scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();
    let cancelled_state = test_residency_continuation(20);
    let surviving_state = test_residency_continuation(21);

    scheduler
        .admit_checkpoint(
            &manager,
            owner("model"),
            test_activation_key(20, "cancelled"),
            &checkpoint,
            vec![descriptor.clone()],
            cancelled_state.clone(),
        )
        .unwrap();
    scheduler
        .admit_checkpoint(
            &manager,
            owner("model"),
            test_activation_key(21, "surviving"),
            &checkpoint,
            vec![descriptor.clone()],
            surviving_state.clone(),
        )
        .unwrap();
    let cancelled = scheduler.cancel_stream("cancelled").unwrap();
    assert_eq!(cancelled.key.activation_id, 20);
    assert_eq!(cancelled.continuation, cancelled_state);

    scheduler
        .pop_load_command()
        .unwrap()
        .publish(resident_test_group(descriptor, drops))
        .unwrap();
    scheduler.poll_load_completions().unwrap();
    let ready = scheduler.pop_ready().unwrap();
    assert_eq!(ready.key().activation_id, 21);
    assert_eq!(ready.continuation(), &surviving_state);
    assert!(scheduler.pop_ready().is_none());
    assert!(scheduler.pop_failed().is_none());
    assert_eq!(
        scheduler.snapshot(),
        VulkanResidencyBackpressureSnapshot::default()
    );
}

#[test]
fn residency_backpressure_load_failure_wakes_every_dependent_activation_in_order() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 16_384, 512,
    )
    .unwrap();
    let descriptor = residency_descriptor('7', '8', 128);
    let checkpoint = test_residency_checkpoint();
    let mut scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();
    let first_state = test_residency_continuation(31);
    let second_state = test_residency_continuation(30);
    for (activation_id, stream_id, continuation) in [
        (31, "first", first_state.clone()),
        (30, "second", second_state.clone()),
    ] {
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(activation_id, stream_id),
                &checkpoint,
                vec![descriptor.clone()],
                continuation,
            )
            .unwrap();
    }

    scheduler
        .pop_load_command()
        .unwrap()
        .fail(DeviceResourceResidencyError::new(
            DeviceResourceResidencyErrorKind::Failed,
            "injected read failure",
        ))
        .unwrap();
    scheduler.poll_load_completions().unwrap();
    let first = scheduler.pop_failed().unwrap();
    let second = scheduler.pop_failed().unwrap();
    assert_eq!(first.key().activation_id, 31);
    assert_eq!(second.key().activation_id, 30);
    assert_eq!(first.continuation(), &first_state);
    assert_eq!(second.continuation(), &second_state);
    assert_eq!(first.error().kind(), VulkanResidencyBackpressureErrorKind::Load);
    assert_eq!(second.error().kind(), VulkanResidencyBackpressureErrorKind::Load);
    assert!(scheduler.pop_ready().is_none());
    assert_eq!(manager.statistics().unwrap().failed_group_count, 1);
}

#[test]
fn residency_pause_keeps_the_runtime_activation_and_transient_reservation_in_flight() {
    let mut runtime_scheduler = RuntimeStreamScheduler::new();
    runtime_scheduler
        .add_stream_with_state_declarations(
            "stateful",
            [(
                TransientStateKey::new("component", "temporal_memory"),
                TransientStateBlockShape::new(32, 4).unwrap(),
            )],
        )
        .unwrap();
    runtime_scheduler
        .enqueue_input_event(
            "stateful",
            RuntimeStreamInputEvent::new("event", [1], 0),
        )
        .unwrap();
    let activation = runtime_scheduler
        .schedule_step(RuntimeStreamSchedulerBudget::new(1, 1, 1))
        .unwrap()
        .activations
        .into_iter()
        .next()
        .unwrap();
    let state_before_pause = runtime_scheduler
        .stream_transient_state_snapshot("stateful")
        .unwrap();

    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 0,
    )
    .unwrap();
    let descriptor = residency_descriptor('d', 'e', 64);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let mut residency_scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();
    residency_scheduler
        .admit_checkpoint(
            &manager,
            owner("model"),
            test_activation_key(activation.id, "stateful"),
            &test_residency_checkpoint(),
            vec![descriptor.clone()],
            activation.clone(),
        )
        .unwrap();

    assert_eq!(
        runtime_scheduler.snapshot().in_flight_activation_count,
        1
    );
    assert_eq!(
        runtime_scheduler
            .stream_transient_state_snapshot("stateful")
            .unwrap(),
        state_before_pause
    );
    residency_scheduler
        .pop_load_command()
        .unwrap()
        .publish(resident_test_group(descriptor, drops))
        .unwrap();
    residency_scheduler.poll_load_completions().unwrap();
    let resumed = residency_scheduler.pop_ready().unwrap();
    assert_eq!(resumed.continuation(), &activation);
    assert_eq!(
        runtime_scheduler
            .stream_transient_state_snapshot("stateful")
            .unwrap(),
        state_before_pause
    );

    runtime_scheduler
        .complete_activation(
            resumed.into_continuation().id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();
    assert_eq!(
        runtime_scheduler.snapshot().in_flight_activation_count,
        0
    );
}

#[test]
fn residency_backpressure_observes_a_shared_manager_load_started_elsewhere() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 0,
    )
    .unwrap();
    let descriptor = residency_descriptor('f', '0', 64);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let external_permit = match manager
        .request(descriptor.clone(), owner("external-graph"))
        .unwrap()
    {
        DeviceResourceResidencyRequest::LoadRequired(permit) => permit,
        _ => panic!("external requester did not own the load"),
    };
    let continuation = test_residency_continuation(35);
    let mut scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();
    assert!(matches!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(35, "shared-manager"),
                &test_residency_checkpoint(),
                vec![descriptor.clone()],
                continuation.clone(),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Blocked {
            new_load_count: 0,
            joined_load_count: 1,
            ..
        }
    ));
    assert!(scheduler.pop_load_command().is_none());
    assert_eq!(scheduler.poll_load_completions().unwrap(), 0);

    drop(
        external_permit
            .publish(resident_test_group(descriptor, drops))
            .unwrap(),
    );
    assert_eq!(scheduler.poll_load_completions().unwrap(), 1);
    let ready = scheduler.pop_ready().unwrap();
    assert_eq!(ready.continuation(), &continuation);
}

#[test]
fn residency_backpressure_resumes_only_after_every_group_is_atomically_published() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 0,
    )
    .unwrap();
    let first = residency_descriptor('1', '5', 64);
    let second = residency_descriptor('2', '6', 96);
    let drops = SyncArc::new(AtomicUsize::new(0));
    let continuation = test_residency_continuation(36);
    let mut scheduler =
        VulkanResidencyBackpressureScheduler::new(test_backpressure_limits())
            .unwrap();
    assert_eq!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(36, "multi-group"),
                &test_residency_checkpoint(),
                vec![first.clone(), second.clone()],
                continuation.clone(),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Blocked {
            activation_id: 36,
            missing_group_ids: vec![first.id.clone(), second.id.clone()],
            new_load_count: 2,
            joined_load_count: 0,
        }
    );
    let first_load = scheduler.pop_load_command().unwrap();
    let second_load = scheduler.pop_load_command().unwrap();

    second_load
        .publish(resident_test_group(
            second.clone(),
            SyncArc::clone(&drops),
        ))
        .unwrap();
    assert_eq!(scheduler.poll_load_completions().unwrap(), 1);
    assert!(scheduler.pop_ready().is_none());
    assert_eq!(scheduler.snapshot().blocked_activation_count, 1);

    first_load
        .publish(resident_test_group(first.clone(), drops))
        .unwrap();
    assert_eq!(scheduler.poll_load_completions().unwrap(), 1);
    let ready = scheduler.pop_ready().unwrap();
    assert_eq!(ready.continuation(), &continuation);
    assert_eq!(
        ready.resident_group_ids(),
        [first.id.as_str(), second.id.as_str()]
    );
}

#[test]
fn residency_batch_capacity_failure_is_atomic_and_scheduler_queues_are_bounded() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 200, 20,
    )
    .unwrap();
    let first = residency_descriptor('9', 'a', 100);
    let second = residency_descriptor('b', 'c', 100);
    let error = manager
        .request_batch([first.clone(), second.clone()], owner("model"))
        .err()
        .unwrap();
    assert_eq!(error.kind(), DeviceResourceResidencyErrorKind::Capacity);
    assert!(manager.directory().unwrap().is_empty());
    assert_eq!(manager.statistics().unwrap().reserved_loading_bytes, 0);

    let mut scheduler = VulkanResidencyBackpressureScheduler::<
        TestResidencyContinuation,
        TestResidentPayload,
    >::new(VulkanResidencyBackpressureLimits {
        maximum_owned_activations: 1,
        maximum_groups_per_activation: 1,
        maximum_outstanding_loads: 1,
    })
    .unwrap();
    let fitting_manager =
        DeviceResourceResidencyManager::new("gpu0", 4096, 0).unwrap();
    scheduler
        .admit_checkpoint(
            &fitting_manager,
            owner("model"),
            test_activation_key(40, "bounded"),
            &test_residency_checkpoint(),
            vec![first],
            test_residency_continuation(40),
        )
        .unwrap();
    let queue_error = scheduler
        .admit_checkpoint(
            &fitting_manager,
            owner("model"),
            test_activation_key(41, "overflow"),
            &test_residency_checkpoint(),
            vec![second],
            test_residency_continuation(41),
        )
        .unwrap_err();
    assert_eq!(
        queue_error.kind(),
        VulkanResidencyBackpressureErrorKind::QueueFull
    );
    assert_eq!(fitting_manager.directory().unwrap().len(), 1);
}

#[test]
fn residency_load_backpressure_does_not_reject_a_resident_hit() {
    let manager = DeviceResourceResidencyManager::<TestResidentPayload>::new(
        "gpu0", 4096, 0,
    )
    .unwrap();
    let miss = residency_descriptor('1', '3', 64);
    let hit = residency_descriptor('2', '4', 64);
    let drops = SyncArc::new(AtomicUsize::new(0));
    preload_test_group(&manager, &hit, &drops);
    let mut scheduler = VulkanResidencyBackpressureScheduler::new(
        VulkanResidencyBackpressureLimits {
            maximum_owned_activations: 2,
            maximum_groups_per_activation: 1,
            maximum_outstanding_loads: 1,
        },
    )
    .unwrap();
    scheduler
        .admit_checkpoint(
            &manager,
            owner("model"),
            test_activation_key(50, "cold"),
            &test_residency_checkpoint(),
            vec![miss],
            test_residency_continuation(50),
        )
        .unwrap();
    assert_eq!(scheduler.snapshot().outstanding_load_count, 1);

    assert_eq!(
        scheduler
            .admit_checkpoint(
                &manager,
                owner("model"),
                test_activation_key(51, "warm"),
                &test_residency_checkpoint(),
                vec![hit],
                test_residency_continuation(51),
            )
            .unwrap(),
        VulkanResidencyCheckpointAdmission::Ready { activation_id: 51 }
    );
    assert_eq!(scheduler.pop_ready().unwrap().key().activation_id, 51);
}
