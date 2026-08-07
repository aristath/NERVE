#[test]
fn placed_prompt_engine_owns_streams_and_submits_input_events() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        0,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let added = engine.add_stream("main", stream).unwrap();
    assert_eq!(added.stream_id, "main");
    assert_eq!(added.pending_input_event_count, 0);
    assert!(added.idle);
    assert_eq!(engine.snapshot().stream_count, 1);
    assert!(engine.snapshot().idle);

    let mut streamed_output_events = Vec::new();
    let submitted = engine
        .submit_input_event_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
            |event| streamed_output_events.push(event),
        )
        .unwrap();

    assert_eq!(submitted.stream_id, "main");
    assert_eq!(submitted.input_event_id, "event_a");
    assert_eq!(submitted.queued_input_event.stream_id, "main");
    assert_eq!(
        submitted
            .queued_input_event
            .queued_input_event
            .pending_input_event_count,
        1
    );
    assert_eq!(submitted.output_events.len(), 1);
    assert_eq!(submitted.output_events[0].stream_id, "main");
    assert_eq!(
        submitted.output_events[0].output_event.input_event_id,
        "event_a"
    );
    assert_eq!(
        submitted.output_events[0].output_event.source_stream_tick,
        0
    );
    assert_eq!(submitted.generated_token_ids.len(), 1);
    assert_eq!(streamed_output_events, submitted.output_events);
    assert_eq!(submitted.engine_run.processed_input_event_count, 1);
    assert_eq!(
        submitted.engine_run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(submitted.engine_run.input_runs.len(), 1);
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .input_event
            .id,
        "event_a"
    );
    assert_eq!(
        submitted.engine_run.end_snapshot.streams[0].next_stream_tick,
        2
    );

    let snapshot = engine.snapshot();
    assert!(snapshot.idle);
    assert_eq!(snapshot.streams[0].next_stream_tick, 2);
    assert_eq!(snapshot.streams[0].completed_prompt_event_count, 1);
    engine
        .enqueue_input_event(
            "main",
            VulkanResidentTokenInputEvent::new(
                "cancelled_by_shutdown",
                vec![2, 3],
                1,
            ),
        )
        .unwrap();
    let shutdown = engine.shutdown();
    assert!(shutdown.complete, "{:?}", shutdown.errors);
    assert_eq!(shutdown.stream_count, 1);
    assert_eq!(shutdown.package_count, 1);
    assert_eq!(shutdown.scheduler_in_flight_activation_count, 0);
    assert_eq!(shutdown.resource_teardowns.len(), 1);
    assert!(shutdown.resource_teardowns[0].complete);

    let baseline_file_descriptors =
        compiled_store_process_file_descriptor_count();
    let baseline_workers = compiled_store_worker_thread_count();
    for cycle_index in 0..3 {
        let runtime_model =
            tiny_fixture_model_runtime_model_with_placement(
                StreamCircuitPlacementSpec::new("gpu0"),
            );
        let stream =
            VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
                BTreeMap::from([("gpu0".to_string(), Rc::clone(&device))]),
                manifest_dir,
                runtime_model,
                Some(64),
                0,
                0,
            )
            .unwrap();
        let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
        engine
            .add_stream(format!("cycle-{cycle_index}"), stream)
            .unwrap();
        engine
            .enqueue_input_event(
                &format!("cycle-{cycle_index}"),
                VulkanResidentTokenInputEvent::new(
                    format!("pending-{cycle_index}"),
                    vec![1, 2],
                    2,
                ),
            )
            .unwrap();
        let shutdown = engine.shutdown();
        assert!(shutdown.complete, "{:?}", shutdown.errors);
        assert_eq!(shutdown.scheduler_in_flight_activation_count, 0);
        assert_eq!(shutdown.stream_count, 1);
        assert_eq!(shutdown.package_count, 1);
        assert_eq!(
            compiled_store_process_file_descriptor_count(),
            baseline_file_descriptors
        );
        assert_eq!(
            compiled_store_worker_thread_count(),
            baseline_workers
        );
    }
}

#[test]
fn placed_package_pool_reuses_immutable_parameters_across_graph_variants() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error)
            if std::env::var_os(
                "NERVE_TEST_VULKAN_DEVICE_INDEX",
            )
            .is_some() =>
        {
            panic!(
                "explicit Vulkan device for resident parameter pool was unavailable: {error}"
            )
        }
        Err(error) => {
            eprintln!(
                "skipping resident parameter pool test: {error}"
            );
            return;
        }
    };
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(
            &manifest_path,
        )
        .unwrap();
    let source = manifest.resolved_source_graph(manifest_dir).unwrap();
    let runtime_graph = manifest
        .runtime_graph_from_controls(
            Some("gpu0"),
            &BTreeMap::new(),
            &[],
            None,
        )
        .unwrap();
    let exact_model =
        manifest.clone().mount_runtime_graph(&runtime_graph).unwrap();
    let duplicated_model = manifest
        .mount_runtime_graph(
            &runtime_graph
                .duplicate_after_instance(
                    &source,
                    "layer_00",
                    "layer_00__duplicate",
                )
                .unwrap(),
        )
        .unwrap();
    let devices = BTreeMap::from([(
        "gpu0".to_string(),
        Rc::new(device),
    )]);
    let pool = VulkanResidentBufferPool::default();

    let exact =
        VulkanResidentInProcessPlacedModelPackage::
            from_runtime_model_for_bound_devices_with_parameter_pool(
                &devices,
                manifest_dir,
                exact_model,
                Some(64),
                0,
                ResourceResidencyPolicy::Eager,
                &pool,
            )
            .unwrap();
    let exact_stats = pool.stats();
    assert!(exact_stats.miss_count > 0);
    assert!(exact_stats.resident_bytes > 0);
    drop(exact);
    let stale_key = VulkanResidentBufferPoolKey::new(
        "nerve.test.stale_variant.v1",
        "gpu0",
        "stale_parameter",
        "0".repeat(64),
        0,
        4096,
    )
    .unwrap();
    let stale_buffer =
        pool.allocate_unpublished(&stale_key).unwrap();
    pool.publish(stale_key, stale_buffer.clone()).unwrap();
    drop(stale_buffer);

    let duplicated =
        VulkanResidentInProcessPlacedModelPackage::
            from_runtime_model_for_bound_devices_with_parameter_pool(
                &devices,
                manifest_dir,
                duplicated_model,
                Some(64),
                0,
                ResourceResidencyPolicy::Eager,
                &pool,
            )
            .unwrap();
    assert_eq!(pool.evict_unreferenced(), 1);
    let duplicated_stats = pool.stats();
    assert!(duplicated_stats.hit_count > exact_stats.hit_count);
    assert_eq!(
        duplicated_stats.miss_count,
        exact_stats.miss_count
    );
    assert_eq!(
        duplicated_stats.resident_bytes,
        exact_stats.resident_bytes
    );
    drop(duplicated);
    drop(pool);
    drop(devices);
}

#[test]
fn placed_prompt_engine_transaction_restores_the_resident_stream_in_place() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for resident stream transaction was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping resident stream transaction test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    let empty_stream_before = engine.snapshot().streams[0].clone();
    let empty_state_before = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("main")
        .unwrap();
    let empty_history_before = engine.stream_histories["main"].clone();
    let empty_transaction = engine
        .submit_input_event_transactionally_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("empty_branch", vec![6], 2),
            |_| VulkanResidentOutputControl::Continue,
        )
        .unwrap();
    assert_eq!(empty_transaction.generated_token_ids.len(), 2);
    assert_eq!(engine.snapshot().streams[0], empty_stream_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .stream_transient_state_snapshot("main")
            .unwrap(),
        empty_state_before,
    );
    assert_eq!(engine.stream_histories["main"], empty_history_before);

    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("canonical", vec![4, 5], 0),
        )
        .unwrap();
    let stream_before = engine.snapshot().streams[0].clone();
    let state_before = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("main")
        .unwrap();
    let history_before = engine.stream_histories["main"].clone();
    let arena_before = engine
        .runtime_scheduler
        .transient_state_arena_snapshot()
        .unwrap();

    let transactional = engine
        .submit_input_event_transactionally_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("branch", vec![6], 3),
            |_| VulkanResidentOutputControl::Continue,
        )
        .unwrap();

    assert_eq!(engine.snapshot().streams[0], stream_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .stream_transient_state_snapshot("main")
            .unwrap(),
        state_before
    );
    assert_eq!(engine.stream_histories["main"], history_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        arena_before.live_block_count
    );

    let mut observed = 0usize;
    let aborted = engine
        .submit_input_event_transactionally_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("aborted_branch", vec![6], 256),
            |_| {
                observed = observed.saturating_add(1);
                if observed >= 2 {
                    VulkanResidentOutputControl::Abort
                } else {
                    VulkanResidentOutputControl::Continue
                }
            },
        )
        .unwrap();
    assert!(observed >= 2);
    assert!(
        aborted.generated_token_ids.len() < 256,
        "abort callback observed {observed} token(s) after the engine had already generated {}",
        aborted.generated_token_ids.len(),
    );
    let aborted_input_run = aborted
        .engine_run
        .input_runs
        .iter()
        .find(|input_run| input_run.submitted_run.input_event.id == "aborted_branch")
        .expect("an output-requested abort must still return the submitted event run");
    assert_eq!(
        aborted_input_run.generated_token_ids,
        aborted.generated_token_ids
    );
    assert_eq!(aborted.engine_run.processed_input_event_count, 1);
    assert!(aborted.engine_run.end_snapshot.idle);
    assert_eq!(engine.snapshot().streams[0], stream_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .stream_transient_state_snapshot("main")
            .unwrap(),
        state_before,
    );
    assert_eq!(engine.stream_histories["main"], history_before);

    let committed = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("committed", vec![6], 3),
        )
        .unwrap();
    assert_eq!(
        transactional.generated_token_ids,
        committed.generated_token_ids
    );
}

#[test]
fn placed_prompt_engine_transaction_checkpoint_memory_is_independent_of_prefix_length() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for page-COW transaction test was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping page-COW transaction test: {error}");
            return;
        }
    };
    let mut runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    runtime_model.package.max_context_activations = 128;
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(128),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("short_prefix", vec![4, 5], 0),
        )
        .unwrap();

    let short = engine.begin_stream_transaction("main").unwrap();
    assert!(short.page_cow);
    let short_checkpoint_bytes = short.resident_state.as_ref().unwrap().byte_count;
    engine.restore_stream_transaction(short).unwrap();

    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("longer_prefix", vec![6; 32], 0),
        )
        .unwrap();
    let before = engine.stream_resident_state_digest("main").unwrap();
    let longer = engine.begin_stream_transaction("main").unwrap();
    assert!(longer.page_cow);
    let longer_checkpoint_bytes = longer.resident_state.as_ref().unwrap().byte_count;
    assert_eq!(
        longer_checkpoint_bytes, short_checkpoint_bytes,
        "transaction storage must contain fixed mutable state only, not accumulated append state",
    );
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("page_cow_branch", vec![7], 2),
        )
        .unwrap();
    engine.restore_stream_transaction(longer).unwrap();
    assert_eq!(engine.stream_resident_state_digest("main").unwrap(), before);
}

#[test]
fn placed_prompt_engine_nested_transactions_use_page_cow_without_full_state_snapshots() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for nested page-COW transaction test was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping nested page-COW transaction test: {error}");
            return;
        }
    };
    let mut runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    runtime_model.package.max_context_activations = 192;
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(192),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("canonical_prefix", vec![4, 5], 0),
        )
        .unwrap();

    let outer = engine.begin_stream_transaction("main").unwrap();
    assert!(outer.page_cow);
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("canonical_user", vec![6], 0),
        )
        .unwrap();
    let after_user = engine.stream_resident_state_digest("main").unwrap();
    let inner = engine.begin_stream_transaction("main").unwrap();
    assert!(inner.page_cow);
    assert_eq!(
        inner.resident_state.as_ref().unwrap().byte_count,
        outer.resident_state.as_ref().unwrap().byte_count,
    );
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("generation_branch", vec![7], 2),
        )
        .unwrap();
    engine.restore_stream_transaction(inner).unwrap();
    assert_eq!(engine.stream_resident_state_digest("main").unwrap(), after_user);
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("canonical_assistant", vec![8], 0),
        )
        .unwrap();
    engine.commit_stream_transaction(outer).unwrap();
    assert!(engine.active_transaction_depths.is_empty());
    assert_eq!(engine.stream("main").unwrap().transaction_page_cow_depth, 0);
}

#[cfg(feature = "tokenizers")]
#[derive(Clone, Copy)]
struct RejectingGeneratedChatCodec;

#[cfg(feature = "tokenizers")]
impl crate::VulkanResidentTokenTextCodec for RejectingGeneratedChatCodec {
    fn encode_text(
        &self,
        _: &str,
    ) -> Result<Vec<u32>, crate::VulkanResidentTokenTextCodecError> {
        Ok(vec![1])
    }

    fn decode_tokens(
        &self,
        _: &[u32],
    ) -> Result<String, crate::VulkanResidentTokenTextCodecError> {
        Err(crate::VulkanResidentTokenTextCodecError::new(
            "deliberately malformed generated protocol",
        ))
    }
}

#[cfg(feature = "tokenizers")]
#[derive(Clone, Copy)]
struct FixedGeneratedChatCodec;

#[cfg(feature = "tokenizers")]
impl crate::VulkanResidentTokenTextCodec for FixedGeneratedChatCodec {
    fn encode_text(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, crate::VulkanResidentTokenTextCodecError> {
        Ok(text
            .bytes()
            .map(|byte| u32::from(byte % 29).saturating_add(1))
            .collect())
    }

    fn decode_tokens(
        &self,
        _: &[u32],
    ) -> Result<String, crate::VulkanResidentTokenTextCodecError> {
        Ok("answer".to_string())
    }
}

#[cfg(feature = "tokenizers")]
#[test]
fn chat_generation_rejection_restores_state_and_allows_canonical_retry() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for rejected chat transaction was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping rejected chat transaction test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    let stream_before = engine.snapshot().streams[0].clone();
    let scheduler_before = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("main")
        .unwrap();
    let history_before = engine.stream_histories["main"].clone();
    let chat_session = crate::RuntimeChatSession {
        formatter: crate::RuntimeChatFormatter {
            template_source: String::new(),
            template_variables: serde_json::Map::new(),
            render_time: chrono::Local::now().fixed_offset(),
            compiled_codec: None,
        },
        messages: Vec::new(),
        committed_token_ids: Vec::new(),
    };
    let prepared = crate::RuntimePreparedChatTurn {
        canonical_user_token_ids: vec![4],
        user_token_delta: vec![4],
        generation_prompt_token_delta: vec![5],
    };
    let mut phases = Vec::new();

    let error = match crate::execute_vulkan_resident_chat_transaction(
        &mut engine,
        "main",
        &chat_session,
        &RejectingGeneratedChatCodec,
        &[],
        0,
        "first",
        &prepared,
        2,
        |_| Ok(crate::RuntimeChatGeneratedOutputControl::Continue),
        |phase, _| {
            phases.push(phase);
            Ok(())
        },
    ) {
        Ok(_) => panic!("malformed generated chat output was committed"),
        Err(error) => error,
    };

    assert!(
        error
            .downcast_ref::<crate::RuntimeRecoverableChatTurnError>()
            .is_some(),
        "unexpected chat transaction error: {error}",
    );
    assert_eq!(
        phases,
        vec![
            crate::VulkanResidentChatTransactionPhase::UserCommitted,
            crate::VulkanResidentChatTransactionPhase::GenerationBranchCompleted,
        ],
    );
    assert_eq!(engine.snapshot().streams[0], stream_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .stream_transient_state_snapshot("main")
            .unwrap(),
        scheduler_before,
    );
    assert_eq!(engine.stream_histories["main"], history_before);
    assert!(engine.active_transaction_depths.is_empty());

    let retry_session = crate::RuntimeChatSession {
        formatter: crate::RuntimeChatFormatter {
            template_source: "{%- for message in messages -%}{{- ('U' if message.role == 'user' else 'A') + message.content -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- 'A' -}}{%- endif -%}".to_string(),
            template_variables: serde_json::Map::new(),
            render_time: chrono::Local::now().fixed_offset(),
            compiled_codec: None,
        },
        messages: Vec::new(),
        committed_token_ids: Vec::new(),
    };
    let retry_prepared = retry_session
        .prepare_user_turn("first", &FixedGeneratedChatCodec)
        .unwrap();
    let mut retry_phases = Vec::new();
    let mut retry_output_count = 0usize;
    let retry = crate::execute_vulkan_resident_chat_transaction(
        &mut engine,
        "main",
        &retry_session,
        &FixedGeneratedChatCodec,
        &[],
        0,
        "first",
        &retry_prepared,
        2,
        |_| {
            retry_output_count = retry_output_count.saturating_add(1);
            Ok(if retry_output_count == 2 {
                crate::RuntimeChatGeneratedOutputControl::TerminateAndTrim { token_count: 1 }
            } else {
                crate::RuntimeChatGeneratedOutputControl::Continue
            })
        },
        |phase, _| {
            retry_phases.push(phase);
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(retry_output_count, 2);
    assert_eq!(retry.generation_run.generated_token_ids.len(), 2);
    assert_eq!(retry.generated_token_ids.len(), 1);
    assert!(retry.generation_terminated_by_protocol);
    let generation_input_run = retry
        .generation_run
        .engine_run
        .input_runs
        .iter()
        .find(|input_run| {
            input_run.submitted_run.input_event.id == retry.generation_event_id
        })
        .expect("protocol termination must retain the generation event report");
    assert_eq!(
        generation_input_run.generated_token_ids,
        retry.generation_run.generated_token_ids
    );
    assert_eq!(
        retry_phases,
        vec![
            crate::VulkanResidentChatTransactionPhase::UserCommitted,
            crate::VulkanResidentChatTransactionPhase::GenerationBranchCompleted,
            crate::VulkanResidentChatTransactionPhase::CanonicalTurnCommitted,
        ],
    );
    assert_eq!(
        retry.canonical_turn_token_delta,
        [
            retry_prepared.user_token_delta.as_slice(),
            retry.assistant_token_delta.as_slice(),
        ]
        .concat(),
    );
    assert_eq!(
        engine.stream_histories["main"].committed_state_token_ids,
        retry.canonical_turn_token_delta,
    );
    assert_eq!(
        engine.snapshot().streams[0].next_stream_tick,
        retry.canonical_turn_token_delta.len() as u64,
    );
}

#[test]
fn placed_prompt_engine_prefill_scheduler_tracks_partial_temporal_blocks_exactly() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for partial temporal prefill was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping partial temporal prefill test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let mut stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        7,
        0,
    )
    .unwrap();
    let prompt = (0..80)
        .map(|index| u32::try_from(index % 7 + 1).unwrap())
        .collect::<Vec<_>>();
    stream.enqueue_input_event(VulkanResidentTokenInputEvent::new(
        "partial-temporal-prefill",
        prompt.clone(),
        0,
    ));
    let scheduler_activation = RuntimeStreamActivation {
        id: 1,
        stream_id: "main".to_string(),
        execution_class_id: stream.package().stream_execution_class_id(),
        input_event_id: "partial-temporal-prefill".to_string(),
        kind: RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: prompt.clone(),
            remaining_prompt_token_count: 0,
        },
        max_state_activation_count: prompt.len(),
        state_reservations: Vec::new(),
    };

    let completed = stream
        .run_scheduled_prefill_chunk_with_output(&scheduler_activation, &prompt, &mut |_| {})
        .unwrap();

    assert_eq!(completed.unwrap().input_event.token_ids, prompt);
    assert_eq!(stream.next_stream_tick(), 80);
    assert!(stream.is_idle());

    stream.enqueue_input_event(VulkanResidentTokenInputEvent::new(
        "divergent-prefill",
        vec![4, 5, 6],
        0,
    ));
    let divergent_tokens = vec![4, 7, 6];
    let divergent_activation = RuntimeStreamActivation {
        id: 2,
        stream_id: "main".to_string(),
        execution_class_id: stream.package().stream_execution_class_id(),
        input_event_id: "divergent-prefill".to_string(),
        kind: RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: divergent_tokens.clone(),
            remaining_prompt_token_count: 0,
        },
        max_state_activation_count: divergent_tokens.len(),
        state_reservations: Vec::new(),
    };
    let error = stream
        .run_scheduled_prefill_chunk_with_output(
            &divergent_activation,
            &divergent_tokens,
            &mut |_| {},
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("diverged at external offset 1: scheduled=Some(7), backend=Some(5)"),
        "unexpected divergence diagnostic: {error}"
    );
    assert_eq!(stream.next_stream_tick(), 80);
}

#[test]
fn placed_prompt_engine_transaction_restores_after_backend_failure() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for failed stream transaction was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping failed resident stream transaction test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("canonical", vec![4], 0),
        )
        .unwrap();
    engine.streams.get_mut("main").unwrap().session.next_stream_tick = u64::MAX;
    let stream_before = engine.snapshot().streams[0].clone();
    let scheduler_before = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("main")
        .unwrap();

    let error = engine
        .submit_input_event_transactionally_until_idle_with_output(
            "main",
            VulkanResidentTokenInputEvent::new("failing_branch", vec![6], 0),
            |_| VulkanResidentOutputControl::Continue,
        )
        .unwrap_err();

    assert!(
        error.to_string().contains("tick overflow"),
        "unexpected transaction error: {error}"
    );
    assert_eq!(engine.snapshot().streams[0], stream_before);
    assert_eq!(
        engine
            .runtime_scheduler
            .stream_transient_state_snapshot("main")
            .unwrap(),
        scheduler_before
    );
    assert!(engine.snapshot().idle);
    assert!(engine.active_transaction_depths.is_empty());
}

#[test]
fn placed_prompt_engine_reuses_physical_state_pages_beyond_context_capacity() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for physical state paging was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping placed prompt engine context-wrap test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(4),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();

    let run = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("wrap", vec![4, 5, 6, 7, 8, 9], 1),
        )
        .unwrap();

    assert_eq!(run.generated_token_ids.len(), 1);
    assert_eq!(
        run.engine_run.end_snapshot.streams[0].next_stream_tick,
        7
    );
    let state = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("main")
        .unwrap();
    assert_eq!(state.logical_activation_count, 7);
    assert!(state.logical_activation_count > 4);
    assert_eq!(state.block_count, 1);
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        2
    );
    assert_eq!(engine.snapshot().prefix_state_cache.resident_entry_count, 1);

    engine.remove_stream("main").unwrap();
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        0
    );
    assert_eq!(engine.snapshot().prefix_state_cache.resident_entry_count, 0);
}

#[test]
fn placed_prompt_engine_restores_device_resident_prefix_pages_for_branches() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for resident prefix caching was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping resident prefix page cache test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let source = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model,
        Some(8),
        17,
        0,
    )
    .unwrap();
    let package = Arc::clone(&source.package);
    let branch_a =
        VulkanResidentInProcessPlacedPromptStream::new(package.clone(), devices.clone(), 29)
            .unwrap();
    let branch_b = VulkanResidentInProcessPlacedPromptStream::new(package, devices, 29).unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("source", source).unwrap();
    engine.add_stream("branch_a", branch_a).unwrap();
    engine.add_stream("branch_b", branch_b).unwrap();

    reset_vulkan_resident_execution_counters();
    engine
        .submit_input_event_until_idle(
            "source",
            VulkanResidentTokenInputEvent::new(
                "source_prompt",
                (4..24).collect::<Vec<_>>(),
                0,
            ),
        )
        .unwrap();
    let capture_counters = vulkan_resident_execution_counters();
    assert!(capture_counters.resident_copy_queue_submits > 0);
    assert_eq!(
        capture_counters.resident_copy_waits, 0,
        "prefix capture must not block the source stream"
    );
    let branch_tokens = (4..25).collect::<Vec<_>>();
    let branch_a_run = engine
        .submit_input_event_until_idle(
            "branch_a",
            VulkanResidentTokenInputEvent::new("branch_a_prompt", branch_tokens.clone(), 1),
        )
        .unwrap();
    assert!(
        vulkan_resident_execution_counters().resident_copy_waits
            > capture_counters.resident_copy_waits,
        "the first cache consumer must establish capture completion"
    );
    let branch_b_run = engine
        .submit_input_event_until_idle(
            "branch_b",
            VulkanResidentTokenInputEvent::new("branch_b_prompt", branch_tokens, 1),
        )
        .unwrap();

    assert_eq!(branch_a_run.queued_input_event.original_token_count, 21);
    assert_eq!(
        branch_a_run.queued_input_event.reused_prefix_token_count,
        16
    );
    assert_eq!(
        branch_b_run.queued_input_event.reused_prefix_token_count,
        16
    );
    assert_eq!(branch_a_run.engine_run.prefill_activation_count, 1);
    assert_eq!(branch_b_run.engine_run.prefill_activation_count, 1);
    assert_eq!(
        branch_a_run.generated_token_ids,
        branch_b_run.generated_token_ids
    );
    let stats = engine.snapshot().prefix_state_cache;
    assert_eq!(stats.hit_count, 2);
    assert_eq!(stats.miss_count, 1);
    assert_eq!(stats.reused_token_count, 32);
    assert_eq!(stats.saved_prefill_token_count, 32);
    assert_eq!(stats.insertion_count, 1);
    assert_eq!(stats.resident_entry_count, 1);
    assert!(stats.resident_byte_count > 0);

    let continued = engine
        .submit_input_event_until_idle(
            "branch_a",
            VulkanResidentTokenInputEvent::new("next_turn", vec![25, 26], 1),
        )
        .unwrap();
    assert_eq!(continued.queued_input_event.reused_prefix_token_count, 0);
    assert_eq!(continued.generated_token_ids.len(), 1);

    let source_state = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("source")
        .unwrap();
    let source_activation_count = source_state
        .entries
        .iter()
        .find(|entry| entry.shape.retention == TransientStateRetention::Append)
        .unwrap()
        .logical_activation_count;
    let advance_to_boundary = 8 - (source_activation_count % 8);
    engine
        .submit_input_event_until_idle(
            "source",
            VulkanResidentTokenInputEvent::new(
                "advance_source_checkpoint",
                vec![24; advance_to_boundary + 1],
                1,
            ),
        )
        .unwrap();
    let stats = engine.snapshot().prefix_state_cache;
    assert_eq!(stats.resident_entry_count, 1);
    assert_eq!(stats.insertion_count, 2);
    assert_eq!(stats.eviction_count, 1);
}

#[test]
fn placed_prompt_engine_reclaims_unused_state_after_cancelled_decode_window() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for cancellation rollback was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping early-stop state rollback test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stopped =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_dir,
            runtime_model,
            Some(16),
            41,
            0,
        )
        .unwrap();
    let feedback_loop = stopped
        .processor
        .resident_feedback_loop
        .as_ref()
        .expect("tiny fixture supports resident feedback");
    feedback_loop
        .window_policy
        .next_tick_count
        .set(feedback_loop.window_policy.maximum_tick_count);
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stopped", stopped).unwrap();
    engine
        .enqueue_input_event(
            "stopped",
            VulkanResidentTokenInputEvent::new("stopped_prompt", vec![4], 8),
        )
        .unwrap();
    engine
        .stream("stopped")
        .unwrap()
        .resident_feedback_cancellation_handle()
        .expect("tiny fixture supports resident feedback cancellation")
        .request_cancel();
    let stopped_run = engine.run_until_idle_bounded(1).unwrap();

    assert_eq!(stopped_run.generated_token_ids.len(), 2);
    assert_eq!(
        stopped_run
            .end_snapshot
            .streams
            .iter()
            .find(|stream| stream.stream_id == "stopped")
            .unwrap()
            .next_stream_tick,
        3
    );
    let state = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("stopped")
        .unwrap();
    assert!(state
        .entries
        .iter()
        .filter(|entry| entry.shape.retention == TransientStateRetention::Append)
        .all(|entry| entry.logical_activation_count == 3));
    let submitted = &stopped_run.input_runs[0].submitted_run;
    assert!(
        submitted.session_run.run.resident_feedback.planned_tick_count
            > submitted.session_run.run.resident_feedback.executed_tick_count
    );
    assert!(
        submitted.session_run.run.resident_feedback.discarded_tick_count > 0
    );
}

#[test]
fn placed_prompt_engine_fork_cow_reset_and_removal_are_physically_consistent() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine physical state lifecycle test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(64),
        7,
        0,
    )
    .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("parent", stream).unwrap();
    engine
        .submit_input_event_until_idle(
            "parent",
            VulkanResidentTokenInputEvent::new("seed", vec![4], 1),
        )
        .unwrap();

    let parent_before = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("parent")
        .unwrap();
    assert_eq!(parent_before.block_count, 1);
    let shared_block = parent_before.entries[0].block_ids[0];
    engine.fork_stream("parent", "child", 7).unwrap();
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .blocks
            .into_iter()
            .find(|block| block.block_id == shared_block)
            .unwrap()
            .ref_count,
        2,
    );

    let parent_run = engine
        .submit_input_event_until_idle(
            "parent",
            VulkanResidentTokenInputEvent::new("parent_next", vec![5], 1),
        )
        .unwrap();
    let child_run = engine
        .submit_input_event_until_idle(
            "child",
            VulkanResidentTokenInputEvent::new("child_next", vec![5], 1),
        )
        .unwrap();
    assert_eq!(parent_run.generated_token_ids, child_run.generated_token_ids);

    let parent_after = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("parent")
        .unwrap();
    let child_after = engine
        .runtime_scheduler
        .stream_transient_state_snapshot("child")
        .unwrap();
    assert_ne!(
        parent_after.entries[0].block_ids[0],
        child_after.entries[0].block_ids[0]
    );
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        2
    );

    engine.remove_stream("child").unwrap();
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        1
    );
    let zeroed = engine.reset_stream_transient_state("parent").unwrap();
    assert!(zeroed > 0);
    assert_eq!(engine.stream("parent").unwrap().next_stream_tick(), 0);
    assert_eq!(
        engine
            .runtime_scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .live_block_count,
        0
    );
}

#[test]
fn placed_prompt_engine_new_session_reset_restores_exact_initial_state() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!(
                "skipping placed prompt engine new-session reset test: {error}"
            );
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let devices =
        BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);
    let stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            runtime_model,
            Some(64),
            7,
            0,
        )
        .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    let initial = engine.stream_resident_state_digest("main").unwrap();
    let first = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("first", vec![4], 1),
        )
        .unwrap();
    assert_ne!(
        engine.stream_resident_state_digest("main").unwrap(),
        initial
    );

    let zeroed = engine.reset_stream_for_new_session("main", 7).unwrap();
    assert!(zeroed > 0);
    assert_eq!(
        engine.stream_resident_state_digest("main").unwrap(),
        initial
    );
    let replay = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("replay", vec![4], 1),
        )
        .unwrap();
    assert_eq!(replay.generated_token_ids, first.generated_token_ids);
}

#[test]
fn placed_prompt_engine_returns_completion_from_a_boundary_closing_drain() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine closing-drain test: {error}");
            return;
        }
    };
    let manifest_path = fixture_model_package_manifest_path();
    let devices = BTreeMap::from([(
        RUNTIME_DEFAULT_LOGICAL_DEVICE_ID.to_string(),
        Rc::new(device),
    )]);
    let stream =
        VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
            devices,
            manifest_path.parent().unwrap(),
            fixture_model_runtime_model(),
            Some(64),
            0,
            0,
        )
        .unwrap();
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();

    let submitted = engine
        .submit_input_event_until_idle(
            "main",
            VulkanResidentTokenInputEvent::new("boundary", vec![1], 3),
        )
        .unwrap();

    assert_eq!(submitted.generated_token_ids.len(), 3);
    assert_eq!(submitted.engine_run.input_runs.len(), 1);
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .session_run
            .run
            .stop_reason,
        "max_new_tokens"
    );
    assert!(submitted.engine_run.end_snapshot.idle);
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_single_submit_runs_the_engine_queue() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine single-submit queue test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            runtime_model,
            Some(64),
            0,
        )
        .unwrap(),
    );
    let stream_a =
        VulkanResidentInProcessPlacedPromptStream::new(model.clone(), devices.clone(), 0).unwrap();
    let stream_b =
        VulkanResidentInProcessPlacedPromptStream::new(model.clone(), devices, 1).unwrap();
    assert!(Arc::ptr_eq(&stream_a.package, &stream_b.package));
    assert!(Arc::ptr_eq(
        &stream_a.processor.device_slices[0]
            .package_slice
            .parameter_buffers,
        &stream_b.processor.device_slices[0]
            .package_slice
            .parameter_buffers,
    ));
    assert!(!std::ptr::eq(
        &stream_a.processor.device_slices[0]
            .mounted
            .buffers
            .state_buffers[0]
            .buffer,
        &stream_b.processor.device_slices[0]
            .mounted
            .buffers
            .state_buffers[0]
            .buffer,
    ));

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
        )
        .unwrap();

    let submitted = engine
        .submit_input_event_until_idle(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![5], 1),
        )
        .unwrap();

    assert_eq!(submitted.output_events.len(), 1);
    assert_eq!(submitted.output_events[0].stream_id, "stream_a");
    assert_eq!(submitted.engine_run.processed_input_event_count, 2);
    assert_eq!(submitted.engine_run.input_runs.len(), 2);
    assert!(
        submitted.engine_run.physical_multi_stream_batch_count > 0,
        "shared-package streams must execute as physical Vulkan batches"
    );
    assert_eq!(
        submitted.engine_run.max_physical_multi_stream_batch_width,
        2
    );
    assert_eq!(submitted.engine_run.input_runs[0].stream_id, "stream_b");
    assert_eq!(
        submitted.engine_run.input_runs[0]
            .submitted_run
            .input_event
            .id,
        "event_b"
    );
    assert_eq!(submitted.engine_run.input_runs[1].stream_id, "stream_a");
    assert_eq!(
        submitted.engine_run.input_runs[1]
            .submitted_run
            .input_event
            .id,
        "event_a"
    );
    assert!(submitted.engine_run.end_snapshot.idle);
}

#[test]
fn placed_prompt_engine_batches_fairly_and_cancels_between_physical_batches() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for physical batching was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping placed prompt engine batch cancellation test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device)]);
    let short_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            runtime_model.clone(),
            Some(64),
            0,
        )
        .unwrap(),
    );
    let long_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices(
            &devices,
            manifest_dir,
            runtime_model,
            Some(64),
            0,
        )
        .unwrap(),
    );
    assert!(!Arc::ptr_eq(&short_model, &long_model));
    assert_eq!(
        short_model.runtime_execution_identity,
        long_model.runtime_execution_identity
    );
    let short =
        VulkanResidentInProcessPlacedPromptStream::new(short_model, devices.clone(), 0).unwrap();
    let long = VulkanResidentInProcessPlacedPromptStream::new(long_model, devices, 1).unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("short", short).unwrap();
    engine.add_stream("long", long).unwrap();
    engine
        .enqueue_input_event(
            "short",
            VulkanResidentTokenInputEvent::new("short_event", vec![4], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "long",
            VulkanResidentTokenInputEvent::new("long_event", vec![5], 5),
        )
        .unwrap();

    let first_completion = engine.run_until_idle_bounded(1).unwrap();

    assert_eq!(first_completion.processed_input_event_count, 1);
    assert_eq!(first_completion.input_runs[0].stream_id, "short");
    assert!(first_completion.physical_multi_stream_batch_count > 0);
    assert_eq!(first_completion.max_physical_multi_stream_batch_width, 2);
    assert_eq!(
        first_completion
            .output_events
            .iter()
            .filter(|event| event.stream_id == "short")
            .count(),
        1
    );
    assert_eq!(
        first_completion
            .output_events
            .iter()
            .filter(|event| event.stream_id == "long")
            .count(),
        1
    );
    assert_eq!(first_completion.end_snapshot.active_stream_ids, ["long"]);

    let cancellation = engine.interrupt_stream("long", "test cancellation").unwrap();
    assert!(cancellation.stream_control_run.completed_input_run.is_some());
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_runs_queued_streams_until_idle() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine run-until-idle test: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device.clone()),
    ]);

    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(8),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b_repeat", vec![4], 1),
        )
        .unwrap();
    let queued_snapshot = engine.snapshot();
    assert!(!queued_snapshot.idle);
    assert_eq!(queued_snapshot.active_stream_count, 2);
    assert_eq!(
        queued_snapshot.active_stream_ids,
        vec!["stream_a".to_string(), "stream_b".to_string()]
    );

    let run = engine.run_until_idle_bounded(3).unwrap();

    assert_eq!(
        run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(run.processed_input_event_count, 3);
    assert_eq!(run.input_runs.len(), 3);
    assert_eq!(run.input_runs[0].stream_id, "stream_b");
    assert_eq!(run.input_runs[0].submitted_run.input_event.id, "event_b");
    assert_eq!(run.input_runs[1].stream_id, "stream_a");
    assert_eq!(run.input_runs[1].submitted_run.input_event.id, "event_a");
    assert_eq!(run.input_runs[2].stream_id, "stream_b");
    assert_eq!(
        run.input_runs[2].submitted_run.input_event.id,
        "event_b_repeat"
    );
    assert_eq!(run.output_events.len(), 3);
    assert_eq!(run.output_events[0].stream_id, "stream_b");
    assert_eq!(run.output_events[0].output_event.source_stream_tick, 0);
    assert_eq!(run.output_events[1].stream_id, "stream_a");
    assert_eq!(run.output_events[1].output_event.source_stream_tick, 0);
    assert_eq!(run.output_events[2].stream_id, "stream_b");
    assert_eq!(run.output_events[2].output_event.source_stream_tick, 2);
    assert_eq!(run.generated_token_ids.len(), 3);
    assert_eq!(run.physical_multi_stream_batch_count, 0);
    assert_eq!(run.max_physical_multi_stream_batch_width, 0);
    assert!(!run.start_snapshot.idle);
    assert!(run.end_snapshot.idle);
    assert_eq!(run.end_snapshot.active_stream_count, 0);
    assert_eq!(run.end_snapshot.streams[0].stream_id, "stream_a");
    assert_eq!(run.end_snapshot.streams[0].next_stream_tick, 2);
    assert_eq!(run.end_snapshot.streams[1].stream_id, "stream_b");
    assert_eq!(run.end_snapshot.streams[1].next_stream_tick, 4);
}

#[test]
fn placed_prompt_engine_batches_input_events_across_streams() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for multi-stream batching was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping placed prompt engine batch test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let devices = BTreeMap::from([("gpu0".to_string(), Rc::new(device))]);

    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(8),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();

    let batch = engine
        .submit_input_events_until_idle_bounded(
            vec![
                VulkanResidentInProcessPlacedPromptEngineInputRequest::new(
                    "stream_b",
                    VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
                ),
                VulkanResidentInProcessPlacedPromptEngineInputRequest::new(
                    "stream_a",
                    VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
                ),
            ],
            2,
        )
        .unwrap();

    assert_eq!(batch.queued_input_events.len(), 2);
    assert_eq!(batch.queued_input_events[0].stream_id, "stream_b");
    assert_eq!(batch.queued_input_events[1].stream_id, "stream_a");
    assert_eq!(
        batch.engine_run.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(batch.engine_run.input_runs.len(), 2);
    assert_eq!(batch.engine_run.processed_input_event_count, 2);
    assert_eq!(batch.engine_run.input_runs[0].stream_id, "stream_b");
    assert_eq!(
        batch.engine_run.input_runs[0].submitted_run.input_event.id,
        "event_b"
    );
    assert_eq!(batch.engine_run.input_runs[1].stream_id, "stream_a");
    assert_eq!(
        batch.engine_run.input_runs[1].submitted_run.input_event.id,
        "event_a"
    );
    assert_eq!(batch.output_events.len(), 2);
    assert_eq!(batch.output_events[0].stream_id, "stream_b");
    assert_eq!(batch.output_events[1].stream_id, "stream_a");
    assert_eq!(batch.generated_token_ids.len(), 2);
    assert!(batch.engine_run.physical_multi_stream_batch_count > 0);
    assert_eq!(batch.engine_run.max_physical_multi_stream_batch_width, 2);
    assert!(engine.snapshot().idle);
}

#[test]
fn placed_prompt_engine_overlaps_resident_feedback_windows_across_streams() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) if std::env::var_os("NERVE_TEST_VULKAN_DEVICE_INDEX").is_some() => {
            panic!("explicit Vulkan device for cross-stream overlap was unavailable: {error}")
        }
        Err(error) => {
            eprintln!("skipping placed prompt engine asynchronous feedback test: {error}");
            return;
        }
    };
    let runtime_model = tiny_fixture_model_runtime_model_with_placement(
        StreamCircuitPlacementSpec::new("gpu0"),
    );
    let manifest_path = tiny_fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([("gpu0".to_string(), device.clone())]);
    let stream_a = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices.clone(),
        manifest_dir,
        runtime_model.clone(),
        Some(64),
        0,
        0,
    )
    .unwrap();
    let stream_b = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        // A distinct capacity gives this stream a distinct execution identity,
        // deliberately exercising asynchronous overlap instead of the physical
        // batch path covered by the adjacent batching tests.
        Some(65),
        1,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("stream_a", stream_a).unwrap();
    engine.add_stream("stream_b", stream_b).unwrap();
    engine
        .enqueue_input_event(
            "stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 5),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![1], 5),
        )
        .unwrap();

    let run = engine.run_until_idle_bounded(2).unwrap();

    assert_eq!(run.processed_input_event_count, 2);
    assert_eq!(run.max_pending_activation_count, 2);
    for input_run in &run.input_runs {
        let feedback = input_run
            .submitted_run
            .session_run
            .run
            .resident_feedback;
        assert!(feedback.asynchronous_submission_count > 0);
        assert!(feedback.completion_poll_count > 0);
        assert!(feedback.bounded_wait_count > 0);
        assert_eq!(
            feedback.asynchronous_submission_count,
            feedback.window_count
        );
    }
    assert!(run.end_snapshot.idle);
}

#[test]
fn placed_prompt_engine_preserves_queued_work_at_input_event_budget() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed prompt engine budget test: {error}");
            return;
        }
    };
    let runtime_model = fixture_model_runtime_model_with_remote_middle();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let device = Rc::new(device);
    let devices = BTreeMap::from([
        ("gpu0".to_string(), device.clone()),
        ("gpu1".to_string(), device.clone()),
    ]);
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        devices,
        manifest_dir,
        runtime_model,
        Some(8),
        0,
        0,
    )
    .unwrap();

    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    engine.add_stream("main", stream).unwrap();
    engine
        .enqueue_input_event(
            "main",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 1),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "main",
            VulkanResidentTokenInputEvent::new("event_b", vec![4], 1),
        )
        .unwrap();

    let budgeted = engine.run_until_idle_bounded(1).unwrap();

    assert_eq!(
        budgeted.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::InputEventBudget
    );
    assert_eq!(budgeted.processed_input_event_count, 1);
    assert_eq!(budgeted.input_runs.len(), 1);
    assert_eq!(
        budgeted.input_runs[0].submitted_run.input_event.id,
        "event_a"
    );
    assert_eq!(budgeted.output_events.len(), 1);
    assert_eq!(
        budgeted.output_events[0].output_event.input_event_id,
        "event_a"
    );
    assert!(!budgeted.end_snapshot.idle);
    assert_eq!(
        budgeted.end_snapshot.streams[0].pending_input_event_count,
        1
    );
    assert_eq!(budgeted.end_snapshot.streams[0].next_stream_tick, 2);

    let completed_b = engine.run_until_idle_bounded(1).unwrap();
    assert_eq!(
        completed_b.stop_condition,
        VulkanResidentInProcessPlacedPromptEngineRunStopCondition::Idle
    );
    assert_eq!(completed_b.processed_input_event_count, 1);
    assert_eq!(
        completed_b.input_runs[0].submitted_run.input_event.id,
        "event_b"
    );
    assert_eq!(completed_b.output_events.len(), 1);
    assert_eq!(
        completed_b.output_events[0].output_event.input_event_id,
        "event_b"
    );
    assert!(completed_b.end_snapshot.idle);
    assert_eq!(completed_b.end_snapshot.streams[0].next_stream_tick, 4);
}

#[test]
fn placed_model_package_runs_runtime_graphed_duplicate_layer() {
    let device = match selected_test_vulkan_device() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping placed model package duplicate layer runtime graph: {error}");
            return;
        }
    };
    let manifest = fixture_model_package_manifest();
    let manifest_path = fixture_model_package_manifest_path();
    let manifest_dir = manifest_path.parent().unwrap();
    let source_graph = manifest
        .circuit_graph
        .to_resolved_lowered_execution_graph(manifest_dir)
        .unwrap();
    let layer_stage_count = source_graph
        .circuits
        .iter()
        .find(|circuit| circuit.component.id == "layer_00")
        .expect("fixture source graph contains layer_00")
        .circuit
        .nodes
        .len();
    let runtime_graph = StreamCircuitRuntimeGraph::from_source_series(&source_graph, "gpu0")
        .unwrap()
        .duplicate_after_instance(&source_graph, "layer_00", "layer_00_repeat")
        .unwrap()
        .with_instance_device("layer_00_repeat", "gpu1")
        .unwrap();
    let runtime_model = manifest.mount_runtime_graph(&runtime_graph).unwrap();

    let placed_model = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_devices(
            &device,
            manifest_dir,
            runtime_model,
            Some(4),
        )
        .unwrap(),
    );
    let placed_package = placed_model
        .create_stream_processor_for_devices(&device, 0)
        .unwrap();
    assert_eq!(placed_model.device_ids, vec!["gpu0", "gpu1"]);
    assert_eq!(placed_model.device_count, 2);
    assert_eq!(placed_model.hosted_component_count, 2);
    assert_eq!(placed_package.device("gpu1").unwrap().hosted_component_count, 1);

    let run = placed_package
        .sample_token_id_stream_tick_in_process(&device, 1, 0)
        .unwrap();

    assert_eq!(
        run.tick_run.placed_run.status,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed
    );
    // Every source-layer kernel runs in both instances, plus the two
    // logical-device boundary stages introduced by the duplicated wiring.
    assert_eq!(
        run.tick_run.placed_run.completed_stage_delta,
        layer_stage_count * 2 + 2
    );
    assert_eq!(run.sampler_run.descriptor_count, 5);
}
