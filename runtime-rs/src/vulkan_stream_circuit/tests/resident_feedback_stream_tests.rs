fn create_fixture_model_resident_greedy_stream_processor(
    device: &VulkanComputeDevice,
    _label: &str,
) -> Option<VulkanResidentStreamProcessor> {
    create_fixture_model_resident_greedy_stream_processor_with_capacity(device, _label, 4, "")
}

fn create_fixture_model_resident_greedy_stream_processor_with_capacity(
    device: &VulkanComputeDevice,
    _label: &str,
    dynamic_state_capacity_activations: usize,
    _obsolete_attention_shader: &str,
) -> Option<VulkanResidentStreamProcessor> {
    Some(
        fixture_model_resident_greedy_model(device, dynamic_state_capacity_activations)
        .unwrap()
        .create_stream_processor(device, 0)
        .unwrap(),
    )
}

#[test]
fn resident_greedy_feedback_loop_runs_two_ticks() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor =
        create_fixture_model_resident_greedy_stream_processor(&device, "feedback").unwrap();
    assert_eq!(processor.device_id, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    assert_eq!(processor.component_count, 1);
    assert_eq!(processor.per_tick_dispatch_count, 13);
    assert!(processor.per_tick_descriptor_count > processor.per_tick_dispatch_count);
    assert_eq!(processor.per_tick_push_constant_byte_count, 0);
    assert_eq!(processor.dynamic_state_capacity_activations, 4);

    let run = processor.run_bounded(&device, 1, 0, 2).unwrap();
    assert_eq!(run.device_id, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    assert_eq!(run.initial_token_id, 1);
    assert_eq!(run.tick_runs.len(), 2);
    assert_eq!(run.per_tick_dispatch_count, processor.per_tick_dispatch_count);
    assert_eq!(
        run.per_tick_descriptor_count,
        processor.per_tick_descriptor_count
    );
    assert_eq!(run.per_tick_push_constant_byte_count, 0);
    assert_eq!(run.tick_runs[0].stream_tick, 0);
    assert_eq!(run.tick_runs[0].input_token_id, 1);
    assert_eq!(run.tick_runs[1].stream_tick, 1);
    assert_eq!(
        run.tick_runs[1].input_token_id,
        run.tick_runs[0].sampled_token_id
    );
    assert_eq!(run.tick_runs[0].tick_run.dispatch_count, 12);
    assert_eq!(run.tick_runs[0].sampler_run.descriptor_count, 5);
    assert_eq!(run.tick_runs[1].tick_run.dispatch_count, 12);
    assert_eq!(run.tick_runs[1].sampler_run.descriptor_count, 5);
    assert_eq!(run.sampled_token_ids, vec![16, 16]);
    assert_eq!(run.tick_runs[0].sampler_run.token_id, 16);
    assert_eq!(run.tick_runs[1].sampler_run.token_id, 16);
    for (actual, expected) in run
        .tick_runs
        .iter()
        .map(|tick| tick.sampler_run.selected_logit_bits)
        .zip([1_067_658_104, 1_077_467_248])
    {
        assert_f32_bits_close(actual, expected, 0.01, 0.01);
    }
}

#[test]
fn resident_greedy_prompt_event_drains_external_input_before_feedback() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor =
        create_fixture_model_resident_greedy_stream_processor(&device, "prompt event").unwrap();

    let run = processor
        .run_prompt_event_bounded(&device, &[1, 4], 0, 1, None)
        .unwrap();

    assert_eq!(run.device_id, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    assert_eq!(run.prompt_token_ids, vec![1, 4]);
    assert_eq!(run.generated_token_ids.len(), 1);
    assert_eq!(
        run.output_token_ids,
        vec![1, 4, run.generated_token_ids[0]]
    );
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.tick_runs.len(), 3);
    assert_eq!(run.per_tick_dispatch_count, 13);
    assert!(run.per_tick_descriptor_count > run.per_tick_dispatch_count);
    assert_eq!(run.per_tick_push_constant_byte_count, 0);

    assert_eq!(run.tick_runs[0].stream_tick, 0);
    assert_eq!(run.tick_runs[0].input_token_id, 1);
    assert_eq!(
        run.tick_runs[0].input_route,
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(run.tick_runs[0].public_output_token_id, None);
    assert_eq!(run.tick_runs[0].private_feedback_token_id, None);
    assert!(run.tick_runs[0].sampler_run.is_none());
    assert_eq!(run.tick_runs[0].tick_run.dispatch_count, 10);
    assert!(run.tick_runs[0].tick_run.output_run.is_none());

    assert_eq!(run.tick_runs[1].stream_tick, 1);
    assert_eq!(run.tick_runs[1].input_token_id, 4);
    assert_eq!(
        run.tick_runs[1].input_route,
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(
        run.tick_runs[1].public_output_token_id,
        Some(run.generated_token_ids[0])
    );
    assert_eq!(
        run.tick_runs[1].private_feedback_token_id,
        Some(run.generated_token_ids[0])
    );
    assert_eq!(
        run.tick_runs[1].private_feedback_closes_loop_after_processing,
        Some(true)
    );
    assert_eq!(
        run.tick_runs[1].sampler_run.as_ref().unwrap().token_id,
        run.generated_token_ids[0]
    );
    assert_eq!(run.tick_runs[1].tick_run.dispatch_count, 12);
    assert!(run.tick_runs[1].tick_run.output_run.is_some());

    assert_eq!(run.tick_runs[2].stream_tick, 2);
    assert_eq!(run.tick_runs[2].input_token_id, run.generated_token_ids[0]);
    assert_eq!(
        run.tick_runs[2].input_route,
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(run.tick_runs[2].input_feedback_depth, 1);
    assert!(run.tick_runs[2].input_closes_loop_after_processing);
    assert_eq!(run.tick_runs[2].public_output_token_id, None);
    assert_eq!(run.tick_runs[2].private_feedback_token_id, None);
    assert!(run.tick_runs[2].sampler_run.is_none());
    assert_eq!(run.tick_runs[2].tick_run.dispatch_count, 10);
    assert!(run.tick_runs[2].tick_run.output_run.is_none());
}

#[test]
fn resident_greedy_running_stream_accepts_later_input_without_resetting_state() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor =
        create_fixture_model_resident_greedy_stream_processor(&device, "running stream").unwrap();
    let mut stream = processor.into_running_stream("stream_0");
    assert_eq!(stream.stream_id, "stream_0");
    assert_eq!(stream.next_stream_tick, 0);
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.pending_private_feedback_count(), 0);

    let first = stream.run_prompt(&device, &[1], 1, None).unwrap();
    assert_eq!(first.stream_id, "stream_0");
    assert_eq!(first.prompt_token_ids, vec![1]);
    assert_eq!(first.generated_token_ids.len(), 1);
    assert_eq!(
        first.output_token_ids,
        vec![1, first.generated_token_ids[0]]
    );
    assert_eq!(first.stop_reason, "max_new_tokens");
    assert_eq!(first.start_stream_tick, 0);
    assert_eq!(first.next_stream_tick, 2);
    assert_eq!(first.ticks.len(), 3);
    assert_eq!(
        first.ticks[0].status,
        VulkanResidentRunningStreamTickStatus::Processed
    );
    assert_eq!(first.ticks[0].stream_tick, Some(0));
    assert_eq!(
        first.ticks[0].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(
        first.ticks[0].public_output.as_ref().unwrap().token_id,
        first.generated_token_ids[0]
    );
    assert_eq!(first.ticks[1].stream_tick, Some(1));
    assert_eq!(
        first.ticks[1].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        first.ticks[1].input_signal.as_ref().unwrap().token_id(),
        first.generated_token_ids[0]
    );
    assert_eq!(
        first.ticks[2].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(first.ticks[2].stream_tick, None);
    assert_eq!(stream.next_stream_tick, 2);
    assert_eq!(stream.public_outputs().len(), 1);
    assert_eq!(stream.private_feedback_history().len(), 1);
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.pending_private_feedback_count(), 0);

    let second = stream.run_prompt(&device, &[4], 1, None).unwrap();
    assert_eq!(second.prompt_token_ids, vec![4]);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(
        second.output_token_ids,
        vec![4, second.generated_token_ids[0]]
    );
    assert_eq!(second.stop_reason, "max_new_tokens");
    assert_eq!(second.start_stream_tick, 2);
    assert_eq!(second.next_stream_tick, 4);
    assert_eq!(second.ticks.len(), 3);
    assert_eq!(second.ticks[0].stream_tick, Some(2));
    assert_eq!(
        second.ticks[0].input_signal.as_ref().unwrap().token_id(),
        4
    );
    assert_eq!(
        second.ticks[0].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert_eq!(second.ticks[1].stream_tick, Some(3));
    assert_eq!(
        second.ticks[1].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        second.ticks[2].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(second.ticks[2].stream_tick, None);
    assert_eq!(stream.next_stream_tick, 4);
    assert_eq!(stream.public_outputs().len(), 2);
    assert_eq!(stream.private_feedback_history().len(), 2);
    assert_eq!(stream.ticks().len(), 6);
    assert!(!stream.loop_open);
    assert_eq!(stream.last_stop_reason.as_deref(), Some("max_new_tokens"));

    stream.inject_prompt(&[1], 0, None).unwrap();
    let rolled = stream.tick(&device).unwrap();
    assert_eq!(rolled.stream_tick, Some(4));
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.next_stream_tick, 5);
}

#[test]
fn resident_greedy_running_stream_uses_configured_capacity() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device, "capacity", 8, "",
    )
    .unwrap();
    assert_eq!(processor.dynamic_state_capacity_activations, 8);

    let mut stream = processor.into_running_stream("stream_0");
    let run = stream.run_prompt(&device, &[1], 7, None).unwrap();
    assert_eq!(run.prompt_token_ids, vec![1]);
    assert_eq!(run.generated_token_ids.len(), 7);
    assert_eq!(run.output_token_ids.len(), 8);
    assert_eq!(run.stop_reason, "max_new_tokens");
    assert_eq!(run.start_stream_tick, 0);
    assert_eq!(run.next_stream_tick, 8);
    assert_eq!(stream.next_stream_tick, 8);
    assert_eq!(stream.public_outputs().len(), 7);
    assert_eq!(stream.private_feedback_history().len(), 7);
    assert_eq!(run.ticks.len(), 9);
    assert_eq!(run.ticks[0].stream_tick, Some(0));
    assert_eq!(run.ticks[7].stream_tick, Some(7));
    assert_eq!(
        run.ticks[7].input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::PrivateFeedback
    );
    assert_eq!(
        run.ticks[8].status,
        VulkanResidentRunningStreamTickStatus::Idle
    );
    assert_eq!(run.ticks[8].stream_tick, None);

    stream.inject_prompt(&[4], 0, None).unwrap();
    let rolled = stream.tick(&device).unwrap();
    assert_eq!(rolled.stream_tick, Some(8));
    assert_eq!(stream.pending_external_input_count(), 0);
    assert_eq!(stream.next_stream_tick, 9);
}

#[test]
fn resident_token_stream_api_accepts_external_events_and_emits_public_events() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "token events",
        8,
        "",
    )
    .unwrap();
    let mut stream = processor.into_token_stream("host_stream_0");
    assert_eq!(stream.stream_id(), "host_stream_0");
    assert_eq!(stream.next_stream_tick(), 0);

    let first_event =
        VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host");
    let first = stream
        .submit_external_event(&device, first_event.clone())
        .unwrap();
    assert_eq!(first.stream_id, "host_stream_0");
    assert_eq!(first.input_event, first_event);
    assert_eq!(first.generated_token_ids.len(), 3);
    assert_eq!(first.output_events.len(), 3);
    assert_eq!(first.stop_reason, "max_new_tokens");
    assert_eq!(first.start_stream_tick, 0);
    assert_eq!(first.next_stream_tick, 4);
    assert_eq!(first.processed_tick_count, 4);
    assert_eq!(first.idle_tick_count, 1);
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.input_event_id.as_str())
            .collect::<Vec<_>>(),
        vec!["event_0", "event_0", "event_0"]
    );
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.output_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| event.source_stream_tick)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );

    let second_event =
        VulkanResidentTokenInputEvent::new("event_1", vec![4], 1).with_origin("test_host");
    let second = stream
        .submit_external_event(&device, second_event.clone())
        .unwrap();
    assert_eq!(second.input_event, second_event);
    assert_eq!(second.generated_token_ids.len(), 1);
    assert_eq!(second.output_events.len(), 1);
    assert_eq!(second.output_events[0].input_event_id, "event_1");
    assert_eq!(second.output_events[0].output_index, 0);
    assert_eq!(second.output_events[0].source_stream_tick, 4);
    assert_eq!(second.start_stream_tick, 4);
    assert_eq!(second.next_stream_tick, 6);
    assert_eq!(second.processed_tick_count, 2);
    assert_eq!(second.idle_tick_count, 1);

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.stream_id, "host_stream_0");
    assert_eq!(snapshot.next_stream_tick, 6);
    assert!(!snapshot.loop_open);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 4);
    assert_eq!(snapshot.total_ticks, 8);
    assert_eq!(snapshot.last_stop_reason.as_deref(), Some("max_new_tokens"));
}

#[test]
fn resident_token_stream_can_be_pumped_one_tick_at_a_time() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device, "token pump", 8, "",
    )
    .unwrap();
    let mut stream = processor.into_token_stream("host_stream_0");
    let event = VulkanResidentTokenInputEvent::new("event_0", vec![1], 2).with_origin("test_host");
    let queued = stream.enqueue_external_event(event.clone()).unwrap();
    assert_eq!(queued.input_event, event);
    assert_eq!(queued.start_stream_tick, 0);
    assert_eq!(queued.enqueued_token_count, 1);
    assert!(!stream.snapshot().idle);

    let first = stream.pump_once(&device).unwrap();
    assert_eq!(first.stream_id, "host_stream_0");
    assert_eq!(
        first.status,
        VulkanResidentRunningStreamTickStatus::Processed
    );
    assert_eq!(first.stream_tick, Some(0));
    assert_eq!(first.input_token_id, Some(1));
    assert_eq!(
        first.input_route,
        Some(VulkanResidentPromptEventInputRoute::ExternalInput)
    );
    assert_eq!(
        first.output_event.as_ref().unwrap().input_event_id,
        "event_0"
    );
    assert_eq!(first.output_event.as_ref().unwrap().output_index, 0);
    assert_eq!(first.output_event.as_ref().unwrap().source_stream_tick, 0);

    let second = stream.pump_once(&device).unwrap();
    assert_eq!(second.stream_tick, Some(1));
    assert_eq!(
        second.input_route,
        Some(VulkanResidentPromptEventInputRoute::PrivateFeedback)
    );
    assert_eq!(
        second.output_event.as_ref().unwrap().input_event_id,
        "event_0"
    );
    assert_eq!(second.output_event.as_ref().unwrap().output_index, 1);
    assert_eq!(second.output_event.as_ref().unwrap().source_stream_tick, 1);

    let closing = stream.pump_once(&device).unwrap();
    assert_eq!(closing.stream_tick, Some(2));
    assert_eq!(
        closing.input_route,
        Some(VulkanResidentPromptEventInputRoute::PrivateFeedback)
    );
    assert!(closing.output_event.is_none());
    assert_eq!(closing.stop_reason.as_deref(), Some("max_new_tokens"));

    let idle = stream.pump_once(&device).unwrap();
    assert_eq!(idle.status, VulkanResidentRunningStreamTickStatus::Idle);
    assert_eq!(idle.stream_tick, None);
    assert!(idle.output_event.is_none());
    assert_eq!(idle.stop_reason.as_deref(), Some("max_new_tokens"));

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.next_stream_tick, 3);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 2);
    assert_eq!(snapshot.total_ticks, 4);
}

#[test]
fn resident_token_stream_can_pump_bounded_runtime_cycles() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let processor = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "bounded token pump",
        8,
        "",
    )
    .unwrap();
    let mut stream = processor.into_token_stream("host_stream_0");
    stream
        .enqueue_external_event(
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
        )
        .unwrap();

    let first_cycle = stream.pump_bounded(&device, 2).unwrap();
    assert_eq!(first_cycle.stream_id, "host_stream_0");
    assert_eq!(first_cycle.start_stream_tick, 0);
    assert_eq!(first_cycle.next_stream_tick, 2);
    assert_eq!(
        first_cycle.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::TickBudget
    );
    assert_eq!(first_cycle.processed_tick_count, 2);
    assert_eq!(first_cycle.idle_tick_count, 0);
    assert_eq!(first_cycle.output_events.len(), 2);
    assert_eq!(first_cycle.ticks.len(), 2);
    assert_eq!(first_cycle.ticks[0].stream_tick, Some(0));
    assert_eq!(first_cycle.ticks[1].stream_tick, Some(1));
    assert_eq!(
        first_cycle
            .output_events
            .iter()
            .map(|event| event.output_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(stream.snapshot().next_stream_tick, 2);
    assert!(!stream.snapshot().idle);

    let second_cycle = stream.pump_bounded(&device, 3).unwrap();
    assert_eq!(second_cycle.start_stream_tick, 2);
    assert_eq!(second_cycle.next_stream_tick, 4);
    assert_eq!(
        second_cycle.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::Idle
    );
    assert_eq!(second_cycle.processed_tick_count, 2);
    assert_eq!(second_cycle.idle_tick_count, 1);
    assert_eq!(second_cycle.output_events.len(), 1);
    assert_eq!(second_cycle.output_events[0].output_index, 2);
    assert_eq!(second_cycle.output_events[0].source_stream_tick, 2);
    assert_eq!(second_cycle.ticks.len(), 3);
    assert_eq!(second_cycle.ticks[0].stream_tick, Some(2));
    assert_eq!(second_cycle.ticks[1].stream_tick, Some(3));
    assert_eq!(second_cycle.ticks[2].stream_tick, None);
    assert_eq!(
        second_cycle.last_stop_reason.as_deref(),
        Some("max_new_tokens")
    );

    let snapshot = stream.snapshot();
    assert_eq!(snapshot.next_stream_tick, 4);
    assert!(snapshot.idle);
    assert_eq!(snapshot.total_public_outputs, 3);
    assert_eq!(snapshot.total_ticks, 5);

    let no_budget = stream.pump_bounded(&device, 0).unwrap();
    assert_eq!(
        no_budget.stop_condition,
        VulkanResidentTokenStreamPumpStopCondition::TickBudget
    );
    assert_eq!(no_budget.processed_tick_count, 0);
    assert_eq!(no_budget.idle_tick_count, 0);
    assert!(no_budget.output_events.is_empty());
    assert!(no_budget.ticks.is_empty());
    assert_eq!(no_budget.start_stream_tick, 4);
    assert_eq!(no_budget.next_stream_tick, 4);
}

#[test]
fn resident_feedback_cycle_restores_recurrent_state_when_eos_arrives_mid_cycle() {
    let device = selected_test_vulkan_device().expect("selected Vulkan test device must open");
    let create_stream = |stream_id: &str| {
        fixture_model_resident_greedy_model(&device, 16)
            .unwrap()
            .create_stream_processor(&device, 0)
            .unwrap()
            .into_token_stream(stream_id)
    };
    let event =
        VulkanResidentTokenInputEvent::new("eos_event", vec![1, 2, 3], 8).with_stop_tokens(vec![23]);

    let mut scalar = create_stream("scalar_stream");
    scalar.enqueue_external_event(event.clone()).unwrap();
    let mut scalar_output = Vec::new();
    loop {
        let tick = scalar.pump_once(&device).unwrap();
        if let Some(output) = tick.output_event {
            scalar_output.push(output.token_id);
        }
        if tick.status == VulkanResidentRunningStreamTickStatus::Idle {
            break;
        }
    }
    let scalar_static_state = scalar
        .inner
        .processor
        ._mounted
        .buffers
        .state_buffers
        .iter()
        .filter(|state| state.static_byte_capacity.is_some())
        .map(|state| {
            (
                state.component_id.clone(),
                state.state_id.clone(),
                state.buffer.read_bytes(state.byte_capacity).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let scalar_snapshot = scalar.snapshot();
    drop(scalar);

    let mut batched = create_stream("batched_stream");
    batched.enqueue_external_event(event).unwrap();
    let mut batched_output = Vec::new();
    loop {
        let cycle = batched.pump_bounded(&device, 4).unwrap();
        batched_output.extend(cycle.output_events.iter().map(|output| output.token_id));
        if cycle.stop_condition == VulkanResidentTokenStreamPumpStopCondition::Idle {
            break;
        }
    }
    let batched_static_state = batched
        .inner
        .processor
        ._mounted
        .buffers
        .state_buffers
        .iter()
        .filter(|state| state.static_byte_capacity.is_some())
        .map(|state| {
            (
                state.component_id.clone(),
                state.state_id.clone(),
                state.buffer.read_bytes(state.byte_capacity).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let batched_snapshot = batched.snapshot();

    assert_eq!(scalar_output, vec![23]);
    assert_eq!(batched_output, scalar_output);
    assert_eq!(
        batched_snapshot.next_stream_tick,
        scalar_snapshot.next_stream_tick
    );
    assert_eq!(
        batched_snapshot.last_stop_reason,
        scalar_snapshot.last_stop_reason
    );
    assert_eq!(batched_static_state, scalar_static_state);
}
