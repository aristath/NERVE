fn run_placed_chat(
    args: &Args,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    capacity: usize,
    codec: &VulkanResidentHfTokenizerTextCodec,
    initial_prompt: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let setup_start = Instant::now();
    let chat_session =
        RuntimeChatSession::from_tokenizer_dir(tokenizer_dir, &args.chat_template_variables)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        manifest_dir,
        tokenizer_dir,
        &runtime_model.package,
        &chat_session.formatter,
    )?;
    let transcript_codec = chat_transcript_codec(tokenizer_dir)?;
    let logical_device_ids = runtime_model.placement_device_ids();
    let sparse_moe_contract = runtime_model.sparse_moe_execution_contract()?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let (runtime_model, implementation_selection) = runtime_model
        .select_and_apply_runtime_implementations(
            manifest_dir,
            &bound_devices.hardware_profiles,
            RuntimeExecutionEnvelope {
                phases: vec![
                    "decode".to_string(),
                    "prefill".to_string(),
                ],
                activation_batch: RuntimeInclusiveRange {
                    minimum: 1,
                    maximum: capacity.max(1),
                },
                context_activations: RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: capacity,
                },
                state_activations: RuntimeInclusiveRange {
                    minimum: 0,
                    maximum: capacity,
                },
                speculative_draft_tokens: args.speculative_draft_tokens,
            },
        )?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(capacity),
        args.random_seed,
        args.speculative_draft_tokens,
        sampler_runtime_config(args),
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let mounted_device_bindings = bound_devices
        .physical_device_ids
        .iter()
        .map(|(logical_device_id, physical_device_id)| {
            format!("{logical_device_id}={physical_device_id}")
        })
        .collect::<Vec<_>>();
    println!(
        "nerve chat ready: placed_in_process, devices={:?}, bindings={:?}, context_size={}, setup_ms={:.3}",
        stream_snapshot.device_ids,
        mounted_device_bindings,
        stream_snapshot.context_window_activations,
        nanos_to_millis(elapsed_nanos_u64(setup_start))
    );

    run_chat_repl(
        initial_prompt,
        chat_session,
        codec,
        &transcript_codec,
        &stop_token_ids,
        |turn_index, chat_session, input_text, prepared| {
            print!("llm> ");
            io::stdout().flush()?;
            let mut decoder = codec.decode_stream();
            let mut output_error = None;
            let generation_context_start = chat_session
                .committed_token_ids
                .len()
                .saturating_add(prepared.user_token_delta.len())
                .saturating_add(
                    prepared.generation_prompt_token_delta.len(),
            );
            let mut previous_output_at = None;
            let mut sustained_decode_samples = Vec::new();
            let transaction = execute_vulkan_resident_chat_transaction(
                &mut engine,
                "main",
                chat_session,
                &transcript_codec,
                &stop_token_ids,
                turn_index,
                input_text,
                prepared,
                args.max_new_tokens,
                |output_event| {
                    let output_at = Instant::now();
                    if let Some(previous) = previous_output_at {
                        sustained_decode_samples.push(
                            RuntimeSustainedDecodeSample {
                                context_activation:
                                    generation_context_start
                                        .saturating_add(
                                            output_event
                                                .output_event
                                                .output_index,
                                        ),
                                transient_state_activation:
                                    output_event
                                        .output_event
                                        .source_stream_tick,
                                inter_token_time_ns:
                                    u64::try_from(
                                        output_at
                                            .duration_since(previous)
                                            .as_nanos(),
                                    )
                                    .unwrap_or(u64::MAX),
                            },
                        );
                    }
                    previous_output_at = Some(output_at);
                    if output_error.is_some() {
                        return;
                    }
                    match decoder.step(output_event.output_event.token_id) {
                        Ok(Some(text)) => {
                            print!("{text}");
                            if let Err(error) = io::stdout().flush() {
                                output_error = Some(error.to_string());
                            }
                        }
                        Ok(None) => {}
                        Err(error) => output_error = Some(error.to_string()),
                    }
                },
            )?;
            let submitted_run = transaction
                .generation_run
                .engine_run
                .input_runs
                .iter()
                .find(|input_run| {
                    input_run.stream_id == "main"
                        && input_run.submitted_run.input_event.id
                            == transaction.generation_event_id
                })
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        "placed chat engine run loop did not return the generation-branch event run",
                    )
                })?;
            let engine_runs = [
                &transaction.user_run.engine_run,
                &transaction.generation_run.engine_run,
                &transaction.commit_run.engine_run,
            ];
            let prefill_activation_count = engine_runs
                .iter()
                .map(|run| run.prefill_activation_count)
                .sum();
            let decode_activation_count = engine_runs
                .iter()
                .map(|run| run.decode_activation_count)
                .sum();
            let timing = runtime_prompt_timing_report(
                0,
                transaction.elapsed_ns,
                prepared
                    .user_token_delta
                    .len()
                    .saturating_add(prepared.generation_prompt_token_delta.len())
                    .saturating_add(transaction.assistant_commit_token_ids.len()),
                transaction.generated_token_ids.len(),
                engine_runs.iter().map(|run| run.scheduler_step_count).sum(),
                engine_runs
                    .iter()
                    .map(|run| run.activation_batch_count)
                    .sum(),
                engine_runs
                    .iter()
                    .map(|run| run.prefill_activation_batch_count)
                    .sum(),
                engine_runs
                    .iter()
                    .map(|run| run.decode_activation_batch_count)
                    .sum(),
                engine_runs
                    .iter()
                    .map(|run| run.max_activation_batch_width)
                    .max()
                    .unwrap_or_default(),
                engine_runs
                    .iter()
                    .map(|run| run.physical_multi_stream_batch_count)
                    .sum(),
                engine_runs
                    .iter()
                    .map(|run| run.max_physical_multi_stream_batch_width)
                    .max()
                    .unwrap_or_default(),
                engine_runs
                    .iter()
                    .map(|run| run.max_pending_activation_count)
                    .max()
                    .unwrap_or_default(),
                prefill_activation_count,
                decode_activation_count,
                engine_runs.iter().map(|run| run.prefill_time_ns).sum(),
                engine_runs.iter().map(|run| run.decode_time_ns).sum(),
                submitted_run.submitted_run.session_run.tick_count,
                submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .scheduler_turn_count,
            );
            let prefix_state_cache =
                transaction.commit_run.engine_run.prefix_state_cache.clone();
            if let Some(error) = output_error {
                return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, error)));
            }
            let speculative_decode =
                submitted_run.submitted_run.session_run.run.speculative_decode.clone();
            let resident_feedback = runtime_feedback_execution_report(
                submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .resident_feedback
                    .clone(),
            );
            let transport_edges = runtime_placed_transport_edge_reports(
                &submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .transport_stats,
            );
            Ok(RuntimeChatTurn {
                generated_token_ids: transaction.generated_token_ids,
                canonical_committed_token_ids:
                    transaction.canonical_committed_token_ids,
                streamed: true,
                timing,
                sustained_decode:
                    RuntimeSustainedDecodeReport::from_samples(
                        &sustained_decode_samples,
                ),
                implementation_selection: implementation_selection.clone(),
                execution_counters: transaction.execution_counters,
                prefix_state_cache,
                speculative_cycle_count: speculative_decode.cycle_count,
                speculative_rollback_cycle_count:
                    speculative_decode.rollback_cycle_count,
                proposed_draft_token_count:
                    speculative_decode.proposed_draft_token_count,
                accepted_draft_token_count:
                    speculative_decode.accepted_draft_token_count,
                speculative_emitted_token_count:
                    speculative_decode.emitted_token_count,
                speculative_draft_time_ns: speculative_decode.draft_time_ns,
                speculative_target_verification_time_ns:
                    speculative_decode.target_verification_time_ns,
                speculative_draft_catch_up_time_ns:
                    speculative_decode.draft_catch_up_time_ns,
                speculative_total_time_ns: speculative_decode.total_time_ns,
                resident_feedback,
                sparse_moe: sparse_moe_contract.work_report(
                    prefill_activation_count,
                    decode_activation_count,
                ),
                transport_edges,
            })
        },
    )
}

fn run_placed_prompt(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let report = execute_placed_prompt_run(context, runtime_model)?;
    print_placed_prompt_report(context.args, &report)
}

fn execute_placed_prompt_run(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<RuntimePlacedPromptRunReport, Box<dyn Error>> {
    let PromptRunContext {
        args,
        package_manifest,
        manifest_dir,
        tokenizer_dir,
        prompt,
        prompt_ids,
        scheduled_token_activations,
        capacity,
        codec,
        ..
    } = context;
    let setup_start = Instant::now();
    let logical_device_ids = runtime_model.placement_device_ids();
    let sparse_moe_contract = runtime_model.sparse_moe_execution_contract()?;
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(*capacity),
        args.random_seed,
        args.speculative_draft_tokens,
        sampler_runtime_config(args),
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);
    let run_start = Instant::now();
    let input_event =
        VulkanResidentTokenInputEvent::new("prompt", prompt_ids.to_vec(), args.max_new_tokens);
    let input_event_id = input_event.id.clone();
    reset_vulkan_resident_execution_counters();
    let submitted_run = engine.submit_input_event_until_idle("main", input_event)?;
    let run_time_ns = elapsed_nanos_u64(run_start);
    let engine_run = submitted_run.engine_run;
    let prefill_activation_count = engine_run.prefill_activation_count;
    let decode_activation_count = engine_run.decode_activation_count;
    let prefill_time_ns = engine_run.prefill_time_ns;
    let decode_time_ns = engine_run.decode_time_ns;
    let scheduler_step_count = engine_run.scheduler_step_count;
    let activation_batch_count = engine_run.activation_batch_count;
    let prefill_activation_batch_count = engine_run.prefill_activation_batch_count;
    let decode_activation_batch_count = engine_run.decode_activation_batch_count;
    let max_activation_batch_width = engine_run.max_activation_batch_width;
    let physical_multi_stream_batch_count = engine_run.physical_multi_stream_batch_count;
    let max_physical_multi_stream_batch_width =
        engine_run.max_physical_multi_stream_batch_width;
    let max_pending_activation_count = engine_run.max_pending_activation_count;
    let run = engine_run
        .input_runs
        .into_iter()
        .find(|run| run.stream_id == "main" && run.submitted_run.input_event.id == input_event_id)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "placed prompt engine run loop did not return the submitted prompt event run",
            )
        })?
        .submitted_run
        .session_run
        .run;
    let generated_text = codec.decode_tokens(&run.generated_token_ids)?;
    let output_text = codec.decode_tokens(&run.output_token_ids)?;
    let total_scheduler_turns = run.scheduler_turn_count;
    let completed_stage_deltas = vec![run.completed_stage_count];
    let tick_count = run.tick_count;
    let generated_token_count = run.generated_token_ids.len();
    let timing = runtime_prompt_timing_report(
        setup_time_ns,
        run_time_ns,
        prompt_ids.len(),
        generated_token_count,
        scheduler_step_count,
        activation_batch_count,
        prefill_activation_batch_count,
        decode_activation_batch_count,
        max_activation_batch_width,
        physical_multi_stream_batch_count,
        max_physical_multi_stream_batch_width,
        max_pending_activation_count,
        prefill_activation_count,
        decode_activation_count,
        prefill_time_ns,
        decode_time_ns,
        tick_count,
        total_scheduler_turns,
    );
    let component_timings = Vec::new();
    let component_timing_summaries = Vec::new();
    let transport_stats_by_tick = Vec::new();
    let transport_published_packet_count = run.transport_stats.published_packet_count;
    let transport_published_byte_count = run.transport_stats.published_byte_count;
    let transport_received_packet_count = run.transport_stats.received_packet_count;
    let transport_received_byte_count = run.transport_stats.received_byte_count;
    let transport_direct_copy_count = run.transport_stats.direct_copy_count;
    let transport_direct_copy_byte_count = run.transport_stats.direct_copy_byte_count;
    let transport_direct_receive_count = run.transport_stats.direct_receive_count;
    let transport_direct_receive_byte_count = run.transport_stats.direct_receive_byte_count;
    let transport_edges = runtime_placed_transport_edge_reports(&run.transport_stats);

    Ok(RuntimePlacedPromptRunReport {
        ok: true,
        execution_mode: "placed_in_process".to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        input_device_id: stream_snapshot.input_device_id.clone(),
        output_device_id: stream_snapshot.output_device_id.clone(),
        device_count: stream_snapshot.device_ids.len(),
        device_ids: stream_snapshot.device_ids.clone(),
        bound_devices: bound_devices_report(&bound_devices),
        edge_routes: bound_edge_routes_report(&bound_devices, &placement.edges),
        runtime_graph: runtime_graph_report(args),
        device_bindings: runtime_device_bindings_report(
            args,
            &stream_snapshot.device_ids,
            &bound_devices.available_devices,
        ),
        hosted_component_count: stream_snapshot.hosted_component_count,
        context_window_activations: stream_snapshot.context_window_activations,
        scheduled_token_activations: *scheduled_token_activations,
        tokenizer: tokenizer_options_report(args),
        prompt_text: prompt.to_string(),
        prompt_ids: run.prompt_token_ids.clone(),
        generated_ids: run.generated_token_ids.clone(),
        generated_text: generated_text.clone(),
        output_text: output_text.clone(),
        stop_reason: run.stop_reason.clone(),
        tick_count,
        scheduler_turns: total_scheduler_turns,
        completed_stage_deltas,
        transport: RuntimePlacedTransportReport {
            published_packet_count: transport_published_packet_count,
            published_byte_count: transport_published_byte_count,
            received_packet_count: transport_received_packet_count,
            received_byte_count: transport_received_byte_count,
            direct_copy_count: transport_direct_copy_count,
            direct_copy_byte_count: transport_direct_copy_byte_count,
            direct_receive_count: transport_direct_receive_count,
            direct_receive_byte_count: transport_direct_receive_byte_count,
            edges: transport_edges,
            by_tick: transport_stats_by_tick,
        },
        timing,
        component_timings,
        component_timing_summaries,
        speculative_cycle_count: run.speculative_decode.cycle_count,
        speculative_rollback_cycle_count: run.speculative_decode.rollback_cycle_count,
        proposed_draft_token_count: run.speculative_decode.proposed_draft_token_count,
        accepted_draft_token_count: run.speculative_decode.accepted_draft_token_count,
        speculative_emitted_token_count: run.speculative_decode.emitted_token_count,
        speculative_draft_time_ns: run.speculative_decode.draft_time_ns,
        speculative_target_verification_time_ns: run.speculative_decode.target_verification_time_ns,
        speculative_draft_catch_up_time_ns: run.speculative_decode.draft_catch_up_time_ns,
        speculative_total_time_ns: run.speculative_decode.total_time_ns,
        resident_feedback: runtime_feedback_execution_report(run.resident_feedback),
        sparse_moe: sparse_moe_contract.work_report(
            prefill_activation_count,
            decode_activation_count,
        ),
    })
}

fn runtime_placed_transport_edge_reports(
    stats: &VulkanPlacedEdgeTransportStats,
) -> Vec<RuntimePlacedTransportEdgeReport> {
    stats
        .edges
        .iter()
        .map(|edge| RuntimePlacedTransportEdgeReport {
            edge_index: edge.key.edge_index,
            from_device_id: edge.key.from_device_id.clone(),
            to_device_id: edge.key.to_device_id.clone(),
            signal: edge.signal.clone(),
            route: match edge.route {
                VulkanPlacedEdgeTransferRoute::SameDeviceAlias => "same_device_alias",
                VulkanPlacedEdgeTransferRoute::DeviceLocalCopy => "device_local_copy",
                VulkanPlacedEdgeTransferRoute::DeviceLocalStaging => "device_local_staging",
                VulkanPlacedEdgeTransferRoute::ExternalDeviceLocal => "external_device_local",
                VulkanPlacedEdgeTransferRoute::SharedHost => "shared_host",
                VulkanPlacedEdgeTransferRoute::HostStaging => "host_staging",
            }
            .to_string(),
            byte_capacity: edge.byte_capacity,
            publish_count: edge.publish_count,
            receive_count: edge.receive_count,
            transferred_byte_count: edge.transferred_byte_count,
            queue_signal_count: edge.queue_signal_count,
            queue_wait_count: edge.queue_wait_count,
            host_wait_count: edge.host_wait_count,
            queue_overlap_eligible: edge.queue_overlap_eligible,
            overlap_submission_count: edge.overlap_submission_count,
        })
        .collect()
}

fn runtime_feedback_execution_report(
    stats: VulkanResidentFeedbackExecutionStats,
) -> RuntimeFeedbackExecutionReport {
    RuntimeFeedbackExecutionReport {
        window_count: stats.window_count,
        planned_tick_count: stats.planned_tick_count,
        submitted_tick_count: stats.submitted_tick_count,
        executed_tick_count: stats.executed_tick_count,
        retained_tick_count: stats.retained_tick_count,
        sampled_tick_count: stats.sampled_tick_count,
        discarded_tick_count: stats.discarded_tick_count,
        template_record_count: stats.template_record_count,
        template_replay_count: stats.template_replay_count,
        asynchronous_submission_count: stats.asynchronous_submission_count,
        completion_poll_count: stats.completion_poll_count,
        bounded_wait_count: stats.bounded_wait_count,
        bounded_wait_timeout_count: stats.bounded_wait_timeout_count,
    }
}

fn print_placed_prompt_report(
    args: &Args,
    report: &RuntimePlacedPromptRunReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else if args.generated_only {
        print_text(&report.generated_text);
    } else {
        print_text(&report.output_text);
        print_runtime_timing_stats("stats", &report.timing);
        print_runtime_execution_counters(&vulkan_resident_execution_counters());
        print_runtime_feedback_stats(&report.resident_feedback);
        print_runtime_sparse_moe_stats(&report.sparse_moe);
        print_runtime_transport_edges(&report.transport.edges);
        print_speculative_profile(report);
        print_placed_component_timing_profile(&report.component_timing_summaries, 5);
    }
    Ok(())
}

fn print_speculative_profile(report: &RuntimePlacedPromptRunReport) {
    print_runtime_speculative_stats(
        report.speculative_cycle_count,
        report.speculative_rollback_cycle_count,
        report.proposed_draft_token_count,
        report.accepted_draft_token_count,
        report.speculative_emitted_token_count,
        report.speculative_draft_time_ns,
        report.speculative_target_verification_time_ns,
        report.speculative_draft_catch_up_time_ns,
        report.speculative_total_time_ns,
    );
}

fn generated_tokens_per_second(generated_token_count: usize, run_time_ns: u64) -> Option<f64> {
    if run_time_ns == 0 {
        None
    } else {
        Some(generated_token_count as f64 / (run_time_ns as f64 / 1_000_000_000.0))
    }
}
