fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

fn print_runtime_timing_stats(label: &str, timing: &RuntimePromptTimingReport) {
    println!("{label}:");
    println!("  setup_ms={:.3}", nanos_to_millis(timing.setup_time_ns));
    println!("  run_ms={:.3}", nanos_to_millis(timing.run_time_ns));
    println!("  total_ms={:.3}", nanos_to_millis(timing.total_time_ns));
    println!("  generated_tokens={}", timing.generated_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.generated_token_count, timing.run_time_ns)
    {
        println!("  generated_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  prefill_tokens={}", timing.prefill_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.prefill_token_count, timing.prefill_time_ns)
    {
        println!("  prefill_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  decode_tokens={}", timing.decode_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.decode_token_count, timing.decode_time_ns)
    {
        println!("  decode_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  prefill_activations={}", timing.prefill_activation_count);
    println!("  decode_activations={}", timing.decode_activation_count);
    println!("  scheduler_steps={}", timing.scheduler_step_count);
    println!("  activation_batches={}", timing.activation_batch_count);
    println!(
        "  prefill_activation_batches={}",
        timing.prefill_activation_batch_count
    );
    println!(
        "  decode_activation_batches={}",
        timing.decode_activation_batch_count
    );
    println!(
        "  max_activation_batch_width={}",
        timing.max_activation_batch_width
    );
    println!(
        "  physical_multi_stream_batches={}",
        timing.physical_multi_stream_batch_count
    );
    println!(
        "  max_physical_multi_stream_batch_width={}",
        timing.max_physical_multi_stream_batch_width
    );
    println!(
        "  max_pending_activations={}",
        timing.max_pending_activation_count
    );
    println!(
        "  prefill_ms={:.3}",
        nanos_to_millis(timing.prefill_time_ns)
    );
    println!("  decode_ms={:.3}", nanos_to_millis(timing.decode_time_ns));
    println!("  ticks={}", timing.tick_count);
    println!("  scheduler_turns={}", timing.scheduler_turn_count);
    if let Some(average) = timing.average_generated_token_time_ns {
        println!("  avg_generated_token_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_prefill_activation_time_ns {
        println!(
            "  avg_prefill_activation_ms={:.3}",
            nanos_to_millis(average)
        );
    }
    if let Some(average) = timing.average_decode_activation_time_ns {
        println!(
            "  avg_decode_activation_ms={:.3}",
            nanos_to_millis(average)
        );
    }
    if let Some(average) = timing.average_tick_time_ns {
        println!("  avg_tick_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_scheduler_turn_time_ns {
        println!("  avg_scheduler_turn_ms={:.3}", nanos_to_millis(average));
    }
}

fn print_runtime_implementation_selection(
    report: &RuntimeImplementationSelectionReport,
) {
    println!("implementations:");
    println!("  selected={}", report.selected.len());
    println!(
        "  exact_instances={}",
        report.exact_instance_ids.len()
    );
    println!(
        "  estimated_saved_ms={:.3}",
        nanos_to_millis(report.total_estimated_saved_ns)
    );
    println!(
        "  representation_boundary_ms={:.3}",
        nanos_to_millis(report.total_conversion_ns)
    );
    println!(
        "  representation_boundary_bytes={}",
        report.total_conversion_bytes
    );
    println!(
        "  representation_boundary_count={}",
        report.total_boundary_count
    );
    for selection in &report.selected {
        let representation = selection
            .representation
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("custom");
        println!(
            "  implementation={} instances={:?} predicate={} representation={} benchmark={} validation={}",
            selection.implementation_id,
            selection.instance_ids,
            selection.predicate.predicate_id,
            representation,
            selection.benchmark_id,
            selection.validation_id,
        );
    }
}

fn print_runtime_sustained_decode_stats(
    report: &RuntimeSustainedDecodeReport,
) {
    println!("sustained_decode:");
    println!(
        "  measured_inter_token_samples={}",
        report.measured_token_count
    );
    for window in &report.windows {
        println!(
            "  window_{}=context:{}-{},state:{}-{},tokens:{},elapsed_ms:{:.3},tokens_per_second:{:.3}",
            window.ordinal,
            window.context_activation_start,
            window.context_activation_end,
            window.transient_state_activation_start,
            window.transient_state_activation_end,
            window.token_count,
            nanos_to_millis(window.elapsed_ns),
            generated_tokens_per_second(
                window.token_count,
                window.elapsed_ns,
            )
            .unwrap_or_default(),
        );
    }
}

fn print_runtime_execution_counters(counters: &VulkanResidentExecutionCounters) {
    println!("execution:");
    print_runtime_distributed_execution_phase_counters("decode", &counters.distributed.decode);
    print_runtime_distributed_execution_phase_counters("prefill", &counters.distributed.prefill);
    println!(
        "  resident_sequence_prepare_calls={}",
        counters.resident_sequence_prepare_calls
    );
    println!(
        "  resident_sequence_recorded_command_buffers={}",
        counters.resident_sequence_recorded_command_buffers
    );
    println!(
        "  resident_sequence_reused_command_buffers={}",
        counters.resident_sequence_reused_command_buffers
    );
    println!(
        "  resident_sequence_queue_submits={}",
        counters.resident_sequence_queue_submits
    );
    println!(
        "  resident_sequence_completion_waits={}",
        counters.resident_sequence_completion_waits
    );
    println!(
        "  resident_queue_batch_submits={}",
        counters.resident_queue_batch_submits
    );
    println!(
        "  resident_queue_batch_commands={}",
        counters.resident_queue_batch_commands
    );
    println!(
        "  resident_copy_queue_submits={}",
        counters.resident_copy_queue_submits
    );
    println!("  resident_copy_waits={}", counters.resident_copy_waits);
    println!(
        "  demand_initial_sequences={}",
        counters.demand_initial_sequence_count
    );
    println!(
        "  demand_initial_device_ms={:.3}",
        nanos_to_millis(counters.demand_initial_device_duration_ns)
    );
    println!(
        "  demand_initial_max_device_ms={:.3}",
        nanos_to_millis(counters.demand_initial_max_device_duration_ns)
    );
    println!(
        "  demand_resume_sequences={}",
        counters.demand_resume_sequence_count
    );
    println!(
        "  demand_resume_device_ms={:.3}",
        nanos_to_millis(counters.demand_resume_device_duration_ns)
    );
    println!(
        "  demand_resume_max_device_ms={:.3}",
        nanos_to_millis(counters.demand_resume_max_device_duration_ns)
    );
    println!(
        "  resident_component_sequences={}",
        counters.resident_component_sequence_count
    );
    println!(
        "  resident_component_device_ms={:.3}",
        nanos_to_millis(counters.resident_component_device_duration_ns)
    );
    println!(
        "  resident_component_max_device_ms={:.3}",
        nanos_to_millis(counters.resident_component_max_device_duration_ns)
    );
    println!(
        "  execution_quanta={}",
        counters.execution_quantum_count
    );
    println!(
        "  execution_quantum_regions={}",
        counters.execution_quantum_region_count
    );
    println!(
        "  execution_quantum_forced_yields={}",
        counters.execution_quantum_forced_yield_count
    );
    println!(
        "  execution_quantum_estimated_work_units={}",
        counters.execution_quantum_estimated_work_units
    );
    println!(
        "  execution_quantum_estimated_memory_bytes={}",
        counters.execution_quantum_estimated_memory_bytes
    );
    println!(
        "  execution_quantum_dispatches={}",
        counters.execution_quantum_dispatch_count
    );
    println!(
        "  execution_quantum_predicted_ms={:.3}",
        nanos_to_millis(counters.execution_quantum_predicted_duration_ns)
    );
    println!(
        "  execution_quantum_host_submit_wait_ms={:.3}",
        nanos_to_millis(counters.execution_quantum_host_submit_wait_duration_ns)
    );
    println!(
        "  execution_quantum_max_regions={}",
        counters.execution_quantum_max_region_count
    );
    println!(
        "  execution_quantum_max_host_submit_wait_ms={:.3}",
        nanos_to_millis(counters.execution_quantum_max_host_submit_wait_duration_ns)
    );
}

fn print_runtime_distributed_execution_phase_counters(
    phase: &str,
    counters: &VulkanResidentDistributedExecutionPhaseCounters,
) {
    for line in runtime_distributed_execution_phase_counter_lines(phase, counters) {
        println!("{line}");
    }
}

fn runtime_distributed_execution_phase_counter_lines(
    phase: &str,
    counters: &VulkanResidentDistributedExecutionPhaseCounters,
) -> Vec<String> {
    vec![
        format!(
            "  distributed_{phase}_island_submissions={}",
            counters.island_submissions
        ),
        format!(
            "  distributed_{phase}_shard_submissions={}",
            counters.shard_submissions
        ),
        format!(
            "  distributed_{phase}_tensor_parallel_island_submissions={}",
            counters.tensor_parallel_island_submissions
        ),
        format!(
            "  distributed_{phase}_whole_expert_parallel_island_submissions={}",
            counters.whole_expert_parallel_island_submissions
        ),
        format!(
            "  distributed_{phase}_intra_expert_tensor_parallel_island_submissions={}",
            counters.intra_expert_tensor_parallel_island_submissions
        ),
        format!(
            "  distributed_{phase}_hybrid_island_submissions={}",
            counters.hybrid_island_submissions
        ),
    ]
}

fn print_runtime_critical_path(report: &RuntimeCriticalPathReport) {
    for line in runtime_critical_path_lines(report) {
        println!("{line}");
    }
}

fn runtime_critical_path_lines(report: &RuntimeCriticalPathReport) -> Vec<String> {
    let mut lines = vec![
        "critical_path:".to_string(),
        format!(
            "  wall_ms={:.3} host_exclusive_work_ms={:.3} attributed_ms={:.3} unattributed_ms={:.3} parallel_overlap_ms={:.3} coverage={:.2}%",
        nanos_to_millis(report.wall_duration_ns),
        nanos_to_millis(report.host_exclusive_work_duration_ns),
        nanos_to_millis(report.host_attributed_critical_path_duration_ns),
        nanos_to_millis(report.host_unattributed_duration_ns),
        nanos_to_millis(report.host_parallel_overlap_duration_ns),
        f64::from(report.host_coverage_basis_points) / 100.0,
        ),
        format!(
            "  device_timestamp_ms={:.3} (reported separately; device intervals may overlap host work and each other)",
            nanos_to_millis(report.device_timestamp_duration_ns),
        ),
        format!(
            "  normalization_generated_tokens={} normalization_execution_windows={} device_detail_sampled_windows={}",
            report.generated_token_count,
            report.execution_window_count,
            report.device_sampled_execution_window_count,
        ),
    ];
    for phase in &report.phases {
        if phase.host_invocation_count == 0 && phase.device_timestamp_count == 0 {
            continue;
        }
        lines.push(format!(
            "  phase={} host_calls={} host_exclusive_ms={:.3} host_inclusive_ms={:.3} host_max_ms={:.3} device_timestamps={} device_ms={:.3} device_max_ms={:.3} host_us/token={} device_us/token={} host_us/window={} device_us/window={} device_us/sampled_window={}",
            phase.phase,
            phase.host_invocation_count,
            nanos_to_millis(phase.host_exclusive_duration_ns),
            nanos_to_millis(phase.host_inclusive_duration_ns),
            nanos_to_millis(phase.host_max_inclusive_duration_ns),
            phase.device_timestamp_count,
            nanos_to_millis(phase.device_duration_ns),
            nanos_to_millis(phase.device_max_duration_ns),
            optional_nanos_to_micros(phase.host_exclusive_per_generated_token_ns),
            optional_nanos_to_micros(phase.device_per_generated_token_ns),
            optional_nanos_to_micros(phase.host_exclusive_per_execution_window_ns),
            optional_nanos_to_micros(phase.device_per_execution_window_ns),
            optional_nanos_to_micros(phase.device_per_sampled_execution_window_ns),
        ));
    }
    lines
}

fn optional_nanos_to_micros(duration_ns: Option<u64>) -> String {
    duration_ns
        .map(|duration_ns| format!("{:.3}", duration_ns as f64 / 1_000.0))
        .unwrap_or_else(|| "n/a".to_string())
}

fn print_runtime_sparse_moe_stats(stats: &RuntimeSparseMoeWorkReport) {
    if stats.component_count == 0 {
        return;
    }
    println!("sparse_moe:");
    println!("  component_count={}", stats.component_count);
    println!("  activation_count={}", stats.activation_count);
    println!("  declared_expert_slots={}", stats.declared_expert_slots);
    println!("  selected_expert_routes={}", stats.selected_expert_routes);
    println!(
        "  submitted_expert_route_slots={}",
        stats.submitted_expert_route_slots
    );
    println!(
        "  grouped_prefill_routes={}",
        stats.grouped_prefill_routes
    );
    println!(
        "  skipped_dense_expert_slots={}",
        stats.skipped_dense_expert_slots
    );
    println!(
        "  empty_shard_route_checks={}",
        stats.empty_shard_route_checks
    );
    println!(
        "  route_weights_device_resident={}",
        stats.route_weights_device_resident
    );
    println!(
        "  reduction_device_resident={}",
        stats.reduction_device_resident
    );
}

fn print_runtime_selection_coverage_stats(
    label: &str,
    stats: &RuntimeSelectionCoverageReport,
) {
    if stats.domain_count == 0 {
        return;
    }
    println!("{label}:");
    println!("  domain_count={}", stats.domain_count);
    println!(
        "  selected_resources={}/{}",
        stats.selected_resource_count, stats.addressable_resource_count
    );
    println!("  selection_count={}", stats.selection_count);
    for domain in &stats.domains {
        println!(
            "  domain={}.{}.{} scope={} selected={}/{} selections={}",
            domain.component_id,
            domain.node_id,
            domain.domain_id,
            domain.execution_scope,
            domain.selected_resource_count,
            domain.resource_count,
            domain.selection_count,
        );
    }
}

fn print_runtime_selection_phase_coverage_stats(
    phases: &[(&str, &RuntimeSelectionCoverageReport)],
) {
    if phases.iter().all(|(_, report)| report.domain_count == 0) {
        return;
    }
    println!("selection_phases:");
    for (phase, report) in phases {
        println!(
            "  phase={} selected={}/{} selections={} domains={}",
            phase,
            report.selected_resource_count,
            report.addressable_resource_count,
            report.selection_count,
            report.domain_count,
        );
    }
}

fn print_runtime_prefix_state_cache_stats(stats: &VulkanResidentPlacedPrefixStateCacheStats) {
    println!("prefix_state_cache:");
    println!("  hits={}", stats.hit_count);
    println!("  misses={}", stats.miss_count);
    println!("  reused_tokens={}", stats.reused_token_count);
    println!(
        "  saved_prefill_tokens={}",
        stats.saved_prefill_token_count
    );
    println!("  insertions={}", stats.insertion_count);
    println!("  evictions={}", stats.eviction_count);
    println!("  resident_entries={}", stats.resident_entry_count);
    println!("  resident_bytes={}", stats.resident_byte_count);
}

fn print_runtime_feedback_stats(stats: &RuntimeFeedbackExecutionReport) {
    if stats.window_count == 0 {
        return;
    }
    println!("resident_feedback:");
    println!("  windows={}", stats.window_count);
    println!("  planned_ticks={}", stats.planned_tick_count);
    println!("  submitted_ticks={}", stats.submitted_tick_count);
    println!("  executed_ticks={}", stats.executed_tick_count);
    println!("  retained_ticks={}", stats.retained_tick_count);
    println!("  sampled_ticks={}", stats.sampled_tick_count);
    println!("  discarded_ticks={}", stats.discarded_tick_count);
    println!("  template_records={}", stats.template_record_count);
    println!("  template_replays={}", stats.template_replay_count);
    println!("  queue_submissions={}", stats.queue_submission_count);
    println!("  host_queue_submits={}", stats.host_queue_submit_count);
    println!(
        "  maximum_host_queue_submits_per_window={}",
        stats.maximum_host_queue_submit_count_per_window
    );
    println!(
        "  asynchronous_submissions={}",
        stats.asynchronous_submission_count
    );
    println!("  completion_polls={}", stats.completion_poll_count);
    println!("  bounded_waits={}", stats.bounded_wait_count);
    println!(
        "  bounded_wait_timeouts={}",
        stats.bounded_wait_timeout_count
    );
}

fn print_runtime_transport_edges(edges: &[RuntimePlacedTransportEdgeReport]) {
    if edges.is_empty() {
        return;
    }
    println!("transport_edges:");
    for edge in edges {
        println!(
            "  edge={} {}->{} signal={} route={} bytes={} transfers={} queue_signals={} queue_waits={} host_waits={} overlap_eligible={} overlap_submissions={} device_samples={} sampled_device_ms={:.3} estimated_device_ms={:.3} max_sampled_transfer_ms={:.3}",
            edge.edge_index,
            edge.from_device_id,
            edge.to_device_id,
            edge.signal,
            edge.route,
            edge.transferred_byte_count,
            edge.publish_count,
            edge.queue_signal_count,
            edge.queue_wait_count,
            edge.host_wait_count,
            edge.queue_overlap_eligible,
            edge.overlap_submission_count,
            edge.device_duration_sample_count,
            nanos_to_millis(edge.sampled_device_duration_ns),
            nanos_to_millis(edge.estimated_device_duration_ns),
            nanos_to_millis(edge.maximum_sampled_transfer_duration_ns),
        );
    }
}

fn print_runtime_speculative_stats(
    cycle_count: usize,
    rollback_cycle_count: usize,
    proposed_draft_token_count: usize,
    accepted_draft_token_count: usize,
    emitted_token_count: usize,
    draft_time_ns: u64,
    target_verification_time_ns: u64,
    draft_catch_up_time_ns: u64,
    total_time_ns: u64,
    windows: &[VulkanSpeculativeWindowStats],
    cycle_traces: &[VulkanSpeculativeCycleTrace],
) {
    if cycle_count == 0 {
        return;
    }
    let acceptance = if proposed_draft_token_count == 0 {
        0.0
    } else {
        100.0 * accepted_draft_token_count as f64 / proposed_draft_token_count as f64
    };
    println!("speculative:");
    println!("  cycles={cycle_count} rollback_cycles={rollback_cycle_count}");
    println!(
        "  drafts proposed={} accepted={} rejected={} acceptance={acceptance:.2}%",
        proposed_draft_token_count,
        accepted_draft_token_count,
        proposed_draft_token_count.saturating_sub(accepted_draft_token_count),
    );
    println!("  useful_tokens={emitted_token_count}");
    println!("  draft_ms={:.3}", nanos_to_millis(draft_time_ns));
    println!(
        "  target_verification_ms={:.3}",
        nanos_to_millis(target_verification_time_ns)
    );
    println!(
        "  draft_catch_up_ms={:.3}",
        nanos_to_millis(draft_catch_up_time_ns)
    );
    println!("  total_ms={:.3}", nanos_to_millis(total_time_ns));
    for window in windows {
        let accepted_throughput = if window.total_time_ns == 0 {
            0.0
        } else {
            window.emitted_token_count as f64
                / (window.total_time_ns as f64 / 1_000_000_000.0)
        };
        println!(
            "  window_{}=cycles:{},useful_tokens:{},total_ms:{:.3},accepted_tokens_per_second:{accepted_throughput:.3}",
            window.draft_width,
            window.cycle_count,
            window.emitted_token_count,
            nanos_to_millis(window.total_time_ns),
        );
    }
    for (index, trace) in cycle_traces.iter().enumerate() {
        println!(
            "  trace_{index}=tick:{} anchor:{} accepted:{} draft:{:?} target:{:?}",
            trace.start_stream_tick,
            trace.initial_token_id,
            trace.accepted_draft_count,
            trace.draft_token_ids,
            trace.target_token_ids,
        );
    }
}

fn print_placed_component_timing_profile(
    summaries: &[RuntimePlacedComponentTimingSummaryReport],
    max_rows: usize,
) {
    if summaries.is_empty() || max_rows == 0 {
        return;
    }
    println!("top_nodes:");
    for summary in summaries.iter().take(max_rows) {
        println!(
            "  {}:{} total_ms={:.3} ticks={} dispatches={} avg_tick_ms={} avg_dispatch_ms={}",
            summary.device_id,
            summary.component_id,
            nanos_to_millis(summary.total_run_time_ns),
            summary.tick_count,
            summary.dispatch_count,
            optional_nanos_to_millis(summary.average_tick_time_ns),
            optional_nanos_to_millis(summary.average_dispatch_time_ns)
        );
    }
}

fn optional_nanos_to_millis(value: Option<u64>) -> String {
    value
        .map(|nanos| format!("{:.3}", nanos_to_millis(nanos)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: nerve-runtime --package <COMPILED_PACKAGE.json> (--prompt <TEXT> | --chat) [OPTIONS]

Options:
  --package <PATH>           Compiled resident model package manifest. Required.
  --prompt <TEXT>            External text event to inject into the resident stream.
                             With --chat, this is the optional first message.
  --chat                     Start an interactive resident text session.
  --chat-template-var <NAME=JSON>
                             Set a model-owned chat template variable; may be repeated.
  --device <DEVICE_ID>       Default logical device for unplaced nodes. May be supplied once.
  --place-node <NODE=DEV>    Assign one runtime node instance to a logical device.
  --shard-component <NODE=DEV,DEV>
                             Shard eligible internal work while preserving the logical node boundary.
  --physical-strategy <NODE=STRATEGY>
                             Select one complete compiler-declared strategy for a manually sharded node.
                             STRATEGY: tensor_parallel, expert_parallel, or tensor_parallel_expert.
  --bind-device <DEV=TARGET> Bind a logical device to a discovered Vulkan device ID.
  --allow-physical-device <vulkan-uuid:UUID>
                             Restrict discovery and execution to this physical device; may be repeated.
  --chain <ITEM[,ITEM...]>    Runtime source chain. ITEM is SOURCE or INSTANCE=SOURCE.
  --duplicate-after <AFTER=NEW>
                             Duplicate runtime node instance AFTER with id NEW.
  --inspect-runtime          Preview UI-ready package, runtime graph, placement, device, and route facts.
  --inspect-package          Summarize the compiled component catalog and available devices.
  --inspect-graph            Preview the effective runtime graph without mounting devices.
  --inspect-placement        Mount and summarize every logical device slice in the runtime graph.
  --inspect-device-slice <DEVICE_ID>
                             Mount and summarize only the runtime graph nodes assigned to DEVICE_ID.
  --inspect-devices          Report physical-device compiler capabilities without loading a model.
  --initialize-device-contexts
                             With --inspect-devices, open and close every allowed execution context
                             before reporting so callers can attest the post-driver idle floor.
  --max-new-tokens <N>       Generation stop condition, independent of context size. Default: 65536
  --speculative-draft-tokens <N>
                             Compiled speculative-decoder tokens proposed per verification cycle.
                             Default: package recommendation when declared, otherwise disabled.
                             Pass 0 explicitly to disable an attached decoder.
  --speculative-confidence-threshold <F32>
                             Keep the contiguous proposed prefix whose compiled confidence is at
                             least this probability. Default: 0 (target verifies every proposal).
  --residency-policy <POLICY>
                             Parameter residency: eager, demand-retained, or demand-paged. Default: eager.
  --context-size <N>         Runtime transient-state window. Default: auto, up to the model maximum.
  --vulkan-device-index <N>  Use Vulkan physical device index N as the default local target.
  --seed <U32>               Explicit sampler randomness seed. Default: 0
  --temperature <F32>        Override the compiled sampler temperature for this stream.
  --top-k <N>                Override top-k, up to the package's compiled runtime capacity.
  --top-p <F32>              Override nucleus probability in (0, 1].
  --min-p <F32>              Override minimum relative token probability in [0, 1].
  --presence-penalty <F32>   Subtract this value once from logits of previously seen tokens.
  --repetition-penalty <F32> Override the positive multiplicative repetition penalty.
  --no-special-tokens        Do not add tokenizer special tokens to raw --prompt input.
                             Chat templates always own their complete special-token framing.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m nerve --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin nerve-runtime -- --package compiled_models/model_xxx/vulkan_resident_package.json --chat"
}
