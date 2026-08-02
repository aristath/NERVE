#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeChatTurn {
    generated_token_ids: Vec<u32>,
    canonical_committed_token_ids: Vec<u32>,
    generated_token_digest: String,
    selection_counter_digest: String,
    resident_state_digest: String,
    streamed: bool,
    timing: RuntimePromptTimingReport,
    sustained_decode: RuntimeSustainedDecodeReport,
    implementation_selection: RuntimeImplementationSelectionReport,
    execution_counters: VulkanResidentExecutionCounters,
    prefix_state_cache: VulkanResidentPlacedPrefixStateCacheStats,
    speculative_cycle_count: usize,
    speculative_rollback_cycle_count: usize,
    proposed_draft_token_count: usize,
    accepted_draft_token_count: usize,
    speculative_emitted_token_count: usize,
    speculative_draft_time_ns: u64,
    speculative_target_verification_time_ns: u64,
    speculative_draft_catch_up_time_ns: u64,
    speculative_total_time_ns: u64,
    resident_feedback: RuntimeFeedbackExecutionReport,
    transport_edges: Vec<RuntimePlacedTransportEdgeReport>,
    sparse_moe: RuntimeSparseMoeWorkReport,
    selection_user_coverage: RuntimeSelectionCoverageReport,
    selection_generation_coverage: RuntimeSelectionCoverageReport,
    selection_commit_coverage: RuntimeSelectionCoverageReport,
    selection_post_generation_cumulative: RuntimeSelectionCoverageReport,
    selection_coverage: RuntimeSelectionCoverageReport,
    cumulative_selection_coverage: RuntimeSelectionCoverageReport,
    resource_residency: VulkanCompiledResourceResidencyReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuntimeSustainedDecodeSample {
    context_activation: usize,
    transient_state_activation: u64,
    inter_token_time_ns: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RuntimeSustainedDecodeReport {
    measured_token_count: usize,
    windows: Vec<RuntimeSustainedDecodeWindow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeSustainedDecodeWindow {
    ordinal: usize,
    context_activation_start: usize,
    context_activation_end: usize,
    transient_state_activation_start: u64,
    transient_state_activation_end: u64,
    token_count: usize,
    elapsed_ns: u64,
}

impl RuntimeSustainedDecodeReport {
    fn from_samples(
        samples: &[RuntimeSustainedDecodeSample],
    ) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut windows = Vec::new();
        for ordinal in 0..4 {
            let start = ordinal * samples.len() / 4;
            let end = (ordinal + 1) * samples.len() / 4;
            if start == end {
                continue;
            }
            let slice = &samples[start..end];
            windows.push(RuntimeSustainedDecodeWindow {
                ordinal: windows.len(),
                context_activation_start:
                    slice[0].context_activation,
                context_activation_end: slice
                    .last()
                    .expect("non-empty sustained window")
                    .context_activation,
                transient_state_activation_start:
                    slice[0].transient_state_activation,
                transient_state_activation_end: slice
                    .last()
                    .expect("non-empty sustained window")
                    .transient_state_activation,
                token_count: slice.len(),
                elapsed_ns: slice.iter().fold(
                    0u64,
                    |elapsed, sample| {
                        elapsed.saturating_add(
                            sample.inter_token_time_ns,
                        )
                    },
                ),
            });
        }
        Self {
            measured_token_count: samples.len(),
            windows,
        }
    }
}

fn run_chat_repl<C, T, F>(
    initial_prompt: Option<&str>,
    mut chat_session: RuntimeChatSession,
    codec: &C,
    transcript_codec: &T,
    stop_token_ids: &[u32],
    mut submit: F,
) -> Result<(), Box<dyn Error>>
where
    C: VulkanResidentTokenTextCodec,
    T: VulkanResidentTokenTextCodec,
    F: FnMut(
        usize,
        &RuntimeChatSession,
        &str,
        &RuntimePreparedChatTurn,
    ) -> Result<RuntimeChatTurn, Box<dyn Error>>,
{
    println!(
        "Type a message and press Enter. Type /exit, /quit, exit, or quit to stop."
    );
    let mut turn_index = 0usize;
    if let Some(initial_prompt) = initial_prompt
        && !initial_prompt.trim().is_empty()
    {
        if !submit_chat_turn(
            &mut chat_session,
            codec,
            transcript_codec,
            stop_token_ids,
            &mut submit,
            turn_index,
            initial_prompt,
        )? {
            return Ok(());
        }
        turn_index = turn_index.saturating_add(1);
    }

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("you> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let input_text = line.trim_end_matches(['\r', '\n']);
        let command = input_text.trim();
        if command.eq_ignore_ascii_case("exit")
            || command.eq_ignore_ascii_case("quit")
            || command.eq_ignore_ascii_case("/exit")
            || command.eq_ignore_ascii_case("/quit")
        {
            break;
        }
        if command.is_empty() {
            continue;
        }

        if !submit_chat_turn(
            &mut chat_session,
            codec,
            transcript_codec,
            stop_token_ids,
            &mut submit,
            turn_index,
            input_text,
        )? {
            break;
        }
        turn_index = turn_index.saturating_add(1);
    }
    Ok(())
}

fn submit_chat_turn<C, T, F>(
    chat_session: &mut RuntimeChatSession,
    codec: &C,
    transcript_codec: &T,
    stop_token_ids: &[u32],
    submit: &mut F,
    turn_index: usize,
    input_text: &str,
) -> Result<bool, Box<dyn Error>>
where
    C: VulkanResidentTokenTextCodec,
    T: VulkanResidentTokenTextCodec,
    F: FnMut(
        usize,
        &RuntimeChatSession,
        &str,
        &RuntimePreparedChatTurn,
    ) -> Result<RuntimeChatTurn, Box<dyn Error>>,
{
    let prepared =
        chat_session.prepare_user_turn(input_text, transcript_codec)?;
    match submit(turn_index, chat_session, input_text, &prepared) {
        Ok(turn) => {
            let generated_text =
                codec.decode_tokens(&turn.generated_token_ids)?;
            let assistant_content_ids = assistant_content_token_ids(
                &turn.generated_token_ids,
                stop_token_ids,
            );
            let assistant_content =
                transcript_codec.decode_tokens(assistant_content_ids)?;
            if turn.streamed {
                println!();
            } else {
                print_chat_response(&generated_text);
            }
            print_runtime_timing_stats("stats", &turn.timing);
            print_runtime_sustained_decode_stats(
                &turn.sustained_decode,
            );
            print_runtime_implementation_selection(
                &turn.implementation_selection,
            );
            print_runtime_execution_counters(&turn.execution_counters);
            print_runtime_prefix_state_cache_stats(
                &turn.prefix_state_cache,
            );
            print_runtime_speculative_stats(
                turn.speculative_cycle_count,
                turn.speculative_rollback_cycle_count,
                turn.proposed_draft_token_count,
                turn.accepted_draft_token_count,
                turn.speculative_emitted_token_count,
                turn.speculative_draft_time_ns,
                turn.speculative_target_verification_time_ns,
                turn.speculative_draft_catch_up_time_ns,
                turn.speculative_total_time_ns,
            );
            print_runtime_feedback_stats(&turn.resident_feedback);
            print_runtime_sparse_moe_stats(&turn.sparse_moe);
            print_runtime_selection_phase_coverage_stats(
                &[
                    ("user", &turn.selection_user_coverage),
                    ("generation", &turn.selection_generation_coverage),
                    ("commit", &turn.selection_commit_coverage),
                    (
                        "post_generation_cumulative",
                        &turn.selection_post_generation_cumulative,
                    ),
                ],
            );
            print_runtime_selection_coverage_stats(
                "selection_coverage",
                &turn.selection_coverage,
            );
            print_runtime_selection_coverage_stats(
                "cumulative_selection_coverage",
                &turn.cumulative_selection_coverage,
            );
            print_runtime_transport_edges(&turn.transport_edges);
            print_runtime_resource_residency(
                &turn.resource_residency,
            );
            println!("determinism:");
            println!(
                "  generated_tokens={}",
                turn.generated_token_digest
            );
            println!(
                "  selection_counters={}",
                turn.selection_counter_digest
            );
            println!("  resident_state={}", turn.resident_state_digest);
            chat_session.commit_assistant_turn(
                input_text,
                &assistant_content,
                turn.canonical_committed_token_ids,
            );
            Ok(true)
        }
        Err(error) => Err(error),
    }
}

fn print_runtime_resource_residency(
    report: &VulkanCompiledResourceResidencyReport,
) {
    let totals = &report.totals;
    println!("resource_residency:");
    println!(
        "  policy={} physical_stores={} device_bytes(initial/current/high_water/maximum)={}/{}/{}/{}",
        report.policy.as_runtime_name(),
        totals.physical_store_count,
        totals.initial_device_bytes,
        totals.current_device_bytes,
        totals.high_water_device_bytes,
        totals.maximum_device_bytes
    );
    println!(
        "  payload_bytes(initial/current/high_water/maximum)={}/{}/{}/{} units(initial/current/high_water/addressable)={}/{}/{}/{}",
        totals.initial_payload_bytes,
        totals.current_payload_bytes,
        totals.high_water_payload_bytes,
        totals.maximum_payload_bytes,
        totals.initial_resident_unit_count,
        totals.resident_unit_count,
        totals.high_water_resident_unit_count,
        totals.addressable_unit_count
    );
    println!(
        "  fixed_device_bytes(always_parameters/runtime_working_set/resource_metadata)={}/{}/{} transfer_staging_host_bytes={}",
        totals.always_resident_parameter_bytes,
        totals.runtime_working_set_device_bytes,
        totals.metadata_device_bytes,
        totals.transfer_staging_host_bytes
    );
    println!(
        "  gpu_accesses(selections/resident_hits/misses)={}/{}/{}",
        totals.gpu_selection_count,
        totals.gpu_resident_hit_count,
        totals.gpu_miss_count,
    );
    println!(
        "  residency_requests(directory_hits/load_required/deduplicated/succeeded/failed/cancelled)={}/{}/{}/{}/{}/{}",
        totals.residency_directory_hit_count,
        totals.residency_load_required_count,
        totals.deduplicated_load_count,
        totals.successful_load_count,
        totals.failed_load_count,
        totals.cancelled_load_count
    );
    println!(
        "  residency_eviction(cycles/units/payload_bytes/device_bytes/reloads)={}/{}/{}/{}/{}",
        totals.eviction_count,
        totals.evicted_unit_count,
        totals.evicted_payload_bytes,
        totals.released_device_bytes,
        totals.reload_count,
    );
    println!(
        "  transfers(reads/read_bytes/uploaded_bytes/read_ms/upload_ms/blocking_ms)={}/{}/{}/{:.3}/{:.3}/{:.3}",
        totals.physical_read_count,
        totals.physical_bytes_read,
        totals.uploaded_bytes,
        nanos_to_millis(totals.read_time_ns),
        nanos_to_millis(totals.upload_time_ns),
        nanos_to_millis(totals.blocking_time_ns)
    );
    println!(
        "  target components={} units={}/{} gpu_accesses={}/{}/{}",
        report.target.component_count,
        report.target.resident_unit_count,
        report.target.addressable_unit_count,
        report.target.gpu_selection_count,
        report.target.gpu_resident_hit_count,
        report.target.gpu_miss_count,
    );
    for scope in &report.mtp {
        println!(
            "  mtp scope={} components={} units={}/{} gpu_accesses={}/{}/{}",
            scope.execution_scope,
            scope.component_count,
            scope.resident_unit_count,
            scope.addressable_unit_count,
            scope.gpu_selection_count,
            scope.gpu_resident_hit_count,
            scope.gpu_miss_count,
        );
    }
    for store in &report.stores {
        println!(
            "  store={} physical_device={} logical_devices={:?} device_bytes={}/{}/{} units={}/{} loading={} failed={}",
            store.store_id,
            store.physical_device_id,
            store.logical_device_ids,
            store.current_device_bytes,
            store.high_water_device_bytes,
            store.maximum_device_bytes,
            store.resident_unit_count,
            store.addressable_unit_count,
            store.loading_unit_count,
            store.failed_unit_count
        );
    }
}

fn print_runtime_shutdown(
    report: &VulkanPlacedPromptEngineShutdownReport,
) {
    println!("shutdown:");
    println!(
        "  complete={} streams={} packages={} scheduler_in_flight={}",
        report.complete,
        report.stream_count,
        report.package_count,
        report.scheduler_in_flight_activation_count,
    );
    println!(
        "  physical_devices_acknowledged={}/{} released_units={} released_payload_bytes={} cancelled_loads={}",
        report.acknowledged_device_count,
        report.physical_device_count,
        report.released_unit_count,
        report.released_payload_bytes,
        report.cancelled_load_count,
    );
    for teardown in &report.resource_teardowns {
        println!(
            "  package={} scope={} physical_devices_acknowledged={}/{}",
            teardown.package_id,
            teardown.execution_scope,
            teardown.acknowledged_device_count,
            teardown.physical_device_count,
        );
        for device in &teardown.devices {
            println!(
                "  store={} physical_device={} acknowledged={} remaining_units={} remaining_payload_bytes={} error={:?}",
                device.store_id,
                device.physical_device_id,
                device.acknowledged,
                device.remaining_unit_count,
                device.remaining_payload_bytes,
                device.error,
            );
        }
    }
}

fn token_id_digest(token_ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};

    let mut digest = Sha256::new();
    digest.update((token_ids.len() as u64).to_le_bytes());
    for token_id in token_ids {
        digest.update(token_id.to_le_bytes());
    }
    format!(
        "nerve.runtime.token_ids_sha256.v1:{:x}",
        digest.finalize()
    )
}

fn assistant_content_token_ids<'a>(
    generated_token_ids: &'a [u32],
    stop_token_ids: &[u32],
) -> &'a [u32] {
    let mut content_len = generated_token_ids.len();
    while content_len > 0
        && stop_token_ids.contains(
            &generated_token_ids[content_len - 1],
        )
    {
        content_len -= 1;
    }
    &generated_token_ids[..content_len]
}

fn print_chat_response(text: &str) {
    print!("llm> ");
    print_text(text);
}
