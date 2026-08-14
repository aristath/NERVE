#[allow(clippy::too_many_arguments)]
fn mount_placed_chat_stream(
    args: &Args,
    manifest_dir: &Path,
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    parameter_pool: &VulkanResidentBufferPool,
    mut runtime_model: VulkanResidentRuntimeModel,
    capacity: usize,
    speculative_draft_tokens: usize,
    physical_execution_plan: Option<VulkanRuntimePhysicalExecutionPlan>,
    placement_calibration_catalog: Option<&VulkanPlacementCalibrationCatalog>,
    retained_stores: Option<&VulkanRetainedCompiledResourceStores>,
) -> Result<VulkanResidentInProcessPlacedPromptStream, Box<dyn Error>> {
    let package = &mut runtime_model.package;
    package.sampler.spec = sampler_runtime_config(args).apply_to(&package.sampler.spec)?;
    let physical_execution_plan = match physical_execution_plan {
        Some(plan) => plan,
        None => explicit_runtime_physical_execution_plan(args, &runtime_model)?,
    };
    let package = Arc::new(
        VulkanResidentInProcessPlacedModelPackage::from_runtime_model_for_bound_devices_with_physical_execution_plan(
            devices,
            manifest_dir,
            runtime_model,
            physical_execution_plan,
            placement_calibration_catalog,
            Some(capacity),
            speculative_draft_tokens,
            args.resource_residency_policy,
            parameter_pool,
            retained_stores,
        )?,
    );
    Ok(
        VulkanResidentInProcessPlacedPromptStream::new(package, devices.clone(), args.random_seed)?
            .with_speculative_draft_tokens(speculative_draft_tokens)?
            .with_speculative_confidence_threshold(args.speculative_confidence_threshold)?,
    )
}

fn explicit_runtime_physical_execution_plan(
    args: &Args,
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<VulkanRuntimePhysicalExecutionPlan, Box<dyn Error>> {
    Ok(
        VulkanRuntimePhysicalExecutionPlan::uniform(runtime_model)
            .with_explicit_distributed_overrides(
                runtime_model,
                &args.component_shard_devices,
                &args.component_physical_strategies,
            )?,
    )
}

fn overlay_explicit_runtime_physical_execution(
    args: &Args,
    runtime_model: &VulkanResidentRuntimeModel,
    automatic_plan: Option<VulkanRuntimePhysicalExecutionPlan>,
) -> Result<Option<VulkanRuntimePhysicalExecutionPlan>, Box<dyn Error>> {
    if args.component_shard_devices.is_empty() {
        return Ok(automatic_plan);
    }
    let plan = automatic_plan
        .unwrap_or_else(|| VulkanRuntimePhysicalExecutionPlan::uniform(runtime_model))
        .with_explicit_distributed_overrides(
            runtime_model,
            &args.component_shard_devices,
            &args.component_physical_strategies,
        )?;
    Ok(Some(plan))
}

/// Applies only the stable-owner part of a caller-declared physical shard
/// pool. The first participant is the component's boundary coordinator; the
/// remaining participants stay phase-local in the physical execution plan.
/// This happens after automatic capacity packing so a local TP request does
/// not disable automatic placement for the rest of the graph, while also
/// avoiding the impossible requirement that a caller predict the measured
/// planner's owner choice.
fn runtime_model_with_explicit_shard_owners(
    args: &Args,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    if args.component_shard_devices.is_empty() {
        return Ok(runtime_model);
    }
    let default_device_id = runtime_model.placement.default_device_id.clone();
    let mut owner_by_component = runtime_model
        .runtime_graph
        .instances
        .iter()
        .map(|instance| (instance.instance_id.clone(), instance.device_id.clone()))
        .collect::<BTreeMap<_, _>>();
    for (component_id, device_ids) in &args.component_shard_devices {
        let owner = device_ids.first().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("explicit shard pool for component {component_id:?} is empty"),
            )
        })?;
        owner_by_component.insert(component_id.clone(), owner.clone());
    }
    vulkan_runtime_model_with_component_placement_owned(
        runtime_model,
        &default_device_id,
        &owner_by_component,
    )
    .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn signal_processor_owner_constraints(
    runtime_model: &VulkanResidentRuntimeModel,
) -> BTreeMap<String, String> {
    runtime_model
        .circuit_graph
        .components
        .iter()
        .filter(|component| component.runtime_role.is_signal_processor())
        .map(|component| {
            (
                component.component_id.clone(),
                runtime_model
                    .placement
                    .device_for_component(&component.component_id)
                    .to_string(),
            )
        })
        .collect()
}

fn resolve_runtime_hybrid_physical_execution(
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    execution: RuntimeExecutionEnvelope,
    auto_placement: Option<&RuntimeAutoPlacementContext>,
    bound_devices: &RuntimeBoundVulkanDevices,
    context_capacity_activations: usize,
    speculative_draft_tokens: usize,
    residency_policy: ResourceResidencyPolicy,
    required_owner_by_component: Option<&BTreeMap<String, String>>,
) -> Result<
    (
        VulkanResidentRuntimeModel,
        Option<VulkanRuntimePhysicalExecutionPlan>,
    ),
    Box<dyn Error>,
> {
    let Some(auto_placement) = auto_placement else {
        return Ok((runtime_model, None));
    };
    let mut available_bytes_by_device = BTreeMap::new();
    for candidate in &auto_placement.candidates {
        let device = bound_devices
            .devices
            .get(&candidate.device_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "hybrid placement candidate {:?} has no bound Vulkan device",
                        candidate.device_id,
                    ),
                )
            })?;
        let identity = VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: device.physical_device_id().to_string(),
            api_version: device.api_version(),
            driver_version: device.driver_version(),
        };
        let reservable_bytes =
            usize::try_from(device.device_local_memory_budget().reservable_bytes)
                .unwrap_or(usize::MAX);
        if available_bytes_by_device
            .insert(identity, reservable_bytes)
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "hybrid placement requires one logical candidate per physical device",
            )
            .into());
        }
    }
    let capacity = VulkanPlacementCapacityEnvelope {
        available_bytes_by_device,
        host_available_bytes: vulkan_safe_host_available_bytes()?,
    };
    let logical_device_id_by_physical_device = bound_devices
        .physical_device_ids
        .iter()
        .map(|(logical_device_id, physical_device_id)| {
            (physical_device_id.clone(), logical_device_id.clone())
        })
        .collect::<BTreeMap<_, _>>();
    if logical_device_id_by_physical_device.len() != bound_devices.physical_device_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "hybrid placement requires one-to-one physical device bindings",
        )
        .into());
    }
    let physical_mount_devices = bound_devices
        .devices
        .iter()
        .map(|(logical_device_id, device)| VulkanRuntimePhysicalPlanningDevice {
            logical_device_id: logical_device_id.clone(),
            identity: VulkanPlacementDeviceExecutionIdentity {
                physical_device_id: device.physical_device_id().to_string(),
                api_version: device.api_version(),
                driver_version: device.driver_version(),
            },
            safe_capacity_bytes: usize::try_from(
                device.device_local_memory_budget().reservable_bytes,
            )
            .unwrap_or(usize::MAX),
            storage_buffer_offset_alignment: device.min_storage_buffer_offset_alignment(),
        })
        .collect::<Vec<_>>();
    let Some(resolution) = resolve_vulkan_runtime_hybrid_physical_execution_with_representations(
        manifest_dir,
        &auto_placement.exact_runtime_model,
        &bound_devices.hardware_profiles,
        execution,
        &auto_placement.calibration_catalog,
        &capacity,
        context_capacity_activations,
        &logical_device_id_by_physical_device,
        &physical_mount_devices,
        speculative_draft_tokens,
        residency_policy,
        capacity.host_available_bytes,
        required_owner_by_component,
    )?
    else {
        eprintln!(
            "nerve runtime hybrid placement unavailable for the current exact device identities and capacity; preserving stable scalar/serialized placement"
        );
        return Ok((runtime_model, None));
    };
    eprintln!(
        "nerve runtime hybrid placement: decode_predicted_ns_per_activation={}, prefill_width={:?}, prefill_predicted_ns_per_activation={:?}, devices={:?}",
        resolution.decode_predicted_duration_ns_per_activation,
        resolution.prefill_activation_batch_width,
        resolution.prefill_predicted_duration_ns_per_activation,
        resolution
            .physical_execution_plan
            .device_ids(&resolution.runtime_model),
    );
    Ok((
        resolution.runtime_model,
        Some(resolution.physical_execution_plan),
    ))
}

fn run_placed_chat(
    args: &Args,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    auto_placement: Option<RuntimeAutoPlacementContext>,
    capacity: usize,
    codec: &VulkanResidentHfTokenizerTextCodec,
    initial_prompt: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let setup_start = Instant::now();
    let mut auto_placement = auto_placement;
    let runtime_model = if !args.component_shard_devices.is_empty()
        && let Some(auto_placement) = &mut auto_placement
    {
        // The automatically selected representation was chosen for the
        // unconstrained owner profile. Re-enter selection from the exact
        // compiled model after applying the caller's local owner constraint;
        // carrying the previous representation across a heterogeneous owner
        // change would be a false compatibility assumption.
        let constrained = runtime_model_with_explicit_shard_owners(
            args,
            auto_placement.exact_runtime_model.clone(),
        )?;
        auto_placement.exact_runtime_model = constrained.clone();
        constrained
    } else {
        runtime_model_with_explicit_shard_owners(args, runtime_model)?
    };
    let speculative_draft_tokens = effective_speculative_draft_tokens(args, &runtime_model)?;
    let chat_session =
        RuntimeChatSession::from_tokenizer_dir(tokenizer_dir, &args.chat_template_variables)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        manifest_dir,
        tokenizer_dir,
        &runtime_model.package,
        &chat_session.formatter,
    )?;
    let transcript_codec = chat_transcript_codec(tokenizer_dir)?;
    let mut logical_device_ids = runtime_model.placement_device_ids();
    if auto_placement.is_some() {
        logical_device_ids.extend(
            auto_placement
                .as_ref()
                .expect("checked above")
                .candidates
                .iter()
                .map(|candidate| candidate.device_id.clone()),
        );
        logical_device_ids.sort();
        logical_device_ids.dedup();
    }
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let execution = RuntimeExecutionEnvelope {
        phases: vec!["decode".to_string(), "prefill".to_string()],
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
        speculative_draft_tokens,
        residency_policy: args
            .resource_residency_policy
            .as_runtime_name()
            .replace('-', "_"),
    };
    let runtime_model =
        if runtime_model.implementation_selection.is_some() {
            runtime_model
        } else {
            runtime_model
                .select_and_apply_runtime_implementations(
                    manifest_dir,
                    &bound_devices.hardware_profiles,
                    execution.clone(),
                )?
                .0
        };
    let required_owner_by_component = (!args.component_shard_devices.is_empty())
        .then(|| signal_processor_owner_constraints(&runtime_model));
    let (runtime_model, automatic_physical_execution_plan) =
        resolve_runtime_hybrid_physical_execution(
            manifest_dir,
            runtime_model,
            execution,
            auto_placement.as_ref(),
            &bound_devices,
            capacity,
            speculative_draft_tokens,
            args.resource_residency_policy,
            required_owner_by_component.as_ref(),
        )?;
    let physical_execution_plan = overlay_explicit_runtime_physical_execution(
        args,
        &runtime_model,
        automatic_physical_execution_plan,
    )?;
    let implementation_selection = runtime_model
        .implementation_selection
        .clone()
        .ok_or_else(|| io::Error::other("placed runtime has no implementation selection"))?;
    let sparse_moe_contract = runtime_model.sparse_moe_execution_contract()?;
    let device_restoration_before =
        capture_vulkan_device_local_memory_restoration_snapshots(
            bound_devices.devices.values().map(Rc::as_ref),
        )?;
    let parameter_pool = VulkanResidentBufferPool::default();
    let stream = mount_placed_chat_stream(
        args,
        manifest_dir,
        &bound_devices.devices,
        &parameter_pool,
        runtime_model.clone(),
        capacity,
        speculative_draft_tokens,
        physical_execution_plan.clone(),
        auto_placement
            .as_ref()
            .map(|placement| &placement.calibration_catalog),
        None,
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let initial_selection = engine
        .stream("main")
        .ok_or_else(|| io::Error::other("placed chat engine lost its mounted main stream"))?
        .selection_telemetry_snapshot()?;
    let mut working_set_baseline = engine
        .stream("main")
        .expect("mounted main stream remains present")
        .package()
        .working_set_pressure_snapshot(&initial_selection)?;
    let mut mounted_runtime_model = runtime_model;
    let mounted_physical_execution_plan = physical_execution_plan;
    let auto_placement = auto_placement;
    let mut conversation_activation_count = 0u64;
    let mounted_device_bindings = bound_devices
        .physical_device_ids
        .iter()
        .map(|(logical_device_id, physical_device_id)| {
            format!("{logical_device_id}={physical_device_id}")
        })
        .collect::<Vec<_>>();
    let exact_calibration_observation_count = auto_placement
        .as_ref()
        .map(|placement| placement.calibration_catalog.observation_count())
        .unwrap_or(0);
    println!(
        "nerve chat ready: placed_in_process, devices={:?}, bindings={:?}, context_size={}, speculative_draft_tokens={}, residency_policy={}, exact_calibration_observations={}, physical_execution={:?}, setup_ms={:.3}",
        stream_snapshot.device_ids,
        mounted_device_bindings,
        stream_snapshot.context_window_activations,
        speculative_draft_tokens,
        args.resource_residency_policy.as_runtime_name(),
        exact_calibration_observation_count,
        stream_snapshot.physical_execution,
        nanos_to_millis(elapsed_nanos_u64(setup_start))
    );

    let chat_result = (|| -> Result<(), Box<dyn Error>> {
        let mut chat_session = chat_session;
        let mut pending_initial_prompt = initial_prompt;
        loop {
            let outcome = run_chat_repl(
                pending_initial_prompt.take(),
                chat_session,
                codec,
                &transcript_codec,
                |turn_index, chat_session, input_text, prepared| {
                    print!("llm> ");
                    io::stdout().flush()?;
                    let mut decoder = codec.decode_stream();
                    let mut protocol_validator = chat_session
                        .formatter
                        .assistant_stream_protocol_validator(&transcript_codec)?;
                    let generation_context_start = chat_session
                        .committed_token_ids
                        .len()
                        .saturating_add(prepared.user_token_delta.len())
                        .saturating_add(prepared.generation_prompt_token_delta.len());
                    let mut previous_output_at = None;
                    let mut sustained_decode_samples = Vec::new();
                    let selection_before = engine
                        .stream("main")
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::NotFound,
                                "placed chat engine lost its main stream",
                            )
                        })?
                        .selection_telemetry_snapshot()?;
                    let mut selection_after_user = None;
                    let mut selection_after_generation_branch = None;
                    let mut selection_after_canonical_commit = None;
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
                            if let Some(validator) = protocol_validator.as_mut() {
                                match validator.observe(output_event.output_event.token_id)? {
                                    RuntimeAssistantStreamProtocolAction::Continue => {}
                                    RuntimeAssistantStreamProtocolAction::TerminateAndTrim {
                                        token_count,
                                    } => {
                                        return Ok(
                                            RuntimeChatGeneratedOutputControl::TerminateAndTrim {
                                                token_count,
                                            },
                                        );
                                    }
                                }
                            }
                            let output_at = Instant::now();
                            if let Some(previous) = previous_output_at {
                                sustained_decode_samples.push(RuntimeSustainedDecodeSample {
                                    context_activation: generation_context_start
                                        .saturating_add(output_event.output_event.output_index),
                                    transient_state_activation: output_event
                                        .output_event
                                        .source_stream_tick,
                                    inter_token_time_ns: u64::try_from(
                                        output_at.duration_since(previous).as_nanos(),
                                    )
                                    .unwrap_or(u64::MAX),
                                });
                            }
                            previous_output_at = Some(output_at);
                            match decoder.step(output_event.output_event.token_id) {
                                Ok(Some(text)) => {
                                    print!("{text}");
                                    io::stdout().flush()?;
                                }
                                Ok(None) => {}
                                Err(error) => return Err(Box::new(error)),
                            }
                            Ok(RuntimeChatGeneratedOutputControl::Continue)
                        },
                        |phase, engine| {
                            let snapshot = engine
                                .stream("main")
                                .ok_or_else(|| {
                                    io::Error::new(
                                        io::ErrorKind::NotFound,
                                        "placed chat engine lost its main stream",
                                    )
                                })?
                                .selection_telemetry_snapshot()?;
                            match phase {
                                VulkanResidentChatTransactionPhase::UserCommitted => {
                                    selection_after_user = Some(snapshot);
                                }
                                VulkanResidentChatTransactionPhase::GenerationBranchCompleted => {
                                    selection_after_generation_branch = Some(snapshot);
                                }
                                VulkanResidentChatTransactionPhase::CanonicalTurnCommitted => {
                                    selection_after_canonical_commit = Some(snapshot);
                                }
                            }
                            Ok(())
                        },
                    )?;
                    let selection_after_user = selection_after_user.ok_or_else(|| {
                        io::Error::other("placed chat transaction did not report its user phase")
                    })?;
                    let selection_after_generation_branch =
                selection_after_generation_branch.ok_or_else(|| {
                    io::Error::other(
                        "placed chat transaction did not report its generation-branch phase",
                    )
                })?;
                    let selection_after = selection_after_canonical_commit.ok_or_else(|| {
                        io::Error::other(
                            "placed chat transaction did not report its canonical-commit phase",
                        )
                    })?;
                    let selection_user_coverage = selection_after_user
                        .delta_since(&selection_before)?
                        .report();
                    let selection_generation_branch_coverage = selection_after_generation_branch
                        .delta_since(&selection_after_user)?
                        .report();
                    let selection_canonical_commit_coverage = selection_after
                        .delta_since(&selection_after_generation_branch)?
                        .report();
                    let selection_post_branch_cumulative =
                        selection_after_generation_branch.report();
                    let selection_coverage =
                        selection_after.delta_since(&selection_before)?.report();
                    let selection_counter_digest = selection_after.digest();
                    let cumulative_selection_coverage = selection_after.report();
                    let resident_state_digest = engine.stream_resident_state_digest("main")?;
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
                        Some(&transaction.user_run.engine_run),
                        Some(&transaction.generation_run.engine_run),
                        transaction
                            .canonical_commit_run
                            .as_ref()
                            .map(|run| &run.engine_run),
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                    let prefill_activation_count: usize = engine_runs
                        .iter()
                        .map(|run| run.prefill_activation_count)
                        .sum();
                    let decode_activation_count: usize = engine_runs
                        .iter()
                        .map(|run| run.decode_activation_count)
                        .sum();
                    conversation_activation_count = conversation_activation_count
                        .checked_add(
                            u64::try_from(
                                prefill_activation_count.saturating_add(decode_activation_count),
                            )
                            .unwrap_or(u64::MAX),
                        )
                        .ok_or_else(|| {
                            io::Error::other("conversation activation count overflowed")
                        })?;
                    let timing = runtime_prompt_timing_report(
                        0,
                        transaction.elapsed_ns,
                        prepared
                            .user_token_delta
                            .len()
                            .saturating_add(prepared.generation_prompt_token_delta.len())
                            .saturating_add(transaction.assistant_token_delta.len()),
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
                    let prefix_state_cache = transaction
                        .canonical_commit_run
                        .as_ref()
                        .map(|run| &run.engine_run)
                        .unwrap_or(&transaction.generation_run.engine_run)
                        .prefix_state_cache
                        .clone();
                    let speculative_decode = submitted_run
                        .submitted_run
                        .session_run
                        .run
                        .speculative_decode
                        .clone();
                    let resident_feedback = runtime_feedback_execution_report(
                        submitted_run
                            .submitted_run
                            .session_run
                            .run
                            .resident_feedback
                            .clone(),
                    );
                    let transport_edges = runtime_placed_transport_edge_reports(
                        &submitted_run.submitted_run.session_run.run.transport_stats,
                    );
                    let generated_token_digest = token_id_digest(&transaction.generated_token_ids);
                    let resource_residency = engine
                        .stream("main")
                        .ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::NotFound,
                                "placed chat engine lost its main stream",
                            )
                        })?
                        .package()
                        .compiled_resource_residency_report(&cumulative_selection_coverage)?;
                    let critical_path = transaction.critical_path.with_normalization(
                        timing.generated_token_count,
                        timing.scheduler_turn_count,
                    );
                    Ok(RuntimeChatTurn {
                        generated_token_ids: transaction.generated_token_ids,
                        assistant_message: transaction.assistant_message,
                        canonical_committed_token_ids: transaction.canonical_committed_token_ids,
                        canonical_commit_mode: transaction.canonical_commit_mode,
                        generated_token_digest,
                        selection_counter_digest,
                        resident_state_digest,
                        streamed: true,
                        timing,
                        sustained_decode: RuntimeSustainedDecodeReport::from_samples(
                            &sustained_decode_samples,
                        ),
                        implementation_selection: implementation_selection.clone(),
                        execution_counters: transaction.execution_counters,
                        critical_path,
                        prefix_state_cache,
                        speculative_cycle_count: speculative_decode.cycle_count,
                        speculative_rollback_cycle_count: speculative_decode.rollback_cycle_count,
                        proposed_draft_token_count: speculative_decode.proposed_draft_token_count,
                        accepted_draft_token_count: speculative_decode.accepted_draft_token_count,
                        speculative_emitted_token_count: speculative_decode.emitted_token_count,
                        speculative_draft_time_ns: speculative_decode.draft_time_ns,
                        speculative_target_verification_time_ns: speculative_decode
                            .target_verification_time_ns,
                        speculative_draft_catch_up_time_ns: speculative_decode
                            .draft_catch_up_time_ns,
                        speculative_total_time_ns: speculative_decode.total_time_ns,
                        speculative_windows: speculative_decode.windows.into_values().collect(),
                        speculative_cycle_traces: speculative_decode.cycle_traces,
                        resident_feedback,
                        sparse_moe: sparse_moe_contract
                            .work_report(prefill_activation_count, decode_activation_count),
                        selection_user_coverage,
                        selection_generation_branch_coverage,
                        selection_canonical_commit_coverage,
                        selection_post_branch_cumulative,
                        selection_coverage,
                        cumulative_selection_coverage,
                        transport_edges,
                        resource_residency,
                    })
                },
            )?;
            match outcome {
                RuntimeChatReplOutcome::Exit => return Ok(()),
                RuntimeChatReplOutcome::NewConversation => {
                    let current_selection = engine
                        .stream("main")
                        .ok_or_else(|| {
                            io::Error::other(
                                "placed chat engine lost its main stream at session boundary",
                            )
                        })?
                        .selection_telemetry_snapshot()?;
                    let current_pressure = engine
                        .stream("main")
                        .expect("session-boundary main stream remains present")
                        .package()
                        .working_set_pressure_snapshot(&current_selection)?;
                    let interval_pressure = current_pressure.delta_since(&working_set_baseline)?;
                    let rebalance = if mounted_physical_execution_plan.is_some() {
                        None
                    } else {
                        auto_placement
                            .as_ref()
                            .map(|placement| {
                                rebalance_demand_paged_vulkan_runtime_model_from_working_set(
                                    manifest_dir,
                                    &mounted_runtime_model,
                                    &placement.candidates,
                                    &placement.costs,
                                    &current_pressure,
                                    &interval_pressure,
                                    conversation_activation_count,
                                    capacity,
                                    speculative_draft_tokens,
                                )
                            })
                            .transpose()?
                            .flatten()
                    };
                    let zeroed = if let Some(rebalance) = rebalance {
                        eprintln!(
                            "nerve runtime working-set rebalance: moved_components={:?}, predicted_ns_per_activation={}=>{}, observed_blocking_ns={}, estimated_remount_ns={}, estimated_net_benefit_ns={}",
                            rebalance.moved_component_ids,
                            rebalance.current_predicted_ns_per_activation,
                            rebalance.proposed_predicted_ns_per_activation,
                            rebalance.observed_blocking_ns,
                            rebalance.estimated_remount_ns,
                            rebalance.estimated_net_benefit_ns,
                        );
                        let new_runtime_model = rebalance.placement.runtime_model;
                        let release = engine.release_stream_for_session_remount(
                            "main",
                            &rebalance.retained_logical_device_ids,
                        )?;
                        let stream = mount_placed_chat_stream(
                            args,
                            manifest_dir,
                            &bound_devices.devices,
                            &parameter_pool,
                            new_runtime_model.clone(),
                            capacity,
                            speculative_draft_tokens,
                            None,
                            None,
                            Some(&release.retained_stores),
                        )?;
                        engine.add_stream("main", stream)?;
                        parameter_pool.evict_unreferenced();
                        mounted_runtime_model = new_runtime_model;
                        let selection = engine
                            .stream("main")
                            .expect("remounted main stream is present")
                            .selection_telemetry_snapshot()?;
                        working_set_baseline = engine
                            .stream("main")
                            .expect("remounted main stream is present")
                            .package()
                            .working_set_pressure_snapshot(&selection)?;
                        println!(
                            "session_remount: released_units={} released_payload_bytes={}",
                            release.teardown.released_unit_count,
                            release.teardown.released_payload_bytes,
                        );
                        0
                    } else {
                        let zeroed =
                            engine.reset_stream_for_new_session("main", args.random_seed)?;
                        working_set_baseline = current_pressure;
                        zeroed
                    };
                    conversation_activation_count = 0;
                    chat_session = RuntimeChatSession::from_tokenizer_dir(
                        tokenizer_dir,
                        &args.chat_template_variables,
                    )?;
                    println!("session_reset: zeroed_state_buffers={zeroed}");
                }
            }
        }
    })();
    let shutdown = engine.shutdown();
    print_runtime_shutdown(&shutdown);
    drop(parameter_pool);
    let device_restoration = quiesce_and_verify_vulkan_device_local_memory_restoration(
        bound_devices.devices.values().map(Rc::as_ref),
        &device_restoration_before,
    );
    print_runtime_device_restoration(&device_restoration);
    let teardown_complete = shutdown.complete && device_restoration.complete;
    match (chat_result, teardown_complete) {
        (Ok(()), true) => Ok(()),
        (Err(error), true) => Err(Box::new(io::Error::other(format!(
            "placed chat failed: {error}; teardown acknowledged on {}/{} physical resource devices and restored {}/{} selected physical devices with {} scheduler activations remaining",
            shutdown.acknowledged_device_count,
            shutdown.physical_device_count,
            device_restoration.restored_device_count,
            device_restoration.physical_device_count,
            shutdown.scheduler_in_flight_activation_count,
        )))),
        (Ok(()), false) => Err(Box::new(io::Error::other(format!(
            "placed chat teardown failed: resource_errors={:?}, device_restoration_errors={:?}",
            shutdown.errors,
            device_restoration.errors,
        )))),
        (Err(error), false) => Err(Box::new(io::Error::other(format!(
            "placed chat failed: {error}; teardown also failed: resource_errors={:?}, device_restoration_errors={:?}",
            shutdown.errors,
            device_restoration.errors,
        )))),
    }
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
    let speculative_draft_tokens = effective_speculative_draft_tokens(args, &runtime_model)?;
    let logical_device_ids = runtime_model.placement_device_ids();
    let sparse_moe_contract = runtime_model.sparse_moe_execution_contract()?;
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config_and_residency_policy(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(*capacity),
        args.random_seed,
        speculative_draft_tokens,
        sampler_runtime_config(args),
        args.resource_residency_policy,
    )?
    .with_speculative_confidence_threshold(args.speculative_confidence_threshold)?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);
    let run_result = (|| -> Result<RuntimePlacedPromptRunReport, Box<dyn Error>> {
        reset_vulkan_resident_execution_counters();
        reset_runtime_critical_path_counters();
        let run_start = Instant::now();
        let protocol_span = runtime_critical_path_span(RuntimeCriticalPathPhase::Protocol);
        let selection_before = engine
            .stream("main")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "placed prompt engine lost its main stream",
                )
            })?
            .selection_telemetry_snapshot()?;
        let input_event =
            VulkanResidentTokenInputEvent::new("prompt", prompt_ids.to_vec(), args.max_new_tokens);
        let input_event_id = input_event.id.clone();
        let submitted_run = engine.submit_input_event_until_idle("main", input_event)?;
        let selection_after = engine
            .stream("main")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "placed prompt engine lost its main stream",
                )
            })?
            .selection_telemetry_snapshot()?;
        let selection_coverage = selection_after.delta_since(&selection_before)?.report();
        let cumulative_selection_coverage = selection_after.report();
        let resource_residency = engine
            .stream("main")
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "placed prompt engine lost its main stream",
                )
            })?
            .package()
            .compiled_resource_residency_report(&cumulative_selection_coverage)?;
        drop(protocol_span);
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
            .find(|run| {
                run.stream_id == "main" && run.submitted_run.input_event.id == input_event_id
            })
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
        let critical_path = runtime_critical_path_report(run_time_ns)
            .with_normalization(generated_token_count, total_scheduler_turns);

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
            critical_path,
            component_timings,
            component_timing_summaries,
            speculative_cycle_count: run.speculative_decode.cycle_count,
            speculative_rollback_cycle_count: run.speculative_decode.rollback_cycle_count,
            proposed_draft_token_count: run.speculative_decode.proposed_draft_token_count,
            accepted_draft_token_count: run.speculative_decode.accepted_draft_token_count,
            speculative_emitted_token_count: run.speculative_decode.emitted_token_count,
            speculative_draft_time_ns: run.speculative_decode.draft_time_ns,
            speculative_target_verification_time_ns: run
                .speculative_decode
                .target_verification_time_ns,
            speculative_draft_catch_up_time_ns: run.speculative_decode.draft_catch_up_time_ns,
            speculative_total_time_ns: run.speculative_decode.total_time_ns,
            speculative_windows: run.speculative_decode.windows.values().cloned().collect(),
            speculative_cycle_traces: run.speculative_decode.cycle_traces.clone(),
            resident_feedback: runtime_feedback_execution_report(run.resident_feedback),
            sparse_moe: sparse_moe_contract
                .work_report(prefill_activation_count, decode_activation_count),
            selection_coverage,
            resource_residency,
            shutdown: Default::default(),
        })
    })();
    let shutdown = engine.shutdown();
    match (run_result, shutdown.complete) {
        (Ok(mut report), true) => {
            report.shutdown = shutdown;
            Ok(report)
        }
        (Err(error), true) => Err(Box::new(io::Error::other(format!(
            "placed prompt failed: {error}; teardown acknowledged on {}/{} physical devices with {} scheduler activations remaining",
            shutdown.acknowledged_device_count,
            shutdown.physical_device_count,
            shutdown.scheduler_in_flight_activation_count,
        )))),
        (Ok(_), false) => Err(Box::new(io::Error::other(format!(
            "placed prompt teardown failed: {:?}",
            shutdown.errors,
        )))),
        (Err(error), false) => Err(Box::new(io::Error::other(format!(
            "placed prompt failed: {error}; teardown also failed: {:?}",
            shutdown.errors,
        )))),
    }
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
            device_duration_sample_count: edge.device_duration_sample_count,
            sampled_device_duration_ns: edge.sampled_device_duration_ns,
            estimated_device_duration_ns: edge.estimated_device_duration_ns,
            maximum_sampled_transfer_duration_ns: edge.maximum_sampled_transfer_duration_ns,
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
        queue_submission_count: stats.queue_submission_count,
        host_queue_submit_count: stats.host_queue_submit_count,
        maximum_host_queue_submit_count_per_window: stats
            .maximum_host_queue_submit_count_per_window,
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
        print_runtime_critical_path(&report.critical_path);
        print_runtime_feedback_stats(&report.resident_feedback);
        print_runtime_sparse_moe_stats(&report.sparse_moe);
        print_runtime_selection_coverage_stats("selection_coverage", &report.selection_coverage);
        print_runtime_resource_residency(&report.resource_residency);
        print_runtime_shutdown(&report.shutdown);
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
        &report.speculative_windows,
        &report.speculative_cycle_traces,
    );
}

fn generated_tokens_per_second(generated_token_count: usize, run_time_ns: u64) -> Option<f64> {
    if run_time_ns == 0 {
        None
    } else {
        Some(generated_token_count as f64 / (run_time_ns as f64 / 1_000_000_000.0))
    }
}
