struct VulkanResidentInProcessPlacedPendingFeedbackWindow {
    start_stream_tick: u64,
    tick_count: usize,
    terminal_output_value: u64,
    template_replayed: bool,
    transport_stats: VulkanPlacedEdgeTransportStats,
    demand_resolved_checkpoints: Vec<VulkanDemandFeedbackCheckpoint>,
    demand_resolution: Option<VulkanResidentDemandFeedbackResolutionState>,
}

struct VulkanResidentDemandFeedbackResolutionState {
    maximum_resolution_count: usize,
    resolved_resource_count: usize,
    resolved_checkpoints:
        BTreeMap<VulkanDemandFeedbackCheckpoint, BTreeSet<usize>>,
}

enum VulkanResidentFeedbackTerminalDisposition {
    Complete(VulkanResidentInProcessPlacedPendingFeedbackWindow),
    Resubmitted(VulkanResidentInProcessPlacedPendingFeedbackWindow),
}

struct VulkanResidentInProcessPlacedMountedFeedbackAttempt {
    submission_template: VulkanResidentQueueSubmissionTemplate,
    pending: VulkanResidentInProcessPlacedPendingFeedbackWindow,
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn reset_resident_feedback_session_state(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let Some(feedback_loop) = &self.resident_feedback_loop else {
            return Ok(());
        };
        feedback_loop
            .control
            .disarm_aborted_window()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if let Some(synchronization) = &feedback_loop.feedback_synchronization {
            // A new chat session has no causal dependency on the final token of
            // the previous one. Timeline semaphore values stay monotonic, but
            // the logical carry and any staged token/tick payload must not.
            synchronization.discard_aborted_turns();
        }
        if let Some(demand) = &feedback_loop.demand_residency {
            demand
                .reset_pipeline_predicate()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.distributed_dispatch_runners
                .reset_residency_predicates()
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::Tick(
                        VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                    )
                })?;
        }
        Ok(())
    }

    pub fn device(
        &self,
        device_id: &str,
    ) -> Option<&VulkanResidentInProcessPlacedStreamProcessorDevice> {
        self.device_slices
            .iter()
            .find(|slice| slice.device_id == device_id)
    }

    fn prepare_token_input(
        &self,
        input: VulkanResidentPlacedTokenInput,
    ) -> Result<VulkanResidentInputEmbeddingTransducerRun, VulkanResidentInProcessPlacedRuntimeError>
    {
        let token_id = input.token_id();
        match input {
            VulkanResidentPlacedTokenInput::HostSupplied(_) => self
                .input_transducer
                .prepare_token_id(token_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer),
            VulkanResidentPlacedTokenInput::ResidentFeedback(_) => {
                Ok(self.input_transducer.completed_run(token_id))
            }
            VulkanResidentPlacedTokenInput::EdgeFeedback(_) => {
                Ok(self.input_transducer.completed_run(token_id))
            }
        }
    }

    fn resident_feedback_next_window_tick_count(&self) -> usize {
        self.resident_feedback_loop
            .as_ref()
            .map(|feedback_loop| feedback_loop.window_policy.next_tick_count())
            .unwrap_or(0)
    }

    fn mount_resident_feedback_submission_template(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        feedback_synchronization: Option<&VulkanResidentPlacedFeedbackTimelineSynchronization>,
        output_synchronization: &VulkanResidentPlacedOutputTimelineSynchronization,
        demand_resume: Option<&VulkanPlacedDemandFeedbackTickResume>,
    ) -> Result<
        (
            VulkanResidentQueueSubmissionTemplate,
            Vec<u64>,
            VulkanPlacedEdgeTransportStats,
        ),
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let mut transport = VulkanInProcessPlacedEdgeTransport::new();
        let submission_batch = VulkanResidentQueueSubmissionBatch::new();
        let start_feedback_lane = demand_resume.map(|resume| resume.feedback_lane).unwrap_or(0);
        let continuation_lanes =
            demand_feedback_continuation_lanes(tick_count, start_feedback_lane)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut output_timeline_values =
            Vec::with_capacity(tick_count - start_feedback_lane);
        let mut transport_stats = VulkanPlacedEdgeTransportStats::default();
        for tick_index in continuation_lanes {
            let stream_tick =
                start_stream_tick
                    .checked_add(u64::try_from(tick_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let completes_window = tick_index + 1 == tick_count;
            let terminal_fault_copy = if completes_window {
                self.resident_feedback_loop
                    .as_ref()
                    .and_then(|feedback_loop| {
                        feedback_loop.demand_residency.as_ref().map(|demand| {
                            demand.terminal_fault_publication_copy(&feedback_loop.control)
                        })
                    })
                    .transpose()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
            } else {
                None
            };
            let mut slices = SmallVec::<
                [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>; 4],
            >::with_capacity(self.device_slices.len());
            for slice in &self.device_slices {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                let mut dispatch_extensions =
                    VulkanMountedPlacedResidentStreamTickDispatchExtensions {
                        sequence_variant: VulkanResidentPlacedTokenTickTail::Sample
                            .sequence_variant(),
                        ..Default::default()
                    };
                if slice.device_id == self.model.input_device_id {
                    dispatch_extensions
                        .prefix_dispatches
                        .push(&self.input_transducer.resident_dispatch);
                }
                if slice.device_id == self.model.output_device_id {
                    dispatch_extensions
                        .prefix_dispatches
                        .extend(self.sampler.input_tracking_dispatches());
                }
                if slice.device_id == self.model.output_device_id {
                    dispatch_extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.embedding_norm_dispatch);
                    dispatch_extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.tied_projection_dispatch);
                    dispatch_extensions
                        .suffix_dispatches
                        .extend(self.sampler.resident_dispatches());
                    dispatch_extensions
                        .suffix_dispatches
                        .push(self.sampler.feedback_control_dispatch());
                    if let Some(copy) = terminal_fault_copy {
                        dispatch_extensions.terminal_snapshot_copies.push(copy);
                    }
                }
                slices.push(
                    VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                        device,
                        &slice.mounted,
                        &slice.resident_execution_plan,
                        dispatch_extensions,
                        stream_tick,
                    ),
                );
            }
            let tick_resume = demand_resume.filter(|resume| resume.feedback_lane == tick_index);
            let feedback_turn = feedback_synchronization
                .map(|synchronization| match tick_resume {
                    Some(_) => {
                        synchronization.prepare_resumed_turn(&self.model.output_device_id)
                    }
                    None => synchronization.prepare_turn(&self.model.output_device_id),
                })
                .transpose()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let output_turn = output_synchronization
                .prepare_turn(&self.model.output_device_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            output_timeline_values.push(output_turn.value);
            let run = run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
                &mut slices,
                &mut transport,
                &self.activation_schedule,
                Some(&self.distributed_dispatch_runners),
                Some(&self.edge_synchronizations),
                VulkanPlacedSubmissionContext {
                    policy: VulkanPlacedSubmissionPolicy {
                        write_stream_control: false,
                        signal_completion: completes_window,
                        wait_for_completion: false,
                        feedback_lane: Some(tick_index),
                    },
                    participant_devices: Some(devices),
                    state_transactions: None,
                    feedback_turn,
                    output_turn: Some(output_turn),
                    demand_resume: tick_resume,
                    submission_batch: Some(&submission_batch),
                },
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)?;
            if run.status != VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed {
                return Err(VulkanResidentInProcessPlacedRuntimeError::IncompleteTick(
                    run.status,
                ));
            }
            if let Some(state) = &self.parallel_speculative_feedback_state {
                state.enqueue_source_tap_capture(
                    devices,
                    tick_index,
                    completes_window,
                    &submission_batch,
                )?;
            }
            if let Some(history) = &self.speculative_target_frame_history {
                let output_device =
                    devices.get(&self.model.output_device_id).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: self.model.output_device_id.clone(),
                        }
                    })?;
                let copy = history.lane_copies.get(tick_index).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "resident feedback target-frame lane {tick_index} exceeds history capacity {}",
                        history.lane_copies.len()
                    )))
                })?;
                submission_batch
                    .enqueue_resident_buffer_copy_batch(
                        output_device,
                        copy,
                        &[],
                        &[],
                        completes_window,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            transport_stats.accumulate(&run.transport_stats);
        }
        let queued_submission_count = submission_batch.pending_submission_count();
        let submission_template = submission_batch
            .mount()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        debug_assert_eq!(
            submission_template.submission_count(),
            queued_submission_count
        );
        Ok((
            submission_template,
            output_timeline_values,
            transport_stats,
        ))
    }

    fn advance_resident_feedback_submission_replay(
        &self,
        feedback_synchronization: Option<&VulkanResidentPlacedFeedbackTimelineSynchronization>,
        output_synchronization: &VulkanResidentPlacedOutputTimelineSynchronization,
        tick_count: usize,
    ) -> Result<Vec<u64>, VulkanResidentInProcessPlacedRuntimeError> {
        // Feedback eligibility requires a completed, bridged traversal of the
        // same graph for every tick. Each feedback edge, remote edge, and
        // distributed dispatch therefore advances once per replayed tick, so
        // the mounted queue template can use one uniform timeline offset.
        if let Some(feedback_synchronization) = feedback_synchronization {
            feedback_synchronization
                .advance_replayed_turns(tick_count)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        self.edge_synchronizations
            .advance_replayed_dependencies(tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.distributed_dispatch_runners
            .advance_replayed_dependency_values(tick_count)
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Tick(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                )
            })?;
        output_synchronization
            .reserve_replayed_turns(tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn resident_feedback_replay_timeline_state(
        &self,
        feedback_synchronization: Option<&VulkanResidentPlacedFeedbackTimelineSynchronization>,
        output_synchronization: &VulkanResidentPlacedOutputTimelineSynchronization,
    ) -> Result<VulkanTimelineSemaphoreReplayState, VulkanResidentInProcessPlacedRuntimeError> {
        let mut state = VulkanTimelineSemaphoreReplayState::default();
        if let Some(feedback_synchronization) = feedback_synchronization {
            feedback_synchronization
                .capture_replay_timeline_state(&mut state)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        self.edge_synchronizations
            .capture_replay_timeline_state(&mut state)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.distributed_dispatch_runners
            .capture_replay_timeline_state(&mut state)
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Tick(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                )
            })?;
        output_synchronization
            .capture_replay_timeline_state(&mut state)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(state)
    }

    fn submit_resident_feedback_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        input_token_id: u32,
        stop_token_ids: &[u32],
        template_catalog: Option<&mut VulkanResidentPlacedFeedbackTemplateCatalog>,
    ) -> Result<
        VulkanResidentInProcessPlacedPendingFeedbackWindow,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.prepare_resident_feedback_initial_control(input_token_id, start_stream_tick)?;
        let demand = self
            .resident_feedback_loop
            .as_ref()
            .and_then(|feedback_loop| feedback_loop.demand_residency.as_ref());
        let Some(demand) = demand else {
            return self.submit_resident_feedback_attempt(
                devices,
                start_stream_tick,
                tick_count,
                stop_token_ids,
                template_catalog,
            );
        };
        let attempt = (|| {
            let maximum_resolution_count = demand
                .resolution_bound(&self.device_slices, tick_count)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            // A cache miss may materialize a previously unseen demand-chain
            // shape. That construction allocates command resources and can
            // invoke reclamation. The preparation hook runs after either that
            // construction or a cached timeline rebase, but before submission,
            // so this is the authoritative execution-headroom boundary.
            let mut pending = self.submit_resident_feedback_attempt_after_preparation(
                devices,
                start_stream_tick,
                tick_count,
                stop_token_ids,
                template_catalog,
                |terminal_output_value| {
                    demand
                        .ensure_execution_headroom(devices)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    if self.resident_feedback_timeline_value_is_complete(
                        devices,
                        terminal_output_value,
                    )? {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "demand feedback terminal timeline value {terminal_output_value} was already complete before its submission",
                            )),
                        ));
                    }
                    Ok(())
                },
            )?;
            pending.demand_resolution = Some(VulkanResidentDemandFeedbackResolutionState {
                maximum_resolution_count,
                resolved_resource_count: 0,
                resolved_checkpoints: BTreeMap::new(),
            });
            Ok(pending)
        })();
        match attempt {
            Ok(pending) => Ok(pending),
            Err(error) => {
                if let Err(abort_error) = self.abort_demand_feedback_attempt() {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "demand feedback failed: {error}; transaction abort also failed: {abort_error}"
                        )),
                    ));
                }
                Err(error)
            }
        }
    }

    fn resolve_resident_feedback_terminal(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        mut pending: VulkanResidentInProcessPlacedPendingFeedbackWindow,
    ) -> Result<
        VulkanResidentFeedbackTerminalDisposition,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "resident feedback terminal resolution has no mounted feedback loop"
                    .to_string(),
            ))
        })?;
        let completion = feedback_loop
            .control
            .completion()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        match resident_feedback_terminal_state(completion, pending.tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
        {
            VulkanResidentFeedbackTerminalState::Complete => {
                Ok(VulkanResidentFeedbackTerminalDisposition::Complete(pending))
            }
            VulkanResidentFeedbackTerminalState::ResidencyFault => {
                let demand = feedback_loop.demand_residency.as_ref().ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "resident feedback published a residency fault without demand residency"
                            .to_string(),
                    ))
                })?;
                let state = pending.demand_resolution.as_mut().ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "resident feedback published a residency fault without resolution state"
                            .to_string(),
                    ))
                })?;
                let resolution = demand.resolve_published_fault(
                    &self.device_slices,
                    devices,
                    &self.distributed_dispatch_runners,
                    VulkanDemandFeedbackPublishedFault {
                        tick_count: pending.tick_count,
                        feedback_lane: completion.executed_tick_count,
                        source_id: completion.fault_source_id,
                        sequence_variant: VulkanResidentPlacedTokenTickTail::Sample
                            .sequence_variant(),
                    },
                )?;
                for (checkpoint, resource_indices) in &resolution.resolved {
                    state.resolved_resource_count = state
                        .resolved_resource_count
                        .checked_add(
                        record_demand_feedback_resolution(
                            &mut state.resolved_checkpoints,
                                *checkpoint,
                                resource_indices,
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    )
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "demand feedback resolved-resource count overflowed".to_string(),
                        ))
                    })?;
                }
                if state.resolved_resource_count > state.maximum_resolution_count {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "demand feedback exceeded its compiled resource-domain resolution bound of {}",
                            state.maximum_resolution_count,
                        )),
                    ));
                }
                pending.demand_resolved_checkpoints =
                    state.resolved_checkpoints.keys().copied().collect();
                let resume = self.demand_feedback_tick_resume(resolution.resume_checkpoint)?;
                let mut mounted = self.mount_demand_resident_feedback_continuation(
                    devices,
                    pending.start_stream_tick,
                    pending.tick_count,
                    &resume,
                )?;
                demand
                    .ensure_execution_headroom(devices)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                if self.resident_feedback_window_is_complete(devices, &mounted.pending)? {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "demand feedback continuation timeline value {} was already complete before its submission",
                            mounted.pending.terminal_output_value,
                        )),
                    ));
                }
                demand
                    .reset_pipeline_predicate()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                feedback_loop
                    .control
                    .acknowledge_residency_fault()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                mounted
                    .submission_template
                    .submit_with_timeline_value_offset(0)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                mounted
                    .pending
                    .transport_stats
                    .accumulate(&pending.transport_stats);
                mounted.pending.demand_resolved_checkpoints =
                    pending.demand_resolved_checkpoints;
                mounted.pending.demand_resolution = pending.demand_resolution;
                Ok(VulkanResidentFeedbackTerminalDisposition::Resubmitted(
                    mounted.pending,
                ))
            }
        }
    }

    fn demand_feedback_tick_resume(
        &self,
        checkpoint: VulkanDemandFeedbackCheckpoint,
    ) -> Result<VulkanPlacedDemandFeedbackTickResume, VulkanResidentInProcessPlacedRuntimeError>
    {
        let tick_plans = self
            .device_slices
            .iter()
            .map(|slice| slice.resident_execution_plan.tick_plan.as_ref())
            .collect::<Vec<_>>();
        let (target_slice_index, target_stage_index, local_gate_index, plan) = match checkpoint.target {
            VulkanDemandFeedbackCheckpointTarget::Local {
                slice_index,
                segment_index,
                gate_index,
            } => {
                let target_slice = self.device_slices.get(slice_index).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "demand feedback checkpoint slice {slice_index} is out of bounds"
                    )))
                })?;
                let target_segment = target_slice
                    .resident_execution_plan
                    .dispatch_segments
                    .get(segment_index)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                            "demand feedback checkpoint segment {segment_index} is out of bounds on {:?}",
                            target_slice.device_id
                        )))
                    })?;
                let gate_count = target_segment
                    .demand_residency
                    .as_ref()
                    .map(|demand| demand.gate_specs.len())
                    .unwrap_or(0);
                if gate_index >= gate_count {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "demand feedback gate {gate_index} is out of bounds for segment {segment_index} on {:?}",
                            target_slice.device_id
                        )),
                    ));
                }
                let target_stage_index = target_segment.start_stage_index;
                let plan = demand_feedback_resume_plan(
                    &tick_plans,
                    slice_index,
                    target_stage_index,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                (slice_index, target_stage_index, Some(gate_index), plan)
            }
            VulkanDemandFeedbackCheckpointTarget::Distributed {
                slice_index,
                dispatch_index,
                ..
            } => {
                let target_slice = self.device_slices.get(slice_index).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed demand feedback checkpoint slice {slice_index} is out of bounds"
                    )))
                })?;
                let (distributed_stage_index, _) = target_slice
                    .resident_execution_plan
                    .distributed_dispatch_stage_range(dispatch_index)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                            "distributed demand feedback dispatch {dispatch_index} has no stage on {:?}",
                            target_slice.device_id
                        )))
                    })?;
                let plan = demand_feedback_resume_plan_after_stage(
                    &tick_plans,
                    slice_index,
                    distributed_stage_index,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                (
                    slice_index,
                    distributed_stage_index + 1,
                    None,
                    plan,
                )
            }
        };
        Ok(VulkanPlacedDemandFeedbackTickResume {
            feedback_lane: checkpoint.feedback_lane,
            schedule_start_turn_index: plan.schedule_start_turn_index,
            next_stage_indices: plan.next_stage_indices,
            target_slice_index,
            target_stage_index,
            local_gate_index,
        })
    }

    fn mount_demand_resident_feedback_continuation(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        resume: &VulkanPlacedDemandFeedbackTickResume,
    ) -> Result<
        VulkanResidentInProcessPlacedMountedFeedbackAttempt,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "demand feedback continuation has no mounted feedback loop".to_string(),
            ))
        })?;
        let (submission_template, output_timeline_values, transport_stats) = self
            .mount_resident_feedback_submission_template(
                devices,
                start_stream_tick,
                tick_count,
                feedback_loop.feedback_synchronization.as_deref(),
                &feedback_loop.output_synchronization,
                Some(resume),
            )?;
        let terminal_output_value = output_timeline_values.last().copied().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "demand feedback continuation has no output timeline value".to_string(),
            ))
        })?;
        Ok(VulkanResidentInProcessPlacedMountedFeedbackAttempt {
            submission_template,
            pending: VulkanResidentInProcessPlacedPendingFeedbackWindow {
                start_stream_tick,
                tick_count,
                terminal_output_value,
                template_replayed: false,
                transport_stats,
                demand_resolved_checkpoints: Vec::new(),
                demand_resolution: None,
            },
        })
    }

    fn prepare_resident_feedback_initial_control(
        &self,
        input_token_id: u32,
        start_stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let dynamic_state_capacity_activations =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "resident feedback context capacity exceeds u32".to_string(),
                ))
            })?;
        let bytes = stream_control_bytes(
            input_token_id,
            VulkanMountedPlacedStreamControl {
                stream_tick: start_stream_tick,
                control_flags: 0,
                dynamic_state_capacity_activations,
            },
        );
        let mut initialized = Vec::<&Arc<VulkanResidentBuffer>>::new();
        for slice in &self.device_slices {
            let buffer = &slice.mounted.stream_control_buffer;
            if initialized.iter().any(|existing| {
                Arc::ptr_eq(existing, buffer)
                    || existing.shares_host_allocation_with(buffer)
            }) {
                continue;
            }
            // Imported device-local views share storage, not host-write
            // visibility. Initial feedback control is a host boundary, so each
            // physical device view must be explicitly initialized before its
            // queue begins the window. Shared-host aliases are one mapped
            // allocation and need only one write.
            buffer
                .write_bytes(&bytes)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            initialized.push(buffer);
        }
        Ok(())
    }

    fn abort_demand_feedback_attempt(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.resident_feedback_loop
            .as_ref()
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "demand feedback abort has no mounted feedback loop".to_string(),
                ))
            })?
            .control
            .disarm_aborted_window()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if let Some(feedback_synchronization) = self
            .resident_feedback_loop
            .as_ref()
            .and_then(|feedback_loop| feedback_loop.feedback_synchronization.as_deref())
        {
            feedback_synchronization.discard_aborted_turns();
        }
        Ok(())
    }

    fn submit_resident_feedback_attempt(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        stop_token_ids: &[u32],
        template_catalog: Option<&mut VulkanResidentPlacedFeedbackTemplateCatalog>,
    ) -> Result<
        VulkanResidentInProcessPlacedPendingFeedbackWindow,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.submit_resident_feedback_attempt_after_preparation(
            devices,
            start_stream_tick,
            tick_count,
            stop_token_ids,
            template_catalog,
            |_| Ok(()),
        )
    }

    fn submit_resident_feedback_attempt_after_preparation<F>(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        start_stream_tick: u64,
        tick_count: usize,
        stop_token_ids: &[u32],
        mut template_catalog: Option<&mut VulkanResidentPlacedFeedbackTemplateCatalog>,
        prepare_submission: F,
    ) -> Result<
        VulkanResidentInProcessPlacedPendingFeedbackWindow,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnOnce(u64) -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    {
        let feedback_loop = self.arm_resident_feedback_attempt(tick_count, stop_token_ids)?;
        let template_key = VulkanResidentPlacedFeedbackTemplateKey {
            runtime_execution_identity: self.model.runtime_execution_identity.clone(),
            tick_count,
        };
        let mut prepare_submission = Some(prepare_submission);
        let mut template_replayed = false;
        let (terminal_output_value, transport_stats) =
            if let Some(replay) = template_catalog
                .as_deref_mut()
                .and_then(|catalog| catalog.get(&template_key))
            {
                template_replayed = true;
                replay
                    .validate_tick_count(tick_count)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let current_timeline_state = self.resident_feedback_replay_timeline_state(
                    feedback_loop.feedback_synchronization.as_deref(),
                    &feedback_loop.output_synchronization,
                )?;
                let output_timeline_values = self.advance_resident_feedback_submission_replay(
                    feedback_loop.feedback_synchronization.as_deref(),
                    &feedback_loop.output_synchronization,
                    tick_count,
                )?;
                let terminal_output_value =
                    output_timeline_values.last().copied().ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "resident feedback replay has no output timeline value".to_string(),
                        ))
                    })?;
                prepare_submission
                    .take()
                    .expect("resident feedback submission preparation runs exactly once")(
                    terminal_output_value,
                )?;
                replay
                    .submit_next(tick_count, &current_timeline_state)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                (terminal_output_value, replay.transport_stats.clone())
            } else {
                let recorded_timeline_state = self.resident_feedback_replay_timeline_state(
                    feedback_loop.feedback_synchronization.as_deref(),
                    &feedback_loop.output_synchronization,
                )?;
                let (submission_template, output_timeline_values, transport_stats) = self
                    .mount_resident_feedback_submission_template(
                        devices,
                        start_stream_tick,
                        tick_count,
                        feedback_loop.feedback_synchronization.as_deref(),
                        &feedback_loop.output_synchronization,
                        None,
                    )?;
                let terminal_output_value =
                    output_timeline_values.last().copied().ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "resident feedback window has no output timeline value".to_string(),
                        ))
                    })?;
                prepare_submission
                    .take()
                    .expect("resident feedback submission preparation runs exactly once")(
                    terminal_output_value,
                )?;
                submission_template
                    .submit_with_timeline_value_offset(0)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                if let Some(catalog) = template_catalog {
                    catalog.insert(
                        template_key,
                        VulkanResidentPlacedFeedbackSubmissionReplay::new(
                            submission_template,
                            tick_count,
                            recorded_timeline_state,
                            transport_stats.clone(),
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                }
                (terminal_output_value, transport_stats)
            };
        Ok(VulkanResidentInProcessPlacedPendingFeedbackWindow {
            start_stream_tick,
            tick_count,
            terminal_output_value,
            template_replayed,
            transport_stats,
            demand_resolved_checkpoints: Vec::new(),
            demand_resolution: None,
        })
    }

    fn arm_resident_feedback_attempt(
        &self,
        tick_count: usize,
        stop_token_ids: &[u32],
    ) -> Result<
        &VulkanResidentInProcessPlacedFeedbackLoop,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        if tick_count < 2 || tick_count > feedback_loop.window_policy.maximum_tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "placed resident feedback window requests {tick_count} ticks, mounted width is {}",
                    feedback_loop.window_policy.maximum_tick_count
                )),
            ));
        }
        feedback_loop
            .control
            .arm(tick_count, stop_token_ids)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(feedback_loop)
    }

    fn resident_feedback_output_device<'a>(
        &self,
        devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError> {
        devices
            .get(&self.model.output_device_id)
            .map(Rc::as_ref)
            .ok_or_else(
                || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.output_device_id.clone(),
                },
            )
    }

    fn resident_feedback_window_is_complete(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        pending: &VulkanResidentInProcessPlacedPendingFeedbackWindow,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        self.resident_feedback_timeline_value_is_complete(
            devices,
            pending.terminal_output_value,
        )
    }

    fn resident_feedback_timeline_value_is_complete(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        terminal_output_value: u64,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        feedback_loop
            .output_synchronization
            .turn_is_complete(
                self.resident_feedback_output_device(devices)?,
                terminal_output_value,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn resident_feedback_window_completion_point<'a>(
        &'a self,
        pending: &VulkanResidentInProcessPlacedPendingFeedbackWindow,
    ) -> Result<VulkanTimelineSemaphorePoint<'a>, VulkanResidentInProcessPlacedRuntimeError> {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        Ok(feedback_loop
            .output_synchronization
            .turn_point(pending.terminal_output_value))
    }

    fn wait_resident_feedback_window_for(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        pending: &VulkanResidentInProcessPlacedPendingFeedbackWindow,
        timeout_ns: u64,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        feedback_loop
            .output_synchronization
            .wait_for_turn_for(
                self.resident_feedback_output_device(devices)?,
                pending.terminal_output_value,
                timeout_ns,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn complete_resident_feedback_window<F>(
        &self,
        pending: VulkanResidentInProcessPlacedPendingFeedbackWindow,
        mut on_sampled_token: F,
    ) -> Result<
        VulkanResidentFeedbackControlCompletion,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    where
        F: FnMut(
            usize,
            VulkanResidentSampledToken,
            usize,
            usize,
            bool,
            &VulkanPlacedEdgeTransportStats,
        )
            -> Result<(), VulkanResidentInProcessPlacedRuntimeError>,
    {
        let feedback_loop = self.resident_feedback_loop.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "placed resident feedback loop is not mounted".to_string(),
            ))
        })?;
        // The output timeline signal is recorded after the terminal output
        // slice. Every upstream slice, distributed shard, and transfer is a
        // semaphore dependency of that slice, so this one wait is the graph
        // completion proof. Waiting every slice fence again only serialized
        // host control on already-completed work.
        let mut completion = feedback_loop
            .control
            .completion()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        completion.template_replayed = pending.template_replayed;
        if resident_feedback_terminal_state(completion, pending.tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
            != VulkanResidentFeedbackTerminalState::Complete
        {
            let demand_diagnostic = feedback_loop
                .demand_residency
                .as_ref()
                .map(|demand| demand.pipeline_predicate_diagnostic())
                .transpose()
                .map(|values| format!("{values:?}"))
                .unwrap_or_else(|error| format!("diagnostic_error={error}"));
            let completion_after_predicate_read = feedback_loop
                .control
                .completion()
                .map(|completion| {
                    format!(
                        "executed={},sampled={},stop={},fault={}",
                        completion.executed_tick_count,
                        completion.sampled_tick_count,
                        completion.stop_reason,
                        completion.fault_reason,
                    )
                })
                .unwrap_or_else(|error| format!("diagnostic_error={error}"));
            let feedback_control_header = feedback_loop
                .control
                .diagnostic_header_words()
                .map(|words| format!("{words:?}"))
                .unwrap_or_else(|error| format!("diagnostic_error={error}"));
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback control reported invalid terminal state stop={} fault={} after {} of {} ticks; resolved_checkpoints={:?}; pipeline_predicates={demand_diagnostic}; completion_after_predicate_read={completion_after_predicate_read}; feedback_control_header={feedback_control_header}",
                    completion.stop_reason,
                    completion.fault_reason,
                    completion.executed_tick_count,
                    pending.tick_count,
                    pending.demand_resolved_checkpoints,
                )),
            ));
        }
        if completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_CANCELLED {
            feedback_loop.control.acknowledge_cancellation();
        }
        let no_transport = VulkanPlacedEdgeTransportStats::default();
        for tick_index in 0..completion.sampled_tick_count {
            let stream_tick = pending
                .start_stream_tick
                .checked_add(u64::try_from(tick_index).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let sampled_token = self
                .sampler
                .completed_run_at(stream_tick)
                .map(|run| VulkanResidentSampledToken::from(&run))
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
            on_sampled_token(
                tick_index,
                sampled_token,
                feedback_loop.scheduler_turn_count_per_tick,
                feedback_loop.completed_stage_count_per_tick,
                completion.stop_reason == VULKAN_FEEDBACK_STOP_REASON_CANCELLED
                    && tick_index + 1 == completion.sampled_tick_count,
                if tick_index == 0 {
                    &pending.transport_stats
                } else {
                    &no_transport
                },
            )?;
        }
        Ok(completion)
    }

}
