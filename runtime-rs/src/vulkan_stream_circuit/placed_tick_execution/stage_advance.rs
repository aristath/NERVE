fn advance_compact_slice_with_distributed_dependencies<'a, 'batch>(
    slice: &mut VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>,
    device_by_id: &BTreeMap<String, &'a VulkanComputeDevice>,
    transport: &mut VulkanInProcessPlacedEdgeTransport,
    distributed_runners: &VulkanDistributedDispatchRunners,
    edge_synchronizations: &'a VulkanPlacedEdgeTimelineSynchronizations,
    submission: VulkanPlacedSliceSubmissionContext<'a, 'batch>,
) -> Result<usize, VulkanMountedPlacedResidentInProcessStreamTickError> {
    let VulkanPlacedSliceSubmissionContext {
        policy: submission_policy,
        state_transaction,
        feedback_turn,
        output_turn,
        demand_resume,
        submission_batch,
    } = submission;
    let can_wait_submitted = submission_policy.signal_completion && submission_batch.is_none();
    debug_assert!(!slice.cursor.capture_execution_trace);
    let dynamic_state_capacity_activations =
        u32::try_from(slice.mounted.buffers.dynamic_state_capacity_activations).map_err(|_| {
            VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                VulkanMountedPlacedResidentStreamTickError::DynamicStateCapacityOverflow {
                    capacity: slice.mounted.buffers.dynamic_state_capacity_activations,
                },
            )
        })?;
    let control = VulkanMountedPlacedStreamControl {
        stream_tick: slice.cursor.stream_tick,
        control_flags: 0,
        dynamic_state_capacity_activations,
    };
    let completed_before = slice.cursor.completed_stage_count;
    let mut last_submitted_segment: Option<&VulkanMountedPlacedResidentDispatchSegmentRunner> =
        None;
    let mut ready_dependency = None;
    let mut completion_dependency = None;
    let mut pending_edge_wait_points = Vec::new();

    while slice.cursor.next_stage_index < slice.cursor.tick_plan.stages.len() {
        let stage = &slice.cursor.tick_plan.stages[slice.cursor.next_stage_index];
        match stage {
            VulkanMountedPlacedStreamTickStage::ReceiveEdge { edge_index, .. } => {
                let incoming = slice
                    .mounted
                    .edge_io
                    .incoming_buffer(*edge_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                            VulkanMountedPlacedResidentStreamTickError::Transport(
                                VulkanPlacedEdgeTransportError::MissingIncomingEdge {
                                    device_id: slice.device_id().to_string(),
                                    edge_index: *edge_index,
                                },
                            ),
                        )
                    })?;
                let edge_key =
                    VulkanPlacedEdgePacketKey::from_incoming_endpoint(&incoming.endpoint);
                let uses_queue_transfer = transport.edge_uses_queue_transfer(&edge_key);
                if !uses_queue_transfer {
                    transport.record_host_wait(&edge_key);
                    if submission_batch.is_some() {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                VulkanError(format!(
                                    "deferred placed submission requires shared edge {edge_key:?}"
                                )),
                            ),
                        );
                    }
                    wait_for_compact_slice_submitted_work(
                        slice,
                        device_by_id,
                        distributed_runners,
                        last_submitted_segment.take(),
                        completion_dependency.take(),
                        submission_policy.feedback_lane,
                        can_wait_submitted,
                    )?;
                }
                match transport.receive_incoming_edge(slice.mounted, *edge_index) {
                    Ok(_) => {
                        if edge_synchronizations.edge_uses_device_local_staging(&edge_key) {
                            if !edge_synchronizations
                                .enqueue_destination_staging_transfer(
                                    &incoming.endpoint,
                                    slice.device,
                                    submission_batch,
                                )
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?
                            {
                                return Err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                        VulkanError(format!(
                                            "staged edge {edge_key:?} has no destination transfer"
                                        )),
                                    ),
                                );
                            }
                            transport.record_queue_wait(&edge_key);
                        } else if uses_queue_transfer
                            && let Some(wait_point) = edge_synchronizations
                                .take_destination_wait(&incoming.endpoint)
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?
                        {
                            transport.record_queue_wait(&edge_key);
                            pending_edge_wait_points.push(wait_point);
                        }
                        slice.cursor.complete_current_stage();
                    }
                    Err(VulkanPlacedEdgeTransportError::MissingPacket { .. }) => break,
                    Err(error) => {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                                VulkanMountedPlacedResidentStreamTickError::Transport(error),
                            ),
                        );
                    }
                }
            }
            VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {
                if let Some(distributed) = slice
                    .execution_plan
                    .distributed_dispatch_at_stage(slice.cursor.next_stage_index)
                {
                    let dependencies = slice
                        .execution_plan
                        .distributed_dispatch_dependencies_at_stage(slice.cursor.next_stage_index)
                        .expect("every distributed stage has a dependency topology");
                    debug_assert_eq!(dependencies.dispatch_index, distributed.dispatch_index);
                    let consumes_ready = ready_dependency.is_some_and(|(dispatch_index, _)| {
                        dispatch_index == distributed.dispatch_index
                    });
                    if consumes_ready != dependencies.has_owner_producer {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                VulkanError(format!(
                                    "distributed dispatch {} on device {:?} expected owner producer={}, but queued ready dependency={consumes_ready}",
                                    distributed.dispatch_index,
                                    slice.device_id(),
                                    dependencies.has_owner_producer
                                )),
                            ),
                        );
                    }
                    let (completion_bridge, completion_staging) = if dependencies
                        .has_owner_continuation
                    {
                        (None, false)
                    } else if let Some(VulkanMountedPlacedStreamTickStage::PublishEdge {
                        edge_index,
                        ..
                    }) = slice.cursor.tick_plan.stages.get(
                        slice
                            .execution_plan
                            .physical_execution_island_at_stage(slice.cursor.next_stage_index)
                            .expect("every distributed dispatch belongs to a stage group")
                            .end_stage_index,
                    ) {
                        let outgoing = slice
                            .mounted
                            .edge_io
                            .outgoing_buffer(*edge_index)
                            .ok_or_else(|| {
                                VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                                    VulkanMountedPlacedResidentStreamTickError::Transport(
                                        VulkanPlacedEdgeTransportError::MissingOutgoingEdge {
                                            device_id: slice.device_id().to_string(),
                                            edge_index: *edge_index,
                                        },
                                    ),
                                )
                            })?;
                        let edge_key =
                            VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
                        if edge_synchronizations.edge_uses_device_local_staging(&edge_key) {
                            (None, true)
                        } else if !transport.edge_uses_shared_storage(&edge_key) {
                            (None, false)
                        } else {
                            let signal_point = edge_synchronizations
                                .prepare_source_signal(&outgoing.endpoint)
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?
                                .ok_or_else(|| {
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                        VulkanError(format!(
                                            "distributed boundary edge {edge_key:?} has shared storage without queue synchronization"
                                        )),
                                    )
                                })?;
                            transport.record_queue_signal(&edge_key);
                            (Some(signal_point), false)
                        }
                    } else {
                        (None, false)
                    };
                    let requires_residency_checkpoint = distributed_runners
                        .requires_residency_checkpoint(
                            slice.device_id(),
                            distributed.dispatch_index,
                        );
                    let requires_host_checkpoint =
                        requires_residency_checkpoint && submission_batch.is_none();
                    let has_owner_continuation =
                        dependencies.has_owner_continuation && !requires_host_checkpoint;
                    if !has_owner_continuation
                        && completion_bridge.is_none()
                        && !completion_staging
                        && (!submission_policy.signal_completion || submission_batch.is_some())
                    {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                VulkanError(format!(
                                    "unfenced placed window cannot end at distributed dispatch {} on {:?}",
                                    distributed.dispatch_index,
                                    slice.device_id()
                                )),
                            ),
                        );
                    }
                    let dependency_value = if let Some((_, dependency_value)) = ready_dependency {
                        dependency_value
                    } else {
                        distributed_runners
                            .reserve_dependency_value(slice.device_id(), distributed.dispatch_index)
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Distributed,
                            )?
                    };
                    let submission = distributed_runners.submit_dispatch_with_device_dependencies(
                        slice.device_id(),
                        distributed.dispatch_index,
                        VulkanDistributedDispatchSubmission {
                            dependency_value,
                            consume_owner_ready_signal: consumes_ready,
                            prepare_owner_continuation: has_owner_continuation
                                || (!requires_host_checkpoint
                                    && (completion_bridge.is_some() || completion_staging)),
                            signal_completion: requires_host_checkpoint
                                || (submission_policy.signal_completion
                                    && submission_batch.is_none()),
                            sequence_kind:
                                VulkanDistributedDispatchSequenceKind::for_feedback_lane(
                                    submission_policy.feedback_lane,
                                ),
                        },
                        submission_batch,
                        |device_id| {
                            device_by_id.get(device_id).copied().ok_or_else(|| {
                                VulkanError(format!(
                                    "distributed shard device {device_id:?} is not mounted"
                                ))
                            })
                        },
                    );
                    if let Err(error) = submission {
                        if can_wait_submitted && let Some(segment) = last_submitted_segment {
                            let _ = segment.wait_submitted(
                                slice.device,
                                slice.dispatch_extensions.sequence_variant,
                                submission_policy.feedback_lane,
                            );
                        }
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                        );
                    }
                    if let Err(error) = slice.cursor.complete_pending_distributed_dispatch(
                        slice.execution_plan,
                        distributed.dispatch_index,
                    ) {
                        if can_wait_submitted {
                            let _ = distributed_runners.wait_dispatch(
                                slice.device_id(),
                                distributed.dispatch_index,
                                VulkanDistributedDispatchSequenceKind::for_feedback_lane(
                                    submission_policy.feedback_lane,
                                ),
                                |device_id| {
                                    device_by_id.get(device_id).copied().ok_or_else(|| {
                                        VulkanError(format!(
                                            "distributed shard device {device_id:?} is not mounted"
                                        ))
                                    })
                                },
                            );
                        }
                        return Err(error.into());
                    }
                    ready_dependency = None;
                    if requires_host_checkpoint {
                        distributed_runners
                            .wait_dispatch(
                                slice.device_id(),
                                distributed.dispatch_index,
                                VulkanDistributedDispatchSequenceKind::for_feedback_lane(
                                    submission_policy.feedback_lane,
                                ),
                                |device_id| {
                                    device_by_id.get(device_id).copied().ok_or_else(|| {
                                        VulkanError(format!(
                                            "distributed shard device {device_id:?} is not mounted"
                                        ))
                                    })
                                },
                            )
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Distributed,
                            )?;
                    } else if has_owner_continuation || completion_staging {
                        completion_dependency =
                            Some((distributed.dispatch_index, dependency_value));
                    } else if let Some(signal_point) = completion_bridge {
                        let wait_points = distributed_runners
                            .owner_completion_wait_points(
                                slice.device_id(),
                                distributed.dispatch_index,
                                dependency_value,
                            )
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Distributed,
                            )?;
                        let signal_points = [signal_point];
                        if let Some(submission_batch) = submission_batch {
                            submission_batch
                                .enqueue_timeline_semaphore_bridge(
                                    slice.device,
                                    &wait_points,
                                    &signal_points,
                                )
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?;
                        } else {
                            slice
                                .device
                                .submit_timeline_semaphore_bridge(&wait_points, &signal_points)
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?;
                        }
                    } else {
                        distributed_runners
                            .wait_dispatch(
                                slice.device_id(),
                                distributed.dispatch_index,
                                VulkanDistributedDispatchSequenceKind::for_feedback_lane(
                                    submission_policy.feedback_lane,
                                ),
                                |device_id| {
                                    device_by_id.get(device_id).copied().ok_or_else(|| {
                                        VulkanError(format!(
                                            "distributed shard device {device_id:?} is not mounted"
                                        ))
                                    })
                                },
                            )
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Distributed,
                            )?;
                    }
                    continue;
                }

                let segment = slice
                    .execution_plan
                    .segment_starting_at(slice.cursor.next_stage_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                            VulkanMountedPlacedResidentStreamTickError::Dispatch(
                                VulkanMountedPlacedResidentKernelDispatchError::MissingDispatchSegment {
                                    device_id: slice.device_id().to_string(),
                                    stage_index: slice.cursor.next_stage_index,
                                },
                            ),
                        )
                    })?;
                let next_distributed = slice
                    .execution_plan
                    .distributed_dispatch_at_stage(segment.end_stage_index)
                    .map(|dispatch| dispatch.dispatch_index);
                let mut wait_points = match completion_dependency
                    .map(|(dispatch_index, dependency_value)| {
                        distributed_runners.owner_completion_wait_points(
                            slice.device_id(),
                            dispatch_index,
                            dependency_value,
                        )
                    })
                    .transpose()
                {
                    Ok(semaphores) => semaphores.unwrap_or_default(),
                    Err(error) => {
                        let _ = wait_for_compact_slice_submitted_work(
                            slice,
                            device_by_id,
                            distributed_runners,
                            last_submitted_segment,
                            completion_dependency,
                            submission_policy.feedback_lane,
                            can_wait_submitted,
                        );
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                        );
                    }
                };
                wait_points.append(&mut pending_edge_wait_points);
                if slice.execution_plan.first_dispatch_segment_stage_index()
                    == Some(segment.start_stage_index)
                    && feedback_turn
                        .and_then(|turn| turn.destination_wait(slice.device_id()))
                        .is_some()
                {
                    let turn = feedback_turn.expect("resident feedback turn was present");
                    let wait = turn
                        .destination_wait(slice.device_id())
                        .expect("resident feedback input wait was present");
                    if let Some(copy) = turn.destination_copy(slice.device_id()) {
                        if let Some(batch) = submission_batch {
                            batch
                                .enqueue_resident_buffer_copy(slice.device, copy, &[wait], &[])
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?;
                        } else {
                            slice
                                .device
                                .submit_resident_buffer_copy_with_timeline_semaphores(
                                    copy,
                                    &[wait],
                                    &[],
                                )
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )?;
                        }
                    } else {
                        wait_points.push(wait);
                    }
                }
                let next_dependency = match next_distributed
                    .map(|dispatch_index| {
                        distributed_runners
                            .reserve_dependency_value(slice.device_id(), dispatch_index)
                            .map(|dependency_value| (dispatch_index, dependency_value))
                    })
                    .transpose()
                {
                    Ok(dependency) => dependency,
                    Err(error) => {
                        let _ = wait_for_compact_slice_submitted_work(
                            slice,
                            device_by_id,
                            distributed_runners,
                            last_submitted_segment,
                            completion_dependency,
                            submission_policy.feedback_lane,
                            can_wait_submitted,
                        );
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                        );
                    }
                };
                let mut signal_points = match next_dependency
                    .map(|(dispatch_index, dependency_value)| {
                        distributed_runners.owner_ready_signal_points(
                            slice.device_id(),
                            dispatch_index,
                            dependency_value,
                        )
                    })
                    .transpose()
                {
                    Ok(points) => points.unwrap_or_default(),
                    Err(error) => {
                        let _ = wait_for_compact_slice_submitted_work(
                            slice,
                            device_by_id,
                            distributed_runners,
                            last_submitted_segment,
                            completion_dependency,
                            submission_policy.feedback_lane,
                            can_wait_submitted,
                        );
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                        );
                    }
                };
                if let Some(VulkanMountedPlacedStreamTickStage::PublishEdge {
                    edge_index, ..
                }) = slice.cursor.tick_plan.stages.get(segment.end_stage_index)
                {
                    let outgoing = slice
                        .mounted
                        .edge_io
                        .outgoing_buffer(*edge_index)
                        .ok_or_else(|| {
                            VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                                VulkanMountedPlacedResidentStreamTickError::Transport(
                                    VulkanPlacedEdgeTransportError::MissingOutgoingEdge {
                                        device_id: slice.device_id().to_string(),
                                        edge_index: *edge_index,
                                    },
                                ),
                            )
                        })?;
                    let edge_key =
                        VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
                    if transport.edge_uses_shared_storage(&edge_key)
                        && let Some(signal_point) = edge_synchronizations
                            .prepare_source_signal(&outgoing.endpoint)
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                            )?
                    {
                        transport.record_queue_signal(&edge_key);
                        signal_points.push(signal_point);
                    }
                }
                let is_terminal_segment = slice.execution_plan.last_dispatch_segment_stage_index()
                    == Some(segment.start_stage_index);
                if is_terminal_segment
                    && feedback_turn.is_some_and(|turn| {
                        turn.output_device_id == slice.device_id() && turn.output_copy().is_none()
                    })
                {
                    signal_points.push(
                        feedback_turn
                            .expect("resident feedback output signal was present")
                            .output_signal(),
                    );
                }
                if is_terminal_segment
                    && output_turn.is_some_and(|turn| turn.output_device_id == slice.device_id())
                    && feedback_turn.is_none_or(|turn| turn.output_copy().is_none())
                {
                    signal_points.push(
                        output_turn
                            .expect("resident output timeline signal was present")
                            .signal,
                    );
                }
                let prefix_dispatches = if slice.execution_plan.first_dispatch_segment_stage_index()
                    == Some(segment.start_stage_index)
                {
                    slice.dispatch_extensions.prefix_dispatches.as_slice()
                } else {
                    &[]
                };
                let suffix_dispatches = if is_terminal_segment {
                    slice.dispatch_extensions.suffix_dispatches.as_slice()
                } else {
                    &[]
                };
                let demand_resume_gate = demand_resume
                    .filter(|(segment_start_stage_index, _)| {
                        *segment_start_stage_index == segment.start_stage_index
                    })
                    .map(|(_, gate_index)| gate_index);
                let terminal_step_index = if is_terminal_segment
                    && (state_transaction.is_some()
                        || !slice
                            .dispatch_extensions
                            .terminal_snapshot_copies
                            .is_empty())
                {
                    Some(
                        segment
                            .feedback_sequence_step_count(
                                prefix_dispatches.len(),
                                suffix_dispatches.len(),
                                slice.dispatch_extensions.sequence_variant,
                                submission_policy.feedback_lane,
                                demand_resume_gate,
                            )
                            .map_err(|error| {
                                VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                                    VulkanMountedPlacedResidentStreamTickError::Dispatch(error),
                                )
                            })?
                            .checked_sub(1)
                            .ok_or_else(|| {
                                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                    VulkanError(
                                        "resident feedback snapshot step index overflowed"
                                            .to_string(),
                                    ),
                                )
                            })?,
                    )
                } else {
                    None
                };
                let mut snapshot_copies = if is_terminal_segment {
                    state_transaction
                        .map(|transaction| {
                            transaction
                                .copies_for_tick(
                                    &slice.mounted.buffers,
                                    terminal_step_index.expect(
                                        "a state transaction requires one terminal step",
                                    ),
                                    submission_policy.feedback_lane.ok_or_else(|| {
                                        VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                            VulkanError(
                                                "resident feedback snapshot requires a sequence lane"
                                                    .to_string(),
                                            ),
                                        )
                                    })?,
                                )
                                .map_err(
                                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                                )
                        })
                        .transpose()?
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                if let Some(after_step_index) = terminal_step_index {
                    snapshot_copies.extend(
                        slice
                            .dispatch_extensions
                            .terminal_snapshot_copies
                            .iter()
                            .copied()
                            .map(|copy| {
                                VulkanResidentKernelSequenceSnapshotCopy::
                                    unconditional_from_range_after_conditional_step(
                                        after_step_index,
                                        copy,
                                    )
                            }),
                    );
                }
                let submission = segment.submit_with_stream_control_and_timeline_semaphores(
                    slice.device,
                    control,
                    prefix_dispatches,
                    suffix_dispatches,
                    slice.dispatch_extensions.sequence_variant,
                    submission_policy.feedback_lane,
                    demand_resume_gate,
                    &snapshot_copies,
                    &wait_points,
                    &signal_points,
                    submission_policy.signal_completion
                        && (submission_batch.is_none() || is_terminal_segment),
                    submission_batch,
                );
                if let Err(error) = submission {
                    let _ = wait_for_compact_slice_submitted_work(
                        slice,
                        device_by_id,
                        distributed_runners,
                        last_submitted_segment,
                        completion_dependency,
                        submission_policy.feedback_lane,
                        can_wait_submitted,
                    );
                    return Err(
                        VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                            VulkanMountedPlacedResidentStreamTickError::Dispatch(error),
                        ),
                    );
                }
                if is_terminal_segment
                    && let Some(turn) = feedback_turn
                    && turn.output_device_id == slice.device_id()
                    && let Some(copy) = turn.output_copy()
                {
                    let mut copy_signals = SmallVec::<[VulkanTimelineSemaphorePoint<'_>; 2]>::new();
                    copy_signals.push(turn.output_signal());
                    if let Some(output_turn) = output_turn
                        && output_turn.output_device_id == slice.device_id()
                    {
                        copy_signals.push(output_turn.signal);
                    }
                    if let Some(batch) = submission_batch {
                        batch
                            .enqueue_resident_buffer_copy(slice.device, copy, &[], &copy_signals)
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                            )?;
                    } else {
                        slice
                            .device
                            .submit_resident_buffer_copy_with_timeline_semaphores(
                                copy,
                                &[],
                                &copy_signals,
                            )
                            .map_err(
                                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule,
                            )?;
                    }
                }
                completion_dependency = None;
                ready_dependency = next_dependency;
                last_submitted_segment = Some(segment);
                while slice.cursor.next_stage_index < segment.end_stage_index {
                    slice.cursor.complete_current_stage();
                }
            }
            VulkanMountedPlacedStreamTickStage::PublishEdge { edge_index, .. } => {
                let outgoing = slice
                    .mounted
                    .edge_io
                    .outgoing_buffer(*edge_index)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                            VulkanMountedPlacedResidentStreamTickError::Transport(
                                VulkanPlacedEdgeTransportError::MissingOutgoingEdge {
                                    device_id: slice.device_id().to_string(),
                                    edge_index: *edge_index,
                                },
                            ),
                        )
                    })?;
                let edge_key =
                    VulkanPlacedEdgePacketKey::from_outgoing_endpoint(&outgoing.endpoint);
                if edge_synchronizations.edge_uses_device_local_staging(&edge_key) {
                    let transfer_wait_points = completion_dependency
                        .take()
                        .map(|(dispatch_index, dependency_value)| {
                            distributed_runners.owner_completion_wait_points(
                                slice.device_id(),
                                dispatch_index,
                                dependency_value,
                            )
                        })
                        .transpose()
                        .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Distributed)?
                        .unwrap_or_default();
                    if !edge_synchronizations
                        .enqueue_source_staging_transfer(
                            &outgoing.endpoint,
                            slice.device,
                            &transfer_wait_points,
                            submission_batch,
                        )
                        .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Schedule)?
                    {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                VulkanError(format!(
                                    "staged edge {edge_key:?} has no source transfer"
                                )),
                            ),
                        );
                    }
                    transport.record_queue_signal(&edge_key);
                } else if !transport.edge_uses_queue_transfer(&edge_key) {
                    transport.record_host_wait(&edge_key);
                    if submission_batch.is_some() {
                        return Err(
                            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(
                                VulkanError(format!(
                                    "deferred placed submission requires shared edge {edge_key:?}"
                                )),
                            ),
                        );
                    }
                    wait_for_compact_slice_submitted_work(
                        slice,
                        device_by_id,
                        distributed_runners,
                        last_submitted_segment.take(),
                        completion_dependency.take(),
                        submission_policy.feedback_lane,
                        can_wait_submitted,
                    )?;
                }
                transport
                    .publish_outgoing_edge(slice.mounted, *edge_index)
                    .map_err(|error| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::StreamTick(
                            VulkanMountedPlacedResidentStreamTickError::Transport(error),
                        )
                    })?;
                slice.cursor.complete_current_stage();
            }
        }
    }
    if !pending_edge_wait_points.is_empty() {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(format!(
                "device {:?} completed without consuming {} edge timeline dependencies",
                slice.device_id(),
                pending_edge_wait_points.len()
            ))),
        );
    }
    Ok(slice.cursor.completed_stage_count - completed_before)
}
