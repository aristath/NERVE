pub fn run_mounted_placed_resident_stream_tick_slices_in_process(
    slices: &mut [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
    transport: &mut VulkanInProcessPlacedEdgeTransport,
) -> Result<
    VulkanMountedPlacedResidentInProcessStreamTickRun,
    VulkanMountedPlacedResidentInProcessStreamTickError,
> {
    let tick_plans = slices
        .iter()
        .map(|slice| slice.execution_plan.tick_plan.as_ref())
        .collect::<Vec<_>>();
    let schedule = VulkanMountedPlacedResidentInProcessSchedule::from_tick_plans(&tick_plans)
        .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Schedule)?;
    run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule(
        slices, transport, &schedule,
    )
}

fn run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule(
    slices: &mut [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
    transport: &mut VulkanInProcessPlacedEdgeTransport,
    schedule: &VulkanMountedPlacedResidentInProcessSchedule,
) -> Result<
    VulkanMountedPlacedResidentInProcessStreamTickRun,
    VulkanMountedPlacedResidentInProcessStreamTickError,
> {
    run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
        slices,
        transport,
        schedule,
        None,
        None,
        VulkanPlacedSubmissionContext::SYNCHRONOUS,
    )
}

fn run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed<
    'a,
    'batch,
>(
    slices: &mut [VulkanMountedPlacedResidentInProcessStreamTickSlice<'a>],
    transport: &mut VulkanInProcessPlacedEdgeTransport,
    schedule: &VulkanMountedPlacedResidentInProcessSchedule,
    distributed_runners: Option<&VulkanDistributedDispatchRunners>,
    edge_synchronizations: Option<&'a VulkanPlacedEdgeTimelineSynchronizations>,
    submission: VulkanPlacedSubmissionContext<'a, 'batch>,
) -> Result<
    VulkanMountedPlacedResidentInProcessStreamTickRun,
    VulkanMountedPlacedResidentInProcessStreamTickError,
> {
    let VulkanPlacedSubmissionContext {
        policy: submission_policy,
        participant_devices,
        state_transactions,
        feedback_turn,
        output_turn,
        demand_resume,
        submission_batch,
    } = submission;
    schedule
        .validate_slices(slices)
        .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Schedule)?;
    if let Some(state_transactions) = state_transactions
        && state_transactions.len() != slices.len()
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(format!(
                "placed feedback state transaction count {} does not match slice count {}",
                state_transactions.len(),
                slices.len()
            ))),
        );
    }
    if (state_transactions.is_some() || feedback_turn.is_some())
        && submission_policy.feedback_lane.is_none()
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "placed feedback resources require a dedicated sequence lane".to_string(),
            )),
        );
    }
    if let Some(resume) = demand_resume {
        if submission_policy.feedback_lane != Some(resume.feedback_lane) {
            return Err(
                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                    "demand feedback resume lane does not match its submission lane".to_string(),
                )),
            );
        }
        if resume.next_stage_indices.len() != slices.len()
            || resume.schedule_start_turn_index >= schedule.turns.len()
            || resume.target_slice_index >= slices.len()
        {
            return Err(
                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                    "demand feedback resume frontier does not match the placed activation schedule"
                        .to_string(),
                )),
            );
        }
        for (slice, next_stage_index) in slices.iter_mut().zip(&resume.next_stage_indices) {
            if *next_stage_index > slice.cursor.tick_plan.stages.len() {
                return Err(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                        format!(
                            "demand feedback resume stage {next_stage_index} exceeds device {:?} stage count {}",
                            slice.device_id(),
                            slice.cursor.tick_plan.stages.len()
                        ),
                    )),
                );
            }
            slice.cursor.next_stage_index = *next_stage_index;
            slice.cursor.completed_stage_count = *next_stage_index;
        }
        if slices[resume.target_slice_index].cursor.next_stage_index
            != resume.target_segment_start_stage_index
        {
            return Err(
                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                    "demand feedback resume target is not the target device causal frontier"
                        .to_string(),
                )),
            );
        }
    }
    if submission_batch.is_some() && submission_policy.wait_for_completion {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "deferred placed submissions cannot be waited before their queue batch is submitted"
                    .to_string(),
            )),
        );
    }
    if let Some(turn) = feedback_turn
        && (!slices
                .iter()
                .any(|slice| slice.device_id() == turn.output_device_id)
            || turn
                .synchronization
                .destination_waits
                .keys()
                .any(|device_id| !slices.iter().any(|slice| slice.device_id() == device_id)))
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "placed feedback timeline endpoints are absent from the scheduled slices"
                    .to_string(),
            )),
        );
    }
    if let Some(turn) = output_turn
        && !slices
            .iter()
            .any(|slice| slice.device_id() == turn.output_device_id)
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "placed output timeline endpoint is absent from the scheduled slices".to_string(),
            )),
        );
    }
    transport.reset_tick_state();
    register_in_process_direct_edge_copies(slices, transport, edge_synchronizations)?;
    if edge_synchronizations
        .is_some_and(VulkanPlacedEdgeTimelineSynchronizations::has_pending_dependencies)
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "placed edge timeline state leaked across stream ticks".to_string(),
            )),
        );
    }
    if distributed_runners.is_some() && submission_policy.write_stream_control {
        prepare_compact_shared_stream_control(slices)?;
    }

    let mut completed_stage_delta = 0usize;
    let mut device_by_id = participant_devices
        .into_iter()
        .flat_map(|devices| {
            devices
                .iter()
                .map(|(device_id, device)| (device_id.clone(), device.as_ref()))
        })
        .collect::<BTreeMap<_, _>>();
    for slice in slices.iter() {
        if let Some(bound) = device_by_id.insert(slice.device_id().to_string(), slice.device)
            && !bound.shares_logical_device_with(slice.device)
        {
            return Err(
                VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                    format!(
                        "placed slice device {:?} disagrees with its bound execution participant",
                        slice.device_id()
                    ),
                )),
            );
        }
    }
    if participant_devices.is_none()
        && slices
            .first()
            .is_some_and(|first| {
                slices
                    .iter()
                    .all(|slice| slice.device.shares_logical_device_with(first.device))
            })
        && let (Some(first), Some(runners)) = (slices.first(), distributed_runners)
    {
        for device_id in runners
            .dispatches
            .iter()
            .flat_map(|dispatch| dispatch.shards.iter().map(|shard| shard.device_id.as_str()))
        {
            device_by_id
                .entry(device_id.to_string())
                .or_insert(first.device);
        }
    }

    let schedule_start_turn_index = demand_resume
        .map(|resume| resume.schedule_start_turn_index)
        .unwrap_or(0);
    for (turn_index, device_indices) in schedule
        .turns
        .iter()
        .enumerate()
        .skip(schedule_start_turn_index)
    {
        for device_index in device_indices {
            if slices[*device_index].cursor.is_completed() {
                continue;
            }
            if let Some(runners) = distributed_runners
                && !slices[*device_index].cursor.capture_execution_trace
            {
                let edge_synchronizations = edge_synchronizations.ok_or_else(|| {
                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                        "compact placed execution requires mounted edge timeline synchronization"
                            .to_string(),
                        ))
                    })?;
                let device_completed_stage_delta =
                    advance_compact_slice_with_distributed_dependencies(
                        &mut slices[*device_index],
                        &device_by_id,
                        transport,
                        runners,
                        edge_synchronizations,
                        VulkanPlacedSliceSubmissionContext {
                            policy: submission_policy,
                            state_transaction: state_transactions
                                .map(|transactions| &transactions[*device_index]),
                            feedback_turn,
                            output_turn,
                            demand_resume: demand_resume
                                .filter(|resume| resume.target_slice_index == *device_index)
                                .map(|resume| {
                                    (
                                        resume.target_segment_start_stage_index,
                                        resume.gate_index,
                                    )
                                }),
                            submission_batch,
                        },
                    )?;
                if device_completed_stage_delta == 0 {
                    return Err(
                        VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                            format!(
                                "placed activation schedule turn {turn_index} made no progress on device {:?}",
                                slices[*device_index].device_id()
                            ),
                        )),
                    );
                }
                completed_stage_delta = completed_stage_delta
                    .checked_add(device_completed_stage_delta)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                            "placed activation schedule progress overflowed".to_string(),
                        ))
                    })?;
                continue;
            }
            let mut device_completed_stage_delta = 0usize;
            loop {
                let advance = {
                    let slice = &mut slices[*device_index];
                    slice
                        .cursor
                        .advance_with_resident_execution_plan_and_in_process_transport(
                            slice.device,
                            slice.mounted,
                            slice.execution_plan,
                            &slice.dispatch_extensions,
                            transport,
                        )?
                };
                device_completed_stage_delta = device_completed_stage_delta
                    .checked_add(advance.completed_stage_delta)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                            "placed distributed stage progress overflowed".to_string(),
                        ))
                    })?;
                let pending_distributed_dispatch = {
                    let slice = &slices[*device_index];
                    slice
                        .cursor
                        .pending_distributed_dispatch(slice.execution_plan)
                        .map(|dispatch| dispatch.dispatch_index)
                };
                let Some(dispatch_index) = pending_distributed_dispatch else {
                    break;
                };
                let runners = distributed_runners.ok_or_else(|| {
                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                        format!(
                            "placed device {:?} reached distributed dispatch {dispatch_index} without mounted distributed runners",
                            slices[*device_index].device_id()
                        ),
                    ))
                })?;
                let owner_device_id = slices[*device_index].device_id().to_string();
                runners
                    .run_dispatch(&owner_device_id, dispatch_index, |shard_device_id| {
                        slices
                            .iter()
                            .find(|slice| slice.device_id() == shard_device_id)
                            .map(|slice| slice.device)
                            .ok_or_else(|| {
                                VulkanError(format!(
                                    "distributed shard device {shard_device_id:?} is not mounted"
                                ))
                            })
                    })
                    .map_err(VulkanMountedPlacedResidentInProcessStreamTickError::Distributed)?;
                let completed_dispatch_stages = {
                    let slice = &mut slices[*device_index];
                    slice.cursor.complete_pending_distributed_dispatch(
                        slice.execution_plan,
                        dispatch_index,
                    )?
                };
                device_completed_stage_delta = device_completed_stage_delta
                    .checked_add(completed_dispatch_stages)
                    .ok_or_else(|| {
                        VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                            "placed distributed stage progress overflowed".to_string(),
                        ))
                    })?;
            }
            if device_completed_stage_delta == 0 {
                return Err(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                        format!(
                            "placed activation schedule turn {turn_index} made no progress on device {:?}",
                            slices[*device_index].device_id()
                        ),
                    )),
                );
            }
            completed_stage_delta = completed_stage_delta
                .checked_add(device_completed_stage_delta)
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                        "placed activation schedule progress overflowed".to_string(),
                    ))
                })?;
        }
    }

    if !slices.iter().all(|slice| slice.cursor.is_completed()) {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(format!(
                "placed activation schedule ended with pending devices {:?}",
                pending_in_process_stream_tick_device_ids(slices)
            ))),
        );
    }
    if edge_synchronizations
        .is_some_and(VulkanPlacedEdgeTimelineSynchronizations::has_pending_dependencies)
    {
        return Err(
            VulkanMountedPlacedResidentInProcessStreamTickError::Schedule(VulkanError(
                "placed edge timeline dependencies were not consumed".to_string(),
            )),
        );
    }
    if submission_policy.wait_for_completion
        && let Some(distributed_runners) = distributed_runners
    {
        for slice in slices
            .iter()
            .filter(|slice| !slice.cursor.capture_execution_trace)
        {
            wait_for_compact_slice_terminal_work(
                slice,
                &device_by_id,
                distributed_runners,
                submission_policy.feedback_lane,
            )?;
        }
    }

    Ok(in_process_stream_tick_run_snapshot(
        slices,
        transport,
        VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed,
        schedule.turns.len() - schedule_start_turn_index,
        completed_stage_delta,
    ))
}

fn pending_in_process_stream_tick_device_ids(
    slices: &[VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
) -> Vec<String> {
    slices
        .iter()
        .filter(|slice| !slice.cursor.is_completed())
        .map(|slice| slice.device_id().to_string())
        .collect()
}

fn in_process_stream_tick_run_snapshot(
    slices: &[VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>],
    transport: &VulkanInProcessPlacedEdgeTransport,
    status: VulkanMountedPlacedResidentInProcessStreamTickRunStatus,
    scheduler_turn_count: usize,
    completed_stage_delta: usize,
) -> VulkanMountedPlacedResidentInProcessStreamTickRun {
    let completed_slice_count = slices
        .iter()
        .filter(|slice| slice.cursor.is_completed())
        .count();
    VulkanMountedPlacedResidentInProcessStreamTickRun {
        status,
        scheduler_turn_count,
        completed_stage_delta,
        completed_slice_count,
        pending_slice_count: slices.len() - completed_slice_count,
        transport_stats: transport.stats(),
        device_runs: if slices
            .iter()
            .any(|slice| slice.cursor.capture_execution_trace)
        {
            slices.iter().map(|slice| slice.cursor.snapshot()).collect()
        } else {
            Vec::new()
        },
    }
}
