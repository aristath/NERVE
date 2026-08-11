pub struct VulkanDistributedDispatchRunners {
    pub dispatches: Vec<VulkanDistributedDispatchRunner>,
    pub dispatch_count: usize,
    pub shard_count: usize,
}

fn distributed_shard_push_constants(
    planned_dispatch: &VulkanDistributedDispatchPlan,
    planned_shard: &VulkanDistributedDispatchShard,
) -> Result<Vec<u8>, VulkanDistributedDispatchRunnerError> {
    let mut bytes = planned_shard.base_workgroup_z.to_le_bytes().to_vec();
    if planned_dispatch.distribution != VulkanDistributedDispatchDistribution::OutputRows {
        let partition_count = u32::try_from(planned_shard.row_count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed repeated partition count exceeds u32".to_string(),
            )
        })?;
        bytes.extend_from_slice(&partition_count.to_le_bytes());
    }
    Ok(bytes)
}

fn create_distributed_resident_dispatch(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    planned_shard: &VulkanDistributedDispatchShard,
    shard_index: usize,
    parameter_buffers: &VulkanDistributedParameterBuffers,
    activation_buffers: &VulkanDistributedActivationBuffers,
    artifact: &VulkanLoadedPhysicalKernelArtifact,
) -> Result<VulkanResidentKernelDispatch, VulkanDistributedDispatchRunnerError> {
    let input = activation_buffers
        .activation_buffer(
            &planned_dispatch.owner_device_id,
            &planned_dispatch.input_activation,
            &planned_shard.device_id,
        )
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {}.{} has no input activation on {:?}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })?;
    let (output, output_byte_offset, output_byte_count) =
        if let Some(reduction) = &planned_dispatch.reduction {
            let output = activation_buffers
                .reduction_partial_buffer(
                    &planned_dispatch.owner_device_id,
                    planned_dispatch.dispatch_index,
                    &planned_shard.device_id,
                )
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} has no partial plane on {:?}",
                        planned_dispatch.component_id,
                        planned_dispatch.node_id,
                        planned_shard.device_id
                    ))
                })?;
            if planned_shard.output_byte_offset != 0
                || planned_shard.output_byte_count != reduction.partial_byte_capacity
            {
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} shard {} has an invalid partial-output range",
                    planned_dispatch.component_id, planned_dispatch.node_id, shard_index
                )));
            }
            let output_byte_offset = shard_index
                .checked_mul(reduction.partial_byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed partial plane offset overflowed".to_string(),
                    )
                })?;
            (output, output_byte_offset, reduction.partial_byte_capacity)
        } else {
            let output = activation_buffers
                .activation_buffer(
                    &planned_dispatch.owner_device_id,
                    &planned_dispatch.output_activation,
                    &planned_shard.device_id,
                )
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} has no output activation on {:?}",
                        planned_dispatch.component_id,
                        planned_dispatch.node_id,
                        planned_shard.device_id
                    ))
                })?;
            (
                output,
                planned_shard.output_byte_offset,
                planned_shard.output_byte_count,
            )
        };
    let mut bindings = Vec::with_capacity(
        2 + planned_dispatch.auxiliary_input_activations.len() + planned_shard.parameters.len(),
    );
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.input_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed primary input binding exceeds u32".to_string(),
                )
            })?,
            input,
            planned_shard.input_range.byte_count,
        )
        .with_byte_offset(planned_shard.input_range.byte_offset)
        .with_access(VulkanResidentKernelBufferAccess::Read),
    );
    if planned_shard.auxiliary_input_ranges.len()
        != planned_dispatch.auxiliary_input_activations.len()
    {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed dispatch {}.{} has {} auxiliary ranges for {} auxiliary inputs on {:?}",
            planned_dispatch.component_id,
            planned_dispatch.node_id,
            planned_shard.auxiliary_input_ranges.len(),
            planned_dispatch.auxiliary_input_activations.len(),
            planned_shard.device_id,
        )));
    }
    for (auxiliary, range) in planned_dispatch
        .auxiliary_input_activations
        .iter()
        .zip(&planned_shard.auxiliary_input_ranges)
    {
        let buffer = activation_buffers
            .activation_buffer(
                &planned_dispatch.owner_device_id,
                auxiliary,
                &planned_shard.device_id,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no auxiliary input {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    auxiliary.signal_id,
                    planned_shard.device_id
                ))
            })?;
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(auxiliary.binding).map_err(|_| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed auxiliary input binding exceeds u32".to_string(),
                    )
                })?,
                buffer,
                range.byte_count,
            )
            .with_byte_offset(range.byte_offset)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.output_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed output binding exceeds u32".to_string(),
                )
            })?,
            output,
            output_byte_count,
        )
        .with_byte_offset(output_byte_offset)
        .with_access(VulkanResidentKernelBufferAccess::Write),
    );
    for fragment in &planned_shard.parameters {
        let allocation = parameter_buffers
            .parameter_buffer(
                &planned_shard.device_id,
                &fragment.tensor,
                fragment.byte_offset,
                fragment.byte_count,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no tensor {:?} range at byte {} with length {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    fragment.tensor,
                    fragment.byte_offset,
                    fragment.byte_count,
                    planned_shard.device_id
                ))
            })?;
        let binding = u32::try_from(fragment.binding).map_err(|_| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed descriptor binding {} exceeds u32",
                fragment.binding
            ))
        })?;
        bindings.push(
            allocation
                .kernel_binding(binding)
                .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    let push_constant_bytes = distributed_shard_push_constants(planned_dispatch, planned_shard)?;
    device
        .create_resident_kernel_dispatch_2d_with_base_z(
            &artifact.words,
            &bindings,
            planned_shard.workgroup_count_x,
            1,
            0,
            artifact.artifact.local_size_x,
            u32::try_from(push_constant_bytes.len()).expect("push constant size is at most 8"),
            Some(format!(
                "component={} node={} distributed=device:{} rows={}..{} base_z={} distribution={:?}",
                planned_dispatch.component_id,
                planned_dispatch.node_id,
                planned_shard.device_id,
                planned_shard.row_start,
                planned_shard.row_start + planned_shard.row_count,
                planned_shard.base_workgroup_z,
                planned_dispatch.distribution,
            )),
        )
        .map_err(|error| {
            VulkanDistributedDispatchRunnerError(format!(
                "failed to create distributed dispatch {}.{} shard on {:?}: {error}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })
}

impl VulkanDistributedDispatchRunners {
    pub fn create<'a, F, E>(
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        activation_buffers: &VulkanDistributedActivationBuffers,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut dispatches = Vec::with_capacity(execution_plan.execution_islands.len());
        let mut shard_count = 0usize;
        for planned_island in &execution_plan.execution_islands {
            let leader = planned_island.leader();
            let tail = planned_island.tail();
            let owner_device = device_for(&planned_island.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed owner device {:?}: {error}",
                    planned_island.owner_device_id
                ))
            })?;
            let mut shards = Vec::with_capacity(leader.shards.len());
            for shard_index in 0..leader.shards.len() {
                let leader_shard = &leader.shards[shard_index];
                let device = device_for(&leader_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed shard device {:?}: {error}",
                        leader_shard.device_id
                    ))
                })?;
                let mut resident_dispatches = Vec::with_capacity(planned_island.dispatches.len());
                let mut planned_shards = Vec::with_capacity(planned_island.dispatches.len());
                for planned_dispatch in &planned_island.dispatches {
                    let planned_shard =
                        planned_dispatch.shards.get(shard_index).ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "physical execution island {}..{} has no shard {shard_index} for {}.{}",
                                leader.dispatch_index,
                                tail.dispatch_index,
                                planned_dispatch.component_id,
                                planned_dispatch.node_id
                            ))
                        })?;
                    if planned_shard.device_id != leader_shard.device_id {
                        return Err(VulkanDistributedDispatchRunnerError(format!(
                            "physical execution island {}..{} changes shard {shard_index} device from {:?} to {:?}",
                            leader.dispatch_index,
                            tail.dispatch_index,
                            leader_shard.device_id,
                            planned_shard.device_id
                        )));
                    }
                    let artifact = loaded_manifest
                        .physical_artifact(&planned_dispatch.physical_artifact_id)
                        .ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "distributed dispatch {}.{} is missing physical artifact {:?}",
                                planned_dispatch.component_id,
                                planned_dispatch.node_id,
                                planned_dispatch.physical_artifact_id
                            ))
                        })?;
                    resident_dispatches.push(create_distributed_resident_dispatch(
                        device,
                        planned_dispatch,
                        planned_shard,
                        shard_index,
                        parameter_buffers,
                        activation_buffers,
                        artifact,
                    )?);
                    planned_shards.push(planned_shard.clone());
                }
                let sequence = device.create_resident_kernel_sequence().map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to create distributed sequence {}..{} shard on {:?}: {error}",
                        leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                    ))
                })?;
                let push_constants = planned_shards
                    .iter()
                    .zip(&planned_island.dispatches)
                    .map(|(shard, dispatch)| distributed_shard_push_constants(dispatch, shard))
                    .collect::<Result<Vec<_>, _>>()?;
                let steps = resident_dispatches
                    .iter()
                    .zip(&push_constants)
                    .map(|(dispatch, push_constants)| {
                        VulkanResidentKernelSequenceStep::new(dispatch, push_constants)
                    })
                    .collect::<Vec<_>>();
                device
                    .record_resident_kernel_sequence(&sequence, &steps)
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to record distributed sequence {}..{} shard on {:?}: {error}",
                            leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                        ))
                    })?;
                shards.push(VulkanDistributedDispatchShardRunner {
                    device_id: leader_shard.device_id.clone(),
                    planned: planned_shards,
                    resident_dispatches,
                    sequence,
                    feedback_sequence: None,
                });
                shard_count = shard_count
                    .checked_add(planned_island.dispatches.len())
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(
                            "distributed dispatch shard count overflowed".to_string(),
                        )
                    })?;
            }
            let mut helper_synchronization = Vec::with_capacity(
                leader
                    .shards
                    .iter()
                    .filter(|shard| shard.device_id != planned_island.owner_device_id)
                    .count(),
            );
            for planned_shard in &leader.shards {
                if planned_shard.device_id == planned_island.owner_device_id {
                    continue;
                }
                let helper_device = device_for(&planned_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed helper device {:?}: {error}",
                        planned_shard.device_id
                    ))
                })?;
                helper_synchronization.push(
                    VulkanDistributedQueueSynchronization::new(
                        owner_device,
                        helper_device,
                        &planned_island.owner_device_id,
                        &planned_shard.device_id,
                        &format!(
                            "distributed dispatch {}.{}",
                            leader.component_id, leader.node_id
                        ),
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?,
                );
            }
            let reduced_dispatches = planned_island
                .dispatches
                .iter()
                .filter(|dispatch| dispatch.reduction.is_some())
                .collect::<Vec<_>>();
            let reduction = match reduced_dispatches.as_slice() {
                [] => None,
                [planned_dispatch]
                    if planned_island.dispatches.len() == 1
                        && planned_dispatch.dispatch_index == tail.dispatch_index =>
                {
                    Some(create_distributed_reduction_runner(
                        owner_device,
                        planned_dispatch,
                        activation_buffers,
                    )?)
                }
                _ => {
                    return Err(VulkanDistributedDispatchRunnerError(format!(
                        "physical execution island {}..{} has {} reductions across {} dispatches",
                        leader.dispatch_index,
                        tail.dispatch_index,
                        reduced_dispatches.len(),
                        planned_island.dispatches.len()
                    )));
                }
            };
            dispatches.push(VulkanDistributedDispatchRunner {
                planned: planned_island.clone(),
                shards,
                helper_synchronization,
                reduction,
                dependency_clock: VulkanDistributedDependencyClock::new(),
            });
        }

        Ok(Self {
            dispatch_count: execution_plan.dispatches.len(),
            dispatches,
            shard_count,
        })
    }

    pub fn dispatch(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedDispatchRunner> {
        self.dispatches.iter().find(|dispatch| {
            dispatch.planned.owner_device_id == owner_device_id
                && dispatch.planned.leader().dispatch_index == dispatch_index
        })
    }

    pub fn execution_island(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanPhysicalExecutionIslandPlan> {
        self.dispatches
            .iter()
            .find(|runner| {
                runner.planned.owner_device_id == owner_device_id
                    && runner.planned.contains_dispatch(dispatch_index)
            })
            .map(|runner| &runner.planned)
    }

    pub fn leader_dispatch_index(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<usize> {
        self.execution_island(owner_device_id, dispatch_index)
            .map(|group| group.leader().dispatch_index)
    }

    pub fn reserve_dependency_value(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<u64, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        dispatch
            .dependency_clock
            .reserve(owner_device_id, dispatch_index)
    }

    pub fn advance_replayed_dependency_values(
        &self,
        count: usize,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        let count = u64::try_from(count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed replay dependency count exceeds u64".to_string(),
            )
        })?;
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.validate_advance(
                count,
                &dispatch.planned.owner_device_id,
                dispatch.planned.leader().dispatch_index,
            )?;
        }
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.advance(count);
        }
        Ok(())
    }

    pub fn capture_replay_timeline_state(
        &self,
        state: &mut VulkanTimelineSemaphoreReplayState,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        for dispatch in &self.dispatches {
            let next_value = dispatch.dependency_clock.next_value.get();
            for synchronization in &dispatch.helper_synchronization {
                state
                    .capture(&synchronization.ready_source, next_value)
                    .and_then(|_| state.capture(&synchronization.ready_wait, next_value))
                    .and_then(|_| state.capture(&synchronization.done_source, next_value))
                    .and_then(|_| state.capture(&synchronization.done_wait, next_value))
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_feedback_indirect_dispatches<'a, F, E>(
        &mut self,
        control: &mut VulkanResidentFeedbackControlPlane,
        mut device_for: F,
    ) -> Result<(), VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        for dispatch in &mut self.dispatches {
            for shard in &mut dispatch.shards {
                let device = device_for(&shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve feedback shard device {:?}: {error}",
                        shard.device_id
                    ))
                })?;
                let indirect = control
                    .register_sequence(&shard.device_id, &shard.resident_dispatches)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let push_constants = shard
                    .planned
                    .iter()
                    .zip(&dispatch.planned.dispatches)
                    .map(|(planned, dispatch)| {
                        distributed_shard_push_constants(dispatch, planned)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let steps = shard
                    .resident_dispatches
                    .iter()
                    .zip(&push_constants)
                    .zip(&indirect.byte_offsets)
                    .map(|((resident_dispatch, push_constants), byte_offset)| {
                        VulkanResidentKernelSequenceStep::new_indirect(
                            resident_dispatch,
                            push_constants,
                            &indirect.buffer,
                            *byte_offset,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                device
                    .record_resident_kernel_sequence(&sequence, &steps)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                shard.feedback_sequence = Some(sequence);
            }
        }
        Ok(())
    }

    pub fn owner_ready_signal_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| sync.owner_ready(dependency_value))
            .collect())
    }

    pub fn owner_completion_wait_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| sync.owner_done(dependency_value))
            .collect())
    }

    pub fn submit_dispatch_with_device_dependencies<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        submission: VulkanDistributedDispatchSubmission,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let VulkanDistributedDispatchSubmission {
            dependency_value,
            consume_owner_ready_signal,
            prepare_owner_continuation,
            signal_completion,
            use_feedback_indirect,
        } = submission;
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut submitted: Vec<(
            &VulkanComputeDevice,
            &VulkanDistributedDispatchShardRunner,
            &VulkanResidentKernelSequence,
        )> = Vec::with_capacity(dispatch.shards.len());
        for (shard, device) in resolved_shards {
            let sequence = if use_feedback_indirect {
                shard.feedback_sequence.as_ref().ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed feedback shard on {:?} has no indirect sequence",
                        shard.device_id
                    ))
                })?
            } else {
                &shard.sequence
            };
            let synchronization = dispatch
                .helper_synchronization
                .iter()
                .find(|sync| sync.device_id == shard.device_id);
            let wait_points = synchronization
                .filter(|_| consume_owner_ready_signal)
                .map(|sync| {
                    vec![sync.helper_ready(dependency_value)]
                })
                .unwrap_or_default();
            let signal_points = synchronization
                .filter(|_| prepare_owner_continuation || dispatch.reduction.is_some())
                .map(|sync| {
                    vec![sync.helper_done(dependency_value)]
                })
                .unwrap_or_default();
            let shard_signal_completion = signal_completion && dispatch.reduction.is_none();
            let submission = if let Some(submission_batch) = submission_batch {
                submission_batch.enqueue_recorded_sequence(
                    device,
                    sequence,
                    &wait_points,
                    &signal_points,
                    shard_signal_completion,
                )
            } else if shard_signal_completion {
                device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    sequence,
                    &wait_points,
                    &signal_points,
                )
            } else {
                device.submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                    sequence,
                    &wait_points,
                    &signal_points,
                )
            };
            if let Err(error) = submission {
                for (submitted_device, _, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.leader().node_id,
                    shard.device_id
                )));
            }
            submitted.push((device, shard, sequence));
        }
        if let Some(reduction) = &dispatch.reduction {
            let owner_device = device_for(&dispatch.planned.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed reduction owner {:?}: {error}",
                    dispatch.planned.owner_device_id
                ))
            })?;
            let wait_points = dispatch
                .helper_synchronization
                .iter()
                .map(|sync| sync.owner_done(dependency_value))
                .collect::<Vec<_>>();
            let reduction_submission = if let Some(submission_batch) = submission_batch {
                submission_batch.enqueue_recorded_sequence(
                    owner_device,
                    &reduction.sequence,
                    &wait_points,
                    &[],
                    signal_completion,
                )
            } else if signal_completion {
                owner_device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &reduction.sequence,
                    &wait_points,
                    &[],
                )
            } else {
                owner_device
                    .submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                        &reduction.sequence,
                        &wait_points,
                        &[],
                    )
            };
            if let Err(error) = reduction_submission {
                for (submitted_device, _, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed reduction {}.{} on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    dispatch.planned.owner_device_id
                )));
            }
        }

        Ok(VulkanDistributedDispatchRun {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            component_id: dispatch.planned.leader().component_id.clone(),
            node_id: dispatch.planned.tail().node_id.clone(),
            shard_count: dispatch.shards.len(),
        })
    }

    pub fn wait_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<(), VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut first_error = None;
        for (shard, device) in resolved_shards {
            if let Err(error) = device.wait_resident_kernel_sequence(&shard.sequence)
                && first_error.is_none()
            {
                first_error = Some(format!(
                    "failed waiting for distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    shard.device_id
                ));
            }
        }
        if let Some(reduction) = &dispatch.reduction {
            let owner_device = device_for(&dispatch.planned.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed reduction owner {:?}: {error}",
                    dispatch.planned.owner_device_id
                ))
            })?;
            if let Err(error) = owner_device.wait_resident_kernel_sequence(&reduction.sequence)
                && first_error.is_none()
            {
                first_error = Some(format!(
                    "failed waiting for distributed reduction {}.{} on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    dispatch.planned.owner_device_id
                ));
            }
        }
        if let Some(error) = first_error {
            return Err(VulkanDistributedDispatchRunnerError(error));
        }
        Ok(())
    }

    pub fn run_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dependency_value = self.reserve_dependency_value(owner_device_id, dispatch_index)?;
        let run = self.submit_dispatch_with_device_dependencies(
            owner_device_id,
            dispatch_index,
            VulkanDistributedDispatchSubmission {
                dependency_value,
                consume_owner_ready_signal: false,
                prepare_owner_continuation: true,
                signal_completion: true,
                use_feedback_indirect: false,
            },
            None,
            |device_id| device_for(device_id),
        )?;
        self.wait_dispatch(owner_device_id, dispatch_index, |device_id| {
            device_for(device_id)
        })?;
        Ok(run)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRun {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub shard_count: usize,
}

pub struct VulkanDistributedDispatchRunner {
    pub planned: VulkanPhysicalExecutionIslandPlan,
    pub shards: Vec<VulkanDistributedDispatchShardRunner>,
    helper_synchronization: Vec<VulkanDistributedQueueSynchronization>,
    reduction: Option<VulkanDistributedReductionRunner>,
    dependency_clock: VulkanDistributedDependencyClock,
}

pub struct VulkanDistributedDispatchShardRunner {
    pub device_id: String,
    pub planned: Vec<VulkanDistributedDispatchShard>,
    pub resident_dispatches: Vec<VulkanResidentKernelDispatch>,
    pub sequence: VulkanResidentKernelSequence,
    feedback_sequence: Option<VulkanResidentKernelSequence>,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRunnerError(pub String);

impl Display for VulkanDistributedDispatchRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedDispatchRunnerError {}

impl From<VulkanError> for VulkanDistributedDispatchRunnerError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}
