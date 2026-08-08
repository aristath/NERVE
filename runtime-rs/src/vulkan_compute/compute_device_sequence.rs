impl VulkanComputeDevice {
    pub fn run_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.run_resident_kernel_sequence_with_snapshot_copies(sequence, steps, &[])
    }

    pub fn run_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    pub fn run_recorded_resident_kernel_sequence_for(
        &self,
        sequence: &VulkanResidentKernelSequence,
        timeout: Duration,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence(sequence)?;
        self.wait_recorded_resident_kernel_sequence_for(sequence, timeout)
    }

    pub fn run_timestamped_recorded_resident_kernel_sequence_for(
        &self,
        sequence: &VulkanResidentKernelSequence,
        timeout: Duration,
    ) -> Result<u64, VulkanError> {
        if sequence.timestamp_query_pool.is_none() {
            return Err(VulkanError(
                "resident kernel sequence was not created with timestamp measurement".to_string(),
            ));
        }
        self.submit_recorded_resident_kernel_sequence(sequence)?;
        self.wait_recorded_resident_kernel_sequence_for(sequence, timeout)?;
        self.read_recorded_resident_kernel_sequence_duration_ns(sequence)
    }

    fn wait_recorded_resident_kernel_sequence_for(
        &self,
        sequence: &VulkanResidentKernelSequence,
        timeout: Duration,
    ) -> Result<(), VulkanError> {
        let _wait = runtime_critical_path_span(RuntimeCriticalPathPhase::HostSynchronization);
        self.require_device_healthy()?;
        let value = sequence
            .completion
            .pending("resident kernel sequence")?;
        let timeout_ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        let completed = self.wait_timeline_semaphore_value_for(
            &sequence.completion.timeline,
            value,
            timeout_ns,
        )?;
        if completed {
            sequence.completion.complete(value)?;
            RESIDENT_SEQUENCE_COMPLETION_WAITS.fetch_add(1, Ordering::Relaxed);
            self.require_device_healthy()
        } else {
            Err(VulkanError(format!(
                "resident kernel sequence exceeded bounded wait of {} ns",
                timeout_ns
            )))
        }
    }

    pub(crate) fn read_recorded_resident_kernel_sequence_duration_ns(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<u64, VulkanError> {
        let query_pool = sequence.timestamp_query_pool.ok_or_else(|| {
            VulkanError(
                "resident kernel sequence was not created with timestamp measurement".to_string(),
            )
        })?;
        let mut timestamps = [0_u64; 2];
        unsafe {
            self.device
                .get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident sequence timestamps: {error:?}"
                    ))
                })?;
        }
        let elapsed_ns = timestamps[1].wrapping_sub(timestamps[0]) as f64
            * f64::from(sequence.timestamp_period_ns);
        if !elapsed_ns.is_finite() || elapsed_ns <= 0.0 || elapsed_ns > u64::MAX as f64 {
            return Err(VulkanError(format!(
                "resident sequence produced invalid device duration {elapsed_ns}"
            )));
        }
        Ok((elapsed_ns.round() as u64).max(1))
    }

    pub(crate) fn read_recorded_resident_kernel_step_durations_ns(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<Vec<u64>, VulkanError> {
        let (query_pool, query_count) =
            sequence.profiling_timestamp_query_pool.ok_or_else(|| {
                VulkanError(
                    "resident kernel sequence was not created with step profiling".to_string(),
                )
            })?;
        let mut timestamps = vec![0_u64; query_count as usize];
        unsafe {
            self.device
                .get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident kernel profile timestamps: {error:?}"
                    ))
                })?;
        }
        timestamps
            .windows(2)
            .map(|pair| {
                let elapsed_ns = pair[1].wrapping_sub(pair[0]) as f64
                    * f64::from(sequence.timestamp_period_ns);
                if !elapsed_ns.is_finite()
                    || elapsed_ns <= 0.0
                    || elapsed_ns > u64::MAX as f64
                {
                    return Err(VulkanError(format!(
                        "resident kernel step produced invalid device duration {elapsed_ns}"
                    )));
                }
                Ok((elapsed_ns.round() as u64).max(1))
            })
            .collect()
    }

    pub(crate) fn read_recorded_resident_kernel_critical_path_region_durations_ns(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<Vec<u64>, VulkanError> {
        let (query_pool, query_count) = sequence
            .critical_path_timestamp_query_pool
            .ok_or_else(|| {
                VulkanError(
                    "resident kernel sequence was not created with critical-path region timing"
                        .to_string(),
                )
            })?;
        let mut timestamps = vec![0_u64; query_count as usize];
        unsafe {
            self.device
                .get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident sequence critical-path timestamps: {error:?}"
                    ))
                })?;
        }
        timestamps
            .windows(2)
            .map(|pair| {
                let elapsed_ns = pair[1].wrapping_sub(pair[0]) as f64
                    * f64::from(sequence.timestamp_period_ns);
                if !elapsed_ns.is_finite()
                    || elapsed_ns <= 0.0
                    || elapsed_ns > u64::MAX as f64
                {
                    return Err(VulkanError(format!(
                        "resident sequence critical-path region produced invalid device duration {elapsed_ns}"
                    )));
                }
                Ok((elapsed_ns.round() as u64).max(1))
            })
            .collect()
    }

    pub fn submit_recorded_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(sequence, &[], &[])
    }

    pub fn submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
        &self,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        let completion_value = sequence
            .completion
            .reserve("resident kernel sequence")?;
        let completion = VulkanTimelineSemaphorePoint::new(
            &sequence.completion.timeline,
            completion_value,
        );
        let submit_result = self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            wait_points,
            signal_points,
            Some(completion),
            "resident kernel sequence",
            true,
        );
        if let Err(error) = submit_result {
            sequence.completion.cancel(completion_value);
            return Err(error);
        }
        *sequence.pending_wait_points.borrow_mut() = wait_points
            .iter()
            .map(|point| (point.semaphore.semaphore, point.value))
            .collect();
        Ok(())
    }

    pub fn submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
        &self,
        sequence: &VulkanResidentKernelSequence,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        if !sequence.has_recorded_commands() {
            return Err(VulkanError(
                "resident kernel sequence has no recorded commands".to_string(),
            ));
        }
        self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            wait_points,
            signal_points,
            None,
            "resident kernel sequence",
            true,
        )
    }

    pub fn submit_timeline_semaphore_bridge(
        &self,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        let _submission = runtime_critical_path_span(RuntimeCriticalPathPhase::QueueSubmission);
        self.require_device_healthy()?;
        if wait_points.is_empty() && signal_points.is_empty() {
            return Err(VulkanError(
                "timeline semaphore bridge has no wait or signal points".to_string(),
            ));
        }
        for point in wait_points.iter().chain(signal_points) {
            self.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let wait_infos = wait_points
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        let signal_infos = signal_points
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        unsafe {
            let submit_info = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_infos)];
            self.device
                .queue_submit2(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    vulkan_error_with_device_quarantine(
                        &self.device_health,
                        error,
                        self.vulkan_operation_error(
                            "failed to submit timeline semaphore bridge",
                            error,
                        ),
                    )
                })?;
        }
        RESIDENT_SEQUENCE_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn submit_resident_kernel_sequence_and_wait(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        self.submit_resident_kernel_sequence(sequence)?;
        self.wait_resident_kernel_sequence(sequence)
    }

    fn submit_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        let completion_value = sequence
            .completion
            .reserve("resident kernel sequence")?;
        let completion = VulkanTimelineSemaphorePoint::new(
            &sequence.completion.timeline,
            completion_value,
        );
        let submit_result = self.submit_command_buffer_with_timeline_semaphores(
            sequence.command_buffer,
            &[],
            &[],
            Some(completion),
            "resident kernel sequence",
            true,
        );
        if let Err(error) = submit_result {
            sequence.completion.cancel(completion_value);
            return Err(error);
        }
        sequence.pending_wait_points.borrow_mut().clear();
        Ok(())
    }

    fn submit_command_buffer_with_timeline_semaphores(
        &self,
        command_buffer: vk::CommandBuffer,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        completion_point: Option<VulkanTimelineSemaphorePoint<'_>>,
        label: &str,
        record_sequence_submission: bool,
    ) -> Result<(), VulkanError> {
        let _submission = runtime_critical_path_span(RuntimeCriticalPathPhase::QueueSubmission);
        self.require_device_healthy()?;
        for point in wait_points
            .iter()
            .chain(signal_points)
            .chain(completion_point.iter())
        {
            self.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let wait_infos = wait_points
            .iter()
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        let signal_infos = signal_points
            .iter()
            .chain(completion_point.iter())
            .map(|point| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(point.semaphore.semaphore)
                    .value(point.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect::<Vec<_>>();
        unsafe {
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
            let submit_info = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(&command_buffers)
                .signal_semaphore_infos(&signal_infos)];
            self.device
                .queue_submit2(self.queue, &submit_info, vk::Fence::null())
                .map_err(|error| {
                    vulkan_error_with_device_quarantine(
                        &self.device_health,
                        error,
                        self.vulkan_operation_error(&format!("failed to submit {label}"), error),
                    )
                })?;
            if record_sequence_submission {
                RESIDENT_SEQUENCE_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            } else {
                RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

}

fn resident_kernel_sequence_watchdog_description(
    sequence: &VulkanResidentKernelSequence,
) -> String {
    let recorded_steps = sequence.recorded_steps.borrow();
    let Some(steps) = recorded_steps.as_ref() else {
        return "resident kernel sequence (unrecorded execution contract)".to_string();
    };
    let mut families = BTreeMap::<&str, usize>::new();
    let mut semantic_nodes = BTreeSet::<&str>::new();
    let mut estimated_work_units = 0u64;
    let mut estimated_memory_bytes = 0u64;
    for step in steps {
        *families.entry(step.execution_family.as_str()).or_default() += 1;
        if let Some(node) = step
            .semantic_label
            .as_deref()
            .and_then(|label| semantic_label_field(label, "node"))
        {
            semantic_nodes.insert(node);
        }
        estimated_work_units = estimated_work_units.saturating_add(step.estimated_work_units);
        estimated_memory_bytes = estimated_memory_bytes.saturating_add(step.estimated_memory_bytes);
    }
    let families = families
        .into_iter()
        .map(|(family, count)| format!("{family}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let nodes = semantic_nodes.into_iter().take(4).collect::<Vec<_>>().join(",");
    format!(
        "resident kernel sequence (steps={}, work_units={}, memory_bytes={}, families=[{}], nodes=[{}])",
        steps.len(), estimated_work_units, estimated_memory_bytes, families, nodes,
    )
}

fn validate_resident_sequence_critical_path_regions(
    region_indices: &[Option<u32>],
    query_count: u32,
) -> Result<(), VulkanError> {
    let expected_region_count = query_count.checked_sub(1).ok_or_else(|| {
        VulkanError(
            "resident kernel critical-path timestamp pool has no boundary query".to_string(),
        )
    })?;
    if expected_region_count == 0 {
        return Err(VulkanError(
            "resident kernel critical-path timestamp pool has no regions".to_string(),
        ));
    }
    let mut previous = None::<u32>;
    for (step_index, region_index) in region_indices.iter().copied().enumerate() {
        let region_index = region_index.ok_or_else(|| {
            VulkanError(format!(
                "resident kernel critical-path sequence step {step_index} has no timing region"
            ))
        })?;
        match previous {
            None if region_index != 0 => {
                return Err(VulkanError(format!(
                    "resident kernel critical-path timing starts at region {region_index}, expected 0"
                )));
            }
            Some(previous)
                if region_index != previous
                    && previous.checked_add(1) != Some(region_index) =>
            {
                return Err(VulkanError(format!(
                    "resident kernel critical-path timing jumps from region {previous} to {region_index} at step {step_index}"
                )));
            }
            _ => {}
        }
        previous = Some(region_index);
    }
    let actual_region_count = previous
        .and_then(|last| last.checked_add(1))
        .unwrap_or_default();
    if actual_region_count != expected_region_count {
        return Err(VulkanError(format!(
            "resident kernel critical-path timing recorded {actual_region_count} regions but allocated {expected_region_count}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[test]
fn critical_path_region_validation_rejects_incomplete_or_noncontiguous_layouts() {
    assert!(
        validate_resident_sequence_critical_path_regions(
            &[Some(0), Some(0), Some(1), Some(2), Some(2)],
            4,
        )
        .is_ok()
    );
    assert!(
        validate_resident_sequence_critical_path_regions(&[Some(0), None], 2)
            .unwrap_err()
            .0
            .contains("has no timing region")
    );
    assert!(
        validate_resident_sequence_critical_path_regions(&[Some(1)], 2)
            .unwrap_err()
            .0
            .contains("expected 0")
    );
    assert!(
        validate_resident_sequence_critical_path_regions(&[Some(0), Some(2)], 3)
            .unwrap_err()
            .0
            .contains("jumps")
    );
    assert!(
        validate_resident_sequence_critical_path_regions(&[Some(0)], 3)
            .unwrap_err()
            .0
            .contains("recorded 1 regions but allocated 2")
    );
}

impl VulkanResidentQueueSubmitter {
    fn submit_prepared_resident_queue_batch(
        &self,
        submissions: &[VulkanPreparedResidentQueueSubmission],
        timeline_value_transform: VulkanTimelineValueTransform<'_>,
        signal_batch_completion: bool,
    ) -> Result<VulkanSubmittedResidentQueueBatch, VulkanError> {
        let _submission = runtime_critical_path_span(RuntimeCriticalPathPhase::QueueSubmission);
        self.device_health.require_healthy()?;
        if submissions.is_empty() {
            return Ok(VulkanSubmittedResidentQueueBatch {
                batch_completion_value: None,
                resource_completions: Vec::new(),
            });
        }
        let mut resource_completions = Vec::new();
        let mut submission_completion_values = Vec::with_capacity(submissions.len());
        for submission in submissions {
            let Some(completion) = submission.completion.as_ref() else {
                submission_completion_values.push(None);
                continue;
            };
            if resource_completions
                .iter()
                .any(|(existing, _)| Rc::ptr_eq(existing, completion))
            {
                for (reserved, value) in resource_completions {
                    reserved.cancel(value);
                }
                return Err(VulkanError(
                    "resident queue batch signals one resource completion more than once"
                        .to_string(),
                ));
            }
            let value = match completion.reserve("resident queue resource") {
                Ok(value) => value,
                Err(error) => {
                    for (reserved, value) in resource_completions {
                        reserved.cancel(value);
                    }
                    return Err(error);
                }
            };
            resource_completions.push((Rc::clone(completion), value));
            submission_completion_values.push(Some(value));
        }
        let batch_completion_value = if signal_batch_completion {
            match self.completion.reserve("resident queue batch") {
                Ok(value) => Some(value),
                Err(error) => {
                    for (completion, value) in resource_completions {
                        completion.cancel(value);
                    }
                    return Err(error);
                }
            }
        } else {
            None
        };
        let wait_infos = submissions
            .iter()
            .map(|submission| {
                submission
                    .wait_points
                    .iter()
                    .map(|(semaphore, value)| {
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*semaphore)
                            .value(
                                timeline_value_transform
                                    .value(self.device_handle, *semaphore, *value)
                                    .expect("resident submission template offsets were validated"),
                            )
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let command_infos = submissions
            .iter()
            .map(|submission| {
                submission
                    .command_buffer
                    .into_iter()
                    .map(|command_buffer| {
                        vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let signal_infos = submissions
            .iter()
            .enumerate()
            .map(|(index, submission)| {
                let mut infos = submission
                    .signal_points
                    .iter()
                    .map(|(semaphore, value)| {
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(*semaphore)
                            .value(
                                timeline_value_transform
                                    .value(self.device_handle, *semaphore, *value)
                                    .expect("resident submission template offsets were validated"),
                            )
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    })
                    .collect::<Vec<_>>();
                if let Some(value) = submission_completion_values[index] {
                    infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(
                                submission
                                    .completion
                                    .as_ref()
                                    .expect("a completion value has a completion timeline")
                                    .semaphore(),
                            )
                            .value(value)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                    );
                }
                if index + 1 == submissions.len()
                    && let Some(value) = batch_completion_value
                {
                    infos.push(
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.completion.semaphore())
                            .value(value)
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                    );
                }
                infos
            })
            .collect::<Vec<_>>();
        let submit_infos = (0..submissions.len())
            .map(|index| {
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait_infos[index])
                    .command_buffer_infos(&command_infos[index])
                    .signal_semaphore_infos(&signal_infos[index])
            })
            .collect::<Vec<_>>();
        unsafe {
            if let Err(error) =
                self.device
                    .queue_submit2(self.queue, &submit_infos, vk::Fence::null())
            {
                for (completion, value) in &resource_completions {
                    completion.cancel(*value);
                }
                if let Some(value) = batch_completion_value {
                    self.completion.cancel(value);
                }
                return Err(vulkan_error_with_device_quarantine(
                    &self.device_health,
                    error,
                    self.vulkan_operation_error(
                        &format!(
                            "failed to submit resident queue batch containing {} commands",
                            submissions.len()
                        ),
                        error,
                    ),
                ));
            }
            RESIDENT_QUEUE_BATCH_SUBMITS.fetch_add(1, Ordering::Relaxed);
            RESIDENT_QUEUE_BATCH_COMMANDS.fetch_add(
                u64::try_from(submissions.len()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
        Ok(VulkanSubmittedResidentQueueBatch {
            batch_completion_value,
            resource_completions,
        })
    }

    fn wait_for_batch_completion(&self, value: u64) -> Result<(), VulkanError> {
        let _wait = runtime_critical_path_span(RuntimeCriticalPathPhase::HostSynchronization);
        wait_for_vulkan_timeline_points_with_progress_watchdog(
            &self.device,
            &[self.completion.semaphore()],
            &[value],
            false,
            &self.device_health,
            "resident execution quantum",
            |error| {
                self.vulkan_operation_error(
                    "failed to wait for resident execution quantum",
                    error,
                )
            },
        )?;
        self.completion.complete(value)?;
        RESIDENT_SEQUENCE_COMPLETION_WAITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl VulkanComputeDevice {
    pub fn wait_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
    ) -> Result<(), VulkanError> {
        let _wait = runtime_critical_path_span(RuntimeCriticalPathPhase::HostSynchronization);
        let operation = resident_kernel_sequence_watchdog_description(sequence);
        let wait_points = sequence.pending_wait_points.borrow().clone();
        let value = sequence
            .completion
            .pending("resident kernel sequence")?;
        let mut progress_points = Vec::with_capacity(wait_points.len() + 1);
        progress_points.push((sequence.completion.semaphore(), value));
        progress_points.extend(wait_points.iter().copied());
        wait_for_vulkan_timeline_points_with_progress_sources(
            &self.device,
            &[sequence.completion.semaphore()],
            &[value],
            false,
            &self.device_health,
            &operation,
            VulkanQueueProgressSources {
                timeline_points: &progress_points,
                timestamp_query_pool: sequence.timestamp_query_pool,
            },
            |error| {
                self.vulkan_operation_error("failed waiting for resident kernel sequence", error)
            },
        )?;
        sequence.completion.complete(value)?;
        sequence.pending_wait_points.borrow_mut().clear();
        RESIDENT_SEQUENCE_COMPLETION_WAITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn run_resident_kernel_sequence_with_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, snapshot_copies, true)
    }

    pub fn run_resident_kernel_sequence_with_input_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, input_copies, steps, &[], true)
    }

    pub fn run_resident_kernel_sequence_with_input_and_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(
            sequence,
            input_copies,
            steps,
            snapshot_copies,
            true,
        )
    }

    pub fn record_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, &[], false)
    }

    pub fn record_resident_kernel_sequence_with_snapshot_copies(
        &self,
        sequence: &VulkanResidentKernelSequence,
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
    ) -> Result<(), VulkanError> {
        self.prepare_resident_kernel_sequence(sequence, &[], steps, snapshot_copies, false)
    }

    fn prepare_resident_kernel_sequence(
        &self,
        sequence: &VulkanResidentKernelSequence,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        steps: &[VulkanResidentKernelSequenceStep<'_>],
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        execute: bool,
    ) -> Result<(), VulkanError> {
        let _preparation = runtime_critical_path_span(RuntimeCriticalPathPhase::CommandPreparation);
        if steps.is_empty() {
            return Err(VulkanError(
                "resident kernel sequence must contain at least one dispatch".to_string(),
            ));
        }
        for (step_index, step) in steps.iter().enumerate() {
            if step.dispatch.pipeline_key.push_constant_byte_count
                != step.push_constants.len() as u32
            {
                return Err(VulkanError(format!(
                    "resident kernel sequence step {step_index} expects {} push-constant bytes, got {}",
                    step.dispatch.pipeline_key.push_constant_byte_count,
                    step.push_constants.len()
                )));
            }
            if let Some(condition) = step.condition {
                if self.conditional_rendering.is_none() {
                    return Err(VulkanError(format!(
                        "resident kernel sequence step {step_index} requires unsupported VK_EXT_conditional_rendering"
                    )));
                }
                if !self.owns_resident_buffer(condition.buffer) {
                    return Err(VulkanError(format!(
                        "resident kernel sequence step {step_index} condition belongs to another logical device"
                    )));
                }
                if snapshot_copies.iter().any(|copy| {
                    copy.after_step_index == step_index
                        && !copy.allow_after_conditional_step
                }) {
                    return Err(VulkanError(format!(
                        "resident kernel sequence step {step_index} cannot combine conditional execution with an unconditional snapshot copy without explicit checkpoint-resume safety"
                    )));
                }
            }
        }
        if let Some(copy) = snapshot_copies
            .iter()
            .find(|copy| copy.after_step_index >= steps.len())
        {
            return Err(VulkanError(format!(
                "resident snapshot follows step {}, but sequence contains {} steps",
                copy.after_step_index,
                steps.len()
            )));
        }
        if let Some((_, query_count)) = sequence.critical_path_timestamp_query_pool {
            validate_resident_sequence_critical_path_regions(
                &steps
                    .iter()
                    .map(|step| step.critical_path_region_index)
                    .collect::<Vec<_>>(),
                query_count,
            )?;
        }

        unsafe {
            RESIDENT_SEQUENCE_PREPARE_CALLS.fetch_add(1, Ordering::Relaxed);
            let profiling_enabled = execute && std::env::var_os("NERVE_VK_PERF_LOGGER").is_some();
            let command_buffer_matches = !profiling_enabled
                && sequence
                    .recorded_input_copies
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == input_copies.len()
                            && recorded
                                .iter()
                                .zip(input_copies)
                                .all(|(recorded, copy)| *recorded == copy.recorded())
                    })
                && sequence
                    .recorded_steps
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == steps.len()
                            && recorded.iter().zip(steps).all(|(recorded, step)| {
                                recorded.pipeline == step.dispatch.pipeline
                                    && recorded.descriptor_set == step.dispatch.descriptor_set
                                    && recorded.workgroup_count_x == step.workgroup_count_x()
                                    && recorded.workgroup_count_y == step.workgroup_count_y()
                                    && recorded.base_workgroup_z == step.dispatch.base_workgroup_z
                                    && recorded.indirect_dispatch
                                        == step.indirect_dispatch.map(|indirect| {
                                            VulkanResidentKernelRecordedIndirectDispatch {
                                                buffer: indirect.buffer.buffer,
                                                offset: indirect.offset,
                                            }
                                        })
                                    && recorded.condition
                                        == step.condition.map(
                                            VulkanResidentKernelSequenceCondition::recorded,
                                        )
                                    && recorded.critical_path_region_index
                                        == step.critical_path_region_index
                                    && recorded.push_constants == step.push_constants
                            })
                    })
                && sequence
                    .recorded_snapshot_copies
                    .borrow()
                    .as_ref()
                    .is_some_and(|recorded| {
                        recorded.len() == snapshot_copies.len()
                            && recorded
                                .iter()
                                .zip(snapshot_copies)
                                .all(|(recorded, copy)| *recorded == copy.recorded())
                    });
            if command_buffer_matches {
                RESIDENT_SEQUENCE_REUSED_COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
            } else {
                RESIDENT_SEQUENCE_RECORDED_COMMAND_BUFFERS.fetch_add(1, Ordering::Relaxed);
            }
            let host_start = profiling_enabled.then(Instant::now);
            let query_count = u32::try_from(steps.len() + 1).map_err(|_| {
                VulkanError("resident kernel timestamp count overflowed".to_string())
            })?;
            let query_pool = if profiling_enabled {
                let query_pool_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(query_count);
                Some(
                    self.device
                        .create_query_pool(&query_pool_info, None)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to create resident kernel timestamp pool: {error:?}"
                            ))
                        })?,
                )
            } else {
                None
            };

            if !command_buffer_matches {
                self.device
                    .reset_command_buffer(
                        sequence.command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to reset resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                let command_begin = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE);
                self.device
                    .begin_command_buffer(sequence.command_buffer, &command_begin)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to begin resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;
            }

            if !command_buffer_matches && let Some(query_pool) = query_pool {
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    query_count,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
            if !command_buffer_matches
                && let Some(query_pool) = sequence.timestamp_query_pool
            {
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    2,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
            if !command_buffer_matches
                && let Some((query_pool, query_count)) =
                    sequence.profiling_timestamp_query_pool
            {
                let expected_query_count = u32::try_from(steps.len())
                    .ok()
                    .and_then(|count| count.checked_add(1))
                    .ok_or_else(|| {
                        VulkanError(
                            "resident kernel profile timestamp count overflowed".to_string(),
                        )
                    })?;
                if query_count != expected_query_count {
                    return Err(VulkanError(format!(
                        "resident kernel profile allocated {query_count} timestamps but recording requires {expected_query_count}"
                    )));
                }
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    query_count,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
            if !command_buffer_matches
                && let Some((query_pool, query_count)) =
                    sequence.critical_path_timestamp_query_pool
            {
                self.device.cmd_reset_query_pool(
                    sequence.command_buffer,
                    query_pool,
                    0,
                    query_count,
                );
                self.device.cmd_write_timestamp(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }

            if !command_buffer_matches {
                let has_conditions =
                    steps.iter().any(|step| step.condition.is_some());
                if input_copies.is_empty() {
                    // A resident sequence is an independently submitted circuit unit. Its
                    // inputs may have been produced by the host, a transfer, or an earlier
                    // compute sequence on this queue, so establish the full producer-to-
                    // consumer dependency at the sequence boundary.
                    let input_visibility_barrier = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::HOST_WRITE
                                | vk::AccessFlags::TRANSFER_WRITE
                                | vk::AccessFlags::SHADER_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::INDIRECT_COMMAND_READ
                                | if has_conditions {
                                    vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
                                } else {
                                    vk::AccessFlags::empty()
                                },
                        )];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::HOST
                            | vk::PipelineStageFlags::TRANSFER
                            | vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT
                            | if has_conditions {
                                vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT
                            } else {
                                vk::PipelineStageFlags::empty()
                            },
                        vk::DependencyFlags::empty(),
                        &input_visibility_barrier,
                        &[],
                        &[],
                    );
                } else {
                    let input_to_transfer = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::HOST_WRITE
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::TRANSFER_WRITE,
                        )
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::HOST
                            | vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &input_to_transfer,
                        &[],
                        &[],
                    );
                    for (copy_index, input_copy) in input_copies.iter().enumerate() {
                        if copy_index != 0 {
                            let transfer_order = [vk::MemoryBarrier::default()
                                .src_access_mask(
                                    vk::AccessFlags::TRANSFER_READ
                                        | vk::AccessFlags::TRANSFER_WRITE,
                                )
                                .dst_access_mask(
                                    vk::AccessFlags::TRANSFER_READ
                                        | vk::AccessFlags::TRANSFER_WRITE,
                                )];
                            self.device.cmd_pipeline_barrier(
                                sequence.command_buffer,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::DependencyFlags::empty(),
                                &transfer_order,
                                &[],
                                &[],
                            );
                        }
                        let regions = [vk::BufferCopy {
                            src_offset: input_copy.source_offset(),
                            dst_offset: input_copy.destination_offset(),
                            size: input_copy.byte_len(),
                        }];
                        self.device.cmd_copy_buffer(
                            sequence.command_buffer,
                            input_copy.source(),
                            input_copy.destination(),
                            &regions,
                        );
                    }
                    let transfer_to_compute = [vk::MemoryBarrier::default()
                        .src_access_mask(
                            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                        )
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ
                                | vk::AccessFlags::SHADER_WRITE
                                | vk::AccessFlags::INDIRECT_COMMAND_READ
                                | if has_conditions {
                                    vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
                                } else {
                                    vk::AccessFlags::empty()
                                },
                        )];
                    self.device.cmd_pipeline_barrier(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                        vk::PipelineStageFlags::COMPUTE_SHADER
                            | vk::PipelineStageFlags::DRAW_INDIRECT
                            | if has_conditions {
                                vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT
                            } else {
                                vk::PipelineStageFlags::empty()
                            },
                        vk::DependencyFlags::empty(),
                        &transfer_to_compute,
                        &[],
                        &[],
                    );
                }
            }

            let mut pending_buffer_accesses = Vec::<VulkanResidentKernelBufferAccessRecord>::new();
            if !command_buffer_matches {
                let mut active_condition =
                    None::<VulkanResidentKernelRecordedCondition>;
                for (step_index, step) in steps.iter().enumerate() {
                    let recorded_condition = step
                        .condition
                        .map(VulkanResidentKernelSequenceCondition::recorded);
                    if active_condition != recorded_condition
                        && active_condition.is_some()
                    {
                        let conditional_rendering =
                            self.conditional_rendering.as_ref().ok_or_else(|| {
                                VulkanError(
                                    "conditional compute dispatch was recorded on a device without VK_EXT_conditional_rendering"
                                        .to_string(),
                                )
                            })?;
                        (conditional_rendering
                            .fp()
                            .cmd_end_conditional_rendering_ext)(
                            sequence.command_buffer,
                        );
                        active_condition = None;
                    }
                    let mut step_buffer_accesses =
                        step.dispatch.buffer_accesses.clone();
                    if let Some(indirect) = step.indirect_dispatch {
                        merge_resident_kernel_buffer_accesses(
                            &mut step_buffer_accesses,
                            &[VulkanResidentKernelBufferAccessRecord {
                                buffer: indirect.buffer.buffer,
                                access: VulkanResidentKernelBufferAccess::Read,
                            }],
                        );
                    }
                    if let Some(condition) = step.condition {
                        merge_resident_kernel_buffer_accesses(
                            &mut step_buffer_accesses,
                            &[VulkanResidentKernelBufferAccessRecord {
                                buffer: condition.buffer.buffer,
                                access: VulkanResidentKernelBufferAccess::Read,
                            }],
                        );
                    }
                    let dependencies = take_resident_kernel_buffer_dependencies(
                        &mut pending_buffer_accesses,
                        &step_buffer_accesses,
                    );
                    if !dependencies.is_empty() {
                        let buffer_barriers = dependencies
                            .iter()
                            .map(|dependency| {
                                let consumes_indirect_command = step
                                    .indirect_dispatch
                                    .is_some_and(|indirect| {
                                        indirect.buffer.buffer == dependency.buffer
                                    });
                                let consumes_condition = step
                                    .condition
                                    .is_some_and(|condition| {
                                        condition.buffer.buffer == dependency.buffer
                                    });
                                vk::BufferMemoryBarrier::default()
                                    .src_access_mask(
                                        vk::AccessFlags::SHADER_READ
                                            | vk::AccessFlags::SHADER_WRITE,
                                    )
                                    .dst_access_mask(
                                        vk::AccessFlags::SHADER_READ
                                            | vk::AccessFlags::SHADER_WRITE
                                            | if consumes_indirect_command {
                                                vk::AccessFlags::INDIRECT_COMMAND_READ
                                            } else {
                                                vk::AccessFlags::empty()
                                            }
                                            | if consumes_condition {
                                                vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
                                            } else {
                                                vk::AccessFlags::empty()
                                            },
                                        )
                                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                                    .buffer(dependency.buffer)
                                    .offset(0)
                                    .size(vk::WHOLE_SIZE)
                            })
                            .collect::<Vec<_>>();
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER
                                | if step.indirect_dispatch.is_some() {
                                    vk::PipelineStageFlags::DRAW_INDIRECT
                                } else {
                                    vk::PipelineStageFlags::empty()
                                }
                                | if step.condition.is_some() {
                                    vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT
                                } else {
                                    vk::PipelineStageFlags::empty()
                                },
                            vk::DependencyFlags::empty(),
                            &[],
                            &buffer_barriers,
                            &[],
                        );
                    }

                    if let Some(condition) = recorded_condition
                        && active_condition.is_none()
                    {
                        let conditional_rendering =
                            self.conditional_rendering.as_ref().ok_or_else(|| {
                                VulkanError(
                                    "conditional compute dispatch was recorded on a device without VK_EXT_conditional_rendering"
                                        .to_string(),
                                )
                            })?;
                        let flags = if condition.inverted {
                            vk::ConditionalRenderingFlagsEXT::INVERTED
                        } else {
                            vk::ConditionalRenderingFlagsEXT::empty()
                        };
                        let begin =
                            vk::ConditionalRenderingBeginInfoEXT::default()
                                .buffer(condition.buffer)
                                .offset(condition.offset)
                                .flags(flags);
                        (conditional_rendering
                            .fp()
                            .cmd_begin_conditional_rendering_ext)(
                            sequence.command_buffer,
                            &begin,
                        );
                        active_condition = Some(condition);
                    }
                    self.device.cmd_bind_pipeline(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline,
                    );
                    self.device.cmd_bind_descriptor_sets(
                        sequence.command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        step.dispatch.pipeline_layout,
                        0,
                        &[step.dispatch.descriptor_set],
                        &[],
                    );
                    if !step.push_constants.is_empty() {
                        self.device.cmd_push_constants(
                            sequence.command_buffer,
                            step.dispatch.pipeline_layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            step.push_constants,
                        );
                    }
                    if let Some(indirect) = step.indirect_dispatch {
                        self.device.cmd_dispatch_indirect(
                            sequence.command_buffer,
                            indirect.buffer.buffer,
                            indirect.offset,
                        );
                    } else if step.dispatch.base_workgroup_z == 0 {
                        self.device.cmd_dispatch(
                            sequence.command_buffer,
                            step.workgroup_count_x(),
                            step.workgroup_count_y(),
                            1,
                        );
                    } else {
                        self.device.cmd_dispatch_base(
                            sequence.command_buffer,
                            0,
                            0,
                            step.dispatch.base_workgroup_z,
                            step.workgroup_count_x(),
                            step.workgroup_count_y(),
                            1,
                        );
                    }
                    merge_resident_kernel_buffer_accesses(
                        &mut pending_buffer_accesses,
                        &step_buffer_accesses,
                    );
                    if let Some(query_pool) = query_pool {
                        self.device.cmd_write_timestamp(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query_pool,
                            u32::try_from(step_index + 1).map_err(|_| {
                                VulkanError(
                                    "resident kernel timestamp index overflowed".to_string(),
                                )
                            })?,
                        );
                    }
                    if let Some((query_pool, _)) =
                        sequence.profiling_timestamp_query_pool
                    {
                        self.device.cmd_write_timestamp(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query_pool,
                            u32::try_from(step_index + 1).map_err(|_| {
                                VulkanError(
                                    "resident kernel profile timestamp index overflowed"
                                        .to_string(),
                                )
                            })?,
                        );
                    }

                    let step_snapshot_copies = snapshot_copies
                        .iter()
                        .filter(|copy| copy.after_step_index == step_index)
                        .collect::<Vec<_>>();
                    if !step_snapshot_copies.is_empty() {
                        let compute_to_transfer = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::DependencyFlags::empty(),
                            &compute_to_transfer,
                            &[],
                            &[],
                        );
                        for copy in step_snapshot_copies {
                            let copy = copy.copy();
                            let regions = [vk::BufferCopy {
                                src_offset: copy.source_offset,
                                dst_offset: copy.destination_offset,
                                size: copy.byte_len,
                            }];
                            self.device.cmd_copy_buffer(
                                sequence.command_buffer,
                                copy.source,
                                copy.destination,
                                &regions,
                            );
                        }
                        let transfer_to_compute = [vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            )];
                        self.device.cmd_pipeline_barrier(
                            sequence.command_buffer,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &transfer_to_compute,
                            &[],
                            &[],
                        );
                        pending_buffer_accesses.clear();
                    }
                    let critical_path_region_ends = step
                        .critical_path_region_index
                        .is_some_and(|region_index| {
                            steps
                                .get(step_index + 1)
                                .and_then(|next| next.critical_path_region_index)
                                != Some(region_index)
                        });
                    if critical_path_region_ends {
                        if active_condition.is_some() {
                            let conditional_rendering =
                                self.conditional_rendering.as_ref().ok_or_else(|| {
                                    VulkanError(
                                        "conditional compute dispatch was recorded on a device without VK_EXT_conditional_rendering"
                                            .to_string(),
                                    )
                                })?;
                            (conditional_rendering
                                .fp()
                                .cmd_end_conditional_rendering_ext)(
                                sequence.command_buffer,
                            );
                            active_condition = None;
                        }
                        if let Some((query_pool, _)) =
                            sequence.critical_path_timestamp_query_pool
                        {
                            let query_index = step
                                .critical_path_region_index
                                .expect("critical-path region end has an index")
                                .checked_add(1)
                                .ok_or_else(|| {
                                    VulkanError(
                                        "resident kernel critical-path timestamp index overflowed"
                                            .to_string(),
                                    )
                                })?;
                            self.device.cmd_write_timestamp(
                                sequence.command_buffer,
                                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                query_pool,
                                query_index,
                            );
                        }
                    }
                }
                if active_condition.is_some() {
                    let conditional_rendering =
                        self.conditional_rendering.as_ref().ok_or_else(|| {
                            VulkanError(
                                "conditional compute dispatch was recorded on a device without VK_EXT_conditional_rendering"
                                    .to_string(),
                            )
                        })?;
                    (conditional_rendering
                        .fp()
                        .cmd_end_conditional_rendering_ext)(
                        sequence.command_buffer,
                    );
                }

                if let Some(query_pool) = sequence.timestamp_query_pool {
                    self.device.cmd_write_timestamp(
                        sequence.command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query_pool,
                        1,
                    );
                }

                let host_visibility_barrier = [vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::HOST_READ)];
                self.device.cmd_pipeline_barrier(
                    sequence.command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &host_visibility_barrier,
                    &[],
                    &[],
                );

                self.device
                    .end_command_buffer(sequence.command_buffer)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to end resident kernel sequence command buffer: {error:?}"
                        ))
                    })?;

                if profiling_enabled {
                    *sequence.recorded_input_copies.borrow_mut() = None;
                    *sequence.recorded_steps.borrow_mut() = None;
                    *sequence.recorded_snapshot_copies.borrow_mut() = None;
                } else {
                    *sequence.recorded_input_copies.borrow_mut() = Some(
                        input_copies
                            .iter()
                            .copied()
                            .map(VulkanResidentKernelSequenceInputCopy::recorded)
                            .collect(),
                    );
                    *sequence.recorded_steps.borrow_mut() = Some(
                        steps
                            .iter()
                            .map(|step| VulkanResidentKernelRecordedStep {
                                pipeline: step.dispatch.pipeline,
                                descriptor_set: step.dispatch.descriptor_set,
                                workgroup_count_x: step.workgroup_count_x(),
                                workgroup_count_y: step.workgroup_count_y(),
                                base_workgroup_z: step.dispatch.base_workgroup_z,
                                indirect_dispatch: step.indirect_dispatch.map(|indirect| {
                                    VulkanResidentKernelRecordedIndirectDispatch {
                                        buffer: indirect.buffer.buffer,
                                        offset: indirect.offset,
                                    }
                                }),
                                condition: step.condition.map(
                                    VulkanResidentKernelSequenceCondition::recorded,
                                ),
                                critical_path_region_index: step.critical_path_region_index,
                                push_constants: step.push_constants.to_vec(),
                                execution_family: step.dispatch.execution_family(),
                                semantic_label: step.dispatch.semantic_label.clone(),
                                estimated_work_units: u64::from(step.workgroup_count_x())
                                    .saturating_mul(u64::from(step.workgroup_count_y()))
                                    .saturating_mul(u64::from(step.dispatch.local_size_x())),
                                estimated_memory_bytes: step.dispatch.estimated_memory_bytes(),
                            })
                            .collect(),
                    );
                    *sequence.recorded_snapshot_copies.borrow_mut() = Some(
                        snapshot_copies
                            .iter()
                            .copied()
                            .map(VulkanResidentKernelSequenceSnapshotCopy::recorded)
                            .collect(),
                    );
                }
            }

            if !execute {
                return Ok(());
            }

            self.submit_resident_kernel_sequence_and_wait(sequence)?;
            let host_submit_wait_ns = host_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();

            if let Some(query_pool) = query_pool {
                let mut timestamps = vec![0u64; query_count as usize];
                let result = self.device.get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                );
                self.device.destroy_query_pool(query_pool, None);
                result.map_err(|error| {
                    VulkanError(format!(
                        "failed to read resident kernel timestamps: {error:?}"
                    ))
                })?;
                print_resident_kernel_timestamp_summary(
                    steps,
                    &timestamps,
                    sequence.timestamp_period_ns,
                    host_submit_wait_ns,
                );
            }

            Ok(())
        }
    }
}
