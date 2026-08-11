#[derive(Clone)]
struct VulkanQueueProgressTimeline {
    semaphore: vk::Semaphore,
    last_submitted_value: Arc<AtomicU64>,
}

impl VulkanQueueProgressTimeline {
    fn new(semaphore: vk::Semaphore) -> Self {
        Self {
            semaphore,
            last_submitted_value: Arc::new(AtomicU64::new(0)),
        }
    }

    fn planned_values(&self, submission_count: usize) -> Result<Vec<u64>, vk::Result> {
        let submission_count = u64::try_from(submission_count)
            .map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        let first = self
            .last_submitted_value
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        let last = first
            .checked_add(submission_count.saturating_sub(1))
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        Ok((first..=last).collect())
    }

    fn publish(&self, value: u64) {
        self.last_submitted_value.store(value, Ordering::Release);
    }

    fn latest_point(&self) -> Option<(vk::Semaphore, u64)> {
        let value = self.last_submitted_value.load(Ordering::Acquire);
        (value != 0).then_some((self.semaphore, value))
    }
}

unsafe fn create_vulkan_queue_progress_timeline(
    device: &ash::Device,
) -> Result<VulkanQueueProgressTimeline, vk::Result> {
    let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type),
            None,
        )
    }?;
    Ok(VulkanQueueProgressTimeline::new(semaphore))
}

#[derive(Clone)]
struct VulkanQueueSubmissionGate {
    queue: vk::Queue,
    submission: Arc<Mutex<()>>,
    memory_lifecycle: Arc<std::sync::RwLock<()>>,
    progress: Option<VulkanQueueProgressTimeline>,
}

impl VulkanQueueSubmissionGate {
    fn new(
        queue: vk::Queue,
        memory_lifecycle: Arc<std::sync::RwLock<()>>,
        progress: Option<VulkanQueueProgressTimeline>,
    ) -> Self {
        Self {
            queue,
            submission: Arc::new(Mutex::new(())),
            memory_lifecycle,
            progress,
        }
    }

    fn paired(
        compute_queue: vk::Queue,
        transfer_queue: vk::Queue,
        memory_lifecycle: Arc<std::sync::RwLock<()>>,
        compute_progress: Option<VulkanQueueProgressTimeline>,
        transfer_progress: Option<VulkanQueueProgressTimeline>,
    ) -> (Self, Self) {
        let compute = Self::new(
            compute_queue,
            Arc::clone(&memory_lifecycle),
            compute_progress,
        );
        let transfer = if transfer_queue == compute_queue {
            compute.clone()
        } else {
            Self::new(transfer_queue, memory_lifecycle, transfer_progress)
        };
        (compute, transfer)
    }

    fn latest_progress_point(&self) -> Option<(vk::Semaphore, u64)> {
        self.progress
            .as_ref()
            .and_then(VulkanQueueProgressTimeline::latest_point)
    }

    #[cfg(test)]
    fn queue(&self) -> vk::Queue {
        self.queue
    }

    #[cfg(test)]
    fn serializes_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.submission, &other.submission)
    }

    unsafe fn submit2(
        &self,
        device: &ash::Device,
        submissions: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        let _memory_lifecycle = self
            .memory_lifecycle
            .read()
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        let _submission = self
            .submission
            .lock()
            .map_err(|_| vk::Result::ERROR_DEVICE_LOST)?;
        let Some(progress) = &self.progress else {
            return unsafe { device.queue_submit2(self.queue, submissions, fence) };
        };
        if submissions.is_empty() {
            return unsafe { device.queue_submit2(self.queue, submissions, fence) };
        }
        let progress_values = progress.planned_values(submissions.len())?;
        let mut signal_infos = Vec::with_capacity(submissions.len());
        for (submission, progress_value) in submissions.iter().zip(&progress_values) {
            let existing = if submission.signal_semaphore_info_count == 0 {
                &[][..]
            } else {
                unsafe {
                    std::slice::from_raw_parts(
                        submission.p_signal_semaphore_infos,
                        submission.signal_semaphore_info_count as usize,
                    )
                }
            };
            let mut signals = Vec::with_capacity(existing.len() + 1);
            signals.extend_from_slice(existing);
            signals.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(progress.semaphore)
                    .value(*progress_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
            signal_infos.push(signals);
        }
        let mut augmented_submissions = submissions.to_vec();
        for (submission, signals) in augmented_submissions.iter_mut().zip(&signal_infos) {
            submission.signal_semaphore_info_count =
                u32::try_from(signals.len()).map_err(|_| vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
            submission.p_signal_semaphore_infos = signals.as_ptr();
        }
        unsafe { device.queue_submit2(self.queue, &augmented_submissions, fence) }?;
        progress.publish(
            *progress_values
                .last()
                .expect("non-empty submission has a progress value"),
        );
        Ok(())
    }

    fn with_exclusive<T>(
        &self,
        operation: impl FnOnce(vk::Queue) -> T,
    ) -> Result<T, VulkanError> {
        let _memory_lifecycle = self
            .memory_lifecycle
            .read()
            .map_err(|_| VulkanError("Vulkan memory lifecycle was poisoned".to_string()))?;
        self.with_exclusive_during_memory_reclamation(operation)
    }

    /// The caller holds the physical device's memory-lifecycle write lock.
    /// Acquiring its read side here would deadlock the reclamation transaction.
    fn with_exclusive_during_memory_reclamation<T>(
        &self,
        operation: impl FnOnce(vk::Queue) -> T,
    ) -> Result<T, VulkanError> {
        let _submission = self
            .submission
            .lock()
            .map_err(|_| VulkanError("Vulkan queue submission gate was poisoned".to_string()))?;
        Ok(operation(self.queue))
    }
}

struct VulkanPhysicalQueueQuiescer {
    physical_device_id: String,
    device: ash::Device,
    compute: VulkanQueueSubmissionGate,
    transfer: VulkanQueueSubmissionGate,
    transfer_is_distinct: bool,
    device_health: VulkanDeviceHealth,
}

impl std::fmt::Debug for VulkanPhysicalQueueQuiescer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanPhysicalQueueQuiescer")
            .field("physical_device_id", &self.physical_device_id)
            .field("transfer_is_distinct", &self.transfer_is_distinct)
            .finish_non_exhaustive()
    }
}

impl VulkanPhysicalQueueQuiescer {
    fn quiesce_during_memory_reclamation(&self) -> Result<(), VulkanError> {
        self.compute
            .with_exclusive_during_memory_reclamation(|queue| {
                quiesce_vulkan_queue_with_progress_watchdog(
                    &self.device,
                    queue,
                    &self.device_health,
                    "compute queue memory-reclamation quiescence",
                    |error| {
                        VulkanError(format!(
                            "failed to quiesce compute queue for physical device {:?} memory reclamation: {error:?}",
                            self.physical_device_id,
                        ))
                    },
                )
            })??;
        if self.transfer_is_distinct {
            self.transfer
                .with_exclusive_during_memory_reclamation(|queue| {
                    quiesce_vulkan_queue_with_progress_watchdog(
                        &self.device,
                        queue,
                        &self.device_health,
                        "transfer queue memory-reclamation quiescence",
                        |error| {
                            VulkanError(format!(
                                "failed to quiesce transfer queue for physical device {:?} memory reclamation: {error:?}",
                                self.physical_device_id,
                            ))
                        },
                    )
                })??;
        }
        Ok(())
    }
}

#[cfg(test)]
mod queue_submission_gate_tests {
    use super::*;

    #[test]
    fn queue_submission_gates_follow_physical_queue_identity() {
        let compute_queue = vk::Queue::from_raw(17);
        let same_transfer_queue = vk::Queue::from_raw(17);
        let distinct_transfer_queue = vk::Queue::from_raw(23);
        let memory_lifecycle = Arc::new(std::sync::RwLock::new(()));

        let (compute, transfer) = VulkanQueueSubmissionGate::paired(
            compute_queue,
            same_transfer_queue,
            Arc::clone(&memory_lifecycle),
            None,
            None,
        );
        assert_eq!(compute.queue(), compute_queue);
        assert_eq!(transfer.queue(), same_transfer_queue);
        assert!(compute.serializes_with(&transfer));

        let (compute, transfer) = VulkanQueueSubmissionGate::paired(
            compute_queue,
            distinct_transfer_queue,
            memory_lifecycle,
            None,
            None,
        );
        assert_eq!(transfer.queue(), distinct_transfer_queue);
        assert!(!compute.serializes_with(&transfer));
    }

    #[test]
    fn physical_memory_lifecycle_excludes_submissions_from_every_logical_gate() {
        let memory_lifecycle = Arc::new(std::sync::RwLock::new(()));
        let first = VulkanQueueSubmissionGate::new(
            vk::Queue::from_raw(17),
            Arc::clone(&memory_lifecycle),
            None,
        );
        let second = VulkanQueueSubmissionGate::new(
            vk::Queue::from_raw(23),
            Arc::clone(&memory_lifecycle),
            None,
        );
        let memory_reclamation = memory_lifecycle.write().unwrap();
        let (completed_send, completed_receive) = std::sync::mpsc::channel();
        let submission = std::thread::spawn(move || {
            second.with_exclusive(|_| ()).unwrap();
            completed_send.send(()).unwrap();
        });

        assert!(
            completed_receive
                .recv_timeout(std::time::Duration::from_millis(50))
                .is_err(),
            "a logical queue gate submitted while physical memory reclamation was exclusive",
        );
        drop(memory_reclamation);
        completed_receive
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        submission.join().unwrap();
        first.with_exclusive(|_| ()).unwrap();
    }

    #[test]
    fn queue_progress_values_publish_only_after_a_successful_submission() {
        let progress = VulkanQueueProgressTimeline::new(vk::Semaphore::from_raw(29));
        assert_eq!(progress.planned_values(3).unwrap(), vec![1, 2, 3]);
        assert_eq!(progress.latest_point(), None);

        progress.publish(3);
        assert_eq!(
            progress.latest_point(),
            Some((vk::Semaphore::from_raw(29), 3))
        );
        assert_eq!(progress.planned_values(2).unwrap(), vec![4, 5]);
    }
}
