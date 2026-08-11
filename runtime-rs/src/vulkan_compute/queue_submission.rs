#[derive(Clone)]
struct VulkanQueueSubmissionGate {
    queue: vk::Queue,
    submission: Arc<Mutex<()>>,
    memory_lifecycle: Arc<std::sync::RwLock<()>>,
}

impl VulkanQueueSubmissionGate {
    fn new(queue: vk::Queue, memory_lifecycle: Arc<std::sync::RwLock<()>>) -> Self {
        Self {
            queue,
            submission: Arc::new(Mutex::new(())),
            memory_lifecycle,
        }
    }

    fn paired(
        compute_queue: vk::Queue,
        transfer_queue: vk::Queue,
        memory_lifecycle: Arc<std::sync::RwLock<()>>,
    ) -> (Self, Self) {
        let compute = Self::new(compute_queue, Arc::clone(&memory_lifecycle));
        let transfer = if transfer_queue == compute_queue {
            compute.clone()
        } else {
            Self::new(transfer_queue, memory_lifecycle)
        };
        (compute, transfer)
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
        unsafe { device.queue_submit2(self.queue, submissions, fence) }
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
        );
        assert_eq!(compute.queue(), compute_queue);
        assert_eq!(transfer.queue(), same_transfer_queue);
        assert!(compute.serializes_with(&transfer));

        let (compute, transfer) = VulkanQueueSubmissionGate::paired(
            compute_queue,
            distinct_transfer_queue,
            memory_lifecycle,
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
        );
        let second = VulkanQueueSubmissionGate::new(
            vk::Queue::from_raw(23),
            Arc::clone(&memory_lifecycle),
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
}
