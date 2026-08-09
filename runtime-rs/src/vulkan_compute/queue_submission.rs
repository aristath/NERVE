#[derive(Clone)]
struct VulkanQueueSubmissionGate {
    queue: vk::Queue,
    submission: Arc<Mutex<()>>,
}

impl VulkanQueueSubmissionGate {
    fn new(queue: vk::Queue) -> Self {
        Self {
            queue,
            submission: Arc::new(Mutex::new(())),
        }
    }

    fn paired(compute_queue: vk::Queue, transfer_queue: vk::Queue) -> (Self, Self) {
        let compute = Self::new(compute_queue);
        let transfer = if transfer_queue == compute_queue {
            compute.clone()
        } else {
            Self::new(transfer_queue)
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
        let _submission = self
            .submission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        unsafe { device.queue_submit2(self.queue, submissions, fence) }
    }

    fn with_exclusive<T>(&self, operation: impl FnOnce(vk::Queue) -> T) -> T {
        let _submission = self
            .submission
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        operation(self.queue)
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

        let (compute, transfer) =
            VulkanQueueSubmissionGate::paired(compute_queue, same_transfer_queue);
        assert_eq!(compute.queue(), compute_queue);
        assert_eq!(transfer.queue(), same_transfer_queue);
        assert!(compute.serializes_with(&transfer));

        let (compute, transfer) =
            VulkanQueueSubmissionGate::paired(compute_queue, distinct_transfer_queue);
        assert_eq!(transfer.queue(), distinct_transfer_queue);
        assert!(!compute.serializes_with(&transfer));
    }
}
