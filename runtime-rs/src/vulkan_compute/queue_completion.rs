/// A reusable queue resource must never recycle binary completion state. Each
/// replay reserves a unique timeline value, and the resource remains in flight
/// until that exact value is observed. Sharing this object lets deferred queue
/// templates reserve completion at submission time without borrowing the
/// command owner.
struct VulkanMonotonicQueueCompletion {
    timeline: VulkanTimelineSemaphore,
    device_health: VulkanDeviceHealth,
    state: VulkanMonotonicCompletionState,
}

#[derive(Default)]
struct VulkanMonotonicCompletionState {
    last_reserved_value: Cell<u64>,
    pending_value: Cell<Option<u64>>,
}

impl VulkanMonotonicQueueCompletion {
    fn new(timeline: VulkanTimelineSemaphore, device_health: VulkanDeviceHealth) -> Self {
        Self {
            timeline,
            device_health,
            state: VulkanMonotonicCompletionState::default(),
        }
    }

    fn semaphore(&self) -> vk::Semaphore {
        self.timeline.semaphore
    }
}

impl VulkanMonotonicCompletionState {
    fn reserve(&self, label: &str) -> Result<u64, VulkanError> {
        if let Some(pending) = self.pending_value.get() {
            return Err(VulkanError(format!(
                "{label} already has pending completion timeline value {pending}"
            )));
        }
        let value = self
            .last_reserved_value
            .get()
            .checked_add(1)
            .ok_or_else(|| VulkanError(format!("{label} completion timeline overflowed")))?;
        self.last_reserved_value.set(value);
        self.pending_value.set(Some(value));
        Ok(value)
    }

    fn pending(&self, label: &str) -> Result<u64, VulkanError> {
        self.pending_value
            .get()
            .ok_or_else(|| VulkanError(format!("{label} has no pending submission")))
    }

    fn complete(&self, value: u64) -> Result<(), VulkanError> {
        if self.pending_value.get() != Some(value) {
            return Err(VulkanError(format!(
                "completion timeline value {value} does not match pending value {:?}",
                self.pending_value.get(),
            )));
        }
        self.pending_value.set(None);
        Ok(())
    }

    fn cancel(&self, value: u64) {
        if self.pending_value.get() == Some(value) {
            self.pending_value.set(None);
        }
    }
}

impl VulkanMonotonicQueueCompletion {
    fn reserve(&self, label: &str) -> Result<u64, VulkanError> {
        if let Some(pending) = self.pending_value() {
            let observed = unsafe {
                self.timeline
                    .device
                    .get_semaphore_counter_value(self.timeline.semaphore)
            }
            .map_err(|error| {
                vulkan_error_with_device_quarantine(
                    &self.device_health,
                    error,
                    VulkanError(format!(
                    "failed to inspect {label} completion timeline: {error:?}"
                    )),
                )
            })?;
            if observed >= pending {
                self.state.complete(pending)?;
            }
        }
        self.state.reserve(label)
    }

    fn pending(&self, label: &str) -> Result<u64, VulkanError> {
        self.state.pending(label)
    }

    fn complete(&self, value: u64) -> Result<(), VulkanError> {
        self.state.complete(value)
    }

    fn cancel(&self, value: u64) {
        self.state.cancel(value);
    }

    fn pending_value(&self) -> Option<u64> {
        self.state.pending_value.get()
    }
}

#[cfg(test)]
mod monotonic_queue_completion_tests {
    use super::*;

    #[test]
    fn values_are_monotonic_and_cancelled_values_are_not_reused() {
        let state = VulkanMonotonicCompletionState::default();
        let first = state.reserve("test queue").unwrap();
        assert_eq!(first, 1);
        assert!(state.reserve("test queue").unwrap_err().0.contains("pending"));
        state.complete(first).unwrap();
        let second = state.reserve("test queue").unwrap();
        state.cancel(second);
        assert_eq!(second, 2);
        assert_eq!(state.reserve("test queue").unwrap(), 3);
    }

    #[test]
    fn completion_rejects_a_different_submission_value() {
        let state = VulkanMonotonicCompletionState::default();
        let value = state.reserve("test queue").unwrap();
        assert!(state.complete(value + 1).unwrap_err().0.contains("does not match"));
        assert_eq!(state.pending("test queue").unwrap(), value);
    }
}
