const VULKAN_QUEUE_PROGRESS_POLL_NS: u64 = RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS;
const VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS: u64 =
    RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS.saturating_mul(4);

/// Tracks observable queue progress independently from total operation time.
/// Long operations remain legal while timeline counters or completed fences
/// advance; a queue is quarantined only after four complete execution-quantum
/// budgets without any observable progress.
struct VulkanQueueProgressWatchdog {
    last_progress_at: Instant,
    last_values: Vec<u64>,
}

fn vulkan_error_with_device_quarantine(
    device_health: &VulkanDeviceHealth,
    error: vk::Result,
    mapped: VulkanError,
) -> VulkanError {
    if error == vk::Result::ERROR_DEVICE_LOST {
        device_health.quarantine(mapped.to_string())
    } else {
        mapped
    }
}

impl VulkanQueueProgressWatchdog {
    fn new(now: Instant, initial_values: Vec<u64>) -> Self {
        Self {
            last_progress_at: now,
            last_values: initial_values,
        }
    }

    fn observe(&mut self, now: Instant, values: &[u64]) -> bool {
        let advanced = values.len() == self.last_values.len()
            && values
                .iter()
                .zip(&self.last_values)
                .any(|(current, previous)| current > previous);
        if advanced {
            self.last_progress_at = now;
            self.last_values.clone_from_slice(values);
        }
        advanced
    }

    fn is_stalled(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last_progress_at)
            >= Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS)
    }

    fn next_wait_ns(&self, now: Instant) -> u64 {
        let elapsed = now.saturating_duration_since(self.last_progress_at);
        let remaining = Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS)
            .saturating_sub(elapsed);
        u64::try_from(
            remaining
                .min(Duration::from_nanos(VULKAN_QUEUE_PROGRESS_POLL_NS))
                .as_nanos(),
        )
        .unwrap_or(VULKAN_QUEUE_PROGRESS_POLL_NS)
        .max(1)
    }
}

fn completed_fence_count(
    device: &ash::Device,
    fences: &[vk::Fence],
) -> Result<u64, vk::Result> {
    fences.iter().try_fold(0u64, |count, fence| {
        unsafe { device.get_fence_status(*fence) }
            .map(|complete| count.saturating_add(u64::from(complete)))
    })
}

fn wait_for_vulkan_fences_with_progress_watchdog<F>(
    device: &ash::Device,
    fences: &[vk::Fence],
    wait_all: bool,
    device_health: &VulkanDeviceHealth,
    operation: &str,
    map_vulkan_error: F,
) -> Result<(), VulkanError>
where
    F: Fn(vk::Result) -> VulkanError,
{
    if fences.is_empty() {
        return Err(VulkanError(format!(
            "{operation} cannot wait on an empty fence set"
        )));
    }
    device_health.require_healthy()?;
    let initial_completed = match completed_fence_count(device, fences) {
        Ok(completed) => completed,
        Err(error) => {
            let mapped = map_vulkan_error(error);
            if error == vk::Result::ERROR_DEVICE_LOST {
                return Err(device_health.quarantine(mapped.to_string()));
            }
            return Err(mapped);
        }
    };
    let mut watchdog = VulkanQueueProgressWatchdog::new(Instant::now(), vec![initial_completed]);
    loop {
        let now = Instant::now();
        if watchdog.is_stalled(now) {
            return Err(device_health.quarantine(format!(
                "{operation} made no observable queue progress for {} ns ({} of {} fences completed)",
                VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS,
                watchdog.last_values[0],
                fences.len()
            )));
        }
        match unsafe {
            device.wait_for_fences(fences, wait_all, watchdog.next_wait_ns(now))
        } {
            Ok(()) => {
                device_health.require_healthy()?;
                return Ok(());
            }
            Err(vk::Result::TIMEOUT) => {
                let completed = match completed_fence_count(device, fences) {
                    Ok(completed) => completed,
                    Err(error) => {
                        let mapped = map_vulkan_error(error);
                        if error == vk::Result::ERROR_DEVICE_LOST {
                            return Err(device_health.quarantine(mapped.to_string()));
                        }
                        return Err(mapped);
                    }
                };
                watchdog.observe(Instant::now(), &[completed]);
            }
            Err(error) => {
                let mapped = map_vulkan_error(error);
                if error == vk::Result::ERROR_DEVICE_LOST {
                    return Err(device_health.quarantine(mapped.to_string()));
                }
                return Err(mapped);
            }
        }
    }
}

fn timeline_counter_values(
    device: &ash::Device,
    semaphores: &[vk::Semaphore],
) -> Result<Vec<u64>, vk::Result> {
    semaphores
        .iter()
        .map(|semaphore| unsafe { device.get_semaphore_counter_value(*semaphore) })
        .collect()
}

fn wait_for_vulkan_timeline_points_with_progress_watchdog<F>(
    device: &ash::Device,
    semaphores: &[vk::Semaphore],
    target_values: &[u64],
    wait_any: bool,
    device_health: &VulkanDeviceHealth,
    operation: &str,
    map_vulkan_error: F,
) -> Result<(), VulkanError>
where
    F: Fn(vk::Result) -> VulkanError,
{
    if semaphores.is_empty() || semaphores.len() != target_values.len() {
        return Err(VulkanError(format!(
            "{operation} requires matching non-empty timeline semaphore and value sets"
        )));
    }
    device_health.require_healthy()?;
    let initial_values = match timeline_counter_values(device, semaphores) {
        Ok(values) => values,
        Err(error) => {
            let mapped = map_vulkan_error(error);
            if error == vk::Result::ERROR_DEVICE_LOST {
                return Err(device_health.quarantine(mapped.to_string()));
            }
            return Err(mapped);
        }
    };
    let mut watchdog = VulkanQueueProgressWatchdog::new(Instant::now(), initial_values);
    let flags = wait_any.then_some(vk::SemaphoreWaitFlags::ANY).unwrap_or_default();
    let wait_info = vk::SemaphoreWaitInfo::default()
        .flags(flags)
        .semaphores(semaphores)
        .values(target_values);
    loop {
        let now = Instant::now();
        if watchdog.is_stalled(now) {
            return Err(device_health.quarantine(format!(
                "{operation} made no observable queue progress for {} ns (current={:?}, target={target_values:?})",
                VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS, watchdog.last_values
            )));
        }
        match unsafe { device.wait_semaphores(&wait_info, watchdog.next_wait_ns(now)) } {
            Ok(()) => {
                device_health.require_healthy()?;
                return Ok(());
            }
            Err(vk::Result::TIMEOUT) => {
                let values = match timeline_counter_values(device, semaphores) {
                    Ok(values) => values,
                    Err(error) => {
                        let mapped = map_vulkan_error(error);
                        if error == vk::Result::ERROR_DEVICE_LOST {
                            return Err(device_health.quarantine(mapped.to_string()));
                        }
                        return Err(mapped);
                    }
                };
                watchdog.observe(Instant::now(), &values);
            }
            Err(error) => {
                let mapped = map_vulkan_error(error);
                if error == vk::Result::ERROR_DEVICE_LOST {
                    return Err(device_health.quarantine(mapped.to_string()));
                }
                return Err(mapped);
            }
        }
    }
}

fn quiesce_vulkan_queue_with_progress_watchdog<F>(
    device: &ash::Device,
    queue: vk::Queue,
    device_health: &VulkanDeviceHealth,
    operation: &str,
    map_vulkan_error: F,
) -> Result<(), VulkanError>
where
    F: Fn(vk::Result) -> VulkanError,
{
    device_health.require_healthy()?;
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .map_err(&map_vulkan_error)?;
    let submit_result = unsafe {
        device.queue_submit2(queue, &[vk::SubmitInfo2::default()], fence)
    };
    if let Err(error) = submit_result {
        unsafe { device.destroy_fence(fence, None) };
        let mapped = map_vulkan_error(error);
        return Err(vulkan_error_with_device_quarantine(
            device_health,
            error,
            mapped,
        ));
    }
    let wait_result = wait_for_vulkan_fences_with_progress_watchdog(
        device,
        &[fence],
        true,
        device_health,
        operation,
        &map_vulkan_error,
    );
    if wait_result.is_ok() {
        unsafe { device.destroy_fence(fence, None) };
    }
    wait_result
}

#[cfg(test)]
mod queue_progress_watchdog_tests {
    use super::*;

    #[test]
    fn progress_resets_the_stall_deadline_but_regression_does_not() {
        let start = Instant::now();
        let mut watchdog = VulkanQueueProgressWatchdog::new(start, vec![2, 7]);
        let almost_stalled = start + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS - 1);
        assert!(!watchdog.is_stalled(almost_stalled));
        assert!(watchdog.observe(almost_stalled, &[3, 7]));

        let before_reset_deadline = almost_stalled
            + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS - 1);
        assert!(!watchdog.is_stalled(before_reset_deadline));
        assert!(!watchdog.observe(before_reset_deadline, &[2, 6]));
        assert!(watchdog.is_stalled(
            almost_stalled + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS)
        ));
    }

    #[test]
    fn poll_wait_is_clamped_to_quantum_and_remaining_stall_budget() {
        let start = Instant::now();
        let watchdog = VulkanQueueProgressWatchdog::new(start, vec![0]);
        assert_eq!(
            watchdog.next_wait_ns(start),
            VULKAN_QUEUE_PROGRESS_POLL_NS
        );
        let nearly_stalled =
            start + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS - 17);
        assert_eq!(watchdog.next_wait_ns(nearly_stalled), 17);
    }
}
