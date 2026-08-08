const VULKAN_QUEUE_PROGRESS_POLL_NS: u64 = RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS;
const VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS: u64 =
    RUNTIME_EXECUTION_TARGET_QUANTUM_DURATION_NS.saturating_mul(4);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct VulkanTimestampQueryProgress {
    value: u64,
    available: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct VulkanQueueProgressSources<'a> {
    timeline_points: &'a [(vk::Semaphore, u64)],
    timestamp_query_pool: Option<vk::QueryPool>,
}

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

fn queue_progress_values(
    device: &ash::Device,
    sources: VulkanQueueProgressSources<'_>,
) -> Result<Vec<u64>, vk::Result> {
    let mut values = Vec::with_capacity(
        sources.timeline_points.len()
            + usize::from(sources.timestamp_query_pool.is_some()) * 4,
    );
    for (semaphore, target) in sources.timeline_points {
        let current = unsafe { device.get_semaphore_counter_value(*semaphore) }?;
        values.push(current.min(*target));
    }
    if let Some(query_pool) = sources.timestamp_query_pool {
        let mut queries = [VulkanTimestampQueryProgress::default(); 2];
        let result = unsafe {
            device.get_query_pool_results(
                query_pool,
                0,
                &mut queries,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        if let Err(error) = result
            && error != vk::Result::NOT_READY
        {
            return Err(error);
        }
        for query in queries {
            values.push(query.available.min(1));
            values.push(query.available.min(1).saturating_mul(query.value));
        }
    }
    Ok(values)
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
    let completion_points = semaphores
        .iter()
        .copied()
        .zip(target_values.iter().copied())
        .collect::<Vec<_>>();
    wait_for_vulkan_timeline_points_with_progress_sources(
        device,
        semaphores,
        target_values,
        wait_any,
        device_health,
        operation,
        VulkanQueueProgressSources {
            timeline_points: &completion_points,
            timestamp_query_pool: None,
        },
        map_vulkan_error,
    )
}

#[allow(clippy::too_many_arguments)]
fn wait_for_vulkan_timeline_points_with_progress_sources<F>(
    device: &ash::Device,
    semaphores: &[vk::Semaphore],
    target_values: &[u64],
    wait_any: bool,
    device_health: &VulkanDeviceHealth,
    operation: &str,
    sources: VulkanQueueProgressSources<'_>,
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
    let initial_values = match queue_progress_values(device, sources) {
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
                let values = match queue_progress_values(device, sources) {
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
    submit_vulkan_command_buffers_and_wait_with_progress_watchdog(
        device,
        queue,
        &[],
        device_health,
        operation,
        map_vulkan_error,
    )
}

/// Submit an optional command-buffer set and establish host-visible queue
/// completion without allocating or recycling a binary fence. A one-shot
/// timeline is destroyed only after its signal is observed; on an uncertain
/// failure it is deliberately retained with the quarantined queue.
fn submit_vulkan_command_buffers_and_wait_with_progress_watchdog<F>(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffers: &[vk::CommandBuffer],
    device_health: &VulkanDeviceHealth,
    operation: &str,
    map_vulkan_error: F,
) -> Result<(), VulkanError>
where
    F: Fn(vk::Result) -> VulkanError,
{
    device_health.require_healthy()?;
    let mut timeline_type = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let timeline = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_type),
            None,
        )
    }
    .map_err(&map_vulkan_error)?;
    let command_infos = command_buffers
        .iter()
        .copied()
        .map(|command_buffer| {
            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
        })
        .collect::<Vec<_>>();
    let signal_infos = [vk::SemaphoreSubmitInfo::default()
        .semaphore(timeline)
        .value(1)
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
    let submit = vk::SubmitInfo2::default()
        .command_buffer_infos(&command_infos)
        .signal_semaphore_infos(&signal_infos);
    if let Err(error) = unsafe { device.queue_submit2(queue, &[submit], vk::Fence::null()) } {
        unsafe { device.destroy_semaphore(timeline, None) };
        let mapped = map_vulkan_error(error);
        return Err(vulkan_error_with_device_quarantine(
            device_health,
            error,
            mapped,
        ));
    }
    let wait_result = wait_for_vulkan_timeline_points_with_progress_watchdog(
        device,
        &[timeline],
        &[1],
        false,
        device_health,
        operation,
        &map_vulkan_error,
    );
    if wait_result.is_ok() {
        unsafe { device.destroy_semaphore(timeline, None) };
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

    #[test]
    fn dependency_and_timestamp_counters_are_first_class_progress() {
        let start = Instant::now();
        // [dependency timeline, timestamp availability, timestamp value,
        // completion availability, completion value]
        let mut watchdog = VulkanQueueProgressWatchdog::new(start, vec![4, 0, 0, 0, 0]);
        let first_deadline = start + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS);
        assert!(watchdog.is_stalled(first_deadline));

        let dependency_advanced = first_deadline - Duration::from_nanos(1);
        assert!(watchdog.observe(dependency_advanced, &[5, 0, 0, 0, 0]));
        let timestamp_started = dependency_advanced
            + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS - 1);
        assert!(watchdog.observe(timestamp_started, &[5, 1, 93, 0, 0]));
        assert!(!watchdog.is_stalled(
            timestamp_started + Duration::from_nanos(VULKAN_QUEUE_NO_PROGRESS_LIMIT_NS - 1)
        ));
    }
}
