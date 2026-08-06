pub const VULKAN_DEVICE_LOCAL_MEMORY_POLICY_SCHEMA: &str =
    "nerve.runtime.device_local_memory_policy.v1";
pub const VULKAN_CAPACITY_PARTS_PER_MILLION: u64 = 1_000_000;
pub const VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM: u64 = 200_000;
pub const VULKAN_DEVICE_LOCAL_RESERVABLE_FRACTION_PPM: u64 =
    VULKAN_CAPACITY_PARTS_PER_MILLION
        - VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM;
const VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_BYTE_CAP: u64 = 16 * 1024 * 1024;
const VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_HEADROOM_DIVISOR: u64 = 4;
const VULKAN_DEVICE_LOCAL_MEMORY_OBSERVER_INTERVAL: Duration = Duration::from_millis(25);
const VULKAN_EXECUTION_HEADROOM_OBSERVATION_MAXIMUM_AGE: Duration = Duration::from_millis(100);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryBudget {
    pub baseline_available_bytes: u64,
    pub reservable_bytes: u64,
    pub protected_headroom_bytes: u64,
    pub counter_tolerance_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryPolicy {
    pub schema: &'static str,
    pub capacity_parts_per_million: u64,
    pub protected_headroom_fraction_ppm: u64,
    pub reservable_free_vram_fraction_ppm: u64,
}

pub fn vulkan_device_local_memory_policy() -> VulkanDeviceLocalMemoryPolicy {
    VulkanDeviceLocalMemoryPolicy {
        schema: VULKAN_DEVICE_LOCAL_MEMORY_POLICY_SCHEMA,
        capacity_parts_per_million: VULKAN_CAPACITY_PARTS_PER_MILLION,
        protected_headroom_fraction_ppm:
            VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM,
        reservable_free_vram_fraction_ppm:
            VULKAN_DEVICE_LOCAL_RESERVABLE_FRACTION_PPM,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryAdmission {
    pub baseline_available_bytes: u64,
    pub currently_available_bytes: u64,
    pub reservable_bytes: u64,
    pub acquired_bytes: u64,
    pub pending_fixed_bytes: u64,
    pub allocatable_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryAccounting {
    pub baseline_available_bytes: u64,
    pub currently_available_bytes: u64,
    pub reservable_bytes: u64,
    pub tracked_allocation_bytes: u64,
    pub pending_reservation_bytes: u64,
    pub untracked_acquired_bytes: u64,
    pub remaining_bytes: u64,
    pub admissible_remaining_bytes: u64,
}

#[derive(Debug)]
struct VulkanDeviceLocalMemoryBudgetTracker {
    budget: VulkanDeviceLocalMemoryBudget,
    tracked_allocation_bytes: u64,
    pending_reservation_bytes: u64,
    allocation_generation: u64,
    execution_memory_observation: Option<(Instant, u64, u64)>,
    next_reclaimer_id: u64,
    reclaimers: BTreeMap<u64, std::sync::Weak<dyn VulkanDeviceLocalMemoryReclaimer>>,
}

pub trait VulkanDeviceLocalMemoryReclaimer: std::fmt::Debug + Send + Sync {
    fn reclaim_device_local_memory(&self, requested_bytes: usize) -> Result<usize, VulkanError>;
}

#[derive(Debug)]
pub struct VulkanDeviceLocalMemoryReclaimerRegistration {
    tracker: Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    reclaimer_id: u64,
    _reclaimer: Arc<dyn VulkanDeviceLocalMemoryReclaimer>,
}

#[derive(Debug)]
struct VulkanDeviceLocalMemoryReservation {
    tracker: Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    byte_count: u64,
}

#[derive(Debug)]
pub struct VulkanDeviceLocalMemoryPermit {
    tracker: Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    byte_count: u64,
}

impl VulkanDeviceLocalMemoryBudgetTracker {
    fn new(budget: VulkanDeviceLocalMemoryBudget) -> Self {
        Self {
            budget,
            tracked_allocation_bytes: 0,
            pending_reservation_bytes: 0,
            allocation_generation: 0,
            execution_memory_observation: None,
            next_reclaimer_id: 0,
            reclaimers: BTreeMap::new(),
        }
    }

    fn register_reclaimer(
        tracker: &Arc<Mutex<Self>>,
        reclaimer: Arc<dyn VulkanDeviceLocalMemoryReclaimer>,
    ) -> Result<VulkanDeviceLocalMemoryReclaimerRegistration, VulkanError> {
        let mut state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        let reclaimer_id = state.next_reclaimer_id;
        state.next_reclaimer_id = state.next_reclaimer_id.checked_add(1).ok_or_else(|| {
            VulkanError("device-local memory reclaimer identity overflowed".to_string())
        })?;
        state
            .reclaimers
            .insert(reclaimer_id, Arc::downgrade(&reclaimer));
        drop(state);
        Ok(VulkanDeviceLocalMemoryReclaimerRegistration {
            tracker: Arc::clone(tracker),
            reclaimer_id,
            _reclaimer: reclaimer,
        })
    }

    fn live_reclaimers(
        tracker: &Arc<Mutex<Self>>,
    ) -> Result<Vec<Arc<dyn VulkanDeviceLocalMemoryReclaimer>>, VulkanError> {
        let mut state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        let mut live = Vec::with_capacity(state.reclaimers.len());
        state.reclaimers.retain(|_, reclaimer| {
            if let Some(reclaimer) = reclaimer.upgrade() {
                live.push(reclaimer);
                true
            } else {
                false
            }
        });
        Ok(live)
    }

    fn accounting_at(
        &self,
        currently_available_bytes: u64,
    ) -> VulkanDeviceLocalMemoryAccounting {
        let acquired_bytes = self.budget.acquired_bytes_at(currently_available_bytes);
        let untracked_acquired_bytes =
            acquired_bytes.saturating_sub(self.tracked_allocation_bytes);
        let remaining_bytes = self
            .budget
            .reservable_bytes
            .saturating_sub(untracked_acquired_bytes)
            .saturating_sub(self.tracked_allocation_bytes)
            .saturating_sub(self.pending_reservation_bytes);
        let admissible_remaining_bytes = self
            .budget
            .reservable_bytes
            .saturating_add(self.budget.counter_tolerance_bytes)
            .saturating_sub(untracked_acquired_bytes)
            .saturating_sub(self.tracked_allocation_bytes)
            .saturating_sub(self.pending_reservation_bytes);
        VulkanDeviceLocalMemoryAccounting {
            baseline_available_bytes: self.budget.baseline_available_bytes,
            currently_available_bytes,
            reservable_bytes: self.budget.reservable_bytes,
            tracked_allocation_bytes: self.tracked_allocation_bytes,
            pending_reservation_bytes: self.pending_reservation_bytes,
            untracked_acquired_bytes,
            remaining_bytes,
            admissible_remaining_bytes,
        }
    }

    fn execution_accounting(
        tracker: &Arc<Mutex<Self>>,
        maximum_age: Duration,
        mut observe_available_bytes: impl FnMut() -> u64,
    ) -> Result<VulkanDeviceLocalMemoryAccounting, VulkanError> {
        loop {
            let allocation_generation = {
                let state = tracker.lock().map_err(|_| {
                    VulkanError("device-local memory budget tracker was poisoned".to_string())
                })?;
                if maximum_age > Duration::ZERO
                    && let Some((observed_at, available_bytes, observed_generation)) =
                        state.execution_memory_observation
                    && observed_generation == state.allocation_generation
                    && observed_at.elapsed() <= maximum_age
                {
                    return Ok(state.accounting_at(available_bytes));
                }
                state.allocation_generation
            };
            let available_bytes = observe_available_bytes();
            let mut state = tracker.lock().map_err(|_| {
                VulkanError("device-local memory budget tracker was poisoned".to_string())
            })?;
            if state.allocation_generation != allocation_generation {
                // An allocation was acquired or released while the driver
                // counter was sampled. That value describes neither side of
                // the completed accounting transition, so retry rather than
                // publishing it as a recent execution observation.
                continue;
            }
            state.execution_memory_observation = Some((
                Instant::now(),
                available_bytes,
                allocation_generation,
            ));
            return Ok(state.accounting_at(available_bytes));
        }
    }

    fn recent_execution_accounting(
        tracker: &Arc<Mutex<Self>>,
        maximum_age: Duration,
    ) -> Result<Option<VulkanDeviceLocalMemoryAccounting>, VulkanError> {
        let state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        Ok(state.execution_memory_observation.and_then(
            |(observed_at, available_bytes, observed_generation)| {
                (observed_generation == state.allocation_generation
                    && observed_at.elapsed() <= maximum_age)
                    .then(|| state.accounting_at(available_bytes))
            },
        ))
    }

    fn record_execution_observation(
        tracker: &Arc<Mutex<Self>>,
        allocation_generation: u64,
        available_bytes: u64,
    ) -> Result<bool, VulkanError> {
        let mut state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        if state.allocation_generation != allocation_generation {
            return Ok(false);
        }
        state.execution_memory_observation = Some((
            Instant::now(),
            available_bytes,
            allocation_generation,
        ));
        Ok(true)
    }

    fn allocation_generation(tracker: &Arc<Mutex<Self>>) -> Result<u64, VulkanError> {
        tracker
            .lock()
            .map(|state| state.allocation_generation)
            .map_err(|_| {
                VulkanError("device-local memory budget tracker was poisoned".to_string())
            })
    }

    fn invalidate_execution_memory_observation(&mut self) {
        self.allocation_generation = self.allocation_generation.wrapping_add(1);
        self.execution_memory_observation = None;
    }
}

fn start_device_local_memory_observer(
    context: &Arc<VulkanInstanceContext>,
    physical_device: vk::PhysicalDevice,
    memory_budget_supported: bool,
    device_local_memory_bytes: u64,
    tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    physical_device_id: &str,
) -> Result<(), VulkanError> {
    if !memory_budget_supported {
        return Ok(());
    }
    let context = Arc::clone(context);
    let tracker = Arc::downgrade(tracker);
    let thread_name = format!("nerve-vram-{}", physical_device_id.replace(':', "-"));
    std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || loop {
            let Some(tracker) = tracker.upgrade() else {
                break;
            };
            let Ok(allocation_generation) =
                VulkanDeviceLocalMemoryBudgetTracker::allocation_generation(&tracker)
            else {
                break;
            };
            let available_bytes = query_available_device_local_memory_bytes(
                &context.instance,
                physical_device,
                memory_budget_supported,
                device_local_memory_bytes,
            );
            let recorded = VulkanDeviceLocalMemoryBudgetTracker::record_execution_observation(
                &tracker,
                allocation_generation,
                available_bytes,
            )
            .unwrap_or(false);
            if recorded {
                let budget = tracker.lock().ok().map(|state| state.budget);
                if let Some(budget) = budget
                    && budget.protected_headroom_deficit_at(available_bytes) > 0
                    && let Ok(reclaimers) =
                        VulkanDeviceLocalMemoryBudgetTracker::live_reclaimers(&tracker)
                    && !reclaimers.is_empty()
                {
                    let _ = restore_protected_device_local_headroom(
                        budget,
                        reclaimers,
                        Duration::from_millis(250),
                        || {
                            let currently_available_bytes =
                                query_available_device_local_memory_bytes(
                                    &context.instance,
                                    physical_device,
                                    memory_budget_supported,
                                    device_local_memory_bytes,
                                );
                            tracker
                                .lock()
                                .map(|state| {
                                    state.accounting_at(currently_available_bytes)
                                })
                                .map_err(|_| {
                                    VulkanError(
                                        "device-local memory budget tracker was poisoned"
                                            .to_string(),
                                    )
                                })
                        },
                    );
                }
            }
            drop(tracker);
            std::thread::sleep(VULKAN_DEVICE_LOCAL_MEMORY_OBSERVER_INTERVAL);
        })
        .map(drop)
        .map_err(|error| {
            VulkanError(format!(
                "could not start device-local memory observer for {physical_device_id}: {error}"
            ))
        })
}

fn restore_protected_device_local_headroom(
    budget: VulkanDeviceLocalMemoryBudget,
    reclaimers: Vec<Arc<dyn VulkanDeviceLocalMemoryReclaimer>>,
    settlement_timeout: Duration,
    mut current_accounting: impl FnMut() -> Result<VulkanDeviceLocalMemoryAccounting, VulkanError>,
) -> Result<VulkanDeviceLocalMemoryAccounting, VulkanError> {
    let initial = current_accounting()?;
    let mut deficit = budget.protected_headroom_deficit_at(initial.currently_available_bytes);
    if deficit == 0 {
        return Ok(initial);
    }
    if reclaimers.is_empty() {
        return Err(VulkanError(format!(
            "device-local execution refused: only {} bytes are currently available, below the protected {}-byte headroom ({} bytes of counter tolerance), and no evictable residency store is registered",
            initial.currently_available_bytes,
            budget.protected_headroom_bytes,
            budget.counter_tolerance_bytes,
        )));
    }
    let mut reclaimed_bytes = 0usize;
    let mut reclaimer_errors = Vec::new();
    for reclaimer in reclaimers {
        let requested_bytes = usize::try_from(deficit).unwrap_or(usize::MAX);
        match reclaimer.reclaim_device_local_memory(requested_bytes) {
            Ok(reclaimed) => reclaimed_bytes = reclaimed_bytes.saturating_add(reclaimed),
            Err(error) => reclaimer_errors.push(error.to_string()),
        }
        let started = Instant::now();
        loop {
            let accounting = current_accounting()?;
            deficit = budget.protected_headroom_deficit_at(accounting.currently_available_bytes);
            if deficit == 0 {
                return Ok(accounting);
            }
            if started.elapsed() >= settlement_timeout {
                break;
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    let final_accounting = current_accounting()?;
    Err(VulkanError(format!(
        "device-local execution refused: protected headroom still lacks {} bytes after registered stores released {reclaimed_bytes} bytes{}",
        budget.protected_headroom_deficit_at(final_accounting.currently_available_bytes),
        if reclaimer_errors.is_empty() {
            String::new()
        } else {
            format!("; reclaimer errors: {}", reclaimer_errors.join(" | "))
        }
    )))
}

impl VulkanDeviceLocalMemoryReservation {
    fn acquire(
        tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
        currently_available_bytes: u64,
        byte_count: u64,
    ) -> Result<Arc<Self>, VulkanError> {
        if byte_count == 0 {
            return Err(VulkanError(
                "device-local memory reservation must not be empty".to_string(),
            ));
        }
        let mut state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        let before = state.accounting_at(currently_available_bytes);
        let projected = state
            .tracked_allocation_bytes
            .checked_add(byte_count)
            .ok_or_else(|| {
                VulkanError("device-local tracked allocation bytes overflowed".to_string())
            })?;
        let maximum_tracked = state
            .budget
            .reservable_bytes
            .saturating_add(state.budget.counter_tolerance_bytes)
            .saturating_sub(before.untracked_acquired_bytes);
        let projected_total = projected
            .checked_add(state.pending_reservation_bytes)
            .ok_or_else(|| {
                VulkanError("device-local accounted allocation bytes overflowed".to_string())
            })?;
        if projected_total > maximum_tracked {
            return Err(VulkanError(format!(
                "device-local allocation of {byte_count} bytes would raise accounted allocations from {} tracked plus {} pending to {projected_total} bytes, beyond the stable {}-byte budget plus {} bytes of bounded counter tolerance after {} untracked acquired bytes",
                state.tracked_allocation_bytes,
                state.pending_reservation_bytes,
                state.budget.reservable_bytes,
                state.budget.counter_tolerance_bytes,
                before.untracked_acquired_bytes,
            )));
        }
        state.tracked_allocation_bytes = projected;
        // The caller acquires this reservation immediately before creating the
        // Vulkan allocation. A cached execution observation taken before this
        // point must never authorize work against the pre-allocation heap
        // availability.
        state.invalidate_execution_memory_observation();
        drop(state);
        Ok(Arc::new(Self {
            tracker: Arc::clone(tracker),
            byte_count,
        }))
    }
}

impl VulkanDeviceLocalMemoryPermit {
    fn acquire(
        tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
        currently_available_bytes: u64,
        byte_count: u64,
    ) -> Result<Self, VulkanError> {
        if byte_count == 0 {
            return Err(VulkanError(
                "device-local memory permit must not be empty".to_string(),
            ));
        }
        let mut state = tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        let before = state.accounting_at(currently_available_bytes);
        let projected_pending = state
            .pending_reservation_bytes
            .checked_add(byte_count)
            .ok_or_else(|| {
                VulkanError("device-local pending reservation bytes overflowed".to_string())
            })?;
        let projected_total = state
            .tracked_allocation_bytes
            .checked_add(projected_pending)
            .ok_or_else(|| {
                VulkanError("device-local accounted allocation bytes overflowed".to_string())
            })?;
        let maximum_accounted = state
            .budget
            .reservable_bytes
            .saturating_add(state.budget.counter_tolerance_bytes)
            .saturating_sub(before.untracked_acquired_bytes);
        if projected_total > maximum_accounted {
            return Err(VulkanError(format!(
                "device-local capacity permit of {byte_count} bytes would raise accounted allocations from {} tracked plus {} pending to {projected_total} bytes, beyond the stable {}-byte budget plus {} bytes of bounded counter tolerance after {} untracked acquired bytes",
                state.tracked_allocation_bytes,
                state.pending_reservation_bytes,
                state.budget.reservable_bytes,
                state.budget.counter_tolerance_bytes,
                before.untracked_acquired_bytes,
            )));
        }
        state.pending_reservation_bytes = projected_pending;
        drop(state);
        Ok(Self {
            tracker: Arc::clone(tracker),
            byte_count,
        })
    }

    fn commit(
        mut self,
        allocation_byte_count: u64,
    ) -> Result<Arc<VulkanDeviceLocalMemoryReservation>, VulkanError> {
        if allocation_byte_count == 0 || allocation_byte_count > self.byte_count {
            return Err(VulkanError(format!(
                "device-local allocation needs {allocation_byte_count} bytes but its capacity permit holds {} bytes",
                self.byte_count,
            )));
        }
        let mut state = self.tracker.lock().map_err(|_| {
            VulkanError("device-local memory budget tracker was poisoned".to_string())
        })?;
        let projected_pending = state
            .pending_reservation_bytes
            .checked_sub(self.byte_count)
            .ok_or_else(|| {
                VulkanError("device-local capacity permit was not accounted".to_string())
            })?;
        let projected_tracked = state
            .tracked_allocation_bytes
            .checked_add(allocation_byte_count)
            .ok_or_else(|| {
                VulkanError("device-local tracked allocation bytes overflowed".to_string())
            })?;
        // Admission already accounted this permit in `pending_reservation_bytes`.
        // Re-reading VK_EXT_memory_budget here would make an admitted operation
        // depend on an asynchronous heap counter for a second time. In
        // particular, delayed accounting for a concurrent release can appear as
        // newly acquired, untracked memory and invalidate a permit even though
        // the transaction has not increased its reservation. Commit therefore
        // only changes the accounting class from pending to tracked. The Vulkan
        // allocation that immediately follows is the authoritative physical
        // capacity check; later admissions take a fresh counter snapshot and can
        // reclaim dynamic residency if external usage has grown.
        state.pending_reservation_bytes = projected_pending;
        state.tracked_allocation_bytes = projected_tracked;
        // The allocation backed by this permit now exists. Force the next
        // execution boundary to observe its physical heap cost instead of
        // reusing an observation captured while the allocation was pending.
        state.invalidate_execution_memory_observation();
        self.byte_count = 0;
        drop(state);
        Ok(Arc::new(VulkanDeviceLocalMemoryReservation {
            tracker: Arc::clone(&self.tracker),
            byte_count: allocation_byte_count,
        }))
    }
}

impl Drop for VulkanDeviceLocalMemoryPermit {
    fn drop(&mut self) {
        if self.byte_count == 0 {
            return;
        }
        if let Ok(mut state) = self.tracker.lock() {
            state.pending_reservation_bytes = state
                .pending_reservation_bytes
                .saturating_sub(self.byte_count);
        }
    }
}

impl Drop for VulkanDeviceLocalMemoryReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.tracker.lock() {
            state.tracked_allocation_bytes = state
                .tracked_allocation_bytes
                .saturating_sub(self.byte_count);
            // VulkanResidentBuffer releases its memory before dropping this
            // reservation, so a subsequent boundary can immediately observe
            // the restored heap capacity.
            state.invalidate_execution_memory_observation();
        }
    }
}

impl Drop for VulkanDeviceLocalMemoryReclaimerRegistration {
    fn drop(&mut self) {
        if let Ok(mut state) = self.tracker.lock() {
            state.reclaimers.remove(&self.reclaimer_id);
        }
    }
}

impl VulkanDeviceLocalMemoryBudget {
    fn capture(baseline_available_bytes: u64) -> Self {
        // A model may remain resident while unrelated workloads grow. Keep a
        // meaningful fraction of the opening availability outside NERVE's
        // stable allocation budget so unrelated display and compute clients do
        // not need TTM eviction merely to allocate. This is intentionally not
        // capped: an absolute cap diluted the safety invariant on larger heaps
        // and allowed every 32 GiB device to sit at the same unsafe 4 GiB
        // watermark.
        let protected_headroom_bytes = baseline_available_bytes
            .saturating_mul(VULKAN_DEVICE_LOCAL_PROTECTED_HEADROOM_FRACTION_PPM)
            .checked_div(VULKAN_CAPACITY_PARTS_PER_MILLION)
            .unwrap_or(0);
        let reservable_bytes = baseline_available_bytes.saturating_sub(protected_headroom_bytes);
        Self {
            baseline_available_bytes,
            reservable_bytes,
            protected_headroom_bytes,
            counter_tolerance_bytes: protected_headroom_bytes
                .checked_div(VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_HEADROOM_DIVISOR)
                .unwrap_or(0)
                .min(VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_BYTE_CAP),
        }
    }

    fn protected_headroom_deficit_at(&self, currently_available_bytes: u64) -> u64 {
        self.protected_headroom_bytes.saturating_sub(
            currently_available_bytes.saturating_add(self.counter_tolerance_bytes),
        )
    }

    pub fn acquired_bytes_at(&self, currently_available_bytes: u64) -> u64 {
        self.baseline_available_bytes
            .saturating_sub(currently_available_bytes)
    }

    pub fn remaining_bytes_at(&self, currently_available_bytes: u64) -> u64 {
        self.reservable_bytes
            .saturating_sub(self.acquired_bytes_at(currently_available_bytes))
    }

    pub fn admit_pending_bytes_at(
        &self,
        currently_available_bytes: u64,
        pending_fixed_bytes: u64,
    ) -> Result<VulkanDeviceLocalMemoryAdmission, VulkanError> {
        let acquired_bytes = self.acquired_bytes_at(currently_available_bytes);
        let remaining_bytes = self.remaining_bytes_at(currently_available_bytes);
        let allocatable_bytes = remaining_bytes.checked_sub(pending_fixed_bytes).ok_or_else(|| {
            VulkanError(format!(
                "pending fixed Vulkan residency needs {pending_fixed_bytes} bytes, but the stable device-local budget has only {remaining_bytes} bytes remaining after {acquired_bytes} bytes were acquired"
            ))
        })?;
        Ok(VulkanDeviceLocalMemoryAdmission {
            baseline_available_bytes: self.baseline_available_bytes,
            currently_available_bytes,
            reservable_bytes: self.reservable_bytes,
            acquired_bytes,
            pending_fixed_bytes,
            allocatable_bytes,
        })
    }
}

fn query_available_device_local_memory_bytes(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    memory_budget_supported: bool,
    device_local_memory_bytes: u64,
) -> u64 {
    if !memory_budget_supported {
        return device_local_memory_bytes;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        instance.get_physical_device_memory_properties2(physical_device, &mut properties);
    }
    (0..properties.memory_properties.memory_heap_count)
        .filter(|heap_index| {
            properties.memory_properties.memory_heaps[*heap_index as usize]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|heap_index| {
            let index = heap_index as usize;
            budget.heap_budget[index].saturating_sub(budget.heap_usage[index])
        })
        .max()
        .unwrap_or(0)
}
