const VULKAN_DEVICE_LOCAL_RESERVABLE_FRACTION_PPM: u64 = 950_000;
const VULKAN_CAPACITY_PARTS_PER_MILLION: u64 = 1_000_000;
const VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_BYTE_CAP: u64 = 16 * 1024 * 1024;
const VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_HEADROOM_DIVISOR: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanDeviceLocalMemoryBudget {
    pub baseline_available_bytes: u64,
    pub reservable_bytes: u64,
    pub protected_headroom_bytes: u64,
    pub counter_tolerance_bytes: u64,
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
        currently_available_bytes: u64,
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
        let before = state.accounting_at(currently_available_bytes);
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
        let projected_total = projected_tracked
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
                "device-local capacity changed before a {}-byte permit could commit its {allocation_byte_count}-byte allocation: {projected_total} accounted bytes would exceed {maximum_accounted}",
                self.byte_count,
            )));
        }
        state.pending_reservation_bytes = projected_pending;
        state.tracked_allocation_bytes = projected_tracked;
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
        let reservable_bytes = baseline_available_bytes
            .saturating_mul(VULKAN_DEVICE_LOCAL_RESERVABLE_FRACTION_PPM)
            / VULKAN_CAPACITY_PARTS_PER_MILLION;
        Self {
            baseline_available_bytes,
            reservable_bytes,
            protected_headroom_bytes: baseline_available_bytes.saturating_sub(reservable_bytes),
            counter_tolerance_bytes: baseline_available_bytes
                .saturating_sub(reservable_bytes)
                .checked_div(VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_HEADROOM_DIVISOR)
                .unwrap_or(0)
                .min(VULKAN_DEVICE_LOCAL_COUNTER_TOLERANCE_BYTE_CAP),
        }
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
