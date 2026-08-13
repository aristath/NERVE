#[derive(Debug, Default)]
struct VulkanHostMemoryBudgetTracker {
    tracked_allocation_bytes: usize,
    pending_reservation_bytes: usize,
}

static VULKAN_HOST_MEMORY_BUDGET_TRACKER: OnceLock<
    Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
> = OnceLock::new();

fn vulkan_host_memory_budget_tracker() -> Arc<Mutex<VulkanHostMemoryBudgetTracker>> {
    Arc::clone(VULKAN_HOST_MEMORY_BUDGET_TRACKER.get_or_init(|| {
        Arc::new(Mutex::new(VulkanHostMemoryBudgetTracker::default()))
    }))
}

#[derive(Debug)]
struct VulkanHostMemoryPermit {
    tracker: Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
    byte_count: usize,
    recycle_pool: Option<Weak<Mutex<VulkanHostMemoryPermitPool>>>,
}

#[derive(Debug)]
struct VulkanHostMemoryReservation {
    tracker: Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
    byte_count: usize,
    recycle_pool: Option<Weak<Mutex<VulkanHostMemoryPermitPool>>>,
}

impl VulkanHostMemoryPermit {
    fn acquire(
        tracker: &Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
        currently_available_bytes: usize,
        byte_count: usize,
    ) -> Result<Self, VulkanError> {
        if byte_count == 0 {
            return Err(VulkanError(
                "host memory permit must not be empty".to_string(),
            ));
        }
        let mut state = tracker
            .lock()
            .map_err(|_| VulkanError("host memory budget tracker was poisoned".to_string()))?;
        let projected_pending = state
            .pending_reservation_bytes
            .checked_add(byte_count)
            .ok_or_else(|| VulkanError("host pending reservation bytes overflowed".to_string()))?;
        // MemAvailable already reflects committed host allocations. Add the
        // tracked NERVE allocation back to establish a stable accounting
        // envelope, then include pending-but-not-yet-allocated transactions.
        let maximum_accounted = currently_available_bytes
            .checked_add(state.tracked_allocation_bytes)
            .unwrap_or(usize::MAX);
        let projected_accounted = state
            .tracked_allocation_bytes
            .checked_add(projected_pending)
            .ok_or_else(|| VulkanError("host accounted allocation bytes overflowed".to_string()))?;
        if projected_accounted > maximum_accounted {
            return Err(VulkanError(format!(
                "host capacity permit of {byte_count} bytes would raise accounted allocations from {} tracked plus {} pending to {projected_accounted} bytes, beyond the current {currently_available_bytes}-byte safe capacity",
                state.tracked_allocation_bytes, state.pending_reservation_bytes,
            )));
        }
        state.pending_reservation_bytes = projected_pending;
        drop(state);
        Ok(Self {
            tracker: Arc::clone(tracker),
            byte_count,
            recycle_pool: None,
        })
    }

    fn take(&mut self, byte_count: usize) -> Result<Self, VulkanError> {
        if byte_count == 0 || byte_count > self.byte_count {
            return Err(VulkanError(format!(
                "host capacity permit holds {} bytes but cannot provide a {byte_count}-byte child",
                self.byte_count,
            )));
        }
        self.byte_count -= byte_count;
        Ok(Self {
            tracker: Arc::clone(&self.tracker),
            byte_count,
            recycle_pool: None,
        })
    }

    fn recycle_into(mut self, pool: Weak<Mutex<VulkanHostMemoryPermitPool>>) -> Self {
        self.recycle_pool = Some(pool);
        self
    }

    fn commit(
        mut self,
        allocation_byte_count: usize,
    ) -> Result<Arc<VulkanHostMemoryReservation>, VulkanError> {
        if allocation_byte_count == 0 || allocation_byte_count > self.byte_count {
            return Err(VulkanError(format!(
                "host allocation needs {allocation_byte_count} bytes but its capacity permit holds {} bytes",
                self.byte_count,
            )));
        }
        if self.recycle_pool.is_some() && allocation_byte_count != self.byte_count {
            return Err(VulkanError(format!(
                "reusable host capacity permit holds {} bytes but the allocation committed {allocation_byte_count}; reusable admissions require exact physical consumption",
                self.byte_count,
            )));
        }
        let mut state = self
            .tracker
            .lock()
            .map_err(|_| VulkanError("host memory budget tracker was poisoned".to_string()))?;
        state.pending_reservation_bytes = state
            .pending_reservation_bytes
            .checked_sub(self.byte_count)
            .ok_or_else(|| VulkanError("host capacity permit was not accounted".to_string()))?;
        state.tracked_allocation_bytes = state
            .tracked_allocation_bytes
            .checked_add(allocation_byte_count)
            .ok_or_else(|| VulkanError("host tracked allocation bytes overflowed".to_string()))?;
        let recycle_pool = self.recycle_pool.take();
        self.byte_count = 0;
        drop(state);
        Ok(Arc::new(VulkanHostMemoryReservation {
            tracker: Arc::clone(&self.tracker),
            byte_count: allocation_byte_count,
            recycle_pool,
        }))
    }

    fn remaining_byte_count(&self) -> usize {
        self.byte_count
    }
}

impl Drop for VulkanHostMemoryPermit {
    fn drop(&mut self) {
        if self.byte_count == 0 {
            return;
        }
        if let Some(pool) = self.recycle_pool.as_ref().and_then(Weak::upgrade)
            && pool.lock().is_ok_and(|mut pool| {
                pool.recycle_pending_permit(&self.tracker, self.byte_count)
            })
        {
            self.byte_count = 0;
            return;
        }
        if let Ok(mut state) = self.tracker.lock() {
            state.pending_reservation_bytes = state
                .pending_reservation_bytes
                .saturating_sub(self.byte_count);
        }
    }
}

impl Drop for VulkanHostMemoryReservation {
    fn drop(&mut self) {
        if let Some(pool) = self.recycle_pool.as_ref().and_then(Weak::upgrade)
            && pool.lock().is_ok_and(|mut pool| {
                pool.recycle_committed_allocation(&self.tracker, self.byte_count)
            })
        {
            self.byte_count = 0;
            return;
        }
        if let Ok(mut state) = self.tracker.lock() {
            state.tracked_allocation_bytes = state
                .tracked_allocation_bytes
                .saturating_sub(self.byte_count);
        }
    }
}

#[derive(Debug)]
struct VulkanDeviceLocalMemoryPermitPool {
    permit: VulkanDeviceLocalMemoryPermit,
}

impl VulkanDeviceLocalMemoryPermitPool {
    fn take(&mut self, byte_count: usize) -> Result<VulkanDeviceLocalMemoryPermit, VulkanError> {
        self.permit.take(u64::try_from(byte_count).map_err(|_| {
            VulkanError("scoped device-local allocation exceeds u64".to_string())
        })?)
    }

    fn remaining_byte_count(&self) -> usize {
        usize::try_from(self.permit.remaining_byte_count()).unwrap_or(usize::MAX)
    }

    fn recycle_pending_permit(
        &mut self,
        tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
        byte_count: u64,
    ) -> bool {
        if !Arc::ptr_eq(&self.permit.tracker, tracker) {
            return false;
        }
        let Some(next) = self.permit.byte_count.checked_add(byte_count) else {
            return false;
        };
        self.permit.byte_count = next;
        true
    }

    fn recycle_committed_allocation(
        &mut self,
        tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
        byte_count: u64,
    ) -> bool {
        if !Arc::ptr_eq(&self.permit.tracker, tracker) {
            return false;
        }
        let Some(next_pool_bytes) = self.permit.byte_count.checked_add(byte_count) else {
            return false;
        };
        let Ok(mut state) = tracker.lock() else {
            return false;
        };
        let Some(next_tracked_bytes) = state.tracked_allocation_bytes.checked_sub(byte_count) else {
            return false;
        };
        let Some(next_pending_bytes) = state.pending_reservation_bytes.checked_add(byte_count) else {
            return false;
        };
        self.permit.byte_count = next_pool_bytes;
        state.tracked_allocation_bytes = next_tracked_bytes;
        state.pending_reservation_bytes = next_pending_bytes;
        state.invalidate_execution_memory_observation();
        true
    }
}

#[derive(Debug)]
struct VulkanHostMemoryPermitPool {
    permit: VulkanHostMemoryPermit,
}

impl VulkanHostMemoryPermitPool {
    fn take(&mut self, byte_count: usize) -> Result<VulkanHostMemoryPermit, VulkanError> {
        self.permit.take(byte_count)
    }

    fn remaining_byte_count(&self) -> usize {
        self.permit.remaining_byte_count()
    }

    fn recycle_pending_permit(
        &mut self,
        tracker: &Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
        byte_count: usize,
    ) -> bool {
        if !Arc::ptr_eq(&self.permit.tracker, tracker) {
            return false;
        }
        let Some(next) = self.permit.byte_count.checked_add(byte_count) else {
            return false;
        };
        self.permit.byte_count = next;
        true
    }

    fn recycle_committed_allocation(
        &mut self,
        tracker: &Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
        byte_count: usize,
    ) -> bool {
        if !Arc::ptr_eq(&self.permit.tracker, tracker) {
            return false;
        }
        let Some(next_pool_bytes) = self.permit.byte_count.checked_add(byte_count) else {
            return false;
        };
        let Ok(mut state) = tracker.lock() else {
            return false;
        };
        let Some(next_tracked_bytes) = state.tracked_allocation_bytes.checked_sub(byte_count) else {
            return false;
        };
        let Some(next_pending_bytes) = state.pending_reservation_bytes.checked_add(byte_count) else {
            return false;
        };
        self.permit.byte_count = next_pool_bytes;
        state.tracked_allocation_bytes = next_tracked_bytes;
        state.pending_reservation_bytes = next_pending_bytes;
        true
    }
}

#[derive(Clone)]
struct VulkanMemoryAdmissionScopeEntry {
    id: u64,
    device_pools: BTreeMap<usize, Arc<Mutex<VulkanDeviceLocalMemoryPermitPool>>>,
    host_pool: Option<(usize, Arc<Mutex<VulkanHostMemoryPermitPool>>)>,
    recycle_released_capacity: bool,
}

thread_local! {
    static VULKAN_MEMORY_ADMISSION_SCOPES: RefCell<Vec<VulkanMemoryAdmissionScopeEntry>> =
        const { RefCell::new(Vec::new()) };
}

static NEXT_VULKAN_MEMORY_ADMISSION_ID: AtomicU64 = AtomicU64::new(1);

/// An all-participant capacity transaction for one runtime stream.
///
/// The transaction is acquired before any stream buffer is created. A scoped
/// construction call consumes exact child permits from it. Permanent children
/// consume their credit once; reusable lazy-runner children return committed
/// capacity to the same transaction when their buffers are released. Unused
/// credit remains reserved for the stream's prompt and verification runners.
#[derive(Debug)]
pub(crate) struct VulkanMemoryAdmission {
    id: u64,
    device_pools: BTreeMap<usize, Arc<Mutex<VulkanDeviceLocalMemoryPermitPool>>>,
    host_pool: Option<(usize, Arc<Mutex<VulkanHostMemoryPermitPool>>)>,
}

pub(crate) struct VulkanMemoryAdmissionScope {
    id: u64,
    _not_send: std::marker::PhantomData<Rc<()>>,
}

impl VulkanMemoryAdmission {
    pub(crate) fn reserve(
        device_requirements: &[(&VulkanComputeDevice, usize)],
        host_requirement: Option<(&VulkanComputeDevice, usize, usize)>,
    ) -> Result<Self, VulkanError> {
        let mut requirements_by_physical_device =
            BTreeMap::<String, (&VulkanComputeDevice, usize, usize)>::new();
        for (device, byte_count) in device_requirements {
            if *byte_count == 0 {
                continue;
            }
            let tracker_key = Arc::as_ptr(&device.device_local_memory_budget_tracker) as usize;
            match requirements_by_physical_device.entry(device.physical_device_id.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((device, *byte_count, tracker_key));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (_, total, existing_tracker_key) = entry.get_mut();
                    if *existing_tracker_key != tracker_key {
                        return Err(VulkanError(format!(
                            "physical device {:?} has multiple memory budget trackers in one stream admission",
                            device.physical_device_id,
                        )));
                    }
                    *total = total.checked_add(*byte_count).ok_or_else(|| {
                        VulkanError("stream device-local admission bytes overflowed".to_string())
                    })?;
                }
            }
        }

        let mut device_pools = BTreeMap::new();
        for (_, (device, byte_count, tracker_key)) in requirements_by_physical_device {
            let permit =
                device.reserve_fixed_device_local_memory_capacity_unscoped(byte_count)?;
            device_pools.insert(
                tracker_key,
                Arc::new(Mutex::new(VulkanDeviceLocalMemoryPermitPool { permit })),
            );
        }

        let host_pool = match host_requirement {
            Some((device, currently_available_bytes, byte_count)) if byte_count > 0 => {
                let tracker = &device.context.host_memory_budget_tracker;
                let tracker_key = Arc::as_ptr(tracker) as usize;
                if device_requirements.iter().any(|(participant, _)| {
                    Arc::as_ptr(&participant.context.host_memory_budget_tracker) as usize
                        != tracker_key
                }) {
                    return Err(VulkanError(
                        "one stream admission spans independent Vulkan host-memory trackers"
                            .to_string(),
                    ));
                }
                let permit = VulkanHostMemoryPermit::acquire(
                    tracker,
                    currently_available_bytes,
                    byte_count,
                )?;
                Some((
                    tracker_key,
                    Arc::new(Mutex::new(VulkanHostMemoryPermitPool { permit })),
                ))
            }
            _ => None,
        };

        Ok(Self {
            id: NEXT_VULKAN_MEMORY_ADMISSION_ID.fetch_add(1, Ordering::Relaxed),
            device_pools,
            host_pool,
        })
    }

    pub(crate) fn enter(&self) -> VulkanMemoryAdmissionScope {
        self.enter_with_recycling(false)
    }

    /// Enters a lazy allocation scope whose committed capacity is returned to
    /// this admission when the corresponding buffers are released. This is
    /// used for cached runners that can be replaced or remounted while the
    /// owning stream remains alive. Permanent mount allocations use `enter`
    /// and consume their credit once.
    pub(crate) fn enter_reusable(&self) -> VulkanMemoryAdmissionScope {
        self.enter_with_recycling(true)
    }

    fn enter_with_recycling(
        &self,
        recycle_released_capacity: bool,
    ) -> VulkanMemoryAdmissionScope {
        VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
            scopes.borrow_mut().push(VulkanMemoryAdmissionScopeEntry {
                id: self.id,
                device_pools: self.device_pools.clone(),
                host_pool: self.host_pool.clone(),
                recycle_released_capacity,
            });
        });
        VulkanMemoryAdmissionScope {
            id: self.id,
            _not_send: std::marker::PhantomData,
        }
    }

    pub(crate) fn ensure_fully_consumed(&self, concern: &str) -> Result<(), VulkanError> {
        if concern.trim().is_empty() {
            return Err(VulkanError(
                "memory admission consumption check requires a concern".to_string(),
            ));
        }
        let mut remaining_device_bytes = 0usize;
        for pool in self.device_pools.values() {
            remaining_device_bytes = remaining_device_bytes
                .checked_add(
                    pool.lock()
                        .map_err(|_| {
                            VulkanError(format!(
                                "{concern} device admission pool is poisoned",
                            ))
                        })?
                        .remaining_byte_count(),
                )
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{concern} remaining device admission bytes overflowed",
                    ))
                })?;
        }
        let remaining_host_bytes = self
            .host_pool
            .as_ref()
            .map(|(_, pool)| {
                pool.lock()
                    .map(|pool| pool.remaining_byte_count())
                    .map_err(|_| VulkanError(format!("{concern} host admission pool is poisoned")))
            })
            .transpose()?
            .unwrap_or_default();
        if remaining_device_bytes == 0 && remaining_host_bytes == 0 {
            return Ok(());
        }
        Err(VulkanError(format!(
            "{concern} left {remaining_device_bytes} device bytes and {remaining_host_bytes} host bytes as unexplained admission credit",
        )))
    }

    #[cfg(test)]
    fn from_test_permits(
        device_permits: Vec<(usize, VulkanDeviceLocalMemoryPermit)>,
        host_permit: Option<(usize, VulkanHostMemoryPermit)>,
    ) -> Self {
        Self {
            id: NEXT_VULKAN_MEMORY_ADMISSION_ID.fetch_add(1, Ordering::Relaxed),
            device_pools: device_permits
                .into_iter()
                .map(|(key, permit)| {
                    (
                        key,
                        Arc::new(Mutex::new(VulkanDeviceLocalMemoryPermitPool { permit })),
                    )
                })
                .collect(),
            host_pool: host_permit.map(|(key, permit)| {
                (
                    key,
                    Arc::new(Mutex::new(VulkanHostMemoryPermitPool { permit })),
                )
            }),
        }
    }

    #[cfg(test)]
    fn remaining_device_bytes(&self, tracker_key: usize) -> usize {
        self.device_pools
            .get(&tracker_key)
            .and_then(|pool| pool.lock().ok())
            .map(|pool| pool.remaining_byte_count())
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn remaining_host_bytes(&self) -> usize {
        self.host_pool
            .as_ref()
            .and_then(|(_, pool)| pool.lock().ok())
            .map(|pool| pool.remaining_byte_count())
            .unwrap_or_default()
    }
}

impl Drop for VulkanMemoryAdmissionScope {
    fn drop(&mut self) {
        VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            if scopes.last().is_some_and(|scope| scope.id == self.id) {
                scopes.pop();
            } else if let Some(index) = scopes.iter().rposition(|scope| scope.id == self.id) {
                scopes.remove(index);
            }
        });
    }
}

fn take_scoped_device_local_memory_capacity(
    tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
    byte_count: usize,
) -> Option<Result<VulkanDeviceLocalMemoryPermit, VulkanError>> {
    let tracker_key = Arc::as_ptr(tracker) as usize;
    let (pool, recycle_released_capacity) = match VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        scopes
            .borrow()
            .last()
            .map(|scope| {
                (
                    scope.device_pools.get(&tracker_key).cloned(),
                    scope.recycle_released_capacity,
                )
            })
    })? {
        (Some(pool), recycle_released_capacity) => (pool, recycle_released_capacity),
        (None, _) => {
            return Some(Err(VulkanError(
                "active stream admission has no capacity permit for this physical device"
                    .to_string(),
            )));
        }
    };
    let child = pool
        .lock()
            .map_err(|_| VulkanError("stream device-local permit pool was poisoned".to_string()))
        .and_then(|mut pool| pool.take(byte_count));
    Some(child.map(|permit| {
        if recycle_released_capacity {
            permit.recycle_into(Arc::downgrade(&pool))
        } else {
            permit
        }
    }))
}

fn take_scoped_host_memory_capacity(
    tracker: &Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
    byte_count: usize,
) -> Option<Result<VulkanHostMemoryPermit, VulkanError>> {
    let tracker_key = Arc::as_ptr(tracker) as usize;
    let (pool, recycle_released_capacity) = match VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        scopes.borrow().last().map(|scope| {
            (
                scope
                    .host_pool
                    .as_ref()
                    .filter(|(key, _)| *key == tracker_key)
                    .map(|(_, pool)| Arc::clone(pool)),
                scope.recycle_released_capacity,
            )
        })
    })? {
        (Some(pool), recycle_released_capacity) => (pool, recycle_released_capacity),
        (None, _) => {
            return Some(Err(VulkanError(
                "active stream admission has no shared-host capacity permit".to_string(),
            )));
        }
    };
    let child = pool
        .lock()
            .map_err(|_| VulkanError("stream host permit pool was poisoned".to_string()))
        .and_then(|mut pool| pool.take(byte_count));
    Some(child.map(|permit| {
        if recycle_released_capacity {
            permit.recycle_into(Arc::downgrade(&pool))
        } else {
            permit
        }
    }))
}
