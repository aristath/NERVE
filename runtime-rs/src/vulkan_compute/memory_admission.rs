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
        let maximum_accounted =
            currently_available_bytes.saturating_add(state.tracked_allocation_bytes);
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
    scope_id: u64,
    allocation_class: VulkanMemoryAdmissionAllocationClass,
    device_pools: BTreeMap<
        (usize, VulkanMemoryAdmissionAllocationClass),
        Arc<Mutex<VulkanDeviceLocalMemoryPermitPool>>,
    >,
    host_pools: BTreeMap<
        VulkanMemoryAdmissionAllocationClass,
        (usize, Arc<Mutex<VulkanHostMemoryPermitPool>>),
    >,
    recycle_released_capacity: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VulkanMemoryAdmissionAllocationClass {
    Permanent,
    PromptRunner,
    VerificationRunner,
    CatchUpRunner,
}

impl VulkanMemoryAdmissionAllocationClass {
    pub(crate) const ALL: [Self; 4] = [
        Self::Permanent,
        Self::PromptRunner,
        Self::VerificationRunner,
        Self::CatchUpRunner,
    ];
}

thread_local! {
    static VULKAN_MEMORY_ADMISSION_SCOPES: RefCell<Vec<VulkanMemoryAdmissionScopeEntry>> =
        const { RefCell::new(Vec::new()) };
}

static NEXT_VULKAN_MEMORY_ADMISSION_SCOPE_ID: AtomicU64 = AtomicU64::new(1);

/// An all-participant capacity transaction for one runtime stream.
///
/// The transaction is acquired before any stream buffer is created. A scoped
/// construction call consumes exact child permits from it. Permanent children
/// consume their credit once; reusable lazy-runner children return committed
/// capacity to the same transaction when their buffers are released. Unused
/// credit remains reserved for the stream's prompt, verification, and catch-up
/// runners without allowing one runner class to consume another's capacity.
#[derive(Debug)]
pub(crate) struct VulkanMemoryAdmission {
    device_pools: BTreeMap<
        (usize, VulkanMemoryAdmissionAllocationClass),
        Arc<Mutex<VulkanDeviceLocalMemoryPermitPool>>,
    >,
    host_pools: BTreeMap<
        VulkanMemoryAdmissionAllocationClass,
        (usize, Arc<Mutex<VulkanHostMemoryPermitPool>>),
    >,
    device_labels_by_tracker_key: BTreeMap<usize, String>,
}

pub(crate) struct VulkanMemoryAdmissionScope {
    scope_id: u64,
    _not_send: std::marker::PhantomData<Rc<()>>,
}

impl VulkanMemoryAdmission {
    pub(crate) fn reserve(
        device_requirements: &[(&VulkanComputeDevice, usize)],
        host_requirement: Option<(&VulkanComputeDevice, usize, usize)>,
    ) -> Result<Self, VulkanError> {
        let classified_device_requirements = device_requirements
            .iter()
            .map(|(device, bytes)| {
                (
                    VulkanMemoryAdmissionAllocationClass::Permanent,
                    *device,
                    *bytes,
                )
            })
            .collect::<Vec<_>>();
        let classified_host_requirements = host_requirement
            .map(|(device, available, bytes)| {
                vec![(
                    VulkanMemoryAdmissionAllocationClass::Permanent,
                    device,
                    available,
                    bytes,
                )]
            })
            .unwrap_or_default();
        Self::reserve_classified(
            &classified_device_requirements,
            &classified_host_requirements,
        )
    }

    pub(crate) fn reserve_classified(
        device_requirements: &[(
            VulkanMemoryAdmissionAllocationClass,
            &VulkanComputeDevice,
            usize,
        )],
        host_requirements: &[(
            VulkanMemoryAdmissionAllocationClass,
            &VulkanComputeDevice,
            usize,
            usize,
        )],
    ) -> Result<Self, VulkanError> {
        let mut requirements_by_physical_device = BTreeMap::<
            String,
            (
                &VulkanComputeDevice,
                usize,
                BTreeMap<VulkanMemoryAdmissionAllocationClass, usize>,
            ),
        >::new();
        for (allocation_class, device, byte_count) in device_requirements {
            if *byte_count == 0 {
                continue;
            }
            let tracker_key = Arc::as_ptr(&device.device_local_memory_budget_tracker) as usize;
            let entry = requirements_by_physical_device
                .entry(device.physical_device_id.clone())
                .or_insert((device, tracker_key, BTreeMap::new()));
            if entry.1 != tracker_key {
                return Err(VulkanError(format!(
                    "physical device {:?} has multiple memory budget trackers in one stream admission",
                    device.physical_device_id,
                )));
            }
            let class_total = entry.2.entry(*allocation_class).or_default();
            *class_total = class_total.checked_add(*byte_count).ok_or_else(|| {
                VulkanError(format!(
                    "{allocation_class:?} stream device-local admission bytes overflowed",
                ))
            })?;
        }

        let mut device_pools = BTreeMap::new();
        let mut device_labels_by_tracker_key = BTreeMap::new();
        for (_, (device, tracker_key, class_requirements)) in requirements_by_physical_device
        {
            device_labels_by_tracker_key
                .insert(tracker_key, device.physical_device_id().to_string());
            let total_bytes = class_requirements.values().try_fold(0usize, |total, bytes| {
                total.checked_add(*bytes).ok_or_else(|| {
                    VulkanError("classified stream device admission overflowed".to_string())
                })
            })?;
            let mut aggregate =
                device.reserve_fixed_device_local_memory_capacity_unscoped(total_bytes)?;
            for (allocation_class, byte_count) in class_requirements {
                let permit = aggregate.take(u64::try_from(byte_count).map_err(|_| {
                    VulkanError("classified device admission exceeds u64".to_string())
                })?)?;
                device_pools.insert(
                    (tracker_key, allocation_class),
                    Arc::new(Mutex::new(VulkanDeviceLocalMemoryPermitPool { permit })),
                );
            }
            debug_assert_eq!(aggregate.remaining_byte_count(), 0);
        }

        let mut host_pools = BTreeMap::new();
        let nonempty_host_requirements = host_requirements
            .iter()
            .filter(|(_, _, _, byte_count)| *byte_count > 0)
            .copied()
            .collect::<Vec<_>>();
        if let Some((_, representative, currently_available_bytes, _)) =
            nonempty_host_requirements.first().copied()
        {
            let tracker = &representative.context.host_memory_budget_tracker;
            let tracker_key = Arc::as_ptr(tracker) as usize;
            if device_requirements.iter().any(|(_, participant, _)| {
                    Arc::as_ptr(&participant.context.host_memory_budget_tracker) as usize
                        != tracker_key
                })
                || nonempty_host_requirements.iter().any(
                    |(_, device, available_bytes, _)| {
                        Arc::as_ptr(&device.context.host_memory_budget_tracker) as usize
                            != tracker_key
                            || *available_bytes != currently_available_bytes
                    },
                )
            {
                return Err(VulkanError(
                    "one classified stream admission spans independent host-memory trackers or capacity snapshots"
                        .to_string(),
                ));
            }
            let mut class_requirements =
                BTreeMap::<VulkanMemoryAdmissionAllocationClass, usize>::new();
            for (allocation_class, _, _, byte_count) in nonempty_host_requirements {
                let class_total = class_requirements.entry(allocation_class).or_default();
                *class_total = class_total.checked_add(byte_count).ok_or_else(|| {
                    VulkanError(format!(
                        "{allocation_class:?} stream host admission bytes overflowed",
                    ))
                })?;
            }
            let total_bytes = class_requirements.values().try_fold(0usize, |total, bytes| {
                total.checked_add(*bytes).ok_or_else(|| {
                    VulkanError("classified stream host admission overflowed".to_string())
                })
            })?;
            let mut aggregate = VulkanHostMemoryPermit::acquire(
                tracker,
                currently_available_bytes,
                total_bytes,
            )?;
            for (allocation_class, byte_count) in class_requirements {
                let permit = aggregate.take(byte_count)?;
                host_pools.insert(
                    allocation_class,
                    (
                        tracker_key,
                        Arc::new(Mutex::new(VulkanHostMemoryPermitPool { permit })),
                    ),
                );
            }
            debug_assert_eq!(aggregate.remaining_byte_count(), 0);
        }

        Ok(Self {
            device_pools,
            host_pools,
            device_labels_by_tracker_key,
        })
    }

    pub(crate) fn enter(&self) -> VulkanMemoryAdmissionScope {
        self.enter_class(VulkanMemoryAdmissionAllocationClass::Permanent, false)
    }

    /// Enters a lazy allocation scope whose committed capacity is returned to
    /// this admission when the corresponding buffers are released. This is
    /// used for cached runners that can be replaced or remounted while the
    /// owning stream remains alive. Permanent mount allocations use `enter`
    /// and consume their credit once.
    pub(crate) fn enter_prompt_runner(&self) -> VulkanMemoryAdmissionScope {
        self.enter_class(VulkanMemoryAdmissionAllocationClass::PromptRunner, true)
    }

    pub(crate) fn enter_verification_runner(&self) -> VulkanMemoryAdmissionScope {
        self.enter_class(
            VulkanMemoryAdmissionAllocationClass::VerificationRunner,
            true,
        )
    }

    pub(crate) fn enter_catch_up_runner(&self) -> VulkanMemoryAdmissionScope {
        self.enter_class(VulkanMemoryAdmissionAllocationClass::CatchUpRunner, true)
    }

    fn enter_class(
        &self,
        allocation_class: VulkanMemoryAdmissionAllocationClass,
        recycle_released_capacity: bool,
    ) -> VulkanMemoryAdmissionScope {
        let scope_id =
            NEXT_VULKAN_MEMORY_ADMISSION_SCOPE_ID.fetch_add(1, Ordering::Relaxed);
        VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
            scopes.borrow_mut().push(VulkanMemoryAdmissionScopeEntry {
                scope_id,
                allocation_class,
                device_pools: self.device_pools.clone(),
                host_pools: self.host_pools.clone(),
                recycle_released_capacity,
            });
        });
        VulkanMemoryAdmissionScope {
            scope_id,
            _not_send: std::marker::PhantomData,
        }
    }

    pub(crate) fn ensure_fully_consumed(&self, concern: &str) -> Result<(), VulkanError> {
        let (
            remaining_device_bytes,
            remaining_host_bytes,
            remaining_device_pools,
        ) =
            self.remaining_bytes_for_class(None, concern)?;
        Self::reject_remaining_credit(
            concern,
            remaining_device_bytes,
            remaining_host_bytes,
            &remaining_device_pools,
        )
    }

    pub(crate) fn ensure_class_fully_consumed(
        &self,
        allocation_class: VulkanMemoryAdmissionAllocationClass,
        concern: &str,
    ) -> Result<(), VulkanError> {
        let (
            remaining_device_bytes,
            remaining_host_bytes,
            remaining_device_pools,
        ) =
            self.remaining_bytes_for_class(Some(allocation_class), concern)?;
        Self::reject_remaining_credit(
            concern,
            remaining_device_bytes,
            remaining_host_bytes,
            &remaining_device_pools,
        )
    }

    fn remaining_bytes_for_class(
        &self,
        allocation_class: Option<VulkanMemoryAdmissionAllocationClass>,
        concern: &str,
    ) -> Result<
        (
            usize,
            usize,
            Vec<(String, VulkanMemoryAdmissionAllocationClass, usize)>,
        ),
        VulkanError,
    > {
        if concern.trim().is_empty() {
            return Err(VulkanError(
                "memory admission consumption check requires a concern".to_string(),
            ));
        }
        let mut remaining_device_bytes = 0usize;
        let mut remaining_device_pools = Vec::new();
        for ((tracker_key, pool_class), pool) in &self.device_pools {
            if allocation_class.is_some_and(|expected| expected != *pool_class) {
                continue;
            }
            let remaining = pool
                .lock()
                .map_err(|_| {
                    VulkanError(format!("{concern} device admission pool is poisoned",))
                })?
                .remaining_byte_count();
            remaining_device_bytes = remaining_device_bytes
                .checked_add(remaining)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{concern} remaining device admission bytes overflowed",
                    ))
                })?;
            if remaining > 0 {
                remaining_device_pools.push((
                    self.device_labels_by_tracker_key
                        .get(tracker_key)
                        .cloned()
                        .unwrap_or_else(|| format!("tracker:{tracker_key}")),
                    *pool_class,
                    remaining,
                ));
            }
        }
        let mut remaining_host_bytes = 0usize;
        for (pool_class, (_, pool)) in &self.host_pools {
            if allocation_class.is_some_and(|expected| expected != *pool_class) {
                continue;
            }
            let pool = pool.lock().map_err(|_| {
                VulkanError(format!("{concern} host admission pool is poisoned"))
            })?;
            remaining_host_bytes = remaining_host_bytes
                .checked_add(pool.remaining_byte_count())
                .ok_or_else(|| {
                    VulkanError(format!(
                        "{concern} remaining host admission bytes overflowed",
                    ))
                })?;
        }
        Ok((
            remaining_device_bytes,
            remaining_host_bytes,
            remaining_device_pools,
        ))
    }

    fn reject_remaining_credit(
        concern: &str,
        remaining_device_bytes: usize,
        remaining_host_bytes: usize,
        remaining_device_pools: &[(String, VulkanMemoryAdmissionAllocationClass, usize)],
    ) -> Result<(), VulkanError> {
        if remaining_device_bytes == 0 && remaining_host_bytes == 0 {
            return Ok(());
        }
        Err(VulkanError(format!(
            "{concern} left {remaining_device_bytes} device bytes and {remaining_host_bytes} host bytes as unexplained admission credit; device_pools={remaining_device_pools:?}",
        )))
    }

    #[cfg(test)]
    fn from_test_permits(
        device_permits: Vec<(usize, VulkanDeviceLocalMemoryPermit)>,
        host_permit: Option<(usize, VulkanHostMemoryPermit)>,
    ) -> Self {
        Self::from_test_partitioned_permits(
            device_permits
                .into_iter()
                .map(|(key, permit)| {
                    (
                        key,
                        VulkanMemoryAdmissionAllocationClass::Permanent,
                        permit,
                    )
                })
                .collect(),
            host_permit
                .map(|(key, permit)| {
                    vec![(
                        VulkanMemoryAdmissionAllocationClass::Permanent,
                        key,
                        permit,
                    )]
                })
                .unwrap_or_default(),
        )
    }

    #[cfg(test)]
    fn from_test_partitioned_permits(
        device_permits: Vec<(
            usize,
            VulkanMemoryAdmissionAllocationClass,
            VulkanDeviceLocalMemoryPermit,
        )>,
        host_permits: Vec<(
            VulkanMemoryAdmissionAllocationClass,
            usize,
            VulkanHostMemoryPermit,
        )>,
    ) -> Self {
        let device_labels_by_tracker_key = device_permits
            .iter()
            .map(|(key, _, _)| (*key, format!("test-tracker:{key}")))
            .collect();
        Self {
            device_pools: device_permits
                .into_iter()
                .map(|(key, allocation_class, permit)| {
                    (
                        (key, allocation_class),
                        Arc::new(Mutex::new(VulkanDeviceLocalMemoryPermitPool { permit })),
                    )
                })
                .collect(),
            host_pools: host_permits
                .into_iter()
                .map(|(allocation_class, key, permit)| {
                    (
                        allocation_class,
                        (
                            key,
                            Arc::new(Mutex::new(VulkanHostMemoryPermitPool { permit })),
                        ),
                    )
                })
                .collect(),
            device_labels_by_tracker_key,
        }
    }

    #[cfg(test)]
    fn remaining_device_bytes(&self, tracker_key: usize) -> usize {
        self.device_pools
            .iter()
            .filter(|((key, _), _)| *key == tracker_key)
            .filter_map(|(_, pool)| pool.lock().ok())
            .map(|pool| pool.remaining_byte_count())
            .sum()
    }

    #[cfg(test)]
    fn remaining_device_bytes_for_class(
        &self,
        tracker_key: usize,
        allocation_class: VulkanMemoryAdmissionAllocationClass,
    ) -> usize {
        self.device_pools
            .get(&(tracker_key, allocation_class))
            .and_then(|pool| pool.lock().ok())
            .map(|pool| pool.remaining_byte_count())
            .unwrap_or_default()
    }

    #[cfg(test)]
    fn remaining_host_bytes(&self) -> usize {
        self.host_pools
            .values()
            .filter_map(|(_, pool)| pool.lock().ok())
            .map(|pool| pool.remaining_byte_count())
            .sum()
    }
}

/// Borrows the active allocation class for short-lived operation scratch.
///
/// The child allocation still consumes the exact physical permit selected by
/// the enclosing stream transaction. Releasing it returns that permit to the
/// same class, so construction-time upload/readback buffers cannot silently
/// consume credit reserved for resident buffers mounted afterward.
fn enter_recyclable_vulkan_memory_admission_subscope(
) -> Option<VulkanMemoryAdmissionScope> {
    VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        let mut scopes = scopes.borrow_mut();
        let mut entry = scopes.last()?.clone();
        entry.scope_id = NEXT_VULKAN_MEMORY_ADMISSION_SCOPE_ID.fetch_add(1, Ordering::Relaxed);
        entry.recycle_released_capacity = true;
        let scope_id = entry.scope_id;
        scopes.push(entry);
        Some(VulkanMemoryAdmissionScope {
            scope_id,
            _not_send: std::marker::PhantomData,
        })
    })
}

fn scoped_device_local_memory_capacity_remaining(
    tracker: &Arc<Mutex<VulkanDeviceLocalMemoryBudgetTracker>>,
) -> Option<Result<usize, VulkanError>> {
    let tracker_key = Arc::as_ptr(tracker) as usize;
    VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        let scopes = scopes.borrow();
        let scope = scopes.last()?;
        Some(
            scope
                .device_pools
                .get(&(tracker_key, scope.allocation_class))
                .ok_or_else(|| {
                    VulkanError(format!(
                        "active stream admission class {:?} has no capacity permit for this physical device",
                        scope.allocation_class
                    ))
                })
                .and_then(|pool| {
                    pool.lock()
                        .map_err(|_| {
                            VulkanError(
                                "stream device-local permit pool was poisoned".to_string(),
                            )
                        })
                        .map(|pool| pool.remaining_byte_count())
                }),
        )
    })
}

fn scoped_host_memory_capacity_remaining(
    tracker: &Arc<Mutex<VulkanHostMemoryBudgetTracker>>,
) -> Option<Result<usize, VulkanError>> {
    let tracker_key = Arc::as_ptr(tracker) as usize;
    VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        let scopes = scopes.borrow();
        let scope = scopes.last()?;
        Some(
            scope
                .host_pools
                .get(&scope.allocation_class)
                .filter(|(key, _)| *key == tracker_key)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "active stream admission class {:?} has no shared-host capacity permit",
                        scope.allocation_class
                    ))
                })
                .and_then(|(_, pool)| {
                    pool.lock()
                        .map_err(|_| {
                            VulkanError("stream host permit pool was poisoned".to_string())
                        })
                        .map(|pool| pool.remaining_byte_count())
                }),
        )
    })
}

impl Drop for VulkanMemoryAdmissionScope {
    fn drop(&mut self) {
        VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            if scopes
                .last()
                .is_some_and(|scope| scope.scope_id == self.scope_id)
            {
                scopes.pop();
            } else if let Some(index) = scopes
                .iter()
                .rposition(|scope| scope.scope_id == self.scope_id)
            {
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
                    scope
                        .device_pools
                        .get(&(tracker_key, scope.allocation_class))
                        .cloned(),
                    scope.allocation_class,
                    scope.recycle_released_capacity,
                )
            })
    })? {
        (Some(pool), _, recycle_released_capacity) => (pool, recycle_released_capacity),
        (None, allocation_class, _) => {
            return Some(Err(VulkanError(
                format!(
                    "active stream admission class {allocation_class:?} has no capacity permit for this physical device",
                ),
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
    let (pool, allocation_class, recycle_released_capacity) = match VULKAN_MEMORY_ADMISSION_SCOPES.with(|scopes| {
        scopes.borrow().last().map(|scope| {
            (
                scope
                    .host_pools
                    .get(&scope.allocation_class)
                    .filter(|(key, _)| *key == tracker_key)
                    .map(|(_, pool)| Arc::clone(pool)),
                scope.allocation_class,
                scope.recycle_released_capacity,
            )
        })
    })? {
        (Some(pool), allocation_class, recycle_released_capacity) => {
            (pool, allocation_class, recycle_released_capacity)
        }
        (None, allocation_class, _) => {
            return Some(Err(VulkanError(
                format!(
                    "active stream admission class {allocation_class:?} has no shared-host capacity permit",
                ),
            )));
        }
    };
    let child = pool
        .lock()
        .map_err(|_| VulkanError("stream host permit pool was poisoned".to_string()))
        .and_then(|mut pool| {
            pool.take(byte_count).map_err(|error| {
                VulkanError(format!(
                    "{allocation_class:?} stream host admission cannot provide {byte_count} bytes: {error}",
                ))
            })
        });
    Some(child.map(|permit| {
        if recycle_released_capacity {
            permit.recycle_into(Arc::downgrade(&pool))
        } else {
            permit
        }
    }))
}
