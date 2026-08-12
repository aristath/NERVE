#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanCompiledResourceDistributedCohortKey {
    selector_id: String,
    resource_index: usize,
    atomic_group_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceDistributedCohortMember {
    logical_device_id: String,
    logical_start: usize,
    logical_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceDistributedCohortPlan {
    key: VulkanCompiledResourceDistributedCohortKey,
    members: Vec<VulkanCompiledResourceDistributedCohortMember>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceDistributedFaultObservation {
    logical_device_id: String,
    selector_id: String,
    checkpoint_tag: u32,
    pending_resource_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceDistributedFaultLoad {
    observation_index: usize,
    resource_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceDistributedFaultPlan {
    loads: Vec<VulkanCompiledResourceDistributedFaultLoad>,
    commit_observation_indices: Vec<usize>,
}

struct VulkanCompiledResourceDistributedCohortCoordinator {
    plans: BTreeMap<
        VulkanCompiledResourceDistributedCohortKey,
        VulkanCompiledResourceDistributedCohortPlan,
    >,
    selection_keys: BTreeMap<
        (String, usize),
        VulkanCompiledResourceDistributedCohortKey,
    >,
    expected_logical_device_ids: BTreeSet<String>,
    physical_store_counts:
        std::sync::Mutex<BTreeMap<VulkanCompiledResourceDistributedCohortKey, usize>>,
    physical_group_keys: std::sync::Mutex<
        BTreeMap<
            (usize, String),
            BTreeSet<VulkanCompiledResourceDistributedCohortKey>,
        >,
    >,
    overlap_keys: std::sync::Mutex<
        BTreeMap<
            VulkanCompiledResourceDistributedCohortKey,
            BTreeSet<VulkanCompiledResourceDistributedCohortKey>,
        >,
    >,
    mutation: std::sync::Mutex<()>,
    stores: std::sync::Mutex<
        BTreeMap<String, std::sync::Weak<VulkanCompiledResourceDeviceStore>>,
    >,
}

struct VulkanCompiledResourceDistributedCohortMutation<'a> {
    coordinator: &'a VulkanCompiledResourceDistributedCohortCoordinator,
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl VulkanCompiledResourceDistributedCohortCoordinator {
    fn new(
        plan: &VulkanDistributedSelectedResourceStorePlan,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let mut plans = BTreeMap::new();
        let mut selection_keys = BTreeMap::new();
        let expected_logical_device_ids = plan
            .devices
            .iter()
            .map(|device| device.device_id.clone())
            .collect::<BTreeSet<_>>();
        if expected_logical_device_ids.len() != plan.devices.len() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency plan repeats a logical device",
            ));
        }
        for cohort in &plan.tensor_sharded_residency_cohorts {
            if cohort.members.len() < 2 {
                continue;
            }
            let key = VulkanCompiledResourceDistributedCohortKey {
                selector_id: cohort.selector_id.clone(),
                resource_index: cohort.resource_index,
                atomic_group_id: cohort.atomic_group_id.clone(),
            };
            let mut members = cohort
                .members
                .iter()
                .map(|member| VulkanCompiledResourceDistributedCohortMember {
                    logical_device_id: member.device_id.clone(),
                    logical_start: member.logical_start,
                    logical_count: member.logical_count,
                })
                .collect::<Vec<_>>();
            members.sort_by(|left, right| {
                (left.logical_start, left.logical_device_id.as_str())
                    .cmp(&(right.logical_start, right.logical_device_id.as_str()))
            });
            let mut frontier = 0usize;
            let mut member_devices = BTreeSet::new();
            for member in &members {
                if !expected_logical_device_ids.contains(&member.logical_device_id)
                    || !member_devices.insert(member.logical_device_id.as_str())
                    || member.logical_count == 0
                    || member.logical_start != frontier
                {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                        "distributed residency cohort {:?} index {} has invalid physical members",
                        key.selector_id, key.resource_index,
                    )));
                }
                frontier = frontier.checked_add(member.logical_count).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "distributed residency cohort logical extent overflowed",
                    )
                })?;
            }
            let cohort_plan = VulkanCompiledResourceDistributedCohortPlan {
                key: key.clone(),
                members,
            };
            if plans.insert(key.clone(), cohort_plan).is_some() {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency cohort {:?} index {} is duplicated",
                    key.selector_id, key.resource_index,
                )));
            }
            if selection_keys
                .insert(
                    (key.selector_id.clone(), key.resource_index),
                    key.clone(),
                )
                .is_some()
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency selector {:?} index {} belongs to multiple cohorts",
                    key.selector_id, key.resource_index,
                )));
            }
        }
        Ok(Self {
            plans,
            selection_keys,
            expected_logical_device_ids,
            physical_store_counts: std::sync::Mutex::new(BTreeMap::new()),
            physical_group_keys: std::sync::Mutex::new(BTreeMap::new()),
            overlap_keys: std::sync::Mutex::new(BTreeMap::new()),
            mutation: std::sync::Mutex::new(()),
            stores: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    fn is_compatible_with(
        &self,
        plan: &VulkanDistributedSelectedResourceStorePlan,
    ) -> Result<bool, VulkanCompiledResourceDeviceStoreError> {
        let candidate = Self::new(plan)?;
        Ok(self.plans == candidate.plans
            && self.selection_keys == candidate.selection_keys
            && self.expected_logical_device_ids == candidate.expected_logical_device_ids)
    }

    fn begin_mutation(
        &self,
    ) -> Result<
        VulkanCompiledResourceDistributedCohortMutation<'_>,
        VulkanCompiledResourceDeviceStoreError,
    > {
        let guard = self.mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency cohort mutation lock was poisoned",
            )
        })?;
        Ok(VulkanCompiledResourceDistributedCohortMutation {
            coordinator: self,
            _guard: guard,
        })
    }

    fn register_store(
        &self,
        store: &Arc<VulkanCompiledResourceDeviceStore>,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let mut stores = self.stores.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency cohort store registry was poisoned",
            )
        })?;
        for logical_device_id in store.logical_device_ids() {
            if !self.expected_logical_device_ids.contains(logical_device_id) {
                continue;
            }
            match stores.get(logical_device_id).and_then(std::sync::Weak::upgrade) {
                Some(existing) if !Arc::ptr_eq(&existing, store) => {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                        "distributed residency logical device {logical_device_id:?} maps to two physical stores",
                    )));
                }
                Some(_) => {}
                None => {
                    stores.insert(logical_device_id.clone(), Arc::downgrade(store));
                }
            }
        }
        Ok(())
    }

    fn validate_complete_registration(&self) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let stores = self.stores.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency cohort store registry was poisoned",
            )
        })?;
        let registered = stores
            .iter()
            .filter_map(|(device_id, store)| store.upgrade().map(|_| device_id.as_str()))
            .collect::<BTreeSet<_>>();
        let required = self
            .plans
            .values()
            .flat_map(|plan| plan.members.iter())
            .map(|member| member.logical_device_id.as_str())
            .collect::<BTreeSet<_>>();
        if !required.is_subset(&registered) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "distributed residency cohorts require logical stores {required:?}, registered {registered:?}",
            )));
        }
        let mut physical_store_counts = BTreeMap::new();
        let mut physical_group_keys = BTreeMap::<
            (usize, String),
            BTreeSet<VulkanCompiledResourceDistributedCohortKey>,
        >::new();
        for (key, plan) in &self.plans {
            let mut physical_stores = BTreeSet::new();
            for member in &plan.members {
                let store = stores
                    .get(&member.logical_device_id)
                    .and_then(std::sync::Weak::upgrade)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "distributed residency cohort {:?} has no live store for logical device {:?}",
                            key.atomic_group_id, member.logical_device_id,
                        ))
                    })?;
                let store_identity = Arc::as_ptr(&store) as usize;
                physical_stores.insert(store_identity);
                let group_id = store.selector_resource_group_id(
                    &key.selector_id,
                    key.resource_index,
                )?;
                physical_group_keys
                    .entry((store_identity, group_id))
                    .or_default()
                    .insert(key.clone());
            }
            physical_store_counts.insert(key.clone(), physical_stores.len());
        }
        let mut overlap_keys = self
            .plans
            .keys()
            .cloned()
            .map(|key| (key.clone(), BTreeSet::from([key])))
            .collect::<BTreeMap<_, _>>();
        for keys in physical_group_keys.values() {
            for key in keys {
                overlap_keys
                    .entry(key.clone())
                    .or_default()
                    .extend(keys.iter().cloned());
            }
        }
        drop(stores);
        *self.physical_store_counts.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency physical-store catalog was poisoned",
            )
        })? = physical_store_counts;
        *self.physical_group_keys.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency physical-group catalog was poisoned",
            )
        })? = physical_group_keys;
        *self.overlap_keys.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency overlap catalog was poisoned",
            )
        })? = overlap_keys;
        Ok(())
    }

    fn protects_group_on_store(
        &self,
        store: &VulkanCompiledResourceDeviceStore,
        group_id: &str,
    ) -> bool {
        self.physical_group_keys
            .lock()
            .ok()
            .and_then(|groups| {
                groups
                    .get(&(store as *const _ as usize, group_id.to_string()))
                    .cloned()
            })
            .is_some_and(|keys| {
                keys.iter()
                    .any(|key| self.cohort_spans_multiple_physical_stores(key))
            })
    }

    fn cohort_spans_multiple_physical_stores(
        &self,
        key: &VulkanCompiledResourceDistributedCohortKey,
    ) -> bool {
        self.physical_store_counts
            .lock()
            .ok()
            .and_then(|counts| counts.get(key).copied())
            .is_some_and(|count| count > 1)
    }

    fn cohort_key_closure(
        &self,
        start: &VulkanCompiledResourceDistributedCohortKey,
    ) -> Result<BTreeSet<VulkanCompiledResourceDistributedCohortKey>, VulkanCompiledResourceDeviceStoreError>
    {
        let overlaps = self.overlap_keys.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency overlap catalog was poisoned",
            )
        })?;
        let mut closure = BTreeSet::new();
        let mut pending = vec![start.clone()];
        while let Some(key) = pending.pop() {
            if !closure.insert(key.clone()) {
                continue;
            }
            let adjacent = overlaps.get(&key).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency overlap catalog omits cohort {:?}",
                    key.atomic_group_id,
                ))
            })?;
            pending.extend(adjacent.iter().filter(|key| !closure.contains(*key)).cloned());
        }
        Ok(closure)
    }

    fn cohort_for_selection(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Option<&VulkanCompiledResourceDistributedCohortPlan> {
        self.selection_keys
            .get(&(selector_id.to_string(), resource_index))
            .and_then(|key| self.plans.get(key))
    }

    fn plan_fault_resolution(
        &self,
        observations: &[VulkanCompiledResourceDistributedFaultObservation],
    ) -> Result<VulkanCompiledResourceDistributedFaultPlan, VulkanCompiledResourceDeviceStoreError>
    {
        let mut observation_indices = BTreeMap::<(&str, &str, u32), Vec<usize>>::new();
        for (observation_index, observation) in observations.iter().enumerate() {
            observation_indices
                .entry((
                    observation.logical_device_id.as_str(),
                    observation.selector_id.as_str(),
                    observation.checkpoint_tag,
                ))
                .or_default()
                .push(observation_index);
        }
        let mut loads = BTreeMap::<usize, BTreeSet<usize>>::new();
        let mut commits = Vec::new();
        for (observation_index, observation) in observations.iter().enumerate() {
            if observation.pending_resource_indices.is_empty() {
                continue;
            }
            if observation
                .pending_resource_indices
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency fault observation {observation_index} has duplicate or unsorted resources",
                )));
            }
            commits.push(observation_index);
            for resource_index in &observation.pending_resource_indices {
                let Some(cohort) = self.cohort_for_selection(
                    &observation.selector_id,
                    *resource_index,
                ) else {
                    loads
                        .entry(observation_index)
                        .or_default()
                        .insert(*resource_index);
                    continue;
                };
                if !cohort.members.iter().any(|member| {
                    member.logical_device_id == observation.logical_device_id
                }) {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                        "distributed selector {:?} resource {} faulted on logical device {:?} outside its residency cohort",
                        observation.selector_id,
                        resource_index,
                        observation.logical_device_id,
                    )));
                }
                for member in &cohort.members {
                    let key = (
                        member.logical_device_id.as_str(),
                        observation.selector_id.as_str(),
                        observation.checkpoint_tag,
                    );
                    let candidates = observation_indices.get(&key).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "distributed selector {:?} resource {} has no residency gate for cohort member {:?} at checkpoint {}",
                            observation.selector_id,
                            resource_index,
                            member.logical_device_id,
                            observation.checkpoint_tag,
                        ))
                    })?;
                    let [member_observation_index] = candidates.as_slice() else {
                        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                            "distributed selector {:?} resource {} has ambiguous residency gates for cohort member {:?} at checkpoint {}",
                            observation.selector_id,
                            resource_index,
                            member.logical_device_id,
                            observation.checkpoint_tag,
                        )));
                    };
                    loads
                        .entry(*member_observation_index)
                        .or_default()
                        .insert(*resource_index);
                }
            }
        }
        Ok(VulkanCompiledResourceDistributedFaultPlan {
            loads: loads
                .into_iter()
                .map(|(observation_index, resource_indices)| {
                    VulkanCompiledResourceDistributedFaultLoad {
                        observation_index,
                        resource_indices: resource_indices.into_iter().collect(),
                    }
                })
                .collect(),
            commit_observation_indices: commits,
        })
    }

    fn eviction_keys_for_store(
        &self,
        store: &VulkanCompiledResourceDeviceStore,
        candidates: &[DeviceResourceResidencyEvictionCandidate],
    ) -> Vec<VulkanCompiledResourceDistributedCohortKey> {
        self.eviction_keys_for_physical_candidates(store as *const _ as usize, candidates)
    }

    fn eviction_keys_for_physical_candidates(
        &self,
        store_identity: usize,
        candidates: &[DeviceResourceResidencyEvictionCandidate],
    ) -> Vec<VulkanCompiledResourceDistributedCohortKey> {
        let Ok(physical_group_keys) = self.physical_group_keys.lock() else {
            return Vec::new();
        };
        let mut seen = BTreeSet::new();
        candidates
            .iter()
            .flat_map(|candidate| {
                physical_group_keys
                    .get(&(store_identity, candidate.group_id.clone()))
                    .into_iter()
                    .flatten()
            })
            .filter(|key| self.cohort_spans_multiple_physical_stores(key))
            .filter(|key| seen.insert((*key).clone()))
            .cloned()
            .collect()
    }

    fn evict_inactive_cohorts_for_store(
        &self,
        triggering_store: &VulkanCompiledResourceDeviceStore,
        candidates: &[DeviceResourceResidencyEvictionCandidate],
        required_local_bytes: usize,
        mutation: &VulkanCompiledResourceDistributedCohortMutation<'_>,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        if required_local_bytes == 0 {
            return Ok(0);
        }
        if !mutation.belongs_to(self) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency eviction received a foreign cohort mutation",
            ));
        }
        let keys = self.eviction_keys_for_store(triggering_store, candidates);
        let registered = self.stores.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "distributed residency cohort store registry was poisoned",
            )
        })?;
        let stores = registered
            .iter()
            .map(|(logical_device_id, store)| {
                store.upgrade().map(|store| (logical_device_id.clone(), store)).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "distributed residency store {logical_device_id:?} was dropped while its coordinator remains mounted",
                    ))
                })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        drop(registered);
        let mut released_local_bytes = 0usize;
        let mut processed_keys = BTreeSet::new();
        for key in keys {
            if released_local_bytes >= required_local_bytes {
                break;
            }
            if processed_keys.contains(&key) {
                continue;
            }
            let mut physical = BTreeMap::<
                usize,
                (Arc<VulkanCompiledResourceDeviceStore>, BTreeSet<String>),
            >::new();
            let closure_keys = self.cohort_key_closure(&key)?;
            for current_key in &closure_keys {
                let plan = self
                    .plans
                    .get(current_key)
                    .expect("eviction closure key came from the cohort catalog");
                for member in &plan.members {
                    let store = stores.get(&member.logical_device_id).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "distributed residency cohort {:?} has no store for logical device {:?}",
                            current_key.atomic_group_id, member.logical_device_id,
                        ))
                    })?;
                    let group_id = store.selector_resource_group_id(
                        &current_key.selector_id,
                        current_key.resource_index,
                    )?;
                    physical
                        .entry(Arc::as_ptr(store) as usize)
                        .or_insert_with(|| (Arc::clone(store), BTreeSet::new()))
                        .1
                        .insert(group_id.clone());
                }
            }
            processed_keys.extend(closure_keys);
            if !physical.values().any(|(store, _)| {
                std::ptr::eq(store.as_ref(), triggering_store)
            }) {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency eviction cohort {:?} does not include its triggering store",
                    key.atomic_group_id,
                )));
            }
            let physical = physical.into_values().collect::<Vec<_>>();
            let mut other_store_mutations = Vec::new();
            for (store, _) in &physical {
                if std::ptr::eq(store.as_ref(), triggering_store) {
                    continue;
                }
                other_store_mutations.push(store.residency_mutation.lock().map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource residency mutation lock was poisoned",
                    )
                })?);
            }
            let members = physical
                .iter()
                .map(|(store, group_ids)| {
                    DeviceResourceResidencyCohortMember::new(
                        store.manager.clone(),
                        group_ids.clone(),
                    )
                    .map_err(compiled_device_store_residency_error)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let cohort = DeviceResourceResidencyCohort::new(
                format!(
                    "{}:{}:{}",
                    key.selector_id, key.resource_index, key.atomic_group_id,
                ),
                members,
            )
            .map_err(compiled_device_store_residency_error)?;
            let eviction = match cohort.evict_inactive() {
                Ok(eviction) => eviction,
                Err(error) if error.kind() == DeviceResourceResidencyErrorKind::InUse => {
                    continue;
                }
                Err(error) => return Err(compiled_device_store_residency_error(error)),
            };
            for (store, group_ids) in &physical {
                store.retire_cohort_evicted_group_publications(group_ids)?;
                if std::ptr::eq(store.as_ref(), triggering_store) {
                    released_local_bytes = group_ids.iter().try_fold(
                        released_local_bytes,
                        |total, group_id| {
                            total
                                .checked_add(
                                    store.group_payload_byte_count(group_id).ok_or_else(|| {
                                        VulkanCompiledResourceDeviceStoreError::new(format!(
                                            "distributed residency group {group_id:?} has no payload accounting",
                                        ))
                                    })?,
                                )
                                .ok_or_else(|| {
                                    VulkanCompiledResourceDeviceStoreError::new(
                                        "distributed residency eviction byte count overflowed",
                                    )
                                })
                        },
                    )?;
                }
            }
            let expected_group_count = physical
                .iter()
                .map(|(_, group_ids)| group_ids.len())
                .sum::<usize>();
            if eviction.release().group_count != expected_group_count {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "distributed residency cohort eviction released {} of {expected_group_count} physical groups",
                    eviction.release().group_count,
                )));
            }
            drop(eviction);
            drop(other_store_mutations);
        }
        Ok(released_local_bytes)
    }
}

impl VulkanCompiledResourceDistributedCohortMutation<'_> {
    fn belongs_to(&self, coordinator: &VulkanCompiledResourceDistributedCohortCoordinator) -> bool {
        std::ptr::eq(self.coordinator, coordinator)
    }
}

fn attach_distributed_compiled_resource_cohorts(
    plan: &VulkanDistributedSelectedResourceStorePlan,
    stores: &BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
) -> Result<Option<Arc<VulkanCompiledResourceDistributedCohortCoordinator>>, VulkanCompiledResourceDeviceStoreError>
{
    if !plan
        .tensor_sharded_residency_cohorts
        .iter()
        .any(|cohort| cohort.members.len() > 1)
    {
        return Ok(None);
    }
    let mut existing = None::<Arc<VulkanCompiledResourceDistributedCohortCoordinator>>;
    for store in stores.values() {
        let Some(coordinator) = store.distributed_cohort_coordinator()? else {
            continue;
        };
        match &existing {
            Some(current) if !Arc::ptr_eq(current, &coordinator) => {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "retained compiled-resource stores belong to different distributed residency coordinators",
                ));
            }
            Some(_) => {}
            None => existing = Some(coordinator),
        }
    }
    let coordinator = match existing {
        Some(coordinator) => {
            if !coordinator.is_compatible_with(plan)? {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "retained distributed residency coordinator is incompatible with the mounted execution plan",
                ));
            }
            coordinator
        }
        None => Arc::new(VulkanCompiledResourceDistributedCohortCoordinator::new(plan)?),
    };
    let mut unique_stores = BTreeMap::new();
    for store in stores.values() {
        unique_stores
            .entry(Arc::as_ptr(store) as usize)
            .or_insert_with(|| Arc::clone(store));
    }
    for store in unique_stores.into_values() {
        coordinator.register_store(&store)?;
        store.attach_distributed_cohort_coordinator(Arc::clone(&coordinator))?;
    }
    coordinator.validate_complete_registration()?;
    Ok(Some(coordinator))
}
