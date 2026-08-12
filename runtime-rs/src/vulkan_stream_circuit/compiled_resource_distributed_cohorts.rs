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
    group_keys: BTreeMap<
        (String, String),
        BTreeSet<VulkanCompiledResourceDistributedCohortKey>,
    >,
    expected_logical_device_ids: BTreeSet<String>,
    mutation: std::sync::Mutex<()>,
    stores: std::sync::Mutex<
        BTreeMap<String, std::sync::Weak<VulkanCompiledResourceDeviceStore>>,
    >,
}

struct VulkanCompiledResourceDistributedCohortMutation<'a> {
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl VulkanCompiledResourceDistributedCohortCoordinator {
    fn new(
        plan: &VulkanDistributedSelectedResourceStorePlan,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let mut plans = BTreeMap::new();
        let mut group_keys = BTreeMap::<
            (String, String),
            BTreeSet<VulkanCompiledResourceDistributedCohortKey>,
        >::new();
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
                group_keys
                    .entry((
                        member.logical_device_id.clone(),
                        key.atomic_group_id.clone(),
                    ))
                    .or_default()
                    .insert(key.clone());
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
        }
        Ok(Self {
            plans,
            group_keys,
            expected_logical_device_ids,
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
        if registered != required {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "distributed residency cohorts require logical stores {required:?}, registered {registered:?}",
            )));
        }
        Ok(())
    }

    fn protects_group_on_store(
        &self,
        store: &VulkanCompiledResourceDeviceStore,
        group_id: &str,
    ) -> bool {
        store.logical_device_ids().iter().any(|device_id| {
            self.group_keys
                .contains_key(&(device_id.clone(), group_id.to_string()))
        })
    }

    fn cohort_for_selection(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Option<&VulkanCompiledResourceDistributedCohortPlan> {
        self.plans
            .values()
            .find(|plan| {
                plan.key.selector_id == selector_id && plan.key.resource_index == resource_index
            })
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
