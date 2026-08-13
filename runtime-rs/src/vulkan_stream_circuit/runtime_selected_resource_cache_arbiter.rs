#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanSelectedResourceCacheDemand {
    weighted_bytes: u128,
    observed_resource_indices: BTreeSet<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanSelectedResourceStreamCacheDemand {
    by_store_selector:
        BTreeMap<String, BTreeMap<String, VulkanSelectedResourceCacheDemand>>,
    concurrent_selectors:
        BTreeMap<String, VulkanSelectedResourceConcurrentSelectorState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanSelectedResourceConcurrentSelectorState {
    telemetry: VulkanSelectionTelemetryDomainSnapshot,
    assignments: Vec<VulkanSelectedResourceAssignment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanSelectedResourceConcurrentPeerStream {
    stream_id: String,
    selectors: BTreeMap<String, VulkanSelectedResourceConcurrentSelectorState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VulkanSelectedResourceConcurrentPeerSnapshot {
    generation: u64,
    streams: Vec<VulkanSelectedResourceConcurrentPeerStream>,
    selector_capacity_bytes_by_logical_device:
        BTreeMap<String, BTreeMap<String, usize>>,
}

struct VulkanSelectedResourceCacheStore {
    store: std::sync::Weak<VulkanCompiledResourceDeviceStore>,
    logical_device_ids: BTreeSet<String>,
    full_baseline_budgets: BTreeMap<String, usize>,
    adaptive_capacity_bytes: usize,
    adaptive_baseline_budgets: BTreeMap<String, usize>,
    resource_payload_bytes: BTreeMap<String, Vec<usize>>,
}

#[derive(Default)]
struct VulkanSelectedResourceCacheArbiterState {
    generation: u64,
    stream_demands: BTreeMap<u64, VulkanSelectedResourceStreamCacheDemand>,
}

struct VulkanSelectedResourceCacheArbiter {
    next_stream_id: std::sync::atomic::AtomicU64,
    stores: BTreeMap<String, VulkanSelectedResourceCacheStore>,
    adaptation_transaction: std::sync::Mutex<()>,
    state: std::sync::Mutex<VulkanSelectedResourceCacheArbiterState>,
}

struct VulkanSelectedResourceCacheRegistration {
    stream_id: u64,
    arbiter: Arc<VulkanSelectedResourceCacheArbiter>,
}

impl VulkanSelectedResourceCacheArbiter {
    fn new(
        stores: &BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
        adaptive_selector_ids: &BTreeSet<String>,
    ) -> Result<Arc<Self>, VulkanResidentTokenModelPackageError> {
        if adaptive_selector_ids.is_empty() {
            return Err(selected_resource_cache_error(
                "selected-resource cache arbiter has no adaptive selectors",
            ));
        }
        let mut unique = BTreeMap::<String, Arc<VulkanCompiledResourceDeviceStore>>::new();
        for store in stores.values() {
            unique
                .entry(store.device_id().to_string())
                .or_insert_with(|| Arc::clone(store));
        }
        let stores = unique
            .into_iter()
            .map(|(store_id, store)| {
                let baseline_budgets = store
                    .selector_payload_budget_snapshot()
                    .map_err(|error| selected_resource_cache_error(error.to_string()))?;
                let adaptive_baseline_budgets = baseline_budgets
                    .iter()
                    .filter(|(selector_id, _)| adaptive_selector_ids.contains(*selector_id))
                    .map(|(selector_id, budget)| (selector_id.clone(), *budget))
                    .collect::<BTreeMap<_, _>>();
                if adaptive_baseline_budgets.is_empty() {
                    return Ok(None);
                }
                let adaptive_selector_ids = adaptive_baseline_budgets
                    .keys()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                let adaptive_resource_payload_bytes = store
                    .selector_cache_resource_payload_bytes(&adaptive_selector_ids)
                    .map_err(|error| selected_resource_cache_error(error.to_string()))?;
                if adaptive_baseline_budgets
                    .keys()
                    .ne(adaptive_resource_payload_bytes.keys())
                {
                    return Err(selected_resource_cache_error(format!(
                        "compiled resource store {store_id:?} adaptive cache selectors differ from their resource payload catalog",
                    )));
                }
                let adaptive_capacity_bytes = adaptive_baseline_budgets.values().try_fold(
                    0usize,
                    |total, budget| {
                        total.checked_add(*budget).ok_or_else(|| {
                            selected_resource_cache_error(
                                "selected-resource adaptive cache capacity overflowed",
                            )
                        })
                    },
                )?;
                if adaptive_capacity_bytes == 0 {
                    return Err(selected_resource_cache_error(format!(
                        "compiled resource store {store_id:?} gives its adaptive selectors no cache capacity",
                    )));
                }
                Ok(Some((
                    store_id,
                    VulkanSelectedResourceCacheStore {
                        full_baseline_budgets: baseline_budgets,
                        adaptive_capacity_bytes,
                        adaptive_baseline_budgets,
                        resource_payload_bytes: adaptive_resource_payload_bytes,
                        logical_device_ids: store.logical_device_ids().iter().cloned().collect(),
                        store: Arc::downgrade(&store),
                    },
                )))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<BTreeMap<_, _>>();
        if stores.is_empty() {
            return Err(selected_resource_cache_error(
                "selected-resource cache arbiter selectors have no physical stores",
            ));
        }
        Ok(Arc::new(Self {
            next_stream_id: std::sync::atomic::AtomicU64::new(1),
            stores,
            adaptation_transaction: std::sync::Mutex::new(()),
            state: std::sync::Mutex::new(VulkanSelectedResourceCacheArbiterState::default()),
        }))
    }

    fn register(
        self: &Arc<Self>,
    ) -> Result<VulkanSelectedResourceCacheRegistration, VulkanResidentTokenModelPackageError> {
        let _transaction = self.adaptation_transaction.lock().map_err(|_| {
            selected_resource_cache_error(
                "selected-resource adaptation transaction was poisoned",
            )
        })?;
        let stream_id = self
            .next_stream_id
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |current| current.checked_add(1),
            )
            .map_err(|_| selected_resource_cache_error("selected-resource stream ID overflowed"))?;
        let mut state = self.state.lock().map_err(|_| {
            selected_resource_cache_error("selected-resource cache arbiter was poisoned")
        })?;
        let next_generation = state.generation.checked_add(1).ok_or_else(|| {
            selected_resource_cache_error("selected-resource cache generation overflowed")
        })?;
        if state
            .stream_demands
            .insert(stream_id, VulkanSelectedResourceStreamCacheDemand::default())
            .is_some()
        {
            return Err(selected_resource_cache_error(
                "selected-resource cache arbiter repeated a stream ID",
            ));
        }
        state.generation = next_generation;
        Ok(VulkanSelectedResourceCacheRegistration {
            stream_id,
            arbiter: Arc::clone(self),
        })
    }

    fn stream_demand(
        &self,
        execution_plan: &VulkanDistributedExecutionPlan,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<VulkanSelectedResourceStreamCacheDemand, VulkanResidentTokenModelPackageError> {
        let placements = selected_resource_placements_from_execution_plan(execution_plan)
            .map_err(|error| selected_resource_cache_error(error.to_string()))?;
        let mut demand = VulkanSelectedResourceStreamCacheDemand::default();
        for placement in placements {
            let partitions = execution_plan
                .dispatches
                .iter()
                .flat_map(|dispatch| {
                    dispatch
                        .selected_resource_partitions
                        .iter()
                        .filter(|partition| partition.selector_id == placement.selector_id)
                        .map(move |partition| (dispatch, partition))
                })
                .collect::<Vec<_>>();
            let Some((dispatch, partition)) = partitions.first().copied() else {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource cache selector {:?} has no executable partition",
                    placement.selector_id,
                )));
            };
            let matching = telemetry
                .domains
                .iter()
                .filter(|domain| {
                    domain.execution_scope == partition.execution_scope
                        && domain.component_id == dispatch.component_id
                        && domain.node_id == partition.node_id
                        && domain.domain_id == partition.domain_id
                })
                .collect::<Vec<_>>();
            let [domain] = matching.as_slice() else {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource cache selector {:?} has {} exact telemetry domains; expected one",
                    placement.selector_id,
                    matching.len(),
                )));
            };
            if domain.resource_count != placement.assignments.len()
                || domain.selection_counts.len() != domain.resource_count
            {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource cache selector {:?} telemetry changes its resource domain",
                    placement.selector_id,
                )));
            }
            if demand
                .concurrent_selectors
                .insert(
                    placement.selector_id.clone(),
                    VulkanSelectedResourceConcurrentSelectorState {
                        telemetry: (*domain).clone(),
                        assignments: placement.assignments.clone(),
                    },
                )
                .is_some()
            {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource cache repeats concurrent selector {:?}",
                    placement.selector_id,
                )));
            }
            for assignment in &placement.assignments {
                let matching_stores = self
                    .stores
                    .iter()
                    .filter(|(_, store)| {
                        store.logical_device_ids.contains(&assignment.device_id)
                            && store
                                .resource_payload_bytes
                                .contains_key(&placement.selector_id)
                    })
                    .collect::<Vec<_>>();
                let [(store_id, store)] = matching_stores.as_slice() else {
                    return Err(selected_resource_cache_error(format!(
                        "selected-resource cache assignment for selector {:?} on {:?} resolves {} stores; expected one",
                        placement.selector_id,
                        assignment.device_id,
                        matching_stores.len(),
                    )));
                };
                let payload_bytes = store.resource_payload_bytes[&placement.selector_id]
                    .get(assignment.resource_index)
                    .copied()
                    .ok_or_else(|| {
                        selected_resource_cache_error(format!(
                            "selected-resource cache assignment for selector {:?} exceeds the store resource catalog",
                            placement.selector_id,
                        ))
                    })?;
                let selection_count = domain.selection_counts[assignment.resource_index];
                if selection_count == 0 || payload_bytes == 0 {
                    continue;
                }
                let selector_demand = demand
                    .by_store_selector
                    .entry((*store_id).clone())
                    .or_default()
                    .entry(placement.selector_id.clone())
                    .or_default();
                selector_demand.weighted_bytes = selector_demand
                    .weighted_bytes
                    .checked_add(u128::from(selection_count) * (payload_bytes as u128))
                    .ok_or_else(|| {
                        selected_resource_cache_error(
                            "selected-resource cache weighted demand overflowed",
                        )
                    })?;
                selector_demand
                    .observed_resource_indices
                    .insert(assignment.resource_index);
            }
        }
        Ok(demand)
    }

    fn peer_snapshot(
        &self,
        stream_id: u64,
    ) -> Result<VulkanSelectedResourceConcurrentPeerSnapshot, VulkanResidentTokenModelPackageError>
    {
        let state = self.state.lock().map_err(|_| {
            selected_resource_cache_error("selected-resource cache arbiter was poisoned")
        })?;
        if !state.stream_demands.contains_key(&stream_id) {
            return Err(selected_resource_cache_error(format!(
                "selected-resource cache arbiter has no stream {stream_id}",
            )));
        }
        let mut selector_capacity_bytes_by_logical_device = BTreeMap::<
            String,
            BTreeMap<String, usize>,
        >::new();
        for (store_id, store) in &self.stores {
            let mounted = store.store.upgrade().ok_or_else(|| {
                selected_resource_cache_error(format!(
                    "selected-resource cache store {store_id:?} was unloaded",
                ))
            })?;
            let budgets = mounted
                .selector_payload_budget_snapshot()
                .map_err(|error| selected_resource_cache_error(error.to_string()))?;
            for selector_id in store.adaptive_baseline_budgets.keys() {
                let budget = budgets.get(selector_id).copied().ok_or_else(|| {
                    selected_resource_cache_error(format!(
                        "selected-resource cache store {store_id:?} lost selector {selector_id:?}",
                    ))
                })?;
                for logical_device_id in &store.logical_device_ids {
                    if selector_capacity_bytes_by_logical_device
                        .entry(selector_id.clone())
                        .or_default()
                        .insert(logical_device_id.clone(), budget)
                        .is_some()
                    {
                        return Err(selected_resource_cache_error(format!(
                            "selected-resource selector {selector_id:?} repeats logical cache target {logical_device_id:?}",
                        )));
                    }
                }
            }
        }
        Ok(VulkanSelectedResourceConcurrentPeerSnapshot {
            generation: state.generation,
            streams: state
                .stream_demands
                .iter()
                .filter(|(candidate, demand)| {
                    **candidate != stream_id && !demand.concurrent_selectors.is_empty()
                })
                .map(|(candidate, demand)| VulkanSelectedResourceConcurrentPeerStream {
                    stream_id: candidate.to_string(),
                    selectors: demand.concurrent_selectors.clone(),
                })
                .collect(),
            selector_capacity_bytes_by_logical_device,
        })
    }

    fn replace_stream_demand_at_generation(
        &self,
        stream_id: u64,
        expected_generation: u64,
        demand: VulkanSelectedResourceStreamCacheDemand,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        let mut state = self.state.lock().map_err(|_| {
            selected_resource_cache_error("selected-resource cache arbiter was poisoned")
        })?;
        if !state.stream_demands.contains_key(&stream_id) {
            return Err(selected_resource_cache_error(format!(
                "selected-resource cache arbiter has no stream {stream_id}",
            )));
        }
        if state.generation != expected_generation {
            return Err(selected_resource_cache_error(format!(
                "selected-resource cache peer generation changed from {expected_generation} to {} while planning",
                state.generation,
            )));
        }
        let next_generation = state.generation.checked_add(1).ok_or_else(|| {
            selected_resource_cache_error("selected-resource cache generation overflowed")
        })?;
        validate_vulkan_selected_resource_stream_cache_demand(&demand, &self.stores)?;
        let previous = state
            .stream_demands
            .insert(stream_id, demand)
            .expect("registered stream was checked above");
        if let Err(error) = self.apply_aggregate_budgets(&state.stream_demands) {
            state.stream_demands.insert(stream_id, previous);
            return Err(error);
        }
        state.generation = next_generation;
        Ok(())
    }

    fn unregister(&self, stream_id: u64) {
        let Ok(_transaction) = self.adaptation_transaction.lock() else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let Some(_) = state.stream_demands.remove(&stream_id) else {
            return;
        };
        state.generation = state.generation.saturating_add(1);
        // Drop cannot report an error. Never retain a dead stream merely
        // because a store is concurrently failing or unloading; the next
        // successful stream update republishes the complete aggregate policy.
        let _ = self.apply_aggregate_budgets(&state.stream_demands);
    }

    fn apply_aggregate_budgets(
        &self,
        demands: &BTreeMap<u64, VulkanSelectedResourceStreamCacheDemand>,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        let replacements = self
            .stores
            .iter()
            .map(|(store_id, store)| {
                let aggregate = aggregate_vulkan_selected_resource_cache_demand(
                    store_id,
                    demands.values(),
                )?;
                vulkan_selected_resource_cache_budget_replacement(store, &aggregate)
                    .map(|replacement| (store_id, replacement))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut applied = Vec::with_capacity(replacements.len());
        for (store_id, replacement) in replacements {
            let store = &self.stores[store_id];
            let mounted = store.store.upgrade().ok_or_else(|| {
                selected_resource_cache_error(format!(
                    "selected-resource cache store {store_id:?} was unloaded",
                ))
            })?;
            match mounted.replace_selector_payload_budgets(replacement) {
                Ok(previous) => applied.push((mounted, previous)),
                Err(error) => {
                    let mut rollback_error = None;
                    for (mounted, previous) in applied.into_iter().rev() {
                        if let Err(error) = mounted.replace_selector_payload_budgets(previous)
                            && rollback_error.is_none()
                        {
                            rollback_error = Some(error.to_string());
                        }
                    }
                    return Err(selected_resource_cache_error(match rollback_error {
                        Some(rollback_error) => format!(
                            "failed to replace selected-resource cache budgets: {error}; rollback also failed: {rollback_error}",
                        ),
                        None => format!(
                            "failed to replace selected-resource cache budgets: {error}",
                        ),
                    }));
                }
            }
        }
        Ok(())
    }
}

fn vulkan_selected_resource_cache_budget_replacement(
    store: &VulkanSelectedResourceCacheStore,
    aggregate: &BTreeMap<String, VulkanSelectedResourceCacheDemand>,
) -> Result<BTreeMap<String, usize>, VulkanResidentTokenModelPackageError> {
    let adaptive = apportion_vulkan_selected_resource_cache_budgets(
        store.adaptive_capacity_bytes,
        &store.adaptive_baseline_budgets,
        &store.resource_payload_bytes,
        aggregate,
    )?;
    let mut replacement = store.full_baseline_budgets.clone();
    for (selector_id, budget) in adaptive {
        replacement.insert(selector_id, budget);
    }
    Ok(replacement)
}

impl VulkanSelectedResourceCacheRegistration {
    #[cfg(test)]
    fn peer_snapshot(
        &self,
    ) -> Result<VulkanSelectedResourceConcurrentPeerSnapshot, VulkanResidentTokenModelPackageError>
    {
        let _transaction = self
            .arbiter
            .adaptation_transaction
            .lock()
            .map_err(|_| {
                selected_resource_cache_error(
                    "selected-resource adaptation transaction was poisoned",
                )
            })?;
        self.arbiter.peer_snapshot(self.stream_id)
    }

    #[cfg(test)]
    fn replace_demand_at_generation(
        &self,
        expected_generation: u64,
        demand: VulkanSelectedResourceStreamCacheDemand,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        let _transaction = self
            .arbiter
            .adaptation_transaction
            .lock()
            .map_err(|_| {
                selected_resource_cache_error(
                    "selected-resource adaptation transaction was poisoned",
                )
            })?;
        self.arbiter.replace_stream_demand_at_generation(
            self.stream_id,
            expected_generation,
            demand,
        )
    }

    fn stream_id(&self) -> u64 {
        self.stream_id
    }
}

impl Drop for VulkanSelectedResourceCacheRegistration {
    fn drop(&mut self) {
        self.arbiter.unregister(self.stream_id);
    }
}

fn selected_resource_cache_error(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message)
}

fn validate_vulkan_selected_resource_stream_cache_demand(
    demand: &VulkanSelectedResourceStreamCacheDemand,
    stores: &BTreeMap<String, VulkanSelectedResourceCacheStore>,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for (selector_id, state) in &demand.concurrent_selectors {
        if selector_id.trim().is_empty()
            || state.telemetry.resource_count == 0
            || state.telemetry.selection_counts.len() != state.telemetry.resource_count
            || state.assignments.len() != state.telemetry.resource_count
        {
            return Err(selected_resource_cache_error(
                "selected-resource concurrent demand has an invalid selector or resource domain",
            ));
        }
        let mut covered = BTreeSet::new();
        for assignment in &state.assignments {
            if assignment.device_id.trim().is_empty()
                || assignment.resource_index >= state.telemetry.resource_count
                || !covered.insert(assignment.resource_index)
                || !stores.values().any(|store| {
                    store.logical_device_ids.contains(&assignment.device_id)
                        && store.resource_payload_bytes.contains_key(selector_id)
                })
            {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource concurrent demand for selector {selector_id:?} repeats ownership or references an unavailable resource target",
                )));
            }
        }
    }
    for (store_id, selectors) in &demand.by_store_selector {
        let store = stores.get(store_id).ok_or_else(|| {
            selected_resource_cache_error(format!(
                "selected-resource cache demand references unknown store {store_id:?}",
            ))
        })?;
        for (selector_id, demand) in selectors {
            let payloads = store.resource_payload_bytes.get(selector_id).ok_or_else(|| {
                selected_resource_cache_error(format!(
                    "selected-resource cache demand references unknown selector {selector_id:?} on {store_id:?}",
                ))
            })?;
            if demand
                .observed_resource_indices
                .iter()
                .any(|resource_index| *resource_index >= payloads.len())
            {
                return Err(selected_resource_cache_error(format!(
                    "selected-resource cache demand for selector {selector_id:?} exceeds its resource domain",
                )));
            }
        }
    }
    Ok(())
}

fn aggregate_vulkan_selected_resource_cache_demand<'a>(
    store_id: &str,
    demands: impl IntoIterator<Item = &'a VulkanSelectedResourceStreamCacheDemand>,
) -> Result<BTreeMap<String, VulkanSelectedResourceCacheDemand>, VulkanResidentTokenModelPackageError>
{
    let mut aggregate = BTreeMap::<String, VulkanSelectedResourceCacheDemand>::new();
    for demand in demands {
        for (selector_id, stream) in demand
            .by_store_selector
            .get(store_id)
            .into_iter()
            .flat_map(|selectors| selectors.iter())
        {
            let destination = aggregate.entry(selector_id.clone()).or_default();
            destination.weighted_bytes = destination
                .weighted_bytes
                .checked_add(stream.weighted_bytes)
                .ok_or_else(|| {
                    selected_resource_cache_error(
                        "selected-resource aggregate cache demand overflowed",
                    )
                })?;
            destination
                .observed_resource_indices
                .extend(&stream.observed_resource_indices);
        }
    }
    Ok(aggregate)
}

fn apportion_vulkan_selected_resource_cache_budgets(
    capacity_bytes: usize,
    baseline: &BTreeMap<String, usize>,
    resource_payload_bytes: &BTreeMap<String, Vec<usize>>,
    demand: &BTreeMap<String, VulkanSelectedResourceCacheDemand>,
) -> Result<BTreeMap<String, usize>, VulkanResidentTokenModelPackageError> {
    let baseline_total = baseline.values().try_fold(0usize, |total, budget| {
        total.checked_add(*budget).ok_or_else(|| {
            selected_resource_cache_error("selected-resource cache baseline capacity overflowed")
        })
    })?;
    if capacity_bytes == 0
        || baseline.is_empty()
        || baseline.keys().ne(resource_payload_bytes.keys())
        || demand.keys().any(|selector_id| !baseline.contains_key(selector_id))
        || baseline_total != capacity_bytes
    {
        return Err(selected_resource_cache_error(
            "selected-resource cache apportionment has incompatible capacity or selector domains",
        ));
    }
    for (selector_id, selector_demand) in demand {
        let payloads = &resource_payload_bytes[selector_id];
        if selector_demand
            .observed_resource_indices
            .iter()
            .any(|resource_index| *resource_index >= payloads.len())
        {
            return Err(selected_resource_cache_error(format!(
                "selected-resource cache demand for selector {selector_id:?} exceeds its resource domain",
            )));
        }
    }
    if demand.values().all(|demand| {
        demand.weighted_bytes == 0 && demand.observed_resource_indices.is_empty()
    }) {
        return Ok(baseline.clone());
    }
    let minimums = baseline
        .keys()
        .map(|selector_id| {
            let minimum = demand
                .get(selector_id)
                .into_iter()
                .flat_map(|demand| demand.observed_resource_indices.iter())
                .try_fold(0usize, |total, resource_index| {
                    total.checked_add(resource_payload_bytes[selector_id][*resource_index])
                        .ok_or_else(|| {
                            selected_resource_cache_error(
                                "selected-resource observed working-set bytes overflowed",
                            )
                        })
                })?;
            Ok((selector_id.clone(), minimum))
        })
        .collect::<Result<BTreeMap<_, _>, VulkanResidentTokenModelPackageError>>()?;
    let minimum_total = minimums.values().try_fold(0usize, |total, minimum| {
        total.checked_add(*minimum).ok_or_else(|| {
            selected_resource_cache_error("selected-resource cache minimum total overflowed")
        })
    })?;
    let mut weights = baseline
        .keys()
        .map(|selector_id| {
            (
                selector_id.clone(),
                demand
                    .get(selector_id)
                    .map(|demand| demand.weighted_bytes)
                    .unwrap_or(0),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if weights.values().all(|weight| *weight == 0) {
        weights = baseline
            .iter()
            .map(|(selector_id, budget)| (selector_id.clone(), *budget as u128))
            .collect();
    }
    let weight_total = weights.values().try_fold(0u128, |total, weight| {
        total.checked_add(*weight).ok_or_else(|| {
            selected_resource_cache_error("selected-resource cache weight total overflowed")
        })
    })?;
    let mut budgets = baseline
        .keys()
        .map(|selector_id| {
            let bytes = if minimum_total <= capacity_bytes {
                minimums[selector_id]
            } else {
                proportional_vulkan_selected_resource_cache_share(
                    capacity_bytes,
                    minimums[selector_id] as u128,
                    minimum_total as u128,
                )?
            };
            Ok((selector_id.clone(), bytes))
        })
        .collect::<Result<BTreeMap<_, _>, VulkanResidentTokenModelPackageError>>()?;
    let used = budgets.values().sum::<usize>();
    let remaining = capacity_bytes.saturating_sub(used);
    if remaining > 0 && minimum_total <= capacity_bytes {
        for selector_id in baseline.keys() {
            let share = proportional_vulkan_selected_resource_cache_share(
                remaining,
                weights[selector_id],
                weight_total,
            )?;
            budgets.insert(
                selector_id.clone(),
                budgets[selector_id].checked_add(share).ok_or_else(|| {
                    selected_resource_cache_error(
                        "selected-resource cache apportioned budget overflowed",
                    )
                })?,
            );
        }
    }
    let mut remainder = capacity_bytes.saturating_sub(budgets.values().sum::<usize>());
    let mut order = baseline
        .keys()
        .map(|selector_id| (weights[selector_id], selector_id.as_str()))
        .collect::<Vec<_>>();
    order.sort_by(|left, right| right.cmp(left));
    for (_, selector_id) in order.into_iter().cycle() {
        if remainder == 0 {
            break;
        }
        budgets.insert(selector_id.to_string(), budgets[selector_id] + 1);
        remainder -= 1;
    }
    debug_assert_eq!(budgets.values().sum::<usize>(), capacity_bytes);
    Ok(budgets)
}

fn proportional_vulkan_selected_resource_cache_share(
    capacity_bytes: usize,
    weight: u128,
    total_weight: u128,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    if total_weight == 0 || weight > total_weight {
        return Err(selected_resource_cache_error(
            "selected-resource cache proportional share has invalid weights",
        ));
    }
    let numerator = (capacity_bytes as u128).checked_mul(weight).ok_or_else(|| {
        selected_resource_cache_error(
            "selected-resource cache proportional share overflowed",
        )
    })?;
    usize::try_from(numerator / total_weight).map_err(|_| {
        selected_resource_cache_error(
            "selected-resource cache proportional share exceeds the host address space",
        )
    })
}

#[cfg(test)]
mod runtime_selected_resource_cache_arbiter_tests {
    use super::*;

    fn test_arbiter_without_stores() -> Arc<VulkanSelectedResourceCacheArbiter> {
        Arc::new(VulkanSelectedResourceCacheArbiter {
            next_stream_id: std::sync::atomic::AtomicU64::new(1),
            stores: BTreeMap::new(),
            adaptation_transaction: std::sync::Mutex::new(()),
            state: std::sync::Mutex::new(VulkanSelectedResourceCacheArbiterState::default()),
        })
    }

    #[test]
    fn peer_snapshot_is_generation_guarded_and_excludes_unpublished_streams() {
        let arbiter = test_arbiter_without_stores();
        let current = arbiter.register().unwrap();
        let peer = arbiter.register().unwrap();
        let initial = current.peer_snapshot().unwrap();
        assert!(initial.streams.is_empty());
        assert_eq!(initial.generation, 2);

        {
            let mut state = arbiter.state.lock().unwrap();
            state
                .stream_demands
                .get_mut(&peer.stream_id)
                .unwrap()
                .concurrent_selectors
                .insert(
                    "experts".to_string(),
                    VulkanSelectedResourceConcurrentSelectorState {
                        telemetry: VulkanSelectionTelemetryDomainSnapshot {
                            execution_scope: "target".to_string(),
                            component_id: "layer".to_string(),
                            node_id: "router".to_string(),
                            domain_id: "bank".to_string(),
                            resource_count: 1,
                            selection_counts: vec![1],
                            co_selection_counts: Vec::new(),
                        },
                        assignments: vec![VulkanSelectedResourceAssignment {
                            resource_index: 0,
                            device_id: "gpu0".to_string(),
                        }],
                    },
                );
        }
        let populated = current.peer_snapshot().unwrap();
        assert_eq!(populated.streams.len(), 1);
        assert_eq!(populated.streams[0].stream_id, peer.stream_id.to_string());

        peer.replace_demand_at_generation(
            populated.generation,
            VulkanSelectedResourceStreamCacheDemand::default(),
        )
        .unwrap();
        assert!(
            current
                .replace_demand_at_generation(
                    populated.generation,
                    VulkanSelectedResourceStreamCacheDemand::default(),
                )
                .unwrap_err()
                .to_string()
                .contains("generation changed")
        );
    }

    #[test]
    fn cache_budget_apportionment_preserves_hot_union_and_exact_capacity() {
        let baseline = BTreeMap::from([
            ("a".to_string(), 50),
            ("b".to_string(), 50),
        ]);
        let payloads = BTreeMap::from([
            ("a".to_string(), vec![10, 20, 30]),
            ("b".to_string(), vec![40, 50]),
        ]);
        let demand = BTreeMap::from([
            (
                "a".to_string(),
                VulkanSelectedResourceCacheDemand {
                    weighted_bytes: 90,
                    observed_resource_indices: BTreeSet::from([0, 1]),
                },
            ),
            (
                "b".to_string(),
                VulkanSelectedResourceCacheDemand {
                    weighted_bytes: 10,
                    observed_resource_indices: BTreeSet::from([0]),
                },
            ),
        ]);
        let budgets = apportion_vulkan_selected_resource_cache_budgets(
            100, &baseline, &payloads, &demand,
        )
        .unwrap();
        assert_eq!(budgets.values().sum::<usize>(), 100);
        assert!(budgets["a"] >= 30);
        assert!(budgets["b"] >= 40);
        assert!(budgets["a"] > budgets["b"]);
    }

    #[test]
    fn cache_demand_aggregates_resource_union_across_streams() {
        let stream = |weighted_bytes, resources| VulkanSelectedResourceStreamCacheDemand {
            by_store_selector: BTreeMap::from([(
                "gpu0".to_string(),
                BTreeMap::from([(
                    "experts".to_string(),
                    VulkanSelectedResourceCacheDemand {
                        weighted_bytes,
                        observed_resource_indices: resources,
                    },
                )]),
            )]),
            ..VulkanSelectedResourceStreamCacheDemand::default()
        };
        let first = stream(10, BTreeSet::from([0, 1]));
        let second = stream(20, BTreeSet::from([1, 2]));
        let aggregate = aggregate_vulkan_selected_resource_cache_demand(
            "gpu0",
            [&first, &second],
        )
        .unwrap();
        assert_eq!(aggregate["experts"].weighted_bytes, 30);
        assert_eq!(
            aggregate["experts"].observed_resource_indices,
            BTreeSet::from([0, 1, 2]),
        );
    }

    #[test]
    fn cache_budget_apportionment_scales_an_oversubscribed_hot_union() {
        let baseline = BTreeMap::from([
            ("a".to_string(), 50),
            ("b".to_string(), 50),
        ]);
        let payloads = BTreeMap::from([
            ("a".to_string(), vec![80]),
            ("b".to_string(), vec![120]),
        ]);
        let demand = BTreeMap::from([
            (
                "a".to_string(),
                VulkanSelectedResourceCacheDemand {
                    weighted_bytes: 80,
                    observed_resource_indices: BTreeSet::from([0]),
                },
            ),
            (
                "b".to_string(),
                VulkanSelectedResourceCacheDemand {
                    weighted_bytes: 120,
                    observed_resource_indices: BTreeSet::from([0]),
                },
            ),
        ]);
        let budgets = apportion_vulkan_selected_resource_cache_budgets(
            100, &baseline, &payloads, &demand,
        )
        .unwrap();
        assert_eq!(
            budgets,
            BTreeMap::from([("a".to_string(), 40), ("b".to_string(), 60)])
        );
    }

    #[test]
    fn cache_budget_apportionment_rejects_unknown_selector_demand() {
        let baseline = BTreeMap::from([("a".to_string(), 10)]);
        let payloads = BTreeMap::from([("a".to_string(), vec![10])]);
        let demand = BTreeMap::from([(
            "missing".to_string(),
            VulkanSelectedResourceCacheDemand::default(),
        )]);
        assert!(
            apportion_vulkan_selected_resource_cache_budgets(
                10, &baseline, &payloads, &demand,
            )
            .unwrap_err()
            .to_string()
            .contains("selector domains")
        );
    }

    #[test]
    fn cache_budget_apportionment_rejects_an_out_of_range_hot_resource() {
        let baseline = BTreeMap::from([("a".to_string(), 10)]);
        let payloads = BTreeMap::from([("a".to_string(), vec![10])]);
        let demand = BTreeMap::from([(
            "a".to_string(),
            VulkanSelectedResourceCacheDemand {
                weighted_bytes: 10,
                observed_resource_indices: BTreeSet::from([1]),
            },
        )]);
        assert!(
            apportion_vulkan_selected_resource_cache_budgets(
                10, &baseline, &payloads, &demand,
            )
            .unwrap_err()
            .to_string()
            .contains("exceeds its resource domain")
        );
    }

    #[test]
    fn cache_budget_apportionment_rejects_a_capacity_mismatch() {
        let baseline = BTreeMap::from([("a".to_string(), 10)]);
        let payloads = BTreeMap::from([("a".to_string(), vec![10])]);
        assert!(
            apportion_vulkan_selected_resource_cache_budgets(
                11,
                &baseline,
                &payloads,
                &BTreeMap::new(),
            )
            .unwrap_err()
            .to_string()
            .contains("incompatible capacity")
        );
    }

    #[test]
    fn cache_budget_replacement_preserves_nonadaptive_selector_share() {
        let store = VulkanSelectedResourceCacheStore {
            store: std::sync::Weak::new(),
            logical_device_ids: BTreeSet::from(["gpu0".to_string()]),
            full_baseline_budgets: BTreeMap::from([
                ("a".to_string(), 40),
                ("b".to_string(), 40),
                ("fixed".to_string(), 20),
            ]),
            adaptive_capacity_bytes: 80,
            adaptive_baseline_budgets: BTreeMap::from([
                ("a".to_string(), 40),
                ("b".to_string(), 40),
            ]),
            resource_payload_bytes: BTreeMap::from([
                ("a".to_string(), vec![10]),
                ("b".to_string(), vec![10]),
            ]),
        };
        let demand = BTreeMap::from([(
            "a".to_string(),
            VulkanSelectedResourceCacheDemand {
                weighted_bytes: 1,
                observed_resource_indices: BTreeSet::from([0]),
            },
        )]);
        let replacement =
            vulkan_selected_resource_cache_budget_replacement(&store, &demand).unwrap();
        assert_eq!(replacement["fixed"], 20);
        assert_eq!(replacement["a"] + replacement["b"], 80);
        assert_eq!(replacement.values().sum::<usize>(), 100);
    }
}
