#[derive(Debug)]
pub struct VulkanCompiledResourceDeviceStoreError(String);

impl VulkanCompiledResourceDeviceStoreError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for VulkanCompiledResourceDeviceStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanCompiledResourceDeviceStoreError {}

fn upload_compiled_resource_tier(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    arena: &VulkanStableResourceArena,
    address_table: &mut VulkanStableResourceAddressTable,
    loaded: &[(
        &VulkanCompiledResourceLoadPlan,
        DeviceResourceLoadPermit<VulkanResidentCompiledResource>,
        LoadedCompiledResourceGroup,
    )],
    indices: &[usize],
    alignment: usize,
    capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
) -> Result<Vec<(usize, VulkanStableCompiledResourceUpload)>, VulkanError> {
    if indices.is_empty() {
        return Ok(Vec::new());
    }
    let requests = indices
        .iter()
        .map(|index| {
            let (plan, _, loaded) = &loaded[*index];
            VulkanStableCompiledResourceUploadRequest {
                descriptor: &plan.descriptor,
                loaded,
                resource_slots: &plan.resource_slots,
            }
        })
        .collect::<Vec<_>>();
    upload_loaded_compiled_resource_groups_to_stable_address_space(
        device,
        transfer,
        arena,
        address_table,
        &requests,
        alignment,
        capacity_permit,
    )
    .map(|uploads| indices.iter().copied().zip(uploads).collect())
}

fn compiled_resource_backing_worker_count_for_parallelism(
    maximum_load_wave_group_count: usize,
    available_parallelism: usize,
) -> usize {
    maximum_load_wave_group_count
        .max(1)
        .min(available_parallelism.max(1))
}

fn compiled_resource_backing_worker_count(maximum_load_wave_group_count: usize) -> usize {
    compiled_resource_backing_worker_count_for_parallelism(
        maximum_load_wave_group_count,
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    )
}

const COMPILED_RESOURCE_DEVICE_CAPACITY_SETTLEMENT_TIMEOUT: Duration =
    Duration::from_millis(250);

fn wait_for_compiled_resource_device_capacity<F>(
    required_bytes: usize,
    timeout: Duration,
    mut remaining_bytes: F,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError>
where
    F: FnMut() -> Result<usize, VulkanCompiledResourceDeviceStoreError>,
{
    if required_bytes == 0 {
        return Ok(0);
    }
    let started = Instant::now();
    loop {
        let remaining = remaining_bytes()?;
        if remaining >= required_bytes {
            return Ok(remaining);
        }
        if started.elapsed() >= timeout {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "Vulkan device-local capacity did not settle within {:.3} ms: the next allocation needs {required_bytes} bytes but only {remaining} bytes are acknowledged as available",
                timeout.as_secs_f64() * 1_000.0,
            )));
        }
        std::thread::sleep(Duration::from_micros(100));
    }
}

#[derive(Debug)]
struct VulkanCompiledResourceEvictionSelection {
    group_ids: BTreeSet<String>,
    payload_bytes: usize,
    device_bytes: usize,
    host_visible_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanCompiledResourceEvictionGranularity {
    AllocationBlock,
    PhysicalSlab,
}

fn detach_compiled_resource_group_cohorts(
    group_id: &str,
    group_cohorts: &mut BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    cohort_groups: &mut BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
) {
    let cohorts = group_cohorts.remove(group_id).unwrap_or_default();
    for cohort in cohorts {
        let remove_cohort = if let Some(groups) = cohort_groups.get_mut(&cohort) {
            groups.remove(group_id);
            groups.is_empty()
        } else {
            false
        };
        if remove_cohort {
            cohort_groups.remove(&cohort);
        }
    }
}

fn compiled_resource_lru_eviction_selection(
    candidates: &[DeviceResourceResidencyEvictionCandidate],
    group_chunks: &BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    chunk_groups: &BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    chunk_byte_capacities: &BTreeMap<VulkanCompiledResourceAllocationCohort, usize>,
    protected_group_ids: &BTreeSet<String>,
    required_payload_bytes: usize,
    required_device_bytes: usize,
    required_host_visible_bytes: usize,
) -> Result<VulkanCompiledResourceEvictionSelection, VulkanCompiledResourceDeviceStoreError> {
    if required_payload_bytes == 0
        && required_device_bytes == 0
        && required_host_visible_bytes == 0
    {
        return Ok(VulkanCompiledResourceEvictionSelection {
            group_ids: BTreeSet::new(),
            payload_bytes: 0,
            device_bytes: 0,
            host_visible_bytes: 0,
        });
    }
    let candidates_by_id = candidates
        .iter()
        .map(|candidate| (candidate.group_id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();
    let mut selected = BTreeSet::new();
    let mut selected_chunks = BTreeSet::new();
    let mut selected_payload_bytes = 0usize;
    let mut selected_device_bytes = 0usize;
    let mut selected_host_visible_bytes = 0usize;
    for candidate in candidates {
        if selected.contains(&candidate.group_id) {
            continue;
        }
        let mut cohort = BTreeSet::new();
        let mut cohort_chunks = BTreeSet::new();
        let mut pending_groups = vec![candidate.group_id.clone()];
        while let Some(group_id) = pending_groups.pop() {
            if !cohort.insert(group_id.clone()) {
                continue;
            }
            let chunks = group_chunks.get(&group_id).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "resident group {group_id:?} has no stable allocation cohort"
                ))
            })?;
            for chunk_id in chunks {
                if !cohort_chunks.insert(*chunk_id) {
                    continue;
                }
                let groups = chunk_groups.get(chunk_id).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "stable allocation cohort {chunk_id:?} has no resident groups"
                    ))
                })?;
                pending_groups.extend(groups.iter().cloned());
            }
        }
        if cohort.is_empty()
            || cohort
                .iter()
                .any(|group_id| protected_group_ids.contains(group_id))
            || cohort
                .iter()
                .any(|group_id| !candidates_by_id.contains_key(group_id.as_str()))
        {
            continue;
        }
        let cohort_payload_bytes = cohort.iter().try_fold(0usize, |total, group_id| {
            if selected.contains(group_id) {
                return Ok(total);
            }
            let candidate = candidates_by_id
                .get(group_id.as_str())
                .expect("eviction cohort was restricted to candidates");
            total.checked_add(candidate.byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource eviction byte count overflowed",
                )
            })
        })?;
        let cohort_device_bytes = cohort_chunks.iter().try_fold(0usize, |total, chunk| {
            if required_device_bytes == 0
                || selected_chunks.contains(chunk)
                || chunk.tier != VulkanCompiledResourceMemoryTier::Device
            {
                return Ok(total);
            }
            let byte_capacity = chunk_byte_capacities.get(chunk).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "stable allocation cohort {chunk:?} has no physical byte capacity"
                ))
            })?;
            total.checked_add(*byte_capacity).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource device eviction byte count overflowed",
                )
            })
        })?;
        let cohort_host_visible_bytes =
            cohort_chunks.iter().try_fold(0usize, |total, chunk| {
                if required_host_visible_bytes == 0
                    || selected_chunks.contains(chunk)
                    || chunk.tier != VulkanCompiledResourceMemoryTier::HostVisible
                {
                    return Ok(total);
                }
                let byte_capacity = chunk_byte_capacities.get(chunk).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "stable allocation cohort {chunk:?} has no physical byte capacity"
                    ))
                })?;
                total.checked_add(*byte_capacity).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource host-visible eviction byte count overflowed",
                    )
                })
            })?;
        let contributes_payload = selected_payload_bytes < required_payload_bytes
            && cohort_payload_bytes > 0;
        let contributes_device = selected_device_bytes < required_device_bytes
            && cohort_device_bytes > 0;
        let contributes_host_visible = selected_host_visible_bytes
            < required_host_visible_bytes
            && cohort_host_visible_bytes > 0;
        if !contributes_payload && !contributes_device && !contributes_host_visible {
            continue;
        }
        selected.extend(cohort);
        selected_chunks.extend(cohort_chunks);
        selected_payload_bytes = selected_payload_bytes
            .checked_add(cohort_payload_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource eviction byte count overflowed",
                )
            })?;
        selected_device_bytes = selected_device_bytes
            .checked_add(cohort_device_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource device eviction byte count overflowed",
                )
            })?;
        selected_host_visible_bytes = selected_host_visible_bytes
            .checked_add(cohort_host_visible_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource host-visible eviction byte count overflowed",
                )
            })?;
        if selected_payload_bytes >= required_payload_bytes
            && selected_device_bytes >= required_device_bytes
            && selected_host_visible_bytes >= required_host_visible_bytes
        {
            break;
        }
    }
    Ok(VulkanCompiledResourceEvictionSelection {
        group_ids: selected,
        payload_bytes: selected_payload_bytes,
        device_bytes: selected_device_bytes,
        host_visible_bytes: selected_host_visible_bytes,
    })
}

#[cfg(test)]
fn compiled_resource_lru_eviction_groups(
    candidates: &[DeviceResourceResidencyEvictionCandidate],
    group_chunks: &BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    chunk_groups: &BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    chunk_byte_capacities: &BTreeMap<VulkanCompiledResourceAllocationCohort, usize>,
    protected_group_ids: &BTreeSet<String>,
    required_payload_bytes: usize,
    required_device_bytes: usize,
) -> Result<BTreeSet<String>, VulkanCompiledResourceDeviceStoreError> {
    let selection = compiled_resource_lru_eviction_selection(
        candidates,
        group_chunks,
        chunk_groups,
        chunk_byte_capacities,
        protected_group_ids,
        required_payload_bytes,
        required_device_bytes,
        0,
    )?;
    if selection.payload_bytes < required_payload_bytes
        || selection.device_bytes < required_device_bytes
    {
        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource residency needs to reclaim {required_payload_bytes} payload and {required_device_bytes} device bytes, but inactive allocation cohorts provide only {} payload and {} device bytes",
            selection.payload_bytes, selection.device_bytes,
        )));
    }
    Ok(selection.group_ids)
}

fn compiled_resource_selector_fair_eviction_candidates(
    candidates: &[DeviceResourceResidencyEvictionCandidate],
    directory: &[DeviceResourceResidencyDirectoryEntry],
    group_selector_ids: &BTreeMap<String, String>,
    selector_payload_budgets: &BTreeMap<String, usize>,
    incoming_selector_id: &str,
    incoming_payload_bytes: usize,
) -> Result<Vec<DeviceResourceResidencyEvictionCandidate>, VulkanCompiledResourceDeviceStoreError> {
    let mut resident_payload_bytes = BTreeMap::<String, usize>::new();
    for entry in directory
        .iter()
        .filter(|entry| entry.state == ResourceResidencyState::Resident)
    {
        let Some(selector_id) = group_selector_ids.get(&entry.group_id) else {
            continue;
        };
        let total = resident_payload_bytes.entry(selector_id.clone()).or_default();
        *total = total.checked_add(entry.byte_count).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled selector resident payload byte count overflowed",
            )
        })?;
    }
    let incoming_total = resident_payload_bytes
        .entry(incoming_selector_id.to_string())
        .or_default();
    *incoming_total = incoming_total
        .checked_add(incoming_payload_bytes)
        .ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled selector incoming payload byte count overflowed",
            )
        })?;

    let selector_overages = resident_payload_bytes
        .into_iter()
        .map(|(selector_id, resident_bytes)| {
            let budget = selector_payload_budgets
                .get(&selector_id)
                .copied()
                .unwrap_or(0);
            (selector_id, resident_bytes.saturating_sub(budget))
        })
        .collect::<BTreeMap<_, _>>();
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| {
        let left_selector = group_selector_ids.get(&left.group_id);
        let right_selector = group_selector_ids.get(&right.group_id);
        let left_overage = left_selector
            .and_then(|selector_id| selector_overages.get(selector_id))
            .copied()
            .unwrap_or(0);
        let right_overage = right_selector
            .and_then(|selector_id| selector_overages.get(selector_id))
            .copied()
            .unwrap_or(0);
        let left_tier = if left_overage > 0 {
            0
        } else if left_selector.is_none() {
            1
        } else {
            2
        };
        let right_tier = if right_overage > 0 {
            0
        } else if right_selector.is_none() {
            1
        } else {
            2
        };
        left_tier
            .cmp(&right_tier)
            .then_with(|| right_overage.cmp(&left_overage))
            .then_with(|| {
                (left.last_access_epoch, left.group_id.as_str())
                    .cmp(&(right.last_access_epoch, right.group_id.as_str()))
            })
    });
    Ok(ordered)
}

struct VulkanCompiledResourceDeviceAddressState {
    transfer: VulkanResidentTransferStream,
    address_table: VulkanStableResourceAddressTable,
    publications: BTreeMap<String, Vec<VulkanStableResourceAddressPublication>>,
    group_chunks: BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    chunk_groups: BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    group_blocks: BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    block_groups: BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    promoted_representations:
        BTreeMap<String, VulkanCompiledResourcePromotedRepresentation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanCompiledResourceStoreLifecycleState {
    Active,
    Failed,
    Quiescing,
    Unloaded,
}

struct VulkanCompiledResourceStoreLifecycle {
    state: VulkanCompiledResourceStoreLifecycleState,
    active_load_operation_count: usize,
    memory_reclamation_in_progress: bool,
    teardown_in_progress: bool,
    terminal_failure: Option<String>,
    pending_release: DeviceResourceResidencyRelease,
}

struct VulkanCompiledResourceStoreLoadGuard<'a> {
    store: &'a VulkanCompiledResourceDeviceStore,
}

struct VulkanCompiledResourceLoadPlan {
    descriptor: DeviceResourceGroupDescriptor,
    resolved: ResolvedCompiledResourceGroup,
    resource_slots: Vec<usize>,
}


struct VulkanCompiledResourceSelectorCachePolicy {
    group_selector_ids: BTreeMap<String, String>,
    group_selections: BTreeMap<String, (String, usize)>,
    group_payload_bytes: BTreeMap<String, usize>,
    selector_payload_budgets: BTreeMap<String, usize>,
}

struct VulkanCompiledResourceDeviceMemoryReclaimer {
    store: std::sync::Weak<VulkanCompiledResourceDeviceStore>,
}

struct VulkanCompiledResourceDeviceMemoryReclamation {
    store: Arc<VulkanCompiledResourceDeviceStore>,
}

impl std::fmt::Debug for VulkanCompiledResourceDeviceMemoryReclaimer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanCompiledResourceDeviceMemoryReclaimer")
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for VulkanCompiledResourceDeviceMemoryReclamation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanCompiledResourceDeviceMemoryReclamation")
            .field("store_id", &self.store.device_id)
            .finish()
    }
}

impl VulkanDeviceLocalMemoryReclaimer for VulkanCompiledResourceDeviceMemoryReclaimer {
    fn begin_device_local_memory_reclamation(
        &self,
        requested_bytes: usize,
    ) -> Result<
        Box<dyn crate::vulkan_compute::VulkanDeviceLocalMemoryReclamation>,
        VulkanError,
    > {
        let Some(store) = self.store.upgrade() else {
            return Err(VulkanError(
                "compiled-resource memory reclaimer outlived its store".to_string(),
            ));
        };
        store
            .begin_memory_reclamation_boundary()
            .map_err(|error| VulkanError(error.to_string()))?;
        if let Err(error) = store.prepare_inactive_device_memory_reclamation(requested_bytes) {
            store.finish_memory_reclamation_boundary();
            return Err(VulkanError(error.to_string()));
        }
        Ok(Box::new(VulkanCompiledResourceDeviceMemoryReclamation {
            store,
        }))
    }
}

impl crate::vulkan_compute::VulkanDeviceLocalMemoryReclamation
    for VulkanCompiledResourceDeviceMemoryReclamation
{
    fn reclaim_device_local_memory(
        &self,
        quiescence: &crate::vulkan_compute::VulkanDeviceLocalMemoryQuiescence<'_>,
        requested_bytes: usize,
    ) -> Result<usize, VulkanError> {
        self.store
            .reclaim_inactive_device_memory(quiescence, requested_bytes)
            .map_err(|error| VulkanError(error.to_string()))
    }
}

impl Drop for VulkanCompiledResourceDeviceMemoryReclamation {
    fn drop(&mut self) {
        self.store.finish_memory_reclamation_boundary();
    }
}

pub struct VulkanCompiledResourceDeviceStore {
    residency_policy: ResourceResidencyPolicy,
    device_id: String,
    physical_device_id: String,
    logical_device_ids: Vec<String>,
    allowed_selector_ids: BTreeSet<String>,
    selector_ownership: VulkanCompiledResourceSelectorOwnership,
    package_root: PathBuf,
    contract: Arc<CompiledResourceResidencyContract>,
    contract_index: CompiledResourceContractIndex,
    layout: Arc<VulkanCompiledResourceAddressLayout>,
    device_arena: VulkanStableResourceArena,
    host_visible_arena: Option<VulkanStableResourceArena>,
    representation_arena: Option<VulkanStableResourceArena>,
    shared_host_cache: Option<Arc<VulkanCompiledResourceSharedHostCache>>,
    memory_plan: Option<std::sync::Mutex<VulkanCompiledResourceMemoryPlan>>,
    address_state: std::sync::Mutex<VulkanCompiledResourceDeviceAddressState>,
    residency_mutation: std::sync::Mutex<()>,
    backing_store: CompiledResourceBackingStore,
    representation_backing_store: Option<CompiledResourceBackingStore>,
    manager: DeviceResourceResidencyManager<VulkanResidentCompiledResource>,
    upload_alignment: usize,
    maximum_dynamic_payload_bytes: usize,
    maximum_dynamic_device_payload_bytes: usize,
    maximum_dynamic_host_visible_payload_bytes: usize,
    maximum_allocation_byte_capacity: usize,
    maximum_group_byte_count: usize,
    maximum_ranges_per_group: usize,
    always_resident_parameter_bytes: usize,
    runtime_working_set_device_bytes: usize,
    metadata_device_bytes: usize,
    transfer_staging_host_bytes: usize,
    maximum_load_wave_group_count: usize,
    group_selector_ids: BTreeMap<String, String>,
    group_selections: BTreeMap<String, (String, usize)>,
    group_payload_bytes: BTreeMap<String, usize>,
    selector_payload_budgets: BTreeMap<String, usize>,
    retiering_last_selection_counts: std::sync::Mutex<BTreeMap<String, u64>>,
    representation_history:
        std::sync::Mutex<VulkanCompiledResourceRepresentationHistory>,
    coverage_index: Vec<VulkanCompiledResourceComponentCoverageIndex>,
    instrumentation: VulkanCompiledResourceStoreInstrumentation,
    lifecycle: std::sync::Mutex<VulkanCompiledResourceStoreLifecycle>,
    lifecycle_changed: std::sync::Condvar,
    memory_reclaimer_registration:
        std::sync::Mutex<Option<VulkanDeviceLocalMemoryReclaimerRegistration>>,
    #[cfg(test)]
    fail_next_teardown_before_address_clear: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    fail_next_upload_as_device_lost: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    fail_next_retiering_after_payload_exchange: std::sync::atomic::AtomicBool,
}

impl VulkanCompiledResourceDeviceStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &VulkanComputeDevice,
        residency_policy: ResourceResidencyPolicy,
        device_id: impl Into<String>,
        physical_device_id: impl Into<String>,
        logical_device_ids: Vec<String>,
        package_root: impl Into<PathBuf>,
        contract: Arc<CompiledResourceResidencyContract>,
        layout: Arc<VulkanCompiledResourceAddressLayout>,
        allowed_selector_ids: BTreeSet<String>,
        maximum_dynamic_payload_bytes: usize,
        available_dynamic_device_bytes: usize,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
        always_resident_parameter_bytes: usize,
        runtime_working_set_device_bytes: usize,
        metadata_device_bytes: usize,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        Self::new_tiered(
            device,
            residency_policy,
            device_id,
            physical_device_id,
            logical_device_ids,
            package_root,
            contract,
            layout,
            allowed_selector_ids,
            maximum_dynamic_payload_bytes,
            maximum_dynamic_payload_bytes,
            0,
            available_dynamic_device_bytes,
            maximum_group_byte_count,
            maximum_ranges_per_group,
            always_resident_parameter_bytes,
            runtime_working_set_device_bytes,
            metadata_device_bytes,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_tiered(
        device: &VulkanComputeDevice,
        residency_policy: ResourceResidencyPolicy,
        device_id: impl Into<String>,
        physical_device_id: impl Into<String>,
        logical_device_ids: Vec<String>,
        package_root: impl Into<PathBuf>,
        contract: Arc<CompiledResourceResidencyContract>,
        layout: Arc<VulkanCompiledResourceAddressLayout>,
        allowed_selector_ids: BTreeSet<String>,
        maximum_dynamic_payload_bytes: usize,
        maximum_dynamic_device_payload_bytes: usize,
        maximum_dynamic_host_visible_payload_bytes: usize,
        available_dynamic_device_bytes: usize,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
        always_resident_parameter_bytes: usize,
        runtime_working_set_device_bytes: usize,
        metadata_device_bytes: usize,
        shared_host_cache: Option<Arc<VulkanCompiledResourceSharedHostCache>>,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let selector_ownership = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &allowed_selector_ids,
        )
        .map_err(|error| VulkanCompiledResourceDeviceStoreError::new(error.to_string()))?;
        Self::new_tiered_with_selector_ownership(
            device,
            residency_policy,
            device_id,
            physical_device_id,
            logical_device_ids,
            package_root,
            contract,
            layout,
            selector_ownership,
            maximum_dynamic_payload_bytes,
            maximum_dynamic_device_payload_bytes,
            maximum_dynamic_host_visible_payload_bytes,
            available_dynamic_device_bytes,
            maximum_group_byte_count,
            maximum_ranges_per_group,
            always_resident_parameter_bytes,
            runtime_working_set_device_bytes,
            metadata_device_bytes,
            shared_host_cache,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_tiered_with_selector_ownership(
        device: &VulkanComputeDevice,
        residency_policy: ResourceResidencyPolicy,
        device_id: impl Into<String>,
        physical_device_id: impl Into<String>,
        logical_device_ids: Vec<String>,
        package_root: impl Into<PathBuf>,
        contract: Arc<CompiledResourceResidencyContract>,
        layout: Arc<VulkanCompiledResourceAddressLayout>,
        selector_ownership: VulkanCompiledResourceSelectorOwnership,
        maximum_dynamic_payload_bytes: usize,
        maximum_dynamic_device_payload_bytes: usize,
        maximum_dynamic_host_visible_payload_bytes: usize,
        available_dynamic_device_bytes: usize,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
        always_resident_parameter_bytes: usize,
        runtime_working_set_device_bytes: usize,
        metadata_device_bytes: usize,
        shared_host_cache: Option<Arc<VulkanCompiledResourceSharedHostCache>>,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let device_id = device_id.into();
        let physical_device_id = physical_device_id.into();
        let allowed_selector_ids = selector_ownership.selector_ids();
        if device_id.trim().is_empty()
            || physical_device_id.trim().is_empty()
            || logical_device_ids.is_empty()
            || logical_device_ids
                .iter()
                .any(|logical_device_id| logical_device_id.trim().is_empty())
            || maximum_dynamic_payload_bytes == 0
            || maximum_dynamic_device_payload_bytes == 0
            || maximum_dynamic_device_payload_bytes > maximum_dynamic_payload_bytes
            || available_dynamic_device_bytes == 0
            || maximum_group_byte_count == 0
            || maximum_ranges_per_group == 0
            || maximum_group_byte_count > maximum_dynamic_payload_bytes
            || layout.slot_count() == 0
            || allowed_selector_ids.is_empty()
            || metadata_device_bytes == 0
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled device-resource store has an invalid capacity or identity",
            ));
        }
        let contract_index = CompiledResourceContractIndex::new(&contract).map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource contract index is invalid: {error}"
            ))
        })?;
        let upload_alignment = compiled_resource_upload_alignment(&contract, device)?;
        let store_residency = plan_compiled_resource_store_residency_for_ownership(
            &contract,
            &layout,
            &selector_ownership,
            maximum_group_byte_count,
            upload_alignment,
        )
        .map_err(|error| VulkanCompiledResourceDeviceStoreError::new(error.to_string()))?;
        if metadata_device_bytes != store_residency.metadata_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled device-resource metadata plan declares {metadata_device_bytes} bytes but the store requires {}",
                store_residency.metadata_device_bytes,
            )));
        }
        let maximum_load_wave_group_count = store_residency.maximum_load_wave_group_count;
        let maximum_addressable_resource_count = layout
            .addressable_slot_count_for_ownership(&selector_ownership)
            .map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource address layout is invalid: {error}"
                ))
            })?;
        if maximum_addressable_resource_count == 0
            || allowed_selector_ids.iter().any(|selector_id| {
                !layout
                    .selectors
                    .iter()
                    .any(|selector| selector.selector_id == *selector_id)
            })
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled device-resource store selector ownership is invalid",
            ));
        }
        let source_maximum_allocation_byte_capacity = maximum_dynamic_device_payload_bytes
            .checked_add(store_residency.maximum_dynamic_allocation_padding_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource allocation capacity overflowed",
                )
            })?;
        if source_maximum_allocation_byte_capacity > available_dynamic_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resources require up to {source_maximum_allocation_byte_capacity} physical allocation bytes for {maximum_dynamic_payload_bytes} payload bytes, but only {available_dynamic_device_bytes} device bytes are available"
            )));
        }
        let staging_byte_capacity = store_residency.transfer_staging_slot_byte_capacity;
        let mut transfer = device
            .create_resident_transfer_stream(
                store_residency.transfer_staging_slot_count,
                staging_byte_capacity,
            )
            .map_err(compiled_device_store_vulkan_error)?;
        let address_table =
            VulkanStableResourceAddressTable::new(device, &mut transfer, layout.slot_count())
                .map_err(compiled_device_store_vulkan_error)?;
        let sparse_group_layouts = compiled_resource_sparse_group_layouts(
            &contract,
            &contract_index,
            &layout,
            &selector_ownership,
            CompiledResourceRepresentation::Source,
        )?;
        let representation_group_layouts = compiled_resource_sparse_group_layouts(
            &contract,
            &contract_index,
            &layout,
            &selector_ownership,
            CompiledResourceRepresentation::ResidentDerivation,
        )?;
        let preferred_device_slab_byte_capacity =
            compiled_resource_stable_slab_payload_bytes(
                device,
                available_dynamic_device_bytes,
                store_residency.maximum_load_wave_payload_bytes,
                upload_alignment,
            )
            .map_err(|error| VulkanCompiledResourceDeviceStoreError::new(error.to_string()))?;
        let device_arena = VulkanStableResourceArena::new(
            device,
            VulkanStableResourceArenaConfig::new(available_dynamic_device_bytes, upload_alignment)
                .map_err(compiled_device_store_vulkan_error)?
                .with_preferred_slab_byte_capacity(preferred_device_slab_byte_capacity)
                .map_err(compiled_device_store_vulkan_error)?,
            &sparse_group_layouts,
        )
        .map_err(compiled_device_store_vulkan_error)?;
        let maximum_allocation_byte_capacity = available_dynamic_device_bytes;
        let representation_arena = if representation_group_layouts.is_empty() {
            None
        } else {
            Some(
                VulkanStableResourceArena::new(
                    device,
                    VulkanStableResourceArenaConfig::new(
                        available_dynamic_device_bytes,
                        upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?
                    .with_preferred_slab_byte_capacity(preferred_device_slab_byte_capacity)
                    .map_err(compiled_device_store_vulkan_error)?,
                    &representation_group_layouts,
                )
                .map_err(compiled_device_store_vulkan_error)?,
            )
        };
        let package_root = package_root.into();
        let selector_cache_policy = compiled_resource_selector_cache_policy(
            &contract,
            &selector_ownership,
            maximum_dynamic_payload_bytes,
        )?;
        let memory_plan = if maximum_dynamic_host_visible_payload_bytes == 0 {
            None
        } else {
            Some(if residency_policy == ResourceResidencyPolicy::Eager {
                VulkanCompiledResourceMemoryPlan::exact_tiered(
                    &selector_cache_policy.group_payload_bytes,
                    maximum_dynamic_device_payload_bytes,
                    maximum_dynamic_host_visible_payload_bytes,
                )?
            } else {
                VulkanCompiledResourceMemoryPlan::dynamic_tiered(
                    &selector_cache_policy.group_payload_bytes,
                    maximum_dynamic_device_payload_bytes,
                    maximum_dynamic_host_visible_payload_bytes,
                )?
            })
        };
        let host_visible_arena = if memory_plan.is_some() {
            let physical_allocation_slack_per_group = device
                .addressable_resident_buffer_memory_requirement_bytes(1)
                .map_err(compiled_device_store_vulkan_error)?
                .checked_sub(1)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "Vulkan host-visible allocation requirement is smaller than its payload",
                    )
                })?;
            let host_visible_allocation_capacity = match &shared_host_cache {
                Some(cache) => cache.capacity_bytes(),
                None => maximum_dynamic_host_visible_payload_bytes
                    .checked_add(
                        maximum_addressable_resource_count
                            .checked_mul(upload_alignment.saturating_sub(1))
                            .ok_or_else(|| {
                                VulkanCompiledResourceDeviceStoreError::new(
                                    "compiled host-visible allocation-padding capacity overflowed",
                                )
                            })?,
                    )
                    .and_then(|capacity| {
                        selector_cache_policy
                            .group_payload_bytes
                            .len()
                            .checked_mul(physical_allocation_slack_per_group)
                            .and_then(|slack| capacity.checked_add(slack))
                    })
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled host-visible physical allocation capacity overflowed",
                        )
                    })?,
            };
            Some(
                VulkanStableResourceArena::new(
                    device,
                    VulkanStableResourceArenaConfig::new(
                        host_visible_allocation_capacity,
                        upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?
                    .with_preferred_slab_byte_capacity(
                        compiled_resource_stable_slab_payload_bytes(
                            device,
                            host_visible_allocation_capacity,
                            store_residency.maximum_load_wave_payload_bytes,
                            upload_alignment,
                        )
                        .map_err(|error| {
                            VulkanCompiledResourceDeviceStoreError::new(error.to_string())
                        })?,
                    )
                    .map_err(compiled_device_store_vulkan_error)?
                    .host_visible(),
                    &sparse_group_layouts,
                )
                .map_err(compiled_device_store_vulkan_error)?,
            )
        } else {
            None
        };
        let coverage_index = compiled_resource_component_coverage_index(
            &contract,
            &contract_index,
            &allowed_selector_ids,
        )?;
        let backing_store = CompiledResourceBackingStore::new(
            package_root.clone(),
            CompiledResourceBackingStoreLimits {
                worker_count: compiled_resource_backing_worker_count(maximum_load_wave_group_count),
                queued_request_capacity: maximum_load_wave_group_count,
                maximum_ranges_per_group,
                maximum_logical_bytes_per_group: maximum_group_byte_count,
                maximum_retained_payload_bytes: store_residency.maximum_load_wave_payload_bytes,
                maximum_coalesced_read_bytes: maximum_group_byte_count,
                maximum_coalescing_gap_bytes: 64 * 1024,
            },
        )
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to create compiled resource backing store: {error}"
            ))
        })?;
        let representation_backing_store = if representation_arena.is_some() {
            let maximum_resident_group_bytes = representation_group_layouts
                .iter()
                .map(compiled_resource_group_layout_payload_bytes)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation arena has no group payload",
                    )
                })?;
            Some(
                CompiledResourceBackingStore::new(
                    package_root.clone(),
                    CompiledResourceBackingStoreLimits {
                        worker_count: 1,
                        queued_request_capacity: 1,
                        maximum_ranges_per_group,
                        maximum_logical_bytes_per_group: maximum_group_byte_count,
                        maximum_retained_payload_bytes: maximum_resident_group_bytes
                            .max(maximum_group_byte_count),
                        maximum_coalesced_read_bytes: maximum_group_byte_count,
                        maximum_coalescing_gap_bytes: 64 * 1024,
                    },
                )
                .map_err(|error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "failed to create compiled representation backing store: {error}"
                    ))
                })?,
            )
        } else {
            None
        };
        let manager = DeviceResourceResidencyManager::new(
            device_id.clone(),
            maximum_dynamic_payload_bytes,
            0,
        )
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to create compiled resource residency manager: {error}"
            ))
        })?;
        Ok(Self {
            residency_policy,
            device_id,
            physical_device_id,
            logical_device_ids,
            allowed_selector_ids,
            selector_ownership,
            package_root,
            contract,
            contract_index,
            layout,
            device_arena,
            host_visible_arena,
            representation_arena,
            shared_host_cache,
            memory_plan: memory_plan.map(std::sync::Mutex::new),
            address_state: std::sync::Mutex::new(VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                publications: BTreeMap::new(),
                group_chunks: BTreeMap::new(),
                chunk_groups: BTreeMap::new(),
                group_blocks: BTreeMap::new(),
                block_groups: BTreeMap::new(),
                promoted_representations: BTreeMap::new(),
            }),
            residency_mutation: std::sync::Mutex::new(()),
            backing_store,
            representation_backing_store,
            manager,
            upload_alignment,
            maximum_dynamic_payload_bytes,
            maximum_dynamic_device_payload_bytes,
            maximum_dynamic_host_visible_payload_bytes,
            maximum_allocation_byte_capacity,
            maximum_group_byte_count,
            maximum_ranges_per_group,
            always_resident_parameter_bytes,
            runtime_working_set_device_bytes,
            metadata_device_bytes,
            transfer_staging_host_bytes: store_residency.transfer_staging_device_bytes,
            maximum_load_wave_group_count,
            group_selector_ids: selector_cache_policy.group_selector_ids,
            group_selections: selector_cache_policy.group_selections,
            group_payload_bytes: selector_cache_policy.group_payload_bytes,
            selector_payload_budgets: selector_cache_policy.selector_payload_budgets,
            retiering_last_selection_counts: std::sync::Mutex::new(BTreeMap::new()),
            representation_history: std::sync::Mutex::new(
                VulkanCompiledResourceRepresentationHistory::default(),
            ),
            coverage_index,
            instrumentation: VulkanCompiledResourceStoreInstrumentation::default(),
            lifecycle: std::sync::Mutex::new(VulkanCompiledResourceStoreLifecycle {
                state: VulkanCompiledResourceStoreLifecycleState::Active,
                active_load_operation_count: 0,
                memory_reclamation_in_progress: false,
                teardown_in_progress: false,
                terminal_failure: None,
                pending_release: DeviceResourceResidencyRelease::default(),
            }),
            lifecycle_changed: std::sync::Condvar::new(),
            memory_reclaimer_registration: std::sync::Mutex::new(None),
            #[cfg(test)]
            fail_next_teardown_before_address_clear: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            fail_next_upload_as_device_lost: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            fail_next_retiering_after_payload_exchange: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn physical_device_id(&self) -> &str {
        &self.physical_device_id
    }

    pub fn logical_device_ids(&self) -> &[String] {
        &self.logical_device_ids
    }

    pub fn register_device_memory_reclaimer(
        self: &Arc<Self>,
        device: &VulkanComputeDevice,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if !self.residency_policy.evicts_inactive_resources() {
            return Ok(());
        }
        let mut registration = self.memory_reclaimer_registration.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource memory reclaimer registration was poisoned",
            )
        })?;
        if registration.is_some() {
            return Ok(());
        }
        let reclaimer: Arc<dyn VulkanDeviceLocalMemoryReclaimer> =
            Arc::new(VulkanCompiledResourceDeviceMemoryReclaimer {
                store: Arc::downgrade(self),
            });
        *registration = Some(
            device
                .register_device_local_memory_reclaimer(reclaimer)
                .map_err(compiled_device_store_vulkan_error)?,
        );
        Ok(())
    }

    fn supports_adaptive_retiering(&self) -> bool {
        self.memory_plan.is_some()
    }

    pub fn allowed_selector_ids(&self) -> &BTreeSet<String> {
        &self.allowed_selector_ids
    }

    fn maximum_load_wave_group_count(&self) -> usize {
        self.maximum_load_wave_group_count
    }

    #[allow(clippy::too_many_arguments)]
    fn is_compatible_with_mount(
        &self,
        residency_policy: ResourceResidencyPolicy,
        device_id: &str,
        physical_device_id: &str,
        logical_device_ids: &[String],
        package_root: &Path,
        contract: &CompiledResourceResidencyContract,
        layout: &VulkanCompiledResourceAddressLayout,
        selector_ownership: &VulkanCompiledResourceSelectorOwnership,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
        always_resident_parameter_bytes: usize,
        runtime_working_set_device_bytes: usize,
        metadata_device_bytes: usize,
    ) -> Result<bool, VulkanCompiledResourceDeviceStoreError> {
        let lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned while validating remount reuse",
            )
        })?;
        Ok(lifecycle.state == VulkanCompiledResourceStoreLifecycleState::Active
            && self.residency_policy == residency_policy
            && self.device_id == device_id
            && self.physical_device_id == physical_device_id
            && self.logical_device_ids == logical_device_ids
            && self.package_root == package_root
            && self.contract.as_ref() == contract
            && self.layout.as_ref() == layout
            && self.selector_ownership == *selector_ownership
            && self.maximum_group_byte_count == maximum_group_byte_count
            && self.maximum_ranges_per_group == maximum_ranges_per_group
            && self.always_resident_parameter_bytes == always_resident_parameter_bytes
            && self.runtime_working_set_device_bytes == runtime_working_set_device_bytes
            && self.metadata_device_bytes == metadata_device_bytes)
    }

    fn retained_mount_capacities(&self) -> VulkanRetainedCompiledResourceStoreCapacities {
        VulkanRetainedCompiledResourceStoreCapacities {
            store_payload_bytes: self.maximum_dynamic_payload_bytes,
            device_payload_bytes: self.maximum_dynamic_device_payload_bytes,
            host_visible_payload_bytes: self.maximum_dynamic_host_visible_payload_bytes,
            available_dynamic_device_bytes: self.maximum_allocation_byte_capacity,
        }
    }

    fn shared_host_cache_for_remount(
        &self,
    ) -> Option<Arc<VulkanCompiledResourceSharedHostCache>> {
        self.shared_host_cache.clone()
    }

    pub fn dynamic_buffers_for_components(
        &self,
        device: &VulkanComputeDevice,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
    ) -> Result<Arc<VulkanDynamicResourceBuffers>, VulkanCompiledResourceDeviceStoreError> {
        let state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        VulkanDynamicResourceBuffers::from_layout_for_components(
            device,
            &state.address_table,
            &self.layout,
            Some(execution_scope),
            component_ids,
        )
        .map(Arc::new)
        .map_err(compiled_device_store_vulkan_error)
    }

    pub(crate) fn owned_selector_resource_indices(
        &self,
        selector_id: &str,
    ) -> Option<&BTreeSet<usize>> {
        self.selector_ownership.resources(selector_id)
    }

    pub(crate) fn residency_policy(&self) -> ResourceResidencyPolicy {
        self.residency_policy
    }

    pub fn load_selector_resource(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_index: usize,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        self.load_selector_resources(device, selector_id, &[resource_index], owner)
            .map(|_| ())
    }

    pub fn load_selector_resources(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let _load = self.begin_load_operation()?;
        self.load_selector_resources_while_active(device, selector_id, resource_indices, owner)
    }

    fn load_selector_resources_while_active(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        // Shared-cache mutation must precede every individual store mutation.
        // A reservation may reclaim another store, so reversing this order
        // allows two simultaneous loads to wait on each other's store lock.
        let shared_host_mutation = self
            .shared_host_cache
            .as_ref()
            .map(|cache| cache.begin_mutation())
            .transpose()?;
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned",
            )
        })?;
        self.load_selector_resources_while_active_locked(
            device,
            selector_id,
            resource_indices,
            owner,
            shared_host_mutation.as_ref(),
        )
    }

    fn load_selector_resources_for_resume(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        device
            .restore_device_local_memory_headroom_after_quiescence()
            .map_err(compiled_device_store_vulkan_error)?;
        let _load = self.begin_load_operation()?;
        let shared_host_mutation = self
            .shared_host_cache
            .as_ref()
            .map(|cache| cache.begin_mutation())
            .transpose()?;
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned",
            )
        })?;
        let loaded = self.load_selector_resources_while_active_locked(
            device,
            selector_id,
            resource_indices,
            owner,
            shared_host_mutation.as_ref(),
        )?;
        Ok(loaded)
    }

    fn load_selector_resources_while_active_locked(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
        shared_host_mutation: Option<&VulkanCompiledResourceSharedHostCacheMutation<'_>>,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        if resource_indices.is_empty() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource load batch is empty",
            ));
        }
        let mut plans_by_group = BTreeMap::new();
        for resource_index in resource_indices {
            let resolved = self.resolve_selector_resource(selector_id, *resource_index)?;
            let descriptor =
                DeviceResourceGroupDescriptor::from_resolved(&resolved).map_err(|error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource descriptor is invalid: {error}"
                    ))
                })?;
            let resource_slots = self
                .layout
                .resource_slots_for_selection(
                    selector_id,
                    *resource_index,
                    &descriptor.resource_ids,
                )
                .map_err(|error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource address layout is invalid: {error}"
                    ))
                })?;
            let plan = VulkanCompiledResourceLoadPlan {
                descriptor: descriptor.clone(),
                resolved,
                resource_slots: resource_slots.clone(),
            };
            if let Some(existing) = plans_by_group.insert(descriptor.id.clone(), plan)
                && (existing.descriptor != descriptor || existing.resource_slots != resource_slots)
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "one compiled resource load batch maps a content identity to conflicting resources or address slots",
                ));
            }
        }
        let plans = plans_by_group.into_values().collect::<Vec<_>>();
        let protected_group_ids = plans
            .iter()
            .map(|plan| plan.descriptor.id.clone())
            .collect::<BTreeSet<_>>();
        for wave in plans.chunks(self.maximum_load_wave_group_count) {
            self.load_compiled_resource_wave(
                device,
                selector_id,
                wave,
                &protected_group_ids,
                owner.clone(),
                shared_host_mutation,
            )?;
        }
        Ok(plans.len())
    }


    pub fn record_gpu_gate_misses(
        &self,
        selector_id: &str,
        miss_count: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let selector = self
            .contract_index
            .selector(&self.contract, selector_id)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource GPU-miss report references unknown selector {selector_id:?}",
                ))
            })?;
        if !self.allowed_selector_ids.contains(selector_id) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource GPU-miss report references selector {selector_id:?} outside store {:?}",
                self.device_id,
            )));
        }
        self.instrumentation.record_gpu_gate_misses(
            &selector.execution_scope,
            &selector.component_id,
            miss_count,
        )
    }

    pub fn load_all_for_components(
        &self,
        device: &VulkanComputeDevice,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let _load = self.begin_load_operation()?;
        let selected = self
            .contract
            .selectors
            .iter()
            .filter(|selector| {
                selector.execution_scope == execution_scope
                    && component_ids.contains(&selector.component_id)
                    && self.allowed_selector_ids.contains(&selector.id)
            })
            .flat_map(|selector| {
                self.selector_ownership
                    .resources(&selector.id)
                    .into_iter()
                    .flatten()
                    .copied()
                    .map(move |resource_index| (selector.id.clone(), resource_index))
            })
            .collect::<Vec<_>>();
        let selected = self.unique_selector_resources(selected)?;
        let mut resources_by_selector = BTreeMap::<String, Vec<usize>>::new();
        for (selector_id, resource_index) in &selected {
            resources_by_selector
                .entry(selector_id.clone())
                .or_default()
                .push(*resource_index);
        }
        for (selector_id, resource_indices) in resources_by_selector {
            self.load_selector_resources_while_active(
                device,
                &selector_id,
                &resource_indices,
                owner.clone(),
            )?;
        }
        Ok(selected.len())
    }

    pub fn load_all_allowed(
        &self,
        device: &VulkanComputeDevice,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let _load = self.begin_load_operation()?;
        let selected = self
            .contract
            .selectors
            .iter()
            .filter(|selector| self.allowed_selector_ids.contains(&selector.id))
            .flat_map(|selector| {
                self.selector_ownership
                    .resources(&selector.id)
                    .into_iter()
                    .flatten()
                    .copied()
                    .map(move |resource_index| (selector.id.clone(), resource_index))
            })
            .collect::<Vec<_>>();
        let selected = self.unique_selector_resources(selected)?;
        let mut resources_by_selector = BTreeMap::<String, Vec<usize>>::new();
        for (selector_id, resource_index) in &selected {
            resources_by_selector
                .entry(selector_id.clone())
                .or_default()
                .push(*resource_index);
        }
        for (selector_id, resource_indices) in resources_by_selector {
            self.load_selector_resources_while_active(
                device,
                &selector_id,
                &resource_indices,
                owner.clone(),
            )?;
        }
        Ok(selected.len())
    }

    fn unique_selector_resources(
        &self,
        candidates: Vec<(String, usize)>,
    ) -> Result<Vec<(String, usize)>, VulkanCompiledResourceDeviceStoreError> {
        let mut selected_by_group = BTreeMap::new();
        for (selector_id, resource_index) in candidates {
            let group = self.resolve_selector_resource(&selector_id, resource_index)?;
            selected_by_group
                .entry(group.id().to_string())
                .or_insert((selector_id, resource_index));
        }
        Ok(selected_by_group.into_values().collect())
    }

    pub fn statistics(
        &self,
    ) -> Result<DeviceResourceResidencyStatistics, VulkanCompiledResourceDeviceStoreError> {
        self.manager
            .statistics()
            .map_err(compiled_device_store_residency_error)
    }

    pub fn mark_mount_complete(&self) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let residency = self.statistics()?;
        let arena = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        let representation_committed_bytes = self
            .representation_arena
            .as_ref()
            .map(VulkanStableResourceArena::stats)
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .map(|stats| stats.committed_byte_capacity)
            .unwrap_or_default();
        self.instrumentation.mark_mount_complete(
            residency.dynamic_resident_bytes,
            residency.resident_group_count,
            arena
                .committed_byte_capacity
                .checked_add(representation_committed_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource initial committed device byte count overflowed",
                    )
                })?,
        );
        Ok(())
    }

    pub fn residency_report(
        &self,
    ) -> Result<VulkanCompiledResourceStoreReport, VulkanCompiledResourceDeviceStoreError> {
        use std::sync::atomic::Ordering;

        let _address_state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let snapshot = self
            .manager
            .snapshot()
            .map_err(compiled_device_store_residency_error)?;
        let residency = snapshot.statistics;
        let directory = snapshot.directory;
        let resident_group_ids = directory
            .iter()
            .filter(|entry| entry.state == ResourceResidencyState::Resident)
            .map(|entry| entry.group_id.as_str())
            .collect::<BTreeSet<_>>();
        let arena = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        let representation_arena = self
            .representation_arena
            .as_ref()
            .map(VulkanStableResourceArena::stats)
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .unwrap_or_default();
        let current_committed_device_bytes = arena
            .committed_byte_capacity
            .checked_add(representation_arena.committed_byte_capacity)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource current committed device byte count overflowed",
                )
            })?;
        self.instrumentation
            .high_water_committed_device_bytes
            .fetch_max(
                u64::try_from(current_committed_device_bytes).unwrap_or(u64::MAX),
                Ordering::AcqRel,
            );
        let backing = self.backing_store.statistics();
        let gpu_misses_by_component = self.instrumentation.gpu_misses_by_component()?;
        let mut components = self
            .coverage_index
            .iter()
            .map(|coverage| VulkanCompiledResourceComponentCoverageReport {
                execution_scope: coverage.execution_scope.clone(),
                component_id: coverage.component_id.clone(),
                addressable_unit_count: coverage.group_ids.len(),
                resident_unit_count: coverage
                    .group_ids
                    .iter()
                    .filter(|group_id| resident_group_ids.contains(group_id.as_str()))
                    .count(),
                gpu_selection_count: 0,
                gpu_resident_hit_count: 0,
                gpu_miss_count: gpu_misses_by_component
                    .get(&(
                        coverage.execution_scope.clone(),
                        coverage.component_id.clone(),
                    ))
                    .copied()
                    .unwrap_or_default(),
            })
            .collect::<Vec<_>>();
        components.sort_by(|left, right| {
            (left.execution_scope.as_str(), left.component_id.as_str())
                .cmp(&(right.execution_scope.as_str(), right.component_id.as_str()))
        });
        let mut scope_group_ids = BTreeMap::<String, BTreeSet<String>>::new();
        for coverage in &self.coverage_index {
            scope_group_ids
                .entry(coverage.execution_scope.clone())
                .or_default()
                .extend(coverage.group_ids.iter().cloned());
        }
        let scopes = scope_group_ids
            .into_iter()
            .map(|(execution_scope, group_ids)| {
                Ok(VulkanCompiledResourceScopeCoverageReport {
                    component_count: components
                        .iter()
                        .filter(|component| component.execution_scope == execution_scope)
                        .count(),
                    addressable_unit_count: group_ids.len(),
                    resident_unit_count: group_ids
                        .iter()
                        .filter(|group_id| resident_group_ids.contains(group_id.as_str()))
                        .count(),
                    gpu_selection_count: 0,
                    gpu_resident_hit_count: 0,
                    gpu_miss_count: components
                        .iter()
                        .filter(|component| component.execution_scope == execution_scope)
                        .map(|component| component.gpu_miss_count)
                        .try_fold(0u64, |total, count| {
                            total.checked_add(count).ok_or_else(|| {
                                VulkanCompiledResourceDeviceStoreError::new(
                                    "compiled resource scope GPU-miss count overflowed",
                                )
                            })
                        })?,
                    execution_scope,
                })
            })
            .collect::<Result<Vec<_>, VulkanCompiledResourceDeviceStoreError>>()?;
        let addressable_unit_count = self
            .coverage_index
            .iter()
            .flat_map(|coverage| coverage.group_ids.iter())
            .collect::<BTreeSet<_>>()
            .len();
        let initial_committed_device_bytes = usize::try_from(
            self.instrumentation
                .initial_committed_device_bytes
                .load(Ordering::Acquire),
        )
        .unwrap_or(usize::MAX);
        let high_water_committed_device_bytes = usize::try_from(
            self.instrumentation
                .high_water_committed_device_bytes
                .load(Ordering::Acquire),
        )
        .unwrap_or(usize::MAX);
        let fixed_device_bytes = self
            .always_resident_parameter_bytes
            .checked_add(self.runtime_working_set_device_bytes)
            .and_then(|bytes| bytes.checked_add(self.metadata_device_bytes))
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource fixed device byte report overflowed",
                )
            })?;
        let total_device_bytes = |dynamic_bytes: usize, label: &str| {
            fixed_device_bytes
                .checked_add(dynamic_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource {label} device byte report overflowed"
                    ))
                })
        };
        let (
            device_tier_payload_bytes,
            host_visible_tier_payload_bytes,
            maximum_device_tier_payload_bytes,
            maximum_host_visible_tier_payload_bytes,
        ) = match &self.memory_plan {
            Some(memory_plan) => {
                let memory_plan = memory_plan.lock().map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned while reporting",
                    )
                })?;
                (
                    memory_plan.device_payload_bytes,
                    memory_plan.host_visible_payload_bytes,
                    memory_plan.device_payload_capacity,
                    memory_plan.host_visible_payload_capacity,
                )
            }
            None => (
                residency.dynamic_resident_bytes,
                0,
                self.maximum_dynamic_payload_bytes,
                0,
            ),
        };
        if device_tier_payload_bytes
            .checked_add(host_visible_tier_payload_bytes)
            != Some(residency.dynamic_resident_bytes)
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource tier payload accounting differs from resident payload bytes",
            ));
        }
        let (
            shared_host_cache_id,
            shared_host_cache_store_committed_bytes,
            shared_host_cache_committed_bytes,
            shared_host_cache_capacity_bytes,
        ) = match &self.shared_host_cache {
            Some(cache) => {
                let snapshot = cache.snapshot()?;
                (
                    Some(cache.cache_id().to_string()),
                    snapshot
                        .committed_bytes_by_store
                        .get(&self.device_id)
                        .copied()
                        .unwrap_or_default(),
                    snapshot.committed_bytes,
                    snapshot.capacity_bytes,
                )
            }
            None => (None, 0, 0, 0),
        };
        Ok(VulkanCompiledResourceStoreReport {
            store_id: self.device_id.clone(),
            physical_device_id: self.physical_device_id.clone(),
            logical_device_ids: self.logical_device_ids.clone(),
            initial_device_bytes: total_device_bytes(initial_committed_device_bytes, "initial")?,
            current_device_bytes: total_device_bytes(current_committed_device_bytes, "current")?,
            maximum_device_bytes: total_device_bytes(
                self.maximum_allocation_byte_capacity,
                "maximum",
            )?,
            high_water_device_bytes: total_device_bytes(
                high_water_committed_device_bytes,
                "high-water",
            )?,
            always_resident_parameter_bytes: self.always_resident_parameter_bytes,
            runtime_working_set_device_bytes: self.runtime_working_set_device_bytes,
            metadata_device_bytes: self.metadata_device_bytes,
            transfer_staging_host_bytes: self.transfer_staging_host_bytes,
            initial_payload_bytes: usize::try_from(
                self.instrumentation
                    .initial_payload_bytes
                    .load(Ordering::Acquire),
            )
            .unwrap_or(usize::MAX),
            current_payload_bytes: residency.dynamic_resident_bytes,
            maximum_payload_bytes: self.maximum_dynamic_payload_bytes,
            high_water_payload_bytes: residency.high_water_dynamic_resident_bytes,
            device_tier_payload_bytes,
            host_visible_tier_payload_bytes,
            maximum_device_tier_payload_bytes,
            maximum_host_visible_tier_payload_bytes,
            shared_host_cache_id,
            shared_host_cache_store_committed_bytes,
            shared_host_cache_committed_bytes,
            shared_host_cache_capacity_bytes,
            addressable_unit_count,
            initial_resident_unit_count: usize::try_from(
                self.instrumentation
                    .initial_resident_unit_count
                    .load(Ordering::Acquire),
            )
            .unwrap_or(usize::MAX),
            resident_unit_count: residency.resident_group_count,
            high_water_resident_unit_count: residency.high_water_resident_group_count,
            loading_unit_count: residency.loading_group_count,
            failed_unit_count: residency.failed_group_count,
            gpu_selection_count: 0,
            gpu_resident_hit_count: 0,
            gpu_miss_count: gpu_misses_by_component
                .values()
                .try_fold(0u64, |total, count| {
                    total.checked_add(*count).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource store GPU-miss count overflowed",
                        )
                    })
                })?,
            residency_directory_hit_count: residency.hit_count,
            residency_load_required_count: residency.miss_count,
            deduplicated_load_count: residency.single_flight_join_count,
            successful_load_count: residency.successful_load_count,
            failed_load_count: residency.failed_load_count,
            cancelled_load_count: residency.cancelled_load_count,
            eviction_count: residency.eviction_count,
            evicted_unit_count: residency.evicted_group_count,
            evicted_payload_bytes: residency.evicted_byte_count,
            released_device_bytes: self
                .instrumentation
                .released_device_bytes
                .load(Ordering::Relaxed),
            reload_count: residency.reload_count,
            logical_read_count: backing.logical_ranges,
            physical_read_count: backing.physical_reads,
            logical_bytes_read: backing.logical_bytes,
            physical_bytes_read: backing.physical_bytes,
            resident_bytes_produced: backing.resident_bytes,
            uploaded_bytes: self.instrumentation.uploaded_bytes.load(Ordering::Relaxed),
            read_time_ns: backing.read_time_ns,
            derivation_time_ns: backing.derivation_time_ns,
            upload_time_ns: self.instrumentation.upload_time_ns.load(Ordering::Relaxed),
            blocking_time_ns: self
                .instrumentation
                .blocking_time_ns
                .load(Ordering::Relaxed),
            retiering_event_count: self
                .instrumentation
                .retiering_event_count
                .load(Ordering::Relaxed),
            retiering_promoted_group_count: self
                .instrumentation
                .retiering_promoted_group_count
                .load(Ordering::Relaxed),
            retiering_promoted_payload_bytes: self
                .instrumentation
                .retiering_promoted_payload_bytes
                .load(Ordering::Relaxed),
            retiering_copied_payload_bytes: self
                .instrumentation
                .retiering_copied_payload_bytes
                .load(Ordering::Relaxed),
            retiering_device_selection_count: self
                .instrumentation
                .retiering_device_selection_count
                .load(Ordering::Relaxed),
            retiering_host_visible_selection_count: self
                .instrumentation
                .retiering_host_visible_selection_count
                .load(Ordering::Relaxed),
            retiering_time_ns: self
                .instrumentation
                .retiering_time_ns
                .load(Ordering::Relaxed),
            scopes,
            components,
        })
    }

    fn unload(
        &self,
    ) -> Result<DeviceResourceResidencyRelease, VulkanCompiledResourceDeviceStoreError> {
        if !self.begin_teardown_attempt()? {
            return Ok(DeviceResourceResidencyRelease::default());
        }
        let teardown = self.teardown_after_quiescence();
        match teardown {
            Ok(()) => self.finish_teardown_attempt(),
            Err(error) => {
                let cleanup = self.fail_teardown_attempt();
                match cleanup {
                    Ok(()) => Err(error),
                    Err(cleanup_error) => Err(VulkanCompiledResourceDeviceStoreError::new(
                        format!("{error}; teardown lifecycle cleanup also failed: {cleanup_error}"),
                    )),
                }
            }
        }
    }

    fn begin_teardown_attempt(&self) -> Result<bool, VulkanCompiledResourceDeviceStoreError> {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        loop {
            while lifecycle.memory_reclamation_in_progress {
                lifecycle = self.lifecycle_changed.wait(lifecycle).map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource store lifecycle was poisoned while waiting for memory reclamation",
                    )
                })?;
            }
            match lifecycle.state {
                VulkanCompiledResourceStoreLifecycleState::Unloaded => {
                    return Ok(false);
                }
                VulkanCompiledResourceStoreLifecycleState::Active => {
                    lifecycle.state = VulkanCompiledResourceStoreLifecycleState::Quiescing;
                    lifecycle.teardown_in_progress = true;
                    break;
                }
                VulkanCompiledResourceStoreLifecycleState::Failed => {
                    lifecycle.state = VulkanCompiledResourceStoreLifecycleState::Quiescing;
                    lifecycle.teardown_in_progress = true;
                    break;
                }
                VulkanCompiledResourceStoreLifecycleState::Quiescing
                    if !lifecycle.teardown_in_progress =>
                {
                    lifecycle.teardown_in_progress = true;
                    break;
                }
                VulkanCompiledResourceStoreLifecycleState::Quiescing => {
                    lifecycle = self
                        .lifecycle_changed
                        .wait(lifecycle)
                        .map_err(|_| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled resource store lifecycle was poisoned while waiting for teardown",
                            )
                        })?;
                }
            }
        }
        while lifecycle.active_load_operation_count != 0 {
            lifecycle = self.lifecycle_changed.wait(lifecycle).map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource store lifecycle was poisoned while quiescing",
                )
            })?;
        }
        Ok(true)
    }

    fn teardown_after_quiescence(&self) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let shared_host_mutation = self
            .shared_host_cache
            .as_ref()
            .map(|cache| cache.begin_mutation())
            .transpose()?;
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned during teardown",
            )
        })?;
        let release = self
            .manager
            .unload_device()
            .map_err(compiled_device_store_residency_error)?;
        {
            let mut lifecycle = self.lifecycle.lock().map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource store lifecycle was poisoned",
                )
            })?;
            let pending_release = DeviceResourceResidencyRelease {
                group_count: lifecycle
                    .pending_release
                    .group_count
                    .checked_add(release.group_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource teardown group count overflowed",
                        )
                    })?,
                byte_count: lifecycle
                    .pending_release
                    .byte_count
                    .checked_add(release.byte_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource teardown byte count overflowed",
                        )
                    })?,
                cancelled_load_count: lifecycle
                    .pending_release
                    .cancelled_load_count
                    .checked_add(release.cancelled_load_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource teardown cancellation count overflowed",
                        )
                    })?,
            };
            lifecycle.pending_release = pending_release;
        }
        #[cfg(test)]
        if self
            .fail_next_teardown_before_address_clear
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "injected compiled resource teardown failure before address clear",
            ));
        }
        let mut state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let publications = state
            .publications
            .values()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        if !publications.is_empty() {
            let VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                ..
            } = &mut *state;
            address_table
                .clear_group(transfer, &publications)
                .map_err(compiled_device_store_vulkan_error)?;
            state.publications.clear();
            state.group_chunks.clear();
            state.chunk_groups.clear();
            state.group_blocks.clear();
            state.block_groups.clear();
        }
        state.promoted_representations.clear();
        drop(state);
        self.device_arena
            .release_backing()
            .map_err(compiled_device_store_vulkan_error)?;
        if let Some(arena) = &self.representation_arena {
            arena
                .release_backing()
                .map_err(compiled_device_store_vulkan_error)?;
        }
        if let Some(arena) = &self.host_visible_arena {
            let committed_bytes = arena
                .stats()
                .map_err(compiled_device_store_vulkan_error)?
                .committed_byte_capacity;
            arena
                .release_backing()
                .map_err(compiled_device_store_vulkan_error)?;
            if let Some(mutation) = &shared_host_mutation {
                mutation.release_store_capacity(&self.device_id, committed_bytes)?;
            }
        }
        let residency = self
            .manager
            .snapshot()
            .map_err(compiled_device_store_residency_error)?;
        let arena = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        let host_visible_arena = self
            .host_visible_arena
            .as_ref()
            .map(VulkanStableResourceArena::stats)
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .unwrap_or_default();
        let representation_arena = self
            .representation_arena
            .as_ref()
            .map(VulkanStableResourceArena::stats)
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .unwrap_or_default();
        if !residency.directory.is_empty()
            || residency.statistics.dynamic_resident_bytes != 0
            || residency.statistics.reserved_loading_bytes != 0
            || residency.statistics.loading_group_count != 0
            || residency.statistics.resident_group_count != 0
            || residency.statistics.failed_group_count != 0
            || arena.active_allocation_count != 0
            || arena.allocated_byte_count != 0
            || arena.committed_byte_capacity != 0
            || arena.chunk_count != 0
            || host_visible_arena.active_allocation_count != 0
            || host_visible_arena.allocated_byte_count != 0
            || host_visible_arena.committed_byte_capacity != 0
            || host_visible_arena.chunk_count != 0
            || representation_arena.active_allocation_count != 0
            || representation_arena.allocated_byte_count != 0
            || representation_arena.committed_byte_capacity != 0
            || representation_arena.chunk_count != 0
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource store teardown did not quiesce cleanly: directory={}, dynamic_bytes={}, reserved_bytes={}, loading={}, resident={}, failed={}, device_arena_allocations={}, device_arena_bytes={}, device_arena_committed={}, device_arena_chunks={}, host_arena_allocations={}, host_arena_bytes={}, host_arena_committed={}, host_arena_chunks={}, representation_arena_allocations={}, representation_arena_bytes={}, representation_arena_committed={}, representation_arena_chunks={}",
                residency.directory.len(),
                residency.statistics.dynamic_resident_bytes,
                residency.statistics.reserved_loading_bytes,
                residency.statistics.loading_group_count,
                residency.statistics.resident_group_count,
                residency.statistics.failed_group_count,
                arena.active_allocation_count,
                arena.allocated_byte_count,
                arena.committed_byte_capacity,
                arena.chunk_count,
                host_visible_arena.active_allocation_count,
                host_visible_arena.allocated_byte_count,
                host_visible_arena.committed_byte_capacity,
                host_visible_arena.chunk_count,
                representation_arena.active_allocation_count,
                representation_arena.allocated_byte_count,
                representation_arena.committed_byte_capacity,
                representation_arena.chunk_count,
            )));
        }
        if let Some(memory_plan) = &self.memory_plan {
            memory_plan
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned during teardown",
                    )
                })?
                .clear_dynamic_admissions()?;
        }
        Ok(())
    }

    fn finish_teardown_attempt(
        &self,
    ) -> Result<DeviceResourceResidencyRelease, VulkanCompiledResourceDeviceStoreError> {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        lifecycle.state = VulkanCompiledResourceStoreLifecycleState::Unloaded;
        lifecycle.teardown_in_progress = false;
        let release = std::mem::take(&mut lifecycle.pending_release);
        self.lifecycle_changed.notify_all();
        Ok(release)
    }

    fn fail_teardown_attempt(&self) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        lifecycle.teardown_in_progress = false;
        self.lifecycle_changed.notify_all();
        Ok(())
    }

    #[cfg(test)]
    fn inject_teardown_failure_before_address_clear(&self) {
        self.fail_next_teardown_before_address_clear
            .store(true, std::sync::atomic::Ordering::Release);
    }

    #[cfg(test)]
    fn inject_next_upload_as_device_lost(&self) {
        self.fail_next_upload_as_device_lost
            .store(true, std::sync::atomic::Ordering::Release);
    }

    #[cfg(test)]
    fn inject_retiering_failure_after_payload_exchange(&self) {
        self.fail_next_retiering_after_payload_exchange
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn ensure_device_work_is_available(
        &self,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        if lifecycle.state == VulkanCompiledResourceStoreLifecycleState::Failed {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                lifecycle
                    .terminal_failure
                    .clone()
                    .unwrap_or_else(|| "compiled resource device is unavailable".to_string()),
            ));
        }
        Ok(())
    }

    fn record_terminal_device_failure(
        &self,
        error: &VulkanError,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        if lifecycle.state == VulkanCompiledResourceStoreLifecycleState::Active {
            lifecycle.state = VulkanCompiledResourceStoreLifecycleState::Failed;
            lifecycle.terminal_failure = Some(format!(
                "compiled resource physical device {:?} is unavailable after a terminal Vulkan failure: {error}",
                self.physical_device_id,
            ));
            self.lifecycle_changed.notify_all();
        }
        Ok(())
    }

    fn begin_load_operation(
        &self,
    ) -> Result<VulkanCompiledResourceStoreLoadGuard<'_>, VulkanCompiledResourceDeviceStoreError>
    {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        while lifecycle.state == VulkanCompiledResourceStoreLifecycleState::Active
            && lifecycle.memory_reclamation_in_progress
        {
            lifecycle = self.lifecycle_changed.wait(lifecycle).map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource store lifecycle was poisoned while waiting for memory reclamation",
                )
            })?;
        }
        if lifecycle.state != VulkanCompiledResourceStoreLifecycleState::Active {
            let terminal_failure = lifecycle
                .terminal_failure
                .as_deref()
                .map(|failure| format!(": {failure}"))
                .unwrap_or_default();
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource store {:?} is {:?} and cannot accept new loads{terminal_failure}",
                self.device_id, lifecycle.state,
            )));
        }
        lifecycle.active_load_operation_count = lifecycle
            .active_load_operation_count
            .checked_add(1)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource active-load count overflowed",
                )
            })?;
        Ok(VulkanCompiledResourceStoreLoadGuard { store: self })
    }

    fn begin_memory_reclamation_boundary(
        &self,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let mut lifecycle = self.lifecycle.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource store lifecycle was poisoned",
            )
        })?;
        while lifecycle.state == VulkanCompiledResourceStoreLifecycleState::Active
            && lifecycle.memory_reclamation_in_progress
        {
            lifecycle = self.lifecycle_changed.wait(lifecycle).map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource store lifecycle was poisoned while serializing memory reclamation",
                )
            })?;
        }
        if lifecycle.state != VulkanCompiledResourceStoreLifecycleState::Active {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource store {:?} is {:?} and cannot begin memory reclamation",
                self.device_id, lifecycle.state,
            )));
        }
        lifecycle.memory_reclamation_in_progress = true;
        while lifecycle.active_load_operation_count != 0 {
            lifecycle = self.lifecycle_changed.wait(lifecycle).map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource store lifecycle was poisoned while draining loads for memory reclamation",
                )
            })?;
        }
        if lifecycle.state != VulkanCompiledResourceStoreLifecycleState::Active {
            lifecycle.memory_reclamation_in_progress = false;
            self.lifecycle_changed.notify_all();
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource store {:?} became {:?} while draining loads for memory reclamation",
                self.device_id, lifecycle.state,
            )));
        }
        Ok(())
    }

    fn finish_memory_reclamation_boundary(&self) {
        let Ok(mut lifecycle) = self.lifecycle.lock() else {
            return;
        };
        lifecycle.memory_reclamation_in_progress = false;
        self.lifecycle_changed.notify_all();
    }

    fn resolve_selector_resource(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Result<ResolvedCompiledResourceGroup, VulkanCompiledResourceDeviceStoreError> {
        let selector = self
            .contract_index
            .selector(&self.contract, selector_id)
            .filter(|selector| self.allowed_selector_ids.contains(&selector.id))
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource selector {selector_id:?} is unknown"
                ))
            })?;
        if resource_index >= selector.resource_count {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource selector {selector_id:?} index {resource_index} exceeds {}",
                selector.resource_count
            )));
        }
        if !self.selector_ownership.owns(selector_id, resource_index) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource selector {selector_id:?} index {resource_index} is outside store {:?} ownership",
                self.device_id
            )));
        }
        let resolved = match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                self.contract_index
                    .resolve_atomic_group(&self.contract, &atomic_group_ids[resource_index])
                    .map(ResolvedCompiledResourceGroup::Atomic)
            }
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => resolve_compiled_partition_group(
                &self.package_root,
                &self.contract,
                partition_template_id,
                resource_index,
            )
            .map(ResolvedCompiledResourceGroup::Partition),
        }
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to resolve compiled resource selection: {error}"
            ))
        })?;
        self.selector_ownership
            .project_resolved_group(selector_id, resource_index, resolved)
            .map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "failed to project compiled resource selection: {error}",
                ))
            })
    }
}

impl Drop for VulkanCompiledResourceStoreLoadGuard<'_> {
    fn drop(&mut self) {
        let Ok(mut lifecycle) = self.store.lifecycle.lock() else {
            return;
        };
        if lifecycle.active_load_operation_count == 0 {
            debug_assert!(
                false,
                "compiled resource load guard released without an active load"
            );
        } else {
            lifecycle.active_load_operation_count -= 1;
        }
        self.store.lifecycle_changed.notify_all();
    }
}

impl Drop for VulkanCompiledResourceDeviceStore {
    fn drop(&mut self) {
        let _ = self.unload();
    }
}

fn compiled_resource_upload_alignment(
    contract: &CompiledResourceResidencyContract,
    device: &VulkanComputeDevice,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    let mut alignment = device
        .min_storage_buffer_offset_alignment()
        .max(std::mem::align_of::<u64>());
    for range_alignment in contract
        .resources
        .iter()
        .flat_map(|resource| resource.ranges.iter().map(|range| range.alignment_bytes))
        .chain(
            contract
                .partition_templates
                .iter()
                .flat_map(|template| &template.member_templates)
                .flat_map(|member| {
                    member
                        .range_templates
                        .iter()
                        .map(|range| range.alignment_bytes)
                }),
        )
    {
        alignment = alignment.max(range_alignment);
    }
    if !alignment.is_power_of_two() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource upload alignment {alignment} is not a power of two"
        )));
    }
    Ok(alignment)
}

fn compiled_resource_selector_cache_policy(
    contract: &CompiledResourceResidencyContract,
    selector_ownership: &VulkanCompiledResourceSelectorOwnership,
    maximum_dynamic_payload_bytes: usize,
) -> Result<VulkanCompiledResourceSelectorCachePolicy, VulkanCompiledResourceDeviceStoreError> {
    let resource_payload_bytes = contract
        .resources
        .iter()
        .map(|resource| {
            let byte_count = resource.source_byte_count().map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(error.to_string())
            })?;
            if byte_count == 0 {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled selector resource is empty",
                ));
            }
            Ok((resource.id.as_str(), byte_count))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let atomic_group_payload_bytes = contract
        .atomic_groups
        .iter()
        .map(|group| {
            let byte_count = group
                .resource_ids
                .iter()
                .try_fold(0usize, |total, resource_id| {
                    let resource_bytes = resource_payload_bytes
                        .get(resource_id.as_str())
                        .copied()
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled selector group references a missing resource",
                            )
                        })?;
                    total.checked_add(resource_bytes).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled selector group byte count overflowed",
                        )
                    })
                })?;
            Ok((group.id.as_str(), byte_count))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let partition_templates = contract
        .partition_templates
        .iter()
        .map(|template| (template.id.as_str(), template))
        .collect::<BTreeMap<_, _>>();

    let mut group_owners = BTreeMap::<String, BTreeSet<String>>::new();
    let mut group_selections = BTreeMap::<String, (String, usize)>::new();
    let mut group_payload_bytes = BTreeMap::<String, usize>::new();
    for selector in contract
        .selectors
        .iter()
        .filter(|selector| selector_ownership.resources(&selector.id).is_some())
    {
        let groups = match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                if atomic_group_ids.len() != selector.resource_count {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector group table length differs from its resource count",
                    ));
                }
                atomic_group_ids
                    .iter()
                    .map(|group_id| {
                        let byte_count = atomic_group_payload_bytes
                            .get(group_id.as_str())
                            .copied()
                            .ok_or_else(|| {
                                VulkanCompiledResourceDeviceStoreError::new(
                                    "compiled selector references a missing atomic group",
                                )
                            })?;
                        Ok((group_id.clone(), byte_count))
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => {
                let template = partition_templates
                    .get(partition_template_id.as_str())
                    .copied()
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled selector references a missing partition template",
                        )
                    })?;
                if template.partition_count != selector.resource_count {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector partition count differs from its resource count",
                    ));
                }
                let byte_count = template.member_templates.iter().try_fold(
                    0usize,
                    |group_total, member| {
                        let member_bytes = member.range_templates.iter().try_fold(
                            0usize,
                            |member_total, range| {
                                member_total.checked_add(range.byte_count).ok_or_else(|| {
                                    VulkanCompiledResourceDeviceStoreError::new(
                                        "compiled selector partition member byte count overflowed",
                                    )
                                })
                            },
                        )?;
                        group_total.checked_add(member_bytes).ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled selector partition group byte count overflowed",
                            )
                        })
                    },
                )?;
                (0..template.partition_count)
                    .map(|partition_index| {
                        derived_partition_resource_id(
                            &template.group_identity_seed,
                            partition_index,
                        )
                        .map(|group_id| (group_id, byte_count))
                        .map_err(|error| {
                            VulkanCompiledResourceDeviceStoreError::new(format!(
                                "failed to derive selector cache group identity: {error}",
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        for (resource_index, (group_id, byte_count)) in groups.into_iter().enumerate() {
            if !selector_ownership.owns(&selector.id, resource_index) {
                continue;
            }
            let byte_count = if let Some(projection) =
                selector_ownership.source_projection(&selector.id, resource_index)
            {
                projection.resources.values().try_fold(
                    0usize,
                    |total, resource| total.checked_add(resource.byte_count),
                ).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector projected group byte count overflowed",
                    )
                })?
            } else {
                byte_count
            };
            group_owners
                .entry(group_id.clone())
                .or_default()
                .insert(selector.id.clone());
            group_selections
                .entry(group_id.clone())
                .or_insert_with(|| (selector.id.clone(), resource_index));
            if let Some(previous) = group_payload_bytes.insert(group_id, byte_count)
                && previous != byte_count
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled selector group has conflicting payload byte counts",
                ));
            }
        }
    }
    if group_payload_bytes.is_empty() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled selectors have no addressable payload",
        ));
    }
    let mut group_selector_ids = BTreeMap::new();
    let mut selector_addressable_payload_bytes = selector_ownership
        .iter()
        .map(|(selector_id, _)| (selector_id.to_string(), 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut total_addressable_payload_bytes = 0usize;
    for (group_id, owners) in group_owners {
        let byte_count = group_payload_bytes.get(&group_id).copied().ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled selector group has no payload byte count",
            )
        })?;
        total_addressable_payload_bytes = total_addressable_payload_bytes
            .checked_add(byte_count)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled selector addressable payload byte count overflowed",
                )
            })?;
        if owners.len() == 1 {
            let selector_id = owners
                .into_iter()
                .next()
                .expect("one selector owner was established");
            let selector_bytes = selector_addressable_payload_bytes
                .get_mut(&selector_id)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector cache owner is outside the device store",
                    )
                })?;
            *selector_bytes = selector_bytes.checked_add(byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled selector cache payload byte count overflowed",
                )
            })?;
            group_selector_ids.insert(group_id, selector_id);
        }
    }
    let selector_payload_budgets = selector_addressable_payload_bytes
        .into_iter()
        .map(|(selector_id, addressable_bytes)| {
            let scaled = (maximum_dynamic_payload_bytes as u128)
                .checked_mul(addressable_bytes as u128)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector cache budget multiplication overflowed",
                    )
                })?
                / total_addressable_payload_bytes as u128;
            let budget = usize::try_from(scaled).map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled selector cache budget exceeds the host address space",
                )
            })?;
            Ok((selector_id, budget))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(VulkanCompiledResourceSelectorCachePolicy {
        group_selector_ids,
        group_selections,
        group_payload_bytes,
        selector_payload_budgets,
    })
}

fn compiled_resource_group_layout_payload_bytes(
    layout: &VulkanStableResourceGroupLayout,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    let resource_byte_counts = match layout {
        VulkanStableResourceGroupLayout::Explicit {
            resource_byte_counts,
            ..
        } => resource_byte_counts.as_slice(),
        VulkanStableResourceGroupLayout::Partitioned {
            resource_byte_counts,
            ..
        } => resource_byte_counts.as_slice(),
    };
    resource_byte_counts
        .iter()
        .try_fold(0usize, |total, byte_count| {
            total.checked_add(*byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled representation group payload byte count overflowed",
                )
            })
        })
}

fn compiled_resource_sparse_group_layouts(
    contract: &CompiledResourceResidencyContract,
    contract_index: &CompiledResourceContractIndex,
    layout: &VulkanCompiledResourceAddressLayout,
    selector_ownership: &VulkanCompiledResourceSelectorOwnership,
    representation: CompiledResourceRepresentation,
) -> Result<Vec<VulkanStableResourceGroupLayout>, VulkanCompiledResourceDeviceStoreError> {
    let resource_byte_count = |resource: &CompiledImmutableResource,
                               label: &str|
     -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let byte_count = resource.resident_byte_count_for(representation).map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "sparse {label} resource is invalid: {error}"
            ))
        })?;
        if byte_count == 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "sparse {label} resource is empty"
            )));
        }
        Ok(byte_count)
    };
    let mut explicit_groups = BTreeMap::new();
    let mut partitioned_groups = BTreeMap::new();
    for selector_layout in layout
        .selectors
        .iter()
        .filter(|selector| selector_ownership.resources(&selector.selector_id).is_some())
    {
        let selector = contract_index
            .selector(contract, &selector_layout.selector_id)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "sparse address layout references a missing selector",
                )
            })?;
        match (&selector.mapping, &selector_layout.mapping) {
            (
                CompiledResourceSelectorMapping::GroupTable { atomic_group_ids },
                VulkanCompiledSelectorAddressMapping::GroupTable { .. },
            ) => {
                for (resource_index, group_id) in atomic_group_ids.iter().enumerate() {
                    if !selector_ownership.owns(&selector.id, resource_index) {
                        continue;
                    }
                    let group = contract_index
                        .atomic_group(contract, group_id)
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "sparse selector references a missing atomic group",
                            )
                        })?;
                    let slots = selector_layout
                        .mapping
                        .resource_slots(resource_index)
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "sparse atomic group has no address slots",
                            )
                        })?;
                    if representation == CompiledResourceRepresentation::ResidentDerivation
                        && !group.resource_ids.iter().any(|resource_id| {
                            contract_index
                                .resource(contract, resource_id)
                                .is_some_and(|resource| {
                                    resource.supports_representation(representation)
                                })
                        })
                    {
                        continue;
                    }
                    let byte_counts = group
                        .resource_ids
                        .iter()
                        .map(|resource_id| {
                            if representation == CompiledResourceRepresentation::Source
                                && let Some(projected) = selector_ownership
                                    .source_projection(&selector.id, resource_index)
                                    .and_then(|projection| {
                                        projection.resources.get(resource_id)
                                    })
                            {
                                return Ok(projected.byte_count);
                            }
                            contract_index
                                .resource(contract, resource_id)
                                .ok_or_else(|| {
                                    VulkanCompiledResourceDeviceStoreError::new(
                                        "sparse atomic group references a missing resource",
                                    )
                                })
                                .and_then(|resource| resource_byte_count(resource, "concrete"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if let Some(previous) = explicit_groups.insert(slots, byte_counts.clone())
                        && previous != byte_counts
                    {
                        return Err(VulkanCompiledResourceDeviceStoreError::new(
                            "sparse atomic group has conflicting byte layouts",
                        ));
                    }
                }
            }
            (
                CompiledResourceSelectorMapping::PartitionTemplate {
                    partition_template_id,
                },
                VulkanCompiledSelectorAddressMapping::PartitionTemplate {
                    member_slot_bases,
                    resource_count,
                    ..
                },
            ) => {
                let template = contract_index
                    .partition_template(contract, partition_template_id)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "sparse selector references a missing partition template",
                        )
                    })?;
                if template.partition_count != *resource_count {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(
                        "sparse partition selector count differs from its address layout",
                    ));
                }
                if representation == CompiledResourceRepresentation::ResidentDerivation
                    && !template
                        .member_templates
                        .iter()
                        .any(|member| member.supports_representation(representation))
                {
                    continue;
                }
                let byte_counts = template
                    .member_templates
                    .iter()
                    .map(|member| {
                        let byte_count = member
                            .resident_byte_count_for(representation)
                            .map_err(|error| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                error.to_string(),
                            )
                        })?;
                        if byte_count == 0 {
                            return Err(VulkanCompiledResourceDeviceStoreError::new(
                                "sparse partition resource is empty",
                            ));
                        }
                        Ok(byte_count)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let key = (member_slot_bases.clone(), template.partition_count);
                if let Some(previous) = partitioned_groups.insert(key, byte_counts.clone())
                    && previous != byte_counts
                {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(
                        "sparse partition group has conflicting byte layouts",
                    ));
                }
            }
            _ => {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "sparse selector contract and address mapping differ",
                ));
            }
        }
    }
    if representation == CompiledResourceRepresentation::Source
        && explicit_groups.is_empty()
        && partitioned_groups.is_empty()
    {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource store has no sparse resource groups",
        ));
    }
    let mut groups = explicit_groups
        .into_iter()
        .map(
            |(resource_slots, resource_byte_counts)| VulkanStableResourceGroupLayout::Explicit {
                resource_slots,
                resource_byte_counts,
            },
        )
        .collect::<Vec<_>>();
    groups.extend(partitioned_groups.into_iter().map(
        |((member_slot_bases, partition_count), resource_byte_counts)| {
            VulkanStableResourceGroupLayout::Partitioned {
                member_slot_bases,
                resource_byte_counts,
                partition_count,
            }
        },
    ));
    Ok(groups)
}

fn compiled_resource_vulkan_error_is_device_loss(error: &VulkanError) -> bool {
    error.0.contains("ERROR_DEVICE_LOST")
}

fn compiled_resource_maximum_ranges_per_group(
    contract: &CompiledResourceResidencyContract,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    let contract_index = CompiledResourceContractIndex::new(contract).map_err(|error| {
        VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource contract index is invalid: {error}"
        ))
    })?;
    contract
        .atomic_groups
        .iter()
        .filter(|group| group.lifetime == CompiledResourceLifetime::Dynamic)
        .map(|group| {
            group
                .resource_ids
                .iter()
                .try_fold(0usize, |total, resource_id| {
                    let resource = contract_index
                        .resource(contract, resource_id)
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(format!(
                                "dynamic group {:?} references missing resource {resource_id:?}",
                                group.id
                            ))
                        })?;
                    total.checked_add(resource.ranges.len()).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource range count overflowed",
                        )
                    })
                })
        })
        .chain(contract.partition_templates.iter().map(|template| {
            template
                .member_templates
                .iter()
                .try_fold(0usize, |total, member| {
                    total
                        .checked_add(member.range_templates.len())
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled partition range count overflowed",
                            )
                        })
                })
        }))
        .try_fold(0usize, |maximum, count| {
            count.map(|count| maximum.max(count))
        })
}

fn compiled_resource_component_coverage_index(
    contract: &CompiledResourceResidencyContract,
    contract_index: &CompiledResourceContractIndex,
    selector_ids: &BTreeSet<String>,
) -> Result<Vec<VulkanCompiledResourceComponentCoverageIndex>, VulkanCompiledResourceDeviceStoreError>
{
    let mut indexed = BTreeMap::<(String, String), BTreeSet<String>>::new();
    for selector in contract
        .selectors
        .iter()
        .filter(|selector| selector_ids.contains(&selector.id))
    {
        let group_ids = match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                atomic_group_ids.to_vec()
            }
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => {
                let template = contract_index
                    .partition_template(contract, partition_template_id)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled selector {:?} references missing partition template {partition_template_id:?}",
                            selector.id
                        ))
                    })?;
                (0..template.partition_count)
                    .map(|partition_index| {
                        derived_partition_resource_id(
                            &template.group_identity_seed,
                            partition_index,
                        )
                        .map_err(|error| {
                            VulkanCompiledResourceDeviceStoreError::new(format!(
                                "failed to derive addressable group identity: {error}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        indexed
            .entry((
                selector.execution_scope.clone(),
                selector.component_id.clone(),
            ))
            .or_default()
            .extend(group_ids);
    }
    Ok(indexed
        .into_iter()
        .map(|((execution_scope, component_id), group_ids)| {
            VulkanCompiledResourceComponentCoverageIndex {
                execution_scope,
                component_id,
                group_ids,
            }
        })
        .collect())
}

fn compiled_device_store_vulkan_error(
    error: VulkanError,
) -> VulkanCompiledResourceDeviceStoreError {
    VulkanCompiledResourceDeviceStoreError::new(error.to_string())
}

fn compiled_device_store_residency_error(
    error: DeviceResourceResidencyError,
) -> VulkanCompiledResourceDeviceStoreError {
    VulkanCompiledResourceDeviceStoreError::new(error.to_string())
}
