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
}

fn compiled_resource_lru_eviction_selection(
    candidates: &[DeviceResourceResidencyEvictionCandidate],
    group_chunks: &BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    chunk_groups: &BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    chunk_byte_capacities: &BTreeMap<VulkanCompiledResourceAllocationCohort, usize>,
    protected_group_ids: &BTreeSet<String>,
    required_payload_bytes: usize,
    required_device_bytes: usize,
) -> Result<VulkanCompiledResourceEvictionSelection, VulkanCompiledResourceDeviceStoreError> {
    if required_payload_bytes == 0 && required_device_bytes == 0 {
        return Ok(VulkanCompiledResourceEvictionSelection {
            group_ids: BTreeSet::new(),
            payload_bytes: 0,
            device_bytes: 0,
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
        let contributes_payload = selected_payload_bytes < required_payload_bytes
            && cohort_payload_bytes > 0;
        let contributes_device = selected_device_bytes < required_device_bytes
            && cohort_device_bytes > 0;
        if !contributes_payload && !contributes_device {
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
        if selected_payload_bytes >= required_payload_bytes
            && selected_device_bytes >= required_device_bytes
        {
            break;
        }
    }
    Ok(VulkanCompiledResourceEvictionSelection {
        group_ids: selected,
        payload_bytes: selected_payload_bytes,
        device_bytes: selected_device_bytes,
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
    teardown_in_progress: bool,
    terminal_failure: Option<String>,
    pending_release: DeviceResourceResidencyRelease,
}

struct VulkanCompiledResourceStoreLoadGuard<'a> {
    store: &'a VulkanCompiledResourceDeviceStore,
}

struct VulkanCompiledResourceExecutionGuard<'a> {
    _guard: std::sync::RwLockReadGuard<'a, ()>,
}

struct VulkanCompiledResourceLoadPlan {
    descriptor: DeviceResourceGroupDescriptor,
    resolved: ResolvedCompiledResourceGroup,
    resource_slots: Vec<usize>,
}

struct VulkanCompiledResourceSelectorCachePolicy {
    group_selector_ids: BTreeMap<String, String>,
    group_payload_bytes: BTreeMap<String, usize>,
    selector_payload_budgets: BTreeMap<String, usize>,
}

struct VulkanCompiledResourceDeviceMemoryReclaimer {
    store: std::sync::Weak<VulkanCompiledResourceDeviceStore>,
}

impl std::fmt::Debug for VulkanCompiledResourceDeviceMemoryReclaimer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanCompiledResourceDeviceMemoryReclaimer")
            .finish_non_exhaustive()
    }
}

impl VulkanDeviceLocalMemoryReclaimer for VulkanCompiledResourceDeviceMemoryReclaimer {
    fn reclaim_device_local_memory(&self, requested_bytes: usize) -> Result<usize, VulkanError> {
        let Some(store) = self.store.upgrade() else {
            return Ok(0);
        };
        store
            .reclaim_inactive_device_memory(requested_bytes)
            .map_err(|error| VulkanError(error.to_string()))
    }
}

pub struct VulkanCompiledResourceDeviceStore {
    residency_policy: ResourceResidencyPolicy,
    device_id: String,
    physical_device_id: String,
    logical_device_ids: Vec<String>,
    allowed_selector_ids: BTreeSet<String>,
    package_root: PathBuf,
    contract: Arc<CompiledResourceResidencyContract>,
    contract_index: CompiledResourceContractIndex,
    layout: Arc<VulkanCompiledResourceAddressLayout>,
    device_arena: VulkanStableResourceArena,
    host_visible_arena: Option<VulkanStableResourceArena>,
    memory_plan: Option<std::sync::Mutex<VulkanCompiledResourceMemoryPlan>>,
    address_state: std::sync::Mutex<VulkanCompiledResourceDeviceAddressState>,
    execution_barrier: std::sync::RwLock<()>,
    residency_mutation: std::sync::Mutex<()>,
    backing_store: CompiledResourceBackingStore,
    manager: DeviceResourceResidencyManager<VulkanResidentCompiledResource>,
    upload_alignment: usize,
    maximum_dynamic_payload_bytes: usize,
    maximum_allocation_byte_capacity: usize,
    always_resident_parameter_bytes: usize,
    runtime_working_set_device_bytes: usize,
    metadata_device_bytes: usize,
    transfer_staging_host_bytes: usize,
    maximum_load_wave_group_count: usize,
    group_selector_ids: BTreeMap<String, String>,
    selector_payload_budgets: BTreeMap<String, usize>,
    retiering_selection_counts: std::sync::Mutex<BTreeMap<String, u64>>,
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
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_tiered(
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
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let device_id = device_id.into();
        let physical_device_id = physical_device_id.into();
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
        let maximum_load_wave_group_count = contract
            .selectors
            .iter()
            .filter(|selector| allowed_selector_ids.contains(&selector.id))
            .map(|selector| selector.encoding.selection_count_per_activation)
            .max()
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled device-resource store has no allowed selector load wave",
                )
            })?;
        if maximum_load_wave_group_count == 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled device-resource store selector load wave is empty",
            ));
        }
        let maximum_addressable_resource_count = layout
            .addressable_slot_count_for_selectors(&allowed_selector_ids)
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
        let per_resource_alignment_slack = upload_alignment.checked_sub(1).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource upload alignment underflowed",
            )
        })?;
        let maximum_allocation_byte_capacity = maximum_dynamic_device_payload_bytes
            .checked_add(
                maximum_addressable_resource_count
                    .checked_mul(per_resource_alignment_slack)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource allocation-padding capacity overflowed",
                        )
                    })?,
            )
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource allocation capacity overflowed",
                )
            })?;
        if maximum_allocation_byte_capacity > available_dynamic_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resources require up to {maximum_allocation_byte_capacity} physical allocation bytes for {maximum_dynamic_payload_bytes} payload bytes, but only {available_dynamic_device_bytes} device bytes are available"
            )));
        }
        let address_table_byte_count = layout.slot_count().checked_mul(32).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address-table capacity overflowed",
            )
        })?;
        let maximum_load_wave_payload_bytes = maximum_group_byte_count
            .checked_mul(maximum_load_wave_group_count)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource load-wave payload capacity overflowed",
                )
            })?;
        let staging_byte_capacity = maximum_load_wave_payload_bytes.max(address_table_byte_count);
        let mut transfer = device
            .create_resident_transfer_stream(2, staging_byte_capacity)
            .map_err(compiled_device_store_vulkan_error)?;
        let address_table =
            VulkanStableResourceAddressTable::new(device, &mut transfer, layout.slot_count())
                .map_err(compiled_device_store_vulkan_error)?;
        let sparse_group_layouts = compiled_resource_sparse_group_layouts(
            &contract,
            &contract_index,
            &layout,
            &allowed_selector_ids,
        )?;
        let device_arena = VulkanStableResourceArena::new(
            device,
            VulkanStableResourceArenaConfig::new(available_dynamic_device_bytes, upload_alignment)
                .map_err(compiled_device_store_vulkan_error)?,
            &sparse_group_layouts,
        )
        .map_err(compiled_device_store_vulkan_error)?;
        let maximum_allocation_byte_capacity = device_arena
            .maximum_backed_byte_capacity()
            .map_err(compiled_device_store_vulkan_error)?
            .min(available_dynamic_device_bytes);
        let package_root = package_root.into();
        let selector_cache_policy = compiled_resource_selector_cache_policy(
            &contract,
            &allowed_selector_ids,
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
            let host_visible_allocation_capacity = maximum_dynamic_host_visible_payload_bytes
                .checked_add(
                    maximum_addressable_resource_count
                        .checked_mul(per_resource_alignment_slack)
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled host-visible allocation-padding capacity overflowed",
                            )
                        })?,
                )
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled host-visible allocation capacity overflowed",
                    )
                })?;
            Some(
                VulkanStableResourceArena::new(
                    device,
                    VulkanStableResourceArenaConfig::new(
                        host_visible_allocation_capacity,
                        upload_alignment,
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
                maximum_retained_payload_bytes: maximum_load_wave_payload_bytes,
                maximum_coalesced_read_bytes: maximum_group_byte_count,
                maximum_coalescing_gap_bytes: 64 * 1024,
            },
        )
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to create compiled resource backing store: {error}"
            ))
        })?;
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
            package_root,
            contract,
            contract_index,
            layout,
            device_arena,
            host_visible_arena,
            memory_plan: memory_plan.map(std::sync::Mutex::new),
            address_state: std::sync::Mutex::new(VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                publications: BTreeMap::new(),
                group_chunks: BTreeMap::new(),
                chunk_groups: BTreeMap::new(),
            }),
            execution_barrier: std::sync::RwLock::new(()),
            residency_mutation: std::sync::Mutex::new(()),
            backing_store,
            manager,
            upload_alignment,
            maximum_dynamic_payload_bytes,
            maximum_allocation_byte_capacity,
            always_resident_parameter_bytes,
            runtime_working_set_device_bytes,
            metadata_device_bytes,
            transfer_staging_host_bytes: staging_byte_capacity.checked_mul(2).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource transfer staging byte count overflowed",
                )
            })?,
            maximum_load_wave_group_count,
            group_selector_ids: selector_cache_policy.group_selector_ids,
            selector_payload_budgets: selector_cache_policy.selector_payload_budgets,
            retiering_selection_counts: std::sync::Mutex::new(BTreeMap::new()),
            coverage_index,
            instrumentation: VulkanCompiledResourceStoreInstrumentation::default(),
            lifecycle: std::sync::Mutex::new(VulkanCompiledResourceStoreLifecycle {
                state: VulkanCompiledResourceStoreLifecycleState::Active,
                active_load_operation_count: 0,
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
        )
    }

    fn load_selector_resources_for_execution(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<
        (usize, VulkanCompiledResourceExecutionGuard<'_>),
        VulkanCompiledResourceDeviceStoreError,
    > {
        let _load = self.begin_load_operation()?;
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
        )?;
        let execution = self.begin_execution()?;
        Ok((loaded, execution))
    }

    fn load_selector_resources_while_active_locked(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
        owner: DeviceResourceResidencyOwnerId,
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
            )?;
        }
        Ok(plans.len())
    }

    fn load_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let device_capacity_permit = if self.residency_policy.evicts_inactive_resources() {
            self.evict_for_compiled_resource_wave(
                device,
                selector_id,
                plans,
                protected_group_ids,
            )?
        } else {
            None
        };
        let requests = self
            .manager
            .request_batch(plans.iter().map(|plan| plan.descriptor.clone()), owner)
            .map_err(compiled_device_store_residency_error)?;
        let mut pending = Vec::new();
        let mut required = Vec::new();
        for (plan, request) in plans.iter().zip(requests) {
            match request {
                DeviceResourceResidencyRequest::Resident(lease) => {
                    drop(lease);
                }
                DeviceResourceResidencyRequest::Pending(waiter) => {
                    pending.push(waiter);
                }
                DeviceResourceResidencyRequest::LoadRequired(permit) => {
                    required.push((plan, permit));
                }
            }
        }
        let _blocking = (!pending.is_empty() || !required.is_empty())
            .then(|| VulkanCompiledResourceBlockingTimer::new(&self.instrumentation));
        let mut submitted = Vec::with_capacity(required.len());
        for (plan, permit) in required {
            match self.backing_store.try_load(plan.resolved.clone()) {
                Ok(ticket) => submitted.push((plan, permit, ticket)),
                Err(error) => {
                    let message = format!("compiled resource backing-store load failed: {error}");
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                    return Err(VulkanCompiledResourceDeviceStoreError::new(message));
                }
            }
        }
        let mut loaded = Vec::with_capacity(submitted.len());
        for (plan, permit, ticket) in submitted {
            match ticket.wait() {
                Ok(group) => loaded.push((plan, permit, group)),
                Err(error) => {
                    let message = format!("compiled resource backing-store load failed: {error}");
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                    return Err(VulkanCompiledResourceDeviceStoreError::new(message));
                }
            }
        }
        if !loaded.is_empty() {
            self.publish_loaded_compiled_resource_wave(
                device,
                loaded,
                device_capacity_permit,
            )?;
        }
        for waiter in pending {
            waiter
                .wait()
                .map(drop)
                .map_err(compiled_device_store_residency_error)?;
        }
        Ok(())
    }

    fn evict_inactive_capacity(
        &self,
        candidates: &[DeviceResourceResidencyEvictionCandidate],
        protected_group_ids: &BTreeSet<String>,
        required_payload_bytes: usize,
        required_device_bytes: usize,
        require_full_capacity: bool,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let mut address_state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let mut chunk_byte_capacities = BTreeMap::new();
        for cohort in address_state.chunk_groups.keys() {
            let arena = match cohort.tier {
                VulkanCompiledResourceMemoryTier::Device => &self.device_arena,
                VulkanCompiledResourceMemoryTier::HostVisible => {
                    self.host_visible_arena.as_ref().ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "host-visible allocation cohort has no stable arena",
                        )
                    })?
                }
            };
            let byte_capacity = arena
                .committed_byte_capacity_for_chunk(cohort.chunk_id)
                .map_err(compiled_device_store_vulkan_error)?;
            chunk_byte_capacities.insert(*cohort, byte_capacity);
        }
        let selection = compiled_resource_lru_eviction_selection(
            candidates,
            &address_state.group_chunks,
            &address_state.chunk_groups,
            &chunk_byte_capacities,
            protected_group_ids,
            required_payload_bytes,
            required_device_bytes,
        )?;
        if require_full_capacity
            && (selection.payload_bytes < required_payload_bytes
                || selection.device_bytes < required_device_bytes)
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource residency needs to reclaim {required_payload_bytes} payload and {required_device_bytes} device bytes, but inactive allocation cohorts provide only {} payload and {} device bytes",
                selection.payload_bytes, selection.device_bytes,
            )));
        }
        let selected_group_ids = selection.group_ids;
        if selected_group_ids.is_empty() {
            return Ok(0);
        }
        let committed_before = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?
            .committed_byte_capacity;
        let host_committed_before = self
            .host_visible_arena
            .as_ref()
            .map(|arena| arena.stats().map(|stats| stats.committed_byte_capacity))
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .unwrap_or(0);
        let eviction = self
            .manager
            .evict_inactive_groups(selected_group_ids.clone())
            .map_err(compiled_device_store_residency_error)?;
        let publications = selected_group_ids
            .iter()
            .flat_map(|group_id| {
                address_state
                    .publications
                    .get(group_id)
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect::<Vec<_>>();
        if publications.is_empty() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "resident eviction cohort has no address publications",
            ));
        }
        {
            let VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                ..
            } = &mut *address_state;
            address_table
                .clear_group(transfer, &publications)
                .map_err(compiled_device_store_vulkan_error)?;
        }
        for group_id in &selected_group_ids {
            address_state.publications.remove(group_id);
            let chunks = address_state
                .group_chunks
                .remove(group_id)
                .unwrap_or_default();
            for chunk_id in chunks {
                let remove_chunk =
                    if let Some(groups) = address_state.chunk_groups.get_mut(&chunk_id) {
                        groups.remove(group_id);
                        groups.is_empty()
                    } else {
                        false
                    };
                if remove_chunk {
                    address_state.chunk_groups.remove(&chunk_id);
                }
            }
        }
        let release = eviction.release();
        drop(eviction);
        if let Some(memory_plan) = &self.memory_plan {
            memory_plan
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned",
                    )
                })?
                .release_dynamic_groups(&selected_group_ids)?;
        }
        let committed_after = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?
            .committed_byte_capacity;
        let host_committed_after = self
            .host_visible_arena
            .as_ref()
            .map(|arena| arena.stats().map(|stats| stats.committed_byte_capacity))
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .unwrap_or(0);
        if committed_after >= committed_before
            && host_committed_after >= host_committed_before
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "evicting {} complete allocation-cohort groups released no device-local or host-visible bytes",
                release.group_count
            )));
        }
        let released_device_bytes = committed_before.saturating_sub(committed_after);
        if released_device_bytes > 0 {
            self.instrumentation
                .record_released_device_bytes(released_device_bytes);
        }
        Ok(released_device_bytes)
    }

    fn reclaim_inactive_device_memory(
        &self,
        requested_bytes: usize,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        if requested_bytes == 0 || !self.residency_policy.evicts_inactive_resources() {
            return Ok(0);
        }
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned",
            )
        })?;
        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        let protected_group_ids = BTreeSet::new();
        let candidates = self
            .manager
            .eviction_candidates(&protected_group_ids)
            .map_err(compiled_device_store_residency_error)?;
        self.evict_inactive_capacity(
            &candidates,
            &protected_group_ids,
            0,
            requested_bytes,
            false,
        )
    }

    fn evict_for_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
    ) -> Result<Option<VulkanDeviceLocalMemoryPermit>, VulkanCompiledResourceDeviceStoreError> {
        let residency_requirement = || {
            let snapshot = self
                .manager
                .snapshot()
                .map_err(compiled_device_store_residency_error)?;
            let known_group_ids = snapshot
                .directory
                .iter()
                .map(|entry| entry.group_id.as_str())
                .collect::<BTreeSet<_>>();
            let new_payload_bytes = plans
                .iter()
                .filter(|plan| !known_group_ids.contains(plan.descriptor.id.as_str()))
                .try_fold(0usize, |total, plan| {
                    total.checked_add(plan.descriptor.byte_count).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource admission byte count overflowed",
                        )
                    })
                })?;
            let new_plans = plans
                .iter()
                .filter(|plan| !known_group_ids.contains(plan.descriptor.id.as_str()))
                .collect::<Vec<_>>();
            let allocation_byte_counts = new_plans
                .iter()
                .map(|plan| {
                    plan.descriptor
                        .resources
                        .iter()
                        .map(|resource| resource.byte_count)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let allocation_requests = new_plans
                .iter()
                .zip(&allocation_byte_counts)
                .map(|(plan, byte_counts)| {
                    (plan.resource_slots.as_slice(), byte_counts.as_slice())
                })
                .collect::<Vec<_>>();
            let new_device_bytes = if allocation_requests.is_empty() {
                0
            } else {
                self.device_arena
                    .additional_committed_byte_capacity_for_groups(
                        device,
                        &allocation_requests,
                        self.upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?
            };
            let globally_available_device_bytes = usize::try_from(
                device
                    .device_local_memory_accounting()
                    .map_err(compiled_device_store_vulkan_error)?
                    .admissible_remaining_bytes,
            )
            .unwrap_or(usize::MAX);
            let arena = self
                .device_arena
                .stats()
                .map_err(compiled_device_store_vulkan_error)?;
            let arena_available_device_bytes = self
                .device_arena
                .config()
                .committed_byte_capacity
                .saturating_sub(arena.committed_byte_capacity);
            let required_device_bytes = new_device_bytes
                .saturating_sub(globally_available_device_bytes)
                .max(new_device_bytes.saturating_sub(arena_available_device_bytes));
            let used_payload_bytes = snapshot
                .statistics
                .always_resident_bytes
                .checked_add(snapshot.statistics.dynamic_resident_bytes)
                .and_then(|used| used.checked_add(snapshot.statistics.reserved_loading_bytes))
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource residency accounting overflowed",
                    )
                })?;
            let required_payload_bytes = used_payload_bytes
                .saturating_add(new_payload_bytes)
                .saturating_sub(snapshot.statistics.capacity_bytes);
            Ok::<_, VulkanCompiledResourceDeviceStoreError>((
                snapshot,
                new_payload_bytes,
                required_payload_bytes,
                new_device_bytes,
                required_device_bytes,
            ))
        };
        let initial_requirement = residency_requirement()?;
        if initial_requirement.2 == 0 && initial_requirement.4 == 0 {
            return if initial_requirement.3 == 0 {
                Ok(None)
            } else {
                device
                    .reserve_device_local_memory_capacity(initial_requirement.3)
                    .map(Some)
                    .map_err(compiled_device_store_vulkan_error)
            };
        }

        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        let (
            snapshot,
            new_payload_bytes,
            required_payload_bytes,
            new_device_bytes,
            required_device_bytes,
        ) = residency_requirement()?;
        if required_payload_bytes == 0 && required_device_bytes == 0 {
            return if new_device_bytes == 0 {
                Ok(None)
            } else {
                device
                    .reserve_device_local_memory_capacity(new_device_bytes)
                    .map(Some)
                    .map_err(compiled_device_store_vulkan_error)
            };
        }
        let candidates = self
            .manager
            .eviction_candidates(protected_group_ids)
            .map_err(compiled_device_store_residency_error)?;
        let candidates = compiled_resource_selector_fair_eviction_candidates(
            &candidates,
            &snapshot.directory,
            &self.group_selector_ids,
            &self.selector_payload_budgets,
            selector_id,
            new_payload_bytes,
        )?;
        self.evict_inactive_capacity(
            &candidates,
            protected_group_ids,
            required_payload_bytes,
            required_device_bytes,
            true,
        )?;
        let committed_after = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?
            .committed_byte_capacity;
        let remaining_arena_bytes = self
            .device_arena
            .config()
            .committed_byte_capacity
            .saturating_sub(committed_after);
        if remaining_arena_bytes < new_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource eviction reclaimed its selected cohorts, but the next {new_device_bytes}-byte device allocation has only {remaining_arena_bytes} arena bytes available"
            )));
        }
        wait_for_compiled_resource_device_capacity(
            new_device_bytes,
            COMPILED_RESOURCE_DEVICE_CAPACITY_SETTLEMENT_TIMEOUT,
            || {
                usize::try_from(
                    device
                        .device_local_memory_accounting()
                        .map_err(compiled_device_store_vulkan_error)?
                        .admissible_remaining_bytes,
                )
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "available Vulkan device-local capacity exceeds the host address space",
                    )
                })
            },
        )?;
        device
            .reserve_device_local_memory_capacity(new_device_bytes)
            .map(Some)
            .map_err(compiled_device_store_vulkan_error)
    }

    fn begin_execution(
        &self,
    ) -> Result<VulkanCompiledResourceExecutionGuard<'_>, VulkanCompiledResourceDeviceStoreError>
    {
        let guard = self.execution_barrier.read().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        Ok(VulkanCompiledResourceExecutionGuard { _guard: guard })
    }

    fn publish_loaded_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        loaded: Vec<(
            &VulkanCompiledResourceLoadPlan,
            DeviceResourceLoadPermit<VulkanResidentCompiledResource>,
            LoadedCompiledResourceGroup,
        )>,
        device_capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        #[cfg(test)]
        if self
            .fail_next_upload_as_device_lost
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            let error = VulkanError(
                "injected compiled resource upload failure: ERROR_DEVICE_LOST".to_string(),
            );
            self.record_terminal_device_failure(&error)?;
            for (_, permit, _) in loaded {
                let _ = permit.fail(DeviceResourceResidencyError::load_failed(error.to_string()));
            }
            return Err(compiled_device_store_vulkan_error(error));
        }
        if let Err(error) = self.ensure_device_work_is_available() {
            for (_, permit, _) in loaded {
                let _ = permit.fail(DeviceResourceResidencyError::load_failed(error.to_string()));
            }
            return Err(error);
        }
        let mut state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        for (plan, _, _) in &loaded {
            if state.publications.contains_key(&plan.descriptor.id) {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address publication exists without a resident directory entry",
                ));
            }
        }
        let upload_started = Instant::now();
        let total_uploaded_bytes = loaded.iter().try_fold(0usize, |total, (plan, _, _)| {
            total
                .checked_add(plan.descriptor.byte_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource upload batch byte count overflowed",
                    )
                })
        })?;
        let VulkanCompiledResourceDeviceAddressState {
            transfer,
            address_table,
            publications: resident_publications,
            group_chunks,
            chunk_groups,
        } = &mut *state;
        let mut device_indices = Vec::new();
        let mut host_visible_indices = Vec::new();
        let mut memory_plan = self
            .memory_plan
            .as_ref()
            .map(|memory_plan| {
                memory_plan.lock().map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned",
                    )
                })
            })
            .transpose()?;
        let tier_admission = memory_plan
            .as_mut()
            .map(|memory_plan| {
                memory_plan.admit_groups(
                    &loaded
                        .iter()
                        .map(|(plan, _, _)| {
                            (plan.descriptor.id.clone(), plan.descriptor.byte_count)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .transpose()?;
        let tiers = tier_admission
            .as_ref()
            .map(|admission| admission.tiers.clone())
            .unwrap_or_else(|| {
                vec![VulkanCompiledResourceMemoryTier::Device; loaded.len()]
            });
        for (index, tier) in tiers.iter().copied().enumerate() {
            match tier {
                VulkanCompiledResourceMemoryTier::Device => device_indices.push(index),
                VulkanCompiledResourceMemoryTier::HostVisible => host_visible_indices.push(index),
            }
        }
        let device_capacity_permit = match device_capacity_permit {
            Some(permit) => Some(permit),
            None if !device_indices.is_empty() => {
                let allocation_byte_counts = device_indices
                    .iter()
                    .map(|index| {
                        loaded[*index]
                            .0
                            .descriptor
                            .resources
                            .iter()
                            .map(|resource| resource.byte_count)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let allocation_requests = device_indices
                    .iter()
                    .zip(&allocation_byte_counts)
                    .map(|(index, byte_counts)| {
                        (loaded[*index].0.resource_slots.as_slice(), byte_counts.as_slice())
                    })
                    .collect::<Vec<_>>();
                let required_bytes = self
                    .device_arena
                    .additional_committed_byte_capacity_for_groups(
                        device,
                        &allocation_requests,
                        self.upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?;
                Some(
                    device
                        .reserve_device_local_memory_capacity(required_bytes)
                        .map_err(compiled_device_store_vulkan_error)?,
                )
            }
            None => None,
        };
        let upload_result = (|| {
            let mut indexed_uploads = upload_compiled_resource_tier(
                device,
                transfer,
                &self.device_arena,
                address_table,
                &loaded,
                &device_indices,
                self.upload_alignment,
                device_capacity_permit,
            )?;
            if !host_visible_indices.is_empty() {
                let host_visible_arena = self.host_visible_arena.as_ref().ok_or_else(|| {
                    VulkanError(
                        "tiered compiled-resource plan has no host-visible arena".to_string(),
                    )
                })?;
                match upload_compiled_resource_tier(
                    device,
                    transfer,
                    host_visible_arena,
                    address_table,
                    &loaded,
                    &host_visible_indices,
                    self.upload_alignment,
                    None,
                ) {
                    Ok(host_uploads) => indexed_uploads.extend(host_uploads),
                    Err(error) => {
                        let publications = indexed_uploads
                            .iter()
                            .flat_map(|(_, upload)| upload.publications().iter().cloned())
                            .collect::<Vec<_>>();
                        if !publications.is_empty() {
                            address_table.clear_group(transfer, &publications)?;
                        }
                        return Err(error);
                    }
                }
            }
            indexed_uploads.sort_by_key(|(index, _)| *index);
            Ok(indexed_uploads
                .into_iter()
                .map(|(_, upload)| upload)
                .collect::<Vec<_>>())
        })();
        let uploads = match upload_result {
            Ok(uploads) => uploads,
            Err(error) => {
                if let (Some(memory_plan), Some(admission)) =
                    (memory_plan.as_mut(), tier_admission.as_ref())
                {
                    memory_plan.rollback_admission(admission)?;
                }
                if compiled_resource_vulkan_error_is_device_loss(&error) {
                    self.record_terminal_device_failure(&error)?;
                }
                let message = format!("compiled resource upload failed: {error}");
                for (_, permit, _) in loaded {
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                }
                return Err(compiled_device_store_vulkan_error(error));
            }
        };
        let mut newly_assigned_group_ids = tier_admission
            .as_ref()
            .map(|admission| {
                admission
                    .newly_assigned_group_ids
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let mut staged = loaded
            .into_iter()
            .zip(uploads)
            .zip(tiers)
            .map(|(((plan, permit, _), upload), tier)| {
                let (group, publications) = upload.into_parts();
                let chunks = group
                    .resources()
                    .iter()
                    .filter_map(|resource| {
                        resource
                            .payload()
                            .stable_chunk_id()
                            .map(|chunk_id| VulkanCompiledResourceAllocationCohort {
                                tier,
                                chunk_id,
                            })
                    })
                    .collect::<BTreeSet<_>>();
                (
                    plan.descriptor.id.clone(),
                    permit,
                    group,
                    publications,
                    chunks,
                )
            })
            .collect::<Vec<_>>();
        while !staged.is_empty() {
            let (group_id, permit, group, publications, chunks) = staged.remove(0);
            if chunks.is_empty() {
                if let Some(memory_plan) = memory_plan.as_mut() {
                    memory_plan.release_dynamic_groups(
                        &newly_assigned_group_ids,
                    )?;
                }
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "stable compiled resource publication has no allocation cohort",
                ));
            }
            match permit.publish(group) {
                Ok(lease) => {
                    resident_publications.insert(group_id.clone(), publications);
                    group_chunks.insert(group_id.clone(), chunks.clone());
                    for chunk_id in chunks {
                        chunk_groups
                            .entry(chunk_id)
                            .or_default()
                            .insert(group_id.clone());
                    }
                    newly_assigned_group_ids.remove(&group_id);
                    drop(lease);
                }
                Err(error) => {
                    let mut unpublished = publications;
                    for (_, _, _, remaining_publications, _) in &staged {
                        unpublished.extend(remaining_publications.iter().cloned());
                    }
                    address_table
                        .clear_group(transfer, &unpublished)
                        .map_err(compiled_device_store_vulkan_error)?;
                    if let Some(memory_plan) = memory_plan.as_mut() {
                        memory_plan.release_dynamic_groups(
                            &newly_assigned_group_ids,
                        )?;
                    }
                    return Err(compiled_device_store_residency_error(error));
                }
            }
        }
        let arena = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        self.instrumentation.record_upload(
            total_uploaded_bytes,
            u64::try_from(upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            arena.committed_byte_capacity,
        );
        Ok(())
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
                (0..selector.resource_count)
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
                (0..selector.resource_count)
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
        self.instrumentation.mark_mount_complete(
            residency.dynamic_resident_bytes,
            residency.resident_group_count,
            arena.committed_byte_capacity,
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
        self.instrumentation
            .high_water_committed_device_bytes
            .fetch_max(
                u64::try_from(arena.committed_byte_capacity).unwrap_or(u64::MAX),
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
        Ok(VulkanCompiledResourceStoreReport {
            store_id: self.device_id.clone(),
            physical_device_id: self.physical_device_id.clone(),
            logical_device_ids: self.logical_device_ids.clone(),
            initial_device_bytes: total_device_bytes(initial_committed_device_bytes, "initial")?,
            current_device_bytes: total_device_bytes(arena.committed_byte_capacity, "current")?,
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
            uploaded_bytes: self.instrumentation.uploaded_bytes.load(Ordering::Relaxed),
            read_time_ns: backing.read_time_ns,
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
        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned during teardown",
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
        }
        drop(state);
        self.device_arena
            .release_backing()
            .map_err(compiled_device_store_vulkan_error)?;
        if let Some(arena) = &self.host_visible_arena {
            arena
                .release_backing()
                .map_err(compiled_device_store_vulkan_error)?;
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
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource store teardown did not quiesce cleanly: directory={}, dynamic_bytes={}, reserved_bytes={}, loading={}, resident={}, failed={}, device_arena_allocations={}, device_arena_bytes={}, device_arena_committed={}, device_arena_chunks={}, host_arena_allocations={}, host_arena_bytes={}, host_arena_committed={}, host_arena_chunks={}",
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
        match &selector.mapping {
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
    allowed_selector_ids: &BTreeSet<String>,
    maximum_dynamic_payload_bytes: usize,
) -> Result<VulkanCompiledResourceSelectorCachePolicy, VulkanCompiledResourceDeviceStoreError> {
    let resource_payload_bytes = contract
        .resources
        .iter()
        .map(|resource| {
            let byte_count = resource.ranges.iter().try_fold(0usize, |total, range| {
                total.checked_add(range.byte_count).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled selector resource byte count overflowed",
                    )
                })
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
    let mut group_payload_bytes = BTreeMap::<String, usize>::new();
    for selector in contract
        .selectors
        .iter()
        .filter(|selector| allowed_selector_ids.contains(&selector.id))
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
        for (group_id, byte_count) in groups {
            group_owners
                .entry(group_id.clone())
                .or_default()
                .insert(selector.id.clone());
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
    let mut selector_addressable_payload_bytes = allowed_selector_ids
        .iter()
        .map(|selector_id| (selector_id.clone(), 0usize))
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
        group_payload_bytes,
        selector_payload_budgets,
    })
}

fn compiled_resource_sparse_group_layouts(
    contract: &CompiledResourceResidencyContract,
    contract_index: &CompiledResourceContractIndex,
    layout: &VulkanCompiledResourceAddressLayout,
    allowed_selector_ids: &BTreeSet<String>,
) -> Result<Vec<VulkanStableResourceGroupLayout>, VulkanCompiledResourceDeviceStoreError> {
    let resource_byte_count = |resource: &CompiledImmutableResource,
                               label: &str|
     -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let byte_count = resource.ranges.iter().try_fold(0usize, |total, range| {
            total.checked_add(range.byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "sparse {label} resource byte count overflowed"
                ))
            })
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
        .filter(|selector| allowed_selector_ids.contains(&selector.selector_id))
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
                    let byte_counts = group
                        .resource_ids
                        .iter()
                        .map(|resource_id| {
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
                let byte_counts = template
                    .member_templates
                    .iter()
                    .map(|member| {
                        let byte_count =
                            member
                                .range_templates
                                .iter()
                                .try_fold(0usize, |total, range| {
                                    total.checked_add(range.byte_count).ok_or_else(|| {
                                        VulkanCompiledResourceDeviceStoreError::new(
                                            "sparse partition resource byte count overflowed",
                                        )
                                    })
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
    if explicit_groups.is_empty() && partitioned_groups.is_empty() {
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
