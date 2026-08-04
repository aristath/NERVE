const VULKAN_PARAMETER_ARENA_HEAP_FRACTION: u64 = 32;
const VULKAN_PARAMETER_ARENA_MIN_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const VULKAN_PARAMETER_ARENA_MAX_CHUNK_BYTES: usize = 1024 * 1024 * 1024;
const VULKAN_PARAMETER_ARENA_COPY_ALIGNMENT: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanResidentBufferPoolKey {
    pub namespace: String,
    pub device_id: String,
    pub resource_id: String,
    pub content_identity: String,
    pub byte_offset: usize,
    pub byte_count: usize,
}

impl VulkanResidentBufferPoolKey {
    pub fn new(
        namespace: impl Into<String>,
        device_id: impl Into<String>,
        resource_id: impl Into<String>,
        content_identity: impl Into<String>,
        byte_offset: usize,
        byte_count: usize,
    ) -> Result<Self, VulkanError> {
        let key = Self {
            namespace: namespace.into(),
            device_id: device_id.into(),
            resource_id: resource_id.into(),
            content_identity: content_identity.into(),
            byte_offset,
            byte_count,
        };
        if key.namespace.is_empty()
            || key.device_id.is_empty()
            || key.resource_id.is_empty()
            || key.content_identity.is_empty()
            || key.byte_count == 0
            || key.byte_offset.checked_add(key.byte_count).is_none()
        {
            return Err(VulkanError(
                "resident buffer pool key is invalid".to_string(),
            ));
        }
        Ok(key)
    }
}

#[derive(Clone)]
pub struct VulkanResidentBufferPoolAllocation {
    pub buffer: Arc<VulkanResidentBuffer>,
    pub byte_offset: usize,
    pub byte_count: usize,
}

impl VulkanResidentBufferPoolAllocation {
    pub fn buffer(&self) -> &VulkanResidentBuffer {
        &self.buffer
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn byte_count(&self) -> usize {
        self.byte_count
    }

    fn same_range(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
            && self.byte_offset == other.byte_offset
            && self.byte_count == other.byte_count
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentBufferPoolStats {
    pub resident_allocation_count: usize,
    pub resident_payload_bytes: usize,
    pub resident_buffer_count: usize,
    pub resident_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub eviction_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentBufferPoolDeviceRelease {
    pub resident_allocation_count: usize,
    pub resident_payload_bytes: usize,
    pub resident_buffer_count: usize,
    pub resident_bytes: usize,
}

#[derive(Default)]
struct VulkanResidentBufferPoolState {
    allocations: BTreeMap<
        VulkanResidentBufferPoolKey,
        VulkanResidentBufferPoolAllocation,
    >,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
    eviction_bytes: u64,
}

#[derive(Default)]
pub struct VulkanResidentBufferPool {
    state: RefCell<VulkanResidentBufferPoolState>,
    device_lifetime_guards:
        RefCell<BTreeMap<String, Rc<VulkanComputeDevice>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanParameterArenaPlacement {
    chunk_index: usize,
    byte_offset: usize,
    byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanParameterArenaPlan {
    chunk_byte_capacities: Vec<usize>,
    placements: Vec<VulkanParameterArenaPlacement>,
}

impl VulkanResidentBufferPool {
    pub fn register_device(
        &self,
        device_id: impl Into<String>,
        device: Rc<VulkanComputeDevice>,
    ) -> Result<(), VulkanError> {
        let device_id = device_id.into();
        if device_id.is_empty() {
            return Err(VulkanError(
                "resident buffer pool device id is empty".to_string(),
            ));
        }
        let mut guards = self.device_lifetime_guards.borrow_mut();
        if let Some(existing) = guards.get(&device_id) {
            if !Rc::ptr_eq(existing, &device) {
                return Err(VulkanError(format!(
                    "resident buffer pool device id {device_id:?} was rebound to another Vulkan device"
                )));
            }
            return Ok(());
        }
        guards.insert(device_id, device);
        Ok(())
    }

    pub fn resident_allocation(
        &self,
        key: &VulkanResidentBufferPoolKey,
    ) -> Option<VulkanResidentBufferPoolAllocation> {
        let mut state = self.state.borrow_mut();
        let allocation = state.allocations.get(key).cloned();
        if allocation.is_some() {
            state.hit_count = state.hit_count.saturating_add(1);
        } else {
            state.miss_count = state.miss_count.saturating_add(1);
        }
        allocation
    }

    pub fn allocate_unpublished(
        &self,
        key: &VulkanResidentBufferPoolKey,
    ) -> Result<VulkanResidentBufferPoolAllocation, VulkanError> {
        self.allocate_unpublished_batch(std::slice::from_ref(key))?
            .pop()
            .ok_or_else(|| {
                VulkanError(
                    "single resident parameter arena allocation disappeared".to_string(),
                )
            })
    }

    pub fn allocate_unpublished_batch(
        &self,
        keys: &[VulkanResidentBufferPoolKey],
    ) -> Result<Vec<VulkanResidentBufferPoolAllocation>, VulkanError> {
        if keys.is_empty() {
            return Err(VulkanError(
                "resident parameter arena allocation batch is empty".to_string(),
            ));
        }
        let device_id = keys[0].device_id.as_str();
        if keys.iter().any(|key| key.device_id != device_id) {
            return Err(VulkanError(
                "resident parameter arena batch spans multiple devices".to_string(),
            ));
        }
        let unique_keys = keys.iter().collect::<BTreeSet<_>>();
        if unique_keys.len() != keys.len() {
            return Err(VulkanError(
                "resident parameter arena batch repeats a pool key".to_string(),
            ));
        }
        if let Some(key) = keys
            .iter()
            .find(|key| self.state.borrow().allocations.contains_key(*key))
        {
            return Err(VulkanError(format!(
                "resident parameter arena key {:?} is already published",
                key.resource_id
            )));
        }
        let device = self
            .device_lifetime_guards
            .borrow()
            .get(device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanError(format!(
                    "resident buffer pool has no registered device {device_id:?}"
                ))
            })?;
        let byte_counts = keys.iter().map(|key| key.byte_count).collect::<Vec<_>>();
        let allocate = || device.allocate_resident_buffer_arena(&byte_counts);
        let chunks = match allocate() {
            Ok(chunks) => chunks,
            Err(first_error) => {
                if self.evict_unreferenced() == 0 {
                    return Err(first_error);
                }
                allocate().map_err(|retry_error| {
                    VulkanError(format!(
                        "resident parameter arena allocation failed before and after evicting idle pooled buffers: first={first_error}; retry={retry_error}"
                    ))
                })?
            }
        };
        Ok(chunks)
    }

    pub fn publish(
        &self,
        key: VulkanResidentBufferPoolKey,
        allocation: VulkanResidentBufferPoolAllocation,
    ) -> Result<(), VulkanError> {
        if let Some(existing) = self.state.borrow().allocations.get(&key) {
            if !existing.same_range(&allocation) {
                return Err(VulkanError(
                    "resident buffer pool key was published twice with different allocations"
                        .to_string(),
                ));
            }
            return Ok(());
        }
        self.publish_batch(vec![(key, allocation)])
    }

    pub fn publish_batch(
        &self,
        publications: Vec<(
            VulkanResidentBufferPoolKey,
            VulkanResidentBufferPoolAllocation,
        )>,
    ) -> Result<(), VulkanError> {
        if publications.is_empty() {
            return Err(VulkanError(
                "resident parameter arena publication batch is empty".to_string(),
            ));
        }
        let publication_keys = publications
            .iter()
            .map(|(key, _)| key)
            .collect::<BTreeSet<_>>();
        if publication_keys.len() != publications.len() {
            return Err(VulkanError(
                "resident parameter arena publication batch repeats a pool key".to_string(),
            ));
        }
        let guards = self.device_lifetime_guards.borrow();
        let state = self.state.borrow();
        if let Some((key, _)) = publications
            .iter()
            .find(|(key, _)| state.allocations.contains_key(key))
        {
            return Err(VulkanError(format!(
                "resident parameter arena key {:?} is already published",
                key.resource_id
            )));
        }
        for (key, allocation) in &publications {
            validate_parameter_arena_publication(&guards, key, allocation)?;
        }
        let mut ranges_by_buffer = BTreeMap::<
            usize,
            Vec<(usize, usize, &str, &str)>,
        >::new();
        for (key, allocation) in state
            .allocations
            .iter()
            .chain(publications.iter().map(|(key, allocation)| (key, allocation)))
        {
            ranges_by_buffer
                .entry(Arc::as_ptr(&allocation.buffer) as usize)
                .or_default()
                .push((
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    key.device_id.as_str(),
                    key.resource_id.as_str(),
                ));
        }
        for ranges in ranges_by_buffer.values_mut() {
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                let (left_start, left_end, left_device, left_resource) = pair[0];
                let (right_start, _, right_device, right_resource) = pair[1];
                if left_device != right_device {
                    return Err(VulkanError(format!(
                        "resident parameter arena buffer cannot span devices {:?} and {:?}",
                        left_device, right_device
                    )));
                }
                if left_start < left_end && right_start < left_end {
                    return Err(VulkanError(format!(
                        "resident parameter arena ranges for {:?} and {:?} overlap",
                        left_resource, right_resource
                    )));
                }
            }
        }
        drop(state);
        drop(guards);
        self.state.borrow_mut().allocations.extend(publications);
        Ok(())
    }

    pub fn evict_unreferenced(&self) -> usize {
        let mut state = self.state.borrow_mut();
        let groups = resident_allocation_groups(&state.allocations, |_| true);
        let removable = groups
            .into_iter()
            .filter_map(|(identity, keys)| {
                let allocation = state
                    .allocations
                    .get(&keys[0])
                    .expect("grouped resident allocation remains present");
                (Arc::strong_count(&allocation.buffer) == keys.len())
                    .then_some((identity, keys))
            })
            .collect::<Vec<_>>();
        let release = remove_resident_allocation_groups(&mut state.allocations, &removable);
        record_pool_eviction(&mut state, release);
        release.resident_buffer_count
    }

    pub fn evict_unreferenced_device(
        &self,
        device_id: &str,
    ) -> Result<VulkanResidentBufferPoolDeviceRelease, VulkanError> {
        self.require_registered_device(device_id)?;
        let mut state = self.state.borrow_mut();
        let all_groups = resident_allocation_groups(&state.allocations, |_| true);
        let removable = all_groups
            .into_iter()
            .filter(|(_, keys)| {
                keys.iter().all(|key| key.device_id == device_id)
                    && {
                        let allocation = state
                            .allocations
                            .get(&keys[0])
                            .expect("grouped resident allocation remains present");
                        Arc::strong_count(&allocation.buffer) == keys.len()
                    }
            })
            .collect::<Vec<_>>();
        let release = remove_resident_allocation_groups(&mut state.allocations, &removable);
        record_pool_eviction(&mut state, release);
        Ok(release)
    }

    pub fn release_device(
        &self,
        device_id: &str,
    ) -> Result<VulkanResidentBufferPoolDeviceRelease, VulkanError> {
        self.require_registered_device(device_id)?;
        let mut state = self.state.borrow_mut();
        let groups = resident_allocation_groups(&state.allocations, |key| {
            key.device_id == device_id
        });
        if let Some((_, keys, owner_count)) = groups.iter().find_map(|(identity, keys)| {
            let allocation = state
                .allocations
                .get(&keys[0])
                .expect("grouped resident allocation remains present");
            let owner_count = Arc::strong_count(&allocation.buffer);
            (owner_count != keys.len()).then_some((*identity, keys, owner_count))
        }) {
            return Err(VulkanError(format!(
                "resident buffer pool cannot release device {device_id:?}: buffer containing {:?} has {owner_count} owners but only {} belong to the pool",
                keys[0].resource_id,
                keys.len()
            )));
        }
        let grouped = groups.into_iter().collect::<Vec<_>>();
        let release = remove_resident_allocation_groups(&mut state.allocations, &grouped);
        record_pool_eviction(&mut state, release);
        drop(state);
        self.device_lifetime_guards
            .borrow_mut()
            .remove(device_id)
            .expect("validated resident buffer pool device guard exists");
        Ok(release)
    }

    pub fn registered_device_count(&self) -> usize {
        self.device_lifetime_guards.borrow().len()
    }

    pub fn stats(&self) -> VulkanResidentBufferPoolStats {
        let state = self.state.borrow();
        let groups = resident_allocation_groups(&state.allocations, |_| true);
        let resident_bytes = groups
            .values()
            .map(|keys| {
                state
                    .allocations
                    .get(&keys[0])
                    .expect("grouped resident allocation remains present")
                    .buffer
                    .byte_capacity()
            })
            .sum();
        VulkanResidentBufferPoolStats {
            resident_allocation_count: state.allocations.len(),
            resident_payload_bytes: state.allocations.values().map(|entry| entry.byte_count).sum(),
            resident_buffer_count: groups.len(),
            resident_bytes,
            hit_count: state.hit_count,
            miss_count: state.miss_count,
            eviction_count: state.eviction_count,
            eviction_bytes: state.eviction_bytes,
        }
    }

    fn require_registered_device(&self, device_id: &str) -> Result<(), VulkanError> {
        if device_id.is_empty() {
            return Err(VulkanError(
                "resident buffer pool device id is empty".to_string(),
            ));
        }
        if !self
            .device_lifetime_guards
            .borrow()
            .contains_key(device_id)
        {
            return Err(VulkanError(format!(
                "resident buffer pool has no registered device {device_id:?}"
            )));
        }
        Ok(())
    }
}

impl VulkanComputeDevice {
    pub fn allocate_resident_buffer_arena(
        &self,
        byte_counts: &[usize],
    ) -> Result<Vec<VulkanResidentBufferPoolAllocation>, VulkanError> {
        let alignment = self
            .min_storage_buffer_offset_alignment()
            .max(VULKAN_PARAMETER_ARENA_COPY_ALIGNMENT);
        let maximum_chunk_bytes = parameter_arena_chunk_limit(self, alignment)?;
        let plan = plan_parameter_arena(byte_counts, alignment, maximum_chunk_bytes)?;
        let chunks = allocate_parameter_arena_buffers(self, &plan)?;
        plan.placements
            .iter()
            .map(|placement| {
                Ok(VulkanResidentBufferPoolAllocation {
                    buffer: Arc::clone(chunks.get(placement.chunk_index).ok_or_else(|| {
                        VulkanError(
                            "resident parameter arena placement references a missing chunk"
                                .to_string(),
                        )
                    })?),
                    byte_offset: placement.byte_offset,
                    byte_count: placement.byte_count,
                })
            })
            .collect()
    }
}

fn parameter_arena_chunk_limit(
    device: &VulkanComputeDevice,
    alignment: usize,
) -> Result<usize, VulkanError> {
    validate_parameter_arena_alignment(alignment)?;
    let heap_scaled = usize::try_from(
        device.device_local_memory_bytes() / VULKAN_PARAMETER_ARENA_HEAP_FRACTION,
    )
    .unwrap_or(usize::MAX);
    let desired = heap_scaled
        .max(VULKAN_PARAMETER_ARENA_MIN_CHUNK_BYTES)
        .min(VULKAN_PARAMETER_ARENA_MAX_CHUNK_BYTES);
    let aligned = desired - (desired % alignment);
    if aligned == 0 {
        return Err(VulkanError(
            "resident parameter arena chunk limit is smaller than its alignment".to_string(),
        ));
    }
    Ok(aligned)
}

fn plan_parameter_arena(
    byte_counts: &[usize],
    alignment: usize,
    maximum_chunk_bytes: usize,
) -> Result<VulkanParameterArenaPlan, VulkanError> {
    if byte_counts.is_empty() || byte_counts.contains(&0) {
        return Err(VulkanError(
            "resident parameter arena byte layout is empty".to_string(),
        ));
    }
    validate_parameter_arena_alignment(alignment)?;
    let maximum_chunk_bytes = maximum_chunk_bytes - (maximum_chunk_bytes % alignment);
    if maximum_chunk_bytes == 0 {
        return Err(VulkanError(
            "resident parameter arena maximum chunk is smaller than its alignment".to_string(),
        ));
    }

    let mut chunk_byte_capacities = Vec::new();
    let mut placements = Vec::with_capacity(byte_counts.len());
    let mut chunk_index = 0usize;
    let mut chunk_end = 0usize;
    for byte_count in byte_counts {
        let mut byte_offset = align_parameter_arena_offset(chunk_end, alignment)?;
        let mut range_end = byte_offset.checked_add(*byte_count).ok_or_else(|| {
            VulkanError("resident parameter arena range overflowed".to_string())
        })?;
        if chunk_end != 0 && range_end > maximum_chunk_bytes {
            chunk_byte_capacities.push(align_parameter_arena_offset(chunk_end, alignment)?);
            chunk_index = chunk_index.checked_add(1).ok_or_else(|| {
                VulkanError("resident parameter arena chunk count overflowed".to_string())
            })?;
            byte_offset = 0;
            range_end = *byte_count;
        }
        placements.push(VulkanParameterArenaPlacement {
            chunk_index,
            byte_offset,
            byte_count: *byte_count,
        });
        chunk_end = range_end;
        if chunk_end > maximum_chunk_bytes {
            chunk_byte_capacities.push(align_parameter_arena_offset(chunk_end, alignment)?);
            chunk_index = chunk_index.checked_add(1).ok_or_else(|| {
                VulkanError("resident parameter arena chunk count overflowed".to_string())
            })?;
            chunk_end = 0;
        }
    }
    if chunk_end != 0 {
        chunk_byte_capacities.push(align_parameter_arena_offset(chunk_end, alignment)?);
    }
    if chunk_byte_capacities.is_empty() {
        return Err(VulkanError(
            "resident parameter arena produced no chunks".to_string(),
        ));
    }
    Ok(VulkanParameterArenaPlan {
        chunk_byte_capacities,
        placements,
    })
}

fn align_parameter_arena_offset(
    byte_offset: usize,
    alignment: usize,
) -> Result<usize, VulkanError> {
    validate_parameter_arena_alignment(alignment)?;
    byte_offset
        .checked_add(alignment - 1)
        .map(|offset| offset & !(alignment - 1))
        .ok_or_else(|| VulkanError("resident parameter arena alignment overflowed".to_string()))
}

fn validate_parameter_arena_alignment(alignment: usize) -> Result<(), VulkanError> {
    if alignment < VULKAN_PARAMETER_ARENA_COPY_ALIGNMENT || !alignment.is_power_of_two() {
        return Err(VulkanError(format!(
            "resident parameter arena alignment {alignment} is invalid"
        )));
    }
    Ok(())
}

fn allocate_parameter_arena_buffers(
    device: &VulkanComputeDevice,
    plan: &VulkanParameterArenaPlan,
) -> Result<Vec<Arc<VulkanResidentBuffer>>, VulkanError> {
    plan.chunk_byte_capacities
        .iter()
        .map(|byte_capacity| device.create_resident_buffer(*byte_capacity).map(Arc::new))
        .collect()
}

fn validate_parameter_arena_publication(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    key: &VulkanResidentBufferPoolKey,
    allocation: &VulkanResidentBufferPoolAllocation,
) -> Result<(), VulkanError> {
    if allocation.byte_count != key.byte_count {
        return Err(VulkanError(format!(
            "resident parameter arena publication byte count {} does not match key byte count {}",
            allocation.byte_count, key.byte_count,
        )));
    }
    let end = allocation
        .byte_offset
        .checked_add(allocation.byte_count)
        .ok_or_else(|| VulkanError("resident parameter arena range overflowed".to_string()))?;
    if end > allocation.buffer.byte_capacity() {
        return Err(VulkanError(format!(
            "resident parameter arena range {end} exceeds backing buffer capacity {}",
            allocation.buffer.byte_capacity()
        )));
    }
    let device = devices.get(&key.device_id).ok_or_else(|| {
        VulkanError(format!(
            "resident buffer pool has no registered device {:?}",
            key.device_id
        ))
    })?;
    if !device.owns_resident_buffer(&allocation.buffer) {
        return Err(VulkanError(format!(
            "resident parameter arena allocation does not belong to device {:?}",
            key.device_id
        )));
    }
    let alignment = device
        .min_storage_buffer_offset_alignment()
        .max(VULKAN_PARAMETER_ARENA_COPY_ALIGNMENT);
    if !allocation.byte_offset.is_multiple_of(alignment) {
        return Err(VulkanError(format!(
            "resident parameter arena offset {} is not aligned to {alignment} bytes",
            allocation.byte_offset
        )));
    }
    Ok(())
}

fn resident_allocation_groups<F>(
    allocations: &BTreeMap<
        VulkanResidentBufferPoolKey,
        VulkanResidentBufferPoolAllocation,
    >,
    mut include: F,
) -> BTreeMap<usize, Vec<VulkanResidentBufferPoolKey>>
where
    F: FnMut(&VulkanResidentBufferPoolKey) -> bool,
{
    let mut groups = BTreeMap::<usize, Vec<VulkanResidentBufferPoolKey>>::new();
    for (key, allocation) in allocations {
        if include(key) {
            groups
                .entry(Arc::as_ptr(&allocation.buffer) as usize)
                .or_default()
                .push(key.clone());
        }
    }
    groups
}

fn remove_resident_allocation_groups(
    allocations: &mut BTreeMap<
        VulkanResidentBufferPoolKey,
        VulkanResidentBufferPoolAllocation,
    >,
    groups: &[(usize, Vec<VulkanResidentBufferPoolKey>)],
) -> VulkanResidentBufferPoolDeviceRelease {
    let mut release = VulkanResidentBufferPoolDeviceRelease::default();
    for (_, keys) in groups {
        let Some(first) = keys.first().and_then(|key| allocations.get(key)) else {
            continue;
        };
        release.resident_buffer_count = release.resident_buffer_count.saturating_add(1);
        release.resident_bytes = release
            .resident_bytes
            .saturating_add(first.buffer.byte_capacity());
        for key in keys {
            if let Some(allocation) = allocations.remove(key) {
                release.resident_allocation_count =
                    release.resident_allocation_count.saturating_add(1);
                release.resident_payload_bytes = release
                    .resident_payload_bytes
                    .saturating_add(allocation.byte_count);
            }
        }
    }
    release
}

fn record_pool_eviction(
    state: &mut VulkanResidentBufferPoolState,
    release: VulkanResidentBufferPoolDeviceRelease,
) {
    state.eviction_count = state
        .eviction_count
        .saturating_add(release.resident_buffer_count as u64);
    state.eviction_bytes = state
        .eviction_bytes
        .saturating_add(release.resident_bytes as u64);
}

#[cfg(test)]
mod resident_buffer_pool_tests {
    use super::*;

    #[test]
    fn parameter_arena_packs_aligned_ranges_into_bounded_chunks() {
        let plan = plan_parameter_arena(&[60, 100, 200], 64, 256).unwrap();

        assert_eq!(plan.chunk_byte_capacities, [192, 256]);
        assert_eq!(
            plan.placements,
            [
                VulkanParameterArenaPlacement {
                    chunk_index: 0,
                    byte_offset: 0,
                    byte_count: 60,
                },
                VulkanParameterArenaPlacement {
                    chunk_index: 0,
                    byte_offset: 64,
                    byte_count: 100,
                },
                VulkanParameterArenaPlacement {
                    chunk_index: 1,
                    byte_offset: 0,
                    byte_count: 200,
                },
            ]
        );
    }

    #[test]
    fn parameter_arena_gives_oversized_resource_a_dedicated_chunk() {
        let plan = plan_parameter_arena(&[64, 300, 64], 64, 256).unwrap();

        assert_eq!(plan.chunk_byte_capacities, [64, 320, 64]);
        assert_eq!(
            plan.placements
                .iter()
                .map(|placement| (placement.chunk_index, placement.byte_offset))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 0), (2, 0)]
        );
    }

    #[test]
    fn parameter_arena_rejects_zero_ranges_and_invalid_alignment() {
        assert!(plan_parameter_arena(&[], 64, 256).is_err());
        assert!(plan_parameter_arena(&[64, 0], 64, 256).is_err());
        assert!(plan_parameter_arena(&[64], 3, 256).is_err());
        assert!(plan_parameter_arena(&[64], 512, 256).is_err());
    }

    #[test]
    fn selected_amd_parameter_arena_binds_ranges_and_releases_shared_backing() {
        let Some(raw_device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX").ok() else {
            eprintln!("skipping parameter arena integration test: explicit Vulkan device index unset");
            return;
        };
        let device_index = raw_device_index
            .parse::<usize>()
            .expect("NERVE_TEST_VULKAN_DEVICE_INDEX must be an integer");
        let device = Rc::new(
            VulkanComputeDevice::new_for_physical_device_index(device_index)
                .expect("selected AMD Vulkan device must open"),
        );
        let pool = VulkanResidentBufferPool::default();
        pool.register_device("arena-test", Rc::clone(&device))
            .unwrap();
        let keys = (0..3)
            .map(|index| {
                VulkanResidentBufferPoolKey::new(
                    "nerve.test.parameter_arena.v1",
                    "arena-test",
                    format!("tensor-{index}"),
                    format!("{index:064x}"),
                    0,
                    64,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let allocations = pool.allocate_unpublished_batch(&keys).unwrap();
        assert_eq!(allocations.len(), 3);
        assert!(Arc::ptr_eq(&allocations[0].buffer, &allocations[1].buffer));
        assert!(Arc::ptr_eq(&allocations[1].buffer, &allocations[2].buffer));
        assert_eq!(allocations[0].byte_offset, 0);
        assert!(allocations[1].byte_offset > allocations[0].byte_offset);
        assert!(allocations[2].byte_offset > allocations[1].byte_offset);

        let words = |base: u32| {
            (base..base + 16)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        for (index, allocation) in allocations.iter().enumerate() {
            allocation
                .buffer
                .write_bytes_at(allocation.byte_offset, &words((index * 100) as u32))
                .unwrap();
        }
        let spirv_words = compile_test_shader_words()
            .expect("parameter arena integration test requires a GLSL compiler");
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &allocations[1].buffer,
                    allocations[1].byte_count,
                )
                .with_byte_offset(allocations[1].byte_offset)],
                1,
                64,
                0,
            )
            .unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();
        let first = allocations[0]
            .buffer
            .read_bytes_at(allocations[0].byte_offset, 64)
            .unwrap();
        let second = allocations[1]
            .buffer
            .read_bytes_at(allocations[1].byte_offset, 64)
            .unwrap();
        let third = allocations[2]
            .buffer
            .read_bytes_at(allocations[2].byte_offset, 64)
            .unwrap();
        assert_eq!(first, words(0));
        assert_eq!(second, words(100).chunks_exact(4).flat_map(|bytes| {
            (u32::from_le_bytes(bytes.try_into().unwrap()) + 1).to_le_bytes()
        }).collect::<Vec<_>>());
        assert_eq!(third, words(200));
        drop(dispatch);

        pool.publish_batch(
            keys.iter()
                .cloned()
                .zip(allocations.iter().cloned())
                .collect(),
        )
        .unwrap();
        let hits = keys
            .iter()
            .map(|key| pool.resident_allocation(key).unwrap())
            .collect::<Vec<_>>();
        let stats = pool.stats();
        assert_eq!(stats.resident_allocation_count, 3);
        assert_eq!(stats.resident_payload_bytes, 192);
        assert_eq!(stats.resident_buffer_count, 1);
        assert!(stats.resident_bytes >= 192);
        assert!(pool.release_device("arena-test").is_err());

        drop(hits);
        drop(allocations);
        let release = pool.release_device("arena-test").unwrap();
        assert_eq!(release.resident_allocation_count, 3);
        assert_eq!(release.resident_payload_bytes, 192);
        assert_eq!(release.resident_buffer_count, 1);
        assert_eq!(release.resident_bytes, stats.resident_bytes);
        assert_eq!(pool.stats().resident_buffer_count, 0);
        device.quiesce().unwrap();
    }
}
