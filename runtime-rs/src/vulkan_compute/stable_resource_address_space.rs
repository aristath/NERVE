const VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanStableResourceArenaConfig {
    pub committed_byte_capacity: usize,
    pub minimum_alignment: usize,
    pub memory_domain: VulkanStableResourceMemoryDomain,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VulkanStableResourceMemoryDomain {
    #[default]
    Device,
    HostVisible,
}

impl VulkanStableResourceArenaConfig {
    pub fn new(
        committed_byte_capacity: usize,
        minimum_alignment: usize,
    ) -> Result<Self, VulkanError> {
        if committed_byte_capacity == 0 {
            return Err(VulkanError(
                "stable resource arena committed capacity must not be zero".to_string(),
            ));
        }
        validate_stable_resource_alignment(minimum_alignment)?;
        Ok(Self {
            committed_byte_capacity,
            minimum_alignment,
            memory_domain: VulkanStableResourceMemoryDomain::Device,
        })
    }

    pub fn host_visible(mut self) -> Self {
        self.memory_domain = VulkanStableResourceMemoryDomain::HostVisible;
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanStableResourceArenaStats {
    pub committed_byte_capacity: usize,
    pub allocated_byte_count: usize,
    pub active_allocation_count: usize,
    pub chunk_count: usize,
}

pub struct VulkanStableResourceArena {
    config: VulkanStableResourceArenaConfig,
    device_handle: vk::Device,
    layouts: Arc<VulkanStableResourceArenaLayouts>,
    allocation_requirement_byte_counts: std::sync::Mutex<BTreeMap<usize, usize>>,
    state: Arc<std::sync::Mutex<VulkanStableResourceArenaState>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanStableResourceGroupLayout {
    Explicit {
        resource_slots: Vec<usize>,
        resource_byte_counts: Vec<usize>,
    },
    Partitioned {
        member_slot_bases: Vec<usize>,
        resource_byte_counts: Vec<usize>,
        partition_count: usize,
    },
}

#[derive(Clone)]
struct VulkanStableResourcePlacement {
    resource_slots: Vec<usize>,
    resource_byte_offsets: Vec<usize>,
    resource_byte_counts: Vec<usize>,
    group_byte_capacity: usize,
}

struct VulkanStableResourceArenaLayouts {
    explicit: BTreeMap<Vec<usize>, VulkanStableResourcePlacement>,
    partitioned: Vec<VulkanPartitionedStableResourcePlacement>,
    maximum_byte_capacity: usize,
}

struct VulkanStableResourceArenaState {
    active_groups: BTreeMap<Vec<usize>, usize>,
    chunks: BTreeMap<u64, VulkanStableResourceChunk>,
    allocations: BTreeMap<u64, VulkanStableResourceAllocationRecord>,
    next_chunk_id: u64,
    next_allocation_id: u64,
    committed_byte_capacity: usize,
    allocated_byte_count: usize,
}

#[derive(Clone)]
struct VulkanPartitionedStableResourcePlacement {
    member_slot_bases: Vec<usize>,
    resource_byte_offsets: Vec<usize>,
    resource_byte_counts: Vec<usize>,
    partition_count: usize,
    group_byte_capacity: usize,
}

struct VulkanStableResourceChunk {
    byte_capacity: usize,
    active_allocation_count: usize,
}

struct VulkanStableResourceAllocationRecord {
    byte_count: usize,
    chunk_id: u64,
}

struct VulkanStableResourceGroupAllocationPlan {
    placements: Vec<(VulkanStableResourcePlacement, Vec<usize>, Vec<usize>)>,
    chunk_byte_capacity: usize,
    payload_byte_count: usize,
    resource_count: usize,
}

pub struct VulkanStableResourceAllocation {
    allocation_id: u64,
    resource_slot: usize,
    chunk_id: u64,
    group_key: Arc<[usize]>,
    buffer: Arc<VulkanResidentBuffer>,
    arena_state: Arc<std::sync::Mutex<VulkanStableResourceArenaState>>,
    byte_offset: usize,
    byte_count: usize,
    device_address: vk::DeviceAddress,
    device_address_registry: Option<Arc<Mutex<VulkanDeviceAddressRegistry>>>,
}

impl VulkanStableResourceArena {
    pub fn new(
        device: &VulkanComputeDevice,
        config: VulkanStableResourceArenaConfig,
        groups: &[VulkanStableResourceGroupLayout],
    ) -> Result<Self, VulkanError> {
        if groups.is_empty() {
            return Err(VulkanError(
                "stable resource arena has no addressable groups".to_string(),
            ));
        }
        if !device.supports_buffer_device_address() {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot host addressable demand-backed resources",
                device.device_name()
            )));
        }
        let mut placements = BTreeMap::new();
        let mut partitioned_placements = Vec::new();
        let mut claimed_slot_ranges = Vec::new();
        let mut maximum_byte_capacity = 0usize;
        for group in groups {
            match group {
                VulkanStableResourceGroupLayout::Explicit {
                    resource_slots,
                    resource_byte_counts,
                } => {
                    let (member_offsets, group_byte_capacity) =
                        stable_group_member_layout(resource_byte_counts, config.minimum_alignment)?;
                    if resource_slots.is_empty()
                        || resource_slots.len() != resource_byte_counts.len()
                    {
                        return Err(VulkanError(
                            "explicit stable resource group layout is invalid".to_string(),
                        ));
                    }
                    maximum_byte_capacity = maximum_byte_capacity
                        .checked_add(group_byte_capacity)
                        .ok_or_else(|| {
                            VulkanError("stable resource maximum capacity overflowed".to_string())
                        })?;
                    for slot in resource_slots {
                        claimed_slot_ranges.push((
                            *slot,
                            slot.checked_add(1).ok_or_else(|| {
                                VulkanError(
                                    "explicit stable address slot range overflowed".to_string(),
                                )
                            })?,
                        ));
                    }
                    let placement = VulkanStableResourcePlacement {
                        resource_slots: resource_slots.clone(),
                        resource_byte_offsets: member_offsets,
                        resource_byte_counts: resource_byte_counts.clone(),
                        group_byte_capacity,
                    };
                    let mut group_key = resource_slots.clone();
                    group_key.sort_unstable();
                    if placements.insert(group_key, placement).is_some() {
                        return Err(VulkanError(
                            "explicit stable resource group layout is duplicated".to_string(),
                        ));
                    }
                }
                VulkanStableResourceGroupLayout::Partitioned {
                    member_slot_bases,
                    resource_byte_counts,
                    partition_count,
                } => {
                    let (resource_byte_offsets, group_byte_capacity) =
                        stable_group_member_layout(resource_byte_counts, config.minimum_alignment)?;
                    if member_slot_bases.is_empty()
                        || member_slot_bases.len() != resource_byte_counts.len()
                        || *partition_count == 0
                    {
                        return Err(VulkanError(
                            "partitioned stable resource group layout is invalid".to_string(),
                        ));
                    }
                    maximum_byte_capacity = group_byte_capacity
                        .checked_mul(*partition_count)
                        .and_then(|bytes| maximum_byte_capacity.checked_add(bytes))
                        .ok_or_else(|| {
                            VulkanError(
                                "partitioned stable resource maximum capacity overflowed"
                                    .to_string(),
                            )
                        })?;
                    for base in member_slot_bases {
                        claimed_slot_ranges.push((
                            *base,
                            base.checked_add(*partition_count).ok_or_else(|| {
                                VulkanError(
                                    "partitioned stable address slot range overflowed".to_string(),
                                )
                            })?,
                        ));
                    }
                    partitioned_placements.push(VulkanPartitionedStableResourcePlacement {
                        member_slot_bases: member_slot_bases.clone(),
                        resource_byte_offsets,
                        resource_byte_counts: resource_byte_counts.clone(),
                        partition_count: *partition_count,
                        group_byte_capacity,
                    });
                }
            }
        }
        claimed_slot_ranges.sort_unstable();
        if claimed_slot_ranges
            .windows(2)
            .any(|ranges| ranges[0].1 > ranges[1].0)
        {
            return Err(VulkanError(
                "stable resource layouts assign one address slot to multiple groups".to_string(),
            ));
        }
        Ok(Self {
            config,
            device_handle: device.device.handle(),
            layouts: Arc::new(VulkanStableResourceArenaLayouts {
                explicit: placements,
                partitioned: partitioned_placements,
                maximum_byte_capacity,
            }),
            allocation_requirement_byte_counts: std::sync::Mutex::new(BTreeMap::new()),
            state: Arc::new(std::sync::Mutex::new(VulkanStableResourceArenaState {
                active_groups: BTreeMap::new(),
                chunks: BTreeMap::new(),
                allocations: BTreeMap::new(),
                next_chunk_id: 0,
                next_allocation_id: 0,
                committed_byte_capacity: 0,
                allocated_byte_count: 0,
            })),
        })
    }

    pub fn config(&self) -> VulkanStableResourceArenaConfig {
        self.config
    }

    pub fn stats(&self) -> Result<VulkanStableResourceArenaStats, VulkanError> {
        let state = self.state.lock().map_err(|_| {
            VulkanError("stable resource arena state lock was poisoned".to_string())
        })?;
        Ok(VulkanStableResourceArenaStats {
            committed_byte_capacity: state.committed_byte_capacity,
            allocated_byte_count: state.allocated_byte_count,
            active_allocation_count: state.allocations.len(),
            chunk_count: state.chunks.len(),
        })
    }

    pub fn maximum_backed_byte_capacity(&self) -> Result<usize, VulkanError> {
        Ok(self.layouts.maximum_byte_capacity)
    }

    pub fn additional_committed_byte_capacity_for_groups(
        &self,
        device: &VulkanComputeDevice,
        groups: &[(&[usize], &[usize])],
        alignment: usize,
    ) -> Result<usize, VulkanError> {
        let state = self.state.lock().map_err(|_| {
            VulkanError("stable resource arena state lock was poisoned".to_string())
        })?;
        let plan = plan_stable_resource_groups(
            &self.layouts,
            &state,
            &self.config,
            groups,
            alignment,
        )?;
        self.physical_allocation_byte_count(device, plan.chunk_byte_capacity)
    }

    pub fn committed_byte_capacity_for_chunk(
        &self,
        chunk_id: u64,
    ) -> Result<usize, VulkanError> {
        let state = self.state.lock().map_err(|_| {
            VulkanError("stable resource arena state lock was poisoned".to_string())
        })?;
        state
            .chunks
            .get(&chunk_id)
            .map(|chunk| chunk.byte_capacity)
            .ok_or_else(|| {
                VulkanError(format!(
                    "stable resource arena has no committed chunk {chunk_id}"
                ))
            })
    }

    pub fn allocate_groups(
        &self,
        device: &VulkanComputeDevice,
        groups: &[(&[usize], &[usize])],
        alignment: usize,
    ) -> Result<Vec<Vec<Arc<VulkanStableResourceAllocation>>>, VulkanError> {
        if groups.is_empty() {
            return Err(VulkanError(
                "stable resource group allocation batch is empty".to_string(),
            ));
        }
        allocate_stable_resource_groups(
            self.device_handle,
            &self.layouts,
            &self.state,
            device,
            &self.config,
            &self.allocation_requirement_byte_counts,
            groups,
            alignment,
            None,
        )
    }

    pub fn allocate_groups_with_capacity_permit(
        &self,
        device: &VulkanComputeDevice,
        groups: &[(&[usize], &[usize])],
        alignment: usize,
        capacity_permit: VulkanDeviceLocalMemoryPermit,
    ) -> Result<Vec<Vec<Arc<VulkanStableResourceAllocation>>>, VulkanError> {
        if self.config.memory_domain != VulkanStableResourceMemoryDomain::Device {
            return Err(VulkanError(
                "device-local capacity permit cannot back a host-visible stable arena"
                    .to_string(),
            ));
        }
        allocate_stable_resource_groups(
            self.device_handle,
            &self.layouts,
            &self.state,
            device,
            &self.config,
            &self.allocation_requirement_byte_counts,
            groups,
            alignment,
            Some(capacity_permit),
        )
    }


    fn physical_allocation_byte_count(
        &self,
        device: &VulkanComputeDevice,
        requested_byte_count: usize,
    ) -> Result<usize, VulkanError> {
        physical_stable_resource_allocation_byte_count(
            self.device_handle,
            &self.allocation_requirement_byte_counts,
            device,
            requested_byte_count,
        )
    }

    pub fn release_backing(&self) -> Result<(), VulkanError> {
        {
            let state = self.state.lock().map_err(|_| {
                VulkanError("stable resource arena state lock was poisoned".to_string())
            })?;
            if !state.allocations.is_empty()
                || !state.active_groups.is_empty()
                || state.allocated_byte_count != 0
            {
                return Err(VulkanError(format!(
                    "stable resource arena still owns {} allocations and {} payload bytes",
                    state.allocations.len(),
                    state.allocated_byte_count
                )));
            }
            if !state.chunks.is_empty() || state.committed_byte_capacity != 0 {
                return Err(VulkanError(
                    "stable resource arena retained chunks without allocations".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn align_stable_resource_offset(
    byte_offset: usize,
    alignment: usize,
) -> Result<usize, VulkanError> {
    validate_stable_resource_alignment(alignment)?;
    byte_offset
        .checked_add(alignment - 1)
        .map(|offset| offset & !(alignment - 1))
        .ok_or_else(|| VulkanError("stable resource aligned offset overflowed".to_string()))
}

fn stable_group_member_layout(
    resource_byte_counts: &[usize],
    minimum_alignment: usize,
) -> Result<(Vec<usize>, usize), VulkanError> {
    if resource_byte_counts.is_empty()
        || resource_byte_counts
            .iter()
            .any(|byte_count| *byte_count == 0)
    {
        return Err(VulkanError(
            "stable resource group byte layout is empty".to_string(),
        ));
    }
    let mut byte_capacity = 0usize;
    let mut offsets = Vec::with_capacity(resource_byte_counts.len());
    for byte_count in resource_byte_counts {
        byte_capacity = align_stable_resource_offset(byte_capacity, minimum_alignment)?;
        offsets.push(byte_capacity);
        byte_capacity = byte_capacity
            .checked_add(*byte_count)
            .ok_or_else(|| VulkanError("stable resource group capacity overflowed".to_string()))?;
    }
    byte_capacity = align_stable_resource_offset(byte_capacity, minimum_alignment)?;
    Ok((offsets, byte_capacity))
}

fn stable_resource_placement_for_slots(
    layouts: &VulkanStableResourceArenaLayouts,
    requested_slots: &[usize],
    sorted_slots: &[usize],
) -> Result<VulkanStableResourcePlacement, VulkanError> {
    if let Some(placement) = layouts.explicit.get(sorted_slots) {
        return Ok(placement.clone());
    }
    'partitioned_layout: for partitioned in &layouts.partitioned {
        if requested_slots.len() != partitioned.member_slot_bases.len() {
            continue;
        }
        let mut partition_index = None;
        let mut requested_member_indices = Vec::with_capacity(requested_slots.len());
        let mut seen_member_indices = BTreeSet::new();
        for slot in requested_slots {
            let Some((member_index, selected_partition_index)) =
                partitioned.member_slot_bases.iter().enumerate().find_map(
                    |(member_index, base)| {
                        slot.checked_sub(*base)
                            .filter(|index| *index < partitioned.partition_count)
                            .map(|index| (member_index, index))
                    },
                )
            else {
                continue 'partitioned_layout;
            };
            if !seen_member_indices.insert(member_index)
                || partition_index.is_some_and(|index| index != selected_partition_index)
            {
                continue 'partitioned_layout;
            }
            partition_index = Some(selected_partition_index);
            requested_member_indices.push(member_index);
        }
        partition_index.expect("partition layout has requested members");
        let resource_byte_offsets = requested_member_indices
            .iter()
            .map(|member_index| Ok(partitioned.resource_byte_offsets[*member_index]))
            .collect::<Result<Vec<_>, _>>()?;
        let resource_byte_counts = requested_member_indices
            .iter()
            .map(|member_index| partitioned.resource_byte_counts[*member_index])
            .collect();
        return Ok(VulkanStableResourcePlacement {
            resource_slots: requested_slots.to_vec(),
            resource_byte_offsets,
            resource_byte_counts,
            group_byte_capacity: partitioned.group_byte_capacity,
        });
    }
    Err(VulkanError(
        "stable resource group has no compiled placement".to_string(),
    ))
}

fn physical_stable_resource_allocation_byte_count(
    device_handle: vk::Device,
    cached_byte_counts: &std::sync::Mutex<BTreeMap<usize, usize>>,
    device: &VulkanComputeDevice,
    requested_byte_count: usize,
) -> Result<usize, VulkanError> {
    if device.device.handle() != device_handle {
        return Err(VulkanError(
            "stable resource allocation requirement was requested from another logical device"
                .to_string(),
        ));
    }
    if let Some(byte_count) = cached_byte_counts
        .lock()
        .map_err(|_| {
            VulkanError(
                "stable resource allocation-requirement cache was poisoned".to_string(),
            )
        })?
        .get(&requested_byte_count)
        .copied()
    {
        return Ok(byte_count);
    }
    let physical_byte_count =
        device.addressable_resident_buffer_memory_requirement_bytes(requested_byte_count)?;
    if physical_byte_count < requested_byte_count {
        return Err(VulkanError(format!(
            "Vulkan reported {physical_byte_count} physical bytes for a {requested_byte_count}-byte stable resource chunk"
        )));
    }
    cached_byte_counts
        .lock()
        .map_err(|_| {
            VulkanError(
                "stable resource allocation-requirement cache was poisoned".to_string(),
            )
        })?
        .insert(requested_byte_count, physical_byte_count);
    Ok(physical_byte_count)
}

fn allocate_stable_resource_groups(
    device_handle: vk::Device,
    layouts: &Arc<VulkanStableResourceArenaLayouts>,
    arena_state: &Arc<std::sync::Mutex<VulkanStableResourceArenaState>>,
    device: &VulkanComputeDevice,
    config: &VulkanStableResourceArenaConfig,
    allocation_requirement_byte_counts: &std::sync::Mutex<BTreeMap<usize, usize>>,
    groups: &[(&[usize], &[usize])],
    alignment: usize,
    capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
) -> Result<Vec<Vec<Arc<VulkanStableResourceAllocation>>>, VulkanError> {
    if device.device.handle() != device_handle {
        return Err(VulkanError(
            "stable resources were requested from another logical device".to_string(),
        ));
    }
    if alignment != config.minimum_alignment {
        return Err(VulkanError(format!(
            "stable resource allocation alignment {alignment} differs from its compiled layout alignment {}",
            config.minimum_alignment
        )));
    }
    let mut state = arena_state
        .lock()
        .map_err(|_| VulkanError("stable resource arena state lock was poisoned".to_string()))?;
    let plan = plan_stable_resource_groups(layouts, &state, config, groups, alignment)?;
    let VulkanStableResourceGroupAllocationPlan {
        placements,
        chunk_byte_capacity,
        payload_byte_count,
        resource_count,
    } = plan;
    let physical_chunk_byte_capacity = physical_stable_resource_allocation_byte_count(
        device_handle,
        allocation_requirement_byte_counts,
        device,
        chunk_byte_capacity,
    )?;
    let committed_byte_capacity = state
        .committed_byte_capacity
        .checked_add(physical_chunk_byte_capacity)
        .ok_or_else(|| VulkanError("stable resource committed capacity overflowed".to_string()))?;
    if committed_byte_capacity > config.committed_byte_capacity {
        return Err(VulkanError(format!(
            "stable resources need {physical_chunk_byte_capacity} additional physical bytes for a {chunk_byte_capacity}-byte logical chunk, but {} of {} physical bytes are already committed",
            state.committed_byte_capacity, config.committed_byte_capacity,
        )));
    }
    state
        .next_allocation_id
        .checked_add(
            u64::try_from(resource_count).expect("stable resource plan prevalidated count"),
        )
        .expect("stable resource plan prevalidated allocation ids");
    let allocated_byte_count = state
        .allocated_byte_count
        .checked_add(payload_byte_count)
        .expect("stable resource plan prevalidated payload capacity");
    let chunk_id = state.next_chunk_id;
    state.next_chunk_id = state
        .next_chunk_id
        .checked_add(1)
        .ok_or_else(|| VulkanError("stable resource chunk ids exhausted".to_string()))?;
    let mut buffer = match config.memory_domain {
        VulkanStableResourceMemoryDomain::Device => {
            match capacity_permit {
                Some(permit) => device
                    .create_addressable_resident_buffer_with_capacity_permit(
                        chunk_byte_capacity,
                        permit,
                    )?,
                None => device.create_addressable_resident_buffer(chunk_byte_capacity)?,
            }
        }
        VulkanStableResourceMemoryDomain::HostVisible => {
            if capacity_permit.is_some() {
                return Err(VulkanError(
                    "device-local capacity permit cannot back a host-visible stable arena"
                        .to_string(),
                ));
            }
            device.create_host_visible_addressable_resident_buffer(chunk_byte_capacity)?
        }
    };
    if config.memory_domain == VulkanStableResourceMemoryDomain::HostVisible {
        buffer.persistently_map()?;
    }
    let buffer = Arc::new(buffer);
    let base_address = buffer.device_address()?;

    let mut allocation_groups = Vec::with_capacity(placements.len());
    let mut group_byte_offset = 0usize;
    for (placement, group_key, requested_slots) in placements {
        group_byte_offset =
            align_stable_resource_offset(group_byte_offset, config.minimum_alignment)?;
        let mut allocations = Vec::with_capacity(placement.resource_slots.len());
        let allocation_group_key = Arc::<[usize]>::from(group_key.clone());
        for slot in requested_slots {
            let placement_index = placement
                .resource_slots
                .iter()
                .position(|candidate| *candidate == slot)
                .expect("stable group member was prevalidated");
            let byte_offset = group_byte_offset
                .checked_add(placement.resource_byte_offsets[placement_index])
                .ok_or_else(|| {
                    VulkanError("stable resource member offset overflowed".to_string())
                })?;
            let byte_count = placement.resource_byte_counts[placement_index];
            let allocation_id = state.next_allocation_id;
            state.next_allocation_id += 1;
            let device_address = base_address
                .checked_add(
                    u64::try_from(byte_offset)
                        .expect("stable resource offset was prevalidated"),
                )
                .expect("stable resource address was prevalidated");
            let device_address_registry = buffer.device_address_registry.clone();
            if let Some(registry) = &device_address_registry {
                registry
                    .lock()
                    .map_err(|_| {
                        VulkanError("device-address registry was poisoned".to_string())
                    })?
                    .register_annotation(
                        allocation_id,
                        device_address,
                        byte_count,
                        format!(
                            "stable resource slot={slot} allocation={allocation_id} chunk={chunk_id}"
                        ),
                    )?;
            }
            state.allocations.insert(
                allocation_id,
                VulkanStableResourceAllocationRecord {
                    byte_count,
                    chunk_id,
                },
            );
            allocations.push(Arc::new(VulkanStableResourceAllocation {
                allocation_id,
                resource_slot: slot,
                chunk_id,
                group_key: Arc::clone(&allocation_group_key),
                buffer: Arc::clone(&buffer),
                arena_state: Arc::clone(arena_state),
                byte_offset,
                byte_count,
                device_address,
                device_address_registry,
            }));
        }
        state
            .active_groups
            .insert(group_key.clone(), allocations.len());
        allocation_groups.push(allocations);
        group_byte_offset = group_byte_offset
            .checked_add(placement.group_byte_capacity)
            .expect("stable chunk capacity was prevalidated");
    }
    state.chunks.insert(
        chunk_id,
        VulkanStableResourceChunk {
            byte_capacity: physical_chunk_byte_capacity,
            active_allocation_count: resource_count,
        },
    );
    state.allocated_byte_count = allocated_byte_count;
    state.committed_byte_capacity = committed_byte_capacity;
    Ok(allocation_groups)
}

fn plan_stable_resource_groups(
    layouts: &VulkanStableResourceArenaLayouts,
    state: &VulkanStableResourceArenaState,
    config: &VulkanStableResourceArenaConfig,
    groups: &[(&[usize], &[usize])],
    alignment: usize,
) -> Result<VulkanStableResourceGroupAllocationPlan, VulkanError> {
    if groups.is_empty() {
        return Err(VulkanError(
            "stable resource group allocation batch is empty".to_string(),
        ));
    }
    if alignment != config.minimum_alignment {
        return Err(VulkanError(format!(
            "stable resource allocation alignment {alignment} differs from its compiled layout alignment {}",
            config.minimum_alignment
        )));
    }
    let mut requested_keys = BTreeSet::new();
    let mut placements = Vec::with_capacity(groups.len());
    let mut chunk_byte_capacity = 0usize;
    let mut payload_byte_count = 0usize;
    let mut resource_count = 0usize;
    for (slots, byte_counts) in groups {
        let mut group_key = slots.to_vec();
        group_key.sort_unstable();
        if slots.is_empty()
            || slots.len() != byte_counts.len()
            || !requested_keys.insert(group_key.clone())
            || state.active_groups.contains_key(&group_key)
        {
            return Err(VulkanError(
                "stable resource group allocation is duplicated or invalid".to_string(),
            ));
        }
        let placement = stable_resource_placement_for_slots(layouts, slots, &group_key)?;
        for (slot, byte_count) in slots.iter().zip(*byte_counts) {
            let placement_index = placement
                .resource_slots
                .iter()
                .position(|candidate| candidate == slot)
                .ok_or_else(|| {
                    VulkanError(
                        "stable resource request contains an unknown group member".to_string(),
                    )
                })?;
            if placement.resource_byte_counts[placement_index] != *byte_count {
                return Err(VulkanError(
                    "stable resource request differs from its compiled placement".to_string(),
                ));
            }
            payload_byte_count = payload_byte_count.checked_add(*byte_count).ok_or_else(|| {
                VulkanError("stable resource payload capacity overflowed".to_string())
            })?;
        }
        chunk_byte_capacity =
            align_stable_resource_offset(chunk_byte_capacity, config.minimum_alignment)?
                .checked_add(placement.group_byte_capacity)
                .ok_or_else(|| {
                    VulkanError("stable resource chunk capacity overflowed".to_string())
                })?;
        resource_count = resource_count.checked_add(slots.len()).ok_or_else(|| {
            VulkanError("stable resource allocation count overflowed".to_string())
        })?;
        placements.push((placement, group_key, slots.to_vec()));
    }
    state
        .next_allocation_id
        .checked_add(
            u64::try_from(resource_count).map_err(|_| {
                VulkanError("stable resource allocation count exceeds u64".to_string())
            })?,
        )
        .ok_or_else(|| VulkanError("stable resource allocation ids exhausted".to_string()))?;
    state
        .allocated_byte_count
        .checked_add(payload_byte_count)
        .ok_or_else(|| VulkanError("stable resource allocated payload overflowed".to_string()))?;
    state
        .next_chunk_id
        .checked_add(1)
        .ok_or_else(|| VulkanError("stable resource chunk ids exhausted".to_string()))?;
    Ok(VulkanStableResourceGroupAllocationPlan {
        placements,
        chunk_byte_capacity,
        payload_byte_count,
        resource_count,
    })
}

impl VulkanStableResourceAllocation {
    pub fn buffer(&self) -> &VulkanResidentBuffer {
        &self.buffer
    }

    pub fn buffer_byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn byte_count(&self) -> usize {
        self.byte_count
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub(crate) fn chunk_id(&self) -> u64 {
        self.chunk_id
    }

    fn allocation_id(&self) -> u64 {
        self.allocation_id
    }
}

impl Drop for VulkanStableResourceAllocation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.arena_state.lock() {
            if let Some(record) = state.allocations.remove(&self.allocation_id) {
                debug_assert_eq!(record.byte_count, self.byte_count);
                debug_assert_eq!(record.chunk_id, self.chunk_id);
                state.allocated_byte_count =
                    state.allocated_byte_count.saturating_sub(record.byte_count);
                let mut remove_active_group = false;
                if let Some(active_allocation_count) =
                    state.active_groups.get_mut(self.group_key.as_ref())
                {
                    *active_allocation_count = active_allocation_count.saturating_sub(1);
                    remove_active_group = *active_allocation_count == 0;
                } else {
                    debug_assert!(false, "stable resource allocation lost its active group");
                }
                if remove_active_group {
                    state.active_groups.remove(self.group_key.as_ref());
                }
                let mut remove_chunk = false;
                if let Some(chunk) = state.chunks.get_mut(&self.chunk_id) {
                    chunk.active_allocation_count =
                        chunk.active_allocation_count.saturating_sub(1);
                    remove_chunk = chunk.active_allocation_count == 0;
                } else {
                    debug_assert!(false, "stable resource allocation lost its chunk");
                }
                if remove_chunk
                    && let Some(chunk) = state.chunks.remove(&self.chunk_id)
                {
                    state.committed_byte_capacity = state
                        .committed_byte_capacity
                        .saturating_sub(chunk.byte_capacity);
                }
            } else {
                debug_assert!(false, "stable resource allocation was released twice");
            }
        }
        if let Some(registry) = &self.device_address_registry
            && let Ok(mut registry) = registry.lock()
        {
            let result = registry.unregister_annotation(self.allocation_id, self.device_address);
            debug_assert!(
                result.is_ok(),
                "stable resource slot {} lost its device-address annotation",
                self.resource_slot
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct VulkanStableResourceAddressRecord {
    pub device_address: u64,
    pub byte_count: u64,
    pub generation: u64,
    pub resident: u32,
    pub reserved: u32,
}

impl VulkanStableResourceAddressRecord {
    fn bytes(self) -> [u8; VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT] {
        let mut bytes = [0u8; VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT];
        bytes[0..8].copy_from_slice(&self.device_address.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.byte_count.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.generation.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.resident.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.reserved.to_le_bytes());
        bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanStableResourceAddressPublication {
    slot: usize,
    generation: u64,
    allocation_id: u64,
    device_address: u64,
}

impl VulkanStableResourceAddressPublication {
    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn device_address(&self) -> u64 {
        self.device_address
    }
}

pub struct VulkanStableResourceAddressTable {
    buffer: Arc<VulkanResidentBuffer>,
    records: Vec<VulkanStableResourceAddressRecord>,
    resident_allocations: Vec<Option<Arc<VulkanStableResourceAllocation>>>,
}

impl VulkanStableResourceAddressTable {
    pub fn new(
        device: &VulkanComputeDevice,
        transfer: &mut VulkanResidentTransferStream,
        slot_count: usize,
    ) -> Result<Self, VulkanError> {
        if slot_count == 0 {
            return Err(VulkanError(
                "stable resource address table must have at least one slot".to_string(),
            ));
        }
        let byte_capacity = slot_count
            .checked_mul(VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT)
            .ok_or_else(|| {
                VulkanError("stable resource address table capacity overflowed".to_string())
            })?;
        if byte_capacity > transfer.staging_byte_capacity() {
            return Err(VulkanError(format!(
                "stable resource address table needs {byte_capacity} staging bytes but the transfer stream provides {}",
                transfer.staging_byte_capacity()
            )));
        }
        let buffer = Arc::new(device.create_resident_buffer(byte_capacity)?);
        let zeros = vec![0u8; byte_capacity];
        let write = VulkanResidentBufferWriteRange::new(&buffer, 0, &zeros)?;
        transfer
            .submit_consumer_serialized(&[write])
            .map_err(|failure| failure.error)?;
        Ok(Self {
            buffer,
            records: vec![VulkanStableResourceAddressRecord::default(); slot_count],
            resident_allocations: (0..slot_count).map(|_| None).collect(),
        })
    }

    pub fn buffer(&self) -> &VulkanResidentBuffer {
        &self.buffer
    }

    pub fn shared_buffer(&self) -> Arc<VulkanResidentBuffer> {
        Arc::clone(&self.buffer)
    }

    pub fn byte_capacity(&self) -> usize {
        self.buffer.byte_capacity()
    }

    pub fn slot_count(&self) -> usize {
        self.records.len()
    }

    pub fn record(&self, slot: usize) -> Result<VulkanStableResourceAddressRecord, VulkanError> {
        self.records.get(slot).copied().ok_or_else(|| {
            VulkanError(format!(
                "stable resource address table slot {slot} is out of bounds for {} slots",
                self.records.len()
            ))
        })
    }

    pub fn publish_group(
        &mut self,
        transfer: &mut VulkanResidentTransferStream,
        resources: &[(usize, Arc<VulkanStableResourceAllocation>)],
    ) -> Result<Vec<VulkanStableResourceAddressPublication>, VulkanError> {
        if resources.is_empty() {
            return Err(VulkanError(
                "stable resource publication group must not be empty".to_string(),
            ));
        }
        let mut slots = BTreeSet::new();
        let mut updates = Vec::with_capacity(resources.len());
        let mut publications = Vec::with_capacity(resources.len());
        for (slot, allocation) in resources {
            if !slots.insert(*slot) {
                return Err(VulkanError(format!(
                    "stable resource publication repeats slot {slot}"
                )));
            }
            let previous = self.records.get(*slot).copied().ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {slot} is out of bounds for {} slots",
                    self.records.len()
                ))
            })?;
            if previous.resident != 0 || self.resident_allocations[*slot].is_some() {
                return Err(VulkanError(format!(
                    "stable resource address table slot {slot} is already resident"
                )));
            }
            if allocation.buffer.device.handle() != self.buffer.device.handle() {
                return Err(VulkanError(format!(
                    "stable resource address table slot {slot} cannot publish an allocation from another logical device"
                )));
            }
            let generation = previous.generation.checked_add(1).ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {slot} exhausted its generations"
                ))
            })?;
            let byte_count = u64::try_from(allocation.byte_count())
                .map_err(|_| VulkanError("stable resource byte count exceeds u64".to_string()))?;
            let record = VulkanStableResourceAddressRecord {
                device_address: allocation.device_address(),
                byte_count,
                generation,
                resident: 1,
                reserved: 0,
            };
            updates.push((*slot, record, Some(Arc::clone(allocation))));
            publications.push(VulkanStableResourceAddressPublication {
                slot: *slot,
                generation,
                allocation_id: allocation.allocation_id(),
                device_address: allocation.device_address(),
            });
        }
        self.submit_updates(transfer, &updates)?;
        Ok(publications)
    }

    pub fn allocations_for_publications(
        &self,
        publications: &[VulkanStableResourceAddressPublication],
    ) -> Result<Vec<Arc<VulkanStableResourceAllocation>>, VulkanError> {
        if publications.is_empty() {
            return Err(VulkanError(
                "stable resource publication group must not be empty".to_string(),
            ));
        }
        let mut slots = BTreeSet::new();
        publications
            .iter()
            .map(|publication| {
                if !slots.insert(publication.slot) {
                    return Err(VulkanError(format!(
                        "stable resource publication group repeats slot {}",
                        publication.slot
                    )));
                }
                self.allocation_for_publication(publication)
            })
            .collect()
    }

    pub fn swap_groups(
        &mut self,
        transfer: &mut VulkanResidentTransferStream,
        left: &[VulkanStableResourceAddressPublication],
        right: &[VulkanStableResourceAddressPublication],
    ) -> Result<
        (
            Vec<VulkanStableResourceAddressPublication>,
            Vec<VulkanStableResourceAddressPublication>,
        ),
        VulkanError,
    > {
        if left.is_empty() || left.len() != right.len() {
            return Err(VulkanError(format!(
                "stable resource exchange needs two non-empty groups of equal length, got {} and {}",
                left.len(),
                right.len()
            )));
        }
        let mut slots = BTreeSet::new();
        for publication in left.iter().chain(right) {
            if !slots.insert(publication.slot) {
                return Err(VulkanError(format!(
                    "stable resource exchange repeats slot {}",
                    publication.slot
                )));
            }
        }
        let left_allocations = left
            .iter()
            .map(|publication| self.allocation_for_publication(publication))
            .collect::<Result<Vec<_>, _>>()?;
        let right_allocations = right
            .iter()
            .map(|publication| self.allocation_for_publication(publication))
            .collect::<Result<Vec<_>, _>>()?;
        let mut updates = Vec::with_capacity(left.len() * 2);
        let mut exchanged_left = Vec::with_capacity(left.len());
        let mut exchanged_right = Vec::with_capacity(right.len());
        for (((left_publication, left_allocation), right_publication), right_allocation) in left
            .iter()
            .zip(&left_allocations)
            .zip(right)
            .zip(&right_allocations)
        {
            if left_allocation.byte_count() != right_allocation.byte_count() {
                return Err(VulkanError(format!(
                    "stable resource exchange slots {} and {} have incompatible byte counts {} and {}",
                    left_publication.slot,
                    right_publication.slot,
                    left_allocation.byte_count(),
                    right_allocation.byte_count()
                )));
            }
            let left_generation = left_publication.generation.checked_add(1).ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {} exhausted its generations",
                    left_publication.slot
                ))
            })?;
            let right_generation =
                right_publication.generation.checked_add(1).ok_or_else(|| {
                    VulkanError(format!(
                        "stable resource address table slot {} exhausted its generations",
                        right_publication.slot
                    ))
                })?;
            let byte_count = u64::try_from(left_allocation.byte_count())
                .map_err(|_| VulkanError("stable resource byte count exceeds u64".to_string()))?;
            updates.push((
                left_publication.slot,
                VulkanStableResourceAddressRecord {
                    device_address: right_allocation.device_address(),
                    byte_count,
                    generation: left_generation,
                    resident: 1,
                    reserved: 0,
                },
                Some(Arc::clone(right_allocation)),
            ));
            updates.push((
                right_publication.slot,
                VulkanStableResourceAddressRecord {
                    device_address: left_allocation.device_address(),
                    byte_count,
                    generation: right_generation,
                    resident: 1,
                    reserved: 0,
                },
                Some(Arc::clone(left_allocation)),
            ));
            exchanged_left.push(VulkanStableResourceAddressPublication {
                slot: left_publication.slot,
                generation: left_generation,
                allocation_id: right_allocation.allocation_id(),
                device_address: right_allocation.device_address(),
            });
            exchanged_right.push(VulkanStableResourceAddressPublication {
                slot: right_publication.slot,
                generation: right_generation,
                allocation_id: left_allocation.allocation_id(),
                device_address: left_allocation.device_address(),
            });
        }
        self.submit_updates(transfer, &updates)?;
        Ok((exchanged_left, exchanged_right))
    }

    pub fn clear_group(
        &mut self,
        transfer: &mut VulkanResidentTransferStream,
        publications: &[VulkanStableResourceAddressPublication],
    ) -> Result<(), VulkanError> {
        if publications.is_empty() {
            return Err(VulkanError(
                "stable resource clear group must not be empty".to_string(),
            ));
        }
        let mut slots = BTreeSet::new();
        let mut updates = Vec::with_capacity(publications.len());
        for publication in publications {
            if !slots.insert(publication.slot) {
                return Err(VulkanError(format!(
                    "stable resource clear group repeats slot {}",
                    publication.slot
                )));
            }
            let record = self.records.get(publication.slot).copied().ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {} is out of bounds for {} slots",
                    publication.slot,
                    self.records.len()
                ))
            })?;
            if record.resident != 1
                || record.generation != publication.generation
                || record.device_address != publication.device_address
                || self.resident_allocations[publication.slot]
                    .as_ref()
                    .map(|allocation| allocation.allocation_id())
                    != Some(publication.allocation_id)
            {
                return Err(VulkanError(format!(
                    "stable resource address table slot {} no longer matches its publication",
                    publication.slot
                )));
            }
            updates.push((
                publication.slot,
                VulkanStableResourceAddressRecord {
                    generation: record.generation.checked_add(1).ok_or_else(|| {
                        VulkanError(format!(
                            "stable resource address table slot {} exhausted its generations",
                            publication.slot
                        ))
                    })?,
                    ..VulkanStableResourceAddressRecord::default()
                },
                None,
            ));
        }
        self.submit_updates(transfer, &updates)
    }

    fn submit_updates(
        &mut self,
        transfer: &mut VulkanResidentTransferStream,
        updates: &[(
            usize,
            VulkanStableResourceAddressRecord,
            Option<Arc<VulkanStableResourceAllocation>>,
        )],
    ) -> Result<(), VulkanError> {
        let encoded = updates
            .iter()
            .map(|(_, record, _)| record.bytes())
            .collect::<Vec<_>>();
        let writes = updates
            .iter()
            .zip(&encoded)
            .map(|((slot, _, _), bytes)| {
                VulkanResidentBufferWriteRange::new(
                    &self.buffer,
                    slot.checked_mul(VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT)
                        .ok_or_else(|| {
                            VulkanError(
                                "stable resource address table byte offset overflowed".to_string(),
                            )
                        })?,
                    bytes,
                )
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        let previous_allocations = updates
            .iter()
            .map(|(slot, _, _)| (*slot, self.resident_allocations[*slot].clone()))
            .collect::<Vec<_>>();
        updates
            .iter()
            .for_each(|(slot, _, allocation)| {
                if let Some(allocation) = allocation {
                    self.resident_allocations[*slot] = Some(Arc::clone(allocation));
                }
            });
        if let Err(failure) = transfer.submit_consumer_serialized(&writes) {
            if !failure.submission_accepted {
                for (slot, allocation) in previous_allocations {
                    self.resident_allocations[slot] = allocation;
                }
            }
            return Err(failure.error);
        }
        for (slot, record, allocation) in updates {
            self.records[*slot] = *record;
            self.resident_allocations[*slot] = allocation.clone();
        }
        Ok(())
    }

    fn allocation_for_publication(
        &self,
        publication: &VulkanStableResourceAddressPublication,
    ) -> Result<Arc<VulkanStableResourceAllocation>, VulkanError> {
        let record = self.records.get(publication.slot).copied().ok_or_else(|| {
            VulkanError(format!(
                "stable resource address table slot {} is out of bounds for {} slots",
                publication.slot,
                self.records.len()
            ))
        })?;
        let allocation = self.resident_allocations[publication.slot]
            .as_ref()
            .ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {} is not resident",
                    publication.slot
                ))
            })?;
        if record.resident != 1
            || record.generation != publication.generation
            || record.device_address != publication.device_address
            || allocation.allocation_id() != publication.allocation_id
        {
            return Err(VulkanError(format!(
                "stable resource address table slot {} no longer matches its publication",
                publication.slot
            )));
        }
        Ok(Arc::clone(allocation))
    }
}

fn validate_stable_resource_alignment(alignment: usize) -> Result<(), VulkanError> {
    if alignment < std::mem::align_of::<u64>() || !alignment.is_power_of_two() {
        return Err(VulkanError(format!(
            "stable resource alignment {alignment} must be a power of two at least {}",
            std::mem::align_of::<u64>()
        )));
    }
    Ok(())
}

const _: () = assert!(
    std::mem::size_of::<VulkanStableResourceAddressRecord>()
        == VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT
);
