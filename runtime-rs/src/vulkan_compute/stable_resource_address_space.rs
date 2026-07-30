const VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanStableResourceArenaConfig {
    pub committed_byte_capacity: usize,
    pub minimum_alignment: usize,
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
        })
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
    sparse: Arc<VulkanSparseStableResourceBacking>,
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
struct VulkanSparseStableResourcePlacement {
    resource_slots: Vec<usize>,
    resource_byte_offsets: Vec<usize>,
    resource_byte_counts: Vec<usize>,
    group_byte_offset: usize,
    group_byte_capacity: usize,
}

struct VulkanSparseStableResourceBacking {
    device_handle: vk::Device,
    requirements: VulkanSparseResidentBufferRequirements,
    state: std::sync::Mutex<VulkanSparseStableResourceState>,
}

struct VulkanSparseStableResourceState {
    buffer: Option<Arc<VulkanResidentBuffer>>,
    placements: BTreeMap<Vec<usize>, VulkanSparseStableResourcePlacement>,
    partitioned_placements: Vec<VulkanSparsePartitionedResourcePlacement>,
    resident_groups: BTreeSet<Vec<usize>>,
    active_groups: BTreeMap<Vec<usize>, usize>,
    blocks: Vec<Arc<VulkanSparseResidentMemoryBlock>>,
    allocations: BTreeMap<u64, usize>,
    next_allocation_id: u64,
    committed_byte_capacity: usize,
    allocated_byte_count: usize,
}

#[derive(Clone)]
struct VulkanSparsePartitionedResourcePlacement {
    member_slot_bases: Vec<usize>,
    resource_byte_offsets: Vec<usize>,
    resource_byte_counts: Vec<usize>,
    partition_count: usize,
    groups_byte_offset: usize,
    group_byte_capacity: usize,
}

pub struct VulkanStableResourceAllocation {
    allocation_id: u64,
    group_key: Arc<[usize]>,
    buffer: Arc<VulkanResidentBuffer>,
    sparse_backing: Arc<VulkanSparseStableResourceBacking>,
    byte_offset: usize,
    byte_count: usize,
    device_address: vk::DeviceAddress,
}

impl VulkanStableResourceArena {
    pub fn new(
        device: &VulkanComputeDevice,
        config: VulkanStableResourceArenaConfig,
        groups: &[VulkanStableResourceGroupLayout],
    ) -> Result<Self, VulkanError> {
        if groups.is_empty() {
            return Err(VulkanError(
                "sparse stable resource arena has no addressable groups"
                    .to_string(),
            ));
        }
        if !device.supports_buffer_device_address()
            || !device.supports_sparse_buffer_residency()
        {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot host demand-backed stable resources",
                device.device_name()
            )));
        }
        let (probe, requirements) = device
            .create_sparse_addressable_resident_buffer(
                config.minimum_alignment,
            )?;
        drop(probe);
        let page_alignment = requirements
            .byte_alignment
            .max(config.minimum_alignment);
        let mut placements = BTreeMap::new();
        let mut partitioned_placements = Vec::new();
        let mut claimed_slot_ranges = Vec::new();
        let mut virtual_byte_capacity = 0usize;
        for group in groups {
            match group {
                VulkanStableResourceGroupLayout::Explicit {
                    resource_slots,
                    resource_byte_counts,
                } => {
                    let (member_offsets, group_byte_capacity) =
                        sparse_group_member_layout(
                            resource_byte_counts,
                            config.minimum_alignment,
                            page_alignment,
                        )?;
                    if resource_slots.is_empty()
                        || resource_slots.len() != resource_byte_counts.len()
                    {
                        return Err(VulkanError(
                            "explicit sparse stable resource group layout is invalid"
                                .to_string(),
                        ));
                    }
                    virtual_byte_capacity = align_stable_resource_offset(
                        virtual_byte_capacity,
                        page_alignment,
                    )?;
                    let group_byte_offset = virtual_byte_capacity;
                    virtual_byte_capacity = virtual_byte_capacity
                        .checked_add(group_byte_capacity)
                        .ok_or_else(|| {
                            VulkanError(
                                "sparse stable resource virtual capacity overflowed"
                                    .to_string(),
                            )
                        })?;
                    let resource_byte_offsets = member_offsets
                        .into_iter()
                        .map(|offset| group_byte_offset + offset)
                        .collect();
                    for slot in resource_slots {
                        claimed_slot_ranges.push((
                            *slot,
                            slot.checked_add(1).ok_or_else(|| {
                                VulkanError(
                                    "explicit sparse address slot range overflowed"
                                        .to_string(),
                                )
                            })?,
                        ));
                    }
                    let placement = VulkanSparseStableResourcePlacement {
                        resource_slots: resource_slots.clone(),
                        resource_byte_offsets,
                        resource_byte_counts: resource_byte_counts.clone(),
                        group_byte_offset,
                        group_byte_capacity,
                    };
                    let mut group_key = resource_slots.clone();
                    group_key.sort_unstable();
                    if placements.insert(group_key, placement).is_some() {
                        return Err(VulkanError(
                            "explicit sparse stable resource group layout is duplicated"
                                .to_string(),
                        ));
                    }
                }
                VulkanStableResourceGroupLayout::Partitioned {
                    member_slot_bases,
                    resource_byte_counts,
                    partition_count,
                } => {
                    let (resource_byte_offsets, group_byte_capacity) =
                        sparse_group_member_layout(
                            resource_byte_counts,
                            config.minimum_alignment,
                            page_alignment,
                        )?;
                    if member_slot_bases.is_empty()
                        || member_slot_bases.len()
                            != resource_byte_counts.len()
                        || *partition_count == 0
                    {
                        return Err(VulkanError(
                            "partitioned sparse stable resource group layout is invalid"
                                .to_string(),
                        ));
                    }
                    virtual_byte_capacity = align_stable_resource_offset(
                        virtual_byte_capacity,
                        page_alignment,
                    )?;
                    let groups_byte_offset = virtual_byte_capacity;
                    virtual_byte_capacity = group_byte_capacity
                        .checked_mul(*partition_count)
                        .and_then(|bytes| {
                            virtual_byte_capacity.checked_add(bytes)
                        })
                        .ok_or_else(|| {
                            VulkanError(
                                "partitioned sparse resource virtual capacity overflowed"
                                    .to_string(),
                            )
                        })?;
                    for base in member_slot_bases {
                        claimed_slot_ranges.push((
                            *base,
                            base.checked_add(*partition_count).ok_or_else(
                                || {
                                    VulkanError(
                                        "partitioned sparse address slot range overflowed"
                                            .to_string(),
                                    )
                                },
                            )?,
                        ));
                    }
                    partitioned_placements.push(
                        VulkanSparsePartitionedResourcePlacement {
                            member_slot_bases: member_slot_bases.clone(),
                            resource_byte_offsets,
                            resource_byte_counts:
                                resource_byte_counts.clone(),
                            partition_count: *partition_count,
                            groups_byte_offset,
                            group_byte_capacity,
                        },
                    );
                }
            }
        }
        claimed_slot_ranges.sort_unstable();
        if claimed_slot_ranges
            .windows(2)
            .any(|ranges| ranges[0].1 > ranges[1].0)
        {
            return Err(VulkanError(
                "sparse stable resource layouts assign one address slot to multiple groups"
                    .to_string(),
            ));
        }
        let (buffer, final_requirements) = device
            .create_sparse_addressable_resident_buffer(
                virtual_byte_capacity,
            )?;
        if final_requirements != requirements {
            return Err(VulkanError(format!(
                "sparse stable resource requirements changed with capacity: probe={requirements:?}, final={final_requirements:?}"
            )));
        }
        let sparse = Arc::new(VulkanSparseStableResourceBacking {
            device_handle: device.device.handle(),
            requirements,
            state: std::sync::Mutex::new(
                VulkanSparseStableResourceState {
                    buffer: Some(Arc::new(buffer)),
                    placements,
                    partitioned_placements,
                    resident_groups: BTreeSet::new(),
                    active_groups: BTreeMap::new(),
                    blocks: Vec::new(),
                    allocations: BTreeMap::new(),
                    next_allocation_id: 0,
                    committed_byte_capacity: 0,
                    allocated_byte_count: 0,
                },
            ),
        });
        Ok(Self {
            config,
            sparse,
        })
    }

    pub fn config(&self) -> VulkanStableResourceArenaConfig {
        self.config
    }

    pub fn stats(&self) -> Result<VulkanStableResourceArenaStats, VulkanError> {
        let state = self.sparse.state.lock().map_err(|_| {
            VulkanError(
                "sparse stable resource arena state lock was poisoned"
                    .to_string(),
            )
        })?;
        Ok(VulkanStableResourceArenaStats {
            committed_byte_capacity: state.committed_byte_capacity,
            allocated_byte_count: state.allocated_byte_count,
            active_allocation_count: state.allocations.len(),
            chunk_count: state.blocks.len(),
        })
    }

    pub fn maximum_backed_byte_capacity(
        &self,
    ) -> Result<usize, VulkanError> {
        let state = self.sparse.state.lock().map_err(|_| {
            VulkanError(
                "sparse stable resource arena state lock was poisoned"
                    .to_string(),
            )
        })?;
        let explicit = state.placements.values().try_fold(
            0usize,
            |total, placement| {
                total
                    .checked_add(placement.group_byte_capacity)
                    .ok_or_else(|| {
                        VulkanError(
                            "sparse stable resource maximum capacity overflowed"
                                .to_string(),
                        )
                    })
            },
        )?;
        state.partitioned_placements.iter().try_fold(
            explicit,
            |total, placement| {
                placement
                    .group_byte_capacity
                    .checked_mul(placement.partition_count)
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| {
                        VulkanError(
                            "partitioned sparse resource maximum capacity overflowed"
                                .to_string(),
                        )
                    })
            },
        )
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
        allocate_sparse_stable_resource_groups(
            &self.sparse,
            device,
            &self.config,
            groups,
            alignment,
        )
    }

    pub fn release_backing(
        &self,
    ) -> Result<(), VulkanError> {
        let (buffer, blocks) = {
            let mut state = self.sparse.state.lock().map_err(|_| {
                VulkanError(
                    "sparse stable resource arena state lock was poisoned"
                        .to_string(),
                )
            })?;
            if !state.allocations.is_empty()
                || !state.active_groups.is_empty()
                || state.allocated_byte_count != 0
            {
                return Err(VulkanError(format!(
                    "sparse stable resource arena still owns {} allocations and {} payload bytes",
                    state.allocations.len(),
                    state.allocated_byte_count
                )));
            }
            state.resident_groups.clear();
            state.committed_byte_capacity = 0;
            (
                state.buffer.take(),
                std::mem::take(&mut state.blocks),
            )
        };
        drop(buffer);
        drop(blocks);
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
        .ok_or_else(|| {
            VulkanError(
                "stable resource aligned offset overflowed".to_string(),
            )
        })
}

fn sparse_group_member_layout(
    resource_byte_counts: &[usize],
    minimum_alignment: usize,
    page_alignment: usize,
) -> Result<(Vec<usize>, usize), VulkanError> {
    if resource_byte_counts.is_empty()
        || resource_byte_counts
            .iter()
            .any(|byte_count| *byte_count == 0)
    {
        return Err(VulkanError(
            "sparse stable resource group byte layout is empty".to_string(),
        ));
    }
    let mut byte_capacity = 0usize;
    let mut offsets = Vec::with_capacity(resource_byte_counts.len());
    for byte_count in resource_byte_counts {
        byte_capacity =
            align_stable_resource_offset(byte_capacity, minimum_alignment)?;
        offsets.push(byte_capacity);
        byte_capacity = byte_capacity.checked_add(*byte_count).ok_or_else(|| {
            VulkanError(
                "sparse stable resource group capacity overflowed".to_string(),
            )
        })?;
    }
    byte_capacity =
        align_stable_resource_offset(byte_capacity, page_alignment)?;
    Ok((offsets, byte_capacity))
}

fn sparse_stable_resource_placement_for_slots(
    state: &VulkanSparseStableResourceState,
    requested_slots: &[usize],
    sorted_slots: &[usize],
) -> Result<VulkanSparseStableResourcePlacement, VulkanError> {
    if let Some(placement) = state.placements.get(sorted_slots) {
        return Ok(placement.clone());
    }
    'partitioned_layout: for partitioned in
        &state.partitioned_placements
    {
        if requested_slots.len() != partitioned.member_slot_bases.len() {
            continue;
        }
        let mut partition_index = None;
        let mut requested_member_indices =
            Vec::with_capacity(requested_slots.len());
        let mut seen_member_indices = BTreeSet::new();
        for slot in requested_slots {
            let Some((member_index, selected_partition_index)) =
                partitioned
                    .member_slot_bases
                    .iter()
                    .enumerate()
                    .find_map(|(member_index, base)| {
                        slot.checked_sub(*base)
                            .filter(|index| {
                                *index < partitioned.partition_count
                            })
                            .map(|index| (member_index, index))
                    })
            else {
                continue 'partitioned_layout;
            };
            if !seen_member_indices.insert(member_index)
                || partition_index
                    .is_some_and(|index| {
                        index != selected_partition_index
                    })
            {
                continue 'partitioned_layout;
            }
            partition_index = Some(selected_partition_index);
            requested_member_indices.push(member_index);
        }
        let partition_index =
            partition_index.expect("partition layout has requested members");
        let group_byte_offset = partitioned
            .group_byte_capacity
            .checked_mul(partition_index)
            .and_then(|offset| {
                partitioned.groups_byte_offset.checked_add(offset)
            })
            .ok_or_else(|| {
                VulkanError(
                    "partitioned sparse resource placement overflowed".to_string(),
                )
            })?;
        let resource_byte_offsets = requested_member_indices
            .iter()
            .map(|member_index| {
                group_byte_offset
                    .checked_add(
                        partitioned.resource_byte_offsets[*member_index],
                    )
                    .ok_or_else(|| {
                        VulkanError(
                            "partitioned sparse member offset overflowed"
                                .to_string(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let resource_byte_counts = requested_member_indices
            .iter()
            .map(|member_index| {
                partitioned.resource_byte_counts[*member_index]
            })
            .collect();
        return Ok(VulkanSparseStableResourcePlacement {
            resource_slots: requested_slots.to_vec(),
            resource_byte_offsets,
            resource_byte_counts,
            group_byte_offset,
            group_byte_capacity: partitioned.group_byte_capacity,
        });
    }
    Err(VulkanError(
        "sparse stable resource group has no compiled virtual placement"
            .to_string(),
    ))
}

fn allocate_sparse_stable_resource_groups(
    sparse: &Arc<VulkanSparseStableResourceBacking>,
    device: &VulkanComputeDevice,
    config: &VulkanStableResourceArenaConfig,
    groups: &[(&[usize], &[usize])],
    alignment: usize,
) -> Result<Vec<Vec<Arc<VulkanStableResourceAllocation>>>, VulkanError> {
    if device.device.handle() != sparse.device_handle {
        return Err(VulkanError(
            "sparse stable resources were requested from another logical device"
                .to_string(),
        ));
    }
    if alignment != config.minimum_alignment {
        return Err(VulkanError(format!(
            "sparse stable resource allocation alignment {alignment} differs from its compiled layout alignment {}",
            config.minimum_alignment
        )));
    }
    let mut state = sparse.state.lock().map_err(|_| {
        VulkanError(
            "sparse stable resource arena state lock was poisoned".to_string(),
        )
    })?;
    let buffer = state.buffer.as_ref().cloned().ok_or_else(|| {
        VulkanError(
            "sparse stable resource arena backing was already released"
                .to_string(),
        )
    })?;
    let mut requested_keys = BTreeSet::new();
    let mut placements = Vec::with_capacity(groups.len());
    let mut physical_byte_count = 0usize;
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
                "sparse stable resource group allocation is duplicated or invalid"
                    .to_string(),
            ));
        }
        let already_backed = state.resident_groups.contains(&group_key);
        let placement =
            sparse_stable_resource_placement_for_slots(
                &state,
                slots,
                &group_key,
            )?;
        for (slot, byte_count) in slots.iter().zip(*byte_counts) {
            let placement_index = placement
                .resource_slots
                .iter()
                .position(|candidate| candidate == slot)
                .ok_or_else(|| {
                    VulkanError(
                        "sparse stable resource request contains an unknown group member"
                            .to_string(),
                    )
                })?;
            if placement.resource_byte_counts[placement_index]
                != *byte_count
            {
                return Err(VulkanError(
                    "sparse stable resource request differs from its compiled virtual placement"
                        .to_string(),
                ));
            }
            payload_byte_count = payload_byte_count
                .checked_add(*byte_count)
                .ok_or_else(|| {
                    VulkanError(
                        "sparse stable resource payload capacity overflowed"
                            .to_string(),
                    )
                })?;
        }
        if !already_backed {
            physical_byte_count = physical_byte_count
                .checked_add(placement.group_byte_capacity)
                .ok_or_else(|| {
                    VulkanError(
                        "sparse stable resource backing capacity overflowed"
                            .to_string(),
                    )
                })?;
        }
        resource_count =
            resource_count.checked_add(slots.len()).ok_or_else(|| {
                VulkanError(
                    "sparse stable resource allocation count overflowed"
                        .to_string(),
                )
            })?;
        placements.push((
            placement,
            group_key,
            slots.to_vec(),
            already_backed,
        ));
    }
    let committed_byte_capacity = state
        .committed_byte_capacity
        .checked_add(physical_byte_count)
        .ok_or_else(|| {
            VulkanError(
                "sparse stable resource committed capacity overflowed"
                    .to_string(),
            )
        })?;
    if committed_byte_capacity > config.committed_byte_capacity {
        return Err(VulkanError(format!(
            "sparse stable resources need {physical_byte_count} additional physical bytes, but {} of {} bytes are already committed",
            state.committed_byte_capacity,
            config.committed_byte_capacity
        )));
    }
    state
        .next_allocation_id
        .checked_add(u64::try_from(resource_count).map_err(|_| {
            VulkanError(
                "sparse stable resource allocation count exceeds u64".to_string(),
            )
        })?)
        .ok_or_else(|| {
            VulkanError(
                "sparse stable resource allocation ids exhausted".to_string(),
            )
        })?;
    let allocated_byte_count = state
        .allocated_byte_count
        .checked_add(payload_byte_count)
        .ok_or_else(|| {
            VulkanError(
                "sparse stable resource allocated payload overflowed"
                    .to_string(),
            )
        })?;
    let base_address = buffer.device_address()?;
    for (placement, _, _, _) in &placements {
        for byte_offset in &placement.resource_byte_offsets {
            base_address
                .checked_add(u64::try_from(*byte_offset).map_err(|_| {
                    VulkanError(
                        "sparse stable resource offset exceeds u64".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    VulkanError(
                        "sparse stable resource address overflowed".to_string(),
                    )
                })?;
        }
    }
    let block = if physical_byte_count == 0 {
        None
    } else {
        let block = Arc::new(device.allocate_sparse_addressable_memory(
            physical_byte_count,
            sparse.requirements,
        )?);
        let mut block_byte_offset = 0usize;
        let binds = placements
            .iter()
            .filter(|(_, _, _, already_backed)| !already_backed)
            .map(|(placement, _, _, _)| {
                let binding = VulkanSparseResidentBufferBind {
                    resource_byte_offset: placement.group_byte_offset,
                    byte_count: placement.group_byte_capacity,
                    memory: block.as_ref(),
                    memory_byte_offset: block_byte_offset,
                };
                block_byte_offset += placement.group_byte_capacity;
                binding
            })
            .collect::<Vec<_>>();
        device.bind_sparse_resident_buffer_ranges(&buffer, &binds)?;
        Some(block)
    };

    let mut allocation_groups = Vec::with_capacity(placements.len());
    for (placement, group_key, requested_slots, _) in placements {
        let mut allocations =
            Vec::with_capacity(placement.resource_slots.len());
        let allocation_group_key = Arc::<[usize]>::from(group_key.clone());
        for slot in requested_slots {
            let placement_index = placement
                .resource_slots
                .iter()
                .position(|candidate| *candidate == slot)
                .expect("sparse group member was prevalidated");
            let byte_offset =
                placement.resource_byte_offsets[placement_index];
            let byte_count =
                placement.resource_byte_counts[placement_index];
            let allocation_id = state.next_allocation_id;
            state.next_allocation_id += 1;
            state.allocations.insert(allocation_id, byte_count);
            allocations.push(Arc::new(VulkanStableResourceAllocation {
                allocation_id,
                group_key: Arc::clone(&allocation_group_key),
                buffer: Arc::clone(&buffer),
                sparse_backing: Arc::clone(sparse),
                byte_offset,
                byte_count,
                device_address: base_address
                    .checked_add(
                        u64::try_from(byte_offset)
                            .expect("sparse resource offset was prevalidated"),
                    )
                    .expect("sparse resource address was prevalidated"),
            }));
        }
        state
            .active_groups
            .insert(group_key.clone(), allocations.len());
        state.resident_groups.insert(group_key);
        allocation_groups.push(allocations);
    }
    state.allocated_byte_count = allocated_byte_count;
    state.committed_byte_capacity = committed_byte_capacity;
    if let Some(block) = block {
        state.blocks.push(block);
    }
    Ok(allocation_groups)
}

impl Drop for VulkanSparseStableResourceBacking {
    fn drop(&mut self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let buffer = state.buffer.take();
        let blocks = std::mem::take(&mut state.blocks);
        drop(state);
        drop(buffer);
        drop(blocks);
    }
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

    fn allocation_id(&self) -> u64 {
        self.allocation_id
    }
}

impl Drop for VulkanStableResourceAllocation {
    fn drop(&mut self) {
        let Ok(mut state) = self.sparse_backing.state.lock() else {
            return;
        };
        let Some(byte_count) =
            state.allocations.remove(&self.allocation_id)
        else {
            debug_assert!(
                false,
                "sparse stable resource allocation was released twice"
            );
            return;
        };
        debug_assert_eq!(byte_count, self.byte_count);
        state.allocated_byte_count =
            state.allocated_byte_count.saturating_sub(byte_count);
        let mut remove_active_group = false;
        if let Some(active_allocation_count) =
            state.active_groups.get_mut(self.group_key.as_ref())
        {
            *active_allocation_count =
                active_allocation_count.saturating_sub(1);
            remove_active_group = *active_allocation_count == 0;
        } else {
            debug_assert!(
                false,
                "sparse stable resource allocation lost its active group"
            );
        }
        if remove_active_group {
            state.active_groups.remove(self.group_key.as_ref());
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
        let mut bytes =
            [0u8; VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT];
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
                "stable resource address table must have at least one slot"
                    .to_string(),
            ));
        }
        let byte_capacity = slot_count
            .checked_mul(VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT)
            .ok_or_else(|| {
                VulkanError(
                    "stable resource address table capacity overflowed".to_string(),
                )
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

    pub fn record(
        &self,
        slot: usize,
    ) -> Result<VulkanStableResourceAddressRecord, VulkanError> {
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
            if previous.resident != 0
                || self.resident_allocations[*slot].is_some()
            {
                return Err(VulkanError(format!(
                    "stable resource address table slot {slot} is already resident"
                )));
            }
            if allocation.buffer.device.handle()
                != self.buffer.device.handle()
            {
                return Err(VulkanError(format!(
                    "stable resource address table slot {slot} cannot publish an allocation from another logical device"
                )));
            }
            let generation = previous.generation.checked_add(1).ok_or_else(|| {
                VulkanError(format!(
                    "stable resource address table slot {slot} exhausted its generations"
                ))
            })?;
            let byte_count =
                u64::try_from(allocation.byte_count()).map_err(|_| {
                    VulkanError(
                        "stable resource byte count exceeds u64".to_string(),
                    )
                })?;
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
            let record = self
                .records
                .get(publication.slot)
                .copied()
                .ok_or_else(|| {
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
                    generation: record.generation.checked_add(1).ok_or_else(
                        || {
                            VulkanError(format!(
                                "stable resource address table slot {} exhausted its generations",
                                publication.slot
                            ))
                        },
                    )?,
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
        updates: &[
            (
                usize,
                VulkanStableResourceAddressRecord,
                Option<Arc<VulkanStableResourceAllocation>>,
            )
        ],
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
                    slot.checked_mul(
                        VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT,
                    )
                    .ok_or_else(|| {
                        VulkanError(
                            "stable resource address table byte offset overflowed"
                                .to_string(),
                        )
                    })?,
                    bytes,
                )
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        let provisional_slots = updates
            .iter()
            .filter_map(|(slot, _, allocation)| {
                allocation.as_ref().map(|allocation| {
                    self.resident_allocations[*slot] =
                        Some(Arc::clone(allocation));
                    *slot
                })
            })
            .collect::<Vec<_>>();
        if let Err(failure) = transfer.submit_consumer_serialized(&writes) {
            if !failure.submission_accepted {
                for slot in provisional_slots {
                    self.resident_allocations[slot] = None;
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
}

fn validate_stable_resource_alignment(
    alignment: usize,
) -> Result<(), VulkanError> {
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
