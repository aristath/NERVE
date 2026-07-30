const VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanStableResourceArenaConfig {
    pub chunk_byte_capacity: usize,
    pub committed_byte_capacity: usize,
    pub minimum_alignment: usize,
}

impl VulkanStableResourceArenaConfig {
    pub fn new(
        chunk_byte_capacity: usize,
        committed_byte_capacity: usize,
        minimum_alignment: usize,
    ) -> Result<Self, VulkanError> {
        if chunk_byte_capacity == 0 {
            return Err(VulkanError(
                "stable resource arena chunk capacity must not be zero".to_string(),
            ));
        }
        if committed_byte_capacity == 0 {
            return Err(VulkanError(
                "stable resource arena committed capacity must not be zero".to_string(),
            ));
        }
        if chunk_byte_capacity > committed_byte_capacity {
            return Err(VulkanError(format!(
                "stable resource arena chunk capacity {chunk_byte_capacity} exceeds committed capacity {committed_byte_capacity}"
            )));
        }
        validate_stable_resource_alignment(minimum_alignment)?;
        Ok(Self {
            chunk_byte_capacity,
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
    device_handle: vk::Device,
    config: VulkanStableResourceArenaConfig,
    state: Arc<std::sync::Mutex<VulkanStableResourceArenaState>>,
}

struct VulkanStableResourceArenaState {
    chunks: BTreeMap<u64, VulkanStableResourceArenaChunk>,
    allocations: BTreeMap<u64, VulkanStableResourceArenaPlacement>,
    next_chunk_id: u64,
    next_allocation_id: u64,
    committed_byte_capacity: usize,
    allocated_byte_count: usize,
}

struct VulkanStableResourceArenaChunk {
    buffer: Arc<VulkanResidentBuffer>,
    base_device_address: vk::DeviceAddress,
    byte_capacity: usize,
    free_ranges: BTreeMap<usize, usize>,
    allocation_count: usize,
}

#[derive(Clone, Copy)]
struct VulkanStableResourceArenaPlacement {
    chunk_id: u64,
    byte_offset: usize,
    byte_count: usize,
}

pub struct VulkanStableResourceAllocation {
    arena: std::sync::Weak<
        std::sync::Mutex<VulkanStableResourceArenaState>,
    >,
    allocation_id: u64,
    buffer: Arc<VulkanResidentBuffer>,
    byte_offset: usize,
    byte_count: usize,
    device_address: vk::DeviceAddress,
}

impl VulkanStableResourceArena {
    pub fn new(
        device: &VulkanComputeDevice,
        config: VulkanStableResourceArenaConfig,
    ) -> Result<Self, VulkanError> {
        if !device.supports_buffer_device_address() {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot host a stable resource arena because buffer device addresses are unavailable",
                device.device_name()
            )));
        }
        Ok(Self {
            device_handle: device.device.handle(),
            config,
            state: Arc::new(std::sync::Mutex::new(
                VulkanStableResourceArenaState {
                    chunks: BTreeMap::new(),
                    allocations: BTreeMap::new(),
                    next_chunk_id: 0,
                    next_allocation_id: 0,
                    committed_byte_capacity: 0,
                    allocated_byte_count: 0,
                },
            )),
        })
    }

    pub fn config(&self) -> VulkanStableResourceArenaConfig {
        self.config
    }

    pub fn allocate(
        &self,
        device: &VulkanComputeDevice,
        byte_count: usize,
        alignment: usize,
    ) -> Result<VulkanStableResourceAllocation, VulkanError> {
        if device.device.handle() != self.device_handle {
            return Err(VulkanError(
                "stable resource allocation requested from another logical device".to_string(),
            ));
        }
        if byte_count == 0 {
            return Err(VulkanError(
                "stable resource allocation byte count must not be zero".to_string(),
            ));
        }
        validate_stable_resource_alignment(alignment)?;
        let alignment = alignment.max(self.config.minimum_alignment);
        let mut state = self.lock_state()?;
        state.next_allocation_id.checked_add(1).ok_or_else(|| {
            VulkanError("stable resource allocation ids exhausted".to_string())
        })?;
        state
            .allocated_byte_count
            .checked_add(byte_count)
            .ok_or_else(|| {
                VulkanError(
                    "stable resource allocated byte count overflowed".to_string(),
                )
            })?;

        if let Some((chunk_id, byte_offset)) =
            best_stable_resource_range(&state.chunks, byte_count, alignment)?
        {
            return reserve_stable_resource_range(
                &self.state,
                &mut state,
                chunk_id,
                byte_offset,
                byte_count,
            );
        }

        let alignment_slack = alignment.checked_sub(1).ok_or_else(|| {
            VulkanError("stable resource alignment underflowed".to_string())
        })?;
        let minimum_new_chunk_capacity =
            byte_count.checked_add(alignment_slack).ok_or_else(|| {
                VulkanError(
                    "stable resource allocation capacity overflowed".to_string(),
                )
            })?;
        let remaining_capacity = self
            .config
            .committed_byte_capacity
            .checked_sub(state.committed_byte_capacity)
            .ok_or_else(|| {
                VulkanError(
                    "stable resource arena committed capacity is internally inconsistent"
                        .to_string(),
                )
            })?;
        if remaining_capacity < minimum_new_chunk_capacity {
            return Err(VulkanError(format!(
                "stable resource arena needs at least {minimum_new_chunk_capacity} additional bytes for a {byte_count}-byte allocation aligned to {alignment}, but only {remaining_capacity} committed bytes remain"
            )));
        }
        let new_chunk_capacity = self
            .config
            .chunk_byte_capacity
            .max(minimum_new_chunk_capacity)
            .min(remaining_capacity);
        let buffer = Arc::new(
            device.create_addressable_resident_buffer(new_chunk_capacity)?,
        );
        let base_device_address = buffer.device_address()?;
        let chunk_id = state.next_chunk_id;
        state.next_chunk_id =
            state.next_chunk_id.checked_add(1).ok_or_else(|| {
                VulkanError("stable resource arena chunk ids exhausted".to_string())
            })?;
        state.committed_byte_capacity = state
            .committed_byte_capacity
            .checked_add(new_chunk_capacity)
            .ok_or_else(|| {
                VulkanError(
                    "stable resource committed capacity overflowed".to_string(),
                )
            })?;
        state.chunks.insert(
            chunk_id,
            VulkanStableResourceArenaChunk {
                buffer,
                base_device_address,
                byte_capacity: new_chunk_capacity,
                free_ranges: BTreeMap::from([(0, new_chunk_capacity)]),
                allocation_count: 0,
            },
        );
        let byte_offset = best_stable_resource_range(
            &state.chunks,
            byte_count,
            alignment,
        )?
        .filter(|(candidate_chunk_id, _)| *candidate_chunk_id == chunk_id)
        .map(|(_, byte_offset)| byte_offset)
        .ok_or_else(|| {
            VulkanError(
                "new stable resource arena chunk cannot satisfy its allocation"
                    .to_string(),
            )
        })?;
        reserve_stable_resource_range(
            &self.state,
            &mut state,
            chunk_id,
            byte_offset,
            byte_count,
        )
    }

    pub fn stats(&self) -> Result<VulkanStableResourceArenaStats, VulkanError> {
        let state = self.lock_state()?;
        Ok(VulkanStableResourceArenaStats {
            committed_byte_capacity: state.committed_byte_capacity,
            allocated_byte_count: state.allocated_byte_count,
            active_allocation_count: state.allocations.len(),
            chunk_count: state.chunks.len(),
        })
    }

    fn lock_state(
        &self,
    ) -> Result<
        std::sync::MutexGuard<'_, VulkanStableResourceArenaState>,
        VulkanError,
    > {
        self.state.lock().map_err(|_| {
            VulkanError("stable resource arena state lock was poisoned".to_string())
        })
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
        let Some(arena) = self.arena.upgrade() else {
            return;
        };
        let Ok(mut state) = arena.lock() else {
            return;
        };
        let Some(placement) = state.allocations.remove(&self.allocation_id)
        else {
            debug_assert!(false, "stable resource allocation was released twice");
            return;
        };
        debug_assert_eq!(placement.byte_offset, self.byte_offset);
        debug_assert_eq!(placement.byte_count, self.byte_count);
        let remove_chunk = if let Some(chunk) =
            state.chunks.get_mut(&placement.chunk_id)
        {
            release_stable_resource_range(
                &mut chunk.free_ranges,
                placement.byte_offset,
                placement.byte_count,
            );
            chunk.allocation_count = chunk.allocation_count.saturating_sub(1);
            chunk.allocation_count == 0
        } else {
            debug_assert!(false, "stable resource allocation lost its chunk");
            false
        };
        state.allocated_byte_count =
            state.allocated_byte_count.saturating_sub(placement.byte_count);
        if remove_chunk
            && let Some(chunk) = state.chunks.remove(&placement.chunk_id)
        {
            state.committed_byte_capacity = state
                .committed_byte_capacity
                .saturating_sub(chunk.byte_capacity);
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

fn best_stable_resource_range(
    chunks: &BTreeMap<u64, VulkanStableResourceArenaChunk>,
    byte_count: usize,
    alignment: usize,
) -> Result<Option<(u64, usize)>, VulkanError> {
    let alignment_mask = u64::try_from(alignment - 1)
        .map_err(|_| VulkanError("stable resource alignment exceeds u64".to_string()))?;
    let mut best: Option<(usize, u64, usize)> = None;
    for (chunk_id, chunk) in chunks {
        for (free_offset, free_byte_count) in &chunk.free_ranges {
            let free_address = chunk
                .base_device_address
                .checked_add(u64::try_from(*free_offset).map_err(|_| {
                    VulkanError(
                        "stable resource free offset exceeds u64".to_string(),
                    )
                })?)
                .ok_or_else(|| {
                    VulkanError(
                        "stable resource device address overflowed".to_string(),
                    )
                })?;
            let aligned_address = free_address
                .checked_add(alignment_mask)
                .map(|address| address & !alignment_mask)
                .ok_or_else(|| {
                    VulkanError(
                        "stable resource aligned address overflowed".to_string(),
                    )
                })?;
            let alignment_padding =
                usize::try_from(aligned_address - free_address).map_err(|_| {
                    VulkanError(
                        "stable resource alignment padding exceeds usize"
                            .to_string(),
                    )
                })?;
            let required = alignment_padding.checked_add(byte_count).ok_or_else(
                || {
                    VulkanError(
                        "stable resource aligned byte count overflowed".to_string(),
                    )
                },
            )?;
            if required > *free_byte_count {
                continue;
            }
            let byte_offset =
                free_offset.checked_add(alignment_padding).ok_or_else(|| {
                    VulkanError(
                        "stable resource aligned offset overflowed".to_string(),
                    )
                })?;
            let waste = free_byte_count - required;
            let candidate = (waste, *chunk_id, byte_offset);
            if best.is_none_or(|current| candidate < current) {
                best = Some(candidate);
            }
        }
    }
    Ok(best.map(|(_, chunk_id, byte_offset)| (chunk_id, byte_offset)))
}

fn reserve_stable_resource_range(
    arena: &Arc<std::sync::Mutex<VulkanStableResourceArenaState>>,
    state: &mut VulkanStableResourceArenaState,
    chunk_id: u64,
    byte_offset: usize,
    byte_count: usize,
) -> Result<VulkanStableResourceAllocation, VulkanError> {
    let chunk = state.chunks.get(&chunk_id).ok_or_else(|| {
        VulkanError("stable resource arena selected a missing chunk".to_string())
    })?;
    let (&free_offset, &free_byte_count) = chunk
        .free_ranges
        .range(..=byte_offset)
        .next_back()
        .filter(|(free_offset, free_byte_count)| {
            byte_offset >= **free_offset
                && byte_offset
                    .checked_add(byte_count)
                    .is_some_and(|end| end <= **free_offset + **free_byte_count)
        })
        .ok_or_else(|| {
            VulkanError(
                "stable resource arena selected an unavailable range".to_string(),
            )
        })?;
    let allocation_end = byte_offset.checked_add(byte_count).ok_or_else(|| {
        VulkanError("stable resource allocation end overflowed".to_string())
    })?;
    let free_end = free_offset
        .checked_add(free_byte_count)
        .ok_or_else(|| VulkanError("stable resource free range overflowed".to_string()))?;
    let allocation_count =
        chunk.allocation_count.checked_add(1).ok_or_else(|| {
            VulkanError("stable resource allocation count overflowed".to_string())
        })?;
    let allocation_id = state.next_allocation_id;
    let next_allocation_id =
        state.next_allocation_id.checked_add(1).ok_or_else(|| {
            VulkanError("stable resource allocation ids exhausted".to_string())
        })?;
    let allocated_byte_count = state
        .allocated_byte_count
        .checked_add(byte_count)
        .ok_or_else(|| {
            VulkanError("stable resource allocated byte count overflowed".to_string())
        })?;
    let device_address = chunk
        .base_device_address
        .checked_add(u64::try_from(byte_offset).map_err(|_| {
            VulkanError("stable resource byte offset exceeds u64".to_string())
        })?)
        .ok_or_else(|| {
            VulkanError("stable resource device address overflowed".to_string())
        })?;
    let buffer = Arc::clone(&chunk.buffer);

    let chunk = state
        .chunks
        .get_mut(&chunk_id)
        .expect("stable resource chunk was validated above");
    chunk.free_ranges.remove(&free_offset);
    if byte_offset > free_offset {
        chunk
            .free_ranges
            .insert(free_offset, byte_offset - free_offset);
    }
    if allocation_end < free_end {
        chunk
            .free_ranges
            .insert(allocation_end, free_end - allocation_end);
    }
    chunk.allocation_count = allocation_count;
    state.next_allocation_id = next_allocation_id;
    state.allocated_byte_count = allocated_byte_count;
    state.allocations.insert(
        allocation_id,
        VulkanStableResourceArenaPlacement {
            chunk_id,
            byte_offset,
            byte_count,
        },
    );
    Ok(VulkanStableResourceAllocation {
        arena: Arc::downgrade(arena),
        allocation_id,
        buffer,
        byte_offset,
        byte_count,
        device_address,
    })
}

fn release_stable_resource_range(
    free_ranges: &mut BTreeMap<usize, usize>,
    byte_offset: usize,
    byte_count: usize,
) {
    let mut merged_offset = byte_offset;
    let mut merged_end = byte_offset.saturating_add(byte_count);
    if let Some((&previous_offset, &previous_byte_count)) =
        free_ranges.range(..byte_offset).next_back()
    {
        let previous_end = previous_offset.saturating_add(previous_byte_count);
        if previous_end == byte_offset {
            merged_offset = previous_offset;
            free_ranges.remove(&previous_offset);
        }
    }
    if let Some((&next_offset, &next_byte_count)) =
        free_ranges.range(merged_end..).next()
        && next_offset == merged_end
    {
        merged_end = next_offset.saturating_add(next_byte_count);
        free_ranges.remove(&next_offset);
    }
    free_ranges.insert(merged_offset, merged_end - merged_offset);
}

const _: () = assert!(
    std::mem::size_of::<VulkanStableResourceAddressRecord>()
        == VULKAN_STABLE_RESOURCE_ADDRESS_RECORD_BYTE_COUNT
);
