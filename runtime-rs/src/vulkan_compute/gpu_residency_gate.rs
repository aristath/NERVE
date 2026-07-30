const VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT: u32 = 12;
const VULKAN_GPU_RESIDENCY_GATE_GROUP_RECORD_WORD_COUNT: usize = 2;
const VULKAN_GPU_RESIDENCY_GATE_RESOLVED_HEADER_WORD_COUNT: usize = 8;
const VULKAN_GPU_RESIDENCY_GATE_RESOLVED_RECORD_WORD_COUNT: usize = 8;
const VULKAN_GPU_RESIDENCY_GATE_MISS_HEADER_WORD_COUNT: usize = 4;
const VULKAN_GPU_RESIDENCY_GATE_MISS_RECORD_WORD_COUNT: usize = 2;
const VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT: usize = 8;
const VULKAN_GPU_RESIDENCY_GATE_CONFIG_DISPATCH_WORD_COUNT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyIndirectDispatch {
    pub byte_offset: usize,
    pub dimensions: [u32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyGateConfig {
    pub maximum_selection_count: usize,
    pub selection_count_per_lane: usize,
    pub selection_lane_stride_words: usize,
    pub selection_index_shift: u32,
    pub selection_index_mask: u32,
    pub address_slots_by_resource_index: Vec<Vec<usize>>,
    pub downstream_dispatches: Vec<VulkanGpuResidencyIndirectDispatch>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyMissingRequest {
    pub checkpoint_tag: u32,
    pub resource_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyMissingSnapshot {
    pub published_count: u32,
    pub consumed_count: u32,
    pub overflowed: bool,
    pub notification_epoch: u32,
    pub requests: Vec<VulkanGpuResidencyMissingRequest>,
}

#[derive(Clone)]
pub struct VulkanGpuResidencyMissQueue {
    capacity: usize,
    buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanGpuResidencyGate {
    config: VulkanGpuResidencyGateConfig,
    _selection_buffer: Arc<VulkanResidentBuffer>,
    _address_table_buffer: Arc<VulkanResidentBuffer>,
    _configuration: Arc<VulkanResidentBuffer>,
    _resource_group_records: Arc<VulkanResidentBuffer>,
    _resource_address_slots: Arc<VulkanResidentBuffer>,
    resolved_addresses: Arc<VulkanResidentBuffer>,
    missing_queue: VulkanGpuResidencyMissQueue,
    indirect_dispatches: Arc<VulkanResidentBuffer>,
    dispatch: VulkanResidentKernelDispatch,
}

impl VulkanGpuResidencyGateConfig {
    pub fn validate(
        &self,
        selection_buffer_byte_capacity: usize,
        address_table_slot_count: usize,
        missing_request_capacity: usize,
        indirect_dispatch_buffer_byte_capacity: usize,
    ) -> Result<(), VulkanError> {
        if self.maximum_selection_count == 0 {
            return Err(VulkanError(
                "GPU residency gate maximum selection count must not be zero".to_string(),
            ));
        }
        if self.maximum_selection_count > u32::MAX as usize {
            return Err(VulkanError(
                "GPU residency gate maximum selection count exceeds u32".to_string(),
            ));
        }
        if self.selection_count_per_lane == 0
            || self.maximum_selection_count % self.selection_count_per_lane != 0
            || self.selection_lane_stride_words < self.selection_count_per_lane
            || self.selection_count_per_lane > u32::MAX as usize
            || self.selection_lane_stride_words > u32::MAX as usize
        {
            return Err(VulkanError(
                "GPU residency gate selection lane layout is invalid".to_string(),
            ));
        }
        let lane_count = self.maximum_selection_count / self.selection_count_per_lane;
        let required_selection_words = lane_count
            .saturating_sub(1)
            .checked_mul(self.selection_lane_stride_words)
            .and_then(|offset| offset.checked_add(self.selection_count_per_lane))
            .ok_or_else(|| {
                VulkanError("GPU residency gate selection capacity overflowed".to_string())
            })?;
        let required_selection_bytes = required_selection_words
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                VulkanError("GPU residency gate selection capacity overflowed".to_string())
            })?;
        if required_selection_bytes > selection_buffer_byte_capacity {
            return Err(VulkanError(format!(
                "GPU residency gate requires {required_selection_bytes} selection bytes but the buffer has {selection_buffer_byte_capacity}"
            )));
        }
        if self.selection_index_shift >= u32::BITS || self.selection_index_mask == 0 {
            return Err(VulkanError(
                "GPU residency gate selection bit field is invalid".to_string(),
            ));
        }
        if self.address_slots_by_resource_index.is_empty() {
            return Err(VulkanError(
                "GPU residency gate must address at least one selectable resource".to_string(),
            ));
        }
        let maximum_resource_index = self.address_slots_by_resource_index.len() - 1;
        if u32::try_from(maximum_resource_index).map_or(true, |index| {
            index & self.selection_index_mask != index
        }) {
            return Err(VulkanError(format!(
                "GPU residency gate selection mask {:#010x} cannot represent resource index {maximum_resource_index}",
                self.selection_index_mask
            )));
        }
        for (resource_index, slots) in
            self.address_slots_by_resource_index.iter().enumerate()
        {
            if slots.is_empty() {
                return Err(VulkanError(format!(
                    "GPU residency gate resource {resource_index} has no address slots"
                )));
            }
            let mut unique = BTreeSet::new();
            for slot in slots {
                if *slot >= address_table_slot_count {
                    return Err(VulkanError(format!(
                        "GPU residency gate resource {resource_index} uses address slot {slot}, but the table has {address_table_slot_count} slots"
                    )));
                }
                if !unique.insert(*slot) {
                    return Err(VulkanError(format!(
                        "GPU residency gate resource {resource_index} repeats address slot {slot}"
                    )));
                }
            }
        }
        if missing_request_capacity == 0
            || missing_request_capacity < self.maximum_selection_count
            || missing_request_capacity > i32::MAX as usize
        {
            return Err(VulkanError(format!(
                "GPU residency miss capacity {} cannot safely hold one maximum-size selection of {} resources",
                missing_request_capacity, self.maximum_selection_count
            )));
        }
        if self.downstream_dispatches.is_empty()
            || self
                .downstream_dispatches
                .iter()
                .any(|dispatch| dispatch.dimensions.contains(&0))
        {
            return Err(VulkanError(
                "GPU residency gate downstream dispatch dimensions must be non-empty and nonzero"
                    .to_string(),
            ));
        }
        let mut target_offsets = BTreeSet::new();
        for dispatch in &self.downstream_dispatches {
            validate_resident_indirect_dispatch_range(
                indirect_dispatch_buffer_byte_capacity,
                dispatch.byte_offset,
            )?;
            if !target_offsets.insert(dispatch.byte_offset) {
                return Err(VulkanError(format!(
                    "GPU residency gate repeats downstream indirect byte offset {}",
                    dispatch.byte_offset
                )));
            }
            if u32::try_from(dispatch.byte_offset / size_of::<u32>()).is_err() {
                return Err(VulkanError(format!(
                    "GPU residency gate downstream indirect byte offset {} exceeds its shader representation",
                    dispatch.byte_offset
                )));
            }
        }
        Ok(())
    }

    fn maximum_resolved_address_count(&self) -> Result<usize, VulkanError> {
        let maximum_members = self
            .address_slots_by_resource_index
            .iter()
            .map(Vec::len)
            .max()
            .expect("validated resource map is non-empty");
        self.maximum_selection_count
            .checked_mul(maximum_members)
            .ok_or_else(|| {
                VulkanError("GPU residency resolved-address capacity overflowed".to_string())
            })
    }
}

impl VulkanGpuResidencyGate {
    pub fn new(
        device: &VulkanComputeDevice,
        spirv_words: &[u32],
        selection_buffer: Arc<VulkanResidentBuffer>,
        address_table_buffer: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
        missing_queue: VulkanGpuResidencyMissQueue,
        indirect_dispatches: Arc<VulkanResidentBuffer>,
        config: VulkanGpuResidencyGateConfig,
    ) -> Result<Self, VulkanError> {
        config.validate(
            selection_buffer.byte_capacity(),
            address_table_slot_count,
            missing_queue.capacity(),
            indirect_dispatches.byte_capacity(),
        )?;
        if !device.owns_resident_buffer(&address_table_buffer)
            || !device.owns_resident_buffer(missing_queue.buffer())
            || !device.owns_resident_buffer(&indirect_dispatches)
        {
            return Err(VulkanError(
                "GPU residency gate buffers belong to another logical device".to_string(),
            ));
        }

        let mut configuration_words = vec![
            config.selection_index_shift,
            config.selection_index_mask,
            u32::try_from(config.address_slots_by_resource_index.len()).map_err(|_| {
                VulkanError("GPU residency resource count exceeds u32".to_string())
            })?,
            u32::try_from(missing_queue.capacity()).map_err(|_| {
                VulkanError("GPU residency missing capacity exceeds u32".to_string())
            })?,
            u32::try_from(config.maximum_resolved_address_count()?).map_err(|_| {
                VulkanError("GPU residency resolved capacity exceeds u32".to_string())
            })?,
            u32::try_from(config.downstream_dispatches.len()).map_err(|_| {
                VulkanError("GPU residency downstream dispatch count exceeds u32".to_string())
            })?,
            u32::try_from(config.selection_count_per_lane).map_err(|_| {
                VulkanError("GPU residency selection count per lane exceeds u32".to_string())
            })?,
            u32::try_from(config.selection_lane_stride_words).map_err(|_| {
                VulkanError("GPU residency selection lane stride exceeds u32".to_string())
            })?,
        ];
        debug_assert_eq!(
            configuration_words.len(),
            VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT
        );
        configuration_words.extend(
            config
                .downstream_dispatches
                .iter()
                .flat_map(|dispatch| {
                    [
                        u32::try_from(dispatch.byte_offset / size_of::<u32>())
                            .expect("validated indirect word offset fits u32"),
                        dispatch.dimensions[0],
                        dispatch.dimensions[1],
                        dispatch.dimensions[2],
                    ]
                }),
        );
        debug_assert_eq!(
            configuration_words.len(),
            VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT
                + config.downstream_dispatches.len()
                    * VULKAN_GPU_RESIDENCY_GATE_CONFIG_DISPATCH_WORD_COUNT
        );
        let configuration = Arc::new(
            device.create_resident_buffer(words_byte_count(configuration_words.len())?)?,
        );
        configuration.write_bytes(&u32_words_bytes(&configuration_words))?;

        let mut resource_group_words = Vec::with_capacity(
            config.address_slots_by_resource_index.len()
                * VULKAN_GPU_RESIDENCY_GATE_GROUP_RECORD_WORD_COUNT,
        );
        let mut resource_address_slot_words = Vec::new();
        for slots in &config.address_slots_by_resource_index {
            resource_group_words.push(
                u32::try_from(resource_address_slot_words.len()).map_err(|_| {
                    VulkanError(
                        "GPU residency address-slot offset exceeds u32".to_string(),
                    )
                })?,
            );
            resource_group_words.push(u32::try_from(slots.len()).map_err(|_| {
                VulkanError("GPU residency resource member count exceeds u32".to_string())
            })?);
            resource_address_slot_words.extend(
                slots
                    .iter()
                    .map(|slot| {
                        u32::try_from(*slot).map_err(|_| {
                            VulkanError(
                                "GPU residency address slot exceeds u32".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }

        let resource_group_records = Arc::new(
            device.create_resident_buffer(words_byte_count(resource_group_words.len())?)?,
        );
        resource_group_records.write_bytes(&u32_words_bytes(&resource_group_words))?;
        let resource_address_slots = Arc::new(
            device.create_resident_buffer(words_byte_count(
                resource_address_slot_words.len(),
            )?)?,
        );
        resource_address_slots.write_bytes(&u32_words_bytes(&resource_address_slot_words))?;

        let resolved_word_count = VULKAN_GPU_RESIDENCY_GATE_RESOLVED_HEADER_WORD_COUNT
            .checked_add(
                config
                    .maximum_resolved_address_count()?
                    .checked_mul(VULKAN_GPU_RESIDENCY_GATE_RESOLVED_RECORD_WORD_COUNT)
                    .ok_or_else(|| {
                        VulkanError(
                            "GPU residency resolved record capacity overflowed".to_string(),
                        )
                    })?,
            )
            .ok_or_else(|| {
                VulkanError("GPU residency resolved buffer capacity overflowed".to_string())
            })?;
        let resolved_addresses =
            Arc::new(device.create_resident_buffer(words_byte_count(resolved_word_count)?)?);
        resolved_addresses.write_bytes(&vec![0; resolved_addresses.byte_capacity()])?;

        let bindings = [
            VulkanResidentKernelBufferBinding {
                binding: 0,
                buffer: &selection_buffer,
                byte_offset: 0,
                byte_len: selection_buffer.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
            VulkanResidentKernelBufferBinding {
                binding: 1,
                buffer: &address_table_buffer,
                byte_offset: 0,
                byte_len: address_table_buffer.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
            VulkanResidentKernelBufferBinding {
                binding: 2,
                buffer: &resource_group_records,
                byte_offset: 0,
                byte_len: resource_group_records.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
            VulkanResidentKernelBufferBinding {
                binding: 3,
                buffer: &resource_address_slots,
                byte_offset: 0,
                byte_len: resource_address_slots.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
            VulkanResidentKernelBufferBinding {
                binding: 4,
                buffer: &resolved_addresses,
                byte_offset: 0,
                byte_len: resolved_addresses.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Write,
            },
            VulkanResidentKernelBufferBinding {
                binding: 5,
                buffer: missing_queue.buffer(),
                byte_offset: 0,
                byte_len: missing_queue.buffer().byte_capacity(),
                access: VulkanResidentKernelBufferAccess::ReadWrite,
            },
            VulkanResidentKernelBufferBinding {
                binding: 6,
                buffer: &indirect_dispatches,
                byte_offset: 0,
                byte_len: indirect_dispatches.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Write,
            },
            VulkanResidentKernelBufferBinding {
                binding: 7,
                buffer: &configuration,
                byte_offset: 0,
                byte_len: configuration.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
        ];
        let dispatch = device.create_resident_kernel_dispatch_labeled(
            spirv_words,
            &bindings,
            1,
            1,
            VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT,
            Some("gpu_residency_gate".to_string()),
        )?;
        Ok(Self {
            config,
            _selection_buffer: selection_buffer,
            _address_table_buffer: address_table_buffer,
            _configuration: configuration,
            _resource_group_records: resource_group_records,
            _resource_address_slots: resource_address_slots,
            resolved_addresses,
            missing_queue,
            indirect_dispatches,
            dispatch,
        })
    }

    pub fn dispatch(&self) -> &VulkanResidentKernelDispatch {
        &self.dispatch
    }

    pub fn push_constants(
        &self,
        selection_count: usize,
        checkpoint_tag: u32,
        restore_downstream: bool,
    ) -> Result<[u8; VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT as usize], VulkanError>
    {
        if selection_count == 0 || selection_count > self.config.maximum_selection_count {
            return Err(VulkanError(format!(
                "GPU residency gate selection count {selection_count} is outside 1..={}",
                self.config.maximum_selection_count
            )));
        }
        let mut bytes =
            [0; VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT as usize];
        bytes[..4].copy_from_slice(
            &u32::try_from(selection_count)
                .map_err(|_| VulkanError("GPU residency selection count exceeds u32".to_string()))?
                .to_le_bytes(),
        );
        bytes[4..8].copy_from_slice(&checkpoint_tag.to_le_bytes());
        bytes[8..12].copy_from_slice(&u32::from(restore_downstream).to_le_bytes());
        Ok(bytes)
    }

    pub fn indirect_dispatch_step<'a>(
        &'a self,
        byte_offset: usize,
        dispatch: &'a VulkanResidentKernelDispatch,
        push_constants: &'a [u8],
    ) -> Result<VulkanResidentKernelSequenceStep<'a>, VulkanError> {
        if !self
            .config
            .downstream_dispatches
            .iter()
            .any(|target| target.byte_offset == byte_offset)
        {
            return Err(VulkanError(format!(
                "GPU residency indirect byte offset {byte_offset} is not controlled by this gate"
            )));
        }
        VulkanResidentKernelSequenceStep::new_indirect(
            dispatch,
            push_constants,
            &self.indirect_dispatches,
            byte_offset,
        )
    }

    pub fn resolved_addresses_buffer(&self) -> &VulkanResidentBuffer {
        &self.resolved_addresses
    }

    pub fn indirect_dispatch_buffer(&self) -> &VulkanResidentBuffer {
        &self.indirect_dispatches
    }

    pub fn notification_epoch(&self) -> Result<u32, VulkanError> {
        self.missing_queue.notification_epoch()
    }

    pub fn missing_snapshot(&self) -> Result<VulkanGpuResidencyMissingSnapshot, VulkanError> {
        self.missing_queue.snapshot()
    }

    pub fn acknowledge_missing_through(&self, published_count: u32) -> Result<(), VulkanError> {
        self.missing_queue.acknowledge_through(published_count)
    }
}

impl VulkanGpuResidencyMissQueue {
    pub fn new(device: &VulkanComputeDevice, capacity: usize) -> Result<Self, VulkanError> {
        if capacity == 0 || capacity > i32::MAX as usize {
            return Err(VulkanError(format!(
                "GPU residency miss queue capacity {capacity} is invalid"
            )));
        }
        let word_count = VULKAN_GPU_RESIDENCY_GATE_MISS_HEADER_WORD_COUNT
            .checked_add(
                capacity
                    .checked_mul(VULKAN_GPU_RESIDENCY_GATE_MISS_RECORD_WORD_COUNT)
                    .ok_or_else(|| {
                        VulkanError("GPU residency miss queue capacity overflowed".to_string())
                    })?,
            )
            .ok_or_else(|| {
                VulkanError("GPU residency miss queue capacity overflowed".to_string())
            })?;
        let mut buffer =
            device.create_host_visible_resident_buffer(words_byte_count(word_count)?)?;
        buffer.persistently_map()?;
        let buffer = Arc::new(buffer);
        buffer.write_bytes(&vec![0; buffer.byte_capacity()])?;
        Ok(Self { capacity, buffer })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn buffer(&self) -> &VulkanResidentBuffer {
        &self.buffer
    }

    pub fn notification_epoch(&self) -> Result<u32, VulkanError> {
        self.buffer
            .read_persistently_mapped_u32_le_at(3 * size_of::<u32>())
    }

    pub fn snapshot(&self) -> Result<VulkanGpuResidencyMissingSnapshot, VulkanError> {
        let published_count = self.buffer.read_persistently_mapped_u32_le_at(0)?;
        let consumed_count = self
            .buffer
            .read_persistently_mapped_u32_le_at(size_of::<u32>())?;
        let pending_count = published_count.wrapping_sub(consumed_count);
        if usize::try_from(pending_count).unwrap_or(usize::MAX) > self.capacity {
            return Err(VulkanError(
                "GPU residency miss queue counters exceed bounded capacity".to_string(),
            ));
        }
        let requests = (0..pending_count)
            .map(|pending_index| {
                let ticket = consumed_count.wrapping_add(pending_index);
                let slot =
                    usize::try_from(ticket).unwrap_or(usize::MAX) % self.capacity;
                let byte_offset = (VULKAN_GPU_RESIDENCY_GATE_MISS_HEADER_WORD_COUNT
                    + slot * VULKAN_GPU_RESIDENCY_GATE_MISS_RECORD_WORD_COUNT)
                    * size_of::<u32>();
                let checkpoint_tag = self
                    .buffer
                    .read_persistently_mapped_u32_le_at(byte_offset)?;
                let resource_index = usize::try_from(
                    self.buffer.read_persistently_mapped_u32_le_at(
                        byte_offset + size_of::<u32>(),
                    )?,
                )
                .map_err(|_| {
                    VulkanError("GPU residency resource index exceeds usize".to_string())
                })?;
                Ok(VulkanGpuResidencyMissingRequest {
                    checkpoint_tag,
                    resource_index,
                })
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        Ok(VulkanGpuResidencyMissingSnapshot {
            published_count,
            consumed_count,
            overflowed: self
                .buffer
                .read_persistently_mapped_u32_le_at(2 * size_of::<u32>())?
                != 0,
            notification_epoch: self.notification_epoch()?,
            requests,
        })
    }

    pub fn acknowledge_through(&self, published_count: u32) -> Result<(), VulkanError> {
        let current = self.buffer.read_persistently_mapped_u32_le_at(0)?;
        if published_count != current {
            return Err(VulkanError(format!(
                "GPU residency acknowledgement {published_count} is stale; current publication is {current}"
            )));
        }
        self.buffer
            .write_bytes_at(size_of::<u32>(), &published_count.to_le_bytes())
    }
}

fn words_byte_count(word_count: usize) -> Result<usize, VulkanError> {
    word_count.checked_mul(size_of::<u32>()).ok_or_else(|| {
        VulkanError("GPU residency word capacity overflowed".to_string())
    })
}

fn u32_words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

pub fn vulkan_gpu_residency_gate_spirv_words() -> Result<Vec<u32>, VulkanError> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/gpu_residency_gate.spv"));
    if bytes.is_empty() || !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(VulkanError(
            "embedded GPU residency gate SPIR-V is empty or misaligned".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(size_of::<u32>())
        .map(|word| u32::from_le_bytes(word.try_into().expect("SPIR-V word is four bytes")))
        .collect())
}
