const VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT: u32 = 16;
const VULKAN_GPU_RESIDENCY_GATE_GROUP_RECORD_WORD_COUNT: usize = 2;
const VULKAN_GPU_RESIDENCY_GATE_RESOLVED_HEADER_WORD_COUNT: usize = 8;
const VULKAN_GPU_RESIDENCY_GATE_RESOLVED_RECORD_WORD_COUNT: usize = 8;
const VULKAN_GPU_RESIDENCY_GATE_MISS_HEADER_WORD_COUNT: usize = 4;
const VULKAN_GPU_RESIDENCY_GATE_MISS_RECORD_WORD_COUNT: usize = 2;
const VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT: usize = 11;
const VULKAN_GPU_RESIDENCY_GATE_GROUP_TABLE_MAPPING: u32 = 0;
const VULKAN_GPU_RESIDENCY_GATE_PARTITIONED_MAPPING: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanGpuResidencyAddressMapping {
    GroupTable {
        resource_address_slots: Vec<usize>,
        resource_address_slot_offsets: Vec<usize>,
    },
    Partitioned {
        member_slot_bases: Vec<usize>,
        resource_count: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyGateConfig {
    pub maximum_selection_count: usize,
    pub selection_count_per_lane: usize,
    pub selection_lane_stride_words: usize,
    pub selection_index_shift: u32,
    pub selection_index_mask: u32,
    pub address_mapping: VulkanGpuResidencyAddressMapping,
    pub owned_resource_indices: Option<BTreeSet<usize>>,
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
    maximum_selection_count: usize,
    selection_buffer: Arc<VulkanResidentBuffer>,
    config: VulkanGpuResidencyGateConfig,
    _address_table_buffer: Arc<VulkanResidentBuffer>,
    _configuration: Arc<VulkanResidentBuffer>,
    _resource_group_records: Arc<VulkanResidentBuffer>,
    _resource_address_slots: Arc<VulkanResidentBuffer>,
    resolved_addresses: Arc<VulkanResidentBuffer>,
    missing_queue: VulkanGpuResidencyMissQueue,
    continuation_predicate: Arc<VulkanResidentBuffer>,
    transaction_predicate: Arc<VulkanResidentBuffer>,
    dispatch: VulkanResidentKernelDispatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyGatePrivateDeviceBytes {
    pub configuration_bytes: usize,
    pub resource_group_record_bytes: usize,
    pub resource_address_slot_bytes: usize,
    pub resolved_address_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanGpuResidencyMissQueueDeviceBytes {
    pub capacity: usize,
    pub byte_count: usize,
}

impl VulkanGpuResidencyGateConfig {
    pub fn validate(
        &self,
        selection_buffer_byte_capacity: usize,
        address_table_slot_count: usize,
        missing_request_capacity: usize,
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
        let resource_count = self
            .address_mapping
            .validate(address_table_slot_count)?;
        if self.owned_resource_indices.as_ref().is_some_and(|indices| {
            indices.is_empty() || indices.iter().any(|index| *index >= resource_count)
        }) {
            return Err(VulkanError(format!(
                "GPU residency gate resource ownership is empty or exceeds {resource_count} resources"
            )));
        }
        let maximum_resource_index = resource_count - 1;
        if u32::try_from(maximum_resource_index).map_or(true, |index| {
            index & self.selection_index_mask != index
        }) {
            return Err(VulkanError(format!(
                "GPU residency gate selection mask {:#010x} cannot represent resource index {maximum_resource_index}",
                self.selection_index_mask
            )));
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
        Ok(())
    }

    fn maximum_resolved_address_count(&self) -> Result<usize, VulkanError> {
        let maximum_members =
            self.address_mapping.maximum_resource_member_count();
        self.maximum_selection_count
            .checked_mul(maximum_members)
            .ok_or_else(|| {
                VulkanError("GPU residency resolved-address capacity overflowed".to_string())
            })
    }

    /// Exact device-local allocations owned by one residency gate. Selection,
    /// address-table, miss-queue, and predicate buffers are caller-owned and
    /// deliberately excluded because a scalar chain may share them across
    /// several gates while a distributed shard may not.
    pub fn private_device_bytes(
        &self,
    ) -> Result<VulkanGpuResidencyGatePrivateDeviceBytes, VulkanError> {
        let (resource_group_words, resource_address_slot_words, _, _) =
            self.address_mapping.gpu_tables()?;
        let ownership_word_count = self
            .owned_resource_indices
            .as_ref()
            .map(|_| self.address_mapping.resource_count().div_ceil(u32::BITS as usize))
            .unwrap_or(0);
        let configuration_word_count = VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT
            .checked_add(ownership_word_count)
            .ok_or_else(|| VulkanError("GPU residency configuration capacity overflowed".to_string()))?;
        let maximum_resolved_address_count = self.maximum_resolved_address_count()?;
        let resolved_record_word_count = maximum_resolved_address_count
            .checked_mul(VULKAN_GPU_RESIDENCY_GATE_RESOLVED_RECORD_WORD_COUNT)
            .ok_or_else(|| {
                VulkanError("GPU residency resolved record capacity overflowed".to_string())
            })?;
        let seen_resource_word_count = self
            .address_mapping
            .resource_count()
            .div_ceil(u32::BITS as usize);
        let resolved_word_count = VULKAN_GPU_RESIDENCY_GATE_RESOLVED_HEADER_WORD_COUNT
            .checked_add(resolved_record_word_count)
            .and_then(|count| count.checked_add(seen_resource_word_count))
            .and_then(|count| count.checked_add(self.maximum_selection_count))
            .ok_or_else(|| {
                VulkanError(
                    "GPU residency resolved and scratch buffer capacity overflowed".to_string(),
                )
            })?;
        let configuration_bytes = words_byte_count(configuration_word_count)?;
        let resource_group_record_bytes = words_byte_count(resource_group_words.len())?;
        let resource_address_slot_bytes = words_byte_count(resource_address_slot_words.len())?;
        let resolved_address_bytes = words_byte_count(resolved_word_count)?;
        let total_bytes = [
            configuration_bytes,
            resource_group_record_bytes,
            resource_address_slot_bytes,
            resolved_address_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                VulkanError("GPU residency private device bytes overflowed".to_string())
            })
        })?;
        Ok(VulkanGpuResidencyGatePrivateDeviceBytes {
            configuration_bytes,
            resource_group_record_bytes,
            resource_address_slot_bytes,
            resolved_address_bytes,
            total_bytes,
        })
    }
}

impl VulkanGpuResidencyAddressMapping {
    fn resource_count(&self) -> usize {
        match self {
            Self::GroupTable {
                resource_address_slot_offsets,
                ..
            } => resource_address_slot_offsets.len().saturating_sub(1),
            Self::Partitioned { resource_count, .. } => *resource_count,
        }
    }

    fn maximum_resource_member_count(&self) -> usize {
        match self {
            Self::GroupTable {
                resource_address_slot_offsets,
                ..
            } => resource_address_slot_offsets
                .windows(2)
                .map(|bounds| bounds[1].saturating_sub(bounds[0]))
                .max()
                .unwrap_or(0),
            Self::Partitioned {
                member_slot_bases,
                ..
            } => member_slot_bases.len(),
        }
    }

    fn validate(
        &self,
        address_table_slot_count: usize,
    ) -> Result<usize, VulkanError> {
        match self {
            Self::GroupTable {
                resource_address_slots,
                resource_address_slot_offsets,
            } => {
                if resource_address_slot_offsets.len() < 2
                    || resource_address_slot_offsets[0] != 0
                    || resource_address_slot_offsets.last().copied()
                        != Some(resource_address_slots.len())
                    || resource_address_slot_offsets
                        .windows(2)
                        .any(|bounds| bounds[0] >= bounds[1])
                {
                    return Err(VulkanError(
                        "GPU residency gate group table is empty or invalid"
                            .to_string(),
                    ));
                }
                for (resource_index, bounds) in
                    resource_address_slot_offsets.windows(2).enumerate()
                {
                    let mut unique = BTreeSet::new();
                    for slot in &resource_address_slots[bounds[0]..bounds[1]] {
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
                Ok(resource_address_slot_offsets.len() - 1)
            }
            Self::Partitioned {
                member_slot_bases,
                resource_count,
            } => {
                if *resource_count == 0 || member_slot_bases.is_empty() {
                    return Err(VulkanError(
                        "GPU residency gate partition mapping is empty"
                            .to_string(),
                    ));
                }
                let mut ranges = member_slot_bases
                    .iter()
                    .map(|base| {
                        base.checked_add(*resource_count)
                            .map(|end| (*base, end))
                            .ok_or_else(|| {
                                VulkanError(
                                    "GPU residency partition slot range overflowed"
                                        .to_string(),
                                )
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ranges.sort_unstable();
                for range in &ranges {
                    if range.1 > address_table_slot_count {
                        return Err(VulkanError(format!(
                            "GPU residency partition address range {}..{} exceeds the table's {address_table_slot_count} slots",
                            range.0, range.1
                        )));
                    }
                }
                if ranges
                    .windows(2)
                    .any(|pair| pair[0].1 > pair[1].0)
                {
                    return Err(VulkanError(
                        "GPU residency partition member address ranges overlap"
                            .to_string(),
                    ));
                }
                Ok(*resource_count)
            }
        }
    }

    fn gpu_tables(
        &self,
    ) -> Result<(Vec<u32>, Vec<u32>, u32, u32), VulkanError> {
        match self {
            Self::GroupTable {
                resource_address_slots,
                resource_address_slot_offsets,
            } => {
                let mut group_words = Vec::with_capacity(
                    (resource_address_slot_offsets.len() - 1)
                        * VULKAN_GPU_RESIDENCY_GATE_GROUP_RECORD_WORD_COUNT,
                );
                for bounds in resource_address_slot_offsets.windows(2) {
                    group_words.push(u32::try_from(bounds[0]).map_err(
                        |_| {
                            VulkanError(
                                "GPU residency address-slot offset exceeds u32"
                                    .to_string(),
                            )
                        },
                    )?);
                    group_words.push(
                        u32::try_from(bounds[1] - bounds[0]).map_err(
                            |_| {
                                VulkanError(
                                    "GPU residency resource member count exceeds u32"
                                        .to_string(),
                                )
                            },
                        )?,
                    );
                }
                let slot_words = resource_address_slots
                    .iter()
                    .map(|slot| {
                        u32::try_from(*slot).map_err(|_| {
                            VulkanError(
                                "GPU residency address slot exceeds u32"
                                    .to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((
                    group_words,
                    slot_words,
                    VULKAN_GPU_RESIDENCY_GATE_GROUP_TABLE_MAPPING,
                    0,
                ))
            }
            Self::Partitioned {
                member_slot_bases,
                ..
            } => {
                let member_count =
                    u32::try_from(member_slot_bases.len()).map_err(|_| {
                        VulkanError(
                            "GPU residency partition member count exceeds u32"
                                .to_string(),
                        )
                    })?;
                let slot_words = member_slot_bases
                    .iter()
                    .map(|slot| {
                        u32::try_from(*slot).map_err(|_| {
                            VulkanError(
                                "GPU residency partition slot base exceeds u32"
                                    .to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((
                    vec![0, member_count],
                    slot_words,
                    VULKAN_GPU_RESIDENCY_GATE_PARTITIONED_MAPPING,
                    member_count,
                ))
            }
        }
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
        continuation_predicate: Arc<VulkanResidentBuffer>,
        transaction_predicate: Option<Arc<VulkanResidentBuffer>>,
        config: VulkanGpuResidencyGateConfig,
    ) -> Result<Self, VulkanError> {
        config.validate(
            selection_buffer.byte_capacity(),
            address_table_slot_count,
            missing_queue.capacity(),
        )?;
        let private_device_bytes = config.private_device_bytes()?;
        if continuation_predicate.byte_capacity() < size_of::<u32>() {
            return Err(VulkanError(format!(
                "GPU residency continuation predicate has {} bytes; expected at least {}",
                continuation_predicate.byte_capacity(),
                size_of::<u32>()
            )));
        }
        let transaction_predicate =
            transaction_predicate.unwrap_or_else(|| Arc::clone(&continuation_predicate));
        if transaction_predicate.byte_capacity() < size_of::<u32>() {
            return Err(VulkanError(format!(
                "GPU residency transaction predicate has {} bytes; expected at least {}",
                transaction_predicate.byte_capacity(),
                size_of::<u32>()
            )));
        }
        if !device.owns_resident_buffer(&address_table_buffer)
            || !device.owns_resident_buffer(missing_queue.buffer())
            || !device.owns_resident_buffer(&continuation_predicate)
            || !device.owns_resident_buffer(&transaction_predicate)
        {
            return Err(VulkanError(
                "GPU residency gate buffers belong to another logical device".to_string(),
            ));
        }

        let (
            resource_group_words,
            resource_address_slot_words,
            mapping_kind,
            partition_member_count,
        ) = config.address_mapping.gpu_tables()?;
        let resource_count = config.address_mapping.resource_count();
        let ownership_words = vulkan_gpu_residency_ownership_words(
            resource_count,
            config.owned_resource_indices.as_ref(),
        )?;
        let mut configuration_words = vec![
            config.selection_index_shift,
            config.selection_index_mask,
            u32::try_from(resource_count).map_err(|_| {
                VulkanError("GPU residency resource count exceeds u32".to_string())
            })?,
            u32::try_from(missing_queue.capacity()).map_err(|_| {
                VulkanError("GPU residency missing capacity exceeds u32".to_string())
            })?,
            u32::try_from(config.maximum_resolved_address_count()?).map_err(|_| {
                VulkanError("GPU residency resolved capacity exceeds u32".to_string())
            })?,
            u32::try_from(config.selection_count_per_lane).map_err(|_| {
                VulkanError("GPU residency selection count per lane exceeds u32".to_string())
            })?,
            u32::try_from(config.selection_lane_stride_words).map_err(|_| {
                VulkanError("GPU residency selection lane stride exceeds u32".to_string())
            })?,
            mapping_kind,
            partition_member_count,
            u32::try_from(VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT).map_err(|_| {
                VulkanError("GPU residency ownership offset exceeds u32".to_string())
            })?,
            u32::try_from(ownership_words.len()).map_err(|_| {
                VulkanError("GPU residency ownership word count exceeds u32".to_string())
            })?,
        ];
        debug_assert_eq!(
            configuration_words.len(),
            VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT
        );
        configuration_words.extend(ownership_words);
        let configuration = Arc::new(
            device.create_resident_buffer(private_device_bytes.configuration_bytes)?,
        );
        configuration.write_bytes(&u32_words_bytes(&configuration_words))?;

        let resource_group_records = Arc::new(
            device.create_resident_buffer(private_device_bytes.resource_group_record_bytes)?,
        );
        resource_group_records.write_bytes(&u32_words_bytes(&resource_group_words))?;
        let resource_address_slots = Arc::new(
            device.create_resident_buffer(private_device_bytes.resource_address_slot_bytes)?,
        );
        resource_address_slots.write_bytes(&u32_words_bytes(&resource_address_slot_words))?;

        let resolved_record_word_count = config
            .maximum_resolved_address_count()?
            .checked_mul(VULKAN_GPU_RESIDENCY_GATE_RESOLVED_RECORD_WORD_COUNT)
            .ok_or_else(|| {
                VulkanError(
                    "GPU residency resolved record capacity overflowed".to_string(),
                )
            })?;
        let seen_resource_word_count =
            resource_count.div_ceil(u32::BITS as usize);
        let resolved_word_count = VULKAN_GPU_RESIDENCY_GATE_RESOLVED_HEADER_WORD_COUNT
            .checked_add(
                resolved_record_word_count,
            )
            .and_then(|count| count.checked_add(seen_resource_word_count))
            .and_then(|count| count.checked_add(config.maximum_selection_count))
            .ok_or_else(|| {
                VulkanError(
                    "GPU residency resolved and scratch buffer capacity overflowed".to_string(),
                )
            })?;
        debug_assert_eq!(
            words_byte_count(resolved_word_count)?,
            private_device_bytes.resolved_address_bytes,
        );
        let resolved_addresses = Arc::new(
            device.create_resident_buffer(private_device_bytes.resolved_address_bytes)?,
        );
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
                buffer: &continuation_predicate,
                byte_offset: 0,
                byte_len: size_of::<u32>(),
                access: VulkanResidentKernelBufferAccess::Write,
            },
            VulkanResidentKernelBufferBinding {
                binding: 7,
                buffer: &configuration,
                byte_offset: 0,
                byte_len: configuration.byte_capacity(),
                access: VulkanResidentKernelBufferAccess::Read,
            },
            VulkanResidentKernelBufferBinding {
                binding: 8,
                buffer: &transaction_predicate,
                byte_offset: 0,
                byte_len: size_of::<u32>(),
                access: VulkanResidentKernelBufferAccess::ReadWrite,
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
            maximum_selection_count: config.maximum_selection_count,
            selection_buffer,
            config,
            _address_table_buffer: address_table_buffer,
            _configuration: configuration,
            _resource_group_records: resource_group_records,
            _resource_address_slots: resource_address_slots,
            resolved_addresses,
            missing_queue,
            continuation_predicate,
            transaction_predicate,
            dispatch,
        })
    }

    pub fn dispatch(&self) -> &VulkanResidentKernelDispatch {
        &self.dispatch
    }

    pub(crate) fn owned_resource_indices(&self) -> Option<&BTreeSet<usize>> {
        self.config.owned_resource_indices.as_ref()
    }

    /// Replaces the GPU-side arithmetic ownership mask while retaining the
    /// recorded dispatch and every stable descriptor binding.
    ///
    /// The caller must establish a quiescent execution boundary. Gates created
    /// without an ownership mask intentionally cannot be converted in place:
    /// their immutable configuration buffer has no ownership-word capacity.
    pub(crate) fn replace_owned_resource_indices_at_quiescent_boundary(
        &mut self,
        owned_resource_indices: BTreeSet<usize>,
    ) -> Result<(), VulkanError> {
        if self.config.owned_resource_indices.is_none() {
            return Err(VulkanError(
                "GPU residency gate was mounted without mutable ownership capacity"
                    .to_string(),
            ));
        }
        let resource_count = self.config.address_mapping.resource_count();
        let ownership_words = vulkan_gpu_residency_ownership_words(
            resource_count,
            Some(&owned_resource_indices),
        )?;
        let ownership_bytes = u32_words_bytes(&ownership_words);
        let ownership_byte_offset = VULKAN_GPU_RESIDENCY_GATE_CONFIG_HEADER_WORD_COUNT
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                VulkanError("GPU residency ownership offset overflowed".to_string())
            })?;
        if ownership_byte_offset
            .checked_add(ownership_bytes.len())
            .is_none_or(|end| end != self._configuration.byte_capacity())
        {
            return Err(VulkanError(
                "GPU residency gate ownership capacity differs from its resource geometry"
                    .to_string(),
            ));
        }
        self._configuration
            .write_bytes_at(ownership_byte_offset, &ownership_bytes)?;
        self.config.owned_resource_indices = Some(owned_resource_indices);
        Ok(())
    }

    /// Device memory owned only by this gate. Shared selection/address-table
    /// inputs and the caller-owned transaction predicate are deliberately
    /// excluded so higher-level accounting can add them exactly once.
    pub fn auxiliary_transient_device_bytes(&self) -> Result<usize, VulkanError> {
        [
            self._configuration.byte_capacity(),
            self._resource_group_records.byte_capacity(),
            self._resource_address_slots.byte_capacity(),
            self.resolved_addresses.byte_capacity(),
            self.missing_queue.buffer().byte_capacity(),
            self.continuation_predicate.byte_capacity(),
        ]
        .into_iter()
        .try_fold(0usize, |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                VulkanError("GPU residency gate transient bytes overflowed".to_string())
            })
        })
    }

    pub fn selected_resource_indices(
        &self,
        active_selection_count: usize,
    ) -> Result<BTreeSet<usize>, VulkanError> {
        if active_selection_count == 0
            || active_selection_count > self.maximum_selection_count
        {
            return Err(VulkanError(format!(
                "GPU residency selected-resource readback count {active_selection_count} exceeds its bounded capacity {}",
                self.maximum_selection_count
            )));
        }
        let lane_count = active_selection_count
            .div_ceil(self.config.selection_count_per_lane);
        let final_lane_count = active_selection_count
            - (lane_count - 1) * self.config.selection_count_per_lane;
        let required_word_count = lane_count
            .saturating_sub(1)
            .checked_mul(self.config.selection_lane_stride_words)
            .and_then(|offset| {
                offset.checked_add(final_lane_count)
            })
            .ok_or_else(|| {
                VulkanError(
                    "GPU residency selected-resource readback overflowed"
                        .to_string(),
                )
            })?;
        let required_byte_count = required_word_count
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| {
                VulkanError(
                    "GPU residency selected-resource readback overflowed"
                        .to_string(),
                )
            })?;
        let bytes = self.selection_buffer.read_bytes(required_byte_count)?;
        let words = bytes
            .chunks_exact(size_of::<u32>())
            .map(|bytes| {
                u32::from_le_bytes(
                    bytes
                        .try_into()
                        .expect("u32 selection chunks are exact"),
                )
            })
            .collect::<Vec<_>>();
        let resource_count = self.config.address_mapping.resource_count();
        let mut selected = BTreeSet::new();
        let mut remaining = active_selection_count;
        for lane_index in 0..lane_count {
            let lane_offset = lane_index
                .checked_mul(self.config.selection_lane_stride_words)
                .expect("selection lane layout was prevalidated");
            let lane_selection_count = remaining
                .min(self.config.selection_count_per_lane);
            for selection_index in 0..lane_selection_count {
                let encoded = words[lane_offset + selection_index];
                let resource_index = usize::try_from(
                    (encoded >> self.config.selection_index_shift)
                        & self.config.selection_index_mask,
                )
                .expect("u32 selection index fits usize");
                if resource_index >= resource_count {
                    return Err(VulkanError(format!(
                        "GPU residency selection index {resource_index} exceeds {resource_count} resources"
                    )));
                }
                if self
                    .config
                    .owned_resource_indices
                    .as_ref()
                    .is_some_and(|indices| !indices.contains(&resource_index))
                {
                    continue;
                }
                selected.insert(resource_index);
            }
            remaining -= lane_selection_count;
        }
        Ok(selected)
    }

    pub fn push_constants(
        &self,
        selection_count: usize,
        checkpoint_tag: u32,
        restore_downstream: bool,
        restore_transaction: bool,
    ) -> Result<[u8; VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT as usize], VulkanError>
    {
        vulkan_gpu_residency_gate_push_constants(
            self.maximum_selection_count,
            selection_count,
            checkpoint_tag,
            restore_downstream,
            restore_transaction,
        )
    }

    pub fn resolved_addresses_buffer(&self) -> &VulkanResidentBuffer {
        &self.resolved_addresses
    }

    pub fn continuation_predicate(&self) -> &VulkanResidentBuffer {
        &self.continuation_predicate
    }

    pub fn transaction_predicate(&self) -> &VulkanResidentBuffer {
        &self.transaction_predicate
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

fn vulkan_gpu_residency_gate_push_constants(
    maximum_selection_count: usize,
    selection_count: usize,
    checkpoint_tag: u32,
    restore_downstream: bool,
    restore_transaction: bool,
) -> Result<[u8; VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT as usize], VulkanError> {
    if selection_count == 0 || selection_count > maximum_selection_count {
        return Err(VulkanError(format!(
            "GPU residency gate selection count {selection_count} is outside 1..={maximum_selection_count}"
        )));
    }
    let mut bytes = [0; VULKAN_GPU_RESIDENCY_GATE_PUSH_CONSTANT_BYTE_COUNT as usize];
    bytes[..4].copy_from_slice(
        &u32::try_from(selection_count)
            .map_err(|_| VulkanError("GPU residency selection count exceeds u32".to_string()))?
            .to_le_bytes(),
    );
    bytes[4..8].copy_from_slice(&checkpoint_tag.to_le_bytes());
    bytes[8..12].copy_from_slice(&u32::from(restore_downstream).to_le_bytes());
    bytes[12..16].copy_from_slice(&u32::from(restore_transaction).to_le_bytes());
    Ok(bytes)
}

fn vulkan_gpu_residency_ownership_words(
    resource_count: usize,
    owned_resource_indices: Option<&BTreeSet<usize>>,
) -> Result<Vec<u32>, VulkanError> {
    let Some(indices) = owned_resource_indices else {
        return Ok(Vec::new());
    };
    if resource_count == 0
        || indices.is_empty()
        || indices.iter().any(|index| *index >= resource_count)
    {
        return Err(VulkanError(format!(
            "GPU residency gate resource ownership is empty or exceeds {resource_count} resources"
        )));
    }
    let mut words = vec![0u32; resource_count.div_ceil(u32::BITS as usize)];
    for index in indices {
        words[*index / u32::BITS as usize] |=
            1u32 << (*index % u32::BITS as usize);
    }
    Ok(words)
}

impl VulkanGpuResidencyMissQueue {
    pub fn device_bytes_for_capacity(
        capacity: usize,
    ) -> Result<VulkanGpuResidencyMissQueueDeviceBytes, VulkanError> {
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
        Ok(VulkanGpuResidencyMissQueueDeviceBytes {
            capacity,
            byte_count: words_byte_count(word_count)?,
        })
    }

    pub fn new(device: &VulkanComputeDevice, capacity: usize) -> Result<Self, VulkanError> {
        let planned = Self::device_bytes_for_capacity(capacity)?;
        let mut buffer =
            device.create_host_visible_resident_buffer(planned.byte_count)?;
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
