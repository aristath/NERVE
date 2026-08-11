#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocationPlan {
    pub allocations: Vec<VulkanDistributedParameterAllocation>,
    pub allocation_count: usize,
    pub tensor_count: usize,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterAllocationPlan {
    pub fn from_execution_plan_set(
        plans: &VulkanDistributedExecutionPlanSet,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let phase_plans = plans
            .all()
            .into_iter()
            .map(|plan| Self::from_execution_plan(plan, tensor_index))
            .collect::<Result<Vec<_>, _>>()?;
        Self::merged(&phase_plans)
    }

    pub fn merged(
        plans: &[VulkanDistributedParameterAllocationPlan],
    ) -> Result<Self, VulkanDistributedPlanError> {
        let mut allocations_by_resource =
            BTreeMap::<(String, String), Vec<VulkanDistributedParameterAllocation>>::new();
        for plan in plans {
            for allocation in &plan.allocations {
                if allocation.byte_count == 0
                    || allocation
                        .byte_offset
                        .checked_add(allocation.byte_count)
                        .is_none()
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed parameter tensor {:?} has an invalid merge range",
                        allocation.tensor
                    )));
                }
                allocations_by_resource
                    .entry((allocation.device_id.clone(), allocation.tensor.clone()))
                    .or_default()
                    .push(allocation.clone());
            }
        }
        let mut allocations = Vec::new();
        for (_, mut ranges) in allocations_by_resource {
            ranges.sort_by_key(|allocation| (allocation.byte_offset, allocation.byte_count));
            let mut merged_ranges = Vec::<VulkanDistributedParameterAllocation>::new();
            for allocation in ranges {
                let Some(current) = merged_ranges.last_mut() else {
                    merged_ranges.push(allocation);
                    continue;
                };
                let current_end = current
                    .byte_offset
                    .checked_add(current.byte_count)
                    .expect("merge ranges were validated above");
                if allocation.byte_offset <= current_end {
                    let allocation_end = allocation
                        .byte_offset
                        .checked_add(allocation.byte_count)
                        .expect("merge ranges were validated above");
                    current.byte_count = current_end
                        .max(allocation_end)
                        .checked_sub(current.byte_offset)
                        .expect("sorted merge range cannot precede its origin");
                    current.use_count = current
                        .use_count
                        .checked_add(allocation.use_count)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "distributed parameter tensor {:?} merged use count overflowed",
                                current.tensor
                            ))
                        })?;
                } else {
                    merged_ranges.push(allocation);
                }
            }
            allocations.extend(merged_ranges);
        }
        let total_byte_capacity = allocations.iter().try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "merged distributed parameter capacity overflowed".to_string(),
                )
            })
        })?;
        let tensor_count = allocations
            .iter()
            .map(|allocation| allocation.tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        Ok(Self {
            allocation_count: allocations.len(),
            allocations,
            tensor_count,
            total_byte_capacity,
        })
    }

    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_execution_plan_with_coverage(execution_plan, tensor_index, true)
    }

    pub fn from_sampled_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_execution_plan_with_coverage(execution_plan, tensor_index, false)
    }

    fn from_execution_plan_with_coverage(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
        require_complete_tensor_coverage: bool,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedParameterAllocationKey,
            VulkanDistributedParameterAllocation,
        >::new();

        for dispatch in &execution_plan.dispatches {
            for shard in &dispatch.shards {
                if !device_ids.contains(shard.device_id.as_str()) {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed parameter shard for {}.{} uses device {:?} outside the execution pool",
                        dispatch.component_id, dispatch.node_id, shard.device_id
                    )));
                }
                for fragment in &shard.parameters {
                    let metadata = tensor_index.tensors.get(&fragment.tensor).ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter fragment has no tensor metadata for {:?}",
                            fragment.tensor
                        ))
                    })?;
                    let tensor_byte_count = metadata.byte_count.ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has no byte count",
                            fragment.tensor
                        ))
                    })?;
                    if fragment.byte_count == 0 {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has an empty fragment",
                            fragment.tensor
                        )));
                    }
                    let byte_end = fragment
                        .byte_offset
                        .checked_add(fragment.byte_count)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "distributed parameter tensor {:?} fragment range overflowed",
                                fragment.tensor
                            ))
                        })?;
                    if byte_end > tensor_byte_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has {tensor_byte_count} bytes but a fragment ends at {byte_end}",
                            fragment.tensor
                        )));
                    }
                    let key = VulkanDistributedParameterAllocationKey {
                        device_id: shard.device_id.clone(),
                        tensor: fragment.tensor.clone(),
                        byte_offset: fragment.byte_offset,
                        byte_count: fragment.byte_count,
                    };
                    if let Some(allocation) = allocations.get_mut(&key) {
                        allocation.use_count =
                            allocation.use_count.checked_add(1).ok_or_else(|| {
                                VulkanDistributedPlanError(format!(
                                    "distributed parameter tensor {:?} use count overflowed",
                                    fragment.tensor
                                ))
                            })?;
                    } else {
                        allocations.insert(
                            key,
                            VulkanDistributedParameterAllocation {
                                device_id: shard.device_id.clone(),
                                tensor: fragment.tensor.clone(),
                                byte_offset: fragment.byte_offset,
                                byte_count: fragment.byte_count,
                                use_count: 1,
                            },
                        );
                    }
                }
            }
        }

        if require_complete_tensor_coverage {
            validate_tensor_partition_coverage(allocations.values(), tensor_index)?;
        }
        let total_byte_capacity = allocations.values().try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed parameter allocation byte count overflowed".to_string(),
                )
            })
        })?;
        let tensor_count = allocations
            .values()
            .map(|allocation| allocation.tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let allocations = allocations.into_values().collect::<Vec<_>>();

        Ok(Self {
            allocation_count: allocations.len(),
            allocations,
            tensor_count,
            total_byte_capacity,
        })
    }

    pub fn load_from_tensor_index<F>(
        &self,
        tensor_index: &TensorIndex,
        mut write: F,
    ) -> Result<VulkanDistributedParameterLoadReport, VulkanDistributedParameterLoadError>
    where
        F: FnMut(
            &VulkanDistributedParameterAllocation,
            &[u8],
        ) -> Result<(), VulkanDistributedParameterLoadError>,
    {
        let mut allocations_by_tensor = BTreeMap::<
            &str,
            BTreeMap<(usize, usize), Vec<&VulkanDistributedParameterAllocation>>,
        >::new();
        for allocation in &self.allocations {
            allocations_by_tensor
                .entry(&allocation.tensor)
                .or_default()
                .entry((allocation.byte_offset, allocation.byte_count))
                .or_default()
                .push(allocation);
        }

        let mut total_bytes_read = 0usize;
        let mut total_bytes_written = 0usize;
        let mut write_count = 0usize;
        let mut source_files = BTreeSet::new();
        for (tensor, ranges) in allocations_by_tensor {
            let storage = TensorStorage::from_index(tensor_index, tensor)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            let storage_ranges = ranges
                .keys()
                .map(|(byte_offset, byte_count)| TensorStorageRange {
                    byte_offset: *byte_offset,
                    byte_count: *byte_count,
                })
                .collect::<Vec<_>>();
            let payloads = storage
                .read_partitions(&storage_ranges)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            total_bytes_read = total_bytes_read
                .checked_add(storage.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterLoadError(
                        "distributed parameter read byte count overflowed".to_string(),
                    )
                })?;
            source_files.insert(storage.source_file);

            for (((_, _), allocations), payload) in ranges.into_iter().zip(payloads) {
                for allocation in allocations {
                    write(allocation, &payload)?;
                    total_bytes_written = total_bytes_written
                        .checked_add(payload.len())
                        .ok_or_else(|| {
                            VulkanDistributedParameterLoadError(
                                "distributed parameter written byte count overflowed".to_string(),
                            )
                        })?;
                    write_count = write_count.checked_add(1).ok_or_else(|| {
                        VulkanDistributedParameterLoadError(
                            "distributed parameter write count overflowed".to_string(),
                        )
                    })?;
                }
            }
        }

        Ok(VulkanDistributedParameterLoadReport {
            tensor_count: self.tensor_count,
            source_file_count: source_files.len(),
            allocation_count: self.allocation_count,
            write_count,
            total_bytes_read,
            total_bytes_written,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocation {
    pub device_id: String,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub use_count: usize,
}

impl VulkanDistributedParameterAllocation {
    fn contains_fragment(
        &self,
        device_id: &str,
        tensor: &str,
        byte_offset: usize,
        byte_count: usize,
    ) -> bool {
        if self.device_id != device_id || self.tensor != tensor || byte_count == 0 {
            return false;
        }
        let Some(allocation_end) = self.byte_offset.checked_add(self.byte_count) else {
            return false;
        };
        let Some(fragment_end) = byte_offset.checked_add(byte_count) else {
            return false;
        };
        byte_offset >= self.byte_offset && fragment_end <= allocation_end
    }
}

pub struct VulkanDistributedParameterBuffers {
    pub plan: VulkanDistributedParameterAllocationPlan,
    pub buffers: Vec<VulkanDistributedParameterBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterBuffers {
    pub fn allocate_and_load<'a, F, E>(
        plan: &VulkanDistributedParameterAllocationPlan,
        tensor_index: &TensorIndex,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedParameterBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut buffers = std::iter::repeat_with(|| None)
            .take(plan.allocations.len())
            .collect::<Vec<Option<VulkanDistributedParameterBufferAllocation>>>();
        let mut buffer_index = BTreeMap::new();
        let mut allocations_by_device = BTreeMap::<String, Vec<usize>>::new();
        let mut total_byte_capacity = 0usize;
        for (allocation_index, allocation) in plan.allocations.iter().enumerate() {
            total_byte_capacity = total_byte_capacity
                .checked_add(allocation.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(
                        "distributed parameter buffer byte capacity overflowed".to_string(),
                    )
                })?;
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            if buffer_index.insert(key, allocation_index).is_some() {
                return Err(VulkanDistributedParameterBufferError(format!(
                    "distributed parameter buffer repeats tensor {:?} range {}..{} on {:?}",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                )));
            }
            allocations_by_device
                .entry(allocation.device_id.clone())
                .or_default()
                .push(allocation_index);
        }
        for (device_id, allocation_indices) in allocations_by_device {
            let device = device_for(&device_id).map_err(|error| {
                VulkanDistributedParameterBufferError(format!(
                    "failed to resolve distributed parameter device {device_id:?}: {error}"
                ))
            })?;
            let byte_counts = allocation_indices
                .iter()
                .map(|index| plan.allocations[*index].byte_count)
                .collect::<Vec<_>>();
            let arena_allocations = device
                .allocate_resident_buffer_arena(&byte_counts)
                .map_err(VulkanDistributedParameterBufferError::from)?;
            for (allocation_index, arena) in allocation_indices.into_iter().zip(arena_allocations)
            {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: plan.allocations[allocation_index].clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
            }
        }
        let buffers = buffers
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                buffer.ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "distributed parameter arena did not allocate plan index {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        plan.load_from_tensor_index(tensor_index, |allocation, bytes| {
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            let index = *buffer_index.get(&key).ok_or_else(|| {
                VulkanDistributedParameterLoadError(format!(
                    "distributed parameter buffer for tensor {:?} range {}..{} on {:?} is missing",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                ))
            })?;
            buffers[index]
                .buffer
                .write_bytes_at(buffers[index].byte_offset, bytes)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))
        })
        .map_err(|error| VulkanDistributedParameterBufferError(error.to_string()))?;

        Ok(Self {
            plan: plan.clone(),
            buffers,
            total_byte_capacity,
        })
    }

    pub fn allocate_and_load_from_pool(
        plan: &VulkanDistributedParameterAllocationPlan,
        tensor_index: &TensorIndex,
        pool: &VulkanResidentBufferPool,
    ) -> Result<Self, VulkanDistributedParameterBufferError> {
        let mut buffers = std::iter::repeat_with(|| None)
            .take(plan.allocations.len())
            .collect::<Vec<Option<VulkanDistributedParameterBufferAllocation>>>();
        let mut buffer_index = BTreeMap::new();
        let mut pool_keys = Vec::with_capacity(plan.allocations.len());
        let mut missing_by_device = BTreeMap::<
            String,
            Vec<(usize, VulkanResidentBufferPoolKey)>,
        >::new();
        let mut total_byte_capacity = 0usize;
        for (allocation_index, allocation) in plan.allocations.iter().enumerate() {
            let metadata = tensor_index
                .tensors
                .get(&allocation.tensor)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "tensor index has no distributed parameter {:?}",
                        allocation.tensor
                    ))
                })?;
            let content_identity = metadata
                .immutable_content_identity(&allocation.tensor)
                .map_err(|error| {
                    VulkanDistributedParameterBufferError(
                        error.to_string(),
                    )
                })?;
            let key = VulkanResidentBufferPoolKey::new(
                "nerve.tensor_parameter.v1",
                &allocation.device_id,
                &allocation.tensor,
                content_identity,
                allocation.byte_offset,
                allocation.byte_count,
            )
            .map_err(VulkanDistributedParameterBufferError::from)?;
            if let Some(arena) = pool.resident_allocation(&key) {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: allocation.clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
            } else {
                missing_by_device
                    .entry(allocation.device_id.clone())
                    .or_default()
                    .push((allocation_index, key.clone()));
            }
            pool_keys.push(key);
            total_byte_capacity = total_byte_capacity
                .checked_add(allocation.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(
                        "distributed parameter buffer byte capacity overflowed"
                            .to_string(),
                    )
                })?;
            let allocation_key =
                VulkanDistributedParameterAllocationKey::from(allocation);
            if buffer_index
                .insert(allocation_key, allocation_index)
                .is_some()
            {
                return Err(VulkanDistributedParameterBufferError(format!(
                    "distributed parameter buffer repeats tensor {:?} range {}..{} on {:?}",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                )));
            }
        }
        let mut unpublished_indices = Vec::new();
        for (_, missing) in missing_by_device {
            let keys = missing
                .iter()
                .map(|(_, key)| key.clone())
                .collect::<Vec<_>>();
            let arena_allocations = pool
                .allocate_unpublished_batch(&keys)
                .map_err(VulkanDistributedParameterBufferError::from)?;
            for ((allocation_index, _), arena) in missing.into_iter().zip(arena_allocations) {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: plan.allocations[allocation_index].clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
                unpublished_indices.push(allocation_index);
            }
        }
        let buffers = buffers
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                buffer.ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "pooled distributed parameter arena did not allocate plan index {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unpublished_allocations = unpublished_indices
            .iter()
            .map(|index| plan.allocations[*index].clone())
            .collect::<Vec<_>>();
        if !unpublished_allocations.is_empty() {
            let total_byte_capacity = unpublished_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_count).ok_or_else(
                        || {
                            VulkanDistributedParameterBufferError(
                                "pooled distributed parameter load byte count overflowed"
                                    .to_string(),
                            )
                        },
                    )
                })?;
            let load_plan = VulkanDistributedParameterAllocationPlan {
                allocation_count: unpublished_allocations.len(),
                tensor_count: unpublished_allocations
                    .iter()
                    .map(|allocation| allocation.tensor.as_str())
                    .collect::<BTreeSet<_>>()
                    .len(),
                total_byte_capacity,
                allocations: unpublished_allocations,
            };
            load_plan
                .load_from_tensor_index(
                    tensor_index,
                    |allocation, bytes| {
                        let key =
                            VulkanDistributedParameterAllocationKey::from(
                                allocation,
                            );
                        let index = *buffer_index.get(&key).ok_or_else(
                            || {
                                VulkanDistributedParameterLoadError(
                                    format!(
                                        "pooled distributed parameter buffer for tensor {:?} range {}..{} on {:?} is missing",
                                        allocation.tensor,
                                        allocation.byte_offset,
                                        allocation.byte_offset
                                            + allocation.byte_count,
                                        allocation.device_id
                                    ),
                                )
                            },
                        )?;
                        buffers[index]
                            .buffer
                            .write_bytes_at(buffers[index].byte_offset, bytes)
                            .map_err(|error| {
                                VulkanDistributedParameterLoadError(
                                    error.to_string(),
                                )
                            })
                    },
                )
                .map_err(|error| {
                    VulkanDistributedParameterBufferError(
                        error.to_string(),
                    )
                })?;
            let publications = unpublished_indices
                .iter()
                .map(|index| {
                    let buffer = &buffers[*index];
                    (
                        pool_keys[*index].clone(),
                        VulkanResidentBufferPoolAllocation {
                            buffer: Arc::clone(&buffer.buffer),
                            byte_offset: buffer.byte_offset,
                            byte_count: buffer.allocation.byte_count,
                        },
                    )
                })
                .collect();
            pool.publish_batch(publications)
                .map_err(VulkanDistributedParameterBufferError::from)?;
        }
        Ok(Self {
            plan: plan.clone(),
            buffers,
            total_byte_capacity,
        })
    }

    pub fn parameter_buffer(
        &self,
        device_id: &str,
        tensor: &str,
        byte_offset: usize,
        byte_count: usize,
    ) -> Option<&VulkanDistributedParameterBufferAllocation> {
        self.buffers.iter().find(|buffer| {
            buffer
                .allocation
                .contains_fragment(device_id, tensor, byte_offset, byte_count)
        })
    }
}

pub struct VulkanDistributedParameterBufferAllocation {
    pub allocation: VulkanDistributedParameterAllocation,
    pub buffer: Arc<VulkanResidentBuffer>,
    pub byte_offset: usize,
}

impl VulkanDistributedParameterBufferAllocation {
    pub fn kernel_binding_for_fragment(
        &self,
        binding: u32,
        byte_offset: usize,
        byte_count: usize,
    ) -> Result<VulkanResidentKernelBufferBinding<'_>, VulkanDistributedParameterBufferError> {
        if !self.allocation.contains_fragment(
            &self.allocation.device_id,
            &self.allocation.tensor,
            byte_offset,
            byte_count,
        ) {
            return Err(VulkanDistributedParameterBufferError(format!(
                "distributed parameter fragment {}..{} is outside tensor {:?} allocation {}..{} on {:?}",
                byte_offset,
                byte_offset.saturating_add(byte_count),
                self.allocation.tensor,
                self.allocation.byte_offset,
                self.allocation
                    .byte_offset
                    .saturating_add(self.allocation.byte_count),
                self.allocation.device_id,
            )));
        }
        let relative_byte_offset = byte_offset
            .checked_sub(self.allocation.byte_offset)
            .expect("fragment containment was checked above");
        let resident_byte_offset = self
            .byte_offset
            .checked_add(relative_byte_offset)
            .ok_or_else(|| {
                VulkanDistributedParameterBufferError(
                    "distributed parameter resident offset overflowed".to_string(),
                )
            })?;
        Ok(VulkanResidentKernelBufferBinding::new(
            binding,
            &self.buffer,
            byte_count,
        )
        .with_byte_offset(resident_byte_offset))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterBufferError(pub String);

impl Display for VulkanDistributedParameterBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedParameterBufferError {}

impl From<VulkanError> for VulkanDistributedParameterBufferError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}
