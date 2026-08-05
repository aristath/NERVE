struct VulkanCausalStateSnapshotEntry {
    state_buffer_index: usize,
    static_byte_capacity: usize,
    snapshots: VulkanResidentBuffer,
}

struct VulkanCausalStateSnapshotBank {
    enabled: bool,
    lane_capacity: usize,
    required_state_buffer_indices: BTreeSet<usize>,
    entries: Vec<VulkanCausalStateSnapshotEntry>,
    commit_batches: Vec<Option<VulkanResidentBufferCopyBatch>>,
    dummy_buffer: VulkanResidentBuffer,
}

impl VulkanCausalStateSnapshotBank {
    fn new(
        device: &VulkanComputeDevice,
        lane_capacity: usize,
        enabled: bool,
    ) -> Result<Self, VulkanError> {
        if lane_capacity == 0 {
            return Err(VulkanError(
                "causal state snapshot lane capacity is zero".to_string(),
            ));
        }
        Ok(Self {
            enabled,
            lane_capacity,
            required_state_buffer_indices: BTreeSet::new(),
            entries: Vec::new(),
            commit_batches: Vec::new(),
            dummy_buffer: device.create_resident_buffer(std::mem::size_of::<u32>())?,
        })
    }

    fn require_state_buffer(&mut self, state_buffer_index: usize) {
        self.required_state_buffer_indices
            .insert(state_buffer_index);
    }

    fn binding_buffer<'a>(
        &'a mut self,
        device: &VulkanComputeDevice,
        buffers: &VulkanStreamCircuitStreamBuffers,
        state_buffer_index: usize,
    ) -> Result<&'a VulkanResidentBuffer, VulkanError> {
        self.require_state_buffer(state_buffer_index);
        if !self.enabled {
            return Ok(&self.dummy_buffer);
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.state_buffer_index == state_buffer_index)
        {
            return Ok(&self.entries[index].snapshots);
        }
        let state = buffers
            .state_buffers
            .get(state_buffer_index)
            .ok_or_else(|| {
                VulkanError(format!(
                    "causal state snapshot references absent state buffer {state_buffer_index}"
                ))
            })?;
        let static_byte_capacity = state.layout.static_byte_capacity;
        if static_byte_capacity == 0 {
            return Err(VulkanError(format!(
                "causal state snapshot references dynamic-only state {}.{}",
                state.component_id, state.state_id
            )));
        }
        let byte_capacity = static_byte_capacity
            .checked_mul(self.lane_capacity)
            .ok_or_else(|| VulkanError("causal state snapshot capacity overflowed".to_string()))?;
        self.entries.push(VulkanCausalStateSnapshotEntry {
            state_buffer_index,
            static_byte_capacity,
            snapshots: device.create_resident_buffer(byte_capacity)?,
        });
        Ok(&self
            .entries
            .last()
            .expect("causal state snapshot entry was inserted")
            .snapshots)
    }

    fn mount_commit_batches(
        &mut self,
        device: &VulkanComputeDevice,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<(), VulkanError> {
        self.commit_batches.clear();
        if !self.enabled
            || self.required_state_buffer_indices
                != self
                    .entries
                    .iter()
                    .map(|entry| entry.state_buffer_index)
                    .collect()
        {
            return Ok(());
        }
        for snapshot_index in 0..self.lane_capacity {
            let copies = self
                .entries
                .iter()
                .map(|entry| {
                    let state = &buffers.state_buffers[entry.state_buffer_index];
                    VulkanResidentBufferRangeCopy::new(
                        &entry.snapshots,
                        &state.buffer,
                        snapshot_index
                            .checked_mul(entry.static_byte_capacity)
                            .ok_or_else(|| {
                                VulkanError(
                                    "causal state snapshot offset overflowed".to_string(),
                                )
                            })?,
                        state.layout.static_data_offset,
                        entry.static_byte_capacity,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.commit_batches.push(
                (!copies.is_empty())
                    .then(|| device.create_resident_buffer_copy_batch(&copies))
                    .transpose()?,
            );
        }
        Ok(())
    }

    fn initialize_from_state_buffers(
        &self,
        buffers: &VulkanStreamCircuitStreamBuffers,
    ) -> Result<(), VulkanError> {
        if !self.enabled {
            return Ok(());
        }
        for entry in &self.entries {
            let state = buffers
                .state_buffers
                .get(entry.state_buffer_index)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "causal state snapshot fixture references absent state buffer {}",
                        entry.state_buffer_index,
                    ))
                })?;
            let source = state.buffer.read_bytes_at(
                state.layout.static_data_offset,
                entry.static_byte_capacity,
            )?;
            let byte_capacity = entry
                .static_byte_capacity
                .checked_mul(self.lane_capacity)
                .ok_or_else(|| {
                    VulkanError("causal state snapshot fixture capacity overflowed".to_string())
                })?;
            let mut snapshots = Vec::with_capacity(byte_capacity);
            for _ in 0..self.lane_capacity {
                snapshots.extend_from_slice(&source);
            }
            entry.snapshots.write_bytes(&snapshots)?;
        }
        Ok(())
    }

    fn update_digest(
        &self,
        buffers: &VulkanStreamCircuitStreamBuffers,
        digest: &mut Sha256,
    ) -> Result<(), VulkanError> {
        digest.update(b"causal_state_snapshots");
        digest.update([u8::from(self.enabled)]);
        let mut entries = self.entries.iter().collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.state_buffer_index);
        for entry in entries {
            let state = buffers
                .state_buffers
                .get(entry.state_buffer_index)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "causal state snapshot digest references absent state buffer {}",
                        entry.state_buffer_index,
                    ))
                })?;
            digest.update(entry.state_buffer_index.to_le_bytes());
            digest.update(state.component_id.as_bytes());
            digest.update(state.state_id.as_bytes());
            digest.update(entry.static_byte_capacity.to_le_bytes());
            digest.update(
                entry
                    .snapshots
                    .read_bytes(entry.snapshots.byte_capacity())?,
            );
        }
        Ok(())
    }

    fn total_byte_capacity(&self) -> usize {
        self.dummy_buffer.byte_capacity().saturating_add(
            self.entries
                .iter()
                .map(|entry| entry.snapshots.byte_capacity())
                .sum::<usize>(),
        )
    }

    fn can_commit_prefix(&self) -> bool {
        self.enabled
            && self.required_state_buffer_indices
                == self
                    .entries
                    .iter()
                    .map(|entry| entry.state_buffer_index)
                    .collect()
            && self.commit_batches.len() == self.lane_capacity
    }

    fn commit_prefix(&self, processed_tick_count: usize) -> Result<bool, VulkanError> {
        if processed_tick_count == 0 || processed_tick_count > self.lane_capacity {
            return Err(VulkanError(format!(
                "causal state snapshot prefix {processed_tick_count} exceeds capacity {}",
                self.lane_capacity
            )));
        }
        if !self.can_commit_prefix() {
            return Ok(false);
        }
        if let Some(batch) = &self.commit_batches[processed_tick_count - 1] {
            batch.run()?;
        }
        Ok(true)
    }
}
