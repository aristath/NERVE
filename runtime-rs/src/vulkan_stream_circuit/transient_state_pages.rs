#[derive(Clone, Default)]
pub struct VulkanResidentTransientStatePageTable {
    states: BTreeMap<TransientStateKey, VulkanResidentTransientStatePages>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanResidentTransientStatePages {
    block_activation_capacity: usize,
    logical_to_physical: Vec<usize>,
    logical_page_blocks: Vec<Option<TransientStateBlockId>>,
    block_to_physical: BTreeMap<TransientStateBlockId, usize>,
    free_physical_pages: BTreeSet<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentTransientStatePageBinding {
    pub key: TransientStateKey,
    pub logical_activation_index: usize,
    pub transient_block_id: TransientStateBlockId,
    pub transient_block_activation_offset: usize,
    pub transient_block_activation_capacity: usize,
    pub resident_page_index: usize,
    pub resident_activation_offset: usize,
    pub resident_byte_offset: usize,
    pub bytes_per_activation: usize,
    pub copy_from_resident_byte_offset: Option<usize>,
    pub copy_byte_len: usize,
}

impl VulkanResidentTransientStatePageTable {
    pub(crate) fn snapshot_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(
            &self
                .states
                .iter()
                .map(|(key, pages)| {
                    serde_json::json!({
                        "node_instance_id": key.node_instance_id,
                        "state_id": key.state_id,
                        "block_activation_capacity": pages.block_activation_capacity,
                        "logical_to_physical": pages.logical_to_physical,
                        "logical_page_blocks": pages
                            .logical_page_blocks
                            .iter()
                            .map(|block| block.map(|value| value.0))
                            .collect::<Vec<_>>(),
                        "block_to_physical": pages
                            .block_to_physical
                            .iter()
                            .map(|(block, physical)| (block.0, *physical))
                            .collect::<Vec<_>>(),
                        "free_physical_pages": pages
                            .free_physical_pages
                            .iter()
                            .copied()
                            .collect::<Vec<_>>(),
                    })
                })
                .collect::<Vec<_>>(),
        )
        .expect("transient-state page table snapshot is serializable")
    }

    pub fn clear(&mut self) {
        self.states.clear();
    }

    fn has_free_physical_page_for(
        &self,
        key: &TransientStateKey,
        resident_state: &VulkanStreamStateBufferAllocation,
    ) -> bool {
        if resident_state.layout.dynamic_page_count == 0 {
            return true;
        }
        self.states
            .get(key)
            .map(|pages| !pages.free_physical_pages.is_empty())
            .unwrap_or(true)
    }

    fn retain_blocks(
        &mut self,
        snapshot: &TransientStateTableSnapshot,
    ) {
        let retained = snapshot
            .entries
            .iter()
            .map(|entry| {
                (
                    entry.key.clone(),
                    entry.block_ids.iter().copied().collect::<BTreeSet<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.states.retain(|key, pages| {
            let Some(retained_blocks) = retained.get(key) else {
                return false;
            };
            let removed_pages = pages
                .block_to_physical
                .iter()
                .filter_map(|(block, page)| (!retained_blocks.contains(block)).then_some(*page))
                .collect::<Vec<_>>();
            pages
                .block_to_physical
                .retain(|block, _| retained_blocks.contains(block));
            pages.free_physical_pages.extend(removed_pages);
            true
        });
    }

    fn sync_page_tables(
        &self,
        processor: &VulkanResidentInProcessPlacedStreamProcessor,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        for (key, pages) in &self.states {
            let state = processor.mounted_state_buffer(key).ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "cannot synchronize non-resident transaction state {}.{}",
                    key.node_instance_id, key.state_id,
                ))
            })?;
            state
                .buffer
                .write_bytes(
                    &state
                        .layout
                        .page_table_bytes_with_mapping(&pages.logical_to_physical)
                        .map_err(|error| {
                            VulkanResidentTokenModelPackageError::new(error.to_string())
                        })?,
                )
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }
        Ok(())
    }

    fn resident_page_indices(
        &self,
        key: &TransientStateKey,
    ) -> Vec<usize> {
        self.states
            .get(key)
            .map(|pages| {
                pages
                    .block_to_physical
                    .values()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn bind_slot(
        &mut self,
        resident_state: &VulkanStreamStateBufferAllocation,
        slot: &TransientStateSlot,
        preserve_copy_source: bool,
    ) -> Result<VulkanResidentTransientStatePageBinding, VulkanResidentTokenModelPackageError> {
        if resident_state.component_id != slot.key.node_instance_id
            || resident_state.state_id != slot.key.state_id
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} cannot bind resident state {}.{}",
                slot.key.node_instance_id,
                slot.key.state_id,
                resident_state.component_id,
                resident_state.state_id
            )));
        }
        if slot.block_activation_capacity == 0
            || slot.block_activation_offset >= slot.block_activation_capacity
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} has invalid block offset {}/{}",
                resident_state.component_id,
                resident_state.state_id,
                slot.block_activation_offset,
                slot.block_activation_capacity
            )));
        }

        if resident_state.layout.dynamic_page_count == 0 {
            if slot.block_activation_capacity != 1 || slot.block_activation_offset != 0 {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "fixed transient state {}.{} must bind one mutable singleton block",
                    resident_state.component_id, resident_state.state_id
                )));
            }
            return Ok(VulkanResidentTransientStatePageBinding {
                key: slot.key.clone(),
                logical_activation_index: 0,
                transient_block_id: slot.block_id,
                transient_block_activation_offset: 0,
                transient_block_activation_capacity: 1,
                resident_page_index: 0,
                resident_activation_offset: 0,
                resident_byte_offset: resident_state.layout.static_data_offset,
                bytes_per_activation: resident_state.layout.static_byte_capacity,
                copy_from_resident_byte_offset: None,
                copy_byte_len: 0,
            });
        }

        if slot.block_activation_capacity != resident_state.layout.block_activation_capacity {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} block capacity {} does not match resident page capacity {}",
                resident_state.component_id,
                resident_state.state_id,
                slot.block_activation_capacity,
                resident_state.layout.block_activation_capacity
            )));
        }
        let resident_activation_index =
            slot.logical_activation_index % resident_state.layout.dynamic_activation_capacity;
        let logical_page_index =
            resident_activation_index / slot.block_activation_capacity;
        if logical_page_index >= resident_state.layout.dynamic_page_count {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} logical page {} exceeds resident capacity {}",
                resident_state.component_id,
                resident_state.state_id,
                logical_page_index,
                resident_state.layout.dynamic_page_count
            )));
        }

        let pages = self
            .states
            .entry(slot.key.clone())
            .or_insert_with(|| VulkanResidentTransientStatePages {
                block_activation_capacity: slot.block_activation_capacity,
                logical_to_physical: (0..resident_state.layout.dynamic_page_count).collect(),
                logical_page_blocks: vec![None; resident_state.layout.dynamic_page_count],
                block_to_physical: BTreeMap::new(),
                free_physical_pages: (0..resident_state.layout.dynamic_page_count).collect(),
            });
        if pages.block_activation_capacity != slot.block_activation_capacity
            || pages.logical_to_physical.len() != resident_state.layout.dynamic_page_count
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "scheduled transient state {}.{} changed its resident page layout",
                resident_state.component_id, resident_state.state_id
            )));
        }

        let mut mapping_changed = false;
        let mut copy_from_resident_page = None;
        let resident_page_index = if let Some(page) =
            pages.block_to_physical.get(&slot.block_id).copied()
        {
            page
        } else {
            let previously_bound_block = pages.logical_page_blocks[logical_page_index];
            let page = if preserve_copy_source
                && let Some(source_block_id) = slot.copy_from_block_id
                && let Some(source_page) = pages.block_to_physical.get(&source_block_id).copied()
            {
                let page = pages.free_physical_pages.pop_first().ok_or_else(|| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "transactional page-COW for {}.{} exhausted {} physical pages before rollback state could be preserved",
                        resident_state.component_id,
                        resident_state.state_id,
                        resident_state.layout.dynamic_page_count,
                    ))
                })?;
                copy_from_resident_page = Some(source_page);
                page
            } else {
                previously_bound_block
                    .and_then(|block_id| pages.block_to_physical.remove(&block_id))
                    .or_else(|| pages.free_physical_pages.pop_first())
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "scheduled transient state {}.{} exhausted {} physical pages",
                            resident_state.component_id,
                            resident_state.state_id,
                            resident_state.layout.dynamic_page_count
                        ))
                    })?
            };
            pages.free_physical_pages.remove(&page);
            pages.block_to_physical.insert(slot.block_id, page);
            mapping_changed = true;
            page
        };
        if pages.logical_page_blocks[logical_page_index] != Some(slot.block_id)
            || pages.logical_to_physical[logical_page_index] != resident_page_index
        {
            pages.logical_page_blocks[logical_page_index] = Some(slot.block_id);
            pages.logical_to_physical[logical_page_index] = resident_page_index;
            mapping_changed = true;
        }

        if mapping_changed {
            resident_state
                .buffer
                .write_bytes(
                    &resident_state
                        .layout
                        .page_table_bytes_with_mapping(&pages.logical_to_physical)
                        .map_err(|error| {
                            VulkanResidentTokenModelPackageError::new(error.to_string())
                        })?,
                )
                .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        }

        let bytes_per_activation = resident_state.layout.bytes_per_activation.ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "dynamic transient state {}.{} has no activation stride",
                resident_state.component_id, resident_state.state_id
            ))
        })?;
        let resident_activation_offset = resident_page_index
            .checked_mul(slot.block_activation_capacity)
            .and_then(|page_offset| page_offset.checked_add(slot.block_activation_offset))
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "scheduled transient state {}.{} resident activation offset overflowed",
                    resident_state.component_id, resident_state.state_id
                ))
            })?;
        let resident_byte_offset = resident_state
            .layout
            .dynamic_physical_page_offset(resident_page_index)
            .and_then(|offset| {
                bytes_per_activation
                    .checked_mul(slot.block_activation_offset)
                    .and_then(|within_page| offset.checked_add(within_page))
                    .ok_or_else(|| {
                        VulkanError("transient state activation byte offset overflowed".to_string())
                    })
            })
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        let copy_byte_len = copy_from_resident_page
            .map(|_| {
                bytes_per_activation
                    .checked_mul(slot.block_activation_offset)
                    .ok_or_else(|| {
                        VulkanResidentTokenModelPackageError::new(format!(
                            "scheduled transient state {}.{} transaction page-copy length overflowed",
                            resident_state.component_id, resident_state.state_id,
                        ))
                    })
            })
            .transpose()?
            .unwrap_or_default();
        let copy_from_resident_byte_offset = copy_from_resident_page
            .filter(|_| copy_byte_len > 0)
            .map(|page| resident_state.layout.dynamic_physical_page_offset(page))
            .transpose()
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;

        Ok(VulkanResidentTransientStatePageBinding {
            key: slot.key.clone(),
            logical_activation_index: slot.logical_activation_index,
            transient_block_id: slot.block_id,
            transient_block_activation_offset: slot.block_activation_offset,
            transient_block_activation_capacity: slot.block_activation_capacity,
            resident_page_index,
            resident_activation_offset,
            resident_byte_offset,
            bytes_per_activation,
            copy_from_resident_byte_offset,
            copy_byte_len,
        })
    }

    fn resident_page_for_block(
        &self,
        key: &TransientStateKey,
        block_id: TransientStateBlockId,
    ) -> Option<usize> {
        self.states
            .get(key)
            .and_then(|pages| pages.block_to_physical.get(&block_id).copied())
    }
}

#[cfg(test)]
mod transient_state_page_tests {
    use super::*;

    fn key() -> TransientStateKey {
        TransientStateKey::new("layer_00", "kv")
    }

    fn resident_state(
        dynamic_activation_capacity: usize,
    ) -> (VulkanComputeDevice, VulkanStreamStateBufferAllocation) {
        let device = tests::selected_test_vulkan_device().unwrap();
        let plan = VulkanResidentStateBuffer {
            component_id: "layer_00".to_string(),
            state_id: "kv".to_string(),
            state_type: "kv".to_string(),
            dtype: None,
            layout: Some("paged".to_string()),
            static_elements: None,
            elements_per_activation: Some(4),
            max_dynamic_activations: Some(dynamic_activation_capacity),
            static_bytes: None,
            bytes_per_activation: Some(16),
            clone_from: None,
        };
        let layout =
            VulkanTransientStateBufferLayout::for_state(&plan, dynamic_activation_capacity)
                .unwrap();
        let buffer = device.create_resident_buffer(layout.byte_capacity).unwrap();
        buffer
            .write_bytes(&layout.initial_page_table_bytes().unwrap())
            .unwrap();
        (
            device,
            VulkanStreamStateBufferAllocation {
                component_id: plan.component_id,
                state_id: plan.state_id,
                state_type: plan.state_type,
                dtype: plan.dtype,
                byte_capacity: layout.byte_capacity,
                layout,
                static_byte_capacity: None,
                bytes_per_activation: Some(16),
                clone_from: None,
                buffer,
            },
        )
    }

    fn slot(
        block: u64,
        logical_activation_index: usize,
        block_activation_offset: usize,
    ) -> TransientStateSlot {
        TransientStateSlot {
            key: key(),
            logical_activation_index,
            block_id: TransientStateBlockId(block),
            block_activation_offset,
            block_activation_capacity: 64,
            allocated_block: false,
            copy_from_block_id: None,
        }
    }

    #[test]
    fn transient_state_page_table_reuses_physical_page_for_same_block() {
        let (_device, state) = resident_state(128);
        let mut table = VulkanResidentTransientStatePageTable::default();
        let first = table.bind_slot(&state, &slot(7, 0, 0), false).unwrap();
        let last = table.bind_slot(&state, &slot(7, 63, 63), false).unwrap();

        assert_eq!(first.resident_page_index, last.resident_page_index);
        assert_eq!(
            last.resident_byte_offset - first.resident_byte_offset,
            16 * 63
        );
    }

    #[test]
    fn transient_state_page_table_rebinds_a_logical_page_without_aliasing() {
        let (_device, state) = resident_state(128);
        let mut table = VulkanResidentTransientStatePageTable::default();
        let old = table.bind_slot(&state, &slot(7, 0, 0), false).unwrap();
        let replacement = table.bind_slot(&state, &slot(8, 0, 0), false).unwrap();
        let next = table.bind_slot(&state, &slot(9, 64, 0), false).unwrap();

        assert_eq!(old.resident_page_index, replacement.resident_page_index);
        assert_ne!(replacement.resident_page_index, next.resident_page_index);
    }

    #[test]
    fn transient_state_page_table_wraps_absolute_ticks_into_resident_pages() {
        let (_device, state) = resident_state(128);
        let mut table = VulkanResidentTransientStatePageTable::default();
        let first = table.bind_slot(&state, &slot(7, 0, 0), false).unwrap();
        let wrapped = table.bind_slot(&state, &slot(7, 128, 0), false).unwrap();

        assert_eq!(wrapped.logical_activation_index, 128);
        assert_eq!(wrapped.resident_page_index, first.resident_page_index);
        assert_eq!(wrapped.resident_byte_offset, first.resident_byte_offset);
    }

    #[test]
    fn transient_state_page_table_copy_on_write_preserves_the_source_page() {
        let (_device, state) = resident_state(128);
        let mut table = VulkanResidentTransientStatePageTable::default();
        let source = table.bind_slot(&state, &slot(7, 0, 0), false).unwrap();
        let mut replacement_slot = slot(8, 10, 10);
        replacement_slot.copy_from_block_id = Some(TransientStateBlockId(7));

        let replacement = table
            .bind_slot(&state, &replacement_slot, true)
            .unwrap();

        assert_ne!(replacement.resident_page_index, source.resident_page_index);
        assert_eq!(
            replacement.copy_from_resident_byte_offset,
            Some(source.resident_byte_offset),
        );
        assert_eq!(replacement.copy_byte_len, 16 * 10);
    }

    #[test]
    fn transient_state_page_table_copy_on_write_rejects_page_exhaustion() {
        let (_device, state) = resident_state(64);
        let mut table = VulkanResidentTransientStatePageTable::default();
        table.bind_slot(&state, &slot(7, 0, 0), false).unwrap();
        let mut replacement_slot = slot(8, 10, 10);
        replacement_slot.copy_from_block_id = Some(TransientStateBlockId(7));

        let error = table
            .bind_slot(&state, &replacement_slot, true)
            .unwrap_err();

        assert!(error.to_string().contains("exhausted 1 physical pages"));
    }
}
