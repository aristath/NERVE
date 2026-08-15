#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanDynamicResourceBindingKey {
    pub component_id: String,
    pub node_id: String,
    pub selection_signal: String,
}

impl VulkanDynamicResourceBindingKey {
    pub fn new(
        component_id: impl Into<String>,
        node_id: impl Into<String>,
        selection_signal: impl Into<String>,
    ) -> Self {
        Self {
            component_id: component_id.into(),
            node_id: node_id.into(),
            selection_signal: selection_signal.into(),
        }
    }
}

pub struct VulkanDynamicResourceBuffers {
    address_table: Arc<VulkanResidentBuffer>,
    address_table_slot_count: usize,
    parameter_slots:
        BTreeMap<VulkanDynamicResourceBindingKey, Arc<VulkanResidentBuffer>>,
    parameter_slot_tables:
        BTreeMap<VulkanDynamicResourceBindingKey, VulkanCompiledParameterSlotTable>,
    execution_ownership_by_selector:
        std::sync::Mutex<BTreeMap<String, BTreeSet<usize>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanSelectedResourceRecordObservation {
    pub component_id: String,
    pub node_id: String,
    pub selector_id: String,
    pub resource_index: usize,
    pub parameter_ids: Vec<String>,
    pub parameter_slots: Vec<u32>,
    pub record_byte_counts: Vec<u64>,
    pub record_representations: Vec<u32>,
    pub invalid_parameter_indices: Vec<usize>,
}

impl VulkanDynamicResourceBuffers {
    pub fn from_layout(
        device: &VulkanComputeDevice,
        address_table: &VulkanStableResourceAddressTable,
        layout: &VulkanCompiledResourceAddressLayout,
    ) -> Result<Self, VulkanError> {
        Self::from_layout_for_components(
            device,
            address_table,
            layout,
            None,
            &BTreeSet::new(),
            None,
        )
    }

    pub fn from_layout_for_components(
        device: &VulkanComputeDevice,
        address_table: &VulkanStableResourceAddressTable,
        layout: &VulkanCompiledResourceAddressLayout,
        execution_scope: Option<&str>,
        component_ids: &BTreeSet<String>,
        selector_ownership: Option<&VulkanCompiledResourceSelectorOwnership>,
    ) -> Result<Self, VulkanError> {
        if address_table.slot_count() != layout.slot_count() {
            return Err(VulkanError(format!(
                "dynamic resource layout has {} slots but its address table has {}",
                layout.slot_count(),
                address_table.slot_count()
            )));
        }
        let mut parameter_slots = BTreeMap::new();
        let mut parameter_slot_tables = BTreeMap::new();
        let mut execution_ownership_by_selector = BTreeMap::new();
        for table in layout.parameter_slot_tables.iter().filter(|table| {
            execution_scope
                .is_none_or(|scope| table.execution_scope == scope)
                && (component_ids.is_empty()
                    || component_ids.contains(&table.key.component_id))
        }) {
            let words = dynamic_parameter_slot_words(table, |resource_index| {
                selector_ownership.is_none_or(|ownership| {
                    ownership.owns(&table.selector_id, resource_index)
                })
            })?;
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            let buffer =
                Arc::new(device.create_resident_buffer(bytes.len())?);
            buffer.write_bytes(&bytes)?;
            parameter_slots.insert(table.key.clone(), buffer);
            parameter_slot_tables.insert(table.key.clone(), table.clone());
            let owned = (0..table.resource_count)
                .filter(|resource_index| {
                    selector_ownership.is_none_or(|ownership| {
                        ownership.owns(&table.selector_id, *resource_index)
                    })
                })
                .collect::<BTreeSet<_>>();
            if let Some(previous) = execution_ownership_by_selector
                .insert(table.selector_id.clone(), owned.clone())
                && previous != owned
            {
                return Err(VulkanError(format!(
                    "dynamic resource selector {:?} changes execution ownership between parameter-slot tables",
                    table.selector_id,
                )));
            }
        }
        Self::new_with_parameter_slot_tables(
            device,
            address_table.shared_buffer(),
            address_table.slot_count(),
            parameter_slots,
            parameter_slot_tables,
            execution_ownership_by_selector,
        )
    }

    pub fn new(
        device: &VulkanComputeDevice,
        address_table: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
        parameter_slots: BTreeMap<
            VulkanDynamicResourceBindingKey,
            Arc<VulkanResidentBuffer>,
        >,
    ) -> Result<Self, VulkanError> {
        if address_table.byte_capacity() == 0 || address_table_slot_count == 0 {
            return Err(VulkanError(
                "dynamic resource address table must not be empty".to_string(),
            ));
        }
        if !device.owns_resident_buffer(&address_table) {
            return Err(VulkanError(
                "dynamic resource address table belongs to another logical device"
                    .to_string(),
            ));
        }
        for (key, slots) in &parameter_slots {
            if key.component_id.trim().is_empty()
                || key.node_id.trim().is_empty()
                || key.selection_signal.trim().is_empty()
                || slots.byte_capacity() == 0
            {
                return Err(VulkanError(
                    "dynamic resource parameter-slot binding is invalid"
                        .to_string(),
                ));
            }
            if !device.owns_resident_buffer(slots) {
                return Err(VulkanError(format!(
                    "dynamic resource parameter slots for {}.{} use another logical device",
                    key.component_id, key.node_id
                )));
            }
        }
        Ok(Self {
            address_table,
            address_table_slot_count,
            parameter_slots,
            parameter_slot_tables: BTreeMap::new(),
            execution_ownership_by_selector: std::sync::Mutex::new(BTreeMap::new()),
        })
    }

    fn new_with_parameter_slot_tables(
        device: &VulkanComputeDevice,
        address_table: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
        parameter_slots: BTreeMap<
            VulkanDynamicResourceBindingKey,
            Arc<VulkanResidentBuffer>,
        >,
        parameter_slot_tables: BTreeMap<
            VulkanDynamicResourceBindingKey,
            VulkanCompiledParameterSlotTable,
        >,
        execution_ownership_by_selector: BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<Self, VulkanError> {
        let mut buffers = Self::new(
            device,
            address_table,
            address_table_slot_count,
            parameter_slots,
        )?;
        if buffers.parameter_slots.keys().ne(parameter_slot_tables.keys()) {
            return Err(VulkanError(
                "dynamic resource parameter-slot metadata differs from its buffers"
                    .to_string(),
            ));
        }
        buffers.parameter_slot_tables = parameter_slot_tables;
        buffers.execution_ownership_by_selector =
            std::sync::Mutex::new(execution_ownership_by_selector);
        Ok(buffers)
    }

    pub fn address_table(&self) -> &VulkanResidentBuffer {
        &self.address_table
    }

    pub fn shared_address_table(&self) -> Arc<VulkanResidentBuffer> {
        Arc::clone(&self.address_table)
    }

    pub fn address_table_slot_count(&self) -> usize {
        self.address_table_slot_count
    }

    pub fn parameter_slots(
        &self,
        component_id: &str,
        node_id: &str,
        selection_signal: &str,
    ) -> Option<&VulkanResidentBuffer> {
        self.parameter_slots
            .get(&VulkanDynamicResourceBindingKey::new(
                component_id,
                node_id,
                selection_signal,
            ))
            .map(Arc::as_ref)
    }

    pub fn parameter_slot_binding_count(&self) -> usize {
        self.parameter_slots.len()
    }

    pub(crate) fn selected_resource_record_observations(
        &self,
        selected_resource_indices: &BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<Vec<VulkanSelectedResourceRecordObservation>, VulkanError> {
        let address_bytes = self
            .address_table
            .read_bytes(self.address_table.byte_capacity())?;
        let mut observations = Vec::new();
        for (key, table) in &self.parameter_slot_tables {
            let Some(selected) = selected_resource_indices.get(&table.selector_id) else {
                continue;
            };
            let slots = self.parameter_slots.get(key).ok_or_else(|| {
                VulkanError(format!(
                    "dynamic resource parameter-slot buffer for {}.{} disappeared",
                    key.component_id, key.node_id,
                ))
            })?;
            let slot_bytes = slots.read_bytes(slots.byte_capacity())?;
            observations.extend(selected_resource_record_observations_from_bytes(
                table,
                selected,
                &slot_bytes,
                &address_bytes,
            )?);
        }
        Ok(observations)
    }

    /// Creates stream-owned arithmetic-ownership tables over the same stable
    /// package address space. Resource residency and addresses remain shared;
    /// later ownership changes cannot leak into another active stream.
    pub(crate) fn fork_for_stream(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<Arc<Self>, VulkanError> {
        if !device.owns_resident_buffer(&self.address_table) {
            return Err(VulkanError(
                "dynamic resource stream fork belongs to another logical device"
                    .to_string(),
            ));
        }
        let execution_ownership_by_selector = self
            .execution_ownership_by_selector
            .lock()
            .map_err(|_| {
                VulkanError(
                    "dynamic resource execution-ownership state was poisoned"
                        .to_string(),
                )
            })?
            .clone();
        let mut parameter_slots = BTreeMap::new();
        for (key, table) in &self.parameter_slot_tables {
            let owned = execution_ownership_by_selector
                .get(&table.selector_id)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "dynamic resource stream fork omits selector {:?}",
                        table.selector_id,
                    ))
                })?;
            let words = dynamic_parameter_slot_words(table, |resource_index| {
                owned.contains(&resource_index)
            })?;
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            let buffer = Arc::new(device.create_resident_buffer(bytes.len())?);
            buffer.write_bytes(&bytes)?;
            parameter_slots.insert(key.clone(), buffer);
        }
        Self::new_with_parameter_slot_tables(
            device,
            Arc::clone(&self.address_table),
            self.address_table_slot_count,
            parameter_slots,
            self.parameter_slot_tables.clone(),
            execution_ownership_by_selector,
        )
        .map(Arc::new)
    }

    pub(crate) fn selector_execution_ownership(
        &self,
        selector_id: &str,
    ) -> Result<Option<BTreeSet<usize>>, VulkanError> {
        self.execution_ownership_by_selector
            .lock()
            .map(|ownership| ownership.get(selector_id).cloned())
            .map_err(|_| {
                VulkanError(
                    "dynamic resource execution-ownership state was poisoned"
                        .to_string(),
                )
            })
    }

    /// Replaces one selector's arithmetic ownership without changing stable
    /// resource addresses or descriptor bindings.
    ///
    /// The caller must establish a quiescent execution boundary. All matching
    /// parameter-slot tables on this logical device are validated before the
    /// first write and rolled back if any write fails.
    pub(crate) fn replace_selector_execution_ownership_at_quiescent_boundary(
        &self,
        selector_id: &str,
        owned_resource_indices: &BTreeSet<usize>,
    ) -> Result<(), VulkanError> {
        let mut execution_ownership = self
            .execution_ownership_by_selector
            .lock()
            .map_err(|_| {
            VulkanError(
                "dynamic resource execution-ownership update lock was poisoned"
                    .to_string(),
            )
        })?;
        if !execution_ownership.contains_key(selector_id) {
            return Err(VulkanError(format!(
                "dynamic resource execution ownership references unbound selector {selector_id:?}",
            )));
        }
        let replacements = dynamic_parameter_slot_replacements(
            &self.parameter_slot_tables,
            selector_id,
            owned_resource_indices,
        )?;
        let mut previous = Vec::with_capacity(replacements.len());
        for (key, words) in &replacements {
            let buffer = self.parameter_slots.get(key).ok_or_else(|| {
                VulkanError(format!(
                    "dynamic resource parameter-slot buffer for {}.{} disappeared",
                    key.component_id, key.node_id,
                ))
            })?;
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            if bytes.len() != buffer.byte_capacity() {
                return Err(VulkanError(format!(
                    "dynamic resource parameter-slot replacement for {}.{} has {} bytes, expected {}",
                    key.component_id,
                    key.node_id,
                    bytes.len(),
                    buffer.byte_capacity(),
                )));
            }
            previous.push((key, buffer.read_bytes(buffer.byte_capacity())?, bytes));
        }
        for (replacement_index, (key, _, bytes)) in previous.iter().enumerate() {
            let buffer = self
                .parameter_slots
                .get(*key)
                .expect("replacement buffers were validated above");
            if let Err(error) = buffer.write_bytes(bytes) {
                let mut rollback_error = None;
                for (written_key, old_bytes, _) in
                    previous[..replacement_index].iter().rev()
                {
                    if let Err(error) = self
                        .parameter_slots
                        .get(*written_key)
                        .expect("rollback buffers were validated above")
                        .write_bytes(old_bytes)
                        && rollback_error.is_none()
                    {
                        rollback_error = Some(error);
                    }
                }
                return Err(match rollback_error {
                    Some(rollback_error) => VulkanError(format!(
                        "failed to replace selector {selector_id:?} execution ownership: {error}; rollback also failed: {rollback_error}",
                    )),
                    None => error,
                });
            }
        }
        execution_ownership.insert(
            selector_id.to_string(),
            owned_resource_indices.clone(),
        );
        Ok(())
    }
}

fn selected_resource_record_observations_from_bytes(
    table: &VulkanCompiledParameterSlotTable,
    selected_resource_indices: &BTreeSet<usize>,
    parameter_slot_bytes: &[u8],
    address_table_bytes: &[u8],
) -> Result<Vec<VulkanSelectedResourceRecordObservation>, VulkanError> {
    const ADDRESS_RECORD_BYTE_COUNT: usize = 32;
    const UNBOUND_PARAMETER_SLOT: u32 = u32::MAX;

    if !parameter_slot_bytes.len().is_multiple_of(size_of::<u32>())
        || !address_table_bytes.len().is_multiple_of(ADDRESS_RECORD_BYTE_COUNT)
    {
        return Err(VulkanError(
            "selected-resource record observation received a misaligned device table"
                .to_string(),
        ));
    }
    let slot_words = parameter_slot_bytes
        .chunks_exact(size_of::<u32>())
        .map(|bytes| {
            u32::from_le_bytes(bytes.try_into().expect("u32 slot chunks are exact"))
        })
        .collect::<Vec<_>>();
    let slot_count = table.slot_count().ok_or_else(|| {
        VulkanError(format!(
            "selected-resource parameter table for selector {:?} overflows",
            table.selector_id,
        ))
    })?;
    if table.resource_count == 0
        || !slot_count.is_multiple_of(table.resource_count)
        || slot_words.len() != slot_count
    {
        return Err(VulkanError(format!(
            "selected-resource parameter table for selector {:?} has invalid geometry",
            table.selector_id,
        )));
    }
    let parameters_per_resource = slot_count / table.resource_count;
    if table.parameter_ids.len() != parameters_per_resource {
        return Err(VulkanError(format!(
            "selected-resource parameter table for selector {:?} names {} parameters but has {parameters_per_resource} slots per resource",
            table.selector_id,
            table.parameter_ids.len(),
        )));
    }

    selected_resource_indices
        .iter()
        .map(|resource_index| {
            let slot_start = resource_index
                .checked_mul(parameters_per_resource)
                .ok_or_else(|| {
                    VulkanError(
                        "selected-resource parameter slot offset overflowed".to_string(),
                    )
                })?;
            let slot_end = slot_start
                .checked_add(parameters_per_resource)
                .ok_or_else(|| {
                    VulkanError(
                        "selected-resource parameter slot range overflowed".to_string(),
                    )
                })?;
            let parameter_slots = slot_words
                .get(slot_start..slot_end)
                .ok_or_else(|| {
                    VulkanError(format!(
                        "selected-resource index {resource_index} exceeds selector {:?} parameter slots",
                        table.selector_id,
                    ))
                })?
                .to_vec();
            let mut record_byte_counts = Vec::with_capacity(parameters_per_resource);
            let mut record_representations = Vec::with_capacity(parameters_per_resource);
            let mut invalid_parameter_indices = Vec::new();
            for (parameter_index, slot) in parameter_slots.iter().copied().enumerate() {
                let record = usize::try_from(slot)
                    .ok()
                    .filter(|_| slot != UNBOUND_PARAMETER_SLOT)
                    .and_then(|slot| slot.checked_mul(ADDRESS_RECORD_BYTE_COUNT))
                    .and_then(|start| {
                        start
                            .checked_add(ADDRESS_RECORD_BYTE_COUNT)
                            .and_then(|end| address_table_bytes.get(start..end))
                    });
                let Some(record) = record else {
                    record_byte_counts.push(0);
                    record_representations.push(0);
                    invalid_parameter_indices.push(parameter_index);
                    continue;
                };
                let device_address = u64::from_le_bytes(
                    record[0..8].try_into().expect("address record u64"),
                );
                let byte_count = u64::from_le_bytes(
                    record[8..16].try_into().expect("address record u64"),
                );
                let generation = u64::from_le_bytes(
                    record[16..24].try_into().expect("address record u64"),
                );
                let resident = u32::from_le_bytes(
                    record[24..28].try_into().expect("address record u32"),
                );
                let representation = u32::from_le_bytes(
                    record[28..32].try_into().expect("address record u32"),
                );
                record_byte_counts.push(byte_count);
                record_representations.push(representation);
                if device_address == 0 || byte_count == 0 || generation == 0 || resident != 1 {
                    invalid_parameter_indices.push(parameter_index);
                }
            }
            Ok(VulkanSelectedResourceRecordObservation {
                component_id: table.key.component_id.clone(),
                node_id: table.key.node_id.clone(),
                selector_id: table.selector_id.clone(),
                resource_index: *resource_index,
                parameter_ids: table.parameter_ids.clone(),
                parameter_slots,
                record_byte_counts,
                record_representations,
                invalid_parameter_indices,
            })
        })
        .collect()
}

fn dynamic_parameter_slot_replacements(
    tables: &BTreeMap<
        VulkanDynamicResourceBindingKey,
        VulkanCompiledParameterSlotTable,
    >,
    selector_id: &str,
    owned_resource_indices: &BTreeSet<usize>,
) -> Result<BTreeMap<VulkanDynamicResourceBindingKey, Vec<u32>>, VulkanError> {
    if selector_id.trim().is_empty() || owned_resource_indices.is_empty() {
        return Err(VulkanError(
            "dynamic resource execution ownership requires a selector and at least one resource"
                .to_string(),
        ));
    }
    let matching = tables
        .values()
        .filter(|table| table.selector_id == selector_id)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(VulkanError(format!(
            "dynamic resource execution ownership references unbound selector {selector_id:?}",
        )));
    }
    if matching.iter().any(|table| {
        owned_resource_indices
            .iter()
            .any(|resource_index| *resource_index >= table.resource_count)
    }) {
        return Err(VulkanError(format!(
            "dynamic resource execution ownership for selector {selector_id:?} exceeds its resource count",
        )));
    }
    matching
        .into_iter()
        .map(|table| {
            dynamic_parameter_slot_words(table, |resource_index| {
                owned_resource_indices.contains(&resource_index)
            })
            .map(|words| (table.key.clone(), words))
        })
        .collect()
}

fn dynamic_parameter_slot_words(
    table: &VulkanCompiledParameterSlotTable,
    owns_resource: impl Fn(usize) -> bool,
) -> Result<Vec<u32>, VulkanError> {
    let slot_count = table.slot_count().ok_or_else(|| {
        VulkanError(format!(
            "dynamic resource parameter-slot table for selector {:?} overflows",
            table.selector_id
        ))
    })?;
    if table.resource_count == 0 || !slot_count.is_multiple_of(table.resource_count) {
        return Err(VulkanError(format!(
            "dynamic resource parameter-slot table for selector {:?} has invalid resource geometry",
            table.selector_id
        )));
    }
    let parameters_per_resource = slot_count / table.resource_count;
    table
        .slots()
        .enumerate()
        .map(|(index, slot)| {
            let resource_index = index / parameters_per_resource;
            if !owns_resource(resource_index) {
                return Ok(u32::MAX);
            }
            u32::try_from(slot).map_err(|_| {
                VulkanError(format!(
                    "dynamic resource parameter slot {slot} exceeds u32"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod dynamic_resource_buffer_tests {
    use super::*;

    fn parameter_slot_table(
        resource_count: usize,
        mapping: VulkanCompiledParameterSlotMapping,
    ) -> VulkanCompiledParameterSlotTable {
        VulkanCompiledParameterSlotTable {
            key: VulkanDynamicResourceBindingKey::new("block", "experts", "routes"),
            selector_id: "experts".to_string(),
            execution_scope: "decode".to_string(),
            parameter_ids: vec!["weight".to_string(), "scale".to_string()],
            resource_count,
            mapping,
        }
    }

    #[test]
    fn device_parameter_slots_encode_exact_ownership_without_residency() {
        let explicit = parameter_slot_table(
            3,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: vec![7, 8, 11, 12, 20, 21],
            },
        );
        assert_eq!(
            dynamic_parameter_slot_words(&explicit, |resource| resource != 1).unwrap(),
            vec![7, 8, u32::MAX, u32::MAX, 20, 21],
        );

        let partitioned = parameter_slot_table(
            3,
            VulkanCompiledParameterSlotMapping::Partitioned {
                parameter_slot_bases: vec![10, 20],
            },
        );
        assert_eq!(
            dynamic_parameter_slot_words(&partitioned, |resource| resource == 1).unwrap(),
            vec![u32::MAX, u32::MAX, 11, 21, u32::MAX, u32::MAX],
        );

        let malformed = parameter_slot_table(
            0,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: Vec::new(),
            },
        );
        assert!(
            dynamic_parameter_slot_words(&malformed, |_| true)
                .unwrap_err()
                .to_string()
                .contains("invalid resource geometry")
        );
    }

    #[test]
    fn selector_ownership_replacement_updates_every_bound_operation_and_rejects_gaps() {
        let first = parameter_slot_table(
            3,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: vec![7, 8, 11, 12, 20, 21],
            },
        );
        let mut second = first.clone();
        second.key.node_id = "experts-down".to_string();
        second.mapping = VulkanCompiledParameterSlotMapping::Partitioned {
            parameter_slot_bases: vec![30, 40],
        };
        let tables = [first, second]
            .into_iter()
            .map(|table| (table.key.clone(), table))
            .collect::<BTreeMap<_, _>>();
        let replacements = dynamic_parameter_slot_replacements(
            &tables,
            "experts",
            &BTreeSet::from([1]),
        )
        .unwrap();
        assert_eq!(replacements.len(), 2);
        assert_eq!(
            replacements[&VulkanDynamicResourceBindingKey::new(
                "block", "experts", "routes"
            )],
            vec![u32::MAX, u32::MAX, 11, 12, u32::MAX, u32::MAX],
        );
        assert_eq!(
            replacements[&VulkanDynamicResourceBindingKey::new(
                "block",
                "experts-down",
                "routes"
            )],
            vec![u32::MAX, u32::MAX, 31, 41, u32::MAX, u32::MAX],
        );
        assert!(dynamic_parameter_slot_replacements(
            &tables,
            "experts",
            &BTreeSet::new(),
        )
        .unwrap_err()
        .to_string()
        .contains("at least one resource"));
        assert!(dynamic_parameter_slot_replacements(
            &tables,
            "missing",
            &BTreeSet::from([0]),
        )
        .unwrap_err()
        .to_string()
        .contains("unbound selector"));
        assert!(dynamic_parameter_slot_replacements(
            &tables,
            "experts",
            &BTreeSet::from([3]),
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds its resource count"));
    }

    #[test]
    fn selected_resource_record_observation_reports_exact_slots_and_invalid_records() {
        let table = parameter_slot_table(
            2,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: vec![0, 1, 2, 3],
            },
        );
        let parameter_slot_bytes = [0u32, 1, 2, u32::MAX]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut address_table_bytes = Vec::new();
        for (address, byte_count, generation, resident, representation) in [
            (0x1000u64, 64u64, 1u64, 1u32, 0u32),
            (0x2000, 16, 2, 1, 1),
            (0, 64, 3, 1, 0),
            (0x4000, 16, 4, 1, 0),
        ] {
            address_table_bytes.extend(address.to_le_bytes());
            address_table_bytes.extend(byte_count.to_le_bytes());
            address_table_bytes.extend(generation.to_le_bytes());
            address_table_bytes.extend(resident.to_le_bytes());
            address_table_bytes.extend(representation.to_le_bytes());
        }

        let observations = selected_resource_record_observations_from_bytes(
            &table,
            &BTreeSet::from([0, 1]),
            &parameter_slot_bytes,
            &address_table_bytes,
        )
        .unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].resource_index, 0);
        assert_eq!(observations[0].parameter_slots, vec![0, 1]);
        assert_eq!(observations[0].record_byte_counts, vec![64, 16]);
        assert_eq!(observations[0].record_representations, vec![0, 1]);
        assert!(observations[0].invalid_parameter_indices.is_empty());
        assert_eq!(observations[1].resource_index, 1);
        assert_eq!(observations[1].parameter_slots, vec![2, u32::MAX]);
        assert_eq!(observations[1].record_byte_counts, vec![64, 0]);
        assert_eq!(observations[1].invalid_parameter_indices, vec![0, 1]);
    }

    #[test]
    fn selected_resource_record_observation_rejects_misaligned_tables_and_oob_resources() {
        let table = parameter_slot_table(
            1,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: vec![0, 1],
            },
        );
        assert!(
            selected_resource_record_observations_from_bytes(
                &table,
                &BTreeSet::from([0]),
                &[0; 7],
                &[0; 64],
            )
            .unwrap_err()
            .to_string()
            .contains("misaligned")
        );
        assert!(
            selected_resource_record_observations_from_bytes(
                &table,
                &BTreeSet::from([1]),
                &[0; 8],
                &[0; 64],
            )
            .unwrap_err()
            .to_string()
            .contains("exceeds")
        );
    }
}
