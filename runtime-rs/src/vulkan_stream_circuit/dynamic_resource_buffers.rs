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
}
