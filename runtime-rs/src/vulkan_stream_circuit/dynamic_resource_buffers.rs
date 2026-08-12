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
        }
        Self::new(
            device,
            address_table.shared_buffer(),
            address_table.slot_count(),
            parameter_slots,
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
        })
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
}
