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
        )
    }

    pub fn from_layout_for_components(
        device: &VulkanComputeDevice,
        address_table: &VulkanStableResourceAddressTable,
        layout: &VulkanCompiledResourceAddressLayout,
        execution_scope: Option<&str>,
        component_ids: &BTreeSet<String>,
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
            let words = table
                .slots
                .iter()
                .map(|slot| {
                    u32::try_from(*slot).map_err(|_| {
                        VulkanError(format!(
                            "dynamic resource parameter slot {slot} exceeds u32"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            let buffer =
                Arc::new(device.create_resident_buffer(bytes.len())?);
            buffer.write_bytes(&bytes)?;
            parameter_slots.insert(table.key.clone(), buffer);
        }
        Self::new(device, address_table.shared_buffer(), parameter_slots)
    }

    pub fn new(
        device: &VulkanComputeDevice,
        address_table: Arc<VulkanResidentBuffer>,
        parameter_slots: BTreeMap<
            VulkanDynamicResourceBindingKey,
            Arc<VulkanResidentBuffer>,
        >,
    ) -> Result<Self, VulkanError> {
        if address_table.byte_capacity() == 0 {
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
            parameter_slots,
        })
    }

    pub fn address_table(&self) -> &VulkanResidentBuffer {
        &self.address_table
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
