#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentParameter {
    pub tensor: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub byte_count: Option<usize>,
    pub use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentStateBuffer {
    pub component_id: String,
    pub state_id: String,
    pub state_type: String,
    pub dtype: Option<String>,
    pub layout: Option<String>,
    pub static_elements: Option<usize>,
    pub elements_per_activation: Option<usize>,
    pub max_dynamic_activations: Option<usize>,
    pub static_bytes: Option<usize>,
    pub bytes_per_activation: Option<usize>,
    pub clone_from: Option<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentActivationBank {
    pub component_id: String,
    pub circuit_id: String,
    pub slot_count: usize,
    pub slots: Vec<VulkanResidentActivationSlot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanResidentActivationSlot {
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub max_elements: Option<usize>,
    pub bytes: Option<usize>,
}

pub struct VulkanStreamCircuitStreamBuffers {
    pub dynamic_state_capacity_activations: usize,
    pub state_buffers: Vec<VulkanStreamStateBufferAllocation>,
    pub selection_telemetry_buffers: Vec<VulkanSelectionTelemetryBufferAllocation>,
    pub activation_slot_buffers: Vec<VulkanActivationSlotBufferAllocation>,
    pub total_byte_capacity: usize,
}

pub struct VulkanStreamStateBufferAllocation {
    pub component_id: String,
    pub state_id: String,
    pub state_type: String,
    pub dtype: Option<String>,
    pub byte_capacity: usize,
    pub layout: VulkanTransientStateBufferLayout,
    pub static_byte_capacity: Option<usize>,
    pub bytes_per_activation: Option<usize>,
    pub clone_from: Option<(String, String)>,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanActivationSlotBufferAllocation {
    pub component_id: String,
    pub circuit_id: String,
    pub slot: usize,
    pub signal_ids: Vec<String>,
    pub byte_capacity: usize,
    pub shared_across_devices: bool,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanSelectionTelemetryBufferAllocation {
    pub component_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub resource_count: usize,
    pub byte_capacity: usize,
    pub buffer: VulkanResidentBuffer,
}

pub struct VulkanActivationSlotBufferOverride {
    pub component_id: String,
    pub slot: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

impl VulkanStreamCircuitStreamBuffers {
    pub fn state_buffer(
        &self,
        component_id: &str,
        state_id: &str,
    ) -> Option<&VulkanStreamStateBufferAllocation> {
        self.state_buffers
            .iter()
            .find(|buffer| buffer.component_id == component_id && buffer.state_id == state_id)
    }

    pub fn state_buffer_index(&self, component_id: &str, state_id: &str) -> Option<usize> {
        self.state_buffers
            .iter()
            .position(|buffer| buffer.component_id == component_id && buffer.state_id == state_id)
    }

    pub fn activation_slot_buffer(
        &self,
        component_id: &str,
        slot: usize,
    ) -> Option<&VulkanActivationSlotBufferAllocation> {
        self.activation_slot_buffers
            .iter()
            .find(|buffer| buffer.component_id == component_id && buffer.slot == slot)
    }

    pub fn activation_slot_buffer_index(&self, component_id: &str, slot: usize) -> Option<usize> {
        self.activation_slot_buffers
            .iter()
            .position(|buffer| buffer.component_id == component_id && buffer.slot == slot)
    }

    pub fn selection_telemetry_buffer(
        &self,
        component_id: &str,
        node_id: &str,
        domain_id: &str,
    ) -> Option<&VulkanSelectionTelemetryBufferAllocation> {
        self.selection_telemetry_buffers.iter().find(|buffer| {
            buffer.component_id == component_id
                && buffer.node_id == node_id
                && buffer.domain_id == domain_id
        })
    }

    pub fn selection_telemetry_buffer_index(
        &self,
        component_id: &str,
        node_id: &str,
        domain_id: &str,
    ) -> Option<usize> {
        self.selection_telemetry_buffers.iter().position(|buffer| {
            buffer.component_id == component_id
                && buffer.node_id == node_id
                && buffer.domain_id == domain_id
        })
    }

    pub fn zero_selection_telemetry_buffers(&self) -> Result<usize, VulkanError> {
        let mut total_zeroed = 0usize;
        for telemetry in &self.selection_telemetry_buffers {
            telemetry
                .buffer
                .write_bytes(&vec![0u8; telemetry.byte_capacity])?;
            total_zeroed = total_zeroed
                .checked_add(telemetry.byte_capacity)
                .ok_or_else(|| {
                    VulkanError("selection telemetry zero byte count overflowed".to_string())
                })?;
        }
        Ok(total_zeroed)
    }

    pub fn zero_state_buffers(&self) -> Result<usize, VulkanError> {
        let mut total_zeroed = 0usize;
        for state in &self.state_buffers {
            let mut bytes = vec![0u8; state.byte_capacity];
            let page_table = state.layout.initial_page_table_bytes()?;
            bytes[..page_table.len()].copy_from_slice(&page_table);
            state.buffer.write_bytes(&bytes)?;
            total_zeroed = total_zeroed
                .checked_add(state.byte_capacity)
                .ok_or_else(|| VulkanError("state zero byte count overflowed".to_string()))?;
        }
        Ok(total_zeroed)
    }

    pub fn initialize_state_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<usize, VulkanError> {
        let payloads = self
            .state_buffers
            .iter()
            .map(|state| {
                let initialized_byte_count = state.layout.dynamic_data_offset;
                let mut bytes = vec![0u8; initialized_byte_count];
                let page_table = state.layout.initial_page_table_bytes()?;
                bytes[..page_table.len()].copy_from_slice(&page_table);
                Ok(bytes)
            })
            .collect::<Result<Vec<_>, VulkanError>>()?;
        let ranges = self
            .state_buffers
            .iter()
            .zip(&payloads)
            .map(|(state, bytes)| {
                VulkanResidentBufferWriteRange::new(&state.buffer, 0, bytes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        device.write_resident_buffer_ranges(&ranges)
    }

    pub fn apply_clone_state_policies(&self) -> Result<usize, VulkanError> {
        self.apply_clone_state_policies_after(&BTreeSet::new())
    }

    fn inherit_matching_state_from(
        &self,
        source: &Self,
    ) -> Result<(usize, BTreeSet<(String, String)>), VulkanError> {
        let source_by_id = source
            .state_buffers
            .iter()
            .map(|state| ((state.component_id.as_str(), state.state_id.as_str()), state))
            .collect::<BTreeMap<_, _>>();
        let mut copied = BTreeSet::new();
        let mut total_copied = 0usize;
        for target in &self.state_buffers {
            let key = (target.component_id.as_str(), target.state_id.as_str());
            let Some(source) = source_by_id.get(&key) else {
                continue;
            };
            validate_state_buffer_copy(target, source)?;
            let bytes = source.buffer.read_bytes(source.byte_capacity)?;
            target.buffer.write_bytes(&bytes)?;
            total_copied = total_copied
                .checked_add(bytes.len())
                .ok_or_else(|| VulkanError("inherited state byte count overflowed".to_string()))?;
            copied.insert((target.component_id.clone(), target.state_id.clone()));
        }
        Ok((total_copied, copied))
    }

    fn apply_clone_state_policies_after(
        &self,
        initialized: &BTreeSet<(String, String)>,
    ) -> Result<usize, VulkanError> {
        let copies = ordered_clone_state_copies(
            self.state_buffers.iter().map(|state| {
                (
                    (state.component_id.clone(), state.state_id.clone()),
                    state.clone_from.clone(),
                )
            }),
            initialized,
        )?;
        let mut total_copied = 0usize;
        for (target_id, source_id) in copies {
            let target = self
                .state_buffer(&target_id.0, &target_id.1)
                .expect("planned clone target must exist");
            let source = self
                .state_buffer(&source_id.0, &source_id.1)
                .expect("planned clone source must exist");
            validate_state_buffer_copy(target, source)?;
            let bytes = source.buffer.read_bytes(source.byte_capacity)?;
            target.buffer.write_bytes(&bytes)?;
            total_copied = total_copied
                .checked_add(bytes.len())
                .ok_or_else(|| VulkanError("clone state byte count overflowed".to_string()))?;
        }
        Ok(total_copied)
    }
}

type VulkanStateBufferId = (String, String);

fn ordered_clone_state_copies(
    states: impl IntoIterator<Item = (VulkanStateBufferId, Option<VulkanStateBufferId>)>,
    initialized: &BTreeSet<VulkanStateBufferId>,
) -> Result<Vec<(VulkanStateBufferId, VulkanStateBufferId)>, VulkanError> {
    let states = states.into_iter().collect::<BTreeMap<_, _>>();
    let available = states.keys().cloned().collect::<BTreeSet<_>>();
    let mut pending = states
        .into_iter()
        .filter(|(target, _)| !initialized.contains(target))
        .filter_map(|(target, source)| source.map(|source| (target, source)))
        .collect::<Vec<_>>();
    for (target, source) in &pending {
        if !available.contains(source) {
            return Err(VulkanError(format!(
                "clone state target {}.{} references unavailable source {}.{}",
                target.0, target.1, source.0, source.1
            )));
        }
    }
    let clone_targets = pending
        .iter()
        .map(|(target, _)| target.clone())
        .collect::<BTreeSet<_>>();
    let mut copied = BTreeSet::new();
    let mut ordered = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let ready_index = pending
            .iter()
            .position(|(_, source)| !clone_targets.contains(source) || copied.contains(source));
        let Some(ready_index) = ready_index else {
            return Err(VulkanError(
                "clone state policies contain a dependency cycle".to_string(),
            ));
        };
        let copy = pending.remove(ready_index);
        copied.insert(copy.0.clone());
        ordered.push(copy);
    }
    Ok(ordered)
}

fn validate_state_buffer_copy(
    target: &VulkanStreamStateBufferAllocation,
    source: &VulkanStreamStateBufferAllocation,
) -> Result<(), VulkanError> {
    if target.state_type != source.state_type
        || target.byte_capacity != source.byte_capacity
        || target.static_byte_capacity != source.static_byte_capacity
        || target.bytes_per_activation != source.bytes_per_activation
        || target.layout != source.layout
    {
        return Err(VulkanError(format!(
            "cannot inherit state {}.{} ({}, {} bytes, static {:?}, per activation {:?}) from incompatible state {}.{} ({}, {} bytes, static {:?}, per activation {:?})",
            target.component_id,
            target.state_id,
            target.state_type,
            target.byte_capacity,
            target.static_byte_capacity,
            target.bytes_per_activation,
            source.component_id,
            source.state_id,
            source.state_type,
            source.byte_capacity,
            source.static_byte_capacity,
            source.bytes_per_activation,
        )));
    }
    Ok(())
}
