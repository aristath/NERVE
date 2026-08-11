#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBufferPlan {
    pub backend_id: String,
    pub device_id: String,
    pub signal_element_bytes: Option<usize>,
    pub inputs: Vec<VulkanModelBoundaryBuffer>,
    pub outputs: Vec<VulkanModelBoundaryBuffer>,
    pub input_count: usize,
    pub output_count: usize,
    pub total_buffer_count: usize,
    pub total_byte_capacity: Option<usize>,
    pub unresolved_byte_signals: Vec<String>,
}

impl VulkanModelBoundaryBufferPlan {
    pub fn from_placed_plan(
        placed_plan: &VulkanPlacedStreamCircuitPlan,
    ) -> Result<Self, VulkanModelBoundaryBufferPlanError> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut total_byte_capacity = Some(0usize);
        let mut unresolved_byte_signals = Vec::new();
        let signal_element_bytes = placed_plan.placed_resident_plan.signal_element_bytes;

        for circuit in &placed_plan.binding_plan.circuits {
            for input in &circuit.input_ports {
                if placed_plan
                    .placed_resident_plan
                    .local_edges
                    .iter()
                    .chain(&placed_plan.placed_resident_plan.incoming_edges)
                    .any(|edge| {
                        edge.destination_component_id == circuit.component_id
                            && edge.destination_port_id == input.id
                    })
                {
                    continue;
                }
                let boundary = VulkanModelBoundaryBuffer::from_port(
                    inputs.len(),
                    &circuit.component_id,
                    input,
                    signal_element_bytes,
                )?;
                total_byte_capacity =
                    add_optional_boundary_bytes(total_byte_capacity, boundary.byte_capacity)?;
                if boundary.byte_capacity.is_none() {
                    unresolved_byte_signals.push(boundary.signal_id.clone());
                }
                inputs.push(boundary);
            }

            for output in &circuit.output_ports {
                if placed_plan
                    .placed_resident_plan
                    .local_edges
                    .iter()
                    .chain(&placed_plan.placed_resident_plan.outgoing_edges)
                    .any(|edge| {
                        edge.source_component_id == circuit.component_id
                            && edge.source_port_id == output.id
                    })
                {
                    continue;
                }
                let boundary = VulkanModelBoundaryBuffer::from_port(
                    outputs.len(),
                    &circuit.component_id,
                    output,
                    signal_element_bytes,
                )?;
                let aliases_input = boundary.source_signal_id.is_some()
                    && inputs.iter().any(|input: &VulkanModelBoundaryBuffer| {
                        input.component_id == boundary.component_id
                            && input.signal_id == boundary.signal_id
                            && input.shape == boundary.shape
                    });
                if !aliases_input {
                    total_byte_capacity =
                        add_optional_boundary_bytes(total_byte_capacity, boundary.byte_capacity)?;
                }
                if boundary.byte_capacity.is_none() {
                    unresolved_byte_signals.push(boundary.signal_id.clone());
                }
                outputs.push(boundary);
            }
        }

        let input_count = inputs.len();
        let output_count = outputs.len();
        Ok(Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            device_id: placed_plan.device_id.clone(),
            signal_element_bytes,
            total_buffer_count: input_count + output_count,
            input_count,
            output_count,
            inputs,
            outputs,
            total_byte_capacity,
            unresolved_byte_signals,
        })
    }

    pub fn allocate_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanModelBoundaryBuffers, VulkanError> {
        self.allocate_buffers_with_overrides(device, &[])
    }

    pub fn allocate_buffers_with_overrides(
        &self,
        device: &VulkanComputeDevice,
        buffer_overrides: &[VulkanModelBoundaryBufferOverride],
    ) -> Result<VulkanModelBoundaryBuffers, VulkanError> {
        let mut input_buffers = Vec::with_capacity(self.inputs.len());
        let mut output_buffers = Vec::with_capacity(self.outputs.len());
        let mut total_byte_capacity = 0usize;
        let mut overrides = BTreeMap::new();

        for override_ in buffer_overrides {
            let key = (
                override_.direction,
                override_.component_id.as_str(),
                override_.signal_id.as_str(),
            );
            if overrides.insert(key, override_).is_some() {
                return Err(VulkanError(format!(
                    "model boundary buffer override repeats {:?} {}.{} on {:?}",
                    override_.direction,
                    override_.component_id,
                    override_.signal_id,
                    self.device_id,
                )));
            }
            if !device.owns_resident_buffer(&override_.buffer) {
                return Err(VulkanError(format!(
                    "model boundary buffer override for {:?} {}.{} on {:?} belongs to a different Vulkan logical device",
                    override_.direction,
                    override_.component_id,
                    override_.signal_id,
                    self.device_id,
                )));
            }
            if !override_.buffer.is_shared_host_backed()
                && !override_.buffer.is_shared_device_memory_backed()
            {
                return Err(VulkanError(format!(
                    "model boundary buffer override for {:?} {}.{} is not backed by shared host or shared device memory",
                    override_.direction, override_.component_id, override_.signal_id,
                )));
            }
            let boundaries = match override_.direction {
                VulkanModelBoundaryDirection::Input => &self.inputs,
                VulkanModelBoundaryDirection::Output => &self.outputs,
            };
            let boundary = boundaries
                .iter()
                .find(|boundary| {
                    boundary.component_id == override_.component_id
                        && boundary.signal_id == override_.signal_id
                })
                .ok_or_else(|| {
                    VulkanError(format!(
                        "model boundary buffer override does not address {:?} {}.{} on {:?}",
                        override_.direction,
                        override_.component_id,
                        override_.signal_id,
                        self.device_id,
                    ))
                })?;
            let required_byte_capacity = boundary.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} model boundary {:?} has unknown byte capacity",
                    self.device_id, boundary.signal_id,
                ))
            })?;
            if override_.buffer.byte_capacity() < required_byte_capacity {
                return Err(VulkanError(format!(
                    "model boundary buffer override for {:?} {}.{} has {} bytes, needs {required_byte_capacity}",
                    override_.direction,
                    override_.component_id,
                    override_.signal_id,
                    override_.buffer.byte_capacity(),
                )));
            }
        }

        for boundary in &self.inputs {
            let byte_capacity = boundary.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} model input boundary {:?} has unknown byte capacity",
                    self.device_id, boundary.signal_id
                ))
            })?;
            total_byte_capacity = checked_add_bytes(
                total_byte_capacity,
                byte_capacity,
                "model input boundary buffer allocation",
            )?;
            let buffer = overrides
                .get(&(
                    VulkanModelBoundaryDirection::Input,
                    boundary.component_id.as_str(),
                    boundary.signal_id.as_str(),
                ))
                .map(|override_| Arc::clone(&override_.buffer))
                .map(Ok)
                .unwrap_or_else(|| device.create_resident_buffer(byte_capacity).map(Arc::new))?;
            input_buffers.push(VulkanModelBoundaryBufferAllocation {
                boundary: boundary.clone(),
                byte_capacity,
                buffer,
            });
        }

        for boundary in &self.outputs {
            let byte_capacity = boundary.byte_capacity.ok_or_else(|| {
                VulkanError(format!(
                    "{} model output boundary {:?} has unknown byte capacity",
                    self.device_id, boundary.signal_id
                ))
            })?;
            let input_alias = boundary.source_signal_id.as_ref().and_then(|_| {
                input_buffers.iter().find(|input| {
                    input.boundary.component_id == boundary.component_id
                        && input.boundary.signal_id == boundary.signal_id
                        && input.boundary.shape == boundary.shape
                })
            });
            let output_override = overrides.get(&(
                VulkanModelBoundaryDirection::Output,
                boundary.component_id.as_str(),
                boundary.signal_id.as_str(),
            ));
            let buffer = if let Some(override_) = output_override {
                if let Some(input) = input_alias
                    && !Arc::ptr_eq(&input.buffer, &override_.buffer)
                {
                    return Err(VulkanError(format!(
                        "{} boundary output {:?} must preserve its input-storage alias",
                        self.device_id, boundary.port_id,
                    )));
                }
                Arc::clone(&override_.buffer)
            } else if let Some(input) = input_alias {
                if input.byte_capacity != byte_capacity {
                    return Err(VulkanError(format!(
                        "{} boundary output {:?} aliases input storage with {} bytes but requires {byte_capacity}",
                        self.device_id, boundary.port_id, input.byte_capacity
                    )));
                }
                Arc::clone(&input.buffer)
            } else {
                total_byte_capacity = checked_add_bytes(
                    total_byte_capacity,
                    byte_capacity,
                    "model output boundary buffer allocation",
                )?;
                Arc::new(device.create_resident_buffer(byte_capacity)?)
            };
            output_buffers.push(VulkanModelBoundaryBufferAllocation {
                boundary: boundary.clone(),
                byte_capacity,
                buffer,
            });
        }

        Ok(VulkanModelBoundaryBuffers {
            plan: self.clone(),
            input_buffers,
            output_buffers,
            total_byte_capacity,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBuffer {
    pub buffer_index: usize,
    pub signal_id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub element_count: usize,
    pub byte_capacity: Option<usize>,
    pub component_id: String,
    pub port_id: String,
    pub source_signal_id: Option<String>,
}

impl VulkanModelBoundaryBuffer {
    fn from_port(
        buffer_index: usize,
        component_id: &str,
        port: &PlannedPort,
        signal_element_bytes: Option<usize>,
    ) -> Result<Self, VulkanModelBoundaryBufferPlanError> {
        let element_count = product(&port.shape).ok_or_else(|| {
            VulkanModelBoundaryBufferPlanError(format!(
                "{} model boundary port {:?} shape {:?} overflows",
                component_id, port.id, port.shape
            ))
        })?;
        if element_count == 0 {
            return Err(VulkanModelBoundaryBufferPlanError(format!(
                "{} model boundary port {:?} shape {:?} has zero elements",
                component_id, port.id, port.shape
            )));
        }
        let byte_capacity = port
            .element_bytes
            .or(signal_element_bytes)
            .map(|bytes| {
                element_count.checked_mul(bytes).ok_or_else(|| {
                    VulkanModelBoundaryBufferPlanError(format!(
                        "{} model boundary port {:?} byte capacity overflowed",
                        component_id, port.id
                    ))
                })
            })
            .transpose()?;

        Ok(Self {
            buffer_index,
            signal_id: port.source.clone().unwrap_or_else(|| port.id.clone()),
            signal: port.signal.clone(),
            shape: port.shape.clone(),
            element_count,
            byte_capacity,
            component_id: component_id.to_string(),
            port_id: port.id.clone(),
            source_signal_id: port.source.clone(),
        })
    }
}

fn add_optional_boundary_bytes(
    total: Option<usize>,
    byte_capacity: Option<usize>,
) -> Result<Option<usize>, VulkanModelBoundaryBufferPlanError> {
    match (total, byte_capacity) {
        (Some(total), Some(bytes)) => total.checked_add(bytes).map(Some).ok_or_else(|| {
            VulkanModelBoundaryBufferPlanError(
                "model boundary total byte capacity overflowed".to_string(),
            )
        }),
        _ => Ok(None),
    }
}

pub struct VulkanModelBoundaryBuffers {
    pub plan: VulkanModelBoundaryBufferPlan,
    pub input_buffers: Vec<VulkanModelBoundaryBufferAllocation>,
    pub output_buffers: Vec<VulkanModelBoundaryBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanModelBoundaryBuffers {
    pub fn input_buffer(&self, signal_id: &str) -> Option<&VulkanModelBoundaryBufferAllocation> {
        self.input_buffers
            .iter()
            .find(|buffer| buffer.boundary.signal_id == signal_id)
    }

    pub fn output_buffer(&self, signal_id: &str) -> Option<&VulkanModelBoundaryBufferAllocation> {
        self.output_buffers
            .iter()
            .find(|buffer| buffer.boundary.signal_id == signal_id)
    }
}

pub struct VulkanModelBoundaryBufferAllocation {
    pub boundary: VulkanModelBoundaryBuffer,
    pub byte_capacity: usize,
    pub buffer: Arc<VulkanResidentBuffer>,
}

pub struct VulkanModelBoundaryBufferOverride {
    pub direction: VulkanModelBoundaryDirection,
    pub component_id: String,
    pub signal_id: String,
    pub buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanModelBoundaryDirection {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanModelBoundaryBufferPlanError(pub String);

impl Display for VulkanModelBoundaryBufferPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanModelBoundaryBufferPlanError {}
