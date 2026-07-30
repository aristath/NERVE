const VULKAN_TARGETED_COMPONENT_QUANTUM_USEFUL_UNITS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum VulkanTargetedComponentExecutionPhase {
    Decode,
    Prefill { activation_batch_width: usize },
}

impl VulkanTargetedComponentExecutionPhase {
    pub fn activation_batch_width(self) -> usize {
        match self {
            Self::Decode => 1,
            Self::Prefill {
                activation_batch_width,
            } => activation_batch_width,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanTargetedComponentThroughputWindow {
    pub index: usize,
    pub start_unit: usize,
    pub end_unit: usize,
    pub duration_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanTargetedComponentExecutionReport {
    pub component_id: String,
    pub node_id: String,
    pub op: String,
    pub phase: String,
    pub activation_batch_width: usize,
    pub useful_units: usize,
    pub execution_ns: u64,
    pub output_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_values_f32_le_hex: Option<String>,
    pub state_digest: String,
    pub throughput_windows: Vec<VulkanTargetedComponentThroughputWindow>,
    pub resident_parameter_bytes: usize,
    pub resident_transient_bytes: usize,
    pub physical_dispatch_count: usize,
    pub queue_submission_count: usize,
    pub synchronization_wait_count: usize,
    pub synchronization_wait_ns: u64,
    pub queue_wait_ns: u64,
}

pub struct VulkanResidentTargetedComponentSession {
    component_id: String,
    node_id: String,
    op: String,
    phase: VulkanTargetedComponentExecutionPhase,
    resident_parameter_bytes: usize,
    mounted: VulkanMountedPlacedStreamCircuit,
    source_dispatch: VulkanMountedPlacedBoundDispatch,
    execution: VulkanTargetedComponentExecution,
}

pub enum VulkanResidentTargetedExecutionSession {
    Component(VulkanResidentTargetedComponentSession),
    OutputTransducer(VulkanResidentTargetedOutputTransducerSession),
}

pub struct VulkanResidentTargetedOutputTransducerSession {
    component_id: String,
    node_id: String,
    resident_parameter_bytes: usize,
    mounted: VulkanMountedPlacedStreamCircuit,
    runner: VulkanResidentOutputTransducerRunner,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
    capture_output_values: bool,
}

enum VulkanTargetedComponentExecution {
    Decode(Box<VulkanTargetedDecodeExecution>),
    Prefill(VulkanTargetedPrefillExecution),
}

struct VulkanTargetedDecodeExecution {
    dispatch: VulkanResidentKernelDispatch,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
    push_constants: Vec<u8>,
}

struct VulkanTargetedPrefillExecution {
    activation_batch_width: usize,
    signal_buffers: Vec<VulkanComponentBatchSignalBuffer>,
    signal_buffer_indices: BTreeMap<VulkanComponentBatchSignalKey, usize>,
    control_buffers:
        BTreeMap<VulkanResidentComponentBatchControlPayload, VulkanResidentBuffer>,
    steps: Vec<VulkanTargetedPrefillStep>,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
}

struct VulkanTargetedPrefillStep {
    dispatch: VulkanResidentKernelDispatch,
    indirect_control: Option<(VulkanResidentComponentBatchControlPayload, usize)>,
}

struct VulkanTargetedComponentRunCounters {
    execution_ns: u64,
    windows: Vec<VulkanTargetedComponentThroughputWindow>,
    physical_dispatch_count: usize,
    queue_submission_count: usize,
    synchronization_wait_ns: u64,
    queue_wait_ns: u64,
}

impl VulkanResidentTargetedExecutionSession {
    pub fn from_device_slice(
        device: &VulkanComputeDevice,
        slice: VulkanResidentModelPackageDeviceSlice,
        component_id: impl AsRef<str>,
        node_id: impl AsRef<str>,
        phase: VulkanTargetedComponentExecutionPhase,
        capture_output_values: bool,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let component_id = component_id.as_ref();
        if slice
            .targeted_output
            .as_ref()
            .is_some_and(|output| output.spec.transducer_id == component_id)
        {
            return VulkanResidentTargetedOutputTransducerSession::from_device_slice(
                device,
                slice,
                component_id,
                node_id,
                phase,
                capture_output_values,
            )
            .map(Self::OutputTransducer);
        }
        VulkanResidentTargetedComponentSession::from_device_slice(
            device,
            slice,
            component_id,
            node_id,
            phase,
        )
        .map(Self::Component)
    }

    pub fn execute(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        seed: u32,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentExecutionReport, VulkanResidentTokenModelPackageError> {
        match self {
            Self::Component(session) => {
                session.execute(device, useful_units, seed, maximum_quantum_wait)
            }
            Self::OutputTransducer(session) => {
                session.execute(device, useful_units, seed, maximum_quantum_wait)
            }
        }
    }

    pub fn resident_parameter_bytes(&self) -> usize {
        match self {
            Self::Component(session) => session.resident_parameter_bytes(),
            Self::OutputTransducer(session) => session.resident_parameter_bytes(),
        }
    }

    pub fn resident_transient_bytes(&self) -> usize {
        match self {
            Self::Component(session) => session.resident_transient_bytes(),
            Self::OutputTransducer(session) => session.resident_transient_bytes(),
        }
    }
}

impl VulkanResidentTargetedOutputTransducerSession {
    fn from_device_slice(
        device: &VulkanComputeDevice,
        slice: VulkanResidentModelPackageDeviceSlice,
        component_id: &str,
        node_id: impl AsRef<str>,
        phase: VulkanTargetedComponentExecutionPhase,
        capture_output_values: bool,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        if phase != VulkanTargetedComponentExecutionPhase::Decode {
            return targeted_component_error(
                "targeted output-transducer execution currently supports decode only",
            );
        }
        let output = slice.targeted_output.as_ref().ok_or_else(|| {
            targeted_component_error_value(
                "resident device slice has no targeted output-transducer resources",
            )
        })?;
        let node_id = node_id.as_ref();
        if !output.spec.node_ids.iter().any(|candidate| candidate == node_id) {
            return targeted_component_error(format!(
                "targeted output transducer {component_id:?} has no node {node_id:?}"
            ));
        }
        let resident_parameter_bytes = output.parameter_buffers.total_byte_capacity;
        let mounted = slice.create_mounted_stream_circuit(device)?;
        let runner = VulkanResidentOutputTransducerRunner::from_mounted_output_transducer(
            device,
            &mounted,
            &output.parameter_buffers,
            &output.embedding_norm_spirv_words,
            &output.projection_spirv_words,
            &output.spec,
        )
        .map_err(|error| {
            targeted_component_error_value(format!(
                "failed to mount targeted output transducer: {error}"
            ))
        })?;
        Ok(Self {
            component_id: component_id.to_string(),
            node_id: node_id.to_string(),
            resident_parameter_bytes,
            mounted,
            runner,
            sequence_catalog: RefCell::new(BTreeMap::new()),
            capture_output_values,
        })
    }

    fn execute(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        seed: u32,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentExecutionReport, VulkanResidentTokenModelPackageError> {
        if useful_units == 0 {
            return targeted_component_error(
                "targeted output-transducer useful work must be at least one unit",
            );
        }
        if maximum_quantum_wait.is_zero() {
            return targeted_component_error(
                "targeted output-transducer quantum wait must be positive",
            );
        }
        self.write_fixture(seed)?;
        let counters =
            self.execute_quanta(device, useful_units, maximum_quantum_wait)?;
        let output_values = self
            .runner
            .read_logits_bytes(self.runner.logits_byte_capacity)
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to read targeted output logits: {error}"
                ))
            })?;
        let output_digest = targeted_finalized_artifact_digest(
            Sha256::digest(&output_values).as_slice(),
        );
        Ok(VulkanTargetedComponentExecutionReport {
            component_id: self.component_id.clone(),
            node_id: self.node_id.clone(),
            op: "output_transducer".to_string(),
            phase: "decode".to_string(),
            activation_batch_width: 1,
            useful_units,
            execution_ns: counters.execution_ns,
            output_digest,
            output_values_f32_le_hex: self
                .capture_output_values
                .then(|| hex_bytes(&output_values)),
            state_digest: targeted_finalized_artifact_digest(
                Sha256::digest([]).as_slice(),
            ),
            throughput_windows: counters.windows,
            resident_parameter_bytes: self.resident_parameter_bytes,
            resident_transient_bytes: self.resident_transient_bytes(),
            physical_dispatch_count: counters.physical_dispatch_count,
            queue_submission_count: counters.queue_submission_count,
            synchronization_wait_count: counters.queue_submission_count,
            synchronization_wait_ns: counters.synchronization_wait_ns,
            queue_wait_ns: counters.queue_wait_ns,
        })
    }

    fn write_fixture(
        &self,
        seed: u32,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        let input = self
            .mounted
            .boundary_io
            .output_buffer(&self.runner.input_signal_id)
            .ok_or_else(|| {
                targeted_component_error_value(
                    "targeted output transducer has no mounted input frame",
                )
            })?;
        input
            .buffer
            .write_bytes(&targeted_fixture_bytes(
                input.byte_capacity,
                seed,
                0,
            ))
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to write targeted output-transducer input: {error}"
                ))
            })
    }

    fn execute_quanta(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentRunCounters, VulkanResidentTokenModelPackageError> {
        let quanta = targeted_execution_quanta(useful_units, 1)?;
        let mut counters = VulkanTargetedComponentRunCounters {
            execution_ns: 0,
            windows: Vec::with_capacity(quanta.len()),
            physical_dispatch_count: 0,
            queue_submission_count: 0,
            synchronization_wait_ns: 0,
            queue_wait_ns: 0,
        };
        let mut start_unit = 0usize;
        for (index, repetitions) in quanta.into_iter().enumerate() {
            self.ensure_sequence(device, repetitions)?;
            let catalog = self.sequence_catalog.borrow();
            let sequence = catalog
                .get(&repetitions)
                .expect("targeted output sequence was inserted");
            let wait_started = Instant::now();
            let duration_ns = device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    sequence,
                    maximum_quantum_wait,
                )
                .map_err(|error| {
                    targeted_component_error_value(format!(
                        "targeted output-transducer quantum failed: {error}"
                    ))
                })?;
            let wait_ns = elapsed_nanoseconds(wait_started);
            counters.execution_ns =
                counters.execution_ns.saturating_add(duration_ns);
            counters.synchronization_wait_ns = counters
                .synchronization_wait_ns
                .saturating_add(wait_ns);
            counters.queue_wait_ns = counters
                .queue_wait_ns
                .saturating_add(wait_ns.saturating_sub(duration_ns));
            let end_unit = start_unit + repetitions;
            counters
                .windows
                .push(VulkanTargetedComponentThroughputWindow {
                    index,
                    start_unit,
                    end_unit,
                    duration_ns,
                });
            start_unit = end_unit;
        }
        counters.execution_ns = counters.execution_ns.max(1);
        counters.physical_dispatch_count = useful_units
            .checked_mul(self.runner.dispatch_count)
            .ok_or_else(|| {
                targeted_component_error_value(
                    "targeted output-transducer dispatch count overflowed",
                )
            })?;
        counters.queue_submission_count = counters.windows.len();
        Ok(counters)
    }

    fn ensure_sequence(
        &self,
        device: &VulkanComputeDevice,
        repetitions: usize,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        if self.sequence_catalog.borrow().contains_key(&repetitions) {
            return Ok(());
        }
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to create targeted output-transducer sequence: {error}"
                ))
            })?;
        let mut steps =
            Vec::with_capacity(repetitions * self.runner.dispatch_count);
        for _ in 0..repetitions {
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.runner.embedding_norm_dispatch,
                &[],
            ));
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.runner.tied_projection_dispatch,
                &[],
            ));
        }
        device
            .record_resident_kernel_sequence(&sequence, &steps)
            .map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to record targeted output-transducer sequence: {error}"
                ))
            })?;
        self.sequence_catalog
            .borrow_mut()
            .insert(repetitions, sequence);
        Ok(())
    }

    fn resident_parameter_bytes(&self) -> usize {
        self.resident_parameter_bytes
    }

    fn resident_transient_bytes(&self) -> usize {
        self.mounted
            .buffers
            .total_byte_capacity
            .saturating_add(self.mounted.boundary_io.total_byte_capacity)
            .saturating_add(self.mounted.edge_io.total_byte_capacity)
            .saturating_add(
                self.mounted.stream_control_buffer.byte_capacity(),
            )
            .saturating_add(
                self.runner.normalized_frame_buffer.byte_capacity(),
            )
            .saturating_add(self.runner.logits_buffer.byte_capacity())
    }
}

impl VulkanResidentTargetedComponentSession {
    pub fn from_device_slice(
        device: &VulkanComputeDevice,
        slice: VulkanResidentModelPackageDeviceSlice,
        component_id: impl AsRef<str>,
        node_id: impl AsRef<str>,
        phase: VulkanTargetedComponentExecutionPhase,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let component_id = component_id.as_ref();
        let node_id = node_id.as_ref();
        let activation_batch_width = phase.activation_batch_width();
        if activation_batch_width == 0 {
            return targeted_component_error(
                "targeted component activation batch width must be at least one",
            );
        }
        let dynamic_state_capacity_activations =
            u32::try_from(slice.dynamic_state_capacity_activations).map_err(|_| {
                targeted_component_error_value(
                    "targeted component dynamic-state capacity exceeds u32",
                )
            })?;
        let resident_parameter_bytes = slice.permanent_parameter_bytes;
        let mounted = slice.create_mounted_stream_circuit(device)?;
        mounted
            .buffers
            .zero_state_buffers()
            .map_err(|error| targeted_component_error_value(format!(
                "failed to reset targeted component state: {error}"
            )))?;
        mounted
            .buffers
            .apply_clone_state_policies()
            .map_err(|error| targeted_component_error_value(format!(
                "failed to initialize targeted component cloned state: {error}"
            )))?;
        let reusable_manifest =
            resident_package_reusable_kernel_manifest(&mounted.placed_plan);
        let mounted_bound = mounted
            .mounted_placed_bound_dispatch_plan(&reusable_manifest)
            .map_err(|error| targeted_component_error_value(format!(
                "failed to bind targeted component dispatches: {error}"
            )))?;
        let source_dispatch = mounted_bound
            .dispatch(component_id, node_id)
            .cloned()
            .ok_or_else(|| {
                targeted_component_error_value(format!(
                    "resident device slice has no dispatch {component_id}.{node_id}"
                ))
            })?;
        let input_count = source_dispatch
            .descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.usage == VulkanKernelDescriptorUsage::InputSignal
            })
            .count();
        let output_count = source_dispatch
            .descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
            })
            .count();
        if input_count == 0 || output_count == 0 {
            return targeted_component_error(format!(
                "targeted dispatch {component_id}.{node_id} requires at least one input and output signal; found {input_count} inputs and {output_count} outputs"
            ));
        }
        let execution = match phase {
            VulkanTargetedComponentExecutionPhase::Decode => {
                VulkanTargetedComponentExecution::Decode(
                    Box::new(VulkanTargetedDecodeExecution::new(
                        device,
                        &mounted,
                        &source_dispatch,
                        slice.loaded_manifest(),
                        dynamic_state_capacity_activations,
                    )?),
                )
            }
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            } => VulkanTargetedComponentExecution::Prefill(
                VulkanTargetedPrefillExecution::new(
                    device,
                    &mounted,
                    &source_dispatch,
                    &slice.batch_kernels,
                    activation_batch_width,
                    dynamic_state_capacity_activations,
                )?,
            ),
        };
        Ok(Self {
            component_id: component_id.to_string(),
            node_id: node_id.to_string(),
            op: source_dispatch.op.clone(),
            phase,
            resident_parameter_bytes,
            mounted,
            source_dispatch,
            execution,
        })
    }

    pub fn execute(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        seed: u32,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentExecutionReport, VulkanResidentTokenModelPackageError> {
        if useful_units == 0 {
            return targeted_component_error(
                "targeted component useful work must be at least one unit",
            );
        }
        if maximum_quantum_wait.is_zero() {
            return targeted_component_error(
                "targeted component quantum wait must be positive",
            );
        }
        self.reset_state_and_write_fixture(seed)?;
        let counters = match &self.execution {
                VulkanTargetedComponentExecution::Decode(execution) => execution.execute(
                    device,
                    useful_units,
                    maximum_quantum_wait,
                )?,
                VulkanTargetedComponentExecution::Prefill(execution) => execution.execute(
                    device,
                    useful_units,
                    maximum_quantum_wait,
                )?,
            };
        let output_digest = self.output_digest()?;
        let state_digest = self.state_digest()?;
        Ok(VulkanTargetedComponentExecutionReport {
            component_id: self.component_id.clone(),
            node_id: self.node_id.clone(),
            op: self.op.clone(),
            phase: match self.phase {
                VulkanTargetedComponentExecutionPhase::Decode => "decode",
                VulkanTargetedComponentExecutionPhase::Prefill { .. } => "prefill",
            }
            .to_string(),
            activation_batch_width: self.phase.activation_batch_width(),
            useful_units,
            execution_ns: counters.execution_ns,
            output_digest,
            output_values_f32_le_hex: None,
            state_digest,
            throughput_windows: counters.windows,
            resident_parameter_bytes: self.resident_parameter_bytes,
            resident_transient_bytes: self.resident_transient_bytes(),
            physical_dispatch_count: counters.physical_dispatch_count,
            queue_submission_count: counters.queue_submission_count,
            synchronization_wait_count: counters.queue_submission_count,
            synchronization_wait_ns: counters.synchronization_wait_ns,
            queue_wait_ns: counters.queue_wait_ns,
        })
    }

    pub fn resident_parameter_bytes(&self) -> usize {
        self.resident_parameter_bytes
    }

    pub fn resident_transient_bytes(&self) -> usize {
        let mounted_bytes = self
            .mounted
            .buffers
            .total_byte_capacity
            .saturating_add(self.mounted.boundary_io.total_byte_capacity)
            .saturating_add(self.mounted.edge_io.total_byte_capacity)
            .saturating_add(self.mounted.stream_control_buffer.byte_capacity());
        match &self.execution {
            VulkanTargetedComponentExecution::Decode(_) => mounted_bytes,
            VulkanTargetedComponentExecution::Prefill(execution) => mounted_bytes
                .saturating_add(
                    execution
                .signal_buffers
                .iter()
                .map(|buffer| buffer.buffer.byte_capacity())
                .sum::<usize>()
                )
                .saturating_add(
                    execution
                        .control_buffers
                        .values()
                        .map(VulkanResidentBuffer::byte_capacity)
                        .sum::<usize>(),
                ),
        }
    }

    fn reset_state_and_write_fixture(
        &self,
        seed: u32,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        self.mounted
            .buffers
            .zero_state_buffers()
            .map_err(|error| targeted_component_error_value(format!(
                "failed to reset targeted component state: {error}"
            )))?;
        match &self.execution {
            VulkanTargetedComponentExecution::Decode(_) => {
                self.write_decode_fixture(seed)
            }
            VulkanTargetedComponentExecution::Prefill(execution) => {
                execution.write_fixture(&self.mounted, &self.source_dispatch, seed)
            }
        }
    }

    fn write_decode_fixture(
        &self,
        seed: u32,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        for descriptor in &self.source_dispatch.descriptors {
            let binding = self
                .mounted
                .resident_kernel_buffer_binding(&self.source_dispatch, descriptor)
                .map_err(|error| targeted_component_error_value(format!(
                    "failed to resolve targeted descriptor {}: {error}",
                    descriptor.name
                )))?;
            match descriptor.usage {
                VulkanKernelDescriptorUsage::InputSignal => {
                    let bytes = targeted_fixture_bytes(
                        binding.byte_len,
                        seed,
                        descriptor.binding,
                    );
                    binding
                        .buffer
                        .write_bytes_at(binding.byte_offset, &bytes)
                        .map_err(|error| targeted_component_error_value(format!(
                            "failed to write targeted input {}: {error}",
                            descriptor.name
                        )))?;
                }
                VulkanKernelDescriptorUsage::OutputSignal => {
                    binding
                        .buffer
                        .write_bytes_at(
                            binding.byte_offset,
                            &vec![0; binding.byte_len],
                        )
                        .map_err(|error| targeted_component_error_value(format!(
                            "failed to clear targeted output {}: {error}",
                            descriptor.name
                        )))?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn output_digest(&self) -> Result<String, VulkanResidentTokenModelPackageError> {
        let mut digest = Sha256::new();
        match &self.execution {
            VulkanTargetedComponentExecution::Decode(_) => {
                for descriptor in self
                    .source_dispatch
                    .descriptors
                    .iter()
                    .filter(|descriptor| {
                        descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
                    })
                {
                    let binding = self
                        .mounted
                        .resident_kernel_buffer_binding(&self.source_dispatch, descriptor)
                        .map_err(|error| targeted_component_error_value(format!(
                            "failed to resolve targeted output {}: {error}",
                            descriptor.name
                        )))?;
                    digest.update(descriptor.binding.to_le_bytes());
                    digest.update(descriptor.name.as_bytes());
                    digest.update(
                        binding
                            .buffer
                            .read_bytes_at(binding.byte_offset, binding.byte_len)
                            .map_err(|error| targeted_component_error_value(format!(
                                "failed to read targeted output {}: {error}",
                                descriptor.name
                            )))?,
                    );
                }
            }
            VulkanTargetedComponentExecution::Prefill(execution) => {
                execution.update_output_digest(
                    &self.mounted,
                    &self.source_dispatch,
                    &mut digest,
                )?;
            }
        }
        Ok(targeted_finalized_artifact_digest(digest.finalize().as_slice()))
    }

    fn state_digest(&self) -> Result<String, VulkanResidentTokenModelPackageError> {
        let mut digest = Sha256::new();
        for state in &self.mounted.buffers.state_buffers {
            digest.update(state.component_id.as_bytes());
            digest.update(state.state_id.as_bytes());
            digest.update(
                state
                    .buffer
                    .read_bytes(state.buffer.byte_capacity())
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to read targeted state {}.{}: {error}",
                        state.component_id, state.state_id
                    )))?,
            );
        }
        Ok(targeted_finalized_artifact_digest(digest.finalize().as_slice()))
    }
}

impl VulkanTargetedDecodeExecution {
    fn new(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        loaded_manifest: &VulkanLoadedReusableKernelArtifactManifest,
        dynamic_state_capacity_activations: u32,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let resident = mounted
            .create_resident_kernel_dispatch_for_bound_dispatch(
                device,
                dispatch,
                loaded_manifest,
            )
            .map_err(|error| targeted_component_error_value(format!(
                "failed to create targeted decode dispatch: {error}"
            )))?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick: 0,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };
        mounted
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(|error| targeted_component_error_value(format!(
                "failed to initialize targeted stream control: {error}"
            )))?;
        let push_constants =
            stream_control_push_constant_bytes(&dispatch.push_constants, control)
                .map_err(|error| targeted_component_error_value(format!(
                    "failed to bind targeted decode stream control: {error}"
                )))?;
        Ok(Self {
            dispatch: resident,
            sequence_catalog: RefCell::new(BTreeMap::new()),
            push_constants,
        })
    }

    fn execute(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentRunCounters, VulkanResidentTokenModelPackageError> {
        let quanta = targeted_execution_quanta(useful_units, 1)?;
        let mut windows = Vec::with_capacity(quanta.len());
        let mut execution_ns = 0u64;
        let mut synchronization_wait_ns = 0u64;
        let mut queue_wait_ns = 0u64;
        let mut start_unit = 0usize;
        for (index, repetitions) in quanta.into_iter().enumerate() {
            self.ensure_sequence(device, repetitions)?;
            let catalog = self.sequence_catalog.borrow();
            let sequence = catalog
                .get(&repetitions)
                .expect("targeted decode sequence was inserted");
            let wait_started = Instant::now();
            let duration_ns = device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    sequence,
                    maximum_quantum_wait,
                )
                .map_err(|error| targeted_component_error_value(format!(
                    "targeted decode quantum failed: {error}"
                )))?;
            let wait_ns = elapsed_nanoseconds(wait_started);
            synchronization_wait_ns =
                synchronization_wait_ns.saturating_add(wait_ns);
            queue_wait_ns = queue_wait_ns
                .saturating_add(wait_ns.saturating_sub(duration_ns));
            let end_unit = start_unit + repetitions;
            execution_ns = execution_ns.saturating_add(duration_ns);
            windows.push(VulkanTargetedComponentThroughputWindow {
                index,
                start_unit,
                end_unit,
                duration_ns,
            });
            start_unit = end_unit;
        }
        let submission_count = windows.len();
        Ok(VulkanTargetedComponentRunCounters {
            execution_ns: execution_ns.max(1),
            windows,
            physical_dispatch_count: useful_units,
            queue_submission_count: submission_count,
            synchronization_wait_ns,
            queue_wait_ns,
        })
    }

    fn ensure_sequence(
        &self,
        device: &VulkanComputeDevice,
        repetitions: usize,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        if self.sequence_catalog.borrow().contains_key(&repetitions) {
            return Ok(());
        }
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .map_err(|error| targeted_component_error_value(format!(
                "failed to create targeted decode sequence: {error}"
            )))?;
        let steps = (0..repetitions)
            .map(|_| {
                VulkanResidentKernelSequenceStep::new(
                    &self.dispatch,
                    &self.push_constants,
                )
            })
            .collect::<Vec<_>>();
        device
            .record_resident_kernel_sequence(&sequence, &steps)
            .map_err(|error| targeted_component_error_value(format!(
                "failed to record targeted decode sequence: {error}"
            )))?;
        self.sequence_catalog
            .borrow_mut()
            .insert(repetitions, sequence);
        Ok(())
    }
}

impl VulkanTargetedPrefillExecution {
    fn new(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        batch_kernels: &[VulkanResidentComponentBatchKernelArtifact],
        activation_batch_width: usize,
        dynamic_state_capacity_activations: u32,
    ) -> Result<Self, VulkanResidentTokenModelPackageError> {
        let (signal_buffer_indices, signal_buffer_plan) =
            component_batch_signal_buffer_plan(
                mounted,
                std::slice::from_ref(dispatch),
            )
            .map_err(|error| targeted_component_error_value(format!(
                "failed to plan targeted prefill signals: {error}"
            )))?;
        let signal_buffers = signal_buffer_plan
            .into_iter()
            .map(|allocation| {
                let byte_capacity = allocation
                    .frame_byte_capacity
                    .checked_mul(activation_batch_width)
                    .ok_or_else(|| targeted_component_error_value(
                        "targeted prefill signal capacity overflowed",
                    ))?;
                let buffer = device
                    .create_resident_buffer(byte_capacity)
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to allocate targeted prefill signal: {error}"
                    )))?;
                Ok(VulkanComponentBatchSignalBuffer {
                    frame_byte_capacity: allocation.frame_byte_capacity,
                    buffer: Arc::new(buffer),
                    shared_device_buffers: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;
        let artifact = select_component_batch_kernel_artifact(
            batch_kernels,
            &dispatch.component_id,
            &dispatch.node_id,
            VulkanComponentBatchExecutionMode::CausalSequence,
            activation_batch_width,
        )
        .filter(|artifact| {
            component_batch_stages_replace_push_constants(
                &artifact.stages,
                &dispatch.push_constants,
            )
        })
        .filter(|artifact| {
            targeted_prefill_batch_mode_is_supported(artifact.batch_mode)
                && !dispatch.descriptors.iter().any(|descriptor| {
                    matches!(
                        descriptor.usage,
                        VulkanKernelDescriptorUsage::StateRead
                            | VulkanKernelDescriptorUsage::StateWrite
                            | VulkanKernelDescriptorUsage::StateView
                    )
                })
        })
        .ok_or_else(|| targeted_component_error_value(format!(
            "dispatch {}.{} has no ordinary stateless prefill implementation for width {activation_batch_width}",
            dispatch.component_id, dispatch.node_id
        )))?;
        if artifact.stages.is_empty() {
            return targeted_component_error(
                "targeted prefill implementation has no executable stages",
            );
        }
        if artifact.stages.iter().any(|stage| stage.state_snapshot_binding.is_some()) {
            return targeted_component_error(
                "targeted stateless prefill cannot mount a state-snapshot stage",
            );
        }
        let control_payloads = artifact
            .stages
            .iter()
            .map(|stage| stage.control.storage_buffer().2)
            .collect::<BTreeSet<_>>();
        let control_buffers = control_payloads
            .into_iter()
            .map(|payload| {
                let mut buffer = device
                    .create_host_visible_resident_buffer(payload.byte_count() as usize)
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to allocate targeted prefill control: {error}"
                    )))?;
                buffer.persistently_map().map_err(|error| {
                    targeted_component_error_value(format!(
                        "failed to map targeted prefill control: {error}"
                    ))
                })?;
                Ok((payload, buffer))
            })
            .collect::<Result<
                BTreeMap<_, _>,
                VulkanResidentTokenModelPackageError,
            >>()?;
        let control = component_batch_control_bytes(
            u32::try_from(activation_batch_width).map_err(|_| {
                targeted_component_error_value(
                    "targeted prefill activation width exceeds u32",
                )
            })?,
            0,
            dynamic_state_capacity_activations,
        );
        for (payload, buffer) in &control_buffers {
            buffer
                .write_bytes(&component_batch_control_payload_bytes(
                    *payload,
                    &control,
                    false,
                ))
                .map_err(|error| targeted_component_error_value(format!(
                    "failed to initialize targeted prefill control: {error}"
                )))?;
        }
        let workgroup_count_y = u32::try_from(
            activation_batch_width
                .checked_add(artifact.lane_tile_width - 1)
                .ok_or_else(|| targeted_component_error_value(
                    "targeted prefill workgroup count overflowed",
                ))?
                / artifact.lane_tile_width,
        )
        .map_err(|_| targeted_component_error_value(
            "targeted prefill workgroup count exceeds u32",
        ))?;
        let parent_bindings = component_batch_bindings(
            mounted,
            dispatch,
            &signal_buffers,
            &signal_buffer_indices,
            None,
            None,
        )
        .map_err(|error| targeted_component_error_value(format!(
            "failed to bind targeted prefill signals: {error}"
        )))?;
        let mut steps = Vec::with_capacity(artifact.stages.len());
        for stage in &artifact.stages {
            let (control_binding, control_bytes, payload) =
                stage.control.storage_buffer();
            let mut bindings = component_batch_stage_bindings(
                &parent_bindings,
                &stage.descriptor_bindings,
                control_binding,
            )
            .map_err(|error| targeted_component_error_value(format!(
                "failed to remap targeted prefill bindings: {error}"
            )))?;
            let control_buffer = control_buffers.get(&payload).ok_or_else(|| {
                targeted_component_error_value(
                    "targeted prefill stage has no control buffer",
                )
            })?;
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    control_binding,
                    control_buffer,
                    control_bytes as usize,
                )
                .with_access(component_batch_control_buffer_access(stage.control)),
            );
            let resident = device
                .create_resident_kernel_dispatch_2d_labeled(
                    &stage.spirv_words,
                    &bindings,
                    stage.workgroup_count_x,
                    workgroup_count_y,
                    stage.local_size_x,
                    0,
                    Some(vulkan_dispatch_semantic_label(
                        dispatch,
                        Some("targeted_prefill"),
                    )),
                )
                .map_err(|error| targeted_component_error_value(format!(
                    "failed to create targeted prefill dispatch: {error}"
                )))?;
            steps.push(VulkanTargetedPrefillStep {
                dispatch: resident,
                indirect_control: stage
                    .indirect_dispatch_byte_offset
                    .map(|offset| (payload, offset as usize)),
            });
        }
        Ok(Self {
            activation_batch_width,
            signal_buffers,
            signal_buffer_indices,
            control_buffers,
            steps,
            sequence_catalog: RefCell::new(BTreeMap::new()),
        })
    }

    fn execute(
        &self,
        device: &VulkanComputeDevice,
        useful_units: usize,
        maximum_quantum_wait: Duration,
    ) -> Result<VulkanTargetedComponentRunCounters, VulkanResidentTokenModelPackageError> {
        let quanta = targeted_execution_quanta(
            useful_units,
            self.activation_batch_width,
        )?;
        let mut windows = Vec::with_capacity(quanta.len());
        let mut execution_ns = 0u64;
        let mut synchronization_wait_ns = 0u64;
        let mut queue_wait_ns = 0u64;
        let mut start_unit = 0usize;
        for (index, repetitions) in quanta.into_iter().enumerate() {
            self.ensure_sequence(device, repetitions)?;
            let catalog = self.sequence_catalog.borrow();
            let sequence = catalog
                .get(&repetitions)
                .expect("targeted prefill sequence was inserted");
            let wait_started = Instant::now();
            let duration_ns = device
                .run_timestamped_recorded_resident_kernel_sequence_for(
                    sequence,
                    maximum_quantum_wait,
                )
                .map_err(|error| targeted_component_error_value(format!(
                    "targeted prefill quantum failed: {error}"
                )))?;
            let wait_ns = elapsed_nanoseconds(wait_started);
            synchronization_wait_ns =
                synchronization_wait_ns.saturating_add(wait_ns);
            queue_wait_ns = queue_wait_ns
                .saturating_add(wait_ns.saturating_sub(duration_ns));
            let quantum_units = repetitions
                .checked_mul(self.activation_batch_width)
                .ok_or_else(|| targeted_component_error_value(
                    "targeted prefill useful-work count overflowed",
                ))?;
            let end_unit = start_unit + quantum_units;
            execution_ns = execution_ns.saturating_add(duration_ns);
            windows.push(VulkanTargetedComponentThroughputWindow {
                index,
                start_unit,
                end_unit,
                duration_ns,
            });
            start_unit = end_unit;
        }
        let physical_dispatch_count = useful_units
            .checked_div(self.activation_batch_width)
            .and_then(|repetitions| repetitions.checked_mul(self.steps.len()))
            .ok_or_else(|| targeted_component_error_value(
                "targeted prefill physical dispatch count overflowed",
            ))?;
        let submission_count = windows.len();
        Ok(VulkanTargetedComponentRunCounters {
            execution_ns: execution_ns.max(1),
            windows,
            physical_dispatch_count,
            queue_submission_count: submission_count,
            synchronization_wait_ns,
            queue_wait_ns,
        })
    }

    fn ensure_sequence(
        &self,
        device: &VulkanComputeDevice,
        repetitions: usize,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        if self.sequence_catalog.borrow().contains_key(&repetitions) {
            return Ok(());
        }
        let sequence = device
            .create_timestamped_resident_kernel_sequence()
            .map_err(|error| targeted_component_error_value(format!(
                "failed to create targeted prefill sequence: {error}"
            )))?;
        let mut sequence_steps = Vec::with_capacity(repetitions * self.steps.len());
        for _ in 0..repetitions {
            for step in &self.steps {
                let sequence_step =
                    if let Some((payload, byte_offset)) = step.indirect_control {
                        VulkanResidentKernelSequenceStep::new_indirect(
                            &step.dispatch,
                            &[],
                            self.control_buffers.get(&payload).ok_or_else(|| {
                                targeted_component_error_value(
                                    "targeted prefill indirect control is absent",
                                )
                            })?,
                            byte_offset,
                        )
                        .map_err(|error| targeted_component_error_value(format!(
                            "failed to bind targeted prefill indirect dispatch: {error}"
                        )))?
                    } else {
                        VulkanResidentKernelSequenceStep::new(
                            &step.dispatch,
                            &[],
                        )
                    };
                sequence_steps.push(sequence_step);
            }
        }
        device
            .record_resident_kernel_sequence(&sequence, &sequence_steps)
            .map_err(|error| targeted_component_error_value(format!(
                "failed to record targeted prefill sequence: {error}"
            )))?;
        self.sequence_catalog
            .borrow_mut()
            .insert(repetitions, sequence);
        Ok(())
    }

    fn write_fixture(
        &self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        seed: u32,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        for descriptor in &dispatch.descriptors {
            if !matches!(
                descriptor.usage,
                VulkanKernelDescriptorUsage::InputSignal
                    | VulkanKernelDescriptorUsage::OutputSignal
            ) {
                continue;
            }
            let (key, frame_byte_capacity) =
                component_batch_signal_target_with_mounted(mounted, descriptor)
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to resolve targeted prefill signal: {error}"
                    )))?
                    .ok_or_else(|| targeted_component_error_value(format!(
                        "targeted prefill descriptor {} has no signal buffer",
                        descriptor.name
                    )))?;
            let buffer_index = *self.signal_buffer_indices.get(&key).ok_or_else(|| {
                targeted_component_error_value(format!(
                    "targeted prefill signal {key:?} was not allocated",
                ))
            })?;
            let buffer = &self.signal_buffers[buffer_index].buffer;
            let byte_count = frame_byte_capacity
                .checked_mul(self.activation_batch_width)
                .ok_or_else(|| targeted_component_error_value(
                    "targeted prefill fixture size overflowed",
                ))?;
            let bytes = match descriptor.usage {
                VulkanKernelDescriptorUsage::InputSignal => {
                    targeted_fixture_bytes(byte_count, seed, descriptor.binding)
                }
                VulkanKernelDescriptorUsage::OutputSignal => vec![0; byte_count],
                _ => unreachable!(),
            };
            buffer.write_bytes(&bytes).map_err(|error| {
                targeted_component_error_value(format!(
                    "failed to initialize targeted prefill signal {}: {error}",
                    descriptor.name
                ))
            })?;
        }
        Ok(())
    }

    fn update_output_digest(
        &self,
        mounted: &VulkanMountedPlacedStreamCircuit,
        dispatch: &VulkanMountedPlacedBoundDispatch,
        digest: &mut Sha256,
    ) -> Result<(), VulkanResidentTokenModelPackageError> {
        for descriptor in dispatch.descriptors.iter().filter(|descriptor| {
            descriptor.usage == VulkanKernelDescriptorUsage::OutputSignal
        }) {
            let (key, frame_byte_capacity) =
                component_batch_signal_target_with_mounted(mounted, descriptor)
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to resolve targeted prefill output: {error}"
                    )))?
                    .ok_or_else(|| targeted_component_error_value(format!(
                        "targeted prefill output {} has no signal buffer",
                        descriptor.name
                    )))?;
            let buffer_index = *self.signal_buffer_indices.get(&key).ok_or_else(|| {
                targeted_component_error_value(format!(
                    "targeted prefill output {key:?} was not allocated",
                ))
            })?;
            let byte_count = frame_byte_capacity
                .checked_mul(self.activation_batch_width)
                .ok_or_else(|| targeted_component_error_value(
                    "targeted prefill output size overflowed",
                ))?;
            digest.update(descriptor.binding.to_le_bytes());
            digest.update(descriptor.name.as_bytes());
            digest.update(
                self.signal_buffers[buffer_index]
                    .buffer
                    .read_bytes(byte_count)
                    .map_err(|error| targeted_component_error_value(format!(
                        "failed to read targeted prefill output {}: {error}",
                        descriptor.name
                    )))?,
            );
        }
        Ok(())
    }
}

fn targeted_prefill_batch_mode_is_supported(
    batch_mode: VulkanResidentComponentKernelBatchMode,
) -> bool {
    matches!(
        batch_mode,
        VulkanResidentComponentKernelBatchMode::WeightShared
            | VulkanResidentComponentKernelBatchMode::CausalScan
    )
}

fn targeted_execution_quanta(
    useful_units: usize,
    activation_batch_width: usize,
) -> Result<Vec<usize>, VulkanResidentTokenModelPackageError> {
    if activation_batch_width == 0 {
        return targeted_component_error(
            "targeted execution activation width must be positive",
        );
    }
    if !useful_units.is_multiple_of(activation_batch_width) {
        return targeted_component_error(format!(
            "targeted useful work {useful_units} is not divisible by activation width {activation_batch_width}"
        ));
    }
    let total_repetitions = useful_units / activation_batch_width;
    let repetitions_per_quantum =
        (VULKAN_TARGETED_COMPONENT_QUANTUM_USEFUL_UNITS
            / activation_batch_width)
            .max(1);
    let mut remaining = total_repetitions;
    let mut quanta = Vec::new();
    while remaining > 0 {
        let repetitions = remaining.min(repetitions_per_quantum);
        quanta.push(repetitions);
        remaining -= repetitions;
    }
    Ok(quanta)
}

fn targeted_fixture_bytes(
    byte_count: usize,
    seed: u32,
    binding: usize,
) -> Vec<u8> {
    let mut state = u64::from(seed)
        ^ (u64::try_from(binding).unwrap_or(u64::MAX) << 32)
        ^ 0x9E37_79B9_7F4A_7C15;
    let mut bytes = Vec::with_capacity(byte_count);
    while bytes.len() + 1 < byte_count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let sample = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let signed = ((sample >> 32) as i32) as f32 / i32::MAX as f32;
        let bounded = signed.clamp(-1.0, 1.0) * 4.0;
        let bits = bounded.to_bits();
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bytes.extend_from_slice(
            &((bits.wrapping_add(rounding_bias) >> 16) as u16).to_le_bytes(),
        );
    }
    if bytes.len() < byte_count {
        bytes.push(0);
    }
    bytes
}

fn targeted_finalized_artifact_digest(payload: &[u8]) -> String {
    format!(
        "nerve.optimizer.artifact_sha256.v1:{}",
        payload
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn hex_bytes(payload: &[u8]) -> String {
    payload
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn elapsed_nanoseconds(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn targeted_component_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(targeted_component_error_value(message))
}

fn targeted_component_error_value(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message.into())
}
