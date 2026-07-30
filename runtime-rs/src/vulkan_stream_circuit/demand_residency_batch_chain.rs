#[derive(Clone)]
struct VulkanDemandResidencyBatchGateSpec {
    checkpoint_id: String,
    selector_id: String,
    command_after_step_index: usize,
    selection_count_per_activation: usize,
    selection_lane_stride_words: usize,
    selection_index_shift: u32,
    selection_index_mask: u32,
    address_slots_by_resource_index: Vec<Vec<usize>>,
    selection_buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDemandResidencyBatchCommand {
    Step(usize),
    Gate(usize),
}

fn demand_batch_commands_are_replay_stable(
    commands: &[VulkanDemandResidencyBatchCommand],
    start_command_index: usize,
    mut step_has_static_push_constants: impl FnMut(usize) -> bool,
) -> bool {
    commands
        .get(start_command_index..)
        .is_some_and(|commands| {
            commands.iter().all(|command| match command {
                VulkanDemandResidencyBatchCommand::Step(step_index) => {
                    step_has_static_push_constants(*step_index)
                }
                VulkanDemandResidencyBatchCommand::Gate(_) => true,
            })
        })
}

struct VulkanDemandResidencyBatchGateRuntime {
    checkpoint_id: String,
    selector_id: String,
    command_index: usize,
    checkpoint_tag: u32,
    selection_count: usize,
    gate: VulkanGpuResidencyGate,
}

struct VulkanDemandResidencyBatchChain {
    commands: Vec<VulkanDemandResidencyBatchCommand>,
    first_gate_command_index: usize,
    continuation_predicate: Arc<VulkanResidentBuffer>,
    continuation_enabled: Cell<bool>,
    missing_queue: VulkanGpuResidencyMissQueue,
    gates: Vec<VulkanDemandResidencyBatchGateRuntime>,
    full_sequence: VulkanResidentKernelSequence,
    resume_sequences: Vec<VulkanResidentKernelSequence>,
    observed_notification_epoch: Cell<u32>,
}

struct VulkanDemandResidencyBatchSegment {
    context: VulkanDemandResidencyExecutionContext,
    gate_specs: Vec<VulkanDemandResidencyBatchGateSpec>,
    address_table: Arc<VulkanResidentBuffer>,
    address_table_slot_count: usize,
    step_start: usize,
    step_end: usize,
    lane_capacity: usize,
    chains: RefCell<BTreeMap<usize, VulkanDemandResidencyBatchChain>>,
}

impl VulkanDemandResidencyBatchSegment {
    #[allow(clippy::too_many_arguments)]
    fn from_slice_steps(
        mounted: &VulkanMountedPlacedStreamCircuit,
        schedule: &VulkanPhysicalResidencySchedule,
        dispatch_spans: &[VulkanComponentBatchDispatchSpan],
        signal_buffers: &[VulkanComponentBatchSignalBuffer],
        signal_buffer_indices: &BTreeMap<VulkanComponentBatchSignalKey, usize>,
        step_start: usize,
        step_end: usize,
        lane_capacity: usize,
        context: VulkanDemandResidencyExecutionContext,
    ) -> Result<Option<Self>, VulkanResidentInProcessPlacedRuntimeError> {
        if schedule.execution_scope != context.execution_scope {
            return Err(demand_batch_error(format!(
                "physical residency scope {:?} does not match batch execution scope {:?}",
                schedule.execution_scope, context.execution_scope
            )));
        }
        if step_start >= step_end {
            return Err(demand_batch_error(
                "demand-resident batch segment has no executable steps",
            ));
        }
        let spans_by_dispatch = dispatch_spans
            .iter()
            .map(|span| (span.dispatch_index, span))
            .collect::<BTreeMap<_, _>>();
        let mut gate_specs = Vec::new();
        for checkpoint in &schedule.checkpoints {
            let Some(selection_span) =
                spans_by_dispatch.get(&checkpoint.selection_dispatch_index)
            else {
                continue;
            };
            if selection_span.step_start < step_start
                || selection_span.step_end > step_end
            {
                continue;
            }
            if selection_span.distributed {
                return Err(demand_batch_error(format!(
                    "batch residency checkpoint {:?} selects inside a distributed dispatch rather than at its owner-device boundary",
                    checkpoint.id
                )));
            }
            for dispatch_index in checkpoint
                .selected_computation_dispatch_indices
                .iter()
                .copied()
                .chain(checkpoint.selected_result_continuation_dispatch_index)
            {
                let span = spans_by_dispatch.get(&dispatch_index).ok_or_else(|| {
                    demand_batch_error(format!(
                        "batch residency checkpoint {:?} references absent dispatch {dispatch_index}",
                        checkpoint.id
                    ))
                })?;
                if span.distributed
                    || span.step_start < step_start
                    || span.step_end > step_end
                {
                    return Err(demand_batch_error(format!(
                        "batch residency checkpoint {:?} crosses a local execution boundary",
                        checkpoint.id
                    )));
                }
            }
            for selector_id in &checkpoint.selector_ids {
                let selector = context
                    .contract
                    .selectors
                    .iter()
                    .find(|selector| selector.id == *selector_id)
                    .ok_or_else(|| {
                        demand_batch_error(format!(
                            "batch checkpoint {:?} references unknown selector {selector_id:?}",
                            checkpoint.id
                        ))
                    })?;
                if selector.execution_scope != context.execution_scope
                    || selector.component_id != checkpoint.component_id
                {
                    return Err(demand_batch_error(format!(
                        "batch selector {selector_id:?} does not belong to checkpoint {:?}",
                        checkpoint.id
                    )));
                }
                let selector_layout = context
                    .layout
                    .selectors
                    .iter()
                    .find(|layout| layout.selector_id == *selector_id)
                    .ok_or_else(|| {
                        demand_batch_error(format!(
                            "batch selector {selector_id:?} has no stable-address layout"
                        ))
                    })?;
                let selection_key = VulkanComponentBatchSignalKey::Activation {
                    component_id: selector.component_id.clone(),
                    signal_id: selector.selection_signal.clone(),
                };
                let selection_buffer_index = signal_buffer_indices
                    .get(&selection_key)
                    .copied()
                    .ok_or_else(|| {
                        demand_batch_error(format!(
                            "batch selector {selector_id:?} selection signal {:?} has no batch buffer",
                            selector.selection_signal
                        ))
                    })?;
                let selection_buffer_allocation = signal_buffers
                    .get(selection_buffer_index)
                    .ok_or_else(|| {
                        demand_batch_error(format!(
                            "batch selector {selector_id:?} selection buffer {selection_buffer_index} is absent"
                        ))
                    })?;
                if selection_buffer_allocation.frame_byte_capacity
                    % size_of::<u32>()
                    != 0
                {
                    return Err(demand_batch_error(format!(
                        "batch selector {selector_id:?} frame capacity {} is not u32-aligned",
                        selection_buffer_allocation.frame_byte_capacity
                    )));
                }
                let selection_lane_stride_words =
                    selection_buffer_allocation.frame_byte_capacity
                        / size_of::<u32>();
                if selection_lane_stride_words
                    < selector.encoding.selection_count_per_activation
                {
                    return Err(demand_batch_error(format!(
                        "batch selector {selector_id:?} frame has {selection_lane_stride_words} words but needs {} selections",
                        selector.encoding.selection_count_per_activation
                    )));
                }
                let selection_buffer =
                    Arc::clone(&selection_buffer_allocation.buffer);
                let required_selection_words = lane_capacity
                    .saturating_sub(1)
                    .checked_mul(selection_lane_stride_words)
                    .and_then(|offset| {
                        offset.checked_add(
                            selector.encoding.selection_count_per_activation,
                        )
                    })
                    .ok_or_else(|| {
                        demand_batch_error(
                            "batch selector selection capacity overflowed",
                        )
                    })?;
                let required_selection_bytes = required_selection_words
                    .checked_mul(size_of::<u32>())
                    .ok_or_else(|| {
                        demand_batch_error(
                            "batch selector selection byte capacity overflowed",
                        )
                    })?;
                if required_selection_bytes > selection_buffer.byte_capacity() {
                    return Err(demand_batch_error(format!(
                        "batch selector {selector_id:?} needs {required_selection_bytes} selection bytes but its signal buffer has {}",
                        selection_buffer.byte_capacity()
                    )));
                }
                gate_specs.push(VulkanDemandResidencyBatchGateSpec {
                    checkpoint_id: checkpoint.id.clone(),
                    selector_id: selector.id.clone(),
                    command_after_step_index: selection_span.step_end,
                    selection_count_per_activation: selector
                        .encoding
                        .selection_count_per_activation,
                    selection_lane_stride_words,
                    selection_index_shift: selector.encoding.index_shift,
                    selection_index_mask: selector.encoding.index_mask,
                    address_slots_by_resource_index: selector_layout
                        .resource_address_slots
                        .clone(),
                    selection_buffer,
                });
            }
        }
        if gate_specs.is_empty() {
            return Ok(None);
        }
        let dynamic_resources = mounted.dynamic_resource_buffers.as_ref().ok_or_else(|| {
            demand_batch_error(
                "demand-resident component batch has no dynamic-resource buffers",
            )
        })?;
        Ok(Some(Self {
            context,
            gate_specs,
            address_table: dynamic_resources.shared_address_table(),
            address_table_slot_count: dynamic_resources.address_table_slot_count(),
            step_start,
            step_end,
            lane_capacity,
            chains: RefCell::new(BTreeMap::new()),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &VulkanComputeDevice,
        steps: &[VulkanComponentBatchDispatchStep],
        batch_width: usize,
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if batch_width == 0 || batch_width > self.lane_capacity {
            return Err(demand_batch_error(format!(
                "demand batch width {batch_width} is outside 1..={}",
                self.lane_capacity
            )));
        }
        if !self.chains.borrow().contains_key(&batch_width) {
            let chain = VulkanDemandResidencyBatchChain::new(
                device,
                steps,
                self.step_start,
                self.step_end,
                batch_width,
                &self.gate_specs,
                Arc::clone(&self.address_table),
                self.address_table_slot_count,
            )?;
            self.chains.borrow_mut().insert(batch_width, chain);
        }
        self.chains
            .borrow()
            .get(&batch_width)
            .expect("demand batch width was initialized")
            .run(
                device,
                steps,
                stream_ticks,
                dynamic_state_capacity_activations,
                &self.context,
            )
    }
}

impl VulkanDemandResidencyBatchChain {
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &VulkanComputeDevice,
        steps: &[VulkanComponentBatchDispatchStep],
        step_start: usize,
        step_end: usize,
        batch_width: usize,
        gate_specs: &[VulkanDemandResidencyBatchGateSpec],
        address_table: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let mut gates_before_step = BTreeMap::<usize, Vec<usize>>::new();
        for (gate_index, gate) in gate_specs.iter().enumerate() {
            gates_before_step
                .entry(gate.command_after_step_index)
                .or_default()
                .push(gate_index);
        }
        let mut commands = Vec::new();
        for step_index in step_start..step_end {
            if let Some(gate_indices) = gates_before_step.remove(&step_index) {
                commands.extend(
                    gate_indices
                        .into_iter()
                        .map(VulkanDemandResidencyBatchCommand::Gate),
                );
            }
            let step = steps.get(step_index).ok_or_else(|| {
                demand_batch_error(format!(
                    "demand batch step {step_index} is absent"
                ))
            })?;
            if step.lane_index.is_none_or(|lane| lane < batch_width) {
                commands.push(VulkanDemandResidencyBatchCommand::Step(step_index));
            }
        }
        if let Some(gate_indices) = gates_before_step.remove(&step_end) {
            commands.extend(
                gate_indices
                    .into_iter()
                    .map(VulkanDemandResidencyBatchCommand::Gate),
            );
        }
        if !gates_before_step.is_empty() {
            return Err(demand_batch_error(format!(
                "batch residency gates reference steps outside {step_start}..{step_end}: {:?}",
                gates_before_step.keys().collect::<Vec<_>>()
            )));
        }
        let first_gate_command_index = commands
            .iter()
            .position(|command| {
                matches!(command, VulkanDemandResidencyBatchCommand::Gate(_))
            })
            .ok_or_else(|| {
                demand_batch_error(
                    "demand-resident batch chain contains no residency gate",
                )
            })?;
        if first_gate_command_index + 1 == commands.len() {
            return Err(demand_batch_error(
                "demand batch gate has no selected computation after it",
            ));
        }
        let continuation_predicate = Arc::new(
            device
                .create_conditional_resident_buffer(size_of::<u32>())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
        );
        continuation_predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let missing_capacity = gate_specs
            .iter()
            .map(|gate| {
                gate.selection_count_per_activation
                    .checked_mul(batch_width)
                    .ok_or_else(|| {
                        demand_batch_error(
                            "demand batch missing-queue capacity overflowed",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .expect("demand batch gates are non-empty");
        let missing_queue =
            VulkanGpuResidencyMissQueue::new(device, missing_capacity)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let gate_shader = vulkan_gpu_residency_gate_spirv_words()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut gates = Vec::with_capacity(gate_specs.len());
        for (gate_index, spec) in gate_specs.iter().enumerate() {
            let command_index = commands
                .iter()
                .position(|command| {
                    *command
                        == VulkanDemandResidencyBatchCommand::Gate(gate_index)
                })
                .expect("expanded demand batch chain contains every gate");
            let checkpoint_tag = u32::try_from(gate_index + 1).map_err(|_| {
                demand_batch_error("demand batch gate count exceeds u32")
            })?;
            let selection_count = spec
                .selection_count_per_activation
                .checked_mul(batch_width)
                .ok_or_else(|| {
                    demand_batch_error(
                        "demand batch active selection count overflowed",
                    )
                })?;
            let gate = VulkanGpuResidencyGate::new(
                device,
                &gate_shader,
                Arc::clone(&spec.selection_buffer),
                Arc::clone(&address_table),
                address_table_slot_count,
                missing_queue.clone(),
                Arc::clone(&continuation_predicate),
                VulkanGpuResidencyGateConfig {
                    maximum_selection_count: selection_count,
                    selection_count_per_lane: spec.selection_count_per_activation,
                    selection_lane_stride_words: spec.selection_lane_stride_words,
                    selection_index_shift: spec.selection_index_shift,
                    selection_index_mask: spec.selection_index_mask,
                    address_slots_by_resource_index: spec
                        .address_slots_by_resource_index
                        .clone(),
                },
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            gates.push(VulkanDemandResidencyBatchGateRuntime {
                checkpoint_id: spec.checkpoint_id.clone(),
                selector_id: spec.selector_id.clone(),
                command_index,
                checkpoint_tag,
                selection_count,
                gate,
            });
        }
        let full_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let resume_sequences = gates
            .iter()
            .map(|_| device.create_resident_kernel_sequence())
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Ok(Self {
            commands,
            first_gate_command_index,
            continuation_predicate,
            continuation_enabled: Cell::new(true),
            missing_queue,
            gates,
            full_sequence,
            resume_sequences,
            observed_notification_epoch: Cell::new(0),
        })
    }

    fn run(
        &self,
        device: &VulkanComputeDevice,
        steps: &[VulkanComponentBatchDispatchStep],
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
        context: &VulkanDemandResidencyExecutionContext,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if !self.continuation_enabled.get() {
            self.continuation_predicate
                .write_bytes(&1u32.to_le_bytes())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.continuation_enabled.set(true);
        }
        self.run_from_gate(
            device,
            None,
            steps,
            stream_ticks,
            dynamic_state_capacity_activations,
        )?;
        let mut resume_count = 0usize;
        loop {
            let notification_epoch = self
                .missing_queue
                .notification_epoch()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if notification_epoch == self.observed_notification_epoch.get() {
                self.continuation_enabled.set(true);
                return Ok(());
            }
            self.continuation_enabled.set(false);
            let missing = self
                .missing_queue
                .snapshot()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if missing.overflowed || missing.requests.is_empty() {
                return Err(demand_batch_error(format!(
                    "GPU batch residency gate reported an invalid or overflowing miss queue: epoch={}, requests={}",
                    missing.notification_epoch,
                    missing.requests.len()
                )));
            }
            let checkpoint_tags = missing
                .requests
                .iter()
                .map(|request| request.checkpoint_tag)
                .collect::<BTreeSet<_>>();
            if checkpoint_tags.len() != 1 {
                return Err(demand_batch_error(format!(
                    "one demand batch traversal reported misses at multiple checkpoints: {checkpoint_tags:?}"
                )));
            }
            let checkpoint_tag = *checkpoint_tags
                .first()
                .expect("one demand batch checkpoint was validated");
            let gate_index = self
                .gates
                .iter()
                .position(|gate| gate.checkpoint_tag == checkpoint_tag)
                .ok_or_else(|| {
                    demand_batch_error(format!(
                        "GPU batch residency miss references unknown checkpoint tag {checkpoint_tag}"
                    ))
                })?;
            let gate = &self.gates[gate_index];
            context
                .store
                .record_gpu_gate_misses(
                    &gate.selector_id,
                    missing.requests.len(),
                )
                .map_err(|error| {
                    demand_batch_error(format!(
                        "failed to record GPU batch residency misses for selector {:?}: {error}",
                        gate.selector_id
                    ))
                })?;
            let resource_indices = missing
                .requests
                .iter()
                .map(|request| request.resource_index)
                .collect::<BTreeSet<_>>();
            let resource_indices =
                resource_indices.into_iter().collect::<Vec<_>>();
            context
                .store
                .load_selector_resources(
                    device,
                    &gate.selector_id,
                    &resource_indices,
                    context.owner.clone(),
                )
                .map_err(|error| {
                    demand_batch_error(format!(
                        "failed to load batch selector {:?} resources {resource_indices:?} at checkpoint {:?}: {error}",
                        gate.selector_id, gate.checkpoint_id
                    ))
                })?;
            self.missing_queue
                .acknowledge_through(missing.published_count)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.observed_notification_epoch
                .set(missing.notification_epoch);
            resume_count = resume_count.checked_add(1).ok_or_else(|| {
                demand_batch_error(
                    "demand batch checkpoint resume count overflowed",
                )
            })?;
            if resume_count > self.gates.len() {
                return Err(demand_batch_error(
                    "demand batch execution exceeded one resume per physical checkpoint",
                ));
            }
            self.run_from_gate(
                device,
                Some(gate_index),
                steps,
                stream_ticks,
                dynamic_state_capacity_activations,
            )?;
        }
    }

    fn run_from_gate(
        &self,
        device: &VulkanComputeDevice,
        resume_gate_index: Option<usize>,
        steps: &[VulkanComponentBatchDispatchStep],
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let (start_command_index, direct_gate_command_index, sequence) =
            match resume_gate_index {
                Some(gate_index) => {
                    let gate = self.gates.get(gate_index).ok_or_else(|| {
                        demand_batch_error(format!(
                            "demand batch resume gate {gate_index} is out of bounds"
                        ))
                    })?;
                    (
                        gate.command_index,
                        gate.command_index,
                        self.resume_sequences.get(gate_index).ok_or_else(|| {
                            demand_batch_error(format!(
                                "demand batch resume sequence {gate_index} is absent"
                            ))
                        })?,
                    )
                }
                None => (
                    0,
                    self.first_gate_command_index,
                    &self.full_sequence,
                ),
            };
        let can_replay_recorded_commands = std::env::var_os("NERVE_VK_PERF_LOGGER").is_none()
            && sequence.has_recorded_commands()
            && demand_batch_commands_are_replay_stable(
                &self.commands,
                start_command_index,
                |step_index| {
                    steps
                        .get(step_index)
                        .is_some_and(|step| step.push_constants.is_empty())
                },
            );
        if can_replay_recorded_commands {
            return device
                .run_recorded_resident_kernel_sequence(sequence)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop);
        }
        let step_push_constants = steps
            .iter()
            .map(|step| {
                if step
                    .lane_index
                    .is_some_and(|lane| lane >= stream_ticks.len())
                {
                    Ok(Vec::new())
                } else {
                    demand_batch_step_push_constants(
                        step,
                        stream_ticks,
                        dynamic_state_capacity_activations,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let gate_push_constants = self
            .gates
            .iter()
            .enumerate()
            .map(|(gate_index, gate)| {
                gate.gate
                    .push_constants(
                        gate.selection_count,
                        gate.checkpoint_tag,
                        resume_gate_index == Some(gate_index),
                    )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut sequence_steps =
            Vec::with_capacity(self.commands.len() - start_command_index);
        let mut conditional_region_id = 1u32;
        for (command_index, command) in self
            .commands
            .iter()
            .copied()
            .enumerate()
            .skip(start_command_index)
        {
            let (dispatch, push_constants): (&VulkanResidentKernelDispatch, &[u8]) =
                match command {
                    VulkanDemandResidencyBatchCommand::Step(step_index) => {
                        let step = steps.get(step_index).ok_or_else(|| {
                            demand_batch_error(format!(
                                "demand batch step {step_index} disappeared"
                            ))
                        })?;
                        (
                            &step.dispatch,
                            step_push_constants
                                .get(step_index)
                                .ok_or_else(|| {
                                    demand_batch_error(format!(
                                        "demand batch push constants {step_index} disappeared"
                                    ))
                                })?
                                .as_slice(),
                        )
                    }
                    VulkanDemandResidencyBatchCommand::Gate(gate_index) => {
                        let gate = self.gates.get(gate_index).ok_or_else(|| {
                            demand_batch_error(format!(
                                "demand batch gate {gate_index} disappeared"
                            ))
                        })?;
                        (
                            gate.gate.dispatch(),
                            gate_push_constants
                                .get(gate_index)
                                .ok_or_else(|| {
                                    demand_batch_error(format!(
                                        "demand batch gate constants {gate_index} disappeared"
                                    ))
                                })?
                                .as_slice(),
                        )
                    }
                };
            if command_index <= direct_gate_command_index {
                sequence_steps.push(VulkanResidentKernelSequenceStep::new(
                    dispatch,
                    push_constants,
                ));
            } else {
                let region_id = if matches!(
                    command,
                    VulkanDemandResidencyBatchCommand::Gate(_)
                ) {
                    conditional_region_id =
                        conditional_region_id.checked_add(1).ok_or_else(|| {
                            demand_batch_error(
                                "demand batch conditional region count exceeds u32",
                            )
                        })?;
                    let gate_region = conditional_region_id;
                    conditional_region_id =
                        conditional_region_id.checked_add(1).ok_or_else(|| {
                            demand_batch_error(
                                "demand batch conditional region count exceeds u32",
                            )
                        })?;
                    gate_region
                } else {
                    conditional_region_id
                };
                sequence_steps.push(
                    VulkanResidentKernelSequenceStep::new_conditional(
                        dispatch,
                        push_constants,
                        &self.continuation_predicate,
                        0,
                        false,
                        region_id,
                    )
                    .map_err(
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                    )?,
                );
            }
        }
        device
            .record_resident_kernel_sequence(sequence, &sequence_steps)
            .and_then(|_| {
                device.run_recorded_resident_kernel_sequence(sequence)
            })
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }
}

fn demand_batch_step_push_constants(
    step: &VulkanComponentBatchDispatchStep,
    stream_ticks: &[u64],
    dynamic_state_capacity_activations: u32,
) -> Result<Vec<u8>, VulkanResidentInProcessPlacedRuntimeError> {
    let Some(lane_index) = step.lane_index else {
        return Ok(Vec::new());
    };
    let stream_tick = *stream_ticks.get(lane_index).ok_or_else(|| {
        demand_batch_error(format!(
            "demand batch has no stream tick for lane {lane_index}"
        ))
    })?;
    stream_control_push_constant_bytes(
        &step.push_constants,
        VulkanMountedPlacedStreamControl {
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        },
    )
    .map_err(|error| {
        demand_batch_error(format!(
            "invalid demand batch stream control: {error}"
        ))
    })
}

fn demand_batch_error(
    message: impl Into<String>,
) -> VulkanResidentInProcessPlacedRuntimeError {
    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(message.into()))
}
