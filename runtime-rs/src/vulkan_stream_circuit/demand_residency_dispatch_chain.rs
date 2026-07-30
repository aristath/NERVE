#[derive(Clone)]
struct VulkanDemandResidencyExecutionContext {
    execution_scope: String,
    contract: Arc<CompiledResourceResidencyContract>,
    layout: Arc<VulkanCompiledResourceAddressLayout>,
    store: Arc<VulkanCompiledResourceDeviceStore>,
    owner: DeviceResourceResidencyOwnerId,
}

#[derive(Clone)]
struct VulkanDemandResidencyGateSpec {
    checkpoint_id: String,
    selector_id: String,
    command_after_dispatch_index: usize,
    selection_count: usize,
    selection_index_shift: u32,
    selection_index_mask: u32,
    address_slots_by_resource_index: Vec<Vec<usize>>,
    selection_buffer: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDemandResidencyCommand {
    Prefix(usize),
    Dispatch(usize),
    Gate(usize),
    Suffix(usize),
}

struct VulkanDemandResidencyGateRuntime {
    checkpoint_id: String,
    selector_id: String,
    command_index: usize,
    checkpoint_tag: u32,
    selection_count: usize,
    gate: VulkanGpuResidencyGate,
}

struct VulkanDemandResidencyDispatchChain {
    commands: Vec<VulkanDemandResidencyCommand>,
    command_indirect_offsets: Vec<Option<usize>>,
    first_gate_command_index: usize,
    indirect_dispatches: Arc<VulkanResidentBuffer>,
    missing_queue: VulkanGpuResidencyMissQueue,
    gates: Vec<VulkanDemandResidencyGateRuntime>,
    full_sequence: VulkanResidentKernelSequence,
    resume_sequences: Vec<VulkanResidentKernelSequence>,
    observed_notification_epoch: Cell<u32>,
}

struct VulkanDemandResidencySegment {
    context: VulkanDemandResidencyExecutionContext,
    gate_specs: Vec<VulkanDemandResidencyGateSpec>,
    address_table: Arc<VulkanResidentBuffer>,
    address_table_slot_count: usize,
    chains: RefCell<BTreeMap<u8, VulkanDemandResidencyDispatchChain>>,
}

impl VulkanDemandResidencySegment {
    fn from_segment(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        schedule: &VulkanPhysicalResidencySchedule,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        context: VulkanDemandResidencyExecutionContext,
    ) -> Result<Option<Self>, VulkanMountedPlacedResidentKernelDispatchError> {
        if schedule.execution_scope != context.execution_scope {
            return Err(demand_dispatch_error(format!(
                "physical residency scope {:?} does not match demand execution scope {:?}",
                schedule.execution_scope, context.execution_scope
            )));
        }
        let dispatch_indices = dispatches
            .iter()
            .map(|dispatch| dispatch.dispatch_index)
            .collect::<BTreeSet<_>>();
        let checkpoints = schedule
            .checkpoints
            .iter()
            .filter(|checkpoint| {
                dispatch_indices.contains(&checkpoint.selection_dispatch_index)
            })
            .collect::<Vec<_>>();
        if checkpoints.is_empty() {
            return Ok(None);
        }
        let mut gate_specs = Vec::new();
        for checkpoint in checkpoints {
            if checkpoint
                .selected_computation_dispatch_indices
                .iter()
                .any(|dispatch_index| !dispatch_indices.contains(dispatch_index))
                || checkpoint
                    .selected_result_continuation_dispatch_index
                    .is_some_and(|dispatch_index| !dispatch_indices.contains(&dispatch_index))
            {
                return Err(demand_dispatch_error(format!(
                    "residency checkpoint {:?} crosses a resident dispatch-segment boundary",
                    checkpoint.id
                )));
            }
            let selection_dispatch = mounted_bound_plan
                .dispatches
                .iter()
                .find(|dispatch| {
                    dispatch.dispatch_index == checkpoint.selection_dispatch_index
                })
                .ok_or_else(|| {
                    demand_dispatch_error(format!(
                        "residency checkpoint {:?} selection dispatch {} is not mounted",
                        checkpoint.id, checkpoint.selection_dispatch_index
                    ))
                })?;
            for selector_id in &checkpoint.selector_ids {
                let selector = context
                    .contract
                    .selectors
                    .iter()
                    .find(|selector| selector.id == *selector_id)
                    .ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "residency checkpoint {:?} references unknown selector {selector_id:?}",
                            checkpoint.id
                        ))
                    })?;
                if selector.execution_scope != context.execution_scope
                    || selector.component_id != checkpoint.component_id
                {
                    return Err(demand_dispatch_error(format!(
                        "selector {selector_id:?} does not belong to checkpoint {:?}",
                        checkpoint.id
                    )));
                }
                let selector_layout = context
                    .layout
                    .selectors
                    .iter()
                    .find(|layout| layout.selector_id == *selector_id)
                    .ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "selector {selector_id:?} has no stable-address layout"
                        ))
                    })?;
                let matching_buffers = selection_dispatch
                    .descriptors
                    .iter()
                    .filter_map(|descriptor| match &descriptor.target {
                        VulkanMountedPlacedBoundDescriptorTarget::Resident {
                            target:
                                VulkanBoundDescriptorTarget::ActivationSlot {
                                    buffer_index,
                                    component_id,
                                    signal_id,
                                    ..
                                },
                        } if component_id == &selector.component_id
                            && signal_id == &selector.selection_signal =>
                        {
                            Some(*buffer_index)
                        }
                        _ => None,
                    })
                    .collect::<BTreeSet<_>>();
                if matching_buffers.len() != 1 {
                    return Err(demand_dispatch_error(format!(
                        "selector {selector_id:?} selection signal {:?} resolves to {} mounted buffers",
                        selector.selection_signal,
                        matching_buffers.len()
                    )));
                }
                let selection_buffer_index = *matching_buffers
                    .first()
                    .expect("one matching selection buffer was validated");
                if mounted
                    .buffers
                    .activation_slot_buffers
                    .get(selection_buffer_index)
                    .is_none()
                {
                    return Err(demand_dispatch_error(format!(
                        "selector {selector_id:?} selection buffer {selection_buffer_index} is absent"
                    )));
                }
                let selection_buffer = mounted
                    .buffers
                    .activation_slot_buffers
                    .get(selection_buffer_index)
                    .map(|buffer| Arc::clone(&buffer.buffer))
                    .ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "selector {selector_id:?} selection buffer {selection_buffer_index} is absent"
                        ))
                    })?;
                gate_specs.push(VulkanDemandResidencyGateSpec {
                    checkpoint_id: checkpoint.id.clone(),
                    selector_id: selector.id.clone(),
                    command_after_dispatch_index: checkpoint.selection_dispatch_index,
                    selection_count: selector.encoding.selection_count_per_activation,
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
            return Err(demand_dispatch_error(
                "demand-resident segment contains checkpoints without selectors",
            ));
        }
        let dynamic_resources = mounted.dynamic_resource_buffers.as_ref().ok_or_else(|| {
            demand_dispatch_error(
                "demand-resident segment has no mounted dynamic-resource buffers",
            )
        })?;
        Ok(Some(Self {
            context,
            gate_specs,
            address_table: dynamic_resources.shared_address_table(),
            address_table_slot_count: dynamic_resources.address_table_slot_count(),
            chains: RefCell::new(BTreeMap::new()),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if !self.chains.borrow().contains_key(&sequence_variant) {
            let chain = VulkanDemandResidencyDispatchChain::new(
                device,
                dispatches,
                &self.gate_specs,
                Arc::clone(&self.address_table),
                self.address_table_slot_count,
                prefix_dispatches,
                suffix_dispatches,
            )?;
            self.chains.borrow_mut().insert(sequence_variant, chain);
        }
        let chains = self.chains.borrow();
        let chain = chains
            .get(&sequence_variant)
            .expect("demand chain variant was initialized");
        chain.run(
            device,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            wait_points,
            signal_points,
            &self.context,
        )
    }
}

impl VulkanDemandResidencyDispatchChain {
    fn new(
        device: &VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        gate_specs: &[VulkanDemandResidencyGateSpec],
        address_table: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let mut gates_after_dispatch = BTreeMap::<usize, Vec<usize>>::new();
        for (gate_index, gate) in gate_specs.iter().enumerate() {
            gates_after_dispatch
                .entry(gate.command_after_dispatch_index)
                .or_default()
                .push(gate_index);
        }
        let mut commands = prefix_dispatches
            .iter()
            .enumerate()
            .map(|(index, _)| VulkanDemandResidencyCommand::Prefix(index))
            .collect::<Vec<_>>();
        for (dispatch_index, dispatch) in dispatches.iter().enumerate() {
            commands.push(VulkanDemandResidencyCommand::Dispatch(dispatch_index));
            if let Some(gate_indices) =
                gates_after_dispatch.remove(&dispatch.dispatch_index)
            {
                commands.extend(
                    gate_indices
                        .into_iter()
                        .map(VulkanDemandResidencyCommand::Gate),
                );
            }
        }
        if !gates_after_dispatch.is_empty() {
            return Err(demand_dispatch_error(format!(
                "demand gates reference dispatches outside their resident segment: {:?}",
                gates_after_dispatch.keys().collect::<Vec<_>>()
            )));
        }
        commands.extend(
            suffix_dispatches
                .iter()
                .enumerate()
                .map(|(index, _)| VulkanDemandResidencyCommand::Suffix(index)),
        );
        let first_gate_command_index = commands
            .iter()
            .position(|command| matches!(command, VulkanDemandResidencyCommand::Gate(_)))
            .expect("demand segment has at least one gate");
        let mut command_indirect_offsets = vec![None; commands.len()];
        let mut next_indirect_offset = 0usize;
        for (command_index, offset) in command_indirect_offsets
            .iter_mut()
            .enumerate()
            .skip(first_gate_command_index + 1)
        {
            *offset = Some(next_indirect_offset);
            next_indirect_offset = next_indirect_offset
                .checked_add(VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT)
                .ok_or_else(|| {
                    demand_dispatch_error(format!(
                        "demand command {command_index} indirect offset overflowed"
                    ))
                })?;
        }
        if next_indirect_offset == 0 {
            return Err(demand_dispatch_error(
                "demand gate has no selected computation after it",
            ));
        }
        let indirect_dispatches =
            Arc::new(device.create_resident_buffer(next_indirect_offset).map_err(
                VulkanMountedPlacedResidentKernelDispatchError::Vulkan,
            )?);
        indirect_dispatches
            .write_bytes(&vec![0; indirect_dispatches.byte_capacity()])
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let missing_capacity = gate_specs
            .iter()
            .map(|gate| gate.selection_count)
            .max()
            .expect("demand segment gates are non-empty");
        let missing_queue =
            VulkanGpuResidencyMissQueue::new(device, missing_capacity)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let gate_shader = vulkan_gpu_residency_gate_spirv_words()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let mut gates = Vec::with_capacity(gate_specs.len());
        for (gate_index, spec) in gate_specs.iter().enumerate() {
            let command_index = commands
                .iter()
                .position(|command| {
                    *command == VulkanDemandResidencyCommand::Gate(gate_index)
                })
                .expect("expanded command chain contains every gate");
            let downstream_dispatches = commands
                .iter()
                .enumerate()
                .skip(command_index + 1)
                .map(|(downstream_index, command)| {
                    let byte_offset = command_indirect_offsets[downstream_index]
                        .expect("every command after a gate is indirect");
                    Ok(VulkanGpuResidencyIndirectDispatch {
                        byte_offset,
                        dimensions: demand_command_dimensions(
                            *command,
                            dispatches,
                            prefix_dispatches,
                            suffix_dispatches,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, VulkanMountedPlacedResidentKernelDispatchError>>()?;
            let checkpoint_tag = u32::try_from(gate_index + 1).map_err(|_| {
                demand_dispatch_error("demand gate count exceeds u32")
            })?;
            let gate = VulkanGpuResidencyGate::new(
                device,
                &gate_shader,
                Arc::clone(&spec.selection_buffer),
                Arc::clone(&address_table),
                address_table_slot_count,
                missing_queue.clone(),
                Arc::clone(&indirect_dispatches),
                VulkanGpuResidencyGateConfig {
                    maximum_selection_count: spec.selection_count,
                    selection_count_per_lane: spec.selection_count,
                    selection_lane_stride_words: spec.selection_count,
                    selection_index_shift: spec.selection_index_shift,
                    selection_index_mask: spec.selection_index_mask,
                    address_slots_by_resource_index: spec
                        .address_slots_by_resource_index
                        .clone(),
                    downstream_dispatches,
                },
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            gates.push(VulkanDemandResidencyGateRuntime {
                checkpoint_id: spec.checkpoint_id.clone(),
                selector_id: spec.selector_id.clone(),
                command_index,
                checkpoint_tag,
                selection_count: spec.selection_count,
                gate,
            });
        }
        let full_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let resume_sequences = gates
            .iter()
            .map(|_| device.create_resident_kernel_sequence())
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        Ok(Self {
            commands,
            command_indirect_offsets,
            first_gate_command_index,
            indirect_dispatches,
            missing_queue,
            gates,
            full_sequence,
            resume_sequences,
            observed_notification_epoch: Cell::new(0),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        context: &VulkanDemandResidencyExecutionContext,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        self.run_from_gate(
            device,
            None,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            wait_points,
            signal_points,
        )?;
        let mut resume_count = 0usize;
        loop {
            let notification_epoch = self
                .missing_queue
                .notification_epoch()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            if notification_epoch == self.observed_notification_epoch.get() {
                return Ok(());
            }
            let missing = self
                .missing_queue
                .snapshot()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            if missing.overflowed || missing.requests.is_empty() {
                return Err(demand_dispatch_error(format!(
                    "GPU residency gate reported an invalid or overflowing miss queue: epoch={}, requests={}",
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
                return Err(demand_dispatch_error(format!(
                    "one synchronous demand traversal reported misses at multiple checkpoints: {checkpoint_tags:?}"
                )));
            }
            let checkpoint_tag = *checkpoint_tags
                .first()
                .expect("one checkpoint tag was validated");
            let gate_index = self
                .gates
                .iter()
                .position(|gate| gate.checkpoint_tag == checkpoint_tag)
                .ok_or_else(|| {
                    demand_dispatch_error(format!(
                        "GPU residency miss references unknown checkpoint tag {checkpoint_tag}"
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
                    demand_dispatch_error(format!(
                        "failed to record GPU residency misses for selector {:?}: {error}",
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
                    demand_dispatch_error(format!(
                        "failed to load selector {:?} resources {resource_indices:?} at checkpoint {:?}: {error}",
                        gate.selector_id, gate.checkpoint_id
                    ))
                })?;
            self.missing_queue
                .acknowledge_through(missing.published_count)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            self.observed_notification_epoch
                .set(missing.notification_epoch);
            resume_count = resume_count.checked_add(1).ok_or_else(|| {
                demand_dispatch_error("demand checkpoint resume count overflowed")
            })?;
            if resume_count > self.gates.len() {
                return Err(demand_dispatch_error(
                    "demand execution exceeded one resume per physical checkpoint",
                ));
            }
            self.run_from_gate(
                device,
                Some(gate_index),
                dispatches,
                control,
                prefix_dispatches,
                suffix_dispatches,
                &[],
                &[],
            )?;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_from_gate(
        &self,
        device: &VulkanComputeDevice,
        resume_gate_index: Option<usize>,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        let model_push_constants = dispatches
            .iter()
            .map(|dispatch| {
                stream_control_push_constant_bytes(&dispatch.push_constants, control)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (start_command_index, direct_gate_command_index, sequence) =
            match resume_gate_index {
                Some(gate_index) => {
                    let gate = self.gates.get(gate_index).ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "demand resume gate {gate_index} is out of bounds"
                        ))
                    })?;
                    (
                        gate.command_index,
                        gate.command_index,
                        self.resume_sequences.get(gate_index).ok_or_else(|| {
                            demand_dispatch_error(format!(
                                "demand resume sequence {gate_index} is absent"
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
        let gate_push_constants = self
            .gates
            .iter()
            .map(|gate| {
                gate.gate
                    .push_constants(gate.selection_count, gate.checkpoint_tag)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let mut steps = Vec::with_capacity(self.commands.len() - start_command_index);
        for (command_index, command) in self
            .commands
            .iter()
            .copied()
            .enumerate()
            .skip(start_command_index)
        {
            let (dispatch, push_constants): (&VulkanResidentKernelDispatch, &[u8]) =
                match command {
                    VulkanDemandResidencyCommand::Prefix(index) => (
                        *prefix_dispatches.get(index).ok_or_else(|| {
                            demand_dispatch_error(format!(
                                "demand prefix dispatch {index} disappeared"
                            ))
                        })?,
                        &[],
                    ),
                    VulkanDemandResidencyCommand::Dispatch(index) => (
                        &dispatches
                            .get(index)
                            .ok_or_else(|| {
                                demand_dispatch_error(format!(
                                    "demand model dispatch {index} disappeared"
                                ))
                            })?
                            .resident_dispatch,
                        model_push_constants
                            .get(index)
                            .ok_or_else(|| {
                                demand_dispatch_error(format!(
                                    "demand model push constants {index} disappeared"
                                ))
                            })?
                            .as_slice(),
                    ),
                    VulkanDemandResidencyCommand::Gate(index) => (
                        self.gates
                            .get(index)
                            .ok_or_else(|| {
                                demand_dispatch_error(format!(
                                    "demand gate dispatch {index} disappeared"
                                ))
                            })?
                            .gate
                            .dispatch(),
                        gate_push_constants
                            .get(index)
                            .ok_or_else(|| {
                                demand_dispatch_error(format!(
                                    "demand gate push constants {index} disappeared"
                                ))
                            })?
                            .as_slice(),
                    ),
                    VulkanDemandResidencyCommand::Suffix(index) => (
                        *suffix_dispatches.get(index).ok_or_else(|| {
                            demand_dispatch_error(format!(
                                "demand suffix dispatch {index} disappeared"
                            ))
                        })?,
                        &[],
                    ),
                };
            if command_index <= direct_gate_command_index {
                steps.push(VulkanResidentKernelSequenceStep::new(
                    dispatch,
                    push_constants,
                ));
            } else {
                steps.push(
                    VulkanResidentKernelSequenceStep::new_indirect(
                        dispatch,
                        push_constants,
                        &self.indirect_dispatches,
                        self.command_indirect_offsets[command_index]
                            .expect("every command after the direct gate is indirect"),
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
                );
            }
        }
        device
            .record_resident_kernel_sequence(sequence, &steps)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        device
            .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                sequence,
                wait_points,
                signal_points,
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        device
            .wait_resident_kernel_sequence(sequence)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }
}

fn demand_command_dimensions(
    command: VulkanDemandResidencyCommand,
    dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
    prefix_dispatches: &[&VulkanResidentKernelDispatch],
    suffix_dispatches: &[&VulkanResidentKernelDispatch],
) -> Result<[u32; 3], VulkanMountedPlacedResidentKernelDispatchError> {
    if matches!(command, VulkanDemandResidencyCommand::Gate(_)) {
        return Ok([1, 1, 1]);
    }
    let dispatch = match command {
        VulkanDemandResidencyCommand::Prefix(index) => {
            *prefix_dispatches.get(index).ok_or_else(|| {
                demand_dispatch_error(format!(
                    "demand prefix dispatch {index} is absent"
                ))
            })?
        }
        VulkanDemandResidencyCommand::Dispatch(index) => &dispatches
            .get(index)
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "demand model dispatch {index} is absent"
                ))
            })?
            .resident_dispatch,
        VulkanDemandResidencyCommand::Gate(_) => unreachable!("gate dimensions returned early"),
        VulkanDemandResidencyCommand::Suffix(index) => {
            *suffix_dispatches.get(index).ok_or_else(|| {
                demand_dispatch_error(format!(
                    "demand suffix dispatch {index} is absent"
                ))
            })?
        }
    };
    Ok([
        dispatch.workgroup_count_x(),
        dispatch.workgroup_count_y(),
        1,
    ])
}

fn demand_dispatch_error(
    message: impl Into<String>,
) -> VulkanMountedPlacedResidentKernelDispatchError {
    VulkanMountedPlacedResidentKernelDispatchError::Vulkan(VulkanError(message.into()))
}
