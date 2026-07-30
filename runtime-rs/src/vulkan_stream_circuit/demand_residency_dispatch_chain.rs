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
    address_mapping: VulkanCompiledSelectorAddressMapping,
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
    first_gate_command_index: usize,
    continuation_predicate: Arc<VulkanResidentBuffer>,
    continuation_enabled: Cell<bool>,
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
                    address_mapping: selector_layout.mapping.clone(),
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
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        post_copies: &[VulkanResidentBufferRangeCopy<'_>],
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
            input_copies,
            post_copies,
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
        if first_gate_command_index + 1 == commands.len() {
            return Err(demand_dispatch_error(
                "demand gate has no selected computation after it",
            ));
        }
        let continuation_predicate = Arc::new(
            device
                .create_conditional_resident_buffer(size_of::<u32>())
                .map_err(
                VulkanMountedPlacedResidentKernelDispatchError::Vulkan,
            )?,
        );
        continuation_predicate
            .write_bytes(&1u32.to_le_bytes())
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
            let checkpoint_tag = u32::try_from(gate_index + 1).map_err(|_| {
                demand_dispatch_error("demand gate count exceeds u32")
            })?;
            let address_mapping = match &spec.address_mapping {
                VulkanCompiledSelectorAddressMapping::GroupTable {
                    resource_address_slots,
                    resource_address_slot_offsets,
                } => VulkanGpuResidencyAddressMapping::GroupTable {
                    resource_address_slots: resource_address_slots.clone(),
                    resource_address_slot_offsets:
                        resource_address_slot_offsets.clone(),
                },
                VulkanCompiledSelectorAddressMapping::PartitionTemplate {
                    member_slot_bases,
                    resource_count,
                    ..
                } => VulkanGpuResidencyAddressMapping::Partitioned {
                    member_slot_bases: member_slot_bases.clone(),
                    resource_count: *resource_count,
                },
            };
            let gate = VulkanGpuResidencyGate::new(
                device,
                &gate_shader,
                Arc::clone(&spec.selection_buffer),
                Arc::clone(&address_table),
                address_table_slot_count,
                missing_queue.clone(),
                Arc::clone(&continuation_predicate),
                VulkanGpuResidencyGateConfig {
                    maximum_selection_count: spec.selection_count,
                    selection_count_per_lane: spec.selection_count,
                    selection_lane_stride_words: spec.selection_count,
                    selection_index_shift: spec.selection_index_shift,
                    selection_index_mask: spec.selection_index_mask,
                    address_mapping,
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
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        post_copies: &[VulkanResidentBufferRangeCopy<'_>],
        context: &VulkanDemandResidencyExecutionContext,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if !self.continuation_enabled.get() {
            self.continuation_predicate
                .write_bytes(&1u32.to_le_bytes())
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            self.continuation_enabled.set(true);
        }
        self.run_from_gate(
            device,
            None,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            wait_points,
            signal_points,
            input_copies,
            post_copies,
        )?;
        let mut resume_count = 0usize;
        loop {
            let notification_epoch = self
                .missing_queue
                .notification_epoch()
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            if notification_epoch == self.observed_notification_epoch.get() {
                self.continuation_enabled.set(true);
                return Ok(());
            }
            self.continuation_enabled.set(false);
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
                &[],
                post_copies,
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
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        post_copies: &[VulkanResidentBufferRangeCopy<'_>],
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
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let mut steps = Vec::with_capacity(self.commands.len() - start_command_index);
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
                let region_id =
                    if matches!(command, VulkanDemandResidencyCommand::Gate(_)) {
                        conditional_region_id =
                            conditional_region_id.checked_add(1).ok_or_else(|| {
                                demand_dispatch_error(
                                    "demand conditional region count exceeds u32",
                                )
                            })?;
                        let gate_region = conditional_region_id;
                        conditional_region_id =
                            conditional_region_id.checked_add(1).ok_or_else(|| {
                                demand_dispatch_error(
                                    "demand conditional region count exceeds u32",
                                )
                            })?;
                        gate_region
                    } else {
                        conditional_region_id
                    };
                steps.push(
                    VulkanResidentKernelSequenceStep::new_conditional(
                        dispatch,
                        push_constants,
                        &self.continuation_predicate,
                        0,
                        false,
                        region_id,
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
                );
            }
        }
        if input_copies.is_empty() && post_copies.is_empty() {
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
            return device
                .wait_resident_kernel_sequence(sequence)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan);
        }
        if !wait_points.is_empty() || !signal_points.is_empty() {
            return Err(demand_dispatch_error(
                "demand-resident inline copies cannot cross a timeline boundary",
            ));
        }
        let after_step_index = steps
            .len()
            .checked_sub(1)
            .expect("demand-resident chains contain at least one step");
        let snapshot_copies = post_copies
            .iter()
            .copied()
            .map(|copy| {
                VulkanResidentKernelSequenceSnapshotCopy::
                    unconditional_from_range_after_conditional_step(
                        after_step_index,
                        copy,
                    )
            })
            .collect::<Vec<_>>();
        device
            .run_resident_kernel_sequence_with_input_and_snapshot_copies(
                sequence,
                input_copies,
                &steps,
                &snapshot_copies,
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }
}

fn demand_dispatch_error(
    message: impl Into<String>,
) -> VulkanMountedPlacedResidentKernelDispatchError {
    VulkanMountedPlacedResidentKernelDispatchError::Vulkan(VulkanError(message.into()))
}
