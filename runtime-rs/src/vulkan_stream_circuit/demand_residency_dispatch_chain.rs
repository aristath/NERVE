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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanDemandResidencyChainLane {
    Scalar,
    Feedback(usize),
}

fn demand_feedback_chain_keys(
    sequence_variant: u8,
    lane_capacity: usize,
) -> Result<Vec<(u8, VulkanDemandResidencyChainLane)>, VulkanError> {
    if lane_capacity == 0 {
        return Err(VulkanError(
            "demand feedback chain capacity must not be zero".to_string(),
        ));
    }
    Ok((0..lane_capacity)
        .map(|lane| {
            (
                sequence_variant,
                VulkanDemandResidencyChainLane::Feedback(lane),
            )
        })
        .collect())
}

impl VulkanDemandResidencyChainLane {
    fn uses_shared_pipeline_guard(self) -> bool {
        matches!(self, Self::Feedback(_))
    }
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
    command_critical_path_phases: Vec<RuntimeCriticalPathPhase>,
    first_gate_command_index: usize,
    continuation_predicate: Arc<VulkanResidentBuffer>,
    continuation_enabled: Cell<bool>,
    missing_queue: VulkanGpuResidencyMissQueue,
    gates: Vec<VulkanDemandResidencyGateRuntime>,
    feedback_full_sequence: VulkanResidentKernelSequence,
    feedback_resume_sequences: Vec<VulkanResidentKernelSequence>,
    profiled_full_sequence: VulkanResidentKernelSequence,
    profiled_resume_sequences: Vec<VulkanResidentKernelSequence>,
    observed_notification_epoch: Cell<u32>,
    shared_pipeline_guard: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDemandResidencySequencePurpose {
    Unprofiled,
    Profiled,
}

impl VulkanDemandResidencySequencePurpose {
    const fn records_critical_path(self) -> bool {
        matches!(self, Self::Profiled)
    }
}

fn demand_command_critical_path_phase(
    command: VulkanDemandResidencyCommand,
    dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
    prefix_dispatches: &[&VulkanResidentKernelDispatch],
    suffix_dispatches: &[&VulkanResidentKernelDispatch],
) -> Result<RuntimeCriticalPathPhase, VulkanError> {
    match command {
        VulkanDemandResidencyCommand::Prefix(index) => prefix_dispatches
            .get(index)
            .map(|dispatch| {
                dispatch
                    .semantic_label()
                    .map(critical_path_phase_for_semantic_label)
                    .unwrap_or(RuntimeCriticalPathPhase::MixedDeviceCompute)
            })
            .ok_or_else(|| {
                VulkanError(format!(
                    "demand critical-path prefix dispatch {index} is absent"
                ))
            }),
        VulkanDemandResidencyCommand::Dispatch(index) => dispatches
            .get(index)
            .map(|dispatch| {
                critical_path_phase_for_component_operation(&dispatch.component_id, &dispatch.op)
            })
            .ok_or_else(|| {
                VulkanError(format!(
                    "demand critical-path model dispatch {index} is absent"
                ))
            }),
        VulkanDemandResidencyCommand::Gate(_) => Ok(RuntimeCriticalPathPhase::ResidencyGate),
        VulkanDemandResidencyCommand::Suffix(index) => suffix_dispatches
            .get(index)
            .map(|dispatch| {
                dispatch
                    .semantic_label()
                    .map(critical_path_phase_for_semantic_label)
                    .unwrap_or(RuntimeCriticalPathPhase::MixedDeviceCompute)
            })
            .ok_or_else(|| {
                VulkanError(format!(
                    "demand critical-path suffix dispatch {index} is absent"
                ))
            }),
    }
}

fn contiguous_critical_path_regions(
    phases: &[RuntimeCriticalPathPhase],
) -> Vec<RuntimeCriticalPathPhase> {
    let mut regions = Vec::new();
    for phase in phases.iter().copied() {
        if regions.last().copied() != Some(phase) {
            regions.push(phase);
        }
    }
    regions
}

struct VulkanDemandResidencySegment {
    context: VulkanDemandResidencyExecutionContext,
    gate_specs: Vec<VulkanDemandResidencyGateSpec>,
    address_table: Arc<VulkanResidentBuffer>,
    address_table_slot_count: usize,
    pipeline_continuation_predicate: Option<Arc<VulkanResidentBuffer>>,
    chains:
        RefCell<BTreeMap<(u8, VulkanDemandResidencyChainLane), VulkanDemandResidencyDispatchChain>>,
}

fn exact_demand_miss_resource_indices(
    requests: &[VulkanGpuResidencyMissingRequest],
) -> Result<Vec<usize>, VulkanError> {
    let resource_indices = requests
        .iter()
        .map(|request| request.resource_index)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if resource_indices.is_empty() {
        return Err(VulkanError(
            "GPU residency miss notification contains no resources".to_string(),
        ));
    }
    Ok(resource_indices)
}

impl VulkanDemandResidencySegment {
    fn resource_domain_counts(&self) -> impl Iterator<Item = usize> + '_ {
        self.gate_specs
            .iter()
            .map(|gate| gate.address_mapping.resource_count())
    }

    fn feedback_dispatch_count(&self, model_dispatch_count: usize) -> Result<usize, VulkanError> {
        model_dispatch_count
            .checked_add(self.gate_specs.len())
            .ok_or_else(|| VulkanError("demand feedback dispatch count overflowed".to_string()))
    }

    fn feedback_sequence_step_count(
        &self,
        sequence_variant: u8,
        feedback_lane: usize,
        resume_gate_index: Option<usize>,
    ) -> Result<usize, VulkanMountedPlacedResidentKernelDispatchError> {
        self.chains
            .borrow()
            .get(&(
                sequence_variant,
                VulkanDemandResidencyChainLane::Feedback(feedback_lane),
            ))
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "resident feedback demand lane {feedback_lane} was not prepared before snapshot placement"
                ))
            })?
            .sequence_step_count(resume_gate_index)
    }

    fn feedback_dispatch_dimensions(
        &self,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
    ) -> Result<Vec<[u32; 3]>, VulkanError> {
        let mut gates_after_dispatch = BTreeMap::<usize, usize>::new();
        for gate in &self.gate_specs {
            *gates_after_dispatch
                .entry(gate.command_after_dispatch_index)
                .or_default() += 1;
        }
        let capacity = prefix_dispatches
            .len()
            .checked_add(self.feedback_dispatch_count(dispatches.len())?)
            .and_then(|count| count.checked_add(suffix_dispatches.len()))
            .ok_or_else(|| {
                VulkanError("demand feedback dispatch dimensions overflowed".to_string())
            })?;
        let mut dimensions = Vec::with_capacity(capacity);
        dimensions.extend(prefix_dispatches.iter().map(|dispatch| {
            [
                dispatch.workgroup_count_x(),
                dispatch.workgroup_count_y(),
                1,
            ]
        }));
        for dispatch in dispatches {
            dimensions.push([
                dispatch.resident_dispatch.workgroup_count_x(),
                dispatch.resident_dispatch.workgroup_count_y(),
                1,
            ]);
            let gate_count = gates_after_dispatch
                .remove(&dispatch.dispatch_index)
                .unwrap_or(0);
            // The residency gate kernel is deliberately one workgroup: its 64
            // lanes cooperatively resolve the bounded selector output. It
            // still needs its own indirect-control slot so EOS/cancellation
            // can suppress the gate together with the selector that feeds it.
            dimensions.extend(std::iter::repeat_n([1, 1, 1], gate_count));
        }
        if !gates_after_dispatch.is_empty() {
            return Err(VulkanError(format!(
                "demand feedback gates reference dispatches outside their segment: {:?}",
                gates_after_dispatch.keys().collect::<Vec<_>>()
            )));
        }
        dimensions.extend(suffix_dispatches.iter().map(|dispatch| {
            [
                dispatch.workgroup_count_x(),
                dispatch.workgroup_count_y(),
                1,
            ]
        }));
        Ok(dimensions)
    }

    fn pipeline_predicate_for_lane(
        &self,
        lane: VulkanDemandResidencyChainLane,
    ) -> Result<Option<Arc<VulkanResidentBuffer>>, VulkanMountedPlacedResidentKernelDispatchError>
    {
        if !lane.uses_shared_pipeline_guard() {
            return Ok(None);
        }
        self.pipeline_continuation_predicate
            .clone()
            .map(Some)
            .ok_or_else(|| {
                demand_dispatch_error(
                    "resident feedback demand execution has no shared pipeline predicate",
                )
            })
    }

    fn from_segment(
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        schedule: &VulkanPhysicalResidencySchedule,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        context: VulkanDemandResidencyExecutionContext,
        pipeline_continuation_predicate: Option<Arc<VulkanResidentBuffer>>,
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
            .filter(|checkpoint| dispatch_indices.contains(&checkpoint.selection_dispatch_index))
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
                .find(|dispatch| dispatch.dispatch_index == checkpoint.selection_dispatch_index)
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
            demand_dispatch_error("demand-resident segment has no mounted dynamic-resource buffers")
        })?;
        Ok(Some(Self {
            context,
            gate_specs,
            address_table: dynamic_resources.shared_address_table(),
            address_table_slot_count: dynamic_resources.address_table_slot_count(),
            pipeline_continuation_predicate,
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
        let chain_lane = VulkanDemandResidencyChainLane::Scalar;
        let chain_key = (sequence_variant, chain_lane);
        if !self.chains.borrow().contains_key(&chain_key) {
            let chain = VulkanDemandResidencyDispatchChain::new(
                device,
                dispatches,
                &self.gate_specs,
                Arc::clone(&self.address_table),
                self.address_table_slot_count,
                prefix_dispatches,
                suffix_dispatches,
                self.pipeline_predicate_for_lane(chain_lane)?,
            )?;
            self.chains.borrow_mut().insert(chain_key, chain);
        }
        let chains = self.chains.borrow();
        let chain = chains
            .get(&chain_key)
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

    #[allow(clippy::too_many_arguments)]
    fn enqueue_feedback_initial<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        feedback_lane: usize,
        feedback_indirect: &VulkanResidentFeedbackIndirectSequence,
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        let chain_lane = VulkanDemandResidencyChainLane::Feedback(feedback_lane);
        let chain_key = (sequence_variant, chain_lane);
        self.chains
            .borrow()
            .get(&chain_key)
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "resident feedback demand lane {feedback_lane} was not preallocated before submission"
                ))
            })?
            .enqueue_feedback_initial(
                device,
                dispatches,
                control,
                prefix_dispatches,
                suffix_dispatches,
                feedback_indirect,
                snapshot_copies,
                wait_points,
                signal_points,
                signal_completion,
                submission_batch,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_feedback_resume<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        feedback_indirect: &VulkanResidentFeedbackIndirectSequence,
        sequence_variant: u8,
        feedback_lane: usize,
        gate_index: usize,
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        let chain_key = (
            sequence_variant,
            VulkanDemandResidencyChainLane::Feedback(feedback_lane),
        );
        self.chains
            .borrow()
            .get(&chain_key)
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "resident feedback demand lane {feedback_lane} was not preallocated before checkpoint resume"
                ))
            })?
            .enqueue_feedback_resume(
                device,
                dispatches,
                control,
                prefix_dispatches,
                suffix_dispatches,
                feedback_indirect,
                gate_index,
                snapshot_copies,
                wait_points,
                signal_points,
                signal_completion,
                submission_batch,
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_feedback_chains(
        &self,
        device: &VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        lane_capacity: usize,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        for (chain_key, chain_lane) in demand_feedback_chain_keys(sequence_variant, lane_capacity)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?
            .into_iter()
            .map(|key @ (_, lane)| (key, lane))
        {
            if self.chains.borrow().contains_key(&chain_key) {
                continue;
            }
            let chain = VulkanDemandResidencyDispatchChain::new(
                device,
                dispatches,
                &self.gate_specs,
                Arc::clone(&self.address_table),
                self.address_table_slot_count,
                prefix_dispatches,
                suffix_dispatches,
                self.pipeline_predicate_for_lane(chain_lane)?,
            )?;
            self.chains.borrow_mut().insert(chain_key, chain);
        }
        Ok(())
    }

    fn feedback_lane_has_pending_miss(
        &self,
        sequence_variant: u8,
        feedback_lane: usize,
    ) -> Result<bool, VulkanMountedPlacedResidentKernelDispatchError> {
        self.chains
            .borrow()
            .get(&(
                sequence_variant,
                VulkanDemandResidencyChainLane::Feedback(feedback_lane),
            ))
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "resident feedback demand lane {feedback_lane} was not prepared"
                ))
            })?
            .has_pending_miss()
    }

    fn resolve_feedback_lane_miss(
        &self,
        device: &VulkanComputeDevice,
        sequence_variant: u8,
        feedback_lane: usize,
    ) -> Result<Option<(usize, Vec<usize>)>, VulkanMountedPlacedResidentKernelDispatchError> {
        self.chains
            .borrow()
            .get(&(
                sequence_variant,
                VulkanDemandResidencyChainLane::Feedback(feedback_lane),
            ))
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "resident feedback demand lane {feedback_lane} was not prepared"
                ))
            })?
            .resolve_pending_miss_without_resume(device, &self.context)
    }
}

impl VulkanDemandResidencyDispatchChain {
    fn sequence_step_count(
        &self,
        resume_gate_index: Option<usize>,
    ) -> Result<usize, VulkanMountedPlacedResidentKernelDispatchError> {
        let start_command_index = resume_gate_index
            .map(|gate_index| {
                self.gates
                    .get(gate_index)
                    .map(|gate| gate.command_index)
                    .ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "demand snapshot resume gate {gate_index} is out of bounds"
                        ))
                    })
            })
            .transpose()?
            .unwrap_or(0);
        demand_feedback_sequence_step_count(self.commands.len(), start_command_index)
    }

    fn new(
        device: &VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        gate_specs: &[VulkanDemandResidencyGateSpec],
        address_table: Arc<VulkanResidentBuffer>,
        address_table_slot_count: usize,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        pipeline_continuation_predicate: Option<Arc<VulkanResidentBuffer>>,
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
            if let Some(gate_indices) = gates_after_dispatch.remove(&dispatch.dispatch_index) {
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
        let command_critical_path_phases = commands
            .iter()
            .copied()
            .map(|command| {
                demand_command_critical_path_phase(
                    command,
                    dispatches,
                    prefix_dispatches,
                    suffix_dispatches,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let first_gate_command_index = commands
            .iter()
            .position(|command| matches!(command, VulkanDemandResidencyCommand::Gate(_)))
            .expect("demand segment has at least one gate");
        if first_gate_command_index + 1 == commands.len() {
            return Err(demand_dispatch_error(
                "demand gate has no selected computation after it",
            ));
        }
        let shared_pipeline_guard = pipeline_continuation_predicate.is_some();
        let continuation_predicate = match pipeline_continuation_predicate {
            Some(predicate) => predicate,
            None => {
                let predicate = Arc::new(
                    device
                        .create_conditional_resident_buffer(size_of::<u32>())
                        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
                );
                predicate
                    .write_bytes(&1u32.to_le_bytes())
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                predicate
            }
        };
        let missing_capacity = gate_specs
            .iter()
            .map(|gate| gate.selection_count)
            .max()
            .expect("demand segment gates are non-empty");
        let missing_queue = VulkanGpuResidencyMissQueue::new(device, missing_capacity)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let gate_shader = vulkan_gpu_residency_gate_spirv_words()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let mut gates = Vec::with_capacity(gate_specs.len());
        for (gate_index, spec) in gate_specs.iter().enumerate() {
            let command_index = commands
                .iter()
                .position(|command| *command == VulkanDemandResidencyCommand::Gate(gate_index))
                .expect("expanded command chain contains every gate");
            let checkpoint_tag = u32::try_from(gate_index + 1)
                .map_err(|_| demand_dispatch_error("demand gate count exceeds u32"))?;
            let address_mapping = match &spec.address_mapping {
                VulkanCompiledSelectorAddressMapping::GroupTable {
                    resource_address_slots,
                    resource_address_slot_offsets,
                } => VulkanGpuResidencyAddressMapping::GroupTable {
                    resource_address_slots: resource_address_slots.clone(),
                    resource_address_slot_offsets: resource_address_slot_offsets.clone(),
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
                None,
                VulkanGpuResidencyGateConfig {
                    maximum_selection_count: spec.selection_count,
                    selection_count_per_lane: spec.selection_count,
                    selection_lane_stride_words: spec.selection_count,
                    selection_index_shift: spec.selection_index_shift,
                    selection_index_mask: spec.selection_index_mask,
                    address_mapping,
                    owned_resource_indices: None,
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
        let feedback_full_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let feedback_resume_sequences = gates
            .iter()
            .map(|_| device.create_resident_kernel_sequence())
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let profiled_full_sequence = device
            .create_critical_path_timestamped_resident_kernel_sequence(
                contiguous_critical_path_regions(&command_critical_path_phases).len(),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let profiled_resume_sequences = gates
            .iter()
            .map(|gate| {
                device.create_critical_path_timestamped_resident_kernel_sequence(
                    contiguous_critical_path_regions(
                        &command_critical_path_phases[gate.command_index..],
                    )
                    .len(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        Ok(Self {
            commands,
            command_critical_path_phases,
            first_gate_command_index,
            continuation_predicate,
            continuation_enabled: Cell::new(true),
            missing_queue,
            gates,
            feedback_full_sequence,
            feedback_resume_sequences,
            profiled_full_sequence,
            profiled_resume_sequences,
            observed_notification_epoch: Cell::new(0),
            shared_pipeline_guard,
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
        context
            .store
            .ensure_execution_headroom(device)
            .map_err(|error| {
                demand_dispatch_error(format!(
                    "failed to establish compiled-resource execution headroom: {error}"
                ))
            })?;
        self.run_from_gate(
            device,
            None,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            wait_points,
            &[],
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
                // A Vulkan semaphore signal is unconditional even when every
                // producer command was skipped by a residency predicate. Edge
                // publication is therefore the commit record for a successful
                // traversal, not part of a speculative traversal that may
                // miss. Queue order plus the completed sequence wait makes
                // this bridge visible only after the real activation exists.
                if !signal_points.is_empty() {
                    device
                        .submit_timeline_semaphore_bridge(&[], signal_points)
                        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                }
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
                .record_gpu_gate_misses(&gate.selector_id, missing.requests.len())
                .map_err(|error| {
                    demand_dispatch_error(format!(
                        "failed to record GPU residency misses for selector {:?}: {error}",
                        gate.selector_id
                    ))
                })?;
            // The queue is the immutable record of the gate invocation that
            // faulted. The selector buffer is shared working memory and a
            // later feedback lane may already have overwritten it by the time
            // the host resolves this checkpoint. Loading from that mutable
            // selector can materialize the wrong experts and make the same
            // checkpoint fault again. Consume only the exact resource indices
            // published by the gate.
            let resource_indices = exact_demand_miss_resource_indices(&missing.requests)
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
            let _ = context
                .store
                .load_selector_resources_for_resume(
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
    fn enqueue_feedback_initial<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        feedback_indirect: &VulkanResidentFeedbackIndirectSequence,
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        self.with_prepared_steps(
            None,
            VulkanDemandResidencySequencePurpose::Unprofiled,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            Some(feedback_indirect),
            |sequence, steps| {
                device
                    .record_resident_kernel_sequence_with_snapshot_copies(
                        sequence,
                        steps,
                        snapshot_copies,
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                submission_batch
                    .enqueue_recorded_sequence(
                        device,
                        sequence,
                        wait_points,
                        signal_points,
                        signal_completion,
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_feedback_resume<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        feedback_indirect: &VulkanResidentFeedbackIndirectSequence,
        gate_index: usize,
        snapshot_copies: &[VulkanResidentKernelSequenceSnapshotCopy<'_>],
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        self.continuation_enabled.set(true);
        self.with_prepared_steps(
            Some(gate_index),
            VulkanDemandResidencySequencePurpose::Unprofiled,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            Some(feedback_indirect),
            |sequence, steps| {
                device
                    .record_resident_kernel_sequence_with_snapshot_copies(
                        sequence,
                        steps,
                        snapshot_copies,
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                submission_batch
                    .enqueue_recorded_sequence(
                        device,
                        sequence,
                        wait_points,
                        signal_points,
                        signal_completion,
                    )
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
            },
        )
    }

    fn has_pending_miss(&self) -> Result<bool, VulkanMountedPlacedResidentKernelDispatchError> {
        self.missing_queue
            .notification_epoch()
            .map(|epoch| epoch != self.observed_notification_epoch.get())
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
    }

    fn resolve_pending_miss_without_resume(
        &self,
        device: &VulkanComputeDevice,
        context: &VulkanDemandResidencyExecutionContext,
    ) -> Result<Option<(usize, Vec<usize>)>, VulkanMountedPlacedResidentKernelDispatchError> {
        if !self.has_pending_miss()? {
            return Ok(None);
        }
        self.continuation_enabled.set(false);
        let missing = self
            .missing_queue
            .snapshot()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        if missing.overflowed || missing.requests.is_empty() {
            return Err(demand_dispatch_error(format!(
                "GPU feedback residency gate reported an invalid or overflowing miss queue: epoch={}, published={}, consumed={}, overflowed={}, requests={}",
                missing.notification_epoch,
                missing.published_count,
                missing.consumed_count,
                missing.overflowed,
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
                "one resident feedback traversal reported misses at multiple checkpoints: {checkpoint_tags:?}"
            )));
        }
        let checkpoint_tag = *checkpoint_tags
            .first()
            .expect("one feedback checkpoint tag was validated");
        let gate_index = self
            .gates
            .iter()
            .position(|gate| gate.checkpoint_tag == checkpoint_tag)
            .ok_or_else(|| {
                demand_dispatch_error(format!(
                    "GPU feedback residency miss references unknown checkpoint tag {checkpoint_tag}"
                ))
            })?;
        let gate = &self.gates[gate_index];
        context
            .store
            .record_gpu_gate_misses(&gate.selector_id, missing.requests.len())
            .map_err(|error| {
                demand_dispatch_error(format!(
                    "failed to record GPU feedback residency misses for selector {:?}: {error}",
                    gate.selector_id
                ))
            })?;
        let resource_indices = exact_demand_miss_resource_indices(&missing.requests)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let _ = context
            .store
            .load_selector_resources_for_resume(
                device,
                &gate.selector_id,
                &resource_indices,
                context.owner.clone(),
            )
            .map_err(|error| {
                demand_dispatch_error(format!(
                    "failed to load feedback selector {:?} resources {resource_indices:?} at checkpoint {:?}: {error}",
                    gate.selector_id, gate.checkpoint_id
                ))
            })?;
        self.missing_queue
            .acknowledge_through(missing.published_count)
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        self.observed_notification_epoch
            .set(missing.notification_epoch);
        Ok(Some((gate_index, resource_indices)))
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
        let start_command_index = resume_gate_index
            .and_then(|gate_index| self.gates.get(gate_index))
            .map(|gate| gate.command_index)
            .unwrap_or_default();
        let region_phases = contiguous_critical_path_regions(
            &self.command_critical_path_phases[start_command_index..],
        );
        let sequence_purpose = if runtime_critical_path_device_detail_enabled() {
            VulkanDemandResidencySequencePurpose::Profiled
        } else {
            VulkanDemandResidencySequencePurpose::Unprofiled
        };
        self.with_prepared_steps(
            resume_gate_index,
            sequence_purpose,
            dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            None,
            |sequence, steps| {
                let execution_result = if input_copies.is_empty() && post_copies.is_empty() {
                    device
                        .record_resident_kernel_sequence(sequence, steps)
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
                } else {
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
                            steps,
                            &snapshot_copies,
                        )
                        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)
                };
                execution_result?;
                if sequence_purpose.records_critical_path() {
                    let region_durations = device
                        .read_recorded_resident_kernel_critical_path_region_durations_ns(sequence)
                        .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
                    if region_durations.len() != region_phases.len() {
                        return Err(demand_dispatch_error(format!(
                            "demand critical-path sequence reported {} region durations for {} semantic regions",
                            region_durations.len(),
                            region_phases.len(),
                        )));
                    }
                    for (phase, duration_ns) in region_phases
                        .iter()
                        .copied()
                        .zip(region_durations)
                    {
                        record_runtime_critical_path_device_duration(phase, duration_ns);
                    }
                }
                Ok(())
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_prepared_steps<T, F>(
        &self,
        resume_gate_index: Option<usize>,
        sequence_purpose: VulkanDemandResidencySequencePurpose,
        dispatches: &[VulkanMountedPlacedResidentComponentDispatch],
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        feedback_indirect: Option<&VulkanResidentFeedbackIndirectSequence>,
        use_steps: F,
    ) -> Result<T, VulkanMountedPlacedResidentKernelDispatchError>
    where
        F: FnOnce(
            &VulkanResidentKernelSequence,
            &[VulkanResidentKernelSequenceStep<'_>],
        ) -> Result<T, VulkanMountedPlacedResidentKernelDispatchError>,
    {
        let model_push_constants = dispatches
            .iter()
            .map(|dispatch| stream_control_push_constant_bytes(&dispatch.push_constants, control))
            .collect::<Result<Vec<_>, _>>()?;
        let (start_command_index, direct_gate_command_index, sequence) = match resume_gate_index {
            Some(gate_index) => {
                let gate = self.gates.get(gate_index).ok_or_else(|| {
                    demand_dispatch_error(format!(
                        "demand resume gate {gate_index} is out of bounds"
                    ))
                })?;
                (
                    gate.command_index,
                    gate.command_index,
                    match sequence_purpose {
                        VulkanDemandResidencySequencePurpose::Unprofiled => {
                            &self.feedback_resume_sequences
                        }
                        VulkanDemandResidencySequencePurpose::Profiled => {
                            &self.profiled_resume_sequences
                        }
                    }
                    .get(gate_index)
                    .ok_or_else(|| {
                        demand_dispatch_error(format!(
                            "demand resume sequence {gate_index} is absent"
                        ))
                    })?,
                )
            }
            None => (
                0,
                self.first_gate_command_index,
                match sequence_purpose {
                    VulkanDemandResidencySequencePurpose::Unprofiled => {
                        &self.feedback_full_sequence
                    }
                    VulkanDemandResidencySequencePurpose::Profiled => &self.profiled_full_sequence,
                },
            ),
        };
        let gate_push_constants = self
            .gates
            .iter()
            .map(|gate| {
                gate.gate.push_constants(
                    gate.selection_count,
                    gate.checkpoint_tag,
                    gate.command_index == direct_gate_command_index,
                    gate.command_index == direct_gate_command_index,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        let conditional_regions = demand_dispatch_conditional_regions(
            &self.commands,
            start_command_index,
            direct_gate_command_index,
            self.shared_pipeline_guard,
            resume_gate_index.is_some(),
        )?;
        let mut steps = Vec::with_capacity(self.commands.len() - start_command_index);
        let mut critical_path_region_index = 0u32;
        let mut previous_critical_path_phase = None;
        let mut feedback_dispatch_index = feedback_indirect
            .map(|indirect| {
                demand_feedback_indirect_command_range(
                    indirect.byte_offsets.len(),
                    start_command_index,
                )
                .map(|range| range.start)
            })
            .transpose()?
            .unwrap_or(0);
        for (command_index, command) in self
            .commands
            .iter()
            .copied()
            .enumerate()
            .skip(start_command_index)
        {
            let critical_path_phase = *self
                .command_critical_path_phases
                .get(command_index)
                .ok_or_else(|| {
                    demand_dispatch_error(format!(
                        "demand command {command_index} has no critical-path phase"
                    ))
                })?;
            if previous_critical_path_phase.is_some_and(|previous| previous != critical_path_phase)
            {
                critical_path_region_index =
                    critical_path_region_index.checked_add(1).ok_or_else(|| {
                        demand_dispatch_error("demand critical-path region index overflowed")
                    })?;
            }
            previous_critical_path_phase = Some(critical_path_phase);
            let (dispatch, push_constants): (&VulkanResidentKernelDispatch, &[u8]) = match command {
                VulkanDemandResidencyCommand::Prefix(index) => (
                    *prefix_dispatches.get(index).ok_or_else(|| {
                        demand_dispatch_error(format!("demand prefix dispatch {index} disappeared"))
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
                        demand_dispatch_error(format!("demand suffix dispatch {index} disappeared"))
                    })?,
                    &[],
                ),
            };
            let step = if let Some(indirect) = feedback_indirect {
                let byte_offset = *indirect
                    .byte_offsets
                    .get(feedback_dispatch_index)
                    .ok_or_else(|| {
                        demand_dispatch_error(
                            "resident feedback indirect sequence is shorter than its demand chain",
                        )
                    })?;
                feedback_dispatch_index += 1;
                VulkanResidentKernelSequenceStep::new_indirect(
                    dispatch,
                    push_constants,
                    &indirect.buffer,
                    byte_offset,
                )
                .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?
            } else {
                VulkanResidentKernelSequenceStep::new(dispatch, push_constants)
            };
            let step = if sequence_purpose.records_critical_path() {
                step.with_critical_path_region(critical_path_region_index)
            } else {
                step
            };
            let conditional_region = conditional_regions
                .get(command_index - start_command_index)
                .copied()
                .flatten();
            steps.push(match conditional_region {
                Some(region_id) => step
                    .with_condition(&self.continuation_predicate, 0, false, region_id)
                    .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?,
                None => step,
            });
        }
        if let Some(indirect) = feedback_indirect
            && feedback_dispatch_index != indirect.byte_offsets.len()
        {
            return Err(demand_dispatch_error(
                "resident feedback indirect sequence is longer than its demand chain",
            ));
        }
        use_steps(sequence, &steps)
    }
}

fn demand_feedback_indirect_command_range(
    command_count: usize,
    start_command_index: usize,
) -> Result<std::ops::Range<usize>, VulkanMountedPlacedResidentKernelDispatchError> {
    if start_command_index >= command_count {
        return Err(demand_dispatch_error(format!(
            "demand feedback indirect start {start_command_index} is outside its {command_count}-command sequence"
        )));
    }
    Ok(start_command_index..command_count)
}

fn demand_feedback_sequence_step_count(
    command_count: usize,
    start_command_index: usize,
) -> Result<usize, VulkanMountedPlacedResidentKernelDispatchError> {
    demand_feedback_indirect_command_range(command_count, start_command_index)
        .map(|range| range.len())
}

fn demand_dispatch_conditional_regions(
    commands: &[VulkanDemandResidencyCommand],
    start_command_index: usize,
    direct_gate_command_index: usize,
    shared_pipeline_guard: bool,
    resuming: bool,
) -> Result<Vec<Option<u32>>, VulkanMountedPlacedResidentKernelDispatchError> {
    if direct_gate_command_index < start_command_index
        || direct_gate_command_index >= commands.len()
        || !matches!(
            commands[direct_gate_command_index],
            VulkanDemandResidencyCommand::Gate(_)
        )
    {
        return Err(demand_dispatch_error(format!(
            "demand direct gate {direct_gate_command_index} is not a gate in the remaining command range {start_command_index}..{}",
            commands.len(),
        )));
    }
    if resuming && direct_gate_command_index != start_command_index {
        return Err(demand_dispatch_error(
            "demand resume must start at its direct gate",
        ));
    }
    let commands = commands.get(start_command_index..).ok_or_else(|| {
        demand_dispatch_error("demand conditional layout starts outside its command list")
    })?;
    let mut region_id = 1u32;
    let mut previous_was_conditional_gate = false;
    let mut resumed_span_reached_next_gate = false;
    let mut regions = Vec::with_capacity(commands.len());
    for (offset, command) in commands.iter().copied().enumerate() {
        let command_index = start_command_index + offset;
        let conditional = if resuming {
            resumed_span_reached_next_gate
        } else {
            command_index > direct_gate_command_index
                || (shared_pipeline_guard && command_index <= direct_gate_command_index)
        };
        if conditional && previous_was_conditional_gate {
            region_id = region_id.checked_add(1).ok_or_else(|| {
                demand_dispatch_error("demand conditional region count exceeds u32")
            })?;
        }
        regions.push(conditional.then_some(region_id));
        previous_was_conditional_gate =
            conditional && matches!(command, VulkanDemandResidencyCommand::Gate(_));
        if resuming
            && command_index > direct_gate_command_index
            && matches!(command, VulkanDemandResidencyCommand::Gate(_))
        {
            // The host loaded and pinned the direct checkpoint before this
            // sequence was submitted. Its selected work and the next gate are
            // therefore safe to execute without consulting the stale shared
            // predicate left by the miss. That next gate becomes the new
            // conditional boundary for all later work.
            resumed_span_reached_next_gate = true;
        }
    }
    Ok(regions)
}

fn demand_dispatch_error(
    message: impl Into<String>,
) -> VulkanMountedPlacedResidentKernelDispatchError {
    VulkanMountedPlacedResidentKernelDispatchError::Vulkan(VulkanError(message.into()))
}
