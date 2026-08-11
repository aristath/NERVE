struct VulkanResidentDemandFeedbackState {
    predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    completion_predicate: Arc<VulkanResidentBuffer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDemandFeedbackCheckpoint {
    feedback_lane: usize,
    slice_index: usize,
    segment_index: usize,
    gate_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanPlacedDemandFeedbackTickResume {
    feedback_lane: usize,
    schedule_start_turn_index: usize,
    next_stage_indices: Vec<usize>,
    target_slice_index: usize,
    target_segment_start_stage_index: usize,
    gate_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDemandFeedbackResumePlan {
    schedule_start_turn_index: usize,
    next_stage_indices: Vec<usize>,
}

fn demand_feedback_resolution_bound(
    tick_count: usize,
    resource_domain_counts: impl IntoIterator<Item = usize>,
) -> Result<usize, VulkanError> {
    if tick_count == 0 {
        return Err(VulkanError(
            "demand feedback resolution bound requires at least one tick".to_string(),
        ));
    }
    let resource_count_per_tick = resource_domain_counts
        .into_iter()
        .try_fold(0usize, |total, resource_count| {
            if resource_count == 0 {
                return Err(VulkanError(
                    "demand feedback checkpoint has an empty resource domain".to_string(),
                ));
            }
            total.checked_add(resource_count).ok_or_else(|| {
                VulkanError(
                    "demand feedback resource-domain count overflowed".to_string(),
                )
            })
        })?;
    if resource_count_per_tick == 0 {
        return Err(VulkanError(
            "demand feedback has no resource domains".to_string(),
        ));
    }
    tick_count
        .checked_mul(resource_count_per_tick)
        .ok_or_else(|| {
            VulkanError("demand feedback resolution bound overflowed".to_string())
        })
}

fn record_demand_feedback_resolution(
    resolved: &mut BTreeMap<VulkanDemandFeedbackCheckpoint, BTreeSet<usize>>,
    checkpoint: VulkanDemandFeedbackCheckpoint,
    resource_indices: &[usize],
) -> Result<usize, VulkanError> {
    if resource_indices.is_empty() {
        return Err(VulkanError(format!(
            "demand feedback checkpoint {checkpoint:?} resolved no resources"
        )));
    }
    let prior = resolved.entry(checkpoint).or_default();
    let repeated = resource_indices
        .iter()
        .copied()
        .filter(|resource_index| prior.contains(resource_index))
        .collect::<Vec<_>>();
    if !repeated.is_empty() {
        return Err(VulkanError(format!(
            "demand feedback checkpoint {checkpoint:?} missed resources {repeated:?} again after they were loaded; previously_resolved={prior:?}; current_resources={resource_indices:?}"
        )));
    }
    prior.extend(resource_indices.iter().copied());
    Ok(resource_indices.len())
}

struct VulkanDemandFeedbackStageTopology {
    node_offsets: Vec<usize>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
}

impl VulkanDemandFeedbackStageTopology {
    fn from_tick_plans(
        tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    ) -> Result<Self, VulkanError> {
        let mut node_offsets = Vec::with_capacity(tick_plans.len());
        let mut node_count = 0usize;
        for plan in tick_plans {
            node_offsets.push(node_count);
            node_count = node_count.checked_add(plan.stages.len()).ok_or_else(|| {
                VulkanError("demand feedback stage topology overflowed".to_string())
            })?;
        }
        let mut outgoing = vec![Vec::new(); node_count];
        let mut incoming = vec![Vec::new(); node_count];
        let mut publishes = BTreeMap::<(usize, String, String), usize>::new();
        let mut receives = BTreeMap::<(usize, String, String), usize>::new();
        for (device_index, plan) in tick_plans.iter().enumerate() {
            for (stage_index, stage) in plan.stages.iter().enumerate() {
                let node = node_offsets[device_index] + stage_index;
                if stage_index > 0 {
                    Self::add_edge(node - 1, node, &mut outgoing, &mut incoming);
                }
                match stage {
                    VulkanMountedPlacedStreamTickStage::PublishEdge {
                        edge_index,
                        remote_device_id,
                        ..
                    } => {
                        let key = (
                            *edge_index,
                            plan.device_id.clone(),
                            remote_device_id.clone(),
                        );
                        if publishes.insert(key.clone(), node).is_some() {
                            return Err(VulkanError(format!(
                                "demand feedback topology repeats published edge {key:?}"
                            )));
                        }
                    }
                    VulkanMountedPlacedStreamTickStage::ReceiveEdge {
                        edge_index,
                        remote_device_id,
                        ..
                    } => {
                        let key = (
                            *edge_index,
                            remote_device_id.clone(),
                            plan.device_id.clone(),
                        );
                        if receives.insert(key.clone(), node).is_some() {
                            return Err(VulkanError(format!(
                                "demand feedback topology repeats received edge {key:?}"
                            )));
                        }
                    }
                    VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {}
                }
            }
        }
        if publishes.keys().collect::<Vec<_>>() != receives.keys().collect::<Vec<_>>() {
            return Err(VulkanError(
                "demand feedback topology has unmatched placed edges".to_string(),
            ));
        }
        for (key, source) in publishes {
            let destination = receives
                .get(&key)
                .copied()
                .expect("matching placed receive edge was validated");
            Self::add_edge(source, destination, &mut outgoing, &mut incoming);
        }
        Ok(Self {
            node_offsets,
            outgoing,
            incoming,
        })
    }

    fn add_edge(
        source: usize,
        destination: usize,
        outgoing: &mut [Vec<usize>],
        incoming: &mut [Vec<usize>],
    ) {
        if !outgoing[source].contains(&destination) {
            outgoing[source].push(destination);
            incoming[destination].push(source);
        }
    }

    fn node(&self, device_index: usize, stage_index: usize) -> Result<usize, VulkanError> {
        let start = self.node_offsets.get(device_index).copied().ok_or_else(|| {
            VulkanError(format!(
                "demand feedback resume device {device_index} is out of bounds"
            ))
        })?;
        let end = self
            .node_offsets
            .get(device_index + 1)
            .copied()
            .unwrap_or(self.outgoing.len());
        let node = start.checked_add(stage_index).ok_or_else(|| {
            VulkanError("demand feedback resume stage overflowed".to_string())
        })?;
        if node >= end {
            return Err(VulkanError(format!(
                "demand feedback resume stage {stage_index} is out of bounds for device {device_index}"
            )));
        }
        Ok(node)
    }

    fn is_total_ordered(&self) -> bool {
        let mut indegree = self.incoming.iter().map(Vec::len).collect::<Vec<_>>();
        let mut ready = indegree
            .iter()
            .enumerate()
            .filter_map(|(node, degree)| (*degree == 0).then_some(node))
            .collect::<BTreeSet<_>>();
        let mut visited = 0usize;
        while let Some(node) = ready.pop_first() {
            if !ready.is_empty() {
                return false;
            }
            visited += 1;
            for destination in &self.outgoing[node] {
                indegree[*destination] -= 1;
                if indegree[*destination] == 0 {
                    ready.insert(*destination);
                }
            }
        }
        visited == self.outgoing.len()
    }

    fn ancestors(&self, node: usize) -> BTreeSet<usize> {
        let mut ancestors = BTreeSet::new();
        let mut pending = self.incoming[node].clone();
        while let Some(parent) = pending.pop() {
            if ancestors.insert(parent) {
                pending.extend(self.incoming[parent].iter().copied());
            }
        }
        ancestors
    }
}

fn demand_feedback_resume_plan(
    tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    target_device_index: usize,
    target_stage_index: usize,
) -> Result<VulkanDemandFeedbackResumePlan, VulkanError> {
    let topology = VulkanDemandFeedbackStageTopology::from_tick_plans(tick_plans)?;
    if !topology.is_total_ordered() {
        return Err(VulkanError(
            "demand feedback checkpoint has an independent parallel branch without explicit GPU progress markers"
                .to_string(),
        ));
    }
    let target = topology.node(target_device_index, target_stage_index)?;
    if !matches!(
        tick_plans[target_device_index].stages[target_stage_index],
        VulkanMountedPlacedStreamTickStage::Dispatch { .. }
    ) {
        return Err(VulkanError(
            "demand feedback checkpoint resume must start at a dispatch segment".to_string(),
        ));
    }
    let ancestors = topology.ancestors(target);
    let next_stage_indices = tick_plans
        .iter()
        .enumerate()
        .map(|(device_index, plan)| {
            let start = topology.node_offsets[device_index];
            plan.stages
                .iter()
                .enumerate()
                .take_while(|(stage_index, _)| ancestors.contains(&(start + stage_index)))
                .count()
        })
        .collect::<Vec<_>>();
    if next_stage_indices[target_device_index] != target_stage_index {
        return Err(VulkanError(
            "demand feedback checkpoint does not follow a contiguous causal prefix".to_string(),
        ));
    }
    let schedule_start_turn_index = demand_feedback_resume_turn_index(
        tick_plans,
        target_device_index,
        target_stage_index,
    )?;
    Ok(VulkanDemandFeedbackResumePlan {
        schedule_start_turn_index,
        next_stage_indices,
    })
}

fn demand_feedback_resume_turn_index(
    tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    target_device_index: usize,
    target_stage_index: usize,
) -> Result<usize, VulkanError> {
    let mut next_stage_indices = vec![0usize; tick_plans.len()];
    let mut ready_edges = BTreeSet::<VulkanPlacedEdgePacketKey>::new();
    let mut turn_index = 0usize;
    loop {
        let mut progressed = false;
        for (device_index, plan) in tick_plans.iter().enumerate() {
            while next_stage_indices[device_index] < plan.stages.len() {
                if device_index == target_device_index
                    && next_stage_indices[device_index] == target_stage_index
                {
                    return Ok(turn_index);
                }
                match &plan.stages[next_stage_indices[device_index]] {
                    VulkanMountedPlacedStreamTickStage::ReceiveEdge {
                        edge_index,
                        remote_device_id,
                        ..
                    } => {
                        let key = VulkanPlacedEdgePacketKey {
                            edge_index: *edge_index,
                            from_device_id: remote_device_id.clone(),
                            to_device_id: plan.device_id.clone(),
                        };
                        if !ready_edges.remove(&key) {
                            break;
                        }
                    }
                    VulkanMountedPlacedStreamTickStage::PublishEdge {
                        edge_index,
                        remote_device_id,
                        ..
                    } => {
                        ready_edges.insert(VulkanPlacedEdgePacketKey {
                            edge_index: *edge_index,
                            from_device_id: plan.device_id.clone(),
                            to_device_id: remote_device_id.clone(),
                        });
                    }
                    VulkanMountedPlacedStreamTickStage::Dispatch { .. } => {}
                }
                next_stage_indices[device_index] += 1;
                progressed = true;
            }
        }
        if !progressed {
            return Err(VulkanError(
                "demand feedback resume target is unreachable in the placed activation topology"
                    .to_string(),
            ));
        }
        turn_index = turn_index.checked_add(1).ok_or_else(|| {
            VulkanError("demand feedback resume turn index overflowed".to_string())
        })?;
    }
}

fn demand_feedback_continuation_lanes(
    tick_count: usize,
    resume_lane: usize,
) -> Result<std::ops::Range<usize>, VulkanError> {
    if tick_count == 0 || resume_lane >= tick_count {
        return Err(VulkanError(format!(
            "demand feedback resume lane {resume_lane} exceeds window width {tick_count}"
        )));
    }
    Ok(resume_lane..tick_count)
}

fn create_demand_feedback_pipeline_predicates<'a, F, E>(
    model: &VulkanResidentInProcessPlacedModelPackage,
    device_for: &F,
) -> Result<Option<BTreeMap<String, Arc<VulkanResidentBuffer>>>, VulkanError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, E>,
    E: Display,
{
    if !model.resource_residency_policy.is_demand_loaded()
        || !model
            .device_slices
            .iter()
            .any(|slice| !slice.physical_residency_schedule().checkpoints.is_empty())
    {
        return Ok(None);
    }
    let device_ids = model
        .device_slices
        .iter()
        .map(|slice| slice.device_id.clone())
        .collect::<Vec<_>>();
    if device_ids.is_empty() {
        return Err(VulkanError(
            "demand-resident feedback has no placed devices".to_string(),
        ));
    }
    let resolved_devices = device_ids
        .iter()
        .map(|device_id| {
            device_for(device_id).map_err(|error| {
                VulkanError(format!(
                    "demand feedback device {device_id:?} resolution failed: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let buffers = if let Some((owner, peers)) = resolved_devices.split_first() {
        if peers.is_empty() {
            vec![Arc::new(
                owner.create_conditional_resident_buffer(size_of::<u32>())?,
            )]
        } else {
            owner
                .create_shared_conditional_resident_buffers(peers, size_of::<u32>())?
                .buffers
        }
    } else {
        unreachable!("non-empty demand feedback devices have an owner")
    };
    if buffers.len() != device_ids.len() {
        return Err(VulkanError(format!(
            "demand feedback predicate produced {} device views for {} placed devices",
            buffers.len(),
            device_ids.len()
        )));
    }
    let mut predicates_by_device = BTreeMap::new();
    for (device_id, buffer) in device_ids.into_iter().zip(buffers) {
        if predicates_by_device.insert(device_id.clone(), buffer).is_some() {
            return Err(VulkanError(format!(
                "demand feedback repeats placed device {device_id:?}"
            )));
        }
    }
    write_shared_device_predicate_views(predicates_by_device.values(), true)?;
    Ok(Some(predicates_by_device))
}

fn write_shared_device_predicate_views<'a>(
    predicates: impl IntoIterator<Item = &'a Arc<VulkanResidentBuffer>>,
    enabled: bool,
) -> Result<(), VulkanError> {
    let value = u32::from(enabled).to_le_bytes();
    let mut written_views = BTreeSet::new();
    for predicate in predicates {
        if written_views.insert(Arc::as_ptr(predicate) as usize) {
            predicate.write_bytes(&value)?;
        }
    }
    if written_views.is_empty() {
        return Err(VulkanError(
            "shared device predicate is absent".to_string(),
        ));
    }
    Ok(())
}

impl VulkanResidentDemandFeedbackState {
    fn new(
        predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        output_device_id: &str,
    ) -> Result<Self, VulkanError> {
        if predicates_by_device.len() != device_slices.len()
            || device_slices
                .iter()
                .any(|slice| !predicates_by_device.contains_key(&slice.device_id))
        {
            return Err(VulkanError(
                "demand feedback predicates do not cover every placed device".to_string(),
            ));
        }
        let checkpoint_count_per_tick = device_slices
            .iter()
            .flat_map(|slice| &slice.resident_execution_plan.dispatch_segments)
            .filter_map(|segment| segment.demand_residency.as_ref())
            .map(|segment| segment.gate_specs.len())
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| {
                VulkanError("demand feedback checkpoint count overflowed".to_string())
            })?;
        if checkpoint_count_per_tick == 0 {
            return Err(VulkanError(
                "demand feedback state contains no residency checkpoints".to_string(),
            ));
        }
        let completion_predicate = predicates_by_device
            .get(output_device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanError(format!(
                    "demand feedback predicates do not contain output device {output_device_id:?}"
                ))
            })?;
        Ok(Self {
            predicates_by_device,
            completion_predicate,
        })
    }

    fn reset_pipeline_predicate(&self) -> Result<(), VulkanError> {
        write_shared_device_predicate_views(self.predicates_by_device.values(), true)
    }

    fn pipeline_predicate_diagnostic(&self) -> Result<BTreeMap<String, u32>, VulkanError> {
        self.predicates_by_device
            .iter()
            .map(|(device_id, predicate)| {
                let bytes = predicate.read_bytes(size_of::<u32>())?;
                let value = u32::from_le_bytes(bytes.try_into().map_err(|_| {
                    VulkanError(format!(
                        "demand feedback predicate on {device_id:?} did not contain one u32"
                    ))
                })?);
                Ok((device_id.clone(), value))
            })
            .collect()
    }

    fn terminal_fault_publication_copy<'a>(
        &'a self,
        control: &'a VulkanResidentFeedbackControlPlane,
    ) -> Result<VulkanResidentBufferRangeCopy<'a>, VulkanError> {
        VulkanResidentBufferRangeCopy::new(
            &self.completion_predicate,
            control.fault_publication_buffer()?,
            0,
            VULKAN_FEEDBACK_CONTINUATION_WORD_OFFSET * size_of::<u32>(),
            size_of::<u32>(),
        )
    }

    fn resolution_bound(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        tick_count: usize,
    ) -> Result<usize, VulkanError> {
        demand_feedback_resolution_bound(
            tick_count,
            device_slices.iter().flat_map(|slice| {
                slice
                    .resident_execution_plan
                    .dispatch_segments
                    .iter()
                    .filter_map(|segment| segment.demand_residency.as_ref())
                    .flat_map(VulkanDemandResidencySegment::resource_domain_counts)
            }),
        )
    }

    fn ensure_execution_headroom(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<(), VulkanError> {
        let mut entered_stores = BTreeSet::new();
        device_slices
            .iter()
            .filter_map(|slice| {
                slice
                    .demand_residency_context
                    .as_ref()
                    .map(|context| (slice, context))
            })
            .filter(|(_, context)| {
                entered_stores.insert(Arc::as_ptr(&context.store) as usize)
            })
            .try_for_each(|(slice, context)| {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanError(format!(
                        "demand feedback has no bound device {:?}",
                        slice.device_id
                    ))
                })?;
                context.store.ensure_execution_headroom(device).map_err(|error| {
                    VulkanError(format!(
                        "failed to establish demand feedback headroom on {:?}: {error}",
                        slice.device_id
                    ))
                })
            })
    }

    fn resolve_published_fault(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        tick_count: usize,
        sequence_variant: u8,
    ) -> Result<
        Option<(VulkanDemandFeedbackCheckpoint, Vec<usize>)>,
        VulkanResidentInProcessPlacedRuntimeError,
    >
    {
        let mut pending = Vec::new();
        for feedback_lane in 0..tick_count {
            for (slice_index, slice) in device_slices.iter().enumerate() {
                for (segment_index, segment) in slice
                    .resident_execution_plan
                    .dispatch_segments
                    .iter()
                    .enumerate()
                {
                    let Some(demand) = &segment.demand_residency else {
                        continue;
                    };
                    if demand
                        .feedback_lane_has_pending_miss(sequence_variant, feedback_lane)
                        .map_err(
                            VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch,
                        )?
                    {
                        pending.push((feedback_lane, slice_index, segment_index));
                    }
                }
            }
        }
        let Some((feedback_lane, slice_index, segment_index)) =
            unique_pending_demand_feedback_checkpoint(&pending)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
        else {
            return Ok(None);
        };
        let slice = &device_slices[slice_index];
        let device = devices.get(&slice.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: slice.device_id.clone(),
            }
        })?;
        let demand = slice.resident_execution_plan.dispatch_segments[segment_index]
            .demand_residency
            .as_ref()
            .expect("pending demand feedback segment remains mounted");
        let (gate_index, resource_indices) = demand
            .resolve_feedback_lane_miss(device, sequence_variant, feedback_lane)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "demand feedback miss disappeared before it was resolved".to_string(),
                ))
            })?;
        Ok(Some((
            VulkanDemandFeedbackCheckpoint {
                feedback_lane,
                slice_index,
                segment_index,
                gate_index,
            },
            resource_indices,
        )))
    }
}

fn unique_pending_demand_feedback_checkpoint(
    pending: &[(usize, usize, usize)],
) -> Result<Option<(usize, usize, usize)>, VulkanError> {
    match pending {
        [] => Ok(None),
        [checkpoint] => Ok(Some(*checkpoint)),
        _ => Err(VulkanError(format!(
            "one guarded resident feedback attempt reported {} independent miss checkpoints: {pending:?}",
            pending.len()
        ))),
    }
}
