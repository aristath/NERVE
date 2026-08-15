const VULKAN_DEMAND_FEEDBACK_PREDICATE_WORD_COUNT: usize = 2;
pub(crate) const VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY: usize =
    VULKAN_DEMAND_FEEDBACK_PREDICATE_WORD_COUNT * size_of::<u32>();

pub(crate) fn demand_feedback_ready_predicate_bytes(
) -> [u8; VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY] {
    let mut bytes = [0u8; VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY];
    bytes[..size_of::<u32>()].copy_from_slice(&1u32.to_le_bytes());
    bytes
}

struct VulkanResidentDemandFeedbackState {
    predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    completion_predicate: Arc<VulkanResidentBuffer>,
    stores_by_device: BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    distributed_resource_domain_counts: Vec<usize>,
    fault_sources: BTreeMap<u32, VulkanDemandFeedbackFaultSource>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanDemandFeedbackFaultSource {
    Local {
        slice_index: usize,
        segment_index: usize,
    },
    Distributed {
        owner_device_id: String,
        dispatch_index: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanDemandFeedbackPublishedFault {
    tick_count: usize,
    feedback_lane: usize,
    source_id: u32,
    sequence_variant: u8,
}

fn demand_feedback_fault_source_id(kind: &str, fields: &[&str]) -> u32 {
    let mut digest = Sha256::new();
    for field in std::iter::once(kind).chain(fields.iter().copied()) {
        digest.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(field.as_bytes());
    }
    let digest = digest.finalize();
    digest
        .chunks_exact(size_of::<u32>())
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("SHA-256 u32 chunks are exact")))
        .find(|value| *value != 0)
        .unwrap_or(1)
}

fn demand_feedback_local_fault_source_id(
    execution_scope: &str,
    checkpoint_id: &str,
    selector_id: &str,
) -> u32 {
    demand_feedback_fault_source_id(
        "local",
        &[execution_scope, checkpoint_id, selector_id],
    )
}

fn demand_feedback_distributed_fault_source_id(
    execution_scope: &str,
    component_id: &str,
    node_id: &str,
    selector_id: &str,
) -> u32 {
    demand_feedback_fault_source_id(
        "distributed",
        &[execution_scope, component_id, node_id, selector_id],
    )
}

fn register_demand_feedback_fault_source(
    sources: &mut BTreeMap<u32, (String, VulkanDemandFeedbackFaultSource)>,
    source_id: u32,
    identity: String,
    source: VulkanDemandFeedbackFaultSource,
) -> Result<(), VulkanError> {
    if source_id == 0 {
        return Err(VulkanError(
            "demand feedback fault source zero is reserved".to_string(),
        ));
    }
    if let Some((existing_identity, existing_source)) = sources.get(&source_id) {
        if existing_identity == &identity && existing_source == &source {
            return Ok(());
        }
        return Err(VulkanError(format!(
            "demand feedback fault-source collision {source_id}: {existing_identity:?} at {existing_source:?} conflicts with {identity:?} at {source:?}",
        )));
    }
    sources.insert(source_id, (identity, source));
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDemandFeedbackCheckpoint {
    feedback_lane: usize,
    target: VulkanDemandFeedbackCheckpointTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanDemandFeedbackCheckpointTarget {
    Local {
        slice_index: usize,
        segment_index: usize,
        gate_index: usize,
    },
    Distributed {
        slice_index: usize,
        dispatch_index: usize,
        shard_index: usize,
        gate_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanPlacedDemandFeedbackTickResume {
    feedback_lane: usize,
    schedule_start_turn_index: usize,
    next_stage_indices: Vec<usize>,
    target_slice_index: usize,
    target_stage_index: usize,
    local_gate_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDemandFeedbackResumePlan {
    schedule_start_turn_index: usize,
    next_stage_indices: Vec<usize>,
}

struct VulkanDemandFeedbackFaultResolution {
    resolved: Vec<(VulkanDemandFeedbackCheckpoint, Vec<usize>)>,
    resume_checkpoint: VulkanDemandFeedbackCheckpoint,
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
    demand_feedback_resume_plan_at_stage(
        tick_plans,
        target_device_index,
        target_stage_index,
        false,
    )
}

fn demand_feedback_resume_plan_after_stage(
    tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    target_device_index: usize,
    target_stage_index: usize,
) -> Result<VulkanDemandFeedbackResumePlan, VulkanError> {
    demand_feedback_resume_plan_at_stage(
        tick_plans,
        target_device_index,
        target_stage_index,
        true,
    )
}

fn demand_feedback_resume_plan_after_dispatch_stage_range(
    tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    target_device_index: usize,
    target_stage_range: std::ops::Range<usize>,
) -> Result<VulkanDemandFeedbackResumePlan, VulkanError> {
    let target_plan = tick_plans.get(target_device_index).ok_or_else(|| {
        VulkanError(format!(
            "demand feedback resume device {target_device_index} is out of bounds"
        ))
    })?;
    if target_stage_range.is_empty()
        || target_stage_range.end > target_plan.stages.len()
        || target_plan.stages[target_stage_range.clone()]
            .iter()
            .any(|stage| !matches!(stage, VulkanMountedPlacedStreamTickStage::Dispatch { .. }))
    {
        return Err(VulkanError(format!(
            "demand feedback physical execution range {:?} is not a non-empty dispatch range on device {target_device_index}",
            target_stage_range,
        )));
    }
    let plan = demand_feedback_resume_plan_after_stage(
        tick_plans,
        target_device_index,
        target_stage_range.end - 1,
    )?;
    if plan.next_stage_indices[target_device_index] != target_stage_range.end {
        return Err(VulkanError(
            "demand feedback physical execution range is not the target device causal frontier"
                .to_string(),
        ));
    }
    Ok(plan)
}

fn demand_feedback_resume_plan_at_stage(
    tick_plans: &[&VulkanMountedPlacedStreamTickPlan],
    target_device_index: usize,
    target_stage_index: usize,
    target_is_completed: bool,
) -> Result<VulkanDemandFeedbackResumePlan, VulkanError> {
    let topology = VulkanDemandFeedbackStageTopology::from_tick_plans(tick_plans)?;
    if !topology.is_total_ordered() {
        return Err(VulkanError(
            "demand feedback checkpoint has an independent parallel branch without explicit GPU progress markers"
                .to_string(),
        ));
    }
    let target = topology.node(target_device_index, target_stage_index)?;
    if !target_is_completed
        && !matches!(
            tick_plans[target_device_index].stages[target_stage_index],
            VulkanMountedPlacedStreamTickStage::Dispatch { .. }
        )
    {
        return Err(VulkanError(
            "demand feedback checkpoint resume must start at a dispatch segment".to_string(),
        ));
    }
    let mut completed = topology.ancestors(target);
    if target_is_completed {
        completed.insert(target);
    }
    let next_stage_indices = tick_plans
        .iter()
        .enumerate()
        .map(|(device_index, plan)| {
            let start = topology.node_offsets[device_index];
            plan.stages
                .iter()
                .enumerate()
                .take_while(|(stage_index, _)| completed.contains(&(start + stage_index)))
                .count()
        })
        .collect::<Vec<_>>();
    let expected_target_frontier = target_stage_index + usize::from(target_is_completed);
    if next_stage_indices[target_device_index] != expected_target_frontier {
        return Err(VulkanError(
            "demand feedback checkpoint does not follow a contiguous causal prefix".to_string(),
        ));
    }
    let schedule_start_turn_index = demand_feedback_resume_turn_index(
        tick_plans,
        target_device_index,
        target_stage_index,
        target_is_completed,
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
    target_is_completed: bool,
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
                    && !target_is_completed
                {
                    return Ok(turn_index);
                }
                let current_stage_index = next_stage_indices[device_index];
                match &plan.stages[current_stage_index] {
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
                if device_index == target_device_index
                    && current_stage_index == target_stage_index
                    && target_is_completed
                {
                    return Ok(turn_index);
                }
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
    let has_local_checkpoints = model
            .device_slices
            .iter()
            .any(|slice| !slice.physical_residency_schedule().checkpoints.is_empty());
    let has_distributed_checkpoints = model
        .distributed_execution_plans
        .decode
        .dispatches
        .iter()
        .any(|dispatch| !dispatch.selected_resource_partitions.is_empty());
    if !model.resource_residency_policy.is_demand_loaded()
        || (!has_local_checkpoints && !has_distributed_checkpoints)
    {
        return Ok(None);
    }
    let device_ids = model.device_ids.clone();
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
                owner.create_conditional_resident_buffer(
                    VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
                )?,
            )]
        } else {
            owner
                .create_shared_conditional_resident_buffers_for_route(
                    peers,
                    VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
                    VulkanSharedResidentBufferRoute::SharedHost,
                )?
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
    write_demand_feedback_predicate_views(predicates_by_device.values(), true)?;
    Ok(Some(predicates_by_device))
}

fn write_shared_resident_predicate_views<'a>(
    predicates: impl IntoIterator<Item = &'a Arc<VulkanResidentBuffer>>,
    value: &[u8],
    contract: &str,
) -> Result<(), VulkanError> {
    let mut written_views = BTreeSet::new();
    for predicate in predicates {
        if written_views.insert(Arc::as_ptr(predicate) as usize) {
            if predicate.byte_capacity() < value.len() {
                return Err(VulkanError(format!(
                    "{contract} predicate capacity {} cannot hold its {}-byte ABI",
                    predicate.byte_capacity(),
                    value.len(),
                )));
            }
            predicate.write_bytes(value)?;
        }
    }
    if written_views.is_empty() {
        return Err(VulkanError(format!("{contract} predicate is absent")));
    }
    Ok(())
}

fn write_demand_feedback_predicate_views<'a>(
    predicates: impl IntoIterator<Item = &'a Arc<VulkanResidentBuffer>>,
    enabled: bool,
) -> Result<(), VulkanError> {
    let mut value = demand_feedback_ready_predicate_bytes();
    value[..size_of::<u32>()].copy_from_slice(&u32::from(enabled).to_le_bytes());
    write_shared_resident_predicate_views(predicates, &value, "demand-feedback")
}

impl VulkanResidentDemandFeedbackState {
    fn new(
        predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
        model: &VulkanResidentInProcessPlacedModelPackage,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        output_device_id: &str,
    ) -> Result<Self, VulkanError> {
        if device_slices
            .iter()
            .any(|slice| !predicates_by_device.contains_key(&slice.device_id))
            || model
                .compiled_resource_device_stores
                .keys()
                .any(|device_id| !predicates_by_device.contains_key(device_id))
        {
            return Err(VulkanError(
                "demand feedback predicates do not cover every execution participant".to_string(),
            ));
        }
        let local_checkpoint_count_per_tick = device_slices
            .iter()
            .flat_map(|slice| &slice.resident_execution_plan.dispatch_segments)
            .filter_map(|segment| segment.demand_residency.as_ref())
            .map(|segment| segment.gate_specs.len())
            .try_fold(0usize, |total, count| total.checked_add(count))
            .ok_or_else(|| {
                VulkanError("demand feedback checkpoint count overflowed".to_string())
            })?;
        let distributed_resource_domain_counts = model
            .distributed_execution_plans
            .decode
            .dispatches
            .iter()
            .flat_map(|dispatch| &dispatch.selected_resource_partitions)
            .map(|partition| partition.resource_count)
            .collect::<Vec<_>>();
        let checkpoint_count_per_tick = local_checkpoint_count_per_tick
            .checked_add(distributed_resource_domain_counts.len())
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
        let mut fault_sources = BTreeMap::new();
        for (slice_index, slice) in device_slices.iter().enumerate() {
            for (segment_index, demand) in slice
                .resident_execution_plan
                .dispatch_segments
                .iter()
                .enumerate()
                .filter_map(|(segment_index, segment)| {
                    segment
                        .demand_residency
                        .as_ref()
                        .map(|demand| (segment_index, demand))
                })
            {
                for gate in &demand.gate_specs {
                    let identity = format!(
                        "local:{}:{}:{}",
                        demand.context.execution_scope, gate.checkpoint_id, gate.selector_id,
                    );
                    register_demand_feedback_fault_source(
                        &mut fault_sources,
                        demand_feedback_local_fault_source_id(
                            &demand.context.execution_scope,
                            &gate.checkpoint_id,
                            &gate.selector_id,
                        ),
                        identity,
                        VulkanDemandFeedbackFaultSource::Local {
                            slice_index,
                            segment_index,
                        },
                    )?;
                }
            }
        }
        for island in &model
            .distributed_execution_plans
            .decode
            .execution_islands
        {
            let leader = island.leader();
            for dispatch in &island.dispatches {
                for partition in &dispatch.selected_resource_partitions {
                    let identity = format!(
                        "distributed:{}:{}:{}:{}",
                        partition.execution_scope,
                        dispatch.component_id,
                        dispatch.node_id,
                        partition.selector_id,
                    );
                    register_demand_feedback_fault_source(
                        &mut fault_sources,
                        demand_feedback_distributed_fault_source_id(
                            &partition.execution_scope,
                            &dispatch.component_id,
                            &dispatch.node_id,
                            &partition.selector_id,
                        ),
                        identity,
                        VulkanDemandFeedbackFaultSource::Distributed {
                            owner_device_id: island.owner_device_id.clone(),
                            dispatch_index: leader.dispatch_index,
                        },
                    )?;
                }
            }
        }
        let fault_sources = fault_sources
            .into_iter()
            .map(|(source_id, (_, source))| (source_id, source))
            .collect();
        Ok(Self {
            predicates_by_device,
            completion_predicate,
            stores_by_device: model.compiled_resource_device_stores.clone(),
            distributed_resource_domain_counts,
            fault_sources,
        })
    }

    fn reset_pipeline_predicate(&self) -> Result<(), VulkanError> {
        write_demand_feedback_predicate_views(self.predicates_by_device.values(), true)
    }

    fn pipeline_predicate_diagnostic(
        &self,
    ) -> Result<BTreeMap<String, [u32; VULKAN_DEMAND_FEEDBACK_PREDICATE_WORD_COUNT]>, VulkanError>
    {
        self.predicates_by_device
            .iter()
            .map(|(device_id, predicate)| {
                let bytes = predicate.read_bytes(VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY)?;
                let value = std::array::from_fn(|word| {
                    let start = word * size_of::<u32>();
                    u32::from_le_bytes(
                        bytes[start..start + size_of::<u32>()]
                            .try_into()
                            .expect("demand feedback predicate words are exact u32s"),
                    )
                });
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
            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
        )
    }

    fn resolution_bound(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        tick_count: usize,
    ) -> Result<usize, VulkanError> {
        demand_feedback_resolution_bound(
            tick_count,
            device_slices
                .iter()
                .flat_map(|slice| {
                    slice
                        .resident_execution_plan
                        .dispatch_segments
                        .iter()
                        .filter_map(|segment| segment.demand_residency.as_ref())
                        .flat_map(VulkanDemandResidencySegment::resource_domain_counts)
                })
                .chain(self.distributed_resource_domain_counts.iter().copied()),
        )
    }

    fn ensure_execution_headroom(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<(), VulkanError> {
        let mut entered_stores = BTreeSet::new();
        self.stores_by_device
            .iter()
            .filter(|(_, store)| {
                store.residency_policy().is_demand_loaded()
                    && entered_stores.insert(Arc::as_ptr(store) as usize)
            })
            .try_for_each(|(device_id, store)| {
                let device = devices.get(device_id).ok_or_else(|| {
                    VulkanError(format!(
                        "demand feedback has no bound device {:?}",
                        device_id
                    ))
                })?;
                store.ensure_execution_headroom(device).map_err(|error| {
                    VulkanError(format!(
                        "failed to establish demand feedback headroom on {:?}: {error}",
                        device_id
                    ))
                })
            })
    }

    fn resolve_published_fault(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        distributed_runners: &VulkanDistributedDispatchRunners,
        fault: VulkanDemandFeedbackPublishedFault,
    ) -> Result<VulkanDemandFeedbackFaultResolution, VulkanResidentInProcessPlacedRuntimeError>
    {
        let VulkanDemandFeedbackPublishedFault {
            tick_count,
            feedback_lane: fault_feedback_lane,
            source_id: fault_source_id,
            sequence_variant,
        } = fault;
        if fault_feedback_lane >= tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "demand feedback fault lane {fault_feedback_lane} exceeds window width {tick_count}"
                )),
            ));
        }
        let fault_source = self.fault_sources.get(&fault_source_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "demand feedback published unknown fault-source ID {fault_source_id}",
            )))
        })?;
        if let VulkanDemandFeedbackFaultSource::Local {
            slice_index,
            segment_index,
        } = *fault_source
        {
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
                .resolve_feedback_lane_miss(device, sequence_variant, fault_feedback_lane)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "demand feedback miss disappeared before it was resolved".to_string(),
                    ))
                })?;
            let checkpoint = VulkanDemandFeedbackCheckpoint {
                feedback_lane: fault_feedback_lane,
                target: VulkanDemandFeedbackCheckpointTarget::Local {
                    slice_index,
                    segment_index,
                    gate_index,
                },
            };
            return Ok(VulkanDemandFeedbackFaultResolution {
                resolved: vec![(checkpoint, resource_indices)],
                resume_checkpoint: checkpoint,
            });
        }
        let VulkanDemandFeedbackFaultSource::Distributed {
            owner_device_id,
            dispatch_index,
        } = fault_source
        else {
            unreachable!("local demand feedback fault returned above");
        };
        let slice_index = device_slices
            .iter()
            .position(|slice| slice.device_id == *owner_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed demand feedback owner {owner_device_id:?} has no placed slice"
                )))
            })?;
        let resolution = distributed_runners
            .resolve_completed_residency_fault(
                owner_device_id,
                *dispatch_index,
                VulkanDistributedDispatchSequenceKind::for_feedback_lane(Some(
                    fault_feedback_lane,
                )),
                |device_id| {
                    devices.get(device_id).map(Rc::as_ref).ok_or_else(|| {
                        VulkanError(format!(
                            "distributed demand feedback device {device_id:?} is not bound"
                        ))
                    })
                },
            )
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::Tick(
                    VulkanMountedPlacedResidentInProcessStreamTickError::Distributed(error),
                )
            })?
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed demand feedback miss disappeared before it was resolved"
                        .to_string(),
                ))
            })?;
        let resolved = resolution
            .misses
            .into_iter()
            .map(|miss| {
                (
                    VulkanDemandFeedbackCheckpoint {
                        feedback_lane: fault_feedback_lane,
                        target: VulkanDemandFeedbackCheckpointTarget::Distributed {
                            slice_index,
                            dispatch_index: *dispatch_index,
                            shard_index: miss.shard_index,
                            gate_index: miss.gate_index,
                        },
                    },
                    miss.resource_indices,
                )
            })
            .collect::<Vec<_>>();
        let resume_checkpoint = resolved
            .first()
            .map(|(checkpoint, _)| *checkpoint)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed demand feedback resolution contained no resource misses"
                        .to_string(),
                ))
            })?;
        Ok(VulkanDemandFeedbackFaultResolution {
            resolved,
            resume_checkpoint,
        })
    }
}
