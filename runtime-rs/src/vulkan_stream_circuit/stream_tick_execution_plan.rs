/// Resident execution structure for one placed device slice. Edge stages stay
/// visible to the scheduler, while every uninterrupted dispatch region becomes
/// one GPU submission.
pub struct VulkanMountedPlacedResidentStreamTickExecutionPlan {
    pub tick_plan: Arc<VulkanMountedPlacedStreamTickPlan>,
    pub dispatch_segment_count: usize,
    pub dispatch_count: usize,
    pub distributed_dispatch_count: usize,
    dispatch_segments: Vec<VulkanMountedPlacedResidentDispatchSegmentRunner>,
    distributed_dispatch_stages: BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    physical_execution_islands: BTreeMap<usize, VulkanMountedPhysicalExecutionIslandStage>,
    distributed_dispatch_dependencies:
        BTreeMap<usize, VulkanMountedPlacedDistributedDispatchDependencies>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanMountedPhysicalExecutionIslandStage {
    dispatches: Vec<VulkanMountedPlacedStreamTickDispatch>,
    end_stage_index: usize,
}

impl VulkanMountedPhysicalExecutionIslandStage {
    fn leader(&self) -> &VulkanMountedPlacedStreamTickDispatch {
        self.dispatches
            .first()
            .expect("distributed stage groups are never empty")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanMountedPlacedDistributedDispatchDependencies {
    dispatch_index: usize,
    has_owner_producer: bool,
    has_owner_continuation: bool,
}

impl VulkanMountedPlacedResidentStreamTickExecutionPlan {
    fn uses_demand_residency(&self) -> bool {
        self.dispatch_segments
            .iter()
            .any(|segment| segment.demand_residency.is_some())
    }

    fn run_single_segment_demand_resident(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        sequence_variant: u8,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        post_copies: &[VulkanResidentBufferRangeCopy<'_>],
    ) -> Result<(), VulkanMountedPlacedResidentKernelDispatchError> {
        if self.dispatch_segments.len() != 1
            || self.distributed_dispatch_count != 0
            || self.tick_plan.receive_stage_count != 0
            || self.tick_plan.publish_stage_count != 0
        {
            return Err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan(
                VulkanError(
                    "single-segment demand execution requires one local dispatch segment"
                        .to_string(),
                ),
            ));
        }
        let segment = self
            .dispatch_segments
            .first()
            .expect("one demand segment was validated");
        let demand = segment.demand_residency.as_ref().ok_or_else(|| {
            VulkanMountedPlacedResidentKernelDispatchError::Vulkan(VulkanError(
                "single-segment demand execution was requested for an eager segment"
                    .to_string(),
            ))
        })?;
        segment
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanMountedPlacedResidentKernelDispatchError::Vulkan)?;
        demand.run(
            device,
            &segment.dispatches,
            control,
            prefix_dispatches,
            suffix_dispatches,
            sequence_variant,
            &[],
            &[],
            input_copies,
            post_copies,
        )
    }

    pub fn from_tick_plan(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        Self::from_tick_plan_with_distributed_dispatches(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            tick_plan,
            &BTreeSet::new(),
        )
    }

    pub fn from_tick_plan_with_distributed_dispatches(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
        distributed_dispatch_indices: &BTreeSet<usize>,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        let physical_execution_islands = distributed_dispatch_indices
            .iter()
            .map(|dispatch_index| vec![*dispatch_index])
            .collect::<Vec<_>>();
        Self::from_tick_plan_with_physical_execution_islands(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            tick_plan,
            &physical_execution_islands,
        )
    }

    pub fn from_tick_plan_with_physical_execution_islands(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
        physical_execution_islands: &[Vec<usize>],
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        Self::from_tick_plan_with_physical_execution_islands_and_demand(
            device,
            mounted,
            mounted_bound_plan,
            loaded_manifest,
            tick_plan,
            physical_execution_islands,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_tick_plan_with_physical_execution_islands_and_demand(
        device: &VulkanComputeDevice,
        mounted: &VulkanMountedPlacedStreamCircuit,
        mounted_bound_plan: &VulkanMountedPlacedBoundDispatchPlan,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        tick_plan: VulkanMountedPlacedStreamTickPlan,
        physical_execution_islands: &[Vec<usize>],
        physical_residency_schedule: Option<&VulkanPhysicalResidencySchedule>,
        demand_context: Option<&VulkanDemandResidencyExecutionContext>,
        demand_pipeline_predicate: Option<Arc<VulkanResidentBuffer>>,
    ) -> Result<Self, VulkanMountedPlacedResidentKernelDispatchError> {
        if tick_plan.device_id != mounted.device_id() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::ExecutionPlanDeviceMismatch {
                    plan_device_id: tick_plan.device_id.clone(),
                    mounted_device_id: mounted.device_id().to_string(),
                },
            );
        }
        if tick_plan.device_id != mounted_bound_plan.device_id {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::ExecutionBoundPlanDeviceMismatch {
                    plan_device_id: tick_plan.device_id.clone(),
                    bound_plan_device_id: mounted_bound_plan.device_id.clone(),
                },
            );
        }

        let distributed_dispatch_indices = physical_execution_islands
            .iter()
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>();
        let distributed_dispatch_stages =
            distributed_dispatch_stages(&tick_plan, &distributed_dispatch_indices)?;
        let physical_execution_islands = physical_execution_island_stage_groups(
            &distributed_dispatch_stages,
            physical_execution_islands,
        )?;

        let dispatch_segment_stage_ranges =
            resident_dispatch_segment_stage_ranges_excluding_dispatches(
                &tick_plan.stages,
                &distributed_dispatch_indices,
            );
        let distributed_dispatch_dependencies = distributed_dispatch_dependency_topologies(
            &physical_execution_islands,
            &dispatch_segment_stage_ranges,
        );
        let mut dispatch_segments = Vec::new();
        for &(start, end) in &dispatch_segment_stage_ranges {
            dispatch_segments.push(
                VulkanMountedPlacedResidentDispatchSegmentRunner::from_dispatch_stages(
                    device,
                    mounted,
                    mounted_bound_plan,
                    loaded_manifest,
                    &tick_plan.stages[start..end],
                    physical_residency_schedule,
                    demand_context,
                    demand_pipeline_predicate.clone(),
                )?,
            );
        }
        if dispatch_segments.is_empty() && distributed_dispatch_stages.is_empty() {
            return Err(
                VulkanMountedPlacedResidentKernelDispatchError::MissingExecutionDispatchSegments {
                    device_id: tick_plan.device_id.clone(),
                },
            );
        }
        let dispatch_count = dispatch_segments
            .iter()
            .map(|segment| segment.dispatch_count)
            .sum();
        let dispatch_segment_count = dispatch_segments.len();
        let distributed_dispatch_count = distributed_dispatch_stages.len();
        Ok(Self {
            tick_plan: Arc::new(tick_plan),
            dispatch_segment_count,
            dispatch_count,
            distributed_dispatch_count,
            dispatch_segments,
            distributed_dispatch_stages,
            physical_execution_islands,
            distributed_dispatch_dependencies,
        })
    }

    fn segment_starting_at(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPlacedResidentDispatchSegmentRunner> {
        self.dispatch_segments
            .iter()
            .find(|segment| segment.start_stage_index == stage_index)
    }

    fn first_dispatch_segment_stage_index(&self) -> Option<usize> {
        self.dispatch_segments
            .first()
            .map(|segment| segment.start_stage_index)
    }

    fn last_dispatch_segment_stage_index(&self) -> Option<usize> {
        self.dispatch_segments
            .last()
            .map(|segment| segment.start_stage_index)
    }

    fn configure_feedback_indirect_dispatches(
        &mut self,
        device: &VulkanComputeDevice,
        control: &mut VulkanResidentFeedbackControlPlane,
        device_id: &str,
        prefix_dispatches: &[&VulkanResidentKernelDispatch],
        suffix_dispatches: &[&VulkanResidentKernelDispatch],
        generation_tail_dispatch_count: Option<usize>,
        lane_capacity: usize,
    ) -> Result<(), VulkanError> {
        let first_stage_index = self.first_dispatch_segment_stage_index();
        let last_stage_index = self.last_dispatch_segment_stage_index();
        for segment in &mut self.dispatch_segments {
            let prefix = if Some(segment.start_stage_index) == first_stage_index {
                prefix_dispatches
            } else {
                &[]
            };
            let suffix = if Some(segment.start_stage_index) == last_stage_index {
                suffix_dispatches
            } else {
                &[]
            };
            segment.configure_feedback_indirect_dispatches(
                device,
                control,
                device_id,
                prefix,
                suffix,
                (Some(segment.start_stage_index) == last_stage_index)
                    .then_some(generation_tail_dispatch_count)
                    .flatten(),
                lane_capacity,
            )?;
        }
        Ok(())
    }

    fn feedback_dispatch_count(&self) -> Result<usize, VulkanError> {
        self.dispatch_segments.iter().try_fold(0usize, |total, segment| {
            total
                .checked_add(segment.feedback_dispatch_count()?)
                .ok_or_else(|| VulkanError("resident feedback dispatch count overflowed".to_string()))
        })
    }

    pub fn distributed_dispatch_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPlacedStreamTickDispatch> {
        self.physical_execution_islands
            .get(&stage_index)
            .map(VulkanMountedPhysicalExecutionIslandStage::leader)
    }

    fn physical_execution_island_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<&VulkanMountedPhysicalExecutionIslandStage> {
        self.physical_execution_islands.get(&stage_index)
    }

    fn distributed_dispatch_dependencies_at_stage(
        &self,
        stage_index: usize,
    ) -> Option<VulkanMountedPlacedDistributedDispatchDependencies> {
        self.distributed_dispatch_dependencies
            .get(&stage_index)
            .copied()
    }

    fn resident_stream_tick_cursor(
        &self,
        stream_tick: u64,
    ) -> VulkanMountedPlacedResidentStreamTickCursor {
        VulkanMountedPlacedResidentStreamTickCursor::new_shared(
            Arc::clone(&self.tick_plan),
            stream_tick,
            true,
        )
    }

    fn compact_resident_stream_tick_cursor(
        &self,
        stream_tick: u64,
    ) -> VulkanMountedPlacedResidentStreamTickCursor {
        VulkanMountedPlacedResidentStreamTickCursor::new_shared(
            Arc::clone(&self.tick_plan),
            stream_tick,
            false,
        )
    }
}

fn distributed_dispatch_dependency_topologies(
    physical_execution_islands: &BTreeMap<usize, VulkanMountedPhysicalExecutionIslandStage>,
    dispatch_segment_stage_ranges: &[(usize, usize)],
) -> BTreeMap<usize, VulkanMountedPlacedDistributedDispatchDependencies> {
    physical_execution_islands
        .iter()
        .map(|(stage_index, group)| {
            (
                *stage_index,
                VulkanMountedPlacedDistributedDispatchDependencies {
                    dispatch_index: group.leader().dispatch_index,
                    has_owner_producer: dispatch_segment_stage_ranges
                        .iter()
                        .any(|(_, end)| end == stage_index),
                    has_owner_continuation: dispatch_segment_stage_ranges
                        .iter()
                        .any(|(start, _)| *start == group.end_stage_index),
                },
            )
        })
        .collect()
}

fn physical_execution_island_stage_groups(
    distributed_dispatch_stages: &BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    physical_execution_islands: &[Vec<usize>],
) -> Result<
    BTreeMap<usize, VulkanMountedPhysicalExecutionIslandStage>,
    VulkanMountedPlacedResidentKernelDispatchError,
> {
    let stages_by_dispatch = distributed_dispatch_stages
        .iter()
        .map(|(stage_index, dispatch)| (dispatch.dispatch_index, (*stage_index, dispatch)))
        .collect::<BTreeMap<_, _>>();
    let mut islands = BTreeMap::new();
    let mut claimed_dispatches = BTreeSet::new();
    for dispatch_indices in physical_execution_islands {
        let Some(leader_dispatch_index) = dispatch_indices.first().copied() else {
            continue;
        };
        let (leader_stage_index, _) = stages_by_dispatch
            .get(&leader_dispatch_index)
            .copied()
            .ok_or_else(|| {
                VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                    device_id: "distributed execution plan".to_string(),
                    dispatch_index: leader_dispatch_index,
                }
            })?;
        let mut dispatches = Vec::with_capacity(dispatch_indices.len());
        for (offset, dispatch_index) in dispatch_indices.iter().copied().enumerate() {
            if !claimed_dispatches.insert(dispatch_index) {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                        device_id: "distributed execution plan".to_string(),
                        stage_index: leader_stage_index + offset,
                        expected_dispatch_index: dispatch_index,
                        completed_dispatch_index: dispatch_index,
                    },
                );
            }
            let expected_stage_index = leader_stage_index + offset;
            let (stage_index, dispatch) = stages_by_dispatch
                .get(&dispatch_index)
                .copied()
                .ok_or_else(|| {
                    VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                        device_id: "distributed execution plan".to_string(),
                        dispatch_index,
                    }
                })?;
            if stage_index != expected_stage_index {
                return Err(
                    VulkanMountedPlacedResidentKernelDispatchError::DistributedDispatchMismatch {
                        device_id: "distributed execution plan".to_string(),
                        stage_index: expected_stage_index,
                        expected_dispatch_index: dispatch_index,
                        completed_dispatch_index: dispatch.dispatch_index,
                    },
                );
            }
            dispatches.push(dispatch.clone());
        }
        islands.insert(
            leader_stage_index,
            VulkanMountedPhysicalExecutionIslandStage {
                dispatches,
                end_stage_index: leader_stage_index + dispatch_indices.len(),
            },
        );
    }
    Ok(islands)
}

#[cfg(test)]
fn resident_dispatch_segment_stage_ranges(
    stages: &[VulkanMountedPlacedStreamTickStage],
) -> Vec<(usize, usize)> {
    resident_dispatch_segment_stage_ranges_excluding_dispatches(stages, &BTreeSet::new())
}

fn resident_dispatch_segment_stage_ranges_excluding_dispatches(
    stages: &[VulkanMountedPlacedStreamTickStage],
    excluded_dispatch_indices: &BTreeSet<usize>,
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut stage_index = 0usize;
    while stage_index < stages.len() {
        if !is_canonical_dispatch_stage(&stages[stage_index], excluded_dispatch_indices) {
            stage_index += 1;
            continue;
        }
        let start = stage_index;
        while stage_index < stages.len()
            && is_canonical_dispatch_stage(&stages[stage_index], excluded_dispatch_indices)
        {
            stage_index += 1;
        }
        ranges.push((start, stage_index));
    }
    ranges
}

fn is_canonical_dispatch_stage(
    stage: &VulkanMountedPlacedStreamTickStage,
    excluded_dispatch_indices: &BTreeSet<usize>,
) -> bool {
    matches!(
        stage,
        VulkanMountedPlacedStreamTickStage::Dispatch { dispatch, .. }
            if !excluded_dispatch_indices.contains(&dispatch.dispatch_index)
    )
}

fn distributed_dispatch_stages(
    tick_plan: &VulkanMountedPlacedStreamTickPlan,
    distributed_dispatch_indices: &BTreeSet<usize>,
) -> Result<
    BTreeMap<usize, VulkanMountedPlacedStreamTickDispatch>,
    VulkanMountedPlacedResidentKernelDispatchError,
> {
    let mut stages = BTreeMap::new();
    let mut found = BTreeSet::new();
    for stage in &tick_plan.stages {
        let VulkanMountedPlacedStreamTickStage::Dispatch {
            stage_index,
            dispatch,
        } = stage
        else {
            continue;
        };
        if distributed_dispatch_indices.contains(&dispatch.dispatch_index) {
            found.insert(dispatch.dispatch_index);
            stages.insert(*stage_index, dispatch.clone());
        }
    }
    if let Some(dispatch_index) = distributed_dispatch_indices.difference(&found).next() {
        return Err(
            VulkanMountedPlacedResidentKernelDispatchError::MissingDistributedDispatchStage {
                device_id: tick_plan.device_id.clone(),
                dispatch_index: *dispatch_index,
            },
        );
    }
    Ok(stages)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanMountedPlacedResidentExecutionGraphRun {
    pub device_id: String,
    pub component_runs: Vec<VulkanMountedPlacedResidentComponentRun>,
}

impl VulkanMountedPlacedResidentExecutionGraphRun {
    pub fn component_count(&self) -> usize {
        self.component_runs.len()
    }

    pub fn dispatch_count(&self) -> usize {
        self.component_runs
            .iter()
            .map(VulkanMountedPlacedResidentComponentRun::dispatch_count)
            .sum()
    }

    pub fn run_time_ns(&self) -> u64 {
        self.component_runs.iter().fold(0u64, |total, component| {
            total.saturating_add(component.run_time_ns())
        })
    }

    pub fn component_ids(&self) -> Vec<&str> {
        self.component_runs
            .iter()
            .map(|component| component.component_id.as_str())
            .collect()
    }
}
