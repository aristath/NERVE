use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::stream_plan::TensorIndex;
use crate::stream_circuit::ComponentEdgePlacement;
use crate::tensor_storage::{TensorStorage, TensorStorageRange};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanResidentBuffer,
    VulkanResidentBufferPool, VulkanResidentBufferPoolAllocation,
    VulkanResidentBufferPoolKey,
    VulkanResidentKernelBufferAccess,
    VulkanResidentKernelBufferBinding, VulkanResidentKernelDispatch,
    VulkanResidentKernelSequence, VulkanResidentKernelSequenceStep,
    VulkanResidentQueueSubmissionBatch, VulkanSharedResidentBufferRoute,
    VulkanTimelineSemaphore, VulkanTimelineSemaphorePoint,
    VulkanTimelineSemaphoreReplayState,
};
use crate::vulkan_stream_circuit::{
    VulkanActivationSlotBufferOverride, VulkanDescriptorResourceAddress,
    VulkanKernelDescriptorUsage, VulkanKernelScalarBinding, VulkanKernelScalarSource,
    VulkanLoadedReusableKernelArtifact, VulkanLoadedReusableKernelArtifactManifest,
    VulkanModelBoundaryBufferOverride, VulkanModelBoundaryDirection,
    VulkanPreparedDispatch, VulkanPreparedDispatchPlan, VulkanResidentFeedbackControlPlane,
    VulkanReusableKernelArtifactManifest,
};

const BF16_BYTE_COUNT: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedExecutionPlan {
    pub device_ids: Vec<String>,
    pub storage_buffer_offset_alignment: usize,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
    pub execution_islands: Vec<VulkanPhysicalExecutionIslandPlan>,
    pub shared_activation_route: VulkanSharedResidentBufferRoute,
    pub shared_input_byte_capacity: usize,
    pub shared_output_byte_capacity: usize,
    pub distributed_parameter_byte_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchSubmission {
    pub dependency_value: u64,
    pub consume_owner_ready_signal: bool,
    pub prepare_owner_continuation: bool,
    pub signal_completion: bool,
    pub use_feedback_indirect: bool,
}

impl VulkanDistributedExecutionPlan {
    pub fn from_prepared_plans(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanReusableKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        if storage_buffer_offset_alignment == 0
            || !storage_buffer_offset_alignment.is_power_of_two()
            || !storage_buffer_offset_alignment.is_multiple_of(BF16_BYTE_COUNT)
        {
            return Err(VulkanDistributedPlanError(format!(
                "distributed storage-buffer offset alignment {storage_buffer_offset_alignment} is invalid"
            )));
        }
        let mut dispatches = Vec::new();
        let mut shared_input_byte_capacity = 0usize;
        let mut shared_output_byte_capacity = 0usize;
        let mut distributed_parameter_byte_count = 0usize;
        let mut device_ids = BTreeSet::new();
        for (component_id, component_devices) in component_device_pools {
            validate_device_pool(component_devices)?;
            if component_devices.len() < 2 {
                return Err(VulkanDistributedPlanError(format!(
                    "internal sharding for component {component_id:?} requires at least two devices"
                )));
            }
            device_ids.extend(component_devices.iter().cloned());
        }
        let mut requested_components = component_device_pools
            .keys()
            .map(|component_id| (component_id.as_str(), false))
            .collect::<BTreeMap<_, _>>();

        for (owner_device_id, prepared_plan) in prepared_plans {
            for dispatch in &prepared_plan.dispatches {
                let Some(component_devices) =
                    component_device_pools.get(dispatch.component_id.as_str())
                else {
                    continue;
                };
                if !component_devices
                    .iter()
                    .any(|device_id| device_id == owner_device_id)
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "internal shard pool for component {:?} omits its owner device {:?}",
                        dispatch.component_id, owner_device_id
                    )));
                }
                let artifact = artifact_manifest
                    .artifacts
                    .iter()
                    .find(|artifact| artifact.family_id == dispatch.reusable_family_id)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed dispatch {}.{} has no artifact for family {:?}",
                            dispatch.component_id, dispatch.node_id, dispatch.reusable_family_id
                        ))
                    })?;
                let Some(contract) = select_distributed_contract(dispatch, artifact)? else {
                    continue;
                };
                let Some(planned) = plan_contract_dispatch(
                    owner_device_id,
                    dispatch,
                    tensor_index,
                    component_devices,
                    edge_placements,
                    artifact,
                    contract,
                    storage_buffer_offset_alignment,
                )?
                else {
                    continue;
                };
                shared_input_byte_capacity =
                    shared_input_byte_capacity.max(planned.input_byte_capacity);
                shared_output_byte_capacity =
                    shared_output_byte_capacity.max(planned.output_byte_capacity);
                distributed_parameter_byte_count = distributed_parameter_byte_count
                    .checked_add(planned.distributed_parameter_byte_count)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed parameter byte count overflowed".to_string(),
                        )
                    })?;
                dispatches.push(planned);
                *requested_components
                    .get_mut(dispatch.component_id.as_str())
                    .expect("component device pool was selected above") = true;
            }
        }
        if let Some((component_id, _)) = requested_components
            .iter()
            .find(|(_, was_planned)| !**was_planned)
        {
            let prepared_dispatches = prepared_plans
                .iter()
                .flat_map(|(owner_device_id, plan)| {
                    plan.dispatches.iter().filter_map(move |dispatch| {
                        (dispatch.component_id == *component_id).then(|| {
                            format!(
                                "{}.{}:{}@{}",
                                dispatch.component_id,
                                dispatch.node_id,
                                dispatch.op,
                                owner_device_id,
                            )
                        })
                    })
                })
                .collect::<Vec<_>>();
            return Err(VulkanDistributedPlanError(format!(
                "requested internal sharding for component {component_id:?} has no compatible distributable dispatch; prepared dispatches for that component: {prepared_dispatches:?}"
            )));
        }

        let shared_activation_route = VulkanSharedResidentBufferRoute::SharedHost;
        let execution_islands =
            resolved_physical_execution_islands(&dispatches, shared_activation_route)?;
        Ok(Self {
            device_ids: device_ids.into_iter().collect(),
            storage_buffer_offset_alignment,
            dispatches,
            execution_islands,
            shared_activation_route,
            shared_input_byte_capacity,
            shared_output_byte_capacity,
            distributed_parameter_byte_count,
        })
    }

    /// Retains an aligned, representative subset of every distributed
    /// dispatch while keeping one total parameter budget across all
    /// participants. The resulting plan runs the package's real shaders and
    /// descriptor contracts; only the dispatched output-row or expert ranges
    /// are reduced.
    pub fn sampled_for_parameter_budget(
        &self,
        tensor_index: &TensorIndex,
        participant_device_ids: &[String],
        maximum_total_parameter_bytes: usize,
    ) -> Result<Option<Self>, VulkanDistributedPlanError> {
        if participant_device_ids.is_empty()
            || maximum_total_parameter_bytes == 0
            || participant_device_ids.iter().any(String::is_empty)
            || participant_device_ids.iter().collect::<BTreeSet<_>>().len()
                != participant_device_ids.len()
        {
            return Err(VulkanDistributedPlanError(
                "sampled distributed execution requires distinct participants and a positive total parameter budget"
                    .to_string(),
            ));
        }
        if self.dispatches.is_empty() {
            return Ok(None);
        }
        let build = |fraction_millionths: usize| {
            sampled_distributed_execution_plan(
                self,
                participant_device_ids,
                fraction_millionths,
            )
        };
        let minimum = build(0)?;
        let minimum_bytes = VulkanDistributedParameterAllocationPlan::
            from_sampled_execution_plan(&minimum, tensor_index)?
            .total_byte_capacity;
        if minimum_bytes > maximum_total_parameter_bytes {
            return Ok(None);
        }
        const SCALE: usize = 1_000_000;
        let mut low = 0usize;
        let mut high = SCALE;
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            let candidate = build(middle)?;
            let bytes = VulkanDistributedParameterAllocationPlan::
                from_sampled_execution_plan(&candidate, tensor_index)?
                .total_byte_capacity;
            if bytes <= maximum_total_parameter_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(Some(build(low)?))
    }
}

fn sampled_distributed_execution_plan(
    source: &VulkanDistributedExecutionPlan,
    participant_device_ids: &[String],
    fraction_millionths: usize,
) -> Result<VulkanDistributedExecutionPlan, VulkanDistributedPlanError> {
    const SCALE: usize = 1_000_000;
    let dispatches = source
        .dispatches
        .iter()
        .map(|dispatch| {
            let source_shards = if participant_device_ids.len() == 1 {
                vec![merged_distributed_dispatch_shard(dispatch)?]
            } else {
                dispatch
                    .shards
                    .iter()
                    .take(participant_device_ids.len())
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let selected = source_shards
                .iter()
                .take(participant_device_ids.len())
                .zip(participant_device_ids)
                .map(|(shard, device_id)| {
                    sampled_distributed_dispatch_shard(
                        dispatch,
                        shard,
                        device_id,
                        fraction_millionths.min(SCALE),
                        SCALE,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            if selected.len() != participant_device_ids.len() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has {} physical shards but {} participants were requested",
                    dispatch.component_id,
                    dispatch.node_id,
                    dispatch.shards.len(),
                    participant_device_ids.len(),
                )));
            }
            let mut sampled = dispatch.clone();
            sampled.owner_device_id = participant_device_ids[0].clone();
            sampled.distributed_parameter_byte_count = selected
                .iter()
                .flat_map(|shard| &shard.parameters)
                .try_fold(0usize, |total, fragment| {
                    total.checked_add(fragment.byte_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "sampled distributed parameter bytes overflowed".to_string(),
                        )
                    })
                })?;
            sampled.shards = selected;
            Ok(sampled)
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    let distributed_parameter_byte_count = dispatches.iter().try_fold(
        0usize,
        |total, dispatch| {
            total
                .checked_add(dispatch.distributed_parameter_byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "sampled distributed parameter total overflowed".to_string(),
                    )
                })
        },
    )?;
    let execution_islands =
        resolved_physical_execution_islands(&dispatches, source.shared_activation_route)?;
    Ok(VulkanDistributedExecutionPlan {
        device_ids: participant_device_ids.to_vec(),
        storage_buffer_offset_alignment: source.storage_buffer_offset_alignment,
        dispatches,
        execution_islands,
        shared_activation_route: source.shared_activation_route,
        shared_input_byte_capacity: source.shared_input_byte_capacity,
        shared_output_byte_capacity: source.shared_output_byte_capacity,
        distributed_parameter_byte_count,
    })
}

fn merged_distributed_dispatch_shard(
    dispatch: &VulkanDistributedDispatchPlan,
) -> Result<VulkanDistributedDispatchShard, VulkanDistributedPlanError> {
    let first = dispatch.shards.first().ok_or_else(|| {
        VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has no shards to merge",
            dispatch.component_id, dispatch.node_id,
        ))
    })?;
    let row_count = dispatch.shards.iter().try_fold(0usize, |total, shard| {
        total.checked_add(shard.row_count).ok_or_else(|| {
            VulkanDistributedPlanError(
                "merged distributed row count overflowed".to_string(),
            )
        })
    })?;
    let merge_ranges = |ranges: Vec<&VulkanDistributedActivationRange>, label: &str| {
        let leading = ranges.first().copied().ok_or_else(|| {
            VulkanDistributedPlanError(format!(
                "distributed dispatch {}.{} has no {label} ranges",
                dispatch.component_id, dispatch.node_id,
            ))
        })?;
        if ranges.iter().all(|range| *range == leading) {
            return Ok(leading.clone());
        }
        let mut byte_end = leading
            .byte_offset
            .checked_add(leading.byte_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "merged distributed {label} range overflowed",
                ))
            })?;
        for range in ranges.iter().skip(1) {
            if range.byte_offset != byte_end {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has non-contiguous {label} ranges",
                    dispatch.component_id, dispatch.node_id,
                )));
            }
            byte_end = byte_end.checked_add(range.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "merged distributed {label} range overflowed",
                ))
            })?;
        }
        Ok(VulkanDistributedActivationRange {
            byte_offset: leading.byte_offset,
            byte_count: byte_end - leading.byte_offset,
        })
    };
    let input_range = merge_ranges(
        dispatch.shards.iter().map(|shard| &shard.input_range).collect(),
        "input",
    )?;
    let auxiliary_count = first.auxiliary_input_ranges.len();
    if dispatch
        .shards
        .iter()
        .any(|shard| shard.auxiliary_input_ranges.len() != auxiliary_count)
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has inconsistent auxiliary ranges",
            dispatch.component_id, dispatch.node_id,
        )));
    }
    let auxiliary_input_ranges = (0..auxiliary_count)
        .map(|index| {
            merge_ranges(
                dispatch
                    .shards
                    .iter()
                    .map(|shard| &shard.auxiliary_input_ranges[index])
                    .collect(),
                "auxiliary input",
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut parameters = BTreeMap::<(usize, String), VulkanDistributedParameterFragment>::new();
    for fragment in dispatch.shards.iter().flat_map(|shard| &shard.parameters) {
        let key = (fragment.binding, fragment.tensor.clone());
        if let Some(merged) = parameters.get_mut(&key) {
            let expected_offset = merged
                .byte_offset
                .checked_add(merged.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "merged distributed parameter range overflowed".to_string(),
                    )
                })?;
            if fragment.byte_offset != expected_offset {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed parameter {:?} binding {} has non-contiguous shard ranges",
                    fragment.tensor, fragment.binding,
                )));
            }
            merged.byte_count = merged
                .byte_count
                .checked_add(fragment.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "merged distributed parameter bytes overflowed".to_string(),
                    )
                })?;
        } else {
            parameters.insert(key, fragment.clone());
        }
    }
    let (workgroup_count_x, output_byte_offset, output_byte_count) = match dispatch.distribution {
        VulkanDistributedDispatchDistribution::OutputRows => (
            dispatch.shards.iter().try_fold(0u32, |total, shard| {
                total.checked_add(shard.workgroup_count_x).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "merged distributed workgroup count overflowed".to_string(),
                    )
                })
            })?,
            first.output_byte_offset,
            dispatch.shards.iter().try_fold(0usize, |total, shard| {
                total.checked_add(shard.output_byte_count).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "merged distributed output bytes overflowed".to_string(),
                    )
                })
            })?,
        ),
        VulkanDistributedDispatchDistribution::InputColumns
        | VulkanDistributedDispatchDistribution::ExpertRange => (
            first.workgroup_count_x,
            first.output_byte_offset,
            first.output_byte_count,
        ),
    };
    Ok(VulkanDistributedDispatchShard {
        device_id: first.device_id.clone(),
        row_start: first.row_start,
        row_count,
        workgroup_count_x,
        base_workgroup_z: first.base_workgroup_z,
        input_range,
        auxiliary_input_ranges,
        output_byte_offset,
        output_byte_count,
        parameters: parameters.into_values().collect(),
    })
}

fn sampled_distributed_dispatch_shard(
    dispatch: &VulkanDistributedDispatchPlan,
    source: &VulkanDistributedDispatchShard,
    device_id: &str,
    numerator: usize,
    denominator: usize,
) -> Result<VulkanDistributedDispatchShard, VulkanDistributedPlanError> {
    if source.row_count == 0
        || source.workgroup_count_x == 0
        || dispatch.row_alignment == 0
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has invalid shard geometry",
            dispatch.component_id, dispatch.node_id,
        )));
    }
    let proportional_rows = source
        .row_count
        .checked_mul(numerator)
        .ok_or_else(|| {
            VulkanDistributedPlanError(
                "sampled distributed row calculation overflowed".to_string(),
            )
        })?
        / denominator;
    let row_count = proportional_rows
        .min(source.row_count)
        .checked_div(dispatch.row_alignment)
        .expect("sampled distributed row alignment is positive")
        .max(1)
        .checked_mul(dispatch.row_alignment)
        .ok_or_else(|| {
            VulkanDistributedPlanError(
                "sampled distributed aligned row count overflowed".to_string(),
            )
        })?
        .min(source.row_count);
    let scale = |value: usize, label: &str| {
        value
            .checked_mul(row_count)
            .filter(|scaled| scaled.is_multiple_of(source.row_count))
            .map(|scaled| scaled / source.row_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "sampled distributed {label} is not row-aligned for {}.{}",
                    dispatch.component_id, dispatch.node_id,
                ))
            })
    };
    let scaled_activation_range =
        |range: &VulkanDistributedActivationRange,
         distribution: InputDistribution,
         label: &str| {
            if distribution == InputDistribution::Sharded {
                Ok(VulkanDistributedActivationRange {
                    byte_offset: range.byte_offset,
                    byte_count: scale(range.byte_count, label)?,
                })
            } else {
                Ok(range.clone())
            }
        };
    let input_range = scaled_activation_range(
        &source.input_range,
        dispatch.input_distribution,
        "primary input range",
    )?;
    if source.auxiliary_input_ranges.len() != dispatch.auxiliary_input_distributions.len() {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has {} sampled auxiliary ranges for {} declared distributions",
            dispatch.component_id,
            dispatch.node_id,
            source.auxiliary_input_ranges.len(),
            dispatch.auxiliary_input_distributions.len(),
        )));
    }
    let scaled_auxiliary_input_ranges = || {
        source
            .auxiliary_input_ranges
            .iter()
            .zip(&dispatch.auxiliary_input_distributions)
            .map(|(range, distribution)| {
                scaled_activation_range(range, *distribution, "auxiliary input range")
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let (workgroup_count_x, output_byte_count, auxiliary_input_ranges) =
        match dispatch.distribution {
            VulkanDistributedDispatchDistribution::OutputRows => {
                let output_byte_count = scale(source.output_byte_count, "output range")?;
                let workgroup_count_x = u32::try_from(scale(
                    source.workgroup_count_x as usize,
                    "workgroup count",
                )?)
                .map_err(|_| {
                    VulkanDistributedPlanError(
                        "sampled distributed workgroup count exceeds u32".to_string(),
                    )
                })?;
                let auxiliary = scaled_auxiliary_input_ranges()?;
                (workgroup_count_x, output_byte_count, auxiliary)
            }
            VulkanDistributedDispatchDistribution::InputColumns => (
                source.workgroup_count_x,
                source.output_byte_count,
                scaled_auxiliary_input_ranges()?,
            ),
            VulkanDistributedDispatchDistribution::ExpertRange => (
                source.workgroup_count_x,
                source.output_byte_count,
                source.auxiliary_input_ranges.clone(),
            ),
        };
    let parameters = source
        .parameters
        .iter()
        .map(|fragment| {
            Ok(VulkanDistributedParameterFragment {
                binding: fragment.binding,
                tensor: fragment.tensor.clone(),
                byte_offset: fragment.byte_offset,
                byte_count: scale(fragment.byte_count, "parameter fragment")?,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    Ok(VulkanDistributedDispatchShard {
        device_id: device_id.to_string(),
        row_start: source.row_start,
        row_count,
        workgroup_count_x,
        base_workgroup_z: source.base_workgroup_z,
        input_range,
        auxiliary_input_ranges,
        output_byte_offset: source.output_byte_offset,
        output_byte_count,
        parameters,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionIslandPlan {
    pub island_id: String,
    pub component_id: String,
    pub member_node_ids: Vec<String>,
    pub contract_ids: Vec<String>,
    pub implementation_digests: Vec<String>,
    pub phase_schedules: Vec<VulkanPhysicalExecutionPhaseSchedule>,
    pub entry_device_id: String,
    pub exit_device_id: String,
    pub owner_device_id: String,
    pub participants: Vec<VulkanPhysicalExecutionParticipant>,
    pub shard_assignments: Vec<VulkanPhysicalExecutionShardAssignment>,
    pub transport_routes: Vec<VulkanPhysicalExecutionTransportRoute>,
    pub synchronization_routes: Vec<VulkanPhysicalExecutionSynchronizationRoute>,
    pub residency: Vec<VulkanPhysicalExecutionResidencyRequirement>,
    pub transient_memory: Vec<VulkanPhysicalExecutionTransientMemoryRequirement>,
    pub dispatches: Vec<VulkanDistributedDispatchPlan>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPhysicalExecutionParticipantRole {
    Coordinator,
    ShardWorker,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionParticipant {
    pub device_id: String,
    pub roles: BTreeSet<VulkanPhysicalExecutionParticipantRole>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionShardAssignment {
    pub dispatch_index: usize,
    pub node_id: String,
    pub device_id: String,
    pub distribution: VulkanDistributedDispatchDistribution,
    pub logical_start: usize,
    pub logical_count: usize,
    pub parameter_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPhysicalExecutionTransportKind {
    ExternalDeviceLocal,
    SharedHost,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanPhysicalExecutionTransportRoute {
    pub source_device_id: String,
    pub destination_device_id: String,
    pub byte_capacity: usize,
    pub kind: VulkanPhysicalExecutionTransportKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPhysicalExecutionSynchronizationKind {
    TimelineSemaphore,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanPhysicalExecutionSynchronizationRoute {
    pub source_device_id: String,
    pub destination_device_id: String,
    pub kind: VulkanPhysicalExecutionSynchronizationKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanPhysicalExecutionScheduleKind {
    PublishInputs,
    ExecuteShards,
    CollectOutputs,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionScheduleStep {
    pub ordinal: usize,
    pub dispatch_index: usize,
    pub kind: VulkanPhysicalExecutionScheduleKind,
    pub device_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionPhaseSchedule {
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub steps: Vec<VulkanPhysicalExecutionScheduleStep>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPhysicalExecutionResidencyKind {
    PermanentParameterShard,
    OwnerState,
    OwnerControl,
    OwnerSelectionTelemetry,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionResidencyRequirement {
    pub device_id: String,
    pub kind: VulkanPhysicalExecutionResidencyKind,
    pub resource_id: String,
    pub byte_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanPhysicalExecutionTransientMemoryKind {
    SharedActivationAllocation,
    PrivateShardIntermediate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanPhysicalExecutionTransientMemoryRequirement {
    pub allocation_device_id: String,
    pub kind: VulkanPhysicalExecutionTransientMemoryKind,
    pub resource_id: String,
    pub fixed_byte_capacity: usize,
    pub per_lane_byte_capacity: usize,
}

impl VulkanPhysicalExecutionIslandPlan {
    pub fn leader(&self) -> &VulkanDistributedDispatchPlan {
        self.dispatches
            .first()
            .expect("distributed dispatch groups are never empty")
    }

    pub fn tail(&self) -> &VulkanDistributedDispatchPlan {
        self.dispatches
            .last()
            .expect("distributed dispatch groups are never empty")
    }

    pub fn contains_dispatch(&self, dispatch_index: usize) -> bool {
        self.dispatches
            .iter()
            .any(|dispatch| dispatch.dispatch_index == dispatch_index)
    }

    pub fn dispatch_indices(&self) -> Vec<usize> {
        self.dispatches
            .iter()
            .map(|dispatch| dispatch.dispatch_index)
            .collect()
    }
}

pub(crate) fn resolved_physical_execution_islands(
    dispatches: &[VulkanDistributedDispatchPlan],
    shared_activation_route: VulkanSharedResidentBufferRoute,
) -> Result<Vec<VulkanPhysicalExecutionIslandPlan>, VulkanDistributedPlanError> {
    let mut groups = Vec::<Vec<VulkanDistributedDispatchPlan>>::new();
    for dispatch in dispatches {
        if let Some(group) = groups.last_mut()
            && distributed_dispatches_can_share_sequence(
                group.last().expect("physical execution islands are never empty"),
                dispatch,
            )
        {
            group.push(dispatch.clone());
        } else {
            groups.push(vec![dispatch.clone()]);
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(index, dispatches)| {
            resolved_physical_execution_island(index, dispatches, shared_activation_route)
        })
        .collect()
}

fn resolved_physical_execution_island(
    island_index: usize,
    dispatches: Vec<VulkanDistributedDispatchPlan>,
    shared_activation_route: VulkanSharedResidentBufferRoute,
) -> Result<VulkanPhysicalExecutionIslandPlan, VulkanDistributedPlanError> {
    let Some(first) = dispatches.first() else {
        return Err(VulkanDistributedPlanError(
            "physical execution island must not be empty".to_string(),
        ));
    };
    let owner_device_id = first.owner_device_id.clone();
    let component_id = first.component_id.clone();
    if dispatches.iter().any(|dispatch| {
        dispatch.owner_device_id != owner_device_id || dispatch.component_id != component_id
    }) {
        return Err(VulkanDistributedPlanError(
            "physical execution island crosses an owner or logical component boundary".to_string(),
        ));
    }
    if dispatches.iter().any(|dispatch| {
        dispatch.has_lazy_resource_requirements
    }) {
        return Err(VulkanDistributedPlanError(format!(
            "physical execution island for component {component_id:?} contains lazy resources without a resolved atomic residency plan",
        )));
    }

    let mut participant_roles = BTreeMap::<
        String,
        BTreeSet<VulkanPhysicalExecutionParticipantRole>,
    >::new();
    participant_roles
        .entry(owner_device_id.clone())
        .or_default()
        .insert(VulkanPhysicalExecutionParticipantRole::Coordinator);
    let mut shard_assignments = Vec::new();
    let mut parameter_bytes_by_device = BTreeMap::<String, usize>::new();
    let mut transport_routes = BTreeSet::new();
    let mut synchronization_routes = BTreeSet::new();
    let mut schedule = Vec::new();
    let mut member_node_ids = Vec::new();
    let mut contract_ids = Vec::new();
    let mut implementation_digests = Vec::new();
    let mut shared_activation_allocations = BTreeMap::<
        (String, VulkanDistributedActivationStorage, String, usize),
        usize,
    >::new();
    let mut private_intermediate_allocations =
        BTreeMap::<(String, String, usize), usize>::new();
    let mut owner_residency = BTreeMap::<
        (String, VulkanPhysicalExecutionResidencyKind, String),
        usize,
    >::new();

    for (dispatch_offset, dispatch) in dispatches.iter().enumerate() {
        for node_id in &dispatch.contract_member_node_ids {
            if !member_node_ids.contains(node_id) {
                member_node_ids.push(node_id.clone());
            }
        }
        let contract_id = &dispatch.physical_execution_contract_id;
        if !contract_ids.contains(contract_id) {
            contract_ids.push(contract_id.clone());
        }
        let implementation_digest = &dispatch.implementation_digest;
        if !implementation_digests.contains(implementation_digest) {
            implementation_digests.push(implementation_digest.clone());
        }
        for requirement in &dispatch.owner_residency_requirements {
            let key = (
                requirement.device_id.clone(),
                requirement.kind,
                requirement.resource_id.clone(),
            );
            if let Some(existing) = owner_residency.insert(key, requirement.byte_capacity)
                && existing != requirement.byte_capacity
            {
                return Err(VulkanDistributedPlanError(format!(
                    "physical execution island resource {:?} has conflicting capacities {existing} and {}",
                    requirement.resource_id,
                    requirement.byte_capacity,
                )));
            }
        }
        for activation in std::iter::once(&dispatch.input_activation)
            .chain(&dispatch.auxiliary_input_activations)
            .chain(std::iter::once(&dispatch.output_activation))
        {
            let allocation_device_id = distributed_activation_owner_device_id(
                &dispatch.owner_device_id,
                activation,
            );
            let key = (
                allocation_device_id,
                activation.storage.clone(),
                activation.component_id.clone(),
                activation.slot,
            );
            if let Some(existing) = shared_activation_allocations.insert(
                key,
                activation.byte_capacity,
            ) && existing != activation.byte_capacity
            {
                return Err(VulkanDistributedPlanError(format!(
                    "physical execution island activation {}.slot_{} has conflicting capacities {existing} and {}",
                    activation.component_id,
                    activation.slot,
                    activation.byte_capacity,
                )));
            }
        }

        let mut worker_device_ids = Vec::new();
        for shard in &dispatch.shards {
            participant_roles
                .entry(shard.device_id.clone())
                .or_default()
                .insert(VulkanPhysicalExecutionParticipantRole::ShardWorker);
            if !worker_device_ids.contains(&shard.device_id) {
                worker_device_ids.push(shard.device_id.clone());
            }
            let parameter_bytes = shard.parameters.iter().try_fold(
                0usize,
                |total, parameter| {
                    total.checked_add(parameter.byte_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "physical execution shard parameter bytes overflowed".to_string(),
                        )
                    })
                },
            )?;
            let entry = parameter_bytes_by_device
                .entry(shard.device_id.clone())
                .or_default();
            *entry = entry.checked_add(parameter_bytes).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "physical execution participant parameter bytes overflowed".to_string(),
                )
            })?;
            shard_assignments.push(VulkanPhysicalExecutionShardAssignment {
                dispatch_index: dispatch.dispatch_index,
                node_id: dispatch.node_id.clone(),
                device_id: shard.device_id.clone(),
                distribution: dispatch.distribution,
                logical_start: shard.row_start,
                logical_count: shard.row_count,
                parameter_bytes,
            });
            if shard.device_id != owner_device_id {
                synchronization_routes.insert(VulkanPhysicalExecutionSynchronizationRoute {
                    source_device_id: owner_device_id.clone(),
                    destination_device_id: shard.device_id.clone(),
                    kind: VulkanPhysicalExecutionSynchronizationKind::TimelineSemaphore,
                });
                synchronization_routes.insert(VulkanPhysicalExecutionSynchronizationRoute {
                    source_device_id: shard.device_id.clone(),
                    destination_device_id: owner_device_id.clone(),
                    kind: VulkanPhysicalExecutionSynchronizationKind::TimelineSemaphore,
                });
            }
        }
        if dispatch_offset == 0 {
            schedule.push(VulkanPhysicalExecutionScheduleStep {
                ordinal: schedule.len(),
                dispatch_index: dispatch.dispatch_index,
                kind: VulkanPhysicalExecutionScheduleKind::PublishInputs,
                device_ids: worker_device_ids.clone(),
            });
        }
        schedule.push(VulkanPhysicalExecutionScheduleStep {
            ordinal: schedule.len(),
            dispatch_index: dispatch.dispatch_index,
            kind: VulkanPhysicalExecutionScheduleKind::ExecuteShards,
            device_ids: worker_device_ids.clone(),
        });
        if dispatch_offset + 1 == dispatches.len() {
            schedule.push(VulkanPhysicalExecutionScheduleStep {
                ordinal: schedule.len(),
                dispatch_index: dispatch.dispatch_index,
                kind: VulkanPhysicalExecutionScheduleKind::CollectOutputs,
                device_ids: vec![owner_device_id.clone()],
            });
        }
    }

    let participants = participant_roles
        .into_iter()
        .map(|(device_id, roles)| VulkanPhysicalExecutionParticipant { device_id, roles })
        .collect::<Vec<_>>();
    let entry_device_id = distributed_activation_owner_device_id(
        &owner_device_id,
        &first.input_activation,
    );
    let tail = dispatches
        .last()
        .expect("physical execution island was checked above");
    let exit_device_id =
        distributed_activation_owner_device_id(&owner_device_id, &tail.output_activation);
    if let Some(collect) = schedule.iter_mut().rev().find(|step| {
        step.kind == VulkanPhysicalExecutionScheduleKind::CollectOutputs
    }) {
        collect.device_ids = vec![exit_device_id.clone()];
    }
    for participant in &participants {
        if participant.device_id != entry_device_id {
            transport_routes.insert(VulkanPhysicalExecutionTransportRoute {
                source_device_id: entry_device_id.clone(),
                destination_device_id: participant.device_id.clone(),
                byte_capacity: first.input_byte_capacity,
                kind: physical_execution_transport_kind(shared_activation_route),
            });
        }
        if participant.device_id != exit_device_id {
            transport_routes.insert(VulkanPhysicalExecutionTransportRoute {
                source_device_id: participant.device_id.clone(),
                destination_device_id: exit_device_id.clone(),
                byte_capacity: tail.output_byte_capacity,
                kind: physical_execution_transport_kind(shared_activation_route),
            });
        }
    }
    let mut residency = parameter_bytes_by_device
        .into_iter()
        .map(|(device_id, byte_capacity)| VulkanPhysicalExecutionResidencyRequirement {
            device_id,
            kind: VulkanPhysicalExecutionResidencyKind::PermanentParameterShard,
            resource_id: "parameter_shards".to_string(),
            byte_capacity,
        })
        .collect::<Vec<_>>();
    residency.extend(owner_residency.into_iter().map(
        |((device_id, kind, resource_id), byte_capacity)| {
            VulkanPhysicalExecutionResidencyRequirement {
                device_id,
                kind,
                resource_id,
                byte_capacity,
            }
        },
    ));
    for pair in dispatches.windows(2) {
        let producer = &pair[0];
        let consumer = &pair[1];
        if producer.output_activation.component_id == consumer.input_activation.component_id
            && producer.output_activation.slot == consumer.input_activation.slot
            && producer.output_activation.signal_id == consumer.input_activation.signal_id
        {
            for shard in &producer.shards {
                let key = (
                    shard.device_id.clone(),
                    producer.output_activation.component_id.clone(),
                    producer.output_activation.slot,
                );
                if let Some(existing) = private_intermediate_allocations.insert(
                    key,
                    producer.output_activation.signal_byte_capacity,
                ) && existing != producer.output_activation.signal_byte_capacity
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "physical execution private intermediate {}.slot_{} has conflicting capacities {existing} and {}",
                        producer.output_activation.component_id,
                        producer.output_activation.slot,
                        producer.output_activation.signal_byte_capacity,
                    )));
                }
            }
        }
    }
    let mut transient_memory = shared_activation_allocations
        .into_iter()
        .map(
            |((allocation_device_id, storage, component_id, slot), per_lane_byte_capacity)| {
                VulkanPhysicalExecutionTransientMemoryRequirement {
                    allocation_device_id,
                    kind: VulkanPhysicalExecutionTransientMemoryKind::SharedActivationAllocation,
                    resource_id: format!("activation:{storage:?}:{component_id}:slot_{slot}"),
                    fixed_byte_capacity: 0,
                    per_lane_byte_capacity,
                }
            },
        )
        .collect::<Vec<_>>();
    transient_memory.extend(private_intermediate_allocations.into_iter().map(
        |((allocation_device_id, component_id, slot), per_lane_byte_capacity)| {
            VulkanPhysicalExecutionTransientMemoryRequirement {
                allocation_device_id,
                kind: VulkanPhysicalExecutionTransientMemoryKind::PrivateShardIntermediate,
                resource_id: format!("private_activation:{component_id}:slot_{slot}"),
                fixed_byte_capacity: 0,
                per_lane_byte_capacity,
            }
        },
    ));
    let tail_dispatch_index = tail.dispatch_index;
    Ok(VulkanPhysicalExecutionIslandPlan {
        island_id: format!(
            "{component_id}:{}-{tail_dispatch_index}:island_{island_index}",
            first.dispatch_index,
        ),
        component_id,
        member_node_ids,
        contract_ids,
        implementation_digests,
        phase_schedules: vec![VulkanPhysicalExecutionPhaseSchedule {
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            steps: schedule,
        }],
        entry_device_id,
        exit_device_id,
        owner_device_id,
        participants,
        shard_assignments,
        transport_routes: transport_routes.into_iter().collect(),
        synchronization_routes: synchronization_routes.into_iter().collect(),
        residency,
        transient_memory,
        dispatches,
    })
}

fn physical_execution_transport_kind(
    route: VulkanSharedResidentBufferRoute,
) -> VulkanPhysicalExecutionTransportKind {
    match route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => {
            VulkanPhysicalExecutionTransportKind::ExternalDeviceLocal
        }
        VulkanSharedResidentBufferRoute::SharedHost => {
            VulkanPhysicalExecutionTransportKind::SharedHost
        }
    }
}

fn distributed_activation_owner_device_id(
    default_owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
) -> String {
    match &activation.storage {
        VulkanDistributedActivationStorage::Edge {
            owner_device_id, ..
        } => owner_device_id.clone(),
        VulkanDistributedActivationStorage::ActivationSlot
        | VulkanDistributedActivationStorage::BoundaryInput
        | VulkanDistributedActivationStorage::BoundaryOutput => {
            default_owner_device_id.to_string()
        }
    }
}

fn distributed_dispatches_can_share_sequence(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    producer.owner_device_id == consumer.owner_device_id
        && producer.component_id == consumer.component_id
        && producer.dispatch_index.checked_add(1) == Some(consumer.dispatch_index)
        && producer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && consumer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && same_distributed_activation(&producer.output_activation, &consumer.input_activation)
        && producer.shards.len() == consumer.shards.len()
        && producer
            .shards
            .iter()
            .zip(&consumer.shards)
            .all(|(producer, consumer)| {
                producer.device_id == consumer.device_id
                    && producer.row_start == consumer.row_start
                    && producer.row_count == consumer.row_count
                    && producer.base_workgroup_z == consumer.base_workgroup_z
            })
}

fn same_distributed_activation(
    left: &VulkanDistributedActivationSlot,
    right: &VulkanDistributedActivationSlot,
) -> bool {
    left.component_id == right.component_id
        && left.signal_id == right.signal_id
        && left.slot == right.slot
        && left.byte_capacity == right.byte_capacity
        && left.signal_byte_capacity == right.signal_byte_capacity
        && left.storage == right.storage
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchPlan {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub reusable_family_id: String,
    pub physical_execution_contract_id: String,
    pub implementation_digest: String,
    pub contract_member_node_ids: Vec<String>,
    pub has_lazy_resource_requirements: bool,
    pub owner_residency_requirements: Vec<VulkanPhysicalExecutionResidencyRequirement>,
    pub input_byte_capacity: usize,
    pub output_byte_capacity: usize,
    pub output_rows: usize,
    pub input_width: usize,
    pub row_alignment: usize,
    pub input_activation: VulkanDistributedActivationSlot,
    pub input_distribution: InputDistribution,
    pub auxiliary_input_activations: Vec<VulkanDistributedActivationSlot>,
    pub auxiliary_input_distributions: Vec<InputDistribution>,
    pub output_activation: VulkanDistributedActivationSlot,
    pub output_collection: OutputCollection,
    pub reduction: Option<VulkanDistributedReductionPlan>,
    pub distribution: VulkanDistributedDispatchDistribution,
    pub distributed_parameter_byte_count: usize,
    pub shards: Vec<VulkanDistributedDispatchShard>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedReductionPlan {
    pub operation: ReductionOperation,
    pub element_count: usize,
    pub partial_byte_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanDistributedDispatchDistribution {
    OutputRows,
    InputColumns,
    ExpertRange,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationSlot {
    pub binding: usize,
    pub component_id: String,
    pub signal_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_byte_capacity: usize,
    pub storage: VulkanDistributedActivationStorage,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanDistributedActivationStorage {
    ActivationSlot,
    BoundaryInput,
    BoundaryOutput,
    Edge {
        edge_index: usize,
        owner_device_id: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferPlan {
    pub allocations: Vec<VulkanDistributedActivationBufferAllocation>,
    pub reduction_allocations: Vec<VulkanDistributedReductionBufferAllocation>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub reference_count: usize,
    pub total_shared_byte_capacity: usize,
    pub route: VulkanSharedResidentBufferRoute,
}

impl VulkanDistributedActivationBufferPlan {
    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedActivationBufferAllocationKey,
            VulkanDistributedActivationBufferAllocation,
        >::new();
        let mut reduction_allocations = Vec::new();
        let mut reduction_keys = BTreeSet::new();

        for dispatch in &execution_plan.dispatches {
            let participant_device_ids = dispatch
                .shards
                .iter()
                .map(|shard| shard.device_id.as_str())
                .collect::<BTreeSet<_>>();
            if participant_device_ids.is_empty() {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has no device shards",
                    dispatch.component_id, dispatch.node_id
                )));
            }
            if !participant_device_ids.contains(dispatch.owner_device_id.as_str()) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} does not include its owner {:?}",
                    dispatch.component_id, dispatch.node_id, dispatch.owner_device_id
                )));
            }
            if let Some(device_id) = participant_device_ids
                .iter()
                .find(|device_id| !device_ids.contains(**device_id))
            {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} uses device {device_id:?} outside the execution pool",
                    dispatch.component_id, dispatch.node_id
                )));
            }

            accumulate_activation_allocation(
                &mut allocations,
                &dispatch.owner_device_id,
                &dispatch.input_activation,
                &participant_device_ids,
                VulkanDistributedActivationAccess::Input,
            )?;
            for activation in &dispatch.auxiliary_input_activations {
                accumulate_activation_allocation(
                    &mut allocations,
                    &dispatch.owner_device_id,
                    activation,
                    &participant_device_ids,
                    VulkanDistributedActivationAccess::Input,
                )?;
            }
            let output_participant_device_ids = if dispatch.reduction.is_some() {
                BTreeSet::from([dispatch.owner_device_id.as_str()])
            } else {
                participant_device_ids.clone()
            };
            accumulate_activation_allocation(
                &mut allocations,
                &dispatch.owner_device_id,
                &dispatch.output_activation,
                &output_participant_device_ids,
                VulkanDistributedActivationAccess::Output,
            )?;
            if let Some(reduction) = &dispatch.reduction {
                if !reduction_keys.insert((
                    dispatch.owner_device_id.as_str(),
                    dispatch.dispatch_index,
                )) {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed reduction repeats dispatch {} owned by {:?}",
                        dispatch.dispatch_index, dispatch.owner_device_id
                    )));
                }
                let device_ids = dispatch
                    .shards
                    .iter()
                    .map(|shard| shard.device_id.clone())
                    .collect::<Vec<_>>();
                if device_ids.iter().collect::<BTreeSet<_>>().len() != device_ids.len() {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed reduction {}.{} repeats a participant device",
                        dispatch.component_id, dispatch.node_id
                    )));
                }
                let byte_capacity = reduction
                    .partial_byte_capacity
                    .checked_mul(device_ids.len())
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed reduction {}.{} byte capacity overflowed",
                            dispatch.component_id, dispatch.node_id
                        ))
                    })?;
                reduction_allocations.push(VulkanDistributedReductionBufferAllocation {
                    owner_device_id: dispatch.owner_device_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    component_id: dispatch.component_id.clone(),
                    node_id: dispatch.node_id.clone(),
                    plane_byte_capacity: reduction.partial_byte_capacity,
                    byte_capacity,
                    device_ids,
                });
            }
        }

        let activation_import_count =
            allocations.values().try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.device_ids.len())
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed activation import count overflowed".to_string(),
                        )
                    })
            })?;
        let reduction_import_count =
            reduction_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.device_ids.len()).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed reduction import count overflowed".to_string(),
                        )
                    })
                })?;
        let import_count = activation_import_count
            .checked_add(reduction_import_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer import count overflowed".to_string(),
                )
            })?;
        let activation_reference_count =
            allocations.values().try_fold(0usize, |total, allocation| {
                total
                    .checked_add(allocation.input_use_count)
                    .and_then(|count| count.checked_add(allocation.output_use_count))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed activation reference count overflowed".to_string(),
                        )
                    })
            })?;
        let reduction_reference_count = reduction_allocations.iter().try_fold(
            0usize,
            |total, allocation| {
                total
                    .checked_add(allocation.device_ids.len())
                    .and_then(|count| count.checked_add(1))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed reduction reference count overflowed".to_string(),
                        )
                    })
            },
        )?;
        let reference_count = activation_reference_count
            .checked_add(reduction_reference_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer reference count overflowed".to_string(),
                )
            })?;
        let activation_byte_capacity =
            allocations.values().try_fold(0usize, |total, allocation| {
                total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })
            })?;
        let reduction_byte_capacity =
            reduction_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_capacity).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "distributed reduction byte capacity overflowed".to_string(),
                        )
                    })
                })?;
        let total_shared_byte_capacity = activation_byte_capacity
            .checked_add(reduction_byte_capacity)
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer byte capacity overflowed".to_string(),
                )
            })?;
        let allocations = allocations.into_values().collect::<Vec<_>>();

        let allocation_count = allocations
            .len()
            .checked_add(reduction_allocations.len())
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed buffer allocation count overflowed".to_string(),
                )
            })?;
        Ok(Self {
            allocation_count,
            allocations,
            reduction_allocations,
            import_count,
            reference_count,
            total_shared_byte_capacity,
            route: execution_plan.shared_activation_route,
        })
    }

    pub fn allocation(
        &self,
        owner_device_id: &str,
        component_id: &str,
        slot: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            allocation.storage == VulkanDistributedActivationStorage::ActivationSlot
                && allocation.owner_device_id == owner_device_id
                && allocation.component_id == component_id
                && allocation.slot == slot
        })
    }

    pub fn edge_allocation(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanDistributedActivationBufferAllocation> {
        self.allocations.iter().find(|allocation| {
            matches!(
                allocation.storage,
                VulkanDistributedActivationStorage::Edge {
                    edge_index: candidate,
                    ..
                } if candidate == edge_index
            )
        })
    }

    pub fn reduction_allocation(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedReductionBufferAllocation> {
        self.reduction_allocations.iter().find(|allocation| {
            allocation.owner_device_id == owner_device_id
                && allocation.dispatch_index == dispatch_index
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferAllocation {
    pub storage: VulkanDistributedActivationStorage,
    pub owner_device_id: String,
    pub component_id: String,
    pub slot: usize,
    pub byte_capacity: usize,
    pub signal_ids: Vec<String>,
    pub device_ids: Vec<String>,
    pub input_use_count: usize,
    pub output_use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedReductionBufferAllocation {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub plane_byte_capacity: usize,
    pub byte_capacity: usize,
    pub device_ids: Vec<String>,
}

pub struct VulkanDistributedActivationBuffers {
    pub plan: VulkanDistributedActivationBufferPlan,
    pub lane_capacity: usize,
    pub allocations: Vec<VulkanDistributedActivationBuffer>,
    pub reduction_allocations: Vec<VulkanDistributedReductionBuffer>,
    pub allocation_count: usize,
    pub import_count: usize,
    pub total_shared_byte_capacity: usize,
}

impl VulkanDistributedActivationBuffers {
    pub fn allocate<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        Self::allocate_for_lanes(plan, 1, device_for)
    }

    pub fn allocate_for_lanes<'a, F, E>(
        plan: &VulkanDistributedActivationBufferPlan,
        lane_capacity: usize,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedActivationBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        if lane_capacity == 0 {
            return Err(VulkanDistributedActivationBufferError(
                "distributed activation lane capacity must not be zero".to_string(),
            ));
        }
        let mut allocations = Vec::with_capacity(plan.allocations.len());
        let mut import_count = 0usize;
        let mut total_shared_byte_capacity = 0usize;
        for planned in &plan.allocations {
            let byte_capacity = planned
                .byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(format!(
                        "distributed activation {}.slot_{} lane capacity overflowed",
                        planned.component_id, planned.slot
                    ))
                })?;
            let shared = allocate_distributed_shared_buffer(
                &planned.owner_device_id,
                &planned.device_ids,
                byte_capacity,
                plan.route,
                &format!("activation {}.slot_{}", planned.component_id, planned.slot),
                &mut device_for,
            )?;
            import_count = import_count
                .checked_add(shared.device_buffers.len())
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation import count overflowed".to_string(),
                    )
                })?;
            total_shared_byte_capacity = total_shared_byte_capacity
                .checked_add(byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed activation byte capacity overflowed".to_string(),
                    )
                })?;
            allocations.push(VulkanDistributedActivationBuffer {
                planned: planned.clone(),
                route: shared.route,
                external_device_local_error: shared.external_device_local_error,
                device_buffers: shared.device_buffers,
            });
        }
        let mut reduction_allocations = Vec::with_capacity(plan.reduction_allocations.len());
        for planned in &plan.reduction_allocations {
            let byte_capacity = planned
                .byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(format!(
                        "distributed reduction {}.{} lane capacity overflowed",
                        planned.component_id, planned.node_id
                    ))
                })?;
            let shared = allocate_distributed_shared_buffer(
                &planned.owner_device_id,
                &planned.device_ids,
                byte_capacity,
                plan.route,
                &format!("reduction {}.{}", planned.component_id, planned.node_id),
                &mut device_for,
            )?;
            import_count = import_count
                .checked_add(shared.device_buffers.len())
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed reduction import count overflowed".to_string(),
                    )
                })?;
            total_shared_byte_capacity = total_shared_byte_capacity
                .checked_add(byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedActivationBufferError(
                        "distributed reduction byte capacity overflowed".to_string(),
                    )
                })?;
            reduction_allocations.push(VulkanDistributedReductionBuffer {
                planned: planned.clone(),
                route: shared.route,
                external_device_local_error: shared.external_device_local_error,
                device_buffers: shared.device_buffers,
            });
        }

        Ok(Self {
            plan: plan.clone(),
            lane_capacity,
            allocation_count: plan.allocation_count,
            allocations,
            reduction_allocations,
            import_count,
            total_shared_byte_capacity,
        })
    }

    pub fn activation_buffer(
        &self,
        dispatch_owner_device_id: &str,
        activation: &VulkanDistributedActivationSlot,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.allocations
            .iter()
            .find(|allocation| {
                distributed_activation_allocation_matches(
                    dispatch_owner_device_id,
                    activation,
                    &allocation.planned,
                )
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn activation_overrides_for_owner_device(
        &self,
        owner_device_id: &str,
    ) -> Vec<VulkanActivationSlotBufferOverride> {
        self.allocations
            .iter()
            .filter(|allocation| {
                allocation.planned.storage
                    == VulkanDistributedActivationStorage::ActivationSlot
                    && allocation.planned.owner_device_id == owner_device_id
            })
            .filter_map(|allocation| {
                allocation
                    .device_buffers
                    .get(owner_device_id)
                    .map(|buffer| VulkanActivationSlotBufferOverride {
                        component_id: allocation.planned.component_id.clone(),
                        slot: allocation.planned.slot,
                        buffer: Arc::clone(buffer),
                    })
            })
            .collect()
    }

    pub fn boundary_overrides_for_owner_device(
        &self,
        owner_device_id: &str,
    ) -> Vec<VulkanModelBoundaryBufferOverride> {
        self.allocations
            .iter()
            .filter(|allocation| allocation.planned.owner_device_id == owner_device_id)
            .filter_map(|allocation| {
                let direction = match allocation.planned.storage {
                    VulkanDistributedActivationStorage::BoundaryInput => {
                        VulkanModelBoundaryDirection::Input
                    }
                    VulkanDistributedActivationStorage::BoundaryOutput => {
                        VulkanModelBoundaryDirection::Output
                    }
                    _ => return None,
                };
                allocation.device_buffers.get(owner_device_id).and_then(|buffer| {
                    Some(VulkanModelBoundaryBufferOverride {
                        direction,
                        component_id: allocation.planned.component_id.clone(),
                        signal_id: allocation.planned.signal_ids.first()?.clone(),
                        buffer: Arc::clone(buffer),
                    })
                })
            })
            .collect()
    }

    pub fn edge_buffer(
        &self,
        edge_index: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.edge_allocation(edge_index)
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub fn reduction_partial_buffer(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        device_id: &str,
    ) -> Option<&Arc<VulkanResidentBuffer>> {
        self.reduction_allocations
            .iter()
            .find(|allocation| {
                allocation.planned.owner_device_id == owner_device_id
                    && allocation.planned.dispatch_index == dispatch_index
            })
            .and_then(|allocation| allocation.device_buffers.get(device_id))
    }

    pub(crate) fn edge_allocation(
        &self,
        edge_index: usize,
    ) -> Option<&VulkanDistributedActivationBuffer> {
        self.allocations.iter().find(|allocation| {
            matches!(
                allocation.planned.storage,
                VulkanDistributedActivationStorage::Edge {
                    edge_index: candidate,
                    ..
                } if candidate == edge_index
            )
        })
    }
}

pub struct VulkanDistributedActivationBuffer {
    pub planned: VulkanDistributedActivationBufferAllocation,
    pub route: VulkanSharedResidentBufferRoute,
    pub external_device_local_error: Option<String>,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

pub struct VulkanDistributedReductionBuffer {
    pub planned: VulkanDistributedReductionBufferAllocation,
    pub route: VulkanSharedResidentBufferRoute,
    pub external_device_local_error: Option<String>,
    pub device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationBufferError(pub String);

impl Display for VulkanDistributedActivationBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedActivationBufferError {}

struct VulkanDistributedSharedBufferAllocation {
    route: VulkanSharedResidentBufferRoute,
    external_device_local_error: Option<String>,
    device_buffers: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

fn allocate_distributed_shared_buffer<'a, F, E>(
    owner_device_id: &str,
    device_ids: &[String],
    byte_capacity: usize,
    route: VulkanSharedResidentBufferRoute,
    label: &str,
    device_for: &mut F,
) -> Result<VulkanDistributedSharedBufferAllocation, VulkanDistributedActivationBufferError>
where
    F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
    E: Display,
{
    let owner = device_for(owner_device_id).map_err(|error| {
        VulkanDistributedActivationBufferError(format!(
            "failed to resolve distributed {label} owner {owner_device_id:?}: {error}"
        ))
    })?;
    let peers = device_ids
        .iter()
        .filter(|device_id| device_id.as_str() != owner_device_id)
        .map(|device_id| {
            device_for(device_id).map_err(|error| {
                VulkanDistributedActivationBufferError(format!(
                    "failed to resolve distributed {label} participant {device_id:?}: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shared = owner
        .create_shared_resident_buffers_for_route(&peers, byte_capacity, route)
        .map_err(|error| {
            VulkanDistributedActivationBufferError(format!(
                "failed to allocate {byte_capacity} shared bytes for distributed {label}: {error}"
            ))
        })?;
    let mut buffers = shared.buffers.into_iter();
    let mut device_buffers = BTreeMap::from([(
        owner_device_id.to_string(),
        buffers
            .next()
            .expect("shared allocation always contains its owner"),
    )]);
    for (device_id, buffer) in device_ids
        .iter()
        .filter(|device_id| device_id.as_str() != owner_device_id)
        .zip(buffers)
    {
        if device_buffers.insert(device_id.clone(), buffer).is_some() {
            return Err(VulkanDistributedActivationBufferError(format!(
                "distributed {label} repeats device {device_id:?}"
            )));
        }
    }
    if device_buffers.len() != device_ids.len() {
        return Err(VulkanDistributedActivationBufferError(format!(
            "distributed {label} resolved {} buffers for {} devices",
            device_buffers.len(),
            device_ids.len()
        )));
    }
    Ok(VulkanDistributedSharedBufferAllocation {
        route: shared.route,
        external_device_local_error: shared.external_device_local_error,
        device_buffers,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanDistributedActivationBufferAllocationKey {
    ActivationSlot {
        owner_device_id: String,
        component_id: String,
        slot: usize,
    },
    BoundaryInput {
        owner_device_id: String,
        component_id: String,
        signal_id: String,
    },
    BoundaryOutput {
        owner_device_id: String,
        component_id: String,
        signal_id: String,
    },
    Edge {
        edge_index: usize,
        owner_device_id: String,
    },
}

fn distributed_activation_allocation_key(
    dispatch_owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
) -> VulkanDistributedActivationBufferAllocationKey {
    match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => {
            VulkanDistributedActivationBufferAllocationKey::ActivationSlot {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                slot: activation.slot,
            }
        }
        VulkanDistributedActivationStorage::BoundaryInput => {
            VulkanDistributedActivationBufferAllocationKey::BoundaryInput {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                signal_id: activation.signal_id.clone(),
            }
        }
        VulkanDistributedActivationStorage::BoundaryOutput => {
            VulkanDistributedActivationBufferAllocationKey::BoundaryOutput {
                owner_device_id: dispatch_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                signal_id: activation.signal_id.clone(),
            }
        }
        VulkanDistributedActivationStorage::Edge {
            edge_index,
            owner_device_id,
        } => VulkanDistributedActivationBufferAllocationKey::Edge {
            edge_index: *edge_index,
            owner_device_id: owner_device_id.clone(),
        },
    }
}

fn distributed_activation_allocation_matches(
    dispatch_owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
    allocation: &VulkanDistributedActivationBufferAllocation,
) -> bool {
    if allocation.storage != activation.storage {
        return false;
    }
    match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => {
            allocation.owner_device_id == dispatch_owner_device_id
                && allocation.component_id == activation.component_id
                && allocation.slot == activation.slot
        }
        VulkanDistributedActivationStorage::BoundaryInput
        | VulkanDistributedActivationStorage::BoundaryOutput => {
            allocation.owner_device_id == dispatch_owner_device_id
                && allocation.component_id == activation.component_id
                && allocation.signal_ids.contains(&activation.signal_id)
        }
        VulkanDistributedActivationStorage::Edge {
            edge_index,
            owner_device_id,
        } => {
            allocation.owner_device_id == *owner_device_id
                && matches!(
                    allocation.storage,
                    VulkanDistributedActivationStorage::Edge {
                        edge_index: candidate,
                        ..
                    } if candidate == *edge_index
                )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDistributedActivationAccess {
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchShard {
    pub device_id: String,
    pub row_start: usize,
    pub row_count: usize,
    pub workgroup_count_x: u32,
    pub base_workgroup_z: u32,
    pub input_range: VulkanDistributedActivationRange,
    pub auxiliary_input_ranges: Vec<VulkanDistributedActivationRange>,
    pub output_byte_offset: usize,
    pub output_byte_count: usize,
    pub parameters: Vec<VulkanDistributedParameterFragment>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedActivationRange {
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterFragment {
    pub binding: usize,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocationPlan {
    pub allocations: Vec<VulkanDistributedParameterAllocation>,
    pub allocation_count: usize,
    pub tensor_count: usize,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterAllocationPlan {
    pub fn from_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_execution_plan_with_coverage(execution_plan, tensor_index, true)
    }

    pub fn from_sampled_execution_plan(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_execution_plan_with_coverage(execution_plan, tensor_index, false)
    }

    fn from_execution_plan_with_coverage(
        execution_plan: &VulkanDistributedExecutionPlan,
        tensor_index: &TensorIndex,
        require_complete_tensor_coverage: bool,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let device_ids = execution_plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut allocations = BTreeMap::<
            VulkanDistributedParameterAllocationKey,
            VulkanDistributedParameterAllocation,
        >::new();

        for dispatch in &execution_plan.dispatches {
            for shard in &dispatch.shards {
                if !device_ids.contains(shard.device_id.as_str()) {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed parameter shard for {}.{} uses device {:?} outside the execution pool",
                        dispatch.component_id, dispatch.node_id, shard.device_id
                    )));
                }
                for fragment in &shard.parameters {
                    let metadata = tensor_index.tensors.get(&fragment.tensor).ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter fragment has no tensor metadata for {:?}",
                            fragment.tensor
                        ))
                    })?;
                    let tensor_byte_count = metadata.byte_count.ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has no byte count",
                            fragment.tensor
                        ))
                    })?;
                    if fragment.byte_count == 0 {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has an empty fragment",
                            fragment.tensor
                        )));
                    }
                    let byte_end = fragment
                        .byte_offset
                        .checked_add(fragment.byte_count)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "distributed parameter tensor {:?} fragment range overflowed",
                                fragment.tensor
                            ))
                        })?;
                    if byte_end > tensor_byte_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed parameter tensor {:?} has {tensor_byte_count} bytes but a fragment ends at {byte_end}",
                            fragment.tensor
                        )));
                    }
                    let key = VulkanDistributedParameterAllocationKey {
                        device_id: shard.device_id.clone(),
                        tensor: fragment.tensor.clone(),
                        byte_offset: fragment.byte_offset,
                        byte_count: fragment.byte_count,
                    };
                    if let Some(allocation) = allocations.get_mut(&key) {
                        allocation.use_count =
                            allocation.use_count.checked_add(1).ok_or_else(|| {
                                VulkanDistributedPlanError(format!(
                                    "distributed parameter tensor {:?} use count overflowed",
                                    fragment.tensor
                                ))
                            })?;
                    } else {
                        allocations.insert(
                            key,
                            VulkanDistributedParameterAllocation {
                                device_id: shard.device_id.clone(),
                                tensor: fragment.tensor.clone(),
                                byte_offset: fragment.byte_offset,
                                byte_count: fragment.byte_count,
                                use_count: 1,
                            },
                        );
                    }
                }
            }
        }

        if require_complete_tensor_coverage {
            validate_tensor_partition_coverage(allocations.values(), tensor_index)?;
        }
        let total_byte_capacity = allocations.values().try_fold(0usize, |total, allocation| {
            total.checked_add(allocation.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "distributed parameter allocation byte count overflowed".to_string(),
                )
            })
        })?;
        let tensor_count = allocations
            .values()
            .map(|allocation| allocation.tensor.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let allocations = allocations.into_values().collect::<Vec<_>>();

        Ok(Self {
            allocation_count: allocations.len(),
            allocations,
            tensor_count,
            total_byte_capacity,
        })
    }

    pub fn load_from_tensor_index<F>(
        &self,
        tensor_index: &TensorIndex,
        mut write: F,
    ) -> Result<VulkanDistributedParameterLoadReport, VulkanDistributedParameterLoadError>
    where
        F: FnMut(
            &VulkanDistributedParameterAllocation,
            &[u8],
        ) -> Result<(), VulkanDistributedParameterLoadError>,
    {
        let mut allocations_by_tensor = BTreeMap::<
            &str,
            BTreeMap<(usize, usize), Vec<&VulkanDistributedParameterAllocation>>,
        >::new();
        for allocation in &self.allocations {
            allocations_by_tensor
                .entry(&allocation.tensor)
                .or_default()
                .entry((allocation.byte_offset, allocation.byte_count))
                .or_default()
                .push(allocation);
        }

        let mut total_bytes_read = 0usize;
        let mut total_bytes_written = 0usize;
        let mut write_count = 0usize;
        let mut source_files = BTreeSet::new();
        for (tensor, ranges) in allocations_by_tensor {
            let storage = TensorStorage::from_index(tensor_index, tensor)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            let storage_ranges = ranges
                .keys()
                .map(|(byte_offset, byte_count)| TensorStorageRange {
                    byte_offset: *byte_offset,
                    byte_count: *byte_count,
                })
                .collect::<Vec<_>>();
            let payloads = storage
                .read_partitions(&storage_ranges)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))?;
            total_bytes_read = total_bytes_read
                .checked_add(storage.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterLoadError(
                        "distributed parameter read byte count overflowed".to_string(),
                    )
                })?;
            source_files.insert(storage.source_file);

            for (((_, _), allocations), payload) in ranges.into_iter().zip(payloads) {
                for allocation in allocations {
                    write(allocation, &payload)?;
                    total_bytes_written = total_bytes_written
                        .checked_add(payload.len())
                        .ok_or_else(|| {
                            VulkanDistributedParameterLoadError(
                                "distributed parameter written byte count overflowed".to_string(),
                            )
                        })?;
                    write_count = write_count.checked_add(1).ok_or_else(|| {
                        VulkanDistributedParameterLoadError(
                            "distributed parameter write count overflowed".to_string(),
                        )
                    })?;
                }
            }
        }

        Ok(VulkanDistributedParameterLoadReport {
            tensor_count: self.tensor_count,
            source_file_count: source_files.len(),
            allocation_count: self.allocation_count,
            write_count,
            total_bytes_read,
            total_bytes_written,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterAllocation {
    pub device_id: String,
    pub tensor: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub use_count: usize,
}

pub struct VulkanDistributedParameterBuffers {
    pub plan: VulkanDistributedParameterAllocationPlan,
    pub buffers: Vec<VulkanDistributedParameterBufferAllocation>,
    pub total_byte_capacity: usize,
}

impl VulkanDistributedParameterBuffers {
    pub fn allocate_and_load<'a, F, E>(
        plan: &VulkanDistributedParameterAllocationPlan,
        tensor_index: &TensorIndex,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedParameterBufferError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut buffers = std::iter::repeat_with(|| None)
            .take(plan.allocations.len())
            .collect::<Vec<Option<VulkanDistributedParameterBufferAllocation>>>();
        let mut buffer_index = BTreeMap::new();
        let mut allocations_by_device = BTreeMap::<String, Vec<usize>>::new();
        let mut total_byte_capacity = 0usize;
        for (allocation_index, allocation) in plan.allocations.iter().enumerate() {
            total_byte_capacity = total_byte_capacity
                .checked_add(allocation.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(
                        "distributed parameter buffer byte capacity overflowed".to_string(),
                    )
                })?;
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            if buffer_index.insert(key, allocation_index).is_some() {
                return Err(VulkanDistributedParameterBufferError(format!(
                    "distributed parameter buffer repeats tensor {:?} range {}..{} on {:?}",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                )));
            }
            allocations_by_device
                .entry(allocation.device_id.clone())
                .or_default()
                .push(allocation_index);
        }
        for (device_id, allocation_indices) in allocations_by_device {
            let device = device_for(&device_id).map_err(|error| {
                VulkanDistributedParameterBufferError(format!(
                    "failed to resolve distributed parameter device {device_id:?}: {error}"
                ))
            })?;
            let byte_counts = allocation_indices
                .iter()
                .map(|index| plan.allocations[*index].byte_count)
                .collect::<Vec<_>>();
            let arena_allocations = device
                .allocate_resident_buffer_arena(&byte_counts)
                .map_err(VulkanDistributedParameterBufferError::from)?;
            for (allocation_index, arena) in allocation_indices.into_iter().zip(arena_allocations)
            {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: plan.allocations[allocation_index].clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
            }
        }
        let buffers = buffers
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                buffer.ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "distributed parameter arena did not allocate plan index {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        plan.load_from_tensor_index(tensor_index, |allocation, bytes| {
            let key = VulkanDistributedParameterAllocationKey::from(allocation);
            let index = *buffer_index.get(&key).ok_or_else(|| {
                VulkanDistributedParameterLoadError(format!(
                    "distributed parameter buffer for tensor {:?} range {}..{} on {:?} is missing",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                ))
            })?;
            buffers[index]
                .buffer
                .write_bytes_at(buffers[index].byte_offset, bytes)
                .map_err(|error| VulkanDistributedParameterLoadError(error.to_string()))
        })
        .map_err(|error| VulkanDistributedParameterBufferError(error.to_string()))?;

        Ok(Self {
            plan: plan.clone(),
            buffers,
            total_byte_capacity,
        })
    }

    pub fn allocate_and_load_from_pool(
        plan: &VulkanDistributedParameterAllocationPlan,
        tensor_index: &TensorIndex,
        pool: &VulkanResidentBufferPool,
    ) -> Result<Self, VulkanDistributedParameterBufferError> {
        let mut buffers = std::iter::repeat_with(|| None)
            .take(plan.allocations.len())
            .collect::<Vec<Option<VulkanDistributedParameterBufferAllocation>>>();
        let mut buffer_index = BTreeMap::new();
        let mut pool_keys = Vec::with_capacity(plan.allocations.len());
        let mut missing_by_device = BTreeMap::<
            String,
            Vec<(usize, VulkanResidentBufferPoolKey)>,
        >::new();
        let mut total_byte_capacity = 0usize;
        for (allocation_index, allocation) in plan.allocations.iter().enumerate() {
            let metadata = tensor_index
                .tensors
                .get(&allocation.tensor)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "tensor index has no distributed parameter {:?}",
                        allocation.tensor
                    ))
                })?;
            let content_identity = metadata
                .immutable_content_identity(&allocation.tensor)
                .map_err(|error| {
                    VulkanDistributedParameterBufferError(
                        error.to_string(),
                    )
                })?;
            let key = VulkanResidentBufferPoolKey::new(
                "nerve.tensor_parameter.v1",
                &allocation.device_id,
                &allocation.tensor,
                content_identity,
                allocation.byte_offset,
                allocation.byte_count,
            )
            .map_err(VulkanDistributedParameterBufferError::from)?;
            if let Some(arena) = pool.resident_allocation(&key) {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: allocation.clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
            } else {
                missing_by_device
                    .entry(allocation.device_id.clone())
                    .or_default()
                    .push((allocation_index, key.clone()));
            }
            pool_keys.push(key);
            total_byte_capacity = total_byte_capacity
                .checked_add(allocation.byte_count)
                .ok_or_else(|| {
                    VulkanDistributedParameterBufferError(
                        "distributed parameter buffer byte capacity overflowed"
                            .to_string(),
                    )
                })?;
            let allocation_key =
                VulkanDistributedParameterAllocationKey::from(allocation);
            if buffer_index
                .insert(allocation_key, allocation_index)
                .is_some()
            {
                return Err(VulkanDistributedParameterBufferError(format!(
                    "distributed parameter buffer repeats tensor {:?} range {}..{} on {:?}",
                    allocation.tensor,
                    allocation.byte_offset,
                    allocation.byte_offset + allocation.byte_count,
                    allocation.device_id
                )));
            }
        }
        let mut unpublished_indices = Vec::new();
        for (_, missing) in missing_by_device {
            let keys = missing
                .iter()
                .map(|(_, key)| key.clone())
                .collect::<Vec<_>>();
            let arena_allocations = pool
                .allocate_unpublished_batch(&keys)
                .map_err(VulkanDistributedParameterBufferError::from)?;
            for ((allocation_index, _), arena) in missing.into_iter().zip(arena_allocations) {
                buffers[allocation_index] = Some(VulkanDistributedParameterBufferAllocation {
                    allocation: plan.allocations[allocation_index].clone(),
                    buffer: arena.buffer,
                    byte_offset: arena.byte_offset,
                });
                unpublished_indices.push(allocation_index);
            }
        }
        let buffers = buffers
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                buffer.ok_or_else(|| {
                    VulkanDistributedParameterBufferError(format!(
                        "pooled distributed parameter arena did not allocate plan index {index}"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unpublished_allocations = unpublished_indices
            .iter()
            .map(|index| plan.allocations[*index].clone())
            .collect::<Vec<_>>();
        if !unpublished_allocations.is_empty() {
            let total_byte_capacity = unpublished_allocations
                .iter()
                .try_fold(0usize, |total, allocation| {
                    total.checked_add(allocation.byte_count).ok_or_else(
                        || {
                            VulkanDistributedParameterBufferError(
                                "pooled distributed parameter load byte count overflowed"
                                    .to_string(),
                            )
                        },
                    )
                })?;
            let load_plan = VulkanDistributedParameterAllocationPlan {
                allocation_count: unpublished_allocations.len(),
                tensor_count: unpublished_allocations
                    .iter()
                    .map(|allocation| allocation.tensor.as_str())
                    .collect::<BTreeSet<_>>()
                    .len(),
                total_byte_capacity,
                allocations: unpublished_allocations,
            };
            load_plan
                .load_from_tensor_index(
                    tensor_index,
                    |allocation, bytes| {
                        let key =
                            VulkanDistributedParameterAllocationKey::from(
                                allocation,
                            );
                        let index = *buffer_index.get(&key).ok_or_else(
                            || {
                                VulkanDistributedParameterLoadError(
                                    format!(
                                        "pooled distributed parameter buffer for tensor {:?} range {}..{} on {:?} is missing",
                                        allocation.tensor,
                                        allocation.byte_offset,
                                        allocation.byte_offset
                                            + allocation.byte_count,
                                        allocation.device_id
                                    ),
                                )
                            },
                        )?;
                        buffers[index]
                            .buffer
                            .write_bytes_at(buffers[index].byte_offset, bytes)
                            .map_err(|error| {
                                VulkanDistributedParameterLoadError(
                                    error.to_string(),
                                )
                            })
                    },
                )
                .map_err(|error| {
                    VulkanDistributedParameterBufferError(
                        error.to_string(),
                    )
                })?;
            let publications = unpublished_indices
                .iter()
                .map(|index| {
                    let buffer = &buffers[*index];
                    (
                        pool_keys[*index].clone(),
                        VulkanResidentBufferPoolAllocation {
                            buffer: Arc::clone(&buffer.buffer),
                            byte_offset: buffer.byte_offset,
                            byte_count: buffer.allocation.byte_count,
                        },
                    )
                })
                .collect();
            pool.publish_batch(publications)
                .map_err(VulkanDistributedParameterBufferError::from)?;
        }
        Ok(Self {
            plan: plan.clone(),
            buffers,
            total_byte_capacity,
        })
    }

    pub fn parameter_buffer(
        &self,
        device_id: &str,
        tensor: &str,
        byte_offset: usize,
        byte_count: usize,
    ) -> Option<&VulkanDistributedParameterBufferAllocation> {
        self.buffers.iter().find(|buffer| {
            buffer.allocation.device_id == device_id
                && buffer.allocation.tensor == tensor
                && buffer.allocation.byte_offset == byte_offset
                && buffer.allocation.byte_count == byte_count
        })
    }
}

pub struct VulkanDistributedParameterBufferAllocation {
    pub allocation: VulkanDistributedParameterAllocation,
    pub buffer: Arc<VulkanResidentBuffer>,
    pub byte_offset: usize,
}

impl VulkanDistributedParameterBufferAllocation {
    pub fn kernel_binding(
        &self,
        binding: u32,
    ) -> VulkanResidentKernelBufferBinding<'_> {
        VulkanResidentKernelBufferBinding::new(
            binding,
            &self.buffer,
            self.allocation.byte_count,
        )
        .with_byte_offset(self.byte_offset)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedParameterBufferError(pub String);

impl Display for VulkanDistributedParameterBufferError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedParameterBufferError {}

impl From<VulkanError> for VulkanDistributedParameterBufferError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}
