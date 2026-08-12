use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::stream_circuit::ComponentEdgePlacement;
use crate::stream_plan::TensorIndex;
use crate::tensor_storage::{TensorStorage, TensorStorageRange};
use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanResidentBuffer, VulkanResidentBufferPool,
    VulkanResidentBufferPoolAllocation, VulkanResidentBufferPoolKey,
    VulkanResidentKernelBufferAccess, VulkanResidentKernelBufferBinding,
    VulkanResidentKernelDispatch, VulkanResidentKernelSequence, VulkanResidentKernelSequenceStep,
    VulkanResidentQueueSubmissionBatch, VulkanSharedResidentBufferRoute, VulkanTimelineSemaphore,
    VulkanTimelineSemaphorePoint, VulkanTimelineSemaphoreReplayState,
};
use crate::vulkan_stream_circuit::{
    CompiledResourceBindingMapping, CompiledResourceLifetime, CompiledResourceResidencyContract,
    CompiledResourceSelectorMapping, VulkanActivationSlotBufferOverride,
    VulkanCompiledResourceDeviceStore, VulkanDescriptorResourceAddress,
    VulkanDistributedSelectedResourceGate, VulkanDynamicResourceBuffers,
    VulkanKernelDescriptorUsage, VulkanKernelScalarBinding, VulkanKernelScalarSource,
    VulkanLoadedKernelArtifactCatalog, VulkanLoadedPhysicalKernelArtifact,
    VulkanModelBoundaryBufferOverride, VulkanModelBoundaryDirection,
    VulkanPhysicalKernelArtifactManifest, VulkanPreparedDispatch, VulkanPreparedDispatchPlan,
    VulkanResidentFeedbackControlPlane,
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
    pub sequence_kind: VulkanDistributedDispatchSequenceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VulkanDistributedDispatchSequenceKind {
    Direct,
    FeedbackIndirect,
}

impl VulkanDistributedDispatchSequenceKind {
    pub fn for_feedback_lane(feedback_lane: Option<usize>) -> Self {
        if feedback_lane.is_some() {
            Self::FeedbackIndirect
        } else {
            Self::Direct
        }
    }
}

impl VulkanDistributedExecutionPlan {
    pub fn replaces_same_logical_dispatches(&self, other: &Self) -> bool {
        if self.dispatches.len() != other.dispatches.len() {
            return false;
        }
        let keys = |plan: &Self| {
            plan.dispatches
                .iter()
                .map(|dispatch| {
                    (
                        dispatch.owner_device_id.clone(),
                        dispatch.dispatch_index,
                        dispatch.component_id.clone(),
                        dispatch.node_id.clone(),
                    )
                })
                .collect::<BTreeSet<_>>()
        };
        keys(self) == keys(other)
    }

    pub fn from_prepared_plans(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_prepared_plans_for_phase(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            component_device_pools,
            edge_placements,
            storage_buffer_offset_alignment,
            ExecutionPhase::Decode,
            ExecutionShape::SingleLane,
        )
    }

    pub fn from_prepared_plans_for_phase(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        phase: ExecutionPhase,
        execution_shape: ExecutionShape,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_prepared_plans_for_phase_and_resources(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            component_device_pools,
            edge_placements,
            storage_buffer_offset_alignment,
            phase,
            execution_shape,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_plans_for_phase_with_resource_contract(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        phase: ExecutionPhase,
        execution_shape: ExecutionShape,
        execution_scope: &str,
        resource_contract: &CompiledResourceResidencyContract,
    ) -> Result<Self, VulkanDistributedPlanError> {
        Self::from_prepared_plans_for_phase_and_resources(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            component_device_pools,
            edge_placements,
            storage_buffer_offset_alignment,
            phase,
            execution_shape,
            Some((execution_scope, resource_contract)),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_plans_for_phase_with_resource_contract_and_contracts(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        phase: ExecutionPhase,
        execution_shape: ExecutionShape,
        execution_scope: &str,
        resource_contract: &CompiledResourceResidencyContract,
        selected_contract_ids: &BTreeSet<String>,
    ) -> Result<Self, VulkanDistributedPlanError> {
        if selected_contract_ids.is_empty() {
            return Err(VulkanDistributedPlanError(
                "exact distributed planning requires selected contract IDs".to_string(),
            ));
        }
        Self::from_prepared_plans_for_phase_and_resources(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            component_device_pools,
            edge_placements,
            storage_buffer_offset_alignment,
            phase,
            execution_shape,
            Some((execution_scope, resource_contract)),
            Some(selected_contract_ids),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_prepared_plans_for_phase_and_resources(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &BTreeMap<String, Vec<String>>,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        phase: ExecutionPhase,
        execution_shape: ExecutionShape,
        resource_context: Option<(&str, &CompiledResourceResidencyContract)>,
        selected_contract_ids: Option<&BTreeSet<String>>,
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
                let Some((contract, artifact)) = select_distributed_contract(
                    dispatch,
                    artifact_manifest,
                    phase,
                    execution_shape,
                    selected_contract_ids,
                )?
                else {
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
                    resource_context,
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
        let execution_islands = resolved_physical_execution_islands_for_phase(
            &dispatches,
            shared_activation_route,
            phase,
        )?;
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
        self.sampled_for_parameter_budget_with_fraction(
            tensor_index,
            participant_device_ids,
            maximum_total_parameter_bytes,
        )
        .map(|sampled| sampled.map(|(plan, _)| plan))
    }

    pub fn sampled_for_parameter_budget_with_fraction(
        &self,
        tensor_index: &TensorIndex,
        participant_device_ids: &[String],
        maximum_total_parameter_bytes: usize,
    ) -> Result<Option<(Self, usize)>, VulkanDistributedPlanError> {
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
            sampled_distributed_execution_plan(self, participant_device_ids, fraction_millionths)
        };
        let minimum = build(0)?;
        let minimum_bytes = VulkanDistributedParameterAllocationPlan::from_sampled_execution_plan(
            &minimum,
            tensor_index,
        )?
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
            let bytes = VulkanDistributedParameterAllocationPlan::from_sampled_execution_plan(
                &candidate,
                tensor_index,
            )?
            .total_byte_capacity;
            if bytes <= maximum_total_parameter_bytes {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(Some((build(low)?, low)))
    }

    pub fn sampled_for_fraction_millionths(
        &self,
        participant_device_ids: &[String],
        fraction_millionths: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        if participant_device_ids.is_empty()
            || participant_device_ids.iter().any(String::is_empty)
            || participant_device_ids.iter().collect::<BTreeSet<_>>().len()
                != participant_device_ids.len()
            || fraction_millionths > 1_000_000
        {
            return Err(VulkanDistributedPlanError(
                "fixed sampled distributed execution requires distinct participants and a fraction in 0..=1000000"
                    .to_string(),
            ));
        }
        sampled_distributed_execution_plan(self, participant_device_ids, fraction_millionths)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedExecutionPlanSet {
    pub decode: VulkanDistributedExecutionPlan,
    pub decode_batch: VulkanDistributedExecutionPlan,
    pub prefill: VulkanDistributedExecutionPlan,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanDistributedPhaseComponentDevicePools {
    pub decode: BTreeMap<String, Vec<String>>,
    pub decode_batch: BTreeMap<String, Vec<String>>,
    pub prefill: BTreeMap<String, Vec<String>>,
}

impl VulkanDistributedPhaseComponentDevicePools {
    pub fn uniform(component_device_pools: &BTreeMap<String, Vec<String>>) -> Self {
        Self {
            decode: component_device_pools.clone(),
            decode_batch: component_device_pools.clone(),
            prefill: component_device_pools.clone(),
        }
    }
}

impl VulkanDistributedExecutionPlanSet {
    pub fn from_prepared_plans(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &VulkanDistributedPhaseComponentDevicePools,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let decode = VulkanDistributedExecutionPlan::from_prepared_plans(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            &component_device_pools.decode,
            edge_placements,
            storage_buffer_offset_alignment,
        )?;
        let decode_batch = VulkanDistributedExecutionPlan::from_prepared_plans_for_phase(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            &component_device_pools.decode_batch,
            edge_placements,
            storage_buffer_offset_alignment,
            ExecutionPhase::Decode,
            ExecutionShape::MultiLane,
        )?;
        let prefill = VulkanDistributedExecutionPlan::from_prepared_plans_for_phase(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            &component_device_pools.prefill,
            edge_placements,
            storage_buffer_offset_alignment,
            ExecutionPhase::Prefill,
            ExecutionShape::MultiLane,
        )?;
        Ok(Self {
            decode,
            decode_batch,
            prefill,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_plans_with_resource_contract(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &VulkanDistributedPhaseComponentDevicePools,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        execution_scope: &str,
        resource_contract: &CompiledResourceResidencyContract,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let resource_context = Some((execution_scope, resource_contract));
        let decode = VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_and_resources(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            &component_device_pools.decode,
            edge_placements,
            storage_buffer_offset_alignment,
            ExecutionPhase::Decode,
            ExecutionShape::SingleLane,
            resource_context,
            None,
        )?;
        let decode_batch =
            VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_and_resources(
                prepared_plans,
                tensor_index,
                artifact_manifest,
                &component_device_pools.decode_batch,
                edge_placements,
                storage_buffer_offset_alignment,
                ExecutionPhase::Decode,
                ExecutionShape::MultiLane,
                resource_context,
                None,
            )?;
        let prefill = VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_and_resources(
            prepared_plans,
            tensor_index,
            artifact_manifest,
            &component_device_pools.prefill,
            edge_placements,
            storage_buffer_offset_alignment,
            ExecutionPhase::Prefill,
            ExecutionShape::MultiLane,
            resource_context,
            None,
        )?;
        Ok(Self {
            decode,
            decode_batch,
            prefill,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_prepared_plans_with_resource_contract_and_execution_cases(
        prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
        tensor_index: &TensorIndex,
        artifact_manifest: &VulkanPhysicalKernelArtifactManifest,
        component_device_pools: &VulkanDistributedPhaseComponentDevicePools,
        edge_placements: &[ComponentEdgePlacement],
        storage_buffer_offset_alignment: usize,
        execution_scope: &str,
        resource_contract: &CompiledResourceResidencyContract,
        decode_cases: &BTreeMap<
            String,
            crate::vulkan_stream_circuit::VulkanPlacementExecutionCaseIdentity,
        >,
        decode_batch_cases: &BTreeMap<
            String,
            crate::vulkan_stream_circuit::VulkanPlacementExecutionCaseIdentity,
        >,
        prefill_cases: &BTreeMap<
            String,
            crate::vulkan_stream_circuit::VulkanPlacementExecutionCaseIdentity,
        >,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let selected_contracts = |cases: &BTreeMap<
            String,
            crate::vulkan_stream_circuit::VulkanPlacementExecutionCaseIdentity,
        >| {
            cases
                .values()
                .flat_map(|case| case.behavior.contract_ids.iter().cloned())
                .collect::<BTreeSet<_>>()
        };
        let decode_contracts = selected_contracts(decode_cases);
        let decode_batch_contracts = selected_contracts(decode_batch_cases);
        let prefill_contracts = selected_contracts(prefill_cases);
        let resource_context = Some((execution_scope, resource_contract));
        let build = |pools: &BTreeMap<String, Vec<String>>,
                     phase,
                     shape,
                     contracts: &BTreeSet<String>| {
            VulkanDistributedExecutionPlan::from_prepared_plans_for_phase_and_resources(
                prepared_plans,
                tensor_index,
                artifact_manifest,
                pools,
                edge_placements,
                storage_buffer_offset_alignment,
                phase,
                shape,
                resource_context,
                (!pools.is_empty() && !contracts.is_empty()).then_some(contracts),
            )
        };
        Ok(Self {
            decode: build(
                &component_device_pools.decode,
                ExecutionPhase::Decode,
                ExecutionShape::SingleLane,
                &decode_contracts,
            )?,
            decode_batch: build(
                &component_device_pools.decode_batch,
                ExecutionPhase::Decode,
                ExecutionShape::MultiLane,
                &decode_batch_contracts,
            )?,
            prefill: build(
                &component_device_pools.prefill,
                ExecutionPhase::Prefill,
                ExecutionShape::MultiLane,
                &prefill_contracts,
            )?,
        })
    }

    pub fn all(&self) -> [&VulkanDistributedExecutionPlan; 3] {
        [&self.decode, &self.decode_batch, &self.prefill]
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
    let distributed_parameter_byte_count =
        dispatches.iter().try_fold(0usize, |total, dispatch| {
            total
                .checked_add(dispatch.distributed_parameter_byte_count)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "sampled distributed parameter total overflowed".to_string(),
                    )
                })
        })?;
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
            VulkanDistributedPlanError("merged distributed row count overflowed".to_string())
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
                VulkanDistributedPlanError(format!("merged distributed {label} range overflowed",))
            })?;
        for range in ranges.iter().skip(1) {
            if range.byte_offset != byte_end {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed dispatch {}.{} has non-contiguous {label} ranges",
                    dispatch.component_id, dispatch.node_id,
                )));
            }
            byte_end = byte_end.checked_add(range.byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(format!("merged distributed {label} range overflowed",))
            })?;
        }
        Ok(VulkanDistributedActivationRange {
            byte_offset: leading.byte_offset,
            byte_count: byte_end - leading.byte_offset,
        })
    };
    let input_range = merge_ranges(
        dispatch
            .shards
            .iter()
            .map(|shard| &shard.input_range)
            .collect(),
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
    let selected_resource_fragments = merged_selected_resource_fragments(dispatch)?;
    Ok(VulkanDistributedDispatchShard {
        device_id: first.device_id.clone(),
        selected_resource_indices: dispatch
            .selected_resource_partitions
            .iter()
            .filter(|partition| partition.parameter_partitions.is_empty())
            .map(|partition| {
                (
                    partition.selector_id.clone(),
                    (0..partition.resource_count).collect(),
                )
            })
            .collect(),
        selected_resource_fragments,
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

fn merged_selected_resource_fragments(
    dispatch: &VulkanDistributedDispatchPlan,
) -> Result<
    BTreeMap<String, Vec<VulkanDistributedSelectedResourceFragmentPlan>>,
    VulkanDistributedPlanError,
> {
    let fragmented_selectors = dispatch
        .selected_resource_partitions
        .iter()
        .filter(|partition| !partition.parameter_partitions.is_empty())
        .map(|partition| partition.selector_id.as_str())
        .collect::<BTreeSet<_>>();
    if dispatch.shards.iter().any(|shard| {
        shard
            .selected_resource_fragments
            .keys()
            .any(|selector| !fragmented_selectors.contains(selector.as_str()))
    }) {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} contains fragments for an undeclared selector",
            dispatch.component_id, dispatch.node_id,
        )));
    }

    let mut merged_by_selector = BTreeMap::new();
    for partition in dispatch
        .selected_resource_partitions
        .iter()
        .filter(|partition| !partition.parameter_partitions.is_empty())
    {
        let partitioned_slots = partition
            .parameter_partitions
            .iter()
            .map(|parameter| parameter.parameter_slot)
            .collect::<BTreeSet<_>>();
        let mut merged_resources = Vec::with_capacity(partition.resource_count);
        for resource_index in 0..partition.resource_count {
            let mut source_fragments = dispatch
                .shards
                .iter()
                .map(|shard| {
                    let fragments = shard
                        .selected_resource_fragments
                        .get(&partition.selector_id)
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(format!(
                                "distributed dispatch {}.{} omits fragmented selector {:?} on device {:?}",
                                dispatch.component_id,
                                dispatch.node_id,
                                partition.selector_id,
                                shard.device_id,
                            ))
                        })?;
                    let mut matching = fragments
                        .iter()
                        .filter(|fragment| fragment.resource_index == resource_index);
                    let fragment = matching.next().ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "distributed selector {:?} omits resource {resource_index} on device {:?}",
                            partition.selector_id, shard.device_id,
                        ))
                    })?;
                    if matching.next().is_some() {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed selector {:?} duplicates resource {resource_index} on device {:?}",
                            partition.selector_id, shard.device_id,
                        )));
                    }
                    Ok(fragment)
                })
                .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
            source_fragments.sort_by_key(|fragment| fragment.logical_start);
            let leading = source_fragments[0];
            if source_fragments.iter().any(|fragment| {
                fragment.atomic_group_id != leading.atomic_group_id
                    || fragment.parameters.len() != partition.parameters_per_resource
            }) {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed selector {:?} resource {resource_index} changes identity between fragments",
                    partition.selector_id,
                )));
            }
            let mut logical_frontier = 0usize;
            for fragment in &source_fragments {
                if fragment.logical_start != logical_frontier || fragment.logical_count == 0 {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed selector {:?} resource {resource_index} has non-contiguous logical fragments",
                        partition.selector_id,
                    )));
                }
                logical_frontier = logical_frontier
                    .checked_add(fragment.logical_count)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "merged selected-resource logical range overflowed".to_string(),
                        )
                    })?;
            }
            if logical_frontier != dispatch.output_rows {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed selector {:?} resource {resource_index} covers {logical_frontier} logical rows, expected {}",
                    partition.selector_id, dispatch.output_rows,
                )));
            }

            let mut merged_parameters = Vec::with_capacity(partition.parameters_per_resource);
            for parameter_slot in 0..partition.parameters_per_resource {
                let source_parameters = source_fragments
                    .iter()
                    .map(|fragment| {
                        fragment
                            .parameters
                            .iter()
                            .find(|parameter| parameter.parameter_slot == parameter_slot)
                            .ok_or_else(|| {
                                VulkanDistributedPlanError(format!(
                                    "distributed selector {:?} resource {resource_index} omits parameter slot {parameter_slot}",
                                    partition.selector_id,
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
                let first_parameter = source_parameters[0];
                if source_parameters.iter().any(|parameter| {
                    parameter.resource_id != first_parameter.resource_id
                        || parameter.resource_byte_count != first_parameter.resource_byte_count
                }) {
                    return Err(VulkanDistributedPlanError(format!(
                        "distributed selector {:?} resource {resource_index} parameter slot {parameter_slot} changes identity between fragments",
                        partition.selector_id,
                    )));
                }
                if partitioned_slots.contains(&parameter_slot) {
                    let mut byte_frontier = 0usize;
                    for parameter in source_parameters {
                        if parameter.byte_offset != byte_frontier || parameter.byte_count == 0 {
                            return Err(VulkanDistributedPlanError(format!(
                                "distributed selector {:?} resource {resource_index} parameter slot {parameter_slot} has non-contiguous byte fragments",
                                partition.selector_id,
                            )));
                        }
                        byte_frontier = byte_frontier.checked_add(parameter.byte_count).ok_or_else(
                            || {
                                VulkanDistributedPlanError(
                                    "merged selected-resource parameter range overflowed"
                                        .to_string(),
                                )
                            },
                        )?;
                    }
                    if byte_frontier != first_parameter.resource_byte_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed selector {:?} resource {resource_index} parameter slot {parameter_slot} covers {byte_frontier} bytes, expected {}",
                            partition.selector_id, first_parameter.resource_byte_count,
                        )));
                    }
                    let mut merged = first_parameter.clone();
                    merged.byte_offset = 0;
                    merged.byte_count = byte_frontier;
                    merged_parameters.push(merged);
                } else {
                    if source_parameters
                        .iter()
                        .any(|parameter| *parameter != first_parameter)
                    {
                        return Err(VulkanDistributedPlanError(format!(
                            "distributed selector {:?} resource {resource_index} parameter slot {parameter_slot} changes an unpartitioned parameter between fragments",
                            partition.selector_id,
                        )));
                    }
                    merged_parameters.push(first_parameter.clone());
                }
            }
            merged_resources.push(VulkanDistributedSelectedResourceFragmentPlan {
                resource_index,
                atomic_group_id: leading.atomic_group_id.clone(),
                logical_start: 0,
                logical_count: logical_frontier,
                parameters: merged_parameters,
            });
        }
        merged_by_selector.insert(partition.selector_id.clone(), merged_resources);
    }
    Ok(merged_by_selector)
}

fn sampled_distributed_dispatch_shard(
    dispatch: &VulkanDistributedDispatchPlan,
    source: &VulkanDistributedDispatchShard,
    device_id: &str,
    numerator: usize,
    denominator: usize,
) -> Result<VulkanDistributedDispatchShard, VulkanDistributedPlanError> {
    if source.row_count == 0 || source.workgroup_count_x == 0 || dispatch.row_alignment == 0 {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has invalid shard geometry",
            dispatch.component_id, dispatch.node_id,
        )));
    }
    // Selected-resource ownership and fragment shards are behaviorally complete
    // physical units. Whole-expert shaders consume the full router domain and
    // skip absent resources; TP-expert shaders consume an exact declared
    // fragment. Scaling either without rebuilding its resource contract would
    // measure work that cannot be replayed by the runtime.
    if (dispatch.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && !source.selected_resource_indices.is_empty())
        || !source.selected_resource_fragments.is_empty()
    {
        let mut exact = source.clone();
        exact.device_id = device_id.to_string();
        return Ok(exact);
    }
    let proportional_rows = source.row_count.checked_mul(numerator).ok_or_else(|| {
        VulkanDistributedPlanError("sampled distributed row calculation overflowed".to_string())
    })? / denominator;
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
        |range: &VulkanDistributedActivationRange, distribution: InputDistribution, label: &str| {
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
    let (workgroup_count_x, output_byte_count, auxiliary_input_ranges) = match dispatch.distribution
    {
        VulkanDistributedDispatchDistribution::OutputRows => {
            let output_byte_count = scale(source.output_byte_count, "output range")?;
            let workgroup_count_x =
                u32::try_from(scale(source.workgroup_count_x as usize, "workgroup count")?)
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
        selected_resource_indices: source.selected_resource_indices.clone(),
        selected_resource_fragments: source.selected_resource_fragments.clone(),
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
    resolved_physical_execution_islands_for_phase(
        dispatches,
        shared_activation_route,
        ExecutionPhase::Decode,
    )
}

pub(crate) fn resolved_physical_execution_islands_for_phase(
    dispatches: &[VulkanDistributedDispatchPlan],
    shared_activation_route: VulkanSharedResidentBufferRoute,
    phase: ExecutionPhase,
) -> Result<Vec<VulkanPhysicalExecutionIslandPlan>, VulkanDistributedPlanError> {
    let mut groups = Vec::<Vec<VulkanDistributedDispatchPlan>>::new();
    for dispatch in dispatches {
        if let Some(group) = groups.last_mut()
            && distributed_dispatches_can_share_sequence(
                group
                    .last()
                    .expect("physical execution islands are never empty"),
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
            resolved_physical_execution_island(index, dispatches, shared_activation_route, phase)
        })
        .collect()
}

fn resolved_physical_execution_island(
    island_index: usize,
    dispatches: Vec<VulkanDistributedDispatchPlan>,
    shared_activation_route: VulkanSharedResidentBufferRoute,
    phase: ExecutionPhase,
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
        dispatch.has_lazy_resource_requirements && dispatch.selected_resource_partitions.is_empty()
    }) {
        return Err(VulkanDistributedPlanError(format!(
            "physical execution island for component {component_id:?} contains lazy resources without a resolved atomic residency plan",
        )));
    }

    let mut participant_roles =
        BTreeMap::<String, BTreeSet<VulkanPhysicalExecutionParticipantRole>>::new();
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
    let mut shared_activation_allocations =
        BTreeMap::<(String, VulkanDistributedActivationStorage, String, usize), usize>::new();
    let mut private_intermediate_allocations =
        BTreeMap::<(String, String, usize, usize, usize), usize>::new();
    let mut owner_residency =
        BTreeMap::<(String, VulkanPhysicalExecutionResidencyKind, String), usize>::new();
    let private_handoff_producers = dispatches
        .windows(2)
        .filter(|pair| local_shard_handoff(&pair[0], &pair[1]))
        .map(|pair| pair[0].dispatch_index)
        .collect::<BTreeSet<_>>();
    let private_handoff_consumers = dispatches
        .windows(2)
        .filter(|pair| local_shard_handoff(&pair[0], &pair[1]))
        .map(|pair| pair[1].dispatch_index)
        .collect::<BTreeSet<_>>();

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
                    requirement.resource_id, requirement.byte_capacity,
                )));
            }
        }
        let activations = std::iter::once(&dispatch.input_activation)
            .filter(|_| !private_handoff_consumers.contains(&dispatch.dispatch_index))
            .chain(&dispatch.auxiliary_input_activations)
            .chain(
                std::iter::once(&dispatch.output_activation)
                    .filter(|_| !private_handoff_producers.contains(&dispatch.dispatch_index)),
            );
        for activation in activations {
            let allocation_device_id =
                distributed_activation_owner_device_id(&dispatch.owner_device_id, activation);
            let key = (
                allocation_device_id,
                activation.storage.clone(),
                activation.component_id.clone(),
                activation.slot,
            );
            if let Some(existing) =
                shared_activation_allocations.insert(key, activation.byte_capacity)
                && existing != activation.byte_capacity
            {
                return Err(VulkanDistributedPlanError(format!(
                    "physical execution island activation {}.slot_{} has conflicting capacities {existing} and {}",
                    activation.component_id, activation.slot, activation.byte_capacity,
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
            let parameter_bytes =
                shard
                    .parameters
                    .iter()
                    .try_fold(0usize, |total, parameter| {
                        total.checked_add(parameter.byte_count).ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "physical execution shard parameter bytes overflowed".to_string(),
                            )
                        })
                    })?;
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
    let entry_device_id =
        distributed_activation_owner_device_id(&owner_device_id, &first.input_activation);
    let tail = dispatches
        .last()
        .expect("physical execution island was checked above");
    let exit_device_id =
        distributed_activation_owner_device_id(&owner_device_id, &tail.output_activation);
    if let Some(collect) = schedule
        .iter_mut()
        .rev()
        .find(|step| step.kind == VulkanPhysicalExecutionScheduleKind::CollectOutputs)
    {
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
        .map(
            |(device_id, byte_capacity)| VulkanPhysicalExecutionResidencyRequirement {
                device_id,
                kind: VulkanPhysicalExecutionResidencyKind::PermanentParameterShard,
                resource_id: "parameter_shards".to_string(),
                byte_capacity,
            },
        )
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
        if local_shard_handoff(producer, consumer) {
            for (producer_shard, consumer_shard) in producer.shards.iter().zip(&consumer.shards) {
                let key = (
                    producer_shard.device_id.clone(),
                    producer.output_activation.component_id.clone(),
                    producer.output_activation.slot,
                    producer.dispatch_index,
                    consumer.dispatch_index,
                );
                if let Some(existing) =
                    private_intermediate_allocations.insert(key, producer_shard.output_byte_count)
                    && existing != producer_shard.output_byte_count
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "physical execution private intermediate {}.slot_{} has conflicting capacities {existing} and {}",
                        producer.output_activation.component_id,
                        producer.output_activation.slot,
                        producer_shard.output_byte_count,
                    )));
                }
                debug_assert_eq!(
                    producer_shard.output_byte_count,
                    consumer_shard.input_range.byte_count
                );
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
        |(
            (
                allocation_device_id,
                component_id,
                slot,
                producer_dispatch_index,
                consumer_dispatch_index,
            ),
            per_lane_byte_capacity,
        )| {
            VulkanPhysicalExecutionTransientMemoryRequirement {
                allocation_device_id,
                kind: VulkanPhysicalExecutionTransientMemoryKind::PrivateShardIntermediate,
                resource_id: format!(
                    "private_activation:{component_id}:slot_{slot}:{producer_dispatch_index}->{consumer_dispatch_index}"
                ),
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
            phase,
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
        | VulkanDistributedActivationStorage::BoundaryOutput => default_owner_device_id.to_string(),
    }
}

fn distributed_dispatches_can_share_sequence(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    if producer.owner_device_id != consumer.owner_device_id
        || producer.component_id != consumer.component_id
        || producer.dispatch_index.checked_add(1) != Some(consumer.dispatch_index)
        || !same_distributed_activation(&producer.output_activation, &consumer.input_activation)
        || producer.shards.len() != consumer.shards.len()
    {
        return false;
    }
    let shards_match = producer
        .shards
        .iter()
        .zip(&consumer.shards)
        .all(|(producer, consumer)| {
            producer.device_id == consumer.device_id
                && producer.row_start == consumer.row_start
                && producer.row_count == consumer.row_count
        });
    if !shards_match {
        return false;
    }
    if producer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && consumer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
    {
        return expert_local_shard_handoff(producer, consumer);
    }
    dense_local_shard_handoff(producer, consumer)
}

fn local_shard_handoff(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    dense_local_shard_handoff(producer, consumer) || expert_local_shard_handoff(producer, consumer)
}

/// Proves that two adjacent selected-resource kernels consume the same atomic
/// cohort on the same devices. Parameter projections and descriptor bindings
/// are intentionally allowed to differ: separate kernels can consume distinct
/// members of one all-or-nothing residency group. If any selector, route,
/// cohort, or shard range differs, the operations remain separate execution
/// islands and therefore separate residency transactions.
fn selected_resource_cohort_handoff(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    if producer.selected_resource_partitions.is_empty()
        || consumer.selected_resource_partitions.is_empty()
    {
        return producer.selected_resource_partitions.is_empty()
            && consumer.selected_resource_partitions.is_empty();
    }
    if producer.selected_resource_partitions.len() != consumer.selected_resource_partitions.len() {
        return false;
    }

    producer
        .selected_resource_partitions
        .iter()
        .all(|producer_partition| {
            let matching = consumer
                .selected_resource_partitions
                .iter()
                .filter(|partition| partition.selector_id == producer_partition.selector_id)
                .collect::<Vec<_>>();
            let [consumer_partition] = matching.as_slice() else {
                return false;
            };
            let selector_identity_matches = producer_partition.execution_scope
                == consumer_partition.execution_scope
                && producer_partition.node_id == consumer_partition.node_id
                && producer_partition.domain_id == consumer_partition.domain_id
                && producer_partition.selection_signal == consumer_partition.selection_signal
                && producer_partition.resource_count == consumer_partition.resource_count
                && producer_partition.selection_count_per_activation
                    == consumer_partition.selection_count_per_activation
                && producer_partition.atomic_group_ids == consumer_partition.atomic_group_ids
                && producer_partition.atomic_group_byte_counts
                    == consumer_partition.atomic_group_byte_counts
                && producer_partition.atomic_group_resource_ids
                    == consumer_partition.atomic_group_resource_ids;
            if !selector_identity_matches {
                return false;
            }

            let producer_activation = distributed_selected_resource_activation(
                producer,
                &producer_partition.selection_signal,
            );
            let consumer_activation = distributed_selected_resource_activation(
                consumer,
                &consumer_partition.selection_signal,
            );
            if !producer_activation
                .zip(consumer_activation)
                .is_some_and(|(producer, consumer)| same_distributed_activation(producer, consumer))
            {
                return false;
            }

            producer.shards.iter().zip(&consumer.shards).all(
                |(producer_shard, consumer_shard)| {
                    let whole_resources_match = producer_shard
                        .selected_resource_indices
                        .get(&producer_partition.selector_id)
                        .zip(
                            consumer_shard
                                .selected_resource_indices
                                .get(&consumer_partition.selector_id),
                        )
                        .is_some_and(|(producer, consumer)| producer == consumer);
                    let fragments_match = producer_shard
                        .selected_resource_fragments
                        .get(&producer_partition.selector_id)
                        .zip(
                            consumer_shard
                                .selected_resource_fragments
                                .get(&consumer_partition.selector_id),
                        )
                        .is_some_and(|(producer, consumer)| {
                            producer.len() == consumer.len()
                                && producer.iter().zip(consumer).all(|(producer, consumer)| {
                                    producer.resource_index == consumer.resource_index
                                        && producer.atomic_group_id == consumer.atomic_group_id
                                        && producer.logical_start == consumer.logical_start
                                        && producer.logical_count == consumer.logical_count
                                })
                        });
                    whole_resources_match ^ fragments_match
                },
            )
        })
}

fn distributed_selected_resource_activation<'a>(
    dispatch: &'a VulkanDistributedDispatchPlan,
    selection_signal: &str,
) -> Option<&'a VulkanDistributedActivationSlot> {
    let mut matching = std::iter::once(&dispatch.input_activation)
        .chain(dispatch.auxiliary_input_activations.iter())
        .filter(|activation| {
            activation.component_id == dispatch.component_id
                && activation.signal_id == selection_signal
        });
    let activation = matching.next()?;
    matching.next().is_none().then_some(activation)
}

fn expert_local_shard_handoff(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    producer.owner_device_id == consumer.owner_device_id
        && producer.component_id == consumer.component_id
        && producer.dispatch_index.checked_add(1) == Some(consumer.dispatch_index)
        && producer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && consumer.distribution == VulkanDistributedDispatchDistribution::ExpertRange
        && producer.output_collection == OutputCollection::Routed
        && consumer.input_distribution == InputDistribution::Routed
        && same_distributed_activation(&producer.output_activation, &consumer.input_activation)
        && selected_resource_cohort_handoff(producer, consumer)
        && producer.shards.len() == consumer.shards.len()
        && producer
            .shards
            .iter()
            .zip(&consumer.shards)
            .all(|(producer_shard, consumer_shard)| {
                producer_shard.device_id == consumer_shard.device_id
                    && producer_shard.row_start == consumer_shard.row_start
                    && producer_shard.row_count == consumer_shard.row_count
                    && producer_shard.base_workgroup_z == consumer_shard.base_workgroup_z
                    && producer_shard.output_byte_count == consumer_shard.input_range.byte_count
            })
        && declared_local_shard_handoff(producer, consumer)
}

fn declared_local_shard_handoff(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    producer.local_intermediates.iter().any(|intermediate| {
        intermediate.signal == producer.output_activation.signal_id
            && usize::try_from(intermediate.producer_binding).ok()
                == Some(producer.output_activation.binding)
            && usize::try_from(intermediate.consumer_binding).ok()
                == Some(consumer.input_activation.binding)
            && consumer.local_intermediates.contains(intermediate)
    })
}

fn dense_local_shard_handoff(
    producer: &VulkanDistributedDispatchPlan,
    consumer: &VulkanDistributedDispatchPlan,
) -> bool {
    producer.owner_device_id == consumer.owner_device_id
        && producer.component_id == consumer.component_id
        && producer.dispatch_index.checked_add(1) == Some(consumer.dispatch_index)
        && same_distributed_activation(&producer.output_activation, &consumer.input_activation)
        && !producer.shards.is_empty()
        && producer.shards.len() == consumer.shards.len()
        && producer.distribution == VulkanDistributedDispatchDistribution::OutputRows
        && producer.output_collection == OutputCollection::Concatenated
        && consumer.distribution == VulkanDistributedDispatchDistribution::InputColumns
        && consumer.input_distribution == InputDistribution::Sharded
        && selected_resource_cohort_handoff(producer, consumer)
        && producer
            .shards
            .iter()
            .zip(&consumer.shards)
            .all(|(producer_shard, consumer_shard)| {
                producer_shard.device_id == consumer_shard.device_id
                    && producer_shard.row_start == consumer_shard.row_start
                    && producer_shard.row_count == consumer_shard.row_count
                    && producer_shard.output_byte_offset == consumer_shard.input_range.byte_offset
                    && producer_shard.output_byte_count == consumer_shard.input_range.byte_count
            })
        && declared_local_shard_handoff(producer, consumer)
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
    pub physical_artifact_id: String,
    pub physical_execution_contract_id: String,
    pub implementation_digest: String,
    pub execution_strategy: nerve_execution_contracts::ExecutionStrategy,
    pub equivalence: VulkanDistributedEquivalencePlan,
    pub contract_member_node_ids: Vec<String>,
    pub local_intermediates: Vec<nerve_execution_contracts::LocalIntermediateContract>,
    pub has_lazy_resource_requirements: bool,
    pub selected_resource_partitions: Vec<VulkanDistributedSelectedResourcePartitionPlan>,
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

pub(crate) fn vulkan_distributed_placement_strategy(
    device_count: usize,
    strategies: impl IntoIterator<Item = nerve_execution_contracts::ExecutionStrategy>,
) -> Result<crate::vulkan_stream_circuit::VulkanPlacementExecutionStrategy, VulkanDistributedPlanError>
{
    use crate::vulkan_stream_circuit::VulkanPlacementExecutionStrategy as Placement;
    use nerve_execution_contracts::ExecutionStrategy;

    if device_count == 0 {
        return Err(VulkanDistributedPlanError(
            "distributed placement strategy requires at least one physical device".to_string(),
        ));
    }
    let mut saw_tensor_parallel = false;
    let mut saw_expert_parallel = false;
    let mut saw_intra_expert_tensor_parallel = false;
    let mut dispatch_count = 0usize;
    for strategy in strategies {
        dispatch_count += 1;
        match strategy {
            ExecutionStrategy::TensorParallel => saw_tensor_parallel = true,
            ExecutionStrategy::ExpertParallel => saw_expert_parallel = true,
            ExecutionStrategy::TensorParallelExpert => {
                saw_intra_expert_tensor_parallel = true
            }
            ExecutionStrategy::SingleDevice => {
                return Err(VulkanDistributedPlanError(
                    "distributed placement contains a single-device physical contract"
                        .to_string(),
                ));
            }
        }
    }
    if dispatch_count == 0 {
        return Err(VulkanDistributedPlanError(
            "distributed placement strategy requires at least one partitioned dispatch"
                .to_string(),
        ));
    }
    if device_count == 1 {
        return Ok(Placement::SingleDevice);
    }
    match (
        saw_tensor_parallel,
        saw_expert_parallel,
        saw_intra_expert_tensor_parallel,
    ) {
        (true, false, false) => Ok(Placement::TensorParallel),
        (false, true, false) => Ok(Placement::WholeExpertParallel),
        (false, false, true) => Ok(Placement::IntraExpertTensorParallel),
        _ => Ok(Placement::Hybrid),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulkanDistributedEquivalenceKind {
    BitExact,
    AbsoluteRelativeTolerance,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedEquivalencePlan {
    pub output: VulkanDistributedEquivalenceKind,
    pub state: VulkanDistributedEquivalenceKind,
    pub absolute_tolerance_bits: Option<u64>,
    pub relative_tolerance_bits: Option<u64>,
}

impl VulkanDistributedEquivalencePlan {
    fn from_contract(equivalence: &nerve_execution_contracts::EquivalenceRequirement) -> Self {
        let kind = |value| match value {
            nerve_execution_contracts::EquivalenceKind::BitExact => {
                VulkanDistributedEquivalenceKind::BitExact
            }
            nerve_execution_contracts::EquivalenceKind::AbsoluteRelativeTolerance => {
                VulkanDistributedEquivalenceKind::AbsoluteRelativeTolerance
            }
        };
        Self {
            output: kind(equivalence.output),
            state: kind(equivalence.state),
            absolute_tolerance_bits: equivalence.absolute_tolerance.map(f64::to_bits),
            relative_tolerance_bits: equivalence.relative_tolerance.map(f64::to_bits),
        }
    }

    pub fn absolute_tolerance(&self) -> Option<f64> {
        self.absolute_tolerance_bits.map(f64::from_bits)
    }

    pub fn relative_tolerance(&self) -> Option<f64> {
        self.relative_tolerance_bits.map(f64::from_bits)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourcePartitionPlan {
    pub execution_scope: String,
    pub selector_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub selection_signal: String,
    pub address_table_binding: usize,
    pub parameter_slots_binding: usize,
    pub resource_count: usize,
    pub parameters_per_resource: usize,
    pub parameter_partitions: Vec<VulkanDistributedSelectedResourceParameterPartitionPlan>,
    pub selection_count_per_activation: usize,
    pub atomic_group_ids: Vec<String>,
    pub atomic_group_byte_counts: Vec<usize>,
    pub atomic_group_resource_ids: Vec<Vec<String>>,
    pub parameter_resource_ids: Vec<Vec<String>>,
    pub parameter_resource_byte_counts: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceParameterPartitionPlan {
    pub parameter_slot: usize,
    pub dimension: usize,
    pub kind: nerve_execution_contracts::ParameterPartitionKind,
    pub alignment_elements: usize,
    pub logical_elements_per_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedReductionPlan {
    pub operation: ReductionOperation,
    pub element_count: usize,
    pub partial_byte_capacity: usize,
    pub finalization: VulkanDistributedReductionFinalizationPlan,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanDistributedReductionFinalizationPlan {
    StoreF32,
    StoreF32ToBf16,
    AddBf16ResidualToBf16 { residual_input_index: usize },
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
