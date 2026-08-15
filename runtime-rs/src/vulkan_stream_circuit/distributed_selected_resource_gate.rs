pub(crate) struct VulkanDistributedSelectedResourceGate {
    logical_device_id: String,
    selector_id: String,
    checkpoint_tag: u32,
    resource_count: usize,
    selection_count_per_lane: usize,
    gate: VulkanGpuResidencyGate,
    pipeline_predicate: Arc<VulkanResidentBuffer>,
    gate_push_constants: Vec<u8>,
    feedback_gate_push_constants: Option<Vec<u8>>,
    context: VulkanDemandResidencyExecutionContext,
    observed_notification_epoch: Cell<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VulkanDistributedSelectedResourceResolvedMiss {
    pub selector_id: String,
    pub checkpoint_tag: u32,
    pub resource_indices: Vec<usize>,
}

struct VulkanDistributedSelectedResourcePendingMiss {
    notification_epoch: u32,
    published_count: u32,
    request_count: usize,
    resource_indices: Vec<usize>,
}

fn reset_distributed_selected_resource_predicate(
    predicate: &VulkanResidentBuffer,
) -> Result<(), VulkanDistributedDispatchRunnerError> {
    if predicate.byte_capacity() >= VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY {
        predicate
            .write_bytes(&demand_feedback_ready_predicate_bytes())
            .map_err(VulkanDistributedDispatchRunnerError::from)
    } else {
        predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }
}

impl VulkanDistributedSelectedResourceGate {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &VulkanComputeDevice,
        logical_device_id: &str,
        execution_scope: &str,
        dispatch: &VulkanDistributedDispatchPlan,
        partition: &VulkanDistributedSelectedResourcePartitionPlan,
        selection_buffer: Arc<VulkanResidentBuffer>,
        selection_lane_stride_bytes: usize,
        lane_count: usize,
        dynamic_resources: &VulkanDynamicResourceBuffers,
        store: Arc<VulkanCompiledResourceDeviceStore>,
        local_predicate: Arc<VulkanResidentBuffer>,
        transaction_predicate: Arc<VulkanResidentBuffer>,
        checkpoint_tag: u32,
    ) -> Result<Self, VulkanDistributedDispatchRunnerError> {
        if logical_device_id.is_empty()
            || !store
                .logical_device_ids()
                .iter()
                .any(|candidate| candidate == logical_device_id)
        {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selected-resource gate {}.{} maps logical device {logical_device_id:?} outside its physical store",
                dispatch.component_id, dispatch.node_id,
            )));
        }
        if lane_count == 0 || selection_lane_stride_bytes % size_of::<u32>() != 0 {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selected-resource gate {}.{} has an invalid lane layout",
                dispatch.component_id, dispatch.node_id
            )));
        }
        let selector = store
            .contract
            .selectors
            .iter()
            .find(|selector| selector.id == partition.selector_id)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource gate references unknown selector {:?}",
                    partition.selector_id
                ))
            })?;
        if selector.execution_scope != execution_scope
            || selector.component_id != dispatch.component_id
            || selector.node_id != partition.node_id
            || selector.domain_id != partition.domain_id
            || selector.selection_signal != partition.selection_signal
            || selector.resource_count != partition.resource_count
            || selector.encoding.selection_count_per_activation
                != partition.selection_count_per_activation
        {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selected-resource selector {:?} disagrees with {}.{}",
                selector.id, dispatch.component_id, dispatch.node_id
            )));
        }
        let selector_layout = store
            .layout
            .selectors
            .iter()
            .find(|layout| layout.selector_id == selector.id)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource selector {:?} has no address layout",
                    selector.id
                ))
            })?;
        let address_mapping = match &selector_layout.mapping {
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
        let addressable_resource_indices = store
            .owned_selector_resource_indices(&selector.id)
            .cloned()
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource store does not own selector {:?}",
                    selector.id
                ))
            })?;
        let owned_resource_indices = distributed_shard_selected_resource_ownership(
            &dispatch.shards,
            &partition.selector_id,
            partition.resource_count,
            logical_device_id,
            &dispatch.component_id,
            &dispatch.node_id,
        )?;
        if !owned_resource_indices.is_subset(&addressable_resource_indices) {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selected-resource shard on {logical_device_id:?} executes selector {:?} resources outside its physical store addressability",
                selector.id,
            )));
        }
        let selection_count = selector
            .encoding
            .selection_count_per_activation
            .checked_mul(lane_count)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed selected-resource selection count overflowed".to_string(),
                )
            })?;
        let missing_queue = VulkanGpuResidencyMissQueue::new(device, selection_count).map_err(
            |error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource miss queue for {}.{} selector {:?} on {logical_device_id:?} with {lane_count} lanes and capacity {selection_count} failed: {error}",
                    dispatch.component_id, dispatch.node_id, selector.id,
                ))
            },
        )?;
        let gate = VulkanGpuResidencyGate::new(
            device,
            &vulkan_gpu_residency_gate_spirv_words()
                .map_err(VulkanDistributedDispatchRunnerError::from)?,
            selection_buffer,
            dynamic_resources.shared_address_table(),
            dynamic_resources.address_table_slot_count(),
            missing_queue,
            local_predicate,
            None,
            VulkanGpuResidencyGateConfig {
                maximum_selection_count: selection_count,
                selection_count_per_lane: selector.encoding.selection_count_per_activation,
                selection_lane_stride_words: selection_lane_stride_bytes / size_of::<u32>(),
                selection_index_shift: selector.encoding.index_shift,
                selection_index_mask: selector.encoding.index_mask,
                address_mapping,
                owned_resource_indices: Some(owned_resource_indices),
            },
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
        let fault_source_id = demand_feedback_distributed_fault_source_id(
            execution_scope,
            &dispatch.component_id,
            &dispatch.node_id,
            &partition.selector_id,
        );
        let gate_push_constants = gate
            .push_constants(selection_count, checkpoint_tag, false, false, 0)
            .map_err(VulkanDistributedDispatchRunnerError::from)?
            .to_vec();
        let feedback_gate_push_constants = (gate.transaction_predicate().byte_capacity()
            >= VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY)
            .then(|| {
                gate.push_constants(
                    selection_count,
                    checkpoint_tag,
                    false,
                    false,
                    fault_source_id,
                )
                .map(|bytes| bytes.to_vec())
                .map_err(VulkanDistributedDispatchRunnerError::from)
            })
            .transpose()?;
        let owner = DeviceResourceResidencyOwnerId::new(format!(
            "distributed:{}:{}:{}:{}",
            execution_scope, dispatch.component_id, dispatch.node_id, checkpoint_tag
        ))
        .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
        Ok(Self {
            logical_device_id: logical_device_id.to_string(),
            selector_id: selector.id.clone(),
            checkpoint_tag,
            resource_count: selector.resource_count,
            selection_count_per_lane: selector.encoding.selection_count_per_activation,
            gate,
            pipeline_predicate: transaction_predicate,
            gate_push_constants,
            feedback_gate_push_constants,
            context: VulkanDemandResidencyExecutionContext {
                execution_scope: execution_scope.to_string(),
                contract: Arc::clone(&store.contract),
                layout: Arc::clone(&store.layout),
                store,
                owner,
            },
            observed_notification_epoch: Cell::new(0),
        })
    }

    pub(crate) fn gate_step(
        &self,
    ) -> Result<VulkanResidentKernelSequenceStep<'_>, VulkanDistributedDispatchRunnerError> {
        VulkanResidentKernelSequenceStep::new(self.gate.dispatch(), &self.gate_push_constants)
            .with_condition(&self.pipeline_predicate, 0, false, self.checkpoint_tag)
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn gate_push_constants_for_lane_count(
        &self,
        lane_count: usize,
    ) -> Result<Vec<u8>, VulkanDistributedDispatchRunnerError> {
        let selection_count = self
            .selection_count_per_lane
            .checked_mul(lane_count)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed selected-resource active selection count overflowed"
                        .to_string(),
                )
            })?;
        self.gate
            .push_constants(selection_count, self.checkpoint_tag, false, false, 0)
            .map(|bytes| bytes.to_vec())
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn gate_step_with_push_constants<'a>(
        &'a self,
        push_constants: &'a [u8],
    ) -> Result<VulkanResidentKernelSequenceStep<'a>, VulkanDistributedDispatchRunnerError> {
        VulkanResidentKernelSequenceStep::new(self.gate.dispatch(), push_constants)
            .with_condition(&self.pipeline_predicate, 0, false, self.checkpoint_tag)
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn resource_count(&self) -> usize {
        self.resource_count
    }

    pub(crate) fn logical_device_id(&self) -> &str {
        &self.logical_device_id
    }

    pub(crate) fn selector_id(&self) -> &str {
        &self.selector_id
    }

    pub(crate) fn checkpoint_tag(&self) -> u32 {
        self.checkpoint_tag
    }

    pub(crate) fn selected_resource_indices(
        &self,
        lane_count: usize,
    ) -> Result<BTreeSet<usize>, VulkanDistributedDispatchRunnerError> {
        let active_selection_count = self
            .selection_count_per_lane
            .checked_mul(lane_count)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed selected-resource observation count overflowed".to_string(),
                )
            })?;
        self.gate
            .selected_resource_indices(active_selection_count)
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn owned_resource_indices(&self) -> &BTreeSet<usize> {
        self.gate
            .owned_resource_indices()
            .expect("distributed selected-resource gates are mounted with exact ownership")
    }

    pub(crate) fn replace_execution_ownership_at_quiescent_boundary(
        &mut self,
        owned_resource_indices: BTreeSet<usize>,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        let addressable_resource_indices = self
            .context
            .store
            .owned_selector_resource_indices(&self.selector_id)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource store no longer addresses selector {:?}",
                    self.selector_id,
                ))
            })?;
        if owned_resource_indices.is_empty()
            || owned_resource_indices
                .iter()
                .any(|resource_index| *resource_index >= self.resource_count)
            || !owned_resource_indices.is_subset(addressable_resource_indices)
        {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selected-resource ownership for selector {:?} on {:?} is empty or exceeds its addressability envelope",
                self.selector_id, self.logical_device_id,
            )));
        }
        self.gate
            .replace_owned_resource_indices_at_quiescent_boundary(
                owned_resource_indices,
            )
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    fn coordinator(
        &self,
    ) -> Result<Option<Arc<VulkanCompiledResourceDistributedCohortCoordinator>>, VulkanDistributedDispatchRunnerError>
    {
        self.context
            .store
            .distributed_cohort_coordinator()
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))
    }

    pub(crate) fn dispatch(&self) -> &VulkanResidentKernelDispatch {
        self.gate.dispatch()
    }

    pub(crate) fn auxiliary_transient_device_bytes(
        &self,
    ) -> Result<usize, VulkanDistributedDispatchRunnerError> {
        self.gate
            .auxiliary_transient_device_bytes()
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn indirect_gate_step<'a>(
        &'a self,
        indirect: &'a VulkanResidentBuffer,
        byte_offset: usize,
    ) -> Result<VulkanResidentKernelSequenceStep<'a>, VulkanDistributedDispatchRunnerError> {
        VulkanResidentKernelSequenceStep::new_indirect(
            self.gate.dispatch(),
            self.feedback_gate_push_constants.as_deref().ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed selected-resource feedback gate has no two-word fault predicate"
                        .to_string(),
                )
            })?,
            indirect,
            byte_offset,
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?
        .with_condition(&self.pipeline_predicate, 0, false, self.checkpoint_tag)
        .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn guard_step<'a>(
        &'a self,
        step: VulkanResidentKernelSequenceStep<'a>,
        region_id: u32,
    ) -> Result<VulkanResidentKernelSequenceStep<'a>, VulkanDistributedDispatchRunnerError> {
        step.with_condition(self.gate.continuation_predicate(), 0, false, region_id)
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn notification_epoch(&self) -> Result<u32, VulkanDistributedDispatchRunnerError> {
        self.gate
            .notification_epoch()
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn observed_notification_epoch(&self) -> u32 {
        self.observed_notification_epoch.get()
    }

    pub(crate) fn observe_notification_epoch(&self, epoch: u32) {
        self.observed_notification_epoch.set(epoch);
    }

    pub(crate) fn reset_local_predicate(&self) -> Result<(), VulkanDistributedDispatchRunnerError> {
        reset_distributed_selected_resource_predicate(self.gate.continuation_predicate())
    }

    pub(crate) fn missing_snapshot(
        &self,
    ) -> Result<VulkanGpuResidencyMissingSnapshot, VulkanDistributedDispatchRunnerError> {
        self.gate
            .missing_snapshot()
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    pub(crate) fn acknowledge_missing_through(
        &self,
        published_count: u32,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.gate
            .acknowledge_missing_through(published_count)
            .map_err(VulkanDistributedDispatchRunnerError::from)
    }

    fn pending_miss(
        &self,
    ) -> Result<Option<VulkanDistributedSelectedResourcePendingMiss>, VulkanDistributedDispatchRunnerError>
    {
        let notification_epoch = self.notification_epoch()?;
        if notification_epoch == self.observed_notification_epoch() {
            return Ok(None);
        }
        let missing = self.missing_snapshot()?;
        if missing.overflowed || missing.requests.is_empty() {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selector {:?} reported an invalid miss queue: epoch={}, requests={}, overflowed={}",
                self.selector_id,
                missing.notification_epoch,
                missing.requests.len(),
                missing.overflowed
            )));
        }
        if missing
            .requests
            .iter()
            .any(|request| request.checkpoint_tag != self.checkpoint_tag)
        {
            return Err(VulkanDistributedDispatchRunnerError(format!(
                "distributed selector {:?} miss queue crossed checkpoint tag {}",
                self.selector_id, self.checkpoint_tag
            )));
        }
        let resource_indices = exact_demand_miss_resource_indices(&missing.requests)
            .map_err(VulkanDistributedDispatchRunnerError::from)?;
        Ok(Some(VulkanDistributedSelectedResourcePendingMiss {
            notification_epoch,
            published_count: missing.published_count,
            request_count: missing.requests.len(),
            resource_indices,
        }))
    }

    fn load_pending_resources(
        &self,
        device: &VulkanComputeDevice,
        resource_indices: &[usize],
        cohort_mutation: Option<&VulkanCompiledResourceDistributedCohortMutation<'_>>,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.context
            .store
            .load_selector_resources_for_resume_with_cohort_mutation(
                device,
                &self.selector_id,
                &resource_indices,
                self.context.owner.clone(),
                cohort_mutation,
            )
            .map(|_| ())
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))
    }

    fn absent_resource_indices(
        &self,
        resource_indices: &[usize],
    ) -> Result<Vec<usize>, VulkanDistributedDispatchRunnerError> {
        self.context
            .store
            .absent_selector_resource_indices(&self.selector_id, resource_indices)
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))
    }

    fn rollback_absent_resources(
        &self,
        resource_indices: &[usize],
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.context
            .store
            .rollback_absent_selector_resources(&self.selector_id, resource_indices)
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))
    }

    fn commit_pending_miss(
        &self,
        pending: &VulkanDistributedSelectedResourcePendingMiss,
    ) -> Result<VulkanDistributedSelectedResourceResolvedMiss, VulkanDistributedDispatchRunnerError>
    {
        self.acknowledge_missing_through(pending.published_count)?;
        self.observe_notification_epoch(pending.notification_epoch);
        reset_distributed_selected_resource_predicate(self.gate.continuation_predicate())?;
        Ok(VulkanDistributedSelectedResourceResolvedMiss {
            selector_id: self.selector_id.clone(),
            checkpoint_tag: self.checkpoint_tag,
            resource_indices: pending.resource_indices.clone(),
        })
    }

}

pub(crate) fn distributed_shard_selected_resource_ownership(
    shards: &[VulkanDistributedDispatchShard],
    selector_id: &str,
    resource_count: usize,
    logical_device_id: &str,
    component_id: &str,
    node_id: &str,
) -> Result<BTreeSet<usize>, VulkanDistributedDispatchRunnerError> {
    let matching = shards
        .iter()
        .filter(|shard| shard.device_id == logical_device_id)
        .collect::<Vec<_>>();
    let [shard] = matching.as_slice() else {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed selected-resource dispatch {}.{} resolves {} shards on {logical_device_id:?}",
            component_id,
            node_id,
            matching.len(),
        )));
    };
    let whole = shard
        .selected_resource_indices
        .get(selector_id)
        .cloned()
        .unwrap_or_default();
    let fragments = shard
        .selected_resource_fragments
        .get(selector_id)
        .map(|fragments| {
            fragments
                .iter()
                .map(|fragment| fragment.resource_index)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if whole.is_empty() == fragments.is_empty() {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed selected-resource shard on {logical_device_id:?} must own whole resources or tensor fragments for selector {:?}",
            selector_id,
        )));
    }
    let ownership = whole
        .into_iter()
        .chain(fragments)
        .collect::<BTreeSet<_>>();
    if ownership.is_empty()
        || ownership.len()
            != shard
                .selected_resource_indices
                .get(selector_id)
                .map_or_else(
                    || {
                        shard
                            .selected_resource_fragments
                            .get(selector_id)
                            .map_or(0, Vec::len)
                    },
                    Vec::len,
                )
        || ownership.iter().any(|index| *index >= resource_count)
    {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed selected-resource shard on {logical_device_id:?} has duplicate or out-of-range ownership for selector {:?}",
            selector_id,
        )));
    }
    Ok(ownership)
}
pub(crate) fn resolve_distributed_selected_resource_misses<'a>(
    gates: &[(&'a VulkanDistributedSelectedResourceGate, &'a VulkanComputeDevice)],
) -> Result<Vec<(usize, VulkanDistributedSelectedResourceResolvedMiss)>, VulkanDistributedDispatchRunnerError>
{
    let mut coordinator = None::<Arc<VulkanCompiledResourceDistributedCohortCoordinator>>;
    let mut pending = Vec::with_capacity(gates.len());
    let mut observations = Vec::with_capacity(gates.len());
    for (gate, _) in gates {
        if let Some(candidate) = gate.coordinator()? {
            match &coordinator {
                Some(current) if !Arc::ptr_eq(current, &candidate) => {
                    return Err(VulkanDistributedDispatchRunnerError(
                        "distributed selected-resource gates belong to different residency coordinators"
                            .to_string(),
                    ));
                }
                Some(_) => {}
                None => coordinator = Some(candidate),
            }
        }
        let gate_pending = gate.pending_miss()?;
        observations.push(VulkanCompiledResourceDistributedFaultObservation {
            logical_device_id: gate.logical_device_id().to_string(),
            selector_id: gate.selector_id().to_string(),
            checkpoint_tag: gate.checkpoint_tag(),
            pending_resource_indices: gate_pending
                .as_ref()
                .map(|pending| pending.resource_indices.clone())
                .unwrap_or_default(),
        });
        pending.push(gate_pending);
    }
    let _mutation = coordinator
        .as_ref()
        .map(|coordinator| coordinator.begin_mutation())
        .transpose()
        .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
    let plan = if let Some(coordinator) = &coordinator {
        coordinator
            .plan_fault_resolution(&observations)
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?
    } else {
        VulkanCompiledResourceDistributedFaultPlan {
            loads: pending
                .iter()
                .enumerate()
                .filter_map(|(observation_index, pending)| {
                    pending.as_ref().map(|pending| VulkanCompiledResourceDistributedFaultLoad {
                        observation_index,
                        resource_indices: pending.resource_indices.clone(),
                    })
                })
                .collect(),
            commit_observation_indices: pending
                .iter()
                .enumerate()
                .filter_map(|(index, pending)| pending.as_ref().map(|_| index))
                .collect(),
        }
    };
    for (gate, pending) in gates.iter().map(|(gate, _)| *gate).zip(&pending) {
        if let Some(pending) = pending {
            gate.context
                .store
                .record_gpu_gate_misses(gate.selector_id(), pending.request_count)
                .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
        }
    }
    let rollback = plan
        .loads
        .iter()
        .map(|load| {
            gates[load.observation_index]
                .0
                .absent_resource_indices(&load.resource_indices)
                .map(|absent| (load.observation_index, absent))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for load in &plan.loads {
        let (gate, device) = gates[load.observation_index];
        if let Err(load_error) = gate.load_pending_resources(
            device,
            &load.resource_indices,
            _mutation.as_ref(),
        ) {
            let rollback_errors = rollback
                .iter()
                .rev()
                .filter_map(|(observation_index, absent)| {
                    gates[*observation_index]
                        .0
                        .rollback_absent_resources(absent)
                        .err()
                        .map(|error| error.to_string())
                })
                .collect::<Vec<_>>();
            return if rollback_errors.is_empty() {
                Err(load_error)
            } else {
                Err(VulkanDistributedDispatchRunnerError(format!(
                    "{load_error}; distributed residency rollback also failed: {}",
                    rollback_errors.join("; "),
                )))
            };
        }
    }
    plan.commit_observation_indices
        .into_iter()
        .map(|observation_index| {
            let pending = pending[observation_index]
                .as_ref()
                .expect("fault plan commits only pending observations");
            gates[observation_index]
                .0
                .commit_pending_miss(pending)
                .map(|miss| (observation_index, miss))
        })
        .collect()
}

#[cfg(test)]
mod distributed_selected_resource_gate_tests {
    use super::*;

    fn shard(
        device_id: &str,
        whole: Vec<usize>,
        fragments: Vec<usize>,
    ) -> VulkanDistributedDispatchShard {
        VulkanDistributedDispatchShard {
            device_id: device_id.to_string(),
            selected_resource_indices: (!whole.is_empty())
                .then(|| ("experts".to_string(), whole))
                .into_iter()
                .collect(),
            selected_resource_fragments: (!fragments.is_empty())
                .then(|| {
                    (
                        "experts".to_string(),
                        fragments
                            .into_iter()
                            .map(|resource_index| {
                                VulkanDistributedSelectedResourceFragmentPlan {
                                    resource_index,
                                    atomic_group_id: format!("expert-{resource_index}"),
                                    logical_start: 0,
                                    logical_count: 4,
                                    parameters: Vec::new(),
                                }
                            })
                            .collect(),
                    )
                })
                .into_iter()
                .collect(),
            row_start: 0,
            row_count: 4,
            workgroup_count_x: 1,
            base_workgroup_z: 0,
            input_range: VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: 8,
            },
            auxiliary_input_ranges: Vec::new(),
            output_byte_offset: 0,
            output_byte_count: 8,
            parameters: Vec::new(),
        }
    }

    #[test]
    fn execution_ownership_comes_from_the_exact_dispatch_shard() {
        let shards = vec![shard("gpu0", vec![0, 2], Vec::new()), shard("gpu1", vec![1, 3], Vec::new())];

        assert_eq!(
            distributed_shard_selected_resource_ownership(
                &shards,
                "experts",
                4,
                "gpu0",
                "layer",
                "gate-up",
            )
            .unwrap(),
            BTreeSet::from([0, 2]),
        );
        assert_eq!(
            distributed_shard_selected_resource_ownership(
                &shards,
                "experts",
                4,
                "gpu1",
                "layer",
                "gate-up",
            )
            .unwrap(),
            BTreeSet::from([1, 3]),
        );
    }

    #[test]
    fn execution_ownership_accepts_tensor_fragments_but_rejects_ambiguity() {
        let fragmented = vec![
            shard("gpu0", Vec::new(), vec![0, 1]),
            shard("gpu1", Vec::new(), vec![0, 1]),
        ];
        assert_eq!(
            distributed_shard_selected_resource_ownership(
                &fragmented,
                "experts",
                2,
                "gpu0",
                "layer",
                "gate-up",
            )
            .unwrap(),
            BTreeSet::from([0, 1]),
        );

        let mixed = vec![VulkanDistributedDispatchShard {
            selected_resource_fragments: fragmented[0]
                .selected_resource_fragments
                .clone(),
            ..shard("gpu0", vec![0], Vec::new())
        }];
        assert!(
            distributed_shard_selected_resource_ownership(
                &mixed,
                "experts",
                2,
                "gpu0",
                "layer",
                "gate-up",
            )
            .unwrap_err()
            .0
            .contains("whole resources or tensor fragments")
        );
        assert!(
            distributed_shard_selected_resource_ownership(
                &fragmented,
                "experts",
                2,
                "missing",
                "layer",
                "gate-up",
            )
            .unwrap_err()
            .0
            .contains("resolves 0 shards")
        );
    }
}
