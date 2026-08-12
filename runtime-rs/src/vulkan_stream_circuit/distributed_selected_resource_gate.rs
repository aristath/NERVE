pub(crate) struct VulkanDistributedSelectedResourceGate {
    selector_id: String,
    checkpoint_tag: u32,
    resource_count: usize,
    selection_count_per_lane: usize,
    gate: VulkanGpuResidencyGate,
    pipeline_predicate: Arc<VulkanResidentBuffer>,
    gate_push_constants: Vec<u8>,
    context: VulkanDemandResidencyExecutionContext,
    observed_notification_epoch: Cell<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VulkanDistributedSelectedResourceResolvedMiss {
    pub selector_id: String,
    pub checkpoint_tag: u32,
    pub resource_indices: Vec<usize>,
}

impl VulkanDistributedSelectedResourceGate {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &VulkanComputeDevice,
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
        let owned_resource_indices = store
            .owned_selector_resource_indices(&selector.id)
            .cloned()
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed selected-resource store does not own selector {:?}",
                    selector.id
                ))
            })?;
        let selection_count = selector
            .encoding
            .selection_count_per_activation
            .checked_mul(lane_count)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed selected-resource selection count overflowed".to_string(),
                )
            })?;
        let missing_queue = VulkanGpuResidencyMissQueue::new(device, selection_count)
            .map_err(VulkanDistributedDispatchRunnerError::from)?;
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
        let gate_push_constants = gate
            .push_constants(selection_count, checkpoint_tag, false, false)
            .map_err(VulkanDistributedDispatchRunnerError::from)?
            .to_vec();
        let owner = DeviceResourceResidencyOwnerId::new(format!(
            "distributed:{}:{}:{}:{}",
            execution_scope, dispatch.component_id, dispatch.node_id, checkpoint_tag
        ))
        .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
        Ok(Self {
            selector_id: selector.id.clone(),
            checkpoint_tag,
            resource_count: selector.resource_count,
            selection_count_per_lane: selector.encoding.selection_count_per_activation,
            gate,
            pipeline_predicate: transaction_predicate,
            gate_push_constants,
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
            .push_constants(selection_count, self.checkpoint_tag, false, false)
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

    pub(crate) fn ensure_execution_headroom(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        self.context
            .store
            .ensure_execution_headroom(device)
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))
    }

    pub(crate) fn store_identity(&self) -> usize {
        Arc::as_ptr(&self.context.store) as usize
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
            &self.gate_push_constants,
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
        self.gate
            .continuation_predicate()
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanDistributedDispatchRunnerError::from)
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

    pub(crate) fn resolve_completed_miss(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        Option<VulkanDistributedSelectedResourceResolvedMiss>,
        VulkanDistributedDispatchRunnerError,
    > {
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
        self.context
            .store
            .record_gpu_gate_misses(&self.selector_id, missing.requests.len())
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
        self.context
            .store
            .load_selector_resources_for_resume(
                device,
                &self.selector_id,
                &resource_indices,
                self.context.owner.clone(),
            )
            .map_err(|error| VulkanDistributedDispatchRunnerError(error.to_string()))?;
        self.acknowledge_missing_through(missing.published_count)?;
        self.observe_notification_epoch(notification_epoch);
        self.gate
            .continuation_predicate()
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanDistributedDispatchRunnerError::from)?;
        Ok(Some(VulkanDistributedSelectedResourceResolvedMiss {
            selector_id: self.selector_id.clone(),
            checkpoint_tag: self.checkpoint_tag,
            resource_indices,
        }))
    }
}
