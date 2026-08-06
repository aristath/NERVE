struct VulkanResidentDemandFeedbackState {
    predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
    state_transactions: Vec<VulkanResidentStateTransactionBank>,
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
    // Every physical device owns a distinct Vulkan buffer view, including when
    // those views import the same external allocation. A host write through one
    // view is not a cross-device visibility operation. Explicitly write every
    // view before submitting another demand attempt so all queues start from
    // the same continuation state.
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
    fn new<'a, F, E>(
        predicates_by_device: BTreeMap<String, Arc<VulkanResidentBuffer>>,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        device_for: &F,
    ) -> Result<Self, VulkanError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        if predicates_by_device.len() != device_slices.len()
            || device_slices
                .iter()
                .any(|slice| !predicates_by_device.contains_key(&slice.device_id))
        {
            return Err(VulkanError(
                "demand feedback predicates do not cover every placed device".to_string(),
            ));
        }
        let state_transactions = device_slices
            .iter()
            .map(|slice| {
                let device = device_for(&slice.device_id).map_err(|error| {
                    VulkanError(format!(
                        "demand feedback transaction device {:?} resolution failed: {error}",
                        slice.device_id
                    ))
                })?;
                VulkanResidentStateTransactionBank::new_transactional(
                    device,
                    &slice.mounted.buffers,
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
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
        Ok(Self {
            predicates_by_device,
            state_transactions,
        })
    }

    fn reset_pipeline_predicate(&self) -> Result<(), VulkanError> {
        write_shared_device_predicate_views(self.predicates_by_device.values(), true)
    }

    fn capture_window_baseline(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    ) -> Result<(), VulkanError> {
        self.for_each_transaction(device_slices, |transaction, slice| {
            transaction.capture_baseline(&slice.mounted.buffers)
        })
    }

    fn restore_window_baseline(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    ) -> Result<(), VulkanError> {
        self.for_each_transaction(device_slices, |transaction, slice| {
            transaction.restore_baseline(&slice.mounted.buffers)
        })
    }

    fn for_each_transaction(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        mut run: impl FnMut(
            &VulkanResidentStateTransactionBank,
            &VulkanResidentInProcessPlacedStreamProcessorDevice,
        ) -> Result<(), VulkanError>,
    ) -> Result<(), VulkanError> {
        if self.state_transactions.len() != device_slices.len() {
            return Err(VulkanError(format!(
                "demand feedback has {} state transactions for {} device slices",
                self.state_transactions.len(),
                device_slices.len()
            )));
        }
        for (transaction, slice) in self.state_transactions.iter().zip(device_slices) {
            run(transaction, slice)?;
        }
        Ok(())
    }

    fn begin_execution<'a>(
        &self,
        device_slices: &'a [VulkanResidentInProcessPlacedStreamProcessorDevice],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<Vec<VulkanCompiledResourceExecutionGuard<'a>>, VulkanError> {
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
            .map(|(slice, context)| {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanError(format!(
                        "demand feedback has no bound device {:?}",
                        slice.device_id
                    ))
                })?;
                context.store.begin_execution(device).map_err(|error| {
                    VulkanError(format!(
                        "failed to enter demand feedback execution epoch on {:?}: {error}",
                        slice.device_id
                    ))
                })
            })
            .collect()
    }

    fn resolve_first_miss(
        &self,
        device_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        tick_count: usize,
        sequence_variant: u8,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
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
            return Ok(false);
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
        demand
            .resolve_feedback_lane_miss(device, sequence_variant, feedback_lane)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)
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
