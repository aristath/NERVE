struct VulkanResidentParallelSpeculativeFeedbackDecoderState {
    state_ingestion: VulkanResidentParallelSpeculativeStateIngestion,
    lane_source_copies: Vec<VulkanResidentBufferCopyBatch>,
    _source_histories: Vec<VulkanResidentBuffer>,
}

struct VulkanResidentParallelSpeculativeFeedbackState {
    decoders: Vec<VulkanResidentParallelSpeculativeFeedbackDecoderState>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanParallelSpeculativeFeedbackAllocationPlan {
    device_allocations: Vec<VulkanRuntimeDeviceLocalTransientAllocation>,
    host_visible_allocations: Vec<VulkanRuntimeHostVisibleTransientAllocation>,
}

impl VulkanParallelSpeculativeFeedbackAllocationPlan {
    fn from_decoder_allocations(
        decoder_allocations: impl IntoIterator<
            Item = (
                String,
                String,
                Vec<VulkanComponentBatchResidentAllocation>,
                Vec<(String, usize)>,
            ),
        >,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let mut device_allocations = Vec::new();
        let mut host_visible_allocations = Vec::new();
        for (decoder_id, device_id, runner_allocations, histories) in decoder_allocations {
            if decoder_id.is_empty() || device_id.is_empty() {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "parallel speculative feedback allocation requires decoder and device identities"
                            .to_string(),
                    ),
                ));
            }
            if runner_allocations
                .iter()
                .any(|allocation| allocation.byte_capacity == 0)
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "parallel speculative feedback runner allocation has zero capacity"
                            .to_string(),
                    ),
                ));
            }
            for allocation in runner_allocations {
                let concern = format!(
                    "speculative decoder {decoder_id} resident feedback state ingestion {:?}",
                    allocation.kind,
                );
                if allocation.host_visible {
                    host_visible_allocations.push(VulkanRuntimeHostVisibleTransientAllocation {
                        logical_device_id: device_id.clone(),
                        byte_capacity: allocation.byte_capacity,
                        concern,
                        allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
                    });
                } else {
                    device_allocations.push(VulkanRuntimeDeviceLocalTransientAllocation {
                        logical_device_id: device_id.clone(),
                        participant_device_ids: vec![device_id.clone()],
                        byte_capacity: allocation.byte_capacity,
                        concern,
                        usage: if allocation.kind
                            == VulkanComponentBatchResidentAllocationKind::DemandPipelinePredicate
                        {
                            VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate
                        } else {
                            VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage
                        },
                        allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
                    });
                }
            }
            for (destination_signal_id, byte_capacity) in histories {
                if destination_signal_id.is_empty() || byte_capacity == 0 {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(
                            "parallel speculative feedback history requires an identity and positive capacity"
                                .to_string(),
                        ),
                    ));
                }
                device_allocations.push(VulkanRuntimeDeviceLocalTransientAllocation {
                    logical_device_id: device_id.clone(),
                    participant_device_ids: vec![device_id.clone()],
                    byte_capacity,
                    concern: format!(
                        "speculative decoder {decoder_id} resident feedback source history {destination_signal_id}",
                    ),
                    usage: VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage,
                    allocation_class: VulkanRuntimeStreamAllocationClass::Permanent,
                });
            }
        }
        Ok(Self {
            device_allocations,
            host_visible_allocations,
        })
    }

    fn from_decoders(
        decoders: &[VulkanResidentSpeculativeDecoderProcessor],
        lane_capacity: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if lane_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "parallel speculative feedback allocation plan has zero lane capacity"
                        .to_string(),
                ),
            ));
        }
        let mut decoder_allocations = Vec::new();
        for decoder in decoders {
            let VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) =
                &decoder.execution
            else {
                continue;
            };
            let execution_scope = VulkanComponentBatchExecutionScope::nodes(
                processor.state_ingestion_node_ids_by_component.clone(),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let allocation_plan = VulkanComponentBatchResidentAllocationPlan::for_single_device(
                &processor.device_slice.package_slice.placed_plan,
                &processor.device_slice.package_slice.prepared_plan,
                &processor.device_slice.package_slice.batch_kernels,
                lane_capacity,
                VulkanComponentBatchExecutionMode::CausalSequence,
                &execution_scope,
                &BTreeSet::new(),
                false,
                processor
                    .device_slice
                    .demand_residency_context
                    .as_ref()
                    .map(|context| VulkanComponentBatchDemandResidencyPlanContext {
                        schedule: processor
                            .device_slice
                            .package_slice
                            .physical_residency_schedule(),
                        contract: &context.contract,
                        layout: &context.layout,
                    }),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let histories = processor
                .batch_source_taps
                .iter()
                .map(|binding| {
                    binding
                        .frame_byte_capacity
                        .checked_mul(lane_capacity)
                        .map(|byte_capacity| {
                            (binding.destination_signal_id.clone(), byte_capacity)
                        })
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "parallel speculative feedback history capacity overflowed"
                                    .to_string(),
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            decoder_allocations.push((
                decoder.id.clone(),
                decoder.device_id.clone(),
                allocation_plan.allocations,
                histories,
            ));
        }
        Self::from_decoder_allocations(decoder_allocations)
    }

    fn logical_bytes_by_device(
        &self,
    ) -> Result<BTreeMap<String, usize>, VulkanResidentInProcessPlacedRuntimeError> {
        let mut totals = BTreeMap::<String, usize>::new();
        for (logical_device_id, byte_capacity) in self
            .device_allocations
            .iter()
            .map(|allocation| (&allocation.logical_device_id, allocation.byte_capacity))
            .chain(
                self.host_visible_allocations
                    .iter()
                    .map(|allocation| (&allocation.logical_device_id, allocation.byte_capacity)),
            )
        {
            let total = totals
                .entry(logical_device_id.clone())
                .or_default();
            *total = total.checked_add(byte_capacity).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "parallel speculative feedback allocation bytes overflowed".to_string(),
                ))
            })?;
        }
        Ok(totals)
    }
}

fn validate_parallel_speculative_feedback_allocation_totals(
    expected: &BTreeMap<String, usize>,
    actual: &BTreeMap<String, usize>,
) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
    if actual == expected {
        return Ok(());
    }
    Err(VulkanResidentInProcessPlacedRuntimeError::Package(
        VulkanResidentTokenModelPackageError::new(format!(
            "parallel speculative feedback mounted allocation totals {actual:?} do not match admitted totals {expected:?}",
        )),
    ))
}

fn reserve_parallel_speculative_feedback_state_memory<'a, F>(
    decoders: &[VulkanResidentSpeculativeDecoderProcessor],
    lane_capacity: usize,
    device_for: &F,
) -> Result<Option<Arc<VulkanMemoryAdmission>>, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    let allocation_plan =
        VulkanParallelSpeculativeFeedbackAllocationPlan::from_decoders(decoders, lane_capacity)?;
    if allocation_plan.device_allocations.is_empty()
        && allocation_plan.host_visible_allocations.is_empty()
    {
        return Ok(None);
    }
    let mut requirement_bytes_by_device = BTreeMap::<String, usize>::new();
    for allocation in &allocation_plan.device_allocations {
        let device = device_for(&allocation.logical_device_id)?;
        let required = match allocation.usage {
            VulkanRuntimeDeviceLocalTransientAllocationUsage::Storage => device
                .resident_buffer_memory_requirement_bytes(allocation.byte_capacity),
            VulkanRuntimeDeviceLocalTransientAllocationUsage::ConditionalPredicate => device
                .conditional_resident_buffer_memory_requirement_bytes(allocation.byte_capacity),
            VulkanRuntimeDeviceLocalTransientAllocationUsage::ExternalSharedStorage => {
                let peers = allocation
                    .participant_device_ids
                    .iter()
                    .map(|device_id| device_for(device_id))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .filter(|candidate| !candidate.shares_logical_device_with(device))
                    .collect::<Vec<_>>();
                device.shared_device_resident_buffer_memory_requirement_bytes(
                    &peers,
                    allocation.byte_capacity,
                )
            }
        }
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let total = requirement_bytes_by_device
            .entry(allocation.logical_device_id.clone())
            .or_default();
        *total = total.checked_add(required).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "parallel speculative feedback memory requirement overflowed".to_string(),
            ))
        })?;
    }
    let mut host_requirement_bytes = 0usize;
    let mut host_representative = None;
    for allocation in &allocation_plan.host_visible_allocations {
        let device = device_for(&allocation.logical_device_id)?;
        let requirement = device
            .host_visible_resident_buffer_memory_requirement(allocation.byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        match requirement.domain {
            VulkanResidentBufferMemoryDomain::DeviceLocal => {
                let total = requirement_bytes_by_device
                    .entry(allocation.logical_device_id.clone())
                    .or_default();
                *total = total.checked_add(requirement.byte_count).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "parallel speculative feedback device-local host-visible requirement overflowed"
                            .to_string(),
                    ))
                })?;
            }
            VulkanResidentBufferMemoryDomain::HostVisible => {
                host_requirement_bytes = host_requirement_bytes
                    .checked_add(requirement.byte_count)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "parallel speculative feedback host requirement overflowed"
                                .to_string(),
                        ))
                    })?;
                host_representative.get_or_insert(device);
            }
        }
    }
    let requirements = requirement_bytes_by_device
        .iter()
        .map(|(device_id, byte_count)| {
            device_for(device_id).map(|device| (device, *byte_count))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let host_requirement = host_representative
        .map(|device| {
            vulkan_safe_host_available_bytes()
                .map(|available| (device, available, host_requirement_bytes))
                .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)
        })
        .transpose()?;
    VulkanMemoryAdmission::reserve(&requirements, host_requirement)
        .map(Arc::new)
        .map(Some)
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed atomic parallel speculative feedback reservation: {error}",
                )),
            )
        })
}

fn parallel_speculative_feedback_state_is_replayable<'a, F>(
    decoders: &[VulkanResidentSpeculativeDecoderProcessor],
    device_for: &F,
) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError>
where
    F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
{
    for decoder in decoders {
        let VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) =
            &decoder.execution
        else {
            continue;
        };
        let decoder_device = device_for(&decoder.device_id)?;
        for binding in &processor.batch_source_taps {
            let source_device = device_for(&binding.source_device_id)?;
            if !source_device.shares_logical_device_with(decoder_device) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

impl VulkanResidentParallelSpeculativeFeedbackState {
    fn new_if_needed<'a, F>(
        decoders: &[VulkanResidentSpeculativeDecoderProcessor],
        lane_capacity: usize,
        device_for: &F,
    ) -> Result<Option<Self>, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        if !decoders.iter().any(|decoder| decoder.is_parallel_block()) {
            return Ok(None);
        }
        if lane_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "parallel speculative feedback history has zero lane capacity".to_string(),
                ),
            ));
        }
        if !parallel_speculative_feedback_state_is_replayable(decoders, device_for)? {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "parallel speculative feedback history requires device-local target source taps"
                        .to_string(),
                ),
            ));
        }

        let mut decoder_states = Vec::new();
        for decoder in decoders {
            let VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) =
                &decoder.execution
            else {
                continue;
            };
            let device = device_for(&decoder.device_id)?;
            let execution_graph =
                VulkanResidentParallelSpeculativeStateIngestion::mount_execution_graph(
                    device,
                    decoder,
                    processor,
                    lane_capacity,
                    "resident-feedback-state-ingestion",
                )?;

            let mut source_histories = Vec::with_capacity(processor.batch_source_taps.len());
            for binding in &processor.batch_source_taps {
                let byte_capacity = binding
                    .frame_byte_capacity
                    .checked_mul(lane_capacity)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "parallel speculative feedback history capacity overflowed"
                                .to_string(),
                        ))
                    })?;
                source_histories.push(
                    device
                        .create_resident_buffer(byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
            }

            let lane_source_copies = (0..lane_capacity)
                .map(|lane| {
                    let ranges = processor
                        .batch_source_taps
                        .iter()
                        .zip(&source_histories)
                        .map(|(binding, history)| {
                            let destination_offset = lane
                                .checked_mul(binding.frame_byte_capacity)
                                .ok_or_else(|| {
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                        VulkanError(
                                            "parallel speculative feedback frame offset overflowed"
                                                .to_string(),
                                        ),
                                    )
                                })?;
                            VulkanResidentBufferRangeCopy::new(
                                &binding.source_scalar_buffer,
                                history,
                                0,
                                destination_offset,
                                binding.frame_byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    device
                        .create_resident_buffer_copy_batch(&ranges)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut state_source_taps = Vec::with_capacity(processor.batch_source_taps.len());
            for (binding, history) in processor
                .batch_source_taps
                .iter()
                .zip(&source_histories)
            {
                let destination = execution_graph.slice(0)?.signal_buffer(
                    &VulkanComponentBatchSignalKey::ModelInput(
                        binding.destination_signal_id.clone(),
                    ),
                )?;
                if destination.frame_byte_capacity != binding.frame_byte_capacity {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "parallel speculative feedback source {:?} has {}-byte frames but state ingestion expects {}",
                            binding.destination_signal_id,
                            binding.frame_byte_capacity,
                            destination.frame_byte_capacity,
                        )),
                    ));
                }
                let byte_capacity = binding
                    .frame_byte_capacity
                    .checked_mul(lane_capacity)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "parallel speculative feedback transfer capacity overflowed"
                                .to_string(),
                        ))
                    })?;
                state_source_taps.push(VulkanSpeculativeSourceTapTransfer::new(
                    device,
                    device,
                    history,
                    &destination.buffer,
                    byte_capacity,
                )?);
            }
            decoder_states.push(VulkanResidentParallelSpeculativeFeedbackDecoderState {
                state_ingestion: VulkanResidentParallelSpeculativeStateIngestion {
                    decoder_id: decoder.id.clone(),
                    device_id: decoder.device_id.clone(),
                    execution_graph,
                    source_taps: state_source_taps,
                },
                lane_source_copies,
                _source_histories: source_histories,
            });
        }
        Ok(Some(Self {
            decoders: decoder_states,
        }))
    }

    fn logical_allocation_bytes_by_device(
        &self,
    ) -> Result<BTreeMap<String, usize>, VulkanResidentInProcessPlacedRuntimeError> {
        let mut totals = BTreeMap::<String, usize>::new();
        for state in &self.decoders {
            let runner_bytes = state
                .state_ingestion
                .execution_graph
                .resident_transient_bytes_by_device()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
                .get(&state.state_ingestion.device_id)
                .copied()
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "parallel speculative feedback state ingestion {:?} has no allocation total for {:?}",
                        state.state_ingestion.decoder_id, state.state_ingestion.device_id,
                    )))
                })?;
            let history_bytes = state
                ._source_histories
                .iter()
                .try_fold(0usize, |total, history| {
                    total.checked_add(history.byte_capacity()).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "parallel speculative feedback mounted history bytes overflowed"
                                .to_string(),
                        ))
                    })
                })?;
            let total = totals
                .entry(state.state_ingestion.device_id.clone())
                .or_default();
            *total = total
                .checked_add(runner_bytes)
                .and_then(|bytes| bytes.checked_add(history_bytes))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "parallel speculative feedback mounted allocation bytes overflowed"
                            .to_string(),
                    ))
                })?;
        }
        Ok(totals)
    }

    fn enqueue_source_tap_capture<'a>(
        &'a self,
        devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
        lane: usize,
        completes_window: bool,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        for state in &self.decoders {
            let copy = state.lane_source_copies.get(lane).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "parallel speculative feedback lane {lane} exceeds history capacity {}",
                    state.lane_source_copies.len(),
                )))
            })?;
            let device = devices.get(&state.state_ingestion.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: state.state_ingestion.device_id.clone(),
                }
            })?;
            submission_batch
                .enqueue_resident_buffer_copy_batch(
                    device,
                    copy,
                    &[],
                    &[],
                    completes_window,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn wait_source_tap_capture(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        planned_tick_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if planned_tick_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "parallel speculative feedback cannot wait for an empty target window"
                        .to_string(),
                ),
            ));
        }
        for state in &self.decoders {
            let capture = state
                .lane_source_copies
                .get(planned_tick_count - 1)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        format!(
                            "parallel speculative feedback planned {planned_tick_count} source frames with history capacity {}",
                            state.lane_source_copies.len(),
                        ),
                    ))
                })?;
            let device = devices.get(&state.state_ingestion.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: state.state_ingestion.device_id.clone(),
                }
            })?;
            device
                .wait_resident_buffer_copy_batch(capture)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn run_state_ingestion(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        decoders: &[VulkanResidentSpeculativeDecoderProcessor],
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        for state in &self.decoders {
            let decoder = decoders
                .iter()
                .find(|decoder| decoder.id == state.state_ingestion.decoder_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "parallel speculative feedback state references absent decoder {:?}",
                        state.state_ingestion.decoder_id,
                    )))
                })?;
            let VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) =
                &decoder.execution
            else {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "parallel speculative feedback state references non-parallel decoder {:?}",
                        decoder.id,
                    )),
                ));
            };
            let device = devices.get(&state.state_ingestion.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: state.state_ingestion.device_id.clone(),
                }
            })?;
            state.state_ingestion.run(
                device,
                processor,
                input_token_ids,
                start_stream_tick,
            )?;
        }
        Ok(())
    }
}
