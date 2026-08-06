struct VulkanResidentParallelSpeculativeFeedbackDecoderState {
    state_ingestion: VulkanResidentParallelSpeculativeStateIngestion,
    lane_source_copies: Vec<VulkanResidentBufferCopyBatch>,
    _source_histories: Vec<VulkanResidentBuffer>,
}

struct VulkanResidentParallelSpeculativeFeedbackState {
    decoders: Vec<VulkanResidentParallelSpeculativeFeedbackDecoderState>,
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
