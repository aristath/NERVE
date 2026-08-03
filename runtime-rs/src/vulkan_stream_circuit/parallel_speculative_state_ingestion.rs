#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanParallelSpeculativeSourceTapBatchBinding {
    source_device_id: String,
    source_batch_signal_key: VulkanComponentBatchSignalKey,
    destination_signal_id: String,
    frame_byte_capacity: usize,
}

struct VulkanResidentParallelSpeculativeStateIngestion {
    decoder_id: String,
    device_id: String,
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    source_taps: Vec<VulkanSpeculativeSourceTapTransfer>,
}

impl VulkanResidentParallelSpeculativeStateIngestion {
    fn mount(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        target_execution_graph: &VulkanResidentPlacedComponentBatchRunner,
        decoder: &VulkanResidentSpeculativeDecoderProcessor,
        processor: &VulkanResidentParallelBlockSpeculativeDecoderProcessor,
        lane_capacity: usize,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(&decoder.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: decoder.device_id.clone(),
            }
        })?;
        let execution_graph =
            VulkanResidentPlacedComponentBatchRunner::new_single_device_for_nodes(
                device,
                &processor.device_slice,
                &format!("draft:{}:retained-state-ingestion", decoder.id),
                lane_capacity,
                VulkanComponentBatchExecutionMode::CausalSequence,
                processor.state_ingestion_node_ids_by_component.clone(),
            )?;
        let mut source_taps = Vec::with_capacity(processor.batch_source_taps.len());
        for binding in &processor.batch_source_taps {
            let source_device_index = target_slices
                .iter()
                .position(|slice| slice.device_id == binding.source_device_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: binding.source_device_id.clone(),
                    }
                })?;
            let source = target_execution_graph
                .slice(source_device_index)?
                .signal_buffer(&binding.source_batch_signal_key)?;
            let destination = execution_graph.slice(0)?.signal_buffer(
                &VulkanComponentBatchSignalKey::ModelInput(
                    binding.destination_signal_id.clone(),
                ),
            )?;
            if source.frame_byte_capacity != binding.frame_byte_capacity
                || destination.frame_byte_capacity != binding.frame_byte_capacity
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "parallel speculative decoder {:?} retained source tap {:?} has frame capacities source={}, destination={}, compiled={}",
                        decoder.id,
                        binding.destination_signal_id,
                        source.frame_byte_capacity,
                        destination.frame_byte_capacity,
                        binding.frame_byte_capacity,
                    )),
                ));
            }
            let byte_capacity = binding
                .frame_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "parallel speculative state-ingestion capacity overflowed".to_string(),
                    ))
                })?;
            if source.buffer.byte_capacity() < byte_capacity
                || destination.buffer.byte_capacity() < byte_capacity
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "parallel speculative decoder {:?} retained source tap {:?} cannot hold {lane_capacity} frames",
                        decoder.id, binding.destination_signal_id,
                    )),
                ));
            }
            let source_device = devices.get(&binding.source_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: binding.source_device_id.clone(),
                }
            })?;
            source_taps.push(VulkanSpeculativeSourceTapTransfer::new(
                source_device,
                device,
                &source.buffer,
                &destination.buffer,
                byte_capacity,
            )?);
        }
        Ok(Self {
            decoder_id: decoder.id.clone(),
            device_id: decoder.device_id.clone(),
            execution_graph,
            source_taps,
        })
    }

    fn run(
        &self,
        device: &VulkanComputeDevice,
        processor: &VulkanResidentParallelBlockSpeculativeDecoderProcessor,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        if input_token_ids.len() > self.execution_graph.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "parallel speculative state ingestion capacity {} cannot process {} tokens",
                    self.execution_graph.lane_capacity,
                    input_token_ids.len(),
                )),
            ));
        }
        for source_tap in &self.source_taps {
            source_tap.run()?;
        }
        let dynamic_state_capacity_activations = u32::try_from(
            processor
                .mounted()
                .buffers
                .dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "parallel speculative state capacity exceeds u32".to_string(),
            ))
        })?;
        self.execution_graph.run_causal_sequence_single_device(
            device,
            &self.device_id,
            processor.mounted(),
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
        )
    }
}

fn mount_parallel_speculative_state_ingestions(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    target_execution_graph: &VulkanResidentPlacedComponentBatchRunner,
    decoders: &[VulkanResidentSpeculativeDecoderProcessor],
    lane_capacity: usize,
) -> Result<Vec<VulkanResidentParallelSpeculativeStateIngestion>, VulkanResidentInProcessPlacedRuntimeError>
{
    decoders
        .iter()
        .filter_map(|decoder| match &decoder.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(_) => None,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                Some(VulkanResidentParallelSpeculativeStateIngestion::mount(
                    devices,
                    target_slices,
                    target_execution_graph,
                    decoder,
                    processor,
                    lane_capacity,
                ))
            }
        })
        .collect()
}

impl VulkanResidentPlacedTemporalBlockRunner {
    fn run_parallel_speculative_state_ingestion(
        &self,
        decoder: &VulkanResidentSpeculativeDecoderProcessor,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) =
            &decoder.execution
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative decoder {:?} has no parallel state-ingestion schedule",
                    decoder.id,
                )),
            ));
        };
        let ingestion = self
            .parallel_speculative_state_ingestions
            .iter()
            .find(|ingestion| ingestion.decoder_id == decoder.id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "speculative decoder {:?} retained state-ingestion schedule is not mounted",
                    decoder.id,
                )))
            })?;
        ingestion.run(device, processor, input_token_ids, start_stream_tick)
    }
}
