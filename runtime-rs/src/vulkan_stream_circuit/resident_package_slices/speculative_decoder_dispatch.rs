impl VulkanResidentSpeculativeDecoderProcessor {
    #[allow(clippy::too_many_arguments)]
    fn from_model<'a, F>(
        device: &VulkanComputeDevice,
        model: &VulkanResidentSpeculativeDecoderModelPackage,
        target_model: &VulkanResidentInProcessPlacedModelPackage,
        target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        target_hidden: &VulkanResidentBuffer,
        target_output_parameters: &VulkanPermanentParameterBuffers,
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
        random_seed: u32,
        device_for: &F,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        let execution = match &model.execution {
            VulkanResidentSpeculativeDecoderModelExecution::Autoregressive { .. } => {
                VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(
                    VulkanResidentAutoregressiveSpeculativeDecoderProcessor::from_model(
                        device,
                        model,
                        target_hidden,
                        target_output_parameters,
                        sampler_kernels,
                        sampler_spec,
                        random_seed,
                    )?,
                )
            }
            VulkanResidentSpeculativeDecoderModelExecution::ParallelBlock { .. } => {
                VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(
                    VulkanResidentParallelBlockSpeculativeDecoderProcessor::from_model(
                        device,
                        model,
                        target_model,
                        target_slices,
                        device_for,
                    )?,
                )
            }
        };
        Ok(Self {
            id: model.id.clone(),
            device_id: model.device_id.clone(),
            execution,
        })
    }

    fn mounted(&self) -> &VulkanMountedPlacedStreamCircuit {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.mounted()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.mounted()
            }
        }
    }

    fn effective_draft_token_count(&self, requested: usize) -> usize {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(_) => requested,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                requested.min(processor.block_width)
            }
        }
    }

    fn is_parallel_block(&self) -> bool {
        matches!(
            self.execution,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_)
        )
    }

    fn capture_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.capture_baseline()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.capture_baseline()
            }
        }
    }

    fn restore_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.restore_baseline()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.restore_baseline()
            }
        }
    }

    fn run_draft_window(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
        confidence_threshold: f32,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.run_draft_window(
                    device,
                    initial_token_id,
                    start_stream_tick,
                    draft_token_count,
                )
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.run_draft_window(
                    device,
                    initial_token_id,
                    start_stream_tick,
                    draft_token_count,
                    confidence_threshold,
                )
            }
        }
    }

    fn run_state_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.run_state_step(device, input_token_id, stream_tick)
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.run_state_step(device, input_token_id, stream_tick)
            }
        }
    }

    fn run_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.run_catch_up_window(
                    device,
                    input_token_ids,
                    start_stream_tick,
                    normalized_target_frames,
                    frame_byte_capacity,
                )
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => Err(
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "parallel speculative state catch-up requires retained source-tap frames"
                        .to_string(),
                )),
            ),
        }
    }

    fn commit_target_hidden(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.commit_target_hidden()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => Ok(()),
        }
    }
}
