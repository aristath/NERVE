impl VulkanResidentSpeculativeDecoderProcessor {
    #[allow(clippy::too_many_arguments)]
    fn from_model<'a, F>(
        device: &VulkanComputeDevice,
        model: &VulkanResidentSpeculativeDecoderModelPackage,
        planned_host_allocations: &[&VulkanRuntimeSharedHostResidentAllocation],
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
                        planned_host_allocations,
                        target_model.normal_prefill_lane_capacity,
                        target_model.speculative_draft_tokens,
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
                        planned_host_allocations,
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

    fn initialize_transient_state_buffers(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let initialized = self
            .mounted()
            .buffers
            .initialize_state_buffers(device)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let cloned = self
            .mounted()
            .buffers
            .apply_clone_state_policies()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        initialized.checked_add(cloned).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder state initialization byte count overflowed"
                    .to_string(),
            ))
        })
    }

    fn reset_transient_state_buffers(
        &self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        let state_bytes = self
            .mounted()
            .buffers
            .zero_state_buffers()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let telemetry_bytes = self
            .mounted()
            .buffers
            .zero_selection_telemetry_buffers()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let auxiliary_bytes = match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor
                    .sampler
                    .reset_token_state()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
                processor.reset_auxiliary_state()?
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => 0,
        };
        state_bytes
            .checked_add(telemetry_bytes)
            .and_then(|bytes| bytes.checked_add(auxiliary_bytes))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative decoder reset byte count overflowed".to_string(),
                ))
            })
    }

    fn reset_session_state(
        &self,
        random_seed: u32,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor
                    .sampler
                    .reset_session_state(random_seed)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
                processor.reset_auxiliary_state()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => Ok(0),
        }
    }

    fn restore_initial_session_state(
        &self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor
                    .sampler
                    .reset_token_state()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
                processor.reset_auxiliary_state()
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => Ok(0),
        }
    }

    fn set_random_seed(
        &self,
        random_seed: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor
                    .sampler
                    .set_random_seed(random_seed)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => Ok(()),
        }
    }

    fn maximum_draft_token_count(&self, requested: usize) -> usize {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(_) => requested,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.block_width
            }
        }
    }

    fn effective_draft_token_count(&self, requested: usize) -> usize {
        requested.min(self.maximum_draft_token_count(requested))
    }

    fn minimum_draft_token_count(&self) -> usize {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(_) => 1,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(processor) => {
                processor.minimum_draft_token_count
            }
        }
    }

    fn is_parallel_block(&self) -> bool {
        matches!(
            self.execution,
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_)
        )
    }

    fn invalidate_catch_up_source_binding(&self, source: &VulkanResidentBuffer) -> bool {
        match &self.execution {
            VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) => {
                processor.invalidate_catch_up_source_binding(source)
            }
            VulkanResidentSpeculativeDecoderExecutionProcessor::ParallelBlock(_) => false,
        }
    }

    fn discard_catch_up_batch(&self) {
        if let VulkanResidentSpeculativeDecoderExecutionProcessor::Autoregressive(processor) =
            &self.execution
        {
            processor.discard_catch_up_batch();
        }
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
