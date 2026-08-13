struct VulkanResidentSpeculativeCatchUpBatch {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    source_binding: Option<VulkanResidentSpeculativeCatchUpSourceBinding>,
}

struct VulkanResidentSpeculativeCatchUpSourceBinding {
    identity: VulkanResidentSpeculativeCatchUpSourceIdentity,
    hidden_copy_batches: [Vec<VulkanResidentBufferCopyBatch>; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VulkanResidentSpeculativeCatchUpSourceIdentity {
    device_handle: u64,
    buffer_handle: u64,
    frame_byte_capacity: usize,
}

impl VulkanResidentSpeculativeCatchUpSourceIdentity {
    fn new(source: &VulkanResidentBuffer, frame_byte_capacity: usize) -> Self {
        let (device_handle, buffer_handle) = source.command_binding_identity();
        Self {
            device_handle,
            buffer_handle,
            frame_byte_capacity,
        }
    }

    fn binds_buffer(self, source: &VulkanResidentBuffer) -> bool {
        self.binds_command_identity(source.command_binding_identity())
    }

    fn binds_command_identity(self, (device_handle, buffer_handle): (u64, u64)) -> bool {
        self.device_handle == device_handle && self.buffer_handle == buffer_handle
    }
}

fn speculative_catch_up_lane_capacity(speculative_draft_tokens: usize) -> Result<usize, VulkanError> {
    let target_tick_count = speculative_draft_tokens.checked_add(1).ok_or_else(|| {
        VulkanError("speculative catch-up target width overflowed".to_string())
    })?;
    causal_component_block_lane_capacity(target_tick_count)
}

fn speculative_catch_up_execution_lane_capacity(
    normal_prefill_lane_capacity: usize,
    speculative_draft_tokens: usize,
) -> Result<usize, VulkanError> {
    Ok(causal_component_block_lane_capacity(normal_prefill_lane_capacity)?
        .max(speculative_catch_up_lane_capacity(speculative_draft_tokens)?))
}

fn speculative_catch_up_preceding_target_bytes(
    batch_width: usize,
    frame_byte_capacity: usize,
) -> Result<usize, VulkanError> {
    batch_width
        .checked_sub(1)
        .and_then(|frame_count| frame_count.checked_mul(frame_byte_capacity))
        .ok_or_else(|| VulkanError("speculative catch-up hidden range overflowed".to_string()))
}

impl VulkanResidentAutoregressiveSpeculativeDecoderProcessor {
    fn discard_catch_up_batch(&self) {
        self.catch_up_batch.borrow_mut().take();
    }

    fn run_batched_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.len() > self.catch_up_lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up batch capacity {} cannot execute {} target ticks",
                    self.catch_up_lane_capacity,
                    input_token_ids.len(),
                )),
            ));
        }
        if self.catch_up_batch.borrow().is_none() {
            let batch = self.create_catch_up_batch(device, self.catch_up_lane_capacity)?;
            *self.catch_up_batch.borrow_mut() = Some(batch);
        }
        let source_identity = VulkanResidentSpeculativeCatchUpSourceIdentity::new(
            normalized_target_frames,
            frame_byte_capacity,
        );
        let mut batch_guard = self.catch_up_batch.borrow_mut();
        let batch = batch_guard
            .as_mut()
            .expect("speculative catch-up batch was inserted");
        if batch
            .source_binding
            .as_ref()
            .is_none_or(|binding| binding.identity != source_identity)
        {
            batch.source_binding = Some(self.create_catch_up_source_binding(
                device,
                batch,
                normalized_target_frames,
                source_identity,
            )?);
        }
        let source_binding = batch
            .source_binding
            .as_ref()
            .expect("speculative catch-up source binding was inserted");
        let active_pending_index = self.active_pending_target_hidden_index();
        device
            .submit_resident_buffer_copy_batch(
                &source_binding.hidden_copy_batches[active_pending_index]
                    [input_token_ids.len() - 1],
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        batch
            .input_embedding
            .submit_deferred(device, input_token_ids)?;
        self.sampler
            .submit_input_tokens_deferred(device, input_token_ids)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let dynamic_state_capacity_activations = self.speculative_dynamic_state_capacity()?;
        batch.execution_graph.run_causal_sequence_single_device(
            device,
            &self.device_id,
            self.mounted(),
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
        )?;
        self.active_pending_target_hidden.set(active_pending_index ^ 1);
        Ok(())
    }

    fn create_catch_up_batch(
        &self,
        device: &VulkanComputeDevice,
        lane_capacity: usize,
    ) -> Result<VulkanResidentSpeculativeCatchUpBatch, VulkanResidentInProcessPlacedRuntimeError>
    {
        let execution_graph = VulkanResidentPlacedComponentBatchRunner::new_single_device(
            device,
            &self.device_slice,
            &format!("draft:{}", self.id),
            lane_capacity,
            VulkanComponentBatchExecutionMode::CausalSequence,
        )?;
        let token_embedding_signal = execution_graph.slice(0)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                self.input_embedding_spec.output_signal_id.clone(),
            ),
        )?;
        let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
            device,
            lane_capacity,
            &self.input_embedding_weight,
            &token_embedding_signal.buffer,
            &self.input_embedding_batch_spirv_words,
            self.input_embedding_batch_control,
            &self.input_embedding_spec,
        )?;
        Ok(VulkanResidentSpeculativeCatchUpBatch {
            execution_graph,
            input_embedding,
            source_binding: None,
        })
    }

    fn create_catch_up_source_binding(
        &self,
        device: &VulkanComputeDevice,
        batch: &VulkanResidentSpeculativeCatchUpBatch,
        normalized_target_frames: &VulkanResidentBuffer,
        identity: VulkanResidentSpeculativeCatchUpSourceIdentity,
    ) -> Result<VulkanResidentSpeculativeCatchUpSourceBinding, VulkanResidentInProcessPlacedRuntimeError>
    {
        let hidden_signal = batch.execution_graph.slice(0)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                self.hidden_input_signal_id.clone(),
            ),
        )?;
        if hidden_signal.frame_byte_capacity != identity.frame_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up batch hidden frame has {} bytes, expected {}",
                    hidden_signal.frame_byte_capacity,
                    identity.frame_byte_capacity,
                )),
            ));
        }
        let build_hidden_copy_batches = |active_pending_index: usize| {
            (1..=batch.execution_graph.lane_capacity)
                .map(|batch_width| {
                    let inactive_pending_index = active_pending_index ^ 1;
                    let preceding_target_bytes = speculative_catch_up_preceding_target_bytes(
                        batch_width,
                        identity.frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let mut copies = vec![VulkanResidentBufferRangeCopy::new(
                        &self.pending_target_hiddens[active_pending_index],
                        &hidden_signal.buffer,
                        0,
                        0,
                        identity.frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?];
                    if preceding_target_bytes > 0 {
                        copies.push(
                            VulkanResidentBufferRangeCopy::new(
                                normalized_target_frames,
                                &hidden_signal.buffer,
                                0,
                                identity.frame_byte_capacity,
                                preceding_target_bytes,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                    }
                    copies.push(
                        VulkanResidentBufferRangeCopy::new(
                            normalized_target_frames,
                            &self.pending_target_hiddens[inactive_pending_index],
                            preceding_target_bytes,
                            0,
                            identity.frame_byte_capacity,
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    );
                    device
                        .create_resident_buffer_copy_batch(&copies)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let hidden_copy_batches = [
            build_hidden_copy_batches(0)?,
            build_hidden_copy_batches(1)?,
        ];
        Ok(VulkanResidentSpeculativeCatchUpSourceBinding {
            identity,
            hidden_copy_batches,
        })
    }

    fn invalidate_catch_up_source_binding(&self, source: &VulkanResidentBuffer) -> bool {
        let mut batch = self.catch_up_batch.borrow_mut();
        let Some(batch) = batch.as_mut() else {
            return false;
        };
        let should_invalidate = batch
            .source_binding
            .as_ref()
            .is_some_and(|binding| binding.identity.binds_buffer(source));
        if should_invalidate {
            batch.source_binding = None;
        }
        should_invalidate
    }
}
