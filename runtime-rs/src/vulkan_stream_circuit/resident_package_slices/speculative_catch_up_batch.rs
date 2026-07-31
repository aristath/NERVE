struct VulkanResidentSpeculativeCatchUpBatch {
    execution_graph: VulkanResidentPlacedComponentBatchRunner,
    input_embedding: VulkanResidentBatchedInputEmbeddingRunner,
    hidden_copy_batches: [Vec<VulkanResidentBufferCopyBatch>; 2],
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

impl VulkanResidentSpeculativeDecoderProcessor {
    fn run_batched_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let lane_capacity = causal_component_block_lane_capacity(input_token_ids.len())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let source_identity = normalized_target_frames as *const VulkanResidentBuffer as usize;
        let key = (lane_capacity, source_identity, frame_byte_capacity);
        if !self.catch_up_batches.borrow().contains_key(&key) {
            let batch = self.create_catch_up_batch(
                device,
                lane_capacity,
                normalized_target_frames,
                frame_byte_capacity,
            )?;
            self.catch_up_batches.borrow_mut().insert(key, batch);
        }

        let batches = self.catch_up_batches.borrow();
        let batch = batches
            .get(&key)
            .expect("speculative catch-up batch was inserted");
        let active_pending_index = self.active_pending_target_hidden_index();
        device
            .submit_resident_buffer_copy_batch(
                &batch.hidden_copy_batches[active_pending_index][input_token_ids.len() - 1],
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
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
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
        let hidden_signal = execution_graph.slice(0)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                self.hidden_input_signal_id.clone(),
            ),
        )?;
        if hidden_signal.frame_byte_capacity != frame_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up batch hidden frame has {} bytes, expected {frame_byte_capacity}",
                    hidden_signal.frame_byte_capacity,
                )),
            ));
        }
        let build_hidden_copy_batches = |active_pending_index: usize| {
            (1..=lane_capacity)
                .map(|batch_width| {
                    let inactive_pending_index = active_pending_index ^ 1;
                    let preceding_target_bytes = speculative_catch_up_preceding_target_bytes(
                        batch_width,
                        frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let mut copies = vec![VulkanResidentBufferRangeCopy::new(
                        &self.pending_target_hiddens[active_pending_index],
                        &hidden_signal.buffer,
                        0,
                        0,
                        frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?];
                    if preceding_target_bytes > 0 {
                        copies.push(
                            VulkanResidentBufferRangeCopy::new(
                                normalized_target_frames,
                                &hidden_signal.buffer,
                                0,
                                frame_byte_capacity,
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
                            frame_byte_capacity,
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
        Ok(VulkanResidentSpeculativeCatchUpBatch {
            execution_graph,
            input_embedding,
            hidden_copy_batches,
        })
    }
}
