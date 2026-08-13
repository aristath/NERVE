impl VulkanResidentInProcessPlacedStreamProcessor {
    pub fn model_package(&self) -> &VulkanResidentInProcessPlacedModelPackage {
        &self.model
    }

    pub fn speculative_decoder_count(&self) -> usize {
        self.speculative_decoders.len()
    }

    fn synchronize_speculative_decoders_after_target_tick(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_id: u32,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _draft = runtime_critical_path_span(RuntimeCriticalPathPhase::SpeculativeDraft);
        if self.speculative_decoders.is_empty() {
            return Ok(());
        }
        for decoder in &self.speculative_decoders {
            let draft_device = devices.get(&decoder.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: decoder.device_id.clone(),
                }
            })?;
            decoder.run_state_step(
                draft_device.as_ref(),
                input_token_id,
                stream_tick,
            )?;
            decoder.commit_target_hidden()?;
        }
        Ok(())
    }

    fn synchronize_speculative_decoders_after_target_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        planned_tick_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _draft = runtime_critical_path_span(RuntimeCriticalPathPhase::SpeculativeDraft);
        if self.speculative_decoders.is_empty() || input_token_ids.is_empty() {
            return Ok(());
        }
        let requirements = resident_speculative_feedback_history_requirements(
            self.speculative_decoders
                .iter()
                .map(VulkanResidentSpeculativeDecoderProcessor::is_parallel_block),
        );
        if requirements.parallel_state {
            let state = self
                .parallel_speculative_feedback_state
                .as_ref()
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "processor did not mount parallel speculative feedback state"
                            .to_string(),
                    ))
                })?;
            // The output timeline proves that target sampling completed, but
            // parallel speculative state also depends on source-tap history
            // copies submitted after that signal. Wait for the terminal
            // capture on each participating decoder device before copying the
            // history into its state-ingestion graph. Queue order alone is not
            // a cross-device or host-observable completion contract.
            state.wait_source_tap_capture(devices, planned_tick_count)?;
            state.run_state_ingestion(
                devices,
                &self.speculative_decoders,
                input_token_ids,
                start_stream_tick,
            )?;
        }
        if !requirements.normalized_frames {
            return Ok(());
        }
        let history = self
            .speculative_target_frame_history
            .as_ref()
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "processor did not mount speculative target-frame history".to_string(),
                ))
            })?;
        if planned_tick_count == 0 || planned_tick_count > history.lane_copies.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback planned {planned_tick_count} target frames with history capacity {}",
                    history.lane_copies.len()
                )),
            ));
        }
        if input_token_ids.len() > planned_tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "resident feedback executed {} target inputs from a {planned_tick_count}-tick window",
                    input_token_ids.len()
                )),
            ));
        }
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        output_device
            .wait_resident_buffer_copy_batch(&history.lane_copies[planned_tick_count - 1])
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.catch_up_speculative_decoders_from_target_frames(
            devices,
            input_token_ids,
            start_stream_tick,
            &history.frames,
            history.frame_byte_capacity,
        )
    }

    fn catch_up_speculative_decoders_from_target_frames(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let _stream_memory_scope = self.stream_memory_admission.enter();
        for decoder in self
            .speculative_decoders
            .iter()
            .filter(|decoder| !decoder.is_parallel_block())
        {
            let draft_device = devices.get(&decoder.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: decoder.device_id.clone(),
                }
            })?;
            decoder.run_catch_up_window(
                draft_device,
                input_token_ids,
                start_stream_tick,
                normalized_target_frames,
                frame_byte_capacity,
            )?;
        }
        Ok(())
    }

    fn ensure_verification_state_transactions(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions_are_sufficient = self
            .verification_state_transactions
            .borrow()
            .as_ref()
            .is_some_and(|transactions| {
                transactions
                    .iter()
                    .all(|transaction| transaction.cycle_width >= 1)
            });
        let causal_window_is_sufficient = self
            .temporal_block_executions
            .borrow()
            .get(&true)
            .is_some_and(|runner| {
                self.causal_block_lane_capacity(batch_width)
                    .is_ok_and(|required| required <= runner.execution_graph.lane_capacity)
            });
        if transactions_are_sufficient && causal_window_is_sufficient {
            return Ok(());
        }
        if !transactions_are_sufficient {
            let transactions = create_placed_state_transactions(
                &self.device_slices,
                1,
                &|device_id| {
                    devices
                        .get(device_id)
                        .map(|device| device.as_ref())
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: device_id.to_string(),
                            }
                        })
                },
            )?;
            *self.verification_state_transactions.borrow_mut() = Some(transactions);
        }
        self.ensure_temporal_block_execution(devices, batch_width, true)
    }

    fn run_causal_verification_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<Vec<VulkanResidentSampledToken>, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        self.run_batched_causal_verification_window(
            devices,
            input_token_ids,
            start_stream_tick,
        )
    }

    fn run_batched_causal_verification_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<Vec<VulkanResidentSampledToken>, VulkanResidentInProcessPlacedRuntimeError> {
        self.run_causal_component_block(
            devices,
            input_token_ids,
            start_stream_tick,
            true,
        )?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "causal verification context capacity exceeds u32".to_string(),
                ))
            })?;
        let stream_ticks = (0..input_token_ids.len())
            .map(|lane| {
                start_stream_tick
                    .checked_add(u64::try_from(lane).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capacities = vec![capacity; input_token_ids.len()];
        let token_prefixes = (0..input_token_ids.len())
            .map(|lane| &input_token_ids[..=lane])
            .collect::<Vec<_>>();
        let runner_guard = self.temporal_block_executions.borrow();
        let runner = runner_guard.get(&true).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "causal verification execution is not mounted".to_string(),
            ))
        })?;
        let target_output = runner.speculative_target_output.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "causal verification output is not mounted".to_string(),
            ))
        })?;
        target_output.project_and_sample_lanes(
            output_device,
            &token_prefixes,
            &stream_ticks,
            &capacities,
        )?;
        let batch_tokens = stream_ticks
            .iter()
            .map(|stream_tick| {
                self.sampler
                    .completed_run_at(*stream_tick)
                    .map(|run| VulkanResidentSampledToken::from(&run))
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(batch_tokens)
    }

    fn commit_causal_verification_prefix(
        &self,
        target_tick_count: usize,
        committed_tick_count: usize,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        let required_capacity = self.causal_block_lane_capacity(target_tick_count)?;
        let executions = self.temporal_block_executions.borrow();
        let runner = executions
            .get(&true)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative causal target window was not initialized".to_string(),
                ))
            })?;
        if required_capacity > runner.execution_graph.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative causal target width {required_capacity} exceeds canonical capacity {}",
                    runner.execution_graph.lane_capacity,
                )),
            ));
        }
        runner
            .execution_graph
            .commit_causal_state_prefix(committed_tick_count)
    }

    fn publish_causal_verification_source_taps(
        &self,
        target_tick_count: usize,
        committed_tick_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if committed_tick_count == 0 || committed_tick_count > target_tick_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative verification cannot publish {committed_tick_count} committed target frames from a {target_tick_count}-frame window"
                )),
            ));
        }
        let required_capacity = self.causal_block_lane_capacity(target_tick_count)?;
        let executions = self.temporal_block_executions.borrow();
        let runner = executions
            .get(&true)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative causal target window was not initialized".to_string(),
                ))
            })?;
        if required_capacity > runner.execution_graph.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative causal target width {required_capacity} exceeds canonical capacity {}",
                    runner.execution_graph.lane_capacity,
                )),
            ));
        }
        runner.publish_speculative_source_tap_frame(committed_tick_count - 1)
    }

    fn catch_up_speculative_decoder_after_verification(
        &self,
        decoder: &VulkanResidentSpeculativeDecoderProcessor,
        draft_device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        target_tick_count: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let required_capacity = self.causal_block_lane_capacity(target_tick_count)?;
        let causal_verification = self.temporal_block_executions.borrow();
        let runner = causal_verification
            .get(&true)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative causal target window was not initialized".to_string(),
                ))
        })?;
        if required_capacity > runner.execution_graph.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative causal target width {required_capacity} exceeds canonical capacity {}",
                    runner.execution_graph.lane_capacity,
                )),
            ));
        }
        if decoder.is_parallel_block() {
            return runner.run_parallel_speculative_state_ingestion(
                decoder,
                draft_device,
                input_token_ids,
                start_stream_tick,
            );
        }
        let normalized_target_frames = &runner
            .speculative_target_output
            .as_ref()
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative causal target output was not initialized".to_string(),
                ))
            })?
            .projection
            .norm
            .normalized_frames_buffer;
        let _stream_memory_scope = self.stream_memory_admission.enter();
        decoder.run_catch_up_window(
            draft_device,
            input_token_ids,
            start_stream_tick,
            normalized_target_frames,
            self.model
                .output_transducer_spec
                .normalized_frame_byte_capacity,
        )
    }

    fn commit_speculative_feedback_control(
        &self,
        token_id: u32,
        next_stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let output_slice = self
            .device_slices
            .iter()
            .find(|slice| slice.device_id == self.model.output_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.output_device_id.clone(),
                }
            })?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative feedback context capacity exceeds u32".to_string(),
                ))
            })?;
        output_slice
            .mounted
            .stream_control_buffer
            .write_bytes(&stream_control_bytes(
                token_id,
                VulkanMountedPlacedStreamControl {
                    stream_tick: next_stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations: capacity,
                },
            ))
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn submit_verification_baseline_capture(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions = self.verification_state_transactions.borrow();
        let transactions = transactions.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "verification state transaction is not mounted".to_string(),
            ))
        })?;
        for (transaction, slice) in transactions.iter().zip(&self.device_slices) {
            let device = devices.get(&slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: slice.device_id.clone(),
                }
            })?;
            transaction
                .submit_baseline_capture(device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

    fn restore_verification_baseline(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let transactions = self.verification_state_transactions.borrow();
        let transactions = transactions.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "verification state transaction is not mounted".to_string(),
            ))
        })?;
        for (transaction, slice) in transactions.iter().zip(&self.device_slices) {
            transaction
                .restore_baseline(&slice.mounted.buffers)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        Ok(())
    }

}
