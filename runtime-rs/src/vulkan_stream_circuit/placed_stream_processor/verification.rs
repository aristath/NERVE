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
        if self.speculative_decoders.is_empty() || input_token_ids.is_empty() {
            return Ok(());
        }
        let history = self
            .resident_feedback_loop
            .as_ref()
            .and_then(|feedback_loop| {
                feedback_loop
                    .speculative_target_frame_history
                    .as_ref()
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "resident feedback did not mount speculative target-frame history".to_string(),
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
        for decoder in &self.speculative_decoders {
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
        let scalar_window_is_sufficient = self
            .scalar_verification_execution
            .borrow()
            .as_ref()
            .is_some_and(|runner| runner.lane_capacity >= batch_width);
        if transactions_are_sufficient && scalar_window_is_sufficient {
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
        self.ensure_scalar_verification_window(devices, batch_width)
    }

    fn capture_verification_baseline(
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
                .capture_baseline(&slice.mounted.buffers)
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
