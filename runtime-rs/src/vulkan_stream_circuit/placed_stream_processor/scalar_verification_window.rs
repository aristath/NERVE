impl VulkanResidentInProcessPlacedStreamProcessor {
    fn ensure_scalar_verification_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        lane_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if self
            .scalar_verification_execution
            .borrow()
            .as_ref()
            .is_some_and(|runner| runner.lane_capacity >= lane_capacity)
        {
            return Ok(());
        }
        if lane_capacity == 0 || lane_capacity > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "scalar verification width {lane_capacity} exceeds resident window {}",
                    VULKAN_BACKEND_LOOP_MAX_WINDOW
                )),
            ));
        }
        let input_device_index = self
            .device_slices
            .iter()
            .position(|slice| slice.device_id == self.model.input_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.input_device_id.clone(),
                }
            })?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        let input_frame_byte_capacity = self
            .model
            .input_transducer_spec
            .output_frame_byte_capacity;
        let input_frames_byte_capacity = input_frame_byte_capacity
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "scalar verification input-frame capacity overflowed".to_string(),
                ))
            })?;
        let input_frames = input_device
            .create_resident_buffer(input_frames_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let embedding_weight = self
            .model
            .input_transducer_parameter_buffers
            .parameter_buffer(&self.model.input_transducer_spec.parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                    VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: self.model.input_transducer_spec.parameter_tensor.clone(),
                    },
                )
            })?;
        let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
            input_device,
            lane_capacity,
            embedding_weight,
            &input_frames,
            &self.model.input_transducer_batch_spirv_words,
            self.model.input_transducer_batch_control,
            &self.model.input_transducer_spec,
        )?;
        let scalar_input = self.device_slices[input_device_index]
            .mounted
            .boundary_io
            .input_buffer(&self.model.input_transducer_spec.output_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "scalar verification input device has no boundary {:?}",
                    self.model.input_transducer_spec.output_signal_id
                )))
            })?;
        let input_frame_copies = (0..lane_capacity)
            .map(|lane| {
                let source_offset = lane
                    .checked_mul(input_frame_byte_capacity)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "scalar verification input-frame offset overflowed".to_string(),
                        ))
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    &input_frames,
                    &scalar_input.buffer,
                    source_offset,
                    0,
                    input_frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                input_device
                    .create_resident_buffer_copy_batch(&[copy])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut stream_control_sources = Vec::with_capacity(self.device_slices.len());
        let mut stream_control_copies = Vec::with_capacity(self.device_slices.len());
        for slice in &self.device_slices {
            let device = devices.get(&slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: slice.device_id.clone(),
                }
            })?;
            let mut sources = Vec::with_capacity(lane_capacity);
            let mut copies = Vec::with_capacity(lane_capacity);
            for _ in 0..lane_capacity {
                let mut source = device
                    .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                source
                    .persistently_map()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                let copy = device
                    .create_resident_buffer_copy(
                        &source,
                        &slice.mounted.stream_control_buffer,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                sources.push(source);
                copies.push(copy);
            }
            stream_control_sources.push(sources);
            stream_control_copies.push(copies);
        }

        let frame_byte_capacity = self
            .model
            .output_transducer_spec
            .normalized_frame_byte_capacity;
        let target_frames_byte_capacity = frame_byte_capacity
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "scalar verification target-frame capacity overflowed".to_string(),
                ))
            })?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        let normalized_target_frames = output_device
            .create_resident_buffer(target_frames_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let normalized_target_frame_copies = (0..lane_capacity)
            .map(|lane| {
                let destination_offset =
                    lane.checked_mul(frame_byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "scalar verification target-frame offset overflowed".to_string(),
                        ))
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    self.output_transducer.normalized_frame_buffer(),
                    &normalized_target_frames,
                    0,
                    destination_offset,
                    frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                output_device
                    .create_resident_buffer_copy_batch(&[copy])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;

        *self.scalar_verification_execution.borrow_mut() =
            Some(VulkanResidentScalarVerificationWindowRunner {
                lane_capacity,
                _input_frames: input_frames,
                input_embedding,
                input_frame_copies,
                stream_control_sources,
                stream_control_copies,
                normalized_target_frames,
                normalized_target_frame_copies,
            });
        Ok(())
    }

    fn run_scalar_verification_window(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        self.ensure_scalar_verification_window(devices, input_token_ids.len())?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "scalar verification context capacity exceeds u32".to_string(),
                ))
            })?;
        let runner_guard = self.scalar_verification_execution.borrow();
        let runner = runner_guard.as_ref().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "scalar verification window is not mounted".to_string(),
            ))
        })?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        runner.input_embedding.run(input_device, input_token_ids)?;

        for (device_index, sources) in runner.stream_control_sources.iter().enumerate() {
            for (lane, source) in sources.iter().take(input_token_ids.len()).enumerate() {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(lane).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                source
                    .write_bytes(&stream_control_bytes(
                        input_token_ids[lane],
                        VulkanMountedPlacedStreamControl {
                            stream_tick,
                            control_flags: 0,
                            dynamic_state_capacity_activations: capacity,
                        },
                    ))
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            debug_assert_eq!(
                sources.len(),
                runner.stream_control_copies[device_index].len()
            );
        }

        let mut transport = VulkanInProcessPlacedEdgeTransport::new();
        let submission_batch = VulkanResidentQueueSubmissionBatch::new();
        for lane in 0..input_token_ids.len() {
            let stream_tick = start_stream_tick
                .checked_add(u64::try_from(lane).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            for (device_index, slice) in self.device_slices.iter().enumerate() {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                submission_batch
                    .enqueue_resident_buffer_copy(
                        device,
                        &runner.stream_control_copies[device_index][lane],
                        &[],
                        &[],
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            submission_batch
                .enqueue_resident_buffer_copy_batch(
                    input_device,
                    &runner.input_frame_copies[lane],
                    &[],
                    &[],
                    false,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

            let sequence_variant = u8::try_from(4usize.checked_add(lane).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "scalar verification sequence variant overflowed".to_string(),
                ))
            })?)
            .map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "scalar verification sequence variant exceeds u8".to_string(),
                ))
            })?;
            let mut slices = SmallVec::<
                [VulkanMountedPlacedResidentInProcessStreamTickSlice<'_>; 4],
            >::with_capacity(self.device_slices.len());
            for slice in &self.device_slices {
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                let mut extensions =
                    VulkanMountedPlacedResidentStreamTickDispatchExtensions::default()
                        .with_sequence_variant(sequence_variant);
                if slice.device_id == self.model.output_device_id {
                    extensions
                        .prefix_dispatches
                        .extend(self.sampler.input_tracking_dispatches());
                    extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.embedding_norm_dispatch);
                    extensions
                        .suffix_dispatches
                        .push(&self.output_transducer.tied_projection_dispatch);
                    extensions
                        .suffix_dispatches
                        .extend(self.sampler.resident_dispatches());
                }
                slices.push(
                    VulkanMountedPlacedResidentInProcessStreamTickSlice::new_with_dispatch_extensions(
                        device,
                        &slice.mounted,
                        &slice.resident_execution_plan,
                        extensions,
                        stream_tick,
                    ),
                );
            }
            let run = run_mounted_placed_resident_stream_tick_slices_in_process_with_schedule_and_distributed(
                &mut slices,
                &mut transport,
                &self.activation_schedule,
                Some(&self.distributed_dispatch_runners),
                Some(&self.edge_synchronizations),
                VulkanPlacedSubmissionContext {
                    policy: VulkanPlacedSubmissionPolicy {
                        write_stream_control: false,
                        signal_completion: false,
                        wait_for_completion: false,
                        feedback_lane: None,
                    },
                    participant_devices: Some(devices),
                    state_transactions: None,
                    feedback_turn: None,
                    output_turn: None,
                    submission_batch: Some(&submission_batch),
                },
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Tick)?;
            if run.status != VulkanMountedPlacedResidentInProcessStreamTickRunStatus::Completed {
                return Err(VulkanResidentInProcessPlacedRuntimeError::IncompleteTick(
                    run.status,
                ));
            }
            let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: self.model.output_device_id.clone(),
                }
            })?;
            submission_batch
                .enqueue_resident_buffer_copy_batch(
                    output_device,
                    &runner.normalized_target_frame_copies[lane],
                    &[],
                    &[],
                    lane + 1 == input_token_ids.len(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        let template = submission_batch
            .mount()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        template
            .submit_with_timeline_value_offset(0)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        output_device
            .wait_resident_buffer_copy_batch(
                &runner.normalized_target_frame_copies[input_token_ids.len() - 1],
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        (0..input_token_ids.len())
            .map(|lane| {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(lane).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                self.sampler
                    .completed_run_at(stream_tick)
                    .map(|run| run.token_id)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
            })
            .collect()
    }
}
