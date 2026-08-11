fn mount_speculative_decoder_device_slice(
    device: &VulkanComputeDevice,
    model: &VulkanResidentSpeculativeDecoderModelPackage,
) -> Result<
    VulkanResidentInProcessPlacedStreamProcessorDevice,
    VulkanResidentInProcessPlacedRuntimeError,
> {
    let mounted = model
        .device_slice
        .create_mounted_stream_circuit(device)
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
    mounted
        .buffers
        .initialize_state_buffers(device)
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to initialize speculative decoder {:?} state: {error}",
                    model.id
                )),
            )
        })?;
    mounted
        .buffers
        .apply_clone_state_policies()
        .map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to initialize speculative decoder {:?} cloned state: {error}",
                    model.id
                )),
            )
        })?;
    let reusable_manifest = resident_package_reusable_kernel_manifest(&mounted.placed_plan);
    let mounted_bound = mounted
        .mounted_placed_bound_dispatch_plan(&reusable_manifest)
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BoundDispatchPlan)?;
    let tick_plan = VulkanMountedPlacedStreamTickPlan::from_mounted_bound_plan(&mounted_bound);
    let execution_plan = VulkanMountedPlacedResidentStreamTickExecutionPlan::
        from_tick_plan_with_physical_execution_islands_and_demand(
            device,
            &mounted,
            &mounted_bound,
            model.device_slice.loaded_manifest(),
            tick_plan,
            &[],
            model
                .demand_residency_context
                .as_ref()
                .map(|_| model.device_slice.physical_residency_schedule()),
            model.demand_residency_context.as_ref(),
            None,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
    if execution_plan.distributed_dispatch_count != 0
        || execution_plan.tick_plan.receive_stage_count != 0
        || execution_plan.tick_plan.publish_stage_count != 0
    {
        return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
            VulkanResidentTokenModelPackageError::new(format!(
                "speculative decoder {:?} did not compile to one device-resident circuit",
                model.id
            )),
        ));
    }
    Ok(VulkanResidentInProcessPlacedStreamProcessorDevice {
        device_id: model.device_id.clone(),
        hosted_component_count: model.device_slice.hosted_component_count,
        incoming_edge_count: model.device_slice.incoming_edge_count,
        outgoing_edge_count: model.device_slice.outgoing_edge_count,
        dispatch_count: mounted_bound.dispatches.len(),
        package_slice: Arc::clone(&model.device_slice),
        mounted,
        mounted_bound,
        resident_execution_plan: execution_plan,
        demand_residency_context: model.demand_residency_context.clone(),
    })
}

impl VulkanResidentAutoregressiveSpeculativeDecoderProcessor {
    fn mounted(&self) -> &VulkanMountedPlacedStreamCircuit {
        &self.device_slice.mounted
    }

    fn reset_auxiliary_state(
        &self,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        self.active_pending_target_hidden.set(0);
        self.pending_target_hiddens.iter().try_fold(
            0usize,
            |total, buffer| {
                buffer
                    .write_bytes(&vec![0; buffer.byte_capacity()])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
                total.checked_add(buffer.byte_capacity()).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge(VulkanError(
                        "speculative decoder auxiliary reset byte count overflowed"
                            .to_string(),
                    ))
                })
            },
        )
    }

    fn execution_plan(&self) -> &VulkanMountedPlacedResidentStreamTickExecutionPlan {
        &self.device_slice.resident_execution_plan
    }

    fn active_pending_target_hidden_index(&self) -> usize {
        self.active_pending_target_hidden.get()
    }

    fn pending_hidden_input_copy(&self) -> &VulkanResidentBufferCopy {
        &self.pending_hidden_input_copies[self.active_pending_target_hidden_index()]
    }

    fn pending_target_hidden(&self) -> &VulkanResidentBuffer {
        &self.pending_target_hiddens[self.active_pending_target_hidden_index()]
    }

    #[allow(clippy::too_many_arguments)]
    fn from_model(
        device: &VulkanComputeDevice,
        model: &VulkanResidentSpeculativeDecoderModelPackage,
        target_hidden: &VulkanResidentBuffer,
        target_output_parameters: &VulkanPermanentParameterBuffers,
        sampler_kernels: &[VulkanResidentSamplerKernelArtifact],
        sampler_spec: &VulkanResidentSamplerSpec,
        random_seed: u32,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let VulkanResidentSpeculativeDecoderModelExecution::Autoregressive {
            input_embedding_spec,
            input_embedding_spirv_words,
            input_embedding_batch_spirv_words,
            input_embedding_batch_control,
            output_norm_spirv_words,
            output_projection_spirv_words,
            ..
        } = &model.execution
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel-block speculative decoder {:?} cannot mount as autoregressive",
                    model.id
                )),
            ));
        };
        let device_slice = mount_speculative_decoder_device_slice(device, model)?;
        let mounted = &device_slice.mounted;

        let adapter = model.package.dedicated_input_adapter().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "autoregressive speculative decoder {:?} has no input adapter",
                    model.id
                )),
            )
        })?;
        let hidden_input = mounted
            .boundary_io
            .input_buffer(&adapter.target_hidden_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "speculative decoder {:?} has no hidden input {:?}",
                        model.id, adapter.target_hidden_signal_id
                    )),
                )
            })?;
        let input_embedding_weight = model
            .parameter(
                target_output_parameters,
                &input_embedding_spec.parameter_tensor,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::InputTransducer(
                    VulkanResidentInputEmbeddingTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: input_embedding_spec.parameter_tensor.clone(),
                    },
                )
            })?;
        let input_embedding_weight_allocation = VulkanPermanentParameterBufferAllocation {
            parameter: input_embedding_weight.parameter.clone(),
            byte_capacity: input_embedding_weight.byte_capacity,
            byte_offset: input_embedding_weight.byte_offset,
            buffer: Arc::clone(&input_embedding_weight.buffer),
        };
        let input_transducer =
            VulkanResidentInputEmbeddingTransducerRunner::from_mounted_token_embedding_with_parameter_allocation(
                device,
                &mounted,
                input_embedding_weight,
                input_embedding_spirv_words,
                input_embedding_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let output_spec = model
            .output_transducer_spec(
                model
                    .package
                    .dedicated_output_transducer()
                    .expect("validated autoregressive decoder has output I/O")
                    .input_signal_id
                    .clone(),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Package)?;
        let norm_weight = model
            .parameter(target_output_parameters, &output_spec.norm_parameter_tensor)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: output_spec.norm_parameter_tensor.clone(),
                    },
                )
            })?;
        let projection_weight = model
            .parameter(
                target_output_parameters,
                &output_spec.projection_parameter_tensor,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                    VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                        tensor: output_spec.projection_parameter_tensor.clone(),
                    },
                )
            })?;
        let projection_scale = match output_spec.projection_scale_parameter_tensor.as_ref() {
            Some(tensor) => Some(model.parameter(target_output_parameters, tensor).ok_or_else(
                || {
                    VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                        VulkanResidentOutputTransducerRunnerError::MissingProjectionScaleParameterBuffer {
                            tensor: tensor.clone(),
                        },
                    )
                },
            )?),
            None => None,
        };
        let output_transducer =
            VulkanResidentOutputTransducerRunner::from_mounted_output_transducer_with_parameter_allocations(
                device,
                &mounted,
                norm_weight,
                projection_weight,
                projection_scale,
                output_norm_spirv_words,
                output_projection_spirv_words,
                &output_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
        let sampler = VulkanResidentSamplerRunner::from_output_transducer_with_spec(
            device,
            &mounted,
            &output_transducer,
            sampler_kernels,
            sampler_spec,
            random_seed,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;
        let pending_target_hiddens = [
            device
                .create_resident_buffer(adapter.target_hidden_byte_capacity)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
            device
                .create_resident_buffer(adapter.target_hidden_byte_capacity)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
        ];
        for pending_target_hidden in &pending_target_hiddens {
            pending_target_hidden
                .write_bytes(&vec![0u8; adapter.target_hidden_byte_capacity])
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        }
        let pending_hidden_input_copies = [
            device
                .create_resident_buffer_copy(
                    &pending_target_hiddens[0],
                    &hidden_input.buffer,
                    adapter.target_hidden_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
            device
                .create_resident_buffer_copy(
                    &pending_target_hiddens[1],
                    &hidden_input.buffer,
                    adapter.target_hidden_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
        ];
        let update_pending_hidden_copies = [
            device
                .create_resident_buffer_copy(
                    target_hidden,
                    &pending_target_hiddens[0],
                    adapter.target_hidden_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
            device
                .create_resident_buffer_copy(
                    target_hidden,
                    &pending_target_hiddens[1],
                    adapter.target_hidden_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?,
        ];
        let state_transaction =
            VulkanResidentStateTransactionBank::new_transactional(device, &mounted.buffers, 1)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let draft_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let state_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let catch_up_sequence = device
            .create_resident_kernel_sequence()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let catch_up_control_byte_capacity = VULKAN_BACKEND_LOOP_MAX_WINDOW
            .checked_mul(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative catch-up control capacity overflowed".to_string(),
                ))
            })?;
        let mut catch_up_controls = device
            .create_host_visible_resident_buffer(catch_up_control_byte_capacity)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        catch_up_controls
            .persistently_map()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let catch_up_controls_initial_copy = device
            .create_resident_buffer_copy(
                &catch_up_controls,
                &mounted.stream_control_buffer,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        Ok(Self {
            id: model.id.clone(),
            device_id: model.device_id.clone(),
            device_slice,
            input_transducer,
            input_embedding_batch_spirv_words: input_embedding_batch_spirv_words.clone(),
            input_embedding_batch_control: *input_embedding_batch_control,
            input_embedding_spec: input_embedding_spec.clone(),
            input_embedding_weight: input_embedding_weight_allocation,
            output_transducer,
            sampler,
            draft_sequence,
            state_sequence,
            catch_up_sequence,
            hidden_input_signal_id: adapter.target_hidden_signal_id.clone(),
            pending_hidden_input_copies,
            update_pending_hidden_copies,
            pending_target_hiddens,
            active_pending_target_hidden: Cell::new(0),
            catch_up_batches: RefCell::new(BTreeMap::new()),
            catch_up_controls,
            catch_up_controls_initial_copy,
            state_transaction,
        })
    }

    fn capture_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.state_transaction
            .capture_baseline(&self.mounted().buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.sampler
            .capture_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn restore_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.state_transaction
            .restore_baseline(&self.mounted().buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.sampler
            .restore_token_state()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn run_draft_window(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        if draft_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        if self.execution_plan().uses_demand_residency() {
            return self.run_demand_draft_window(
                device,
                initial_token_id,
                start_stream_tick,
                draft_token_count,
            );
        }
        self.input_transducer
            .prepare_token_id_only(initial_token_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let dynamic_state_capacity_activations = u32::try_from(
            self.mounted().buffers.dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder dynamic state capacity exceeds u32".to_string(),
            ))
        })?;
        self.mounted()
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(VulkanMountedPlacedStreamControl {
                    stream_tick: start_stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                }),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        let decoder_dispatches = self
            .execution_plan()
            .dispatch_segments
            .iter()
            .flat_map(|segment| segment.dispatches.iter())
            .collect::<Vec<_>>();
        let controls = (0..draft_token_count)
            .map(|draft_index| {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(draft_index).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                    })?)
                    .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                let control = VulkanMountedPlacedStreamControl {
                    stream_tick,
                    control_flags: 0,
                    dynamic_state_capacity_activations,
                };
                decoder_dispatches
                    .iter()
                    .map(|dispatch| {
                        stream_control_push_constant_bytes(&dispatch.push_constants, control)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let steps_per_tick = 1usize
            .checked_add(self.sampler.input_tracking_dispatches().len())
            .and_then(|count| count.checked_add(decoder_dispatches.len()))
            .and_then(|count| count.checked_add(2))
            .and_then(|count| count.checked_add(self.sampler.resident_dispatches().len()))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative draft dispatch count overflowed".to_string(),
                ))
            })?;
        let mut steps = Vec::with_capacity(
            steps_per_tick
                .checked_mul(draft_token_count)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative draft window dispatch count overflowed".to_string(),
                    ))
                })?,
        );
        let hidden_input = self
            .mounted()
            .boundary_io
            .input_buffer(&self.hidden_input_signal_id)
            .expect("validated speculative hidden input must remain mounted");
        let mut hidden_feedback_copies = Vec::with_capacity(draft_token_count);
        for control in &controls {
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.input_transducer.resident_dispatch,
                &[],
            ));
            steps.extend(
                self.sampler
                    .input_tracking_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
            steps.extend(
                decoder_dispatches
                    .iter()
                    .zip(control)
                    .map(|(dispatch, push_constants)| {
                        VulkanResidentKernelSequenceStep::new(
                            &dispatch.resident_dispatch,
                            push_constants,
                        )
                    }),
            );
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.embedding_norm_dispatch,
                &[],
            ));
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.tied_projection_dispatch,
                &[],
            ));
            steps.extend(
                self.sampler
                    .resident_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
            hidden_feedback_copies.push(
                VulkanResidentKernelSequenceSnapshotCopy::new(
                    steps.len() - 1,
                    self.output_transducer.normalized_frame_buffer(),
                    &hidden_input.buffer,
                    0,
                    0,
                    hidden_input.byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
        }
        device
            .run_resident_kernel_sequence_with_input_and_snapshot_copies(
                &self.draft_sequence,
                &[VulkanResidentKernelSequenceInputCopy::new(
                    self.pending_hidden_input_copy(),
                )],
                &steps,
                &hidden_feedback_copies,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        (0..draft_token_count)
            .map(|draft_index| {
                let stream_tick = start_stream_tick
                    .checked_add(u64::try_from(draft_index).map_err(|_| {
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

    fn run_state_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.run_composed_step_with_input_copies(
            device,
            &self.state_sequence,
            input_token_id,
            stream_tick,
            &[VulkanResidentKernelSequenceInputCopy::new(
                self.pending_hidden_input_copy(),
            )],
            false,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_composed_step_with_input_copies(
        &self,
        device: &VulkanComputeDevice,
        sequence: &VulkanResidentKernelSequence,
        input_token_id: u32,
        stream_tick: u64,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        include_output: bool,
    ) -> Result<Option<VulkanResidentSamplerRun>, VulkanResidentInProcessPlacedRuntimeError> {
        if self.execution_plan().uses_demand_residency() {
            if input_copies.len() != 1 {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "demand-resident speculative step requires exactly one hidden-input copy, got {}",
                        input_copies.len()
                    )),
                ));
            }
            return self.run_demand_composed_step(
                device,
                input_token_id,
                stream_tick,
                include_output,
            );
        }
        self.input_transducer
            .prepare_token_id_only(input_token_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let dynamic_state_capacity_activations = u32::try_from(
            self.mounted().buffers.dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder dynamic state capacity exceeds u32".to_string(),
            ))
        })?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };
        self.mounted()
            .stream_control_buffer
            .write_bytes_at(
                VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                &stream_control_metadata_bytes(control),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let decoder_dispatches = self
            .execution_plan()
            .dispatch_segments
            .iter()
            .flat_map(|segment| segment.dispatches.iter())
            .collect::<Vec<_>>();
        let decoder_push_constants = decoder_dispatches
            .iter()
            .map(|dispatch| stream_control_push_constant_bytes(&dispatch.push_constants, control))
            .collect::<Result<Vec<_>, _>>()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        let output_dispatch_count = if include_output {
            2usize
                .checked_add(self.sampler.resident_dispatches().len())
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative decoder composed dispatch count overflowed".to_string(),
                    ))
                })?
        } else {
            0
        };
        let mut steps = Vec::with_capacity(
            1usize
                .checked_add(self.sampler.input_tracking_dispatches().len())
                .and_then(|count| count.checked_add(decoder_dispatches.len()))
                .and_then(|count| count.checked_add(output_dispatch_count))
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative decoder composed dispatch count overflowed".to_string(),
                    ))
                })?,
        );
        steps.push(VulkanResidentKernelSequenceStep::new(
            &self.input_transducer.resident_dispatch,
            &[],
        ));
        steps.extend(
            self.sampler
                .input_tracking_dispatches()
                .iter()
                .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
        );
        steps.extend(decoder_dispatches.iter().zip(&decoder_push_constants).map(
            |(dispatch, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&dispatch.resident_dispatch, push_constants)
            },
        ));
        if include_output {
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.embedding_norm_dispatch,
                &[],
            ));
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.output_transducer.tied_projection_dispatch,
                &[],
            ));
            steps.extend(
                self.sampler
                    .resident_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
        }
        device
            .run_resident_kernel_sequence_with_input_copies(sequence, input_copies, &steps)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        include_output
            .then(|| self.sampler.completed_run())
            .transpose()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn run_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        if input_token_ids.len() > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up width {} exceeds resident window {}",
                    input_token_ids.len(),
                    VULKAN_BACKEND_LOOP_MAX_WINDOW,
                )),
            ));
        }
        if self.execution_plan().uses_demand_residency() || input_token_ids.len() == 1 {
            return self.run_serial_catch_up_window(
                device,
                input_token_ids,
                start_stream_tick,
                normalized_target_frames,
                frame_byte_capacity,
            );
        }
        self.run_batched_catch_up_window(
            device,
            input_token_ids,
            start_stream_tick,
            normalized_target_frames,
            frame_byte_capacity,
        )
    }

    fn run_serial_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        if input_token_ids.len() > VULKAN_BACKEND_LOOP_MAX_WINDOW {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up width {} exceeds resident window {}",
                    input_token_ids.len(),
                    VULKAN_BACKEND_LOOP_MAX_WINDOW,
                )),
            ));
        }
        if self.execution_plan().uses_demand_residency() {
            return self.run_demand_catch_up_window(
                device,
                input_token_ids,
                start_stream_tick,
                normalized_target_frames,
                frame_byte_capacity,
            );
        }
        let dynamic_state_capacity_activations = u32::try_from(
            self.mounted().buffers.dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder dynamic state capacity exceeds u32".to_string(),
            ))
        })?;
        let decoder_dispatches = self
            .execution_plan()
            .dispatch_segments
            .iter()
            .flat_map(|segment| segment.dispatches.iter())
            .collect::<Vec<_>>();
        let mut controls = Vec::with_capacity(input_token_ids.len());
        for (tick_index, input_token_id) in input_token_ids.iter().copied().enumerate() {
            let stream_tick = start_stream_tick
                .checked_add(u64::try_from(tick_index).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let control = VulkanMountedPlacedStreamControl {
                stream_tick,
                control_flags: 0,
                dynamic_state_capacity_activations,
            };
            let control_offset = tick_index
                .checked_mul(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative catch-up control offset overflowed".to_string(),
                    ))
                })?;
            self.catch_up_controls
                .write_bytes_at(
                    control_offset,
                    &stream_control_bytes(input_token_id, control),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            controls.push(
                decoder_dispatches
                    .iter()
                    .map(|dispatch| {
                        stream_control_push_constant_bytes(&dispatch.push_constants, control)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }

        let hidden_input = self
            .mounted()
            .boundary_io
            .input_buffer(&self.hidden_input_signal_id)
            .expect("validated speculative hidden input must remain mounted");
        if hidden_input.byte_capacity != frame_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up hidden frame has {} bytes, expected {frame_byte_capacity}",
                    hidden_input.byte_capacity
                )),
            ));
        }
        let steps_per_tick = 1usize
            .checked_add(self.sampler.input_tracking_dispatches().len())
            .and_then(|count| count.checked_add(decoder_dispatches.len()))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "speculative catch-up dispatch count overflowed".to_string(),
                ))
            })?;
        let mut steps = Vec::with_capacity(
            steps_per_tick
                .checked_mul(input_token_ids.len())
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative catch-up window dispatch count overflowed".to_string(),
                    ))
                })?,
        );
        let mut intermediate_copies = Vec::with_capacity(
            input_token_ids
                .len()
                .saturating_sub(1)
                .saturating_mul(2)
                .saturating_add(1),
        );
        for (tick_index, control) in controls.iter().enumerate() {
            steps.push(VulkanResidentKernelSequenceStep::new(
                &self.input_transducer.resident_dispatch,
                &[],
            ));
            steps.extend(
                self.sampler
                    .input_tracking_dispatches()
                    .iter()
                    .map(|dispatch| VulkanResidentKernelSequenceStep::new(dispatch, &[])),
            );
            steps.extend(
                decoder_dispatches
                    .iter()
                    .zip(control)
                    .map(|(dispatch, push_constants)| {
                        VulkanResidentKernelSequenceStep::new(
                            &dispatch.resident_dispatch,
                            push_constants,
                        )
                    }),
            );
            let after_step_index = steps.len() - 1;
            let target_hidden_offset = tick_index
                .checked_mul(frame_byte_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative catch-up hidden offset overflowed".to_string(),
                    ))
                })?;
            if tick_index + 1 < input_token_ids.len() {
                let next_control_offset = (tick_index + 1)
                    .checked_mul(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "speculative catch-up control offset overflowed".to_string(),
                        ))
                    })?;
                intermediate_copies.push(
                    VulkanResidentKernelSequenceSnapshotCopy::new(
                        after_step_index,
                        &self.catch_up_controls,
                        &self.mounted().stream_control_buffer,
                        next_control_offset,
                        0,
                        VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
                intermediate_copies.push(
                    VulkanResidentKernelSequenceSnapshotCopy::new(
                        after_step_index,
                        normalized_target_frames,
                        &hidden_input.buffer,
                        target_hidden_offset,
                        0,
                        frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
            } else {
                intermediate_copies.push(
                    VulkanResidentKernelSequenceSnapshotCopy::new(
                        after_step_index,
                        normalized_target_frames,
                        self.pending_target_hidden(),
                        target_hidden_offset,
                        0,
                        frame_byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
            }
        }
        device
            .run_resident_kernel_sequence_with_input_and_snapshot_copies(
                &self.catch_up_sequence,
                &[
                    VulkanResidentKernelSequenceInputCopy::new(
                        &self.catch_up_controls_initial_copy,
                    ),
                    VulkanResidentKernelSequenceInputCopy::new(
                        self.pending_hidden_input_copy(),
                    ),
                ],
                &steps,
                &intermediate_copies,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn commit_target_hidden(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let copy = &self.update_pending_hidden_copies[self.active_pending_target_hidden_index()];
        copy.run(copy.byte_len())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)
    }

    fn run_demand_draft_window(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        self.input_transducer
            .prepare_token_id_only(initial_token_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let dynamic_state_capacity_activations = self.speculative_dynamic_state_capacity()?;
        let hidden_input = self
            .mounted()
            .boundary_io
            .input_buffer(&self.hidden_input_signal_id)
            .expect("validated speculative hidden input must remain mounted");
        let feedback_copy = VulkanResidentBufferRangeCopy::new(
            self.output_transducer.normalized_frame_buffer(),
            &hidden_input.buffer,
            0,
            0,
            hidden_input.byte_capacity,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::FeedbackEdge)?;
        let mut tokens = Vec::with_capacity(draft_token_count);
        for draft_index in 0..draft_token_count {
            let stream_tick = start_stream_tick
                .checked_add(u64::try_from(draft_index).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let control = VulkanMountedPlacedStreamControl {
                stream_tick,
                control_flags: 0,
                dynamic_state_capacity_activations,
            };
            if draft_index == 0 {
                self.run_demand_decoder_tick(
                    device,
                    control,
                    true,
                    &[VulkanResidentKernelSequenceInputCopy::new(
                        self.pending_hidden_input_copy(),
                    )],
                    &[feedback_copy],
                )?;
            } else {
                self.run_demand_decoder_tick(
                    device,
                    control,
                    true,
                    &[],
                    &[feedback_copy],
                )?;
            }
            tokens.push(
                self.sampler
                    .completed_run_at(stream_tick)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?
                    .token_id,
            );
        }
        Ok(tokens)
    }

    fn run_demand_composed_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
        include_output: bool,
    ) -> Result<Option<VulkanResidentSamplerRun>, VulkanResidentInProcessPlacedRuntimeError> {
        self.input_transducer
            .prepare_token_id_only(input_token_id)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::InputTransducer)?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations: self.speculative_dynamic_state_capacity()?,
        };
        self.run_demand_decoder_tick(
            device,
            control,
            include_output,
            &[VulkanResidentKernelSequenceInputCopy::new(
                self.pending_hidden_input_copy(),
            )],
            &[],
        )?;
        include_output
            .then(|| self.sampler.completed_run())
            .transpose()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)
    }

    fn run_demand_catch_up_window(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        normalized_target_frames: &VulkanResidentBuffer,
        frame_byte_capacity: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let hidden_input = self
            .mounted()
            .boundary_io
            .input_buffer(&self.hidden_input_signal_id)
            .expect("validated speculative hidden input must remain mounted");
        if hidden_input.byte_capacity != frame_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "speculative catch-up hidden frame has {} bytes, expected {frame_byte_capacity}",
                    hidden_input.byte_capacity
                )),
            ));
        }
        let dynamic_state_capacity_activations = self.speculative_dynamic_state_capacity()?;
        for (tick_index, input_token_id) in input_token_ids.iter().copied().enumerate() {
            let stream_tick = start_stream_tick
                .checked_add(u64::try_from(tick_index).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                })?)
                .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
            let control = VulkanMountedPlacedStreamControl {
                stream_tick,
                control_flags: 0,
                dynamic_state_capacity_activations,
            };
            self.mounted()
                .stream_control_buffer
                .write_bytes(&stream_control_bytes(input_token_id, control))
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            let source_offset = tick_index
                .checked_mul(frame_byte_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "speculative catch-up hidden offset overflowed".to_string(),
                    ))
                })?;
            let destination = if tick_index + 1 < input_token_ids.len() {
                &hidden_input.buffer
            } else {
                self.pending_target_hidden()
            };
            let copy = VulkanResidentBufferRangeCopy::new(
                normalized_target_frames,
                destination,
                source_offset,
                0,
                frame_byte_capacity,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if tick_index == 0 {
                self.run_demand_decoder_tick(
                    device,
                    control,
                    false,
                    &[VulkanResidentKernelSequenceInputCopy::new(
                        self.pending_hidden_input_copy(),
                    )],
                    &[copy],
                )?;
            } else {
                self.run_demand_decoder_tick(
                    device,
                    control,
                    false,
                    &[],
                    &[copy],
                )?;
            }
        }
        Ok(())
    }

    fn run_demand_decoder_tick(
        &self,
        device: &VulkanComputeDevice,
        control: VulkanMountedPlacedStreamControl,
        include_output: bool,
        input_copies: &[VulkanResidentKernelSequenceInputCopy<'_>],
        post_copies: &[VulkanResidentBufferRangeCopy<'_>],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let mut prefix_dispatches =
            Vec::with_capacity(1 + self.sampler.input_tracking_dispatches().len());
        prefix_dispatches.push(&self.input_transducer.resident_dispatch);
        prefix_dispatches.extend(self.sampler.input_tracking_dispatches());
        let mut suffix_dispatches = Vec::new();
        if include_output {
            suffix_dispatches.push(&self.output_transducer.embedding_norm_dispatch);
            suffix_dispatches.push(&self.output_transducer.tied_projection_dispatch);
            suffix_dispatches.extend(self.sampler.resident_dispatches());
        }
        let sequence_variant = u8::from(include_output)
            | (u8::from(!input_copies.is_empty()) << 1)
            | (u8::from(!post_copies.is_empty()) << 2);
        self.execution_plan()
            .run_single_segment_demand_resident(
                device,
                control,
                &prefix_dispatches,
                &suffix_dispatches,
                sequence_variant,
                input_copies,
                post_copies,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)
    }

    fn speculative_dynamic_state_capacity(
        &self,
    ) -> Result<u32, VulkanResidentInProcessPlacedRuntimeError> {
        u32::try_from(self.mounted().buffers.dynamic_state_capacity_activations).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "speculative decoder dynamic state capacity exceeds u32".to_string(),
            ))
        })
    }
}
