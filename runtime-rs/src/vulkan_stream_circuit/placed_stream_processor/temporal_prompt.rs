fn causal_component_block_lane_capacity(block_width: usize) -> Result<usize, VulkanError> {
    if block_width == 0 || block_width > VULKAN_BACKEND_LOOP_MAX_WINDOW {
        return Err(VulkanError(format!(
            "causal component block width {block_width} exceeds resident window {}",
            VULKAN_BACKEND_LOOP_MAX_WINDOW,
        )));
    }
    block_width.checked_next_power_of_two().ok_or_else(|| {
        VulkanError("causal component block capacity overflowed".to_string())
    })
}

fn speculative_source_tap_signal_keys_by_device(
    model: &VulkanResidentInProcessPlacedModelPackage,
    target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
) -> Result<
    BTreeMap<String, BTreeSet<VulkanComponentBatchSignalKey>>,
    VulkanResidentInProcessPlacedRuntimeError,
> {
    let mut retained = BTreeMap::<String, BTreeSet<VulkanComponentBatchSignalKey>>::new();
    for decoder in &model.speculative_decoders {
        for tap in decoder
            .package
            .circuit_graph
            .boundary
            .external_inputs
            .iter()
            .filter_map(|port| port.source_tap.as_ref())
        {
            let resolved =
                resolved_speculative_source_tap_buffer(model, target_slices, tap)?;
            retained
                .entry(resolved.device_id.to_string())
                .or_default()
                .insert(resolved.batch_signal_key);
        }
    }
    Ok(retained)
}

fn mount_speculative_source_tap_frame_copies(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    model: &VulkanResidentInProcessPlacedModelPackage,
    target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    execution_graph: &VulkanResidentPlacedComponentBatchRunner,
    block_width: usize,
) -> Result<Vec<Vec<VulkanResidentBufferCopyBatch>>, VulkanResidentInProcessPlacedRuntimeError> {
    let retained = speculative_source_tap_signal_keys_by_device(model, target_slices)?;
    if retained.is_empty() {
        return Ok(Vec::new());
    }
    (0..block_width)
        .map(|frame_index| {
            let mut ranges_by_device =
                BTreeMap::<String, Vec<VulkanResidentBufferRangeCopy<'_>>>::new();
            let mut mounted_keys = BTreeSet::new();
            for decoder in &model.speculative_decoders {
                for tap in decoder
                    .package
                    .circuit_graph
                    .boundary
                    .external_inputs
                    .iter()
                    .filter_map(|port| port.source_tap.as_ref())
                {
                    let resolved =
                        resolved_speculative_source_tap_buffer(model, target_slices, tap)?;
                    if !mounted_keys.insert((
                        resolved.device_id.to_string(),
                        resolved.batch_signal_key.clone(),
                    )) {
                        continue;
                    }
                    let device_index = target_slices
                        .iter()
                        .position(|slice| slice.device_id == resolved.device_id)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: resolved.device_id.to_string(),
                            }
                        })?;
                    let source = execution_graph
                        .slice(device_index)?
                        .signal_buffer(&resolved.batch_signal_key)?;
                    if source.frame_byte_capacity != resolved.frame_byte_capacity {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "speculative source-tap batch frame has {} bytes, expected {}",
                                source.frame_byte_capacity, resolved.frame_byte_capacity
                            )),
                        ));
                    }
                    let source_offset = frame_index
                        .checked_mul(source.frame_byte_capacity)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "speculative source-tap frame offset overflowed".to_string(),
                            ))
                        })?;
                    ranges_by_device
                        .entry(resolved.device_id.to_string())
                        .or_default()
                        .push(
                            VulkanResidentBufferRangeCopy::new(
                                &source.buffer,
                                resolved.scalar_buffer,
                                source_offset,
                                0,
                                resolved.frame_byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                }
            }
            ranges_by_device
                .into_iter()
                .map(|(device_id, ranges)| {
                    devices
                        .get(&device_id)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: device_id.clone(),
                            }
                        })?
                        .create_resident_buffer_copy_batch(&ranges)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

impl VulkanResidentInProcessPlacedStreamProcessor {
    fn linear_pipeline_device_indices(
        &self,
    ) -> Result<Vec<usize>, VulkanResidentInProcessPlacedRuntimeError> {
        let mut current = self
            .device_slices
            .iter()
            .position(|slice| slice.device_id == self.model.input_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "placed package {:?} has no input pipeline device {:?}",
                    self.model.package_id, self.model.input_device_id
                )))
            })?;
        let output = self
            .device_slices
            .iter()
            .position(|slice| slice.device_id == self.model.output_device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "placed package {:?} has no output pipeline device {:?}",
                    self.model.package_id, self.model.output_device_id
                )))
            })?;
        let mut ordered = Vec::with_capacity(self.device_slices.len());
        let mut visited = BTreeSet::new();
        loop {
            if !visited.insert(current) {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError("placed verification pipeline contains a device cycle".to_string()),
                ));
            }
            ordered.push(current);
            if current == output {
                break;
            }
            let outgoing = &self.device_slices[current]
                .mounted
                .edge_io
                .outgoing_buffers;
            let [outgoing] = outgoing.as_slice() else {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "placed verification pipeline device {:?} has {} outgoing activation edges; expected one",
                        self.device_slices[current].device_id,
                        outgoing.len()
                    )),
                ));
            };
            current = self
                .device_slices
                .iter()
                .position(|slice| {
                    slice.device_id == outgoing.endpoint.remote_device_id
                        && slice
                            .mounted
                            .edge_io
                            .incoming_buffer(outgoing.endpoint.edge_index)
                            .is_some()
                })
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "placed verification pipeline edge {} has no mounted destination {:?}",
                        outgoing.endpoint.edge_index, outgoing.endpoint.remote_device_id
                    )))
                })?;
        }
        if ordered.len() != self.device_slices.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "placed verification pipeline reaches {} of {} mounted devices",
                    ordered.len(),
                    self.device_slices.len()
                )),
            ));
        }
        Ok(ordered)
    }

    fn temporal_block_lane_capacity(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        const MINIMUM_DEVICE_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;
        const RECORDED_DISPATCH_BUDGET_PER_SUBMISSION: usize = 65_536;

        let mut width = VULKAN_BACKEND_LOOP_MAX_WINDOW;
        for slice in &self.device_slices {
            let device = devices.get(&slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: slice.device_id.clone(),
                }
            })?;
            let (_, signal_buffer_plan) =
                component_batch_signal_buffer_plan(&slice.mounted, &slice.mounted_bound.dispatches)?;
            let signal_bytes_per_lane =
                signal_buffer_plan
                    .iter()
                    .try_fold(0usize, |total, allocation| {
                        total
                            .checked_add(allocation.frame_byte_capacity)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "temporal signal byte count overflowed".to_string(),
                                ))
                            })
                    })?;
            if signal_bytes_per_lane > 0 {
                let available = device.available_device_local_memory_bytes();
                let headroom = (device.device_local_memory_bytes() / 32)
                    .max(MINIMUM_DEVICE_HEADROOM_BYTES)
                    .min(available / 2);
                let usable = available.saturating_sub(headroom);
                let memory_width = usize::try_from(usable)
                    .unwrap_or(usize::MAX)
                    .checked_div(signal_bytes_per_lane)
                    .unwrap_or_default();
                width = width.min(memory_width.max(1));
            }

            for artifact in slice.package_slice.batch_kernels.iter().filter(|artifact| {
                artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
            }) {
                width = width.min(artifact.lane_tile_width);
                for stage in &artifact.stages {
                    let dispatch_width = u64::from(device.max_compute_work_group_count_x())
                        .saturating_mul(
                            u64::try_from(artifact.lane_tile_width).unwrap_or(u64::MAX),
                        )
                        .checked_div(u64::from(stage.workgroup_count_x.max(1)))
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(usize::MAX);
                    width = width.min(dispatch_width.max(1));
                }
            }

            let mut scalar_dispatches_per_lane_by_component = BTreeMap::<&str, usize>::new();
            for dispatch in &slice.mounted_bound.dispatches {
                if !slice.package_slice.batch_kernels.iter().any(|artifact| {
                    artifact.component_id == dispatch.component_id && artifact.node_id == dispatch.node_id
                }) {
                    *scalar_dispatches_per_lane_by_component
                        .entry(&dispatch.component_id)
                        .or_default() += 1;
                }
            }
            let scalar_dispatches_per_lane = scalar_dispatches_per_lane_by_component
                .values()
                .copied()
                .max()
                .unwrap_or_default();
            if let Some(dispatch_width) =
                RECORDED_DISPATCH_BUDGET_PER_SUBMISSION.checked_div(scalar_dispatches_per_lane)
            {
                width = width.min(dispatch_width.max(1));
            }
        }
        Ok(width.max(1))
    }

    fn supports_contiguous_device_batch_pipeline(&self) -> bool {
        self.linear_pipeline_device_indices().is_ok()
    }

    fn temporal_block_width(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        available_token_count: usize,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        if available_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        // The causal batch runner currently owns one slice per device. A valid
        // component graph may revisit a device (for example gpu0 -> gpu1 ->
        // gpu0), but batching those two gpu0 regions as one slice would execute
        // the tail before the remote middle. Keep that graph on the ordered
        // scalar tick path until the batch plan can represent device segments.
        if !self.supports_contiguous_device_batch_pipeline() {
            return Ok(1);
        }
        Ok(available_token_count.min(self.temporal_block_lane_capacity(devices)?))
    }

    fn causal_block_lane_capacity(
        &self,
        block_width: usize,
    ) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
        causal_component_block_lane_capacity(block_width)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn ensure_temporal_block_execution(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        block_width: usize,
        capture_causal_state_snapshots: bool,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let block_width = self.causal_block_lane_capacity(block_width)?;
        let execution_key = (block_width, capture_causal_state_snapshots);
        if self
            .temporal_block_executions
            .borrow()
            .contains_key(&execution_key)
        {
            return Ok(());
        }
        let pipeline = self.linear_pipeline_device_indices()?;
        let first_device_index = *pipeline.first().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal pipeline is empty".to_string(),
            ))
        })?;
        let retained_signal_keys =
            speculative_source_tap_signal_keys_by_device(&self.model, &self.device_slices)?;
        let execution_graph = VulkanResidentPlacedComponentBatchRunner::new(
            devices,
            &self.device_slices,
            &self.model.runtime_execution_identity,
            &self.execution_quantum_calibrators,
            block_width,
            VulkanComponentBatchExecutionMode::CausalSequence,
            &retained_signal_keys,
            capture_causal_state_snapshots,
            &self.model.distributed_execution_plan,
            &self.model.distributed_parameter_buffers,
        )?;
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
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
        let input_signal = execution_graph.slice(first_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelInput(
                self.model.input_transducer_spec.output_signal_id.clone(),
            ),
        )?;
        let input_embedding = VulkanResidentBatchedInputEmbeddingRunner::new(
            input_device,
            block_width,
            embedding_weight,
            &input_signal.buffer,
            &self.model.input_transducer_batch_spirv_words,
            self.model.input_transducer_batch_control,
            &self.model.input_transducer_spec,
        )?;
        let last_device_index = *pipeline.last().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal pipeline is empty".to_string(),
            ))
        })?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        let output_signal = execution_graph.slice(last_device_index)?.signal_buffer(
            &VulkanComponentBatchSignalKey::ModelOutput(
                self.model.output_transducer_spec.input_signal_id.clone(),
            ),
        )?;
        let speculative_target_output = if self.speculative_decoders.is_empty() {
            None
        } else {
            let norm_weight = self
                .model
                .output_transducer_parameter_buffers
                .parameter_buffer(&self.model.output_transducer_spec.norm_parameter_tensor)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                        VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                            tensor: self
                                .model
                                .output_transducer_spec
                                .norm_parameter_tensor
                                .clone(),
                        },
                    )
                })?;
            let projection_weight = self
                .model
                .output_transducer_parameter_buffers
                .parameter_buffer(&self.model.output_transducer_spec.projection_parameter_tensor)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::OutputTransducer(
                        VulkanResidentOutputTransducerRunnerError::MissingTransducerParameterBuffer {
                            tensor: self
                                .model
                                .output_transducer_spec
                                .projection_parameter_tensor
                                .clone(),
                        },
                    )
                })?;
            let projection_scale = projection_scale_parameter_buffer(
                &self.model.output_transducer_parameter_buffers,
                &self.model.output_transducer_spec,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
            let sampler_lanes = vec![&self.sampler; block_width];
            Some(VulkanResidentBatchedOutputProjectionRunner::new_for_sampler_lanes(
                output_device,
                self.model.embedding_norm_batch_lane_tile_width,
                self.model.projection_batch_lane_tile_width,
                &output_signal.buffer,
                norm_weight,
                projection_weight,
                projection_scale,
                &self.model.embedding_norm_batch_spirv_words,
                &self.model.tied_projection_batch_spirv_words,
                &self.model.output_transducer_spec,
                &sampler_lanes,
                &self.model.sampler_kernels,
                &self.model.sampler_spec,
            )?)
        };
        let scalar_output = self.device_slices[last_device_index]
            .mounted
            .boundary_io
            .output_buffer(&self.model.output_transducer_spec.input_signal_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "temporal output device has no boundary {:?}",
                    self.model.output_transducer_spec.input_signal_id
                )))
            })?;
        let output_frame_copies = (0..block_width)
            .map(|frame_index| {
                let source_offset = output_signal
                    .frame_byte_capacity
                    .checked_mul(frame_index)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "temporal output frame offset overflowed".to_string(),
                        ))
                    })?;
                let copy = VulkanResidentBufferRangeCopy::new(
                    &output_signal.buffer,
                    &scalar_output.buffer,
                    source_offset,
                    0,
                    output_signal.frame_byte_capacity,
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                output_device
                    .create_resident_buffer_copy_batch(&[copy])
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let speculative_source_tap_frame_copies =
            mount_speculative_source_tap_frame_copies(
                devices,
                &self.model,
                &self.device_slices,
                &execution_graph,
                block_width,
            )?;
        self.temporal_block_executions.borrow_mut().insert(
            execution_key,
            VulkanResidentPlacedTemporalBlockRunner {
                execution_graph,
                input_embedding,
                output_frame_copies,
                speculative_source_tap_frame_copies,
                speculative_target_output,
                pipeline,
            },
        );
        Ok(())
    }

    fn run_causal_component_block(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        capture_causal_state_snapshots: bool,
    ) -> Result<VulkanPlacedEdgeTransportStats, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        self.ensure_temporal_block_execution(
            devices,
            input_token_ids.len(),
            capture_causal_state_snapshots,
        )?;
        let block_capacity = self.causal_block_lane_capacity(input_token_ids.len())?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "causal component batch context capacity exceeds u32".to_string(),
                ))
            })?;
        let runner_guard = self.temporal_block_executions.borrow();
        let runner = runner_guard
            .get(&(block_capacity, capture_causal_state_snapshots))
            .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "causal component batch execution is not mounted".to_string(),
            ))
        })?;
        if input_token_ids.len() > runner.execution_graph.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "causal component batch capacity {} cannot process {} tokens",
                    runner.execution_graph.lane_capacity,
                    input_token_ids.len(),
                )),
            ));
        }
        let pipeline_starts_at_input = runner
            .pipeline
            .first()
            .and_then(|index| self.device_slices.get(*index))
            .is_some_and(|slice| slice.device_id == self.model.input_device_id);
        let pipeline_ends_at_output = runner
            .pipeline
            .last()
            .and_then(|index| self.device_slices.get(*index))
            .is_some_and(|slice| slice.device_id == self.model.output_device_id);
        let completion_mode = if capture_causal_state_snapshots
            && pipeline_starts_at_input
            && pipeline_ends_at_output
            && runner.execution_graph.supports_deferred_completion()
            && std::env::var_os("NERVE_VK_PERF_LOGGER").is_none()
        {
            VulkanComponentBatchCompletionMode::Deferred
        } else {
            VulkanComponentBatchCompletionMode::Blocking
        };
        let input_device = devices.get(&self.model.input_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.input_device_id.clone(),
            }
        })?;
        if completion_mode == VulkanComponentBatchCompletionMode::Deferred {
            runner
                .input_embedding
                .submit_deferred(input_device, input_token_ids)?;
        } else {
            runner.input_embedding.run(input_device, input_token_ids)?;
        }

        let mut transport_stats = VulkanPlacedEdgeTransportStats::default();
        for (pipeline_index, device_index) in runner.pipeline.iter().copied().enumerate() {
            let slice = &self.device_slices[device_index];
            runner.execution_graph.run_causal_sequence(
                devices,
                device_index,
                &slice.device_id,
                &slice.mounted,
                input_token_ids,
                start_stream_tick,
                capacity,
                completion_mode,
            )?;
            if let Some(next_device_index) = runner.pipeline.get(pipeline_index + 1).copied() {
                let [outgoing] = slice.mounted.edge_io.outgoing_buffers.as_slice() else {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "causal component batch device {:?} has {} outgoing edges; expected one",
                            slice.device_id,
                            slice.mounted.edge_io.outgoing_buffers.len()
                        )),
                    ));
                };
                let route = runner.execution_graph.transfer_edge(
                    device_index,
                    next_device_index,
                    outgoing.endpoint.edge_index,
                )?;
                let transferred_bytes =
                    outgoing.byte_capacity.saturating_mul(input_token_ids.len());
                transport_stats.direct_copy_count =
                    transport_stats.direct_copy_count.saturating_add(1);
                transport_stats.direct_copy_byte_count = transport_stats
                    .direct_copy_byte_count
                    .saturating_add(transferred_bytes);
                transport_stats.direct_receive_count =
                    transport_stats.direct_receive_count.saturating_add(1);
                transport_stats.direct_receive_byte_count = transport_stats
                    .direct_receive_byte_count
                    .saturating_add(transferred_bytes);
                transport_stats
                    .edges
                    .push(VulkanPlacedEdgeTransportEdgeStats {
                        key: VulkanPlacedEdgePacketKey::from_outgoing_endpoint(
                            &outgoing.endpoint,
                        ),
                        signal: outgoing.endpoint.signal.clone(),
                        route,
                        byte_capacity: outgoing.byte_capacity,
                        publish_count: 1,
                        receive_count: 1,
                        transferred_byte_count: transferred_bytes,
                        queue_signal_count: usize::from(
                            route == VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
                        ),
                        queue_wait_count: usize::from(
                            route == VulkanPlacedEdgeTransferRoute::DeviceLocalStaging,
                        ),
                        host_wait_count: usize::from(
                            route == VulkanPlacedEdgeTransferRoute::HostStaging,
                        ),
                        queue_overlap_eligible: route.supports_queue_overlap(),
                        overlap_submission_count: usize::from(route.supports_queue_overlap()),
                    });
            }
        }
        Ok(transport_stats)
    }

    fn run_temporal_prompt_block(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        sample_last: bool,
    ) -> Result<VulkanResidentTemporalBlockRun, VulkanResidentInProcessPlacedRuntimeError> {
        if input_token_ids.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let tick_count = u64::try_from(input_token_ids.len())
            .map_err(|_| VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let end_stream_tick = start_stream_tick
            .checked_add(tick_count - 1)
            .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
        let transport_stats =
            self.run_causal_component_block(
                devices,
                input_token_ids,
                start_stream_tick,
                false,
            )?;
        let block_capacity = self.causal_block_lane_capacity(input_token_ids.len())?;
        let capacity =
            u32::try_from(self.model.dynamic_state_capacity_activations).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal context capacity exceeds u32".to_string(),
                ))
            })?;
        let runner_guard = self.temporal_block_executions.borrow();
        let runner = runner_guard.get(&(block_capacity, false)).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "temporal block execution is not mounted".to_string(),
                ))
            })?;
        let output_device = devices.get(&self.model.output_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: self.model.output_device_id.clone(),
            }
        })?;
        self.sampler
            .record_input_tokens(output_device, input_token_ids)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?;

        if !self.speculative_decoders.is_empty() {
            for (lane, input_token_id) in input_token_ids.iter().copied().enumerate() {
                if self
                    .speculative_decoders
                    .iter()
                    .any(VulkanResidentSpeculativeDecoderProcessor::is_parallel_block)
                {
                    runner.publish_speculative_source_tap_frame(lane)?;
                    let stream_tick = start_stream_tick
                        .checked_add(u64::try_from(lane).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow
                        })?)
                        .ok_or(VulkanResidentInProcessPlacedRuntimeError::StreamTickOverflow)?;
                    for decoder in self
                        .speculative_decoders
                        .iter()
                        .filter(|decoder| decoder.is_parallel_block())
                    {
                        let draft_device = devices.get(&decoder.device_id).ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                device_id: decoder.device_id.clone(),
                            }
                        })?;
                        decoder.run_state_step(
                            draft_device,
                            input_token_id,
                            stream_tick,
                        )?;
                    }
                }
            }
            let target_output = runner.speculative_target_output.as_ref().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal speculative target normalization is not mounted".to_string(),
                ))
            })?;
            target_output
                .projection
                .norm
                .run(output_device, input_token_ids.len())?;
            self.catch_up_speculative_decoders_from_target_frames(
                devices,
                input_token_ids,
                start_stream_tick,
                &target_output
                    .projection
                    .norm
                    .normalized_frames_buffer,
                self.model
                    .output_transducer_spec
                    .normalized_frame_byte_capacity,
            )?;
        }

        let sampled_token = if sample_last {
            let last_device_index = *runner.pipeline.last().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "temporal pipeline is empty".to_string(),
                ))
            })?;
            let output_slice = &self.device_slices[last_device_index];
            let output_device = devices.get(&output_slice.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: output_slice.device_id.clone(),
                }
            })?;
            runner
                .output_frame_copies
                .get(input_token_ids.len() - 1)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "temporal output frame copy is not mounted".to_string(),
                    ))
                })?
                .run()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            output_slice
                .mounted
                .stream_control_buffer
                .write_bytes_at(
                    VULKAN_STREAM_CONTROL_METADATA_OFFSET,
                    &stream_control_metadata_bytes(VulkanMountedPlacedStreamControl {
                        stream_tick: end_stream_tick,
                        control_flags: 0,
                        dynamic_state_capacity_activations: capacity,
                    }),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.output_transducer
                .run(output_device)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::OutputTransducer)?;
            Some(VulkanResidentSampledToken::from(
                &self
                    .sampler
                    .run(output_device)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::Sampler)?,
            ))
        } else {
            None
        };

        Ok(VulkanResidentTemporalBlockRun {
            sampled_token,
            scheduler_turn_count_per_tick: self.activation_schedule.turns.len(),
            completed_stage_count_per_tick: self
                .device_slices
                .iter()
                .map(|slice| slice.dispatch_count)
                .sum(),
            transport_stats,
        })
    }

}
