impl VulkanResidentParallelBlockSpeculativeDecoderProcessor {
    fn from_model<'a, F>(
        device: &VulkanComputeDevice,
        model: &VulkanResidentSpeculativeDecoderModelPackage,
        target_model: &VulkanResidentInProcessPlacedModelPackage,
        target_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        device_for: &F,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError>
    where
        F: Fn(&str) -> Result<&'a VulkanComputeDevice, VulkanResidentInProcessPlacedRuntimeError>,
    {
        let VulkanResidentSpeculativeDecoderModelExecution::ParallelBlock { block_width } =
            &model.execution
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "autoregressive speculative decoder {:?} cannot mount as a parallel block",
                    model.id
                )),
            ));
        };
        let device_slice = mount_speculative_decoder_device_slice(device, model)?;
        let graph = &model.package.circuit_graph;
        let input_components = graph
            .components
            .iter()
            .filter(|component| component.runtime_role == CircuitRuntimeRole::DraftInputAdapter)
            .collect::<Vec<_>>();
        let processor_components = graph
            .components
            .iter()
            .filter(|component| component.runtime_role == CircuitRuntimeRole::DraftProcessor)
            .collect::<Vec<_>>();
        let output_components = graph
            .components
            .iter()
            .filter(|component| {
                component.runtime_role == CircuitRuntimeRole::DraftOutputTransducer
            })
            .collect::<Vec<_>>();
        let ([input_component], [output_component]) =
            (input_components.as_slice(), output_components.as_slice())
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} requires one input and one output component",
                    model.id
                )),
            ));
        };
        if processor_components.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} has no processor components",
                    model.id
                )),
            ));
        }

        let input_phase = device_slice
            .mounted
            .create_resident_execution_graph_runner(
                device,
                &device_slice.mounted_bound,
                [input_component.component_id.as_str()],
                model.device_slice.loaded_manifest(),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        let output_phase = device_slice
            .mounted
            .create_resident_execution_graph_runner(
                device,
                &device_slice.mounted_bound,
                [output_component.component_id.as_str()],
                model.device_slice.loaded_manifest(),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        let processor_phase = VulkanResidentPlacedComponentBatchRunner::
            new_single_device_for_components(
                device,
                &device_slice,
                &format!("draft:{}:parallel", model.id),
                *block_width,
                VulkanComponentBatchExecutionMode::ParallelBlock,
                processor_components
                    .iter()
                    .map(|component| component.component_id.clone()),
            )?;

        let anchor_ports = graph
            .boundary
            .external_inputs
            .iter()
            .filter(|port| port.source_tap.is_none())
            .collect::<Vec<_>>();
        let [anchor_port] = anchor_ports.as_slice() else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} requires one untapped anchor input",
                    model.id
                )),
            ));
        };
        let anchor_input = device_slice
            .mounted
            .boundary_io
            .input_buffer(&anchor_port.id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "parallel speculative decoder {:?} anchor input {:?} is not mounted",
                        model.id, anchor_port.id
                    )),
                )
            })?;
        if anchor_input.byte_capacity != size_of::<u32>() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} anchor input has {} bytes, expected {}",
                    model.id,
                    anchor_input.byte_capacity,
                    size_of::<u32>()
                )),
            ));
        }

        let mut source_taps = Vec::new();
        for port in graph
            .boundary
            .external_inputs
            .iter()
            .filter(|port| port.source_tap.is_some())
        {
            let tap = port.source_tap.as_ref().expect("filtered source tap");
            let (source_device_id, source, source_byte_len) =
                resolved_speculative_source_tap_buffer(target_model, target_slices, tap)?;
            let destination = device_slice
                .mounted
                .boundary_io
                .input_buffer(&port.id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "parallel speculative decoder {:?} source-tap input {:?} is not mounted",
                            model.id, port.id
                        )),
                    )
                })?;
            if destination.byte_capacity != source_byte_len {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "parallel speculative decoder {:?} source tap {}.{} has {} bytes, destination {:?} has {}",
                        model.id,
                        tap.component_id,
                        tap.port_id,
                        source_byte_len,
                        port.id,
                        destination.byte_capacity
                    )),
                ));
            }
            source_taps.push(VulkanSpeculativeSourceTapTransfer::new(
                device_for(source_device_id)?,
                device,
                source,
                &destination.buffer,
                source_byte_len,
            )?);
        }

        let processor_ids = processor_components
            .iter()
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut ingress_ranges = Vec::new();
        let mut egress_ranges = Vec::new();
        for (edge_index, edge) in graph.edges.iter().enumerate() {
            let source_is_processor = processor_ids.contains(edge.source.component_id.as_str());
            let destination_is_processor =
                processor_ids.contains(edge.destination.component_id.as_str());
            if source_is_processor == destination_is_processor {
                continue;
            }
            let mounted_edge = device_slice
                .mounted
                .edge_io
                .local_edge_buffer(edge_index)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "parallel speculative decoder {:?} phase edge {edge_index} is not device-local",
                            model.id
                        )),
                    )
                })?;
            let batch_edge = processor_phase.single_device_edge_signal_buffer(edge_index)?;
            if !source_is_processor && destination_is_processor {
                match edge.connection {
                    StreamCircuitConnection::ParallelBlockScatter { width } => {
                        if width != *block_width
                            || mounted_edge.byte_capacity != batch_edge.buffer.byte_capacity()
                        {
                            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "parallel speculative decoder {:?} scatter edge {edge_index} has incompatible storage",
                                    model.id
                                )),
                            ));
                        }
                        ingress_ranges.push(
                            VulkanResidentBufferRangeCopy::new(
                                &mounted_edge.buffer,
                                &batch_edge.buffer,
                                0,
                                0,
                                mounted_edge.byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                    }
                    StreamCircuitConnection::SharedContext { .. } => {
                        if mounted_edge.byte_capacity != batch_edge.frame_byte_capacity {
                            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "parallel speculative decoder {:?} shared-context edge {edge_index} has incompatible storage",
                                    model.id
                                )),
                            ));
                        }
                        for lane in 0..*block_width {
                            ingress_ranges.push(
                                VulkanResidentBufferRangeCopy::new(
                                    &mounted_edge.buffer,
                                    &batch_edge.buffer,
                                    0,
                                    lane * batch_edge.frame_byte_capacity,
                                    batch_edge.frame_byte_capacity,
                                )
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                            );
                        }
                    }
                    _ => {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                            VulkanResidentTokenModelPackageError::new(format!(
                                "parallel speculative decoder {:?} input phase edge {edge_index} has unsupported semantics",
                                model.id
                            )),
                        ))
                    }
                }
            } else {
                if !matches!(
                    edge.connection,
                    StreamCircuitConnection::ParallelBlockGather { width } if width == *block_width
                ) || mounted_edge.byte_capacity != batch_edge.buffer.byte_capacity()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "parallel speculative decoder {:?} gather edge {edge_index} has incompatible storage",
                            model.id
                        )),
                    ));
                }
                egress_ranges.push(
                    VulkanResidentBufferRangeCopy::new(
                        &batch_edge.buffer,
                        &mounted_edge.buffer,
                        0,
                        0,
                        mounted_edge.byte_capacity,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
            }
        }
        if ingress_ranges.is_empty() || egress_ranges.len() != 1 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} has incomplete phase boundaries",
                    model.id
                )),
            ));
        }
        let ingress_copies = device
            .create_resident_buffer_copy_batch(&ingress_ranges)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let egress_copies = device
            .create_resident_buffer_copy_batch(&egress_ranges)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        let mut token_output = None;
        let mut confidence_output = None;
        for public in &graph.boundary.public_outputs {
            let component = graph
                .components
                .iter()
                .find(|component| component.component_id == public.endpoint.component_id)
                .expect("validated public output component exists");
            let port = component
                .circuit
                .boundary
                .outputs
                .iter()
                .find(|port| port.id == public.endpoint.port_id)
                .expect("validated public output port exists");
            match port.signal.as_str() {
                "token_id_block" => token_output = Some(public.id.clone()),
                "scalar_block" => confidence_output = Some(public.id.clone()),
                _ => {}
            }
        }
        let (Some(draft_tokens_output_signal_id), Some(confidence_output_signal_id)) =
            (token_output, confidence_output)
        else {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} does not expose token and confidence blocks",
                    model.id
                )),
            ));
        };
        let state_transaction = VulkanResidentStateTransactionBank::new_transactional(
            device,
            &device_slice.mounted.buffers,
            1,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;

        Ok(Self {
            device_slice,
            input_phase,
            processor_phase,
            output_phase,
            source_taps,
            ingress_copies,
            egress_copies,
            anchor_input_signal_id: anchor_port.id.clone(),
            draft_tokens_output_signal_id,
            confidence_output_signal_id,
            block_width: *block_width,
            state_transaction,
        })
    }

    fn mounted(&self) -> &VulkanMountedPlacedStreamCircuit {
        &self.device_slice.mounted
    }

    fn capture_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.state_transaction
            .capture_baseline(&self.mounted().buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn restore_baseline(&self) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.state_transaction
            .restore_baseline(&self.mounted().buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn run_draft_window(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        let draft_token_count = draft_token_count.min(self.block_width);
        if draft_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        let anchor = self
            .mounted()
            .boundary_io
            .input_buffer(&self.anchor_input_signal_id)
            .expect("validated parallel anchor remains mounted");
        anchor
            .buffer
            .write_bytes(&initial_token_id.to_le_bytes())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        for tap in &self.source_taps {
            tap.run()?;
        }
        let dynamic_state_capacity_activations = u32::try_from(
            self.mounted().buffers.dynamic_state_capacity_activations,
        )
        .map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "parallel speculative state capacity exceeds u32".to_string(),
            ))
        })?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick: start_stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };
        self.input_phase
            .run_with_stream_control(device, control)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        self.ingress_copies
            .run()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.processor_phase.run_parallel_block_single_device(
            device,
            &self.device_slice.device_id,
            &vec![initial_token_id; self.block_width],
            start_stream_tick,
            dynamic_state_capacity_activations,
        )?;
        self.egress_copies
            .run()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.output_phase
            .run_with_stream_control(device, control)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;

        let tokens = self
            .mounted()
            .boundary_io
            .output_buffer(&self.draft_tokens_output_signal_id)
            .expect("validated parallel token output remains mounted")
            .buffer
            .read_bytes(self.block_width * size_of::<u32>())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
            .chunks_exact(size_of::<u32>())
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("u32-sized token chunk")))
            .take(draft_token_count)
            .collect::<Vec<_>>();
        let confidence_bytes = self
            .mounted()
            .boundary_io
            .output_buffer(&self.confidence_output_signal_id)
            .expect("validated parallel confidence output remains mounted")
            .buffer
            .read_bytes(self.block_width * size_of::<f32>())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if confidence_bytes
            .chunks_exact(size_of::<f32>())
            .take(draft_token_count)
            .any(|bytes| {
                !f32::from_le_bytes(bytes.try_into().expect("f32-sized confidence chunk"))
                    .is_finite()
            })
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "parallel speculative decoder produced non-finite confidence logits"
                        .to_string(),
                ),
            ));
        }
        Ok(tokens)
    }
}
