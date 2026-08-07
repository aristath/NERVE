fn committed_context_state_node_ids(
    circuit: &StreamCircuit,
    context_signal: &str,
) -> Result<BTreeSet<String>, VulkanError> {
    let producer_by_signal = circuit
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(index, node)| {
            node.outputs
                .iter()
                .map(move |signal| (signal.as_str(), index))
        })
        .collect::<BTreeMap<_, _>>();
    let mut depends_on_context = BTreeMap::<String, bool>::new();
    depends_on_context.insert(context_signal.to_string(), true);
    for node in &circuit.nodes {
        let depends = node.inputs.iter().any(|signal| {
            depends_on_context.get(signal).copied().unwrap_or(false)
        });
        for output in &node.outputs {
            depends_on_context.insert(output.clone(), depends);
        }
    }
    let state_sinks = circuit
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            !node.state_writes.is_empty()
                && node.inputs.iter().any(|signal| {
                    depends_on_context.get(signal).copied().unwrap_or(false)
                })
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if state_sinks.is_empty() {
        return Err(VulkanError(format!(
            "circuit {:?} has no state update derived from committed context signal {context_signal:?}",
            circuit.source.component_id
        )));
    }

    let mut selected_indices = BTreeSet::new();
    let mut pending = state_sinks;
    while let Some(index) = pending.pop() {
        if !selected_indices.insert(index) {
            continue;
        }
        for input in &circuit.nodes[index].inputs {
            if let Some(producer) = producer_by_signal.get(input.as_str()) {
                pending.push(*producer);
            }
        }
    }
    Ok(selected_indices
        .into_iter()
        .map(|index| circuit.nodes[index].id.clone())
        .collect())
}

fn producer_dependency_node_ids(
    circuit: &StreamCircuit,
    output_signal: &str,
) -> Result<BTreeSet<String>, VulkanError> {
    let producer_by_signal = circuit
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(index, node)| {
            node.outputs
                .iter()
                .map(move |signal| (signal.as_str(), index))
        })
        .collect::<BTreeMap<_, _>>();
    let producer = producer_by_signal.get(output_signal).copied().ok_or_else(|| {
        VulkanError(format!(
            "circuit {:?} has no producer for state-ingestion output signal {output_signal:?}",
            circuit.source.component_id,
        ))
    })?;
    let mut selected_indices = BTreeSet::new();
    let mut pending = vec![producer];
    while let Some(index) = pending.pop() {
        if !selected_indices.insert(index) {
            continue;
        }
        for input in &circuit.nodes[index].inputs {
            if let Some(producer) = producer_by_signal.get(input.as_str()) {
                pending.push(*producer);
            }
        }
    }
    Ok(selected_indices
        .into_iter()
        .map(|index| circuit.nodes[index].id.clone())
        .collect())
}

fn selected_boundary_input_ids(
    circuit: &StreamCircuit,
    selected_node_ids: &BTreeSet<String>,
) -> BTreeSet<String> {
    let boundary_inputs = circuit
        .boundary
        .inputs
        .iter()
        .map(|port| port.id.as_str())
        .collect::<BTreeSet<_>>();
    circuit
        .nodes
        .iter()
        .filter(|node| selected_node_ids.contains(&node.id))
        .flat_map(|node| node.inputs.iter())
        .filter(|signal| boundary_inputs.contains(signal.as_str()))
        .cloned()
        .collect()
}

fn offset_stream_tick(stream_tick: u64, offset: i64) -> Result<u64, VulkanError> {
    if offset < 0 {
        stream_tick.checked_sub(offset.unsigned_abs())
    } else {
        stream_tick.checked_add(offset as u64)
    }
    .ok_or_else(|| {
        VulkanError(format!(
            "stream tick {stream_tick} cannot apply compiled source-context offset {offset}"
        ))
    })
}

fn proposal_node_ids(
    circuit: &StreamCircuit,
    committed_context_node_ids: &BTreeSet<String>,
) -> Result<BTreeSet<String>, VulkanError> {
    let proposal_nodes = circuit
        .nodes
        .iter()
        .filter(|node| !committed_context_node_ids.contains(&node.id))
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    if proposal_nodes.is_empty() {
        return Err(VulkanError(format!(
            "circuit {:?} has no proposal nodes outside its committed-context cone",
            circuit.source.component_id
        )));
    }
    Ok(proposal_nodes)
}

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
        let VulkanResidentSpeculativeDecoderModelExecution::ParallelBlock {
            block_width,
            source_context_tick_offset,
        } = &model.execution
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
        let committed_context_edges = graph
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                matches!(
                    &edge.connection,
                    StreamCircuitConnection::SharedContext { state_update }
                        if state_update == "committed_target_only"
                )
            })
            .collect::<Vec<_>>();
        if committed_context_edges.is_empty()
            || committed_context_edges.iter().any(|(_, edge)| {
                edge.source.component_id != input_component.component_id
                    || !processor_components.iter().any(|component| {
                        component.component_id == edge.destination.component_id
                    })
            })
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} has an invalid committed-context state topology",
                    model.id
                )),
            ));
        }
        let mut state_node_ids_by_component = BTreeMap::new();
        for (_, edge) in &committed_context_edges {
            let component = processor_components
                .iter()
                .copied()
                .find(|component| component.component_id == edge.destination.component_id)
                .expect("validated committed-context destination exists");
            let context_port = component
                .circuit
                .boundary
                .inputs
                .iter()
                .find(|port| port.id == edge.destination.port_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "parallel speculative decoder {:?} committed-context destination {}.{} is absent",
                            model.id, edge.destination.component_id, edge.destination.port_id
                        )),
                    )
                })?;
            let node_ids = committed_context_state_node_ids(
                &component.circuit,
                &context_port.id,
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            state_node_ids_by_component
                .entry(component.component_id.clone())
                .or_insert_with(BTreeSet::new)
                .extend(node_ids);
        }
        if state_node_ids_by_component.len() != processor_components.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} does not update committed context in every processor component",
                    model.id
                )),
            ));
        }
        let mut input_state_node_ids = BTreeSet::new();
        for (_, edge) in &committed_context_edges {
            let output_port = input_component
                .circuit
                .boundary
                .outputs
                .iter()
                .find(|port| port.id == edge.source.port_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::Package(
                        VulkanResidentTokenModelPackageError::new(format!(
                            "parallel speculative decoder {:?} committed-context source {}.{} is absent",
                            model.id, edge.source.component_id, edge.source.port_id,
                        )),
                    )
                })?;
            let source_signal = output_port.source.as_deref().unwrap_or(&output_port.id);
            input_state_node_ids.extend(
                producer_dependency_node_ids(&input_component.circuit, source_signal)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
        }
        let state_input_signal_ids =
            selected_boundary_input_ids(&input_component.circuit, &input_state_node_ids);
        let state_input_ports = graph
            .boundary
            .external_inputs
            .iter()
            .filter(|port| {
                port.endpoint.component_id == input_component.component_id
                    && state_input_signal_ids.contains(&port.endpoint.port_id)
            })
            .collect::<Vec<_>>();
        if state_input_ports.len() != state_input_signal_ids.len()
            || state_input_ports.iter().any(|port| port.source_tap.is_none())
        {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} committed-context input cone must be supplied entirely by target source taps",
                    model.id,
                )),
            ));
        }
        let mut state_ingestion_node_ids_by_component = state_node_ids_by_component.clone();
        state_ingestion_node_ids_by_component
            .insert(input_component.component_id.clone(), input_state_node_ids);
        let proposal_node_ids_by_component = processor_components
            .iter()
            .map(|component| {
                let committed_only = state_node_ids_by_component
                    .get(&component.component_id)
                    .expect("every processor has a validated committed-context cone");
                let proposal_nodes = proposal_node_ids(&component.circuit, committed_only)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                Ok::<_, VulkanResidentInProcessPlacedRuntimeError>((
                    component.component_id.clone(),
                    proposal_nodes,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let processor_phase =
            VulkanResidentPlacedComponentBatchRunner::new_single_device_for_nodes(
                device,
                &device_slice,
                &format!("draft:{}:parallel", model.id),
                *block_width,
                VulkanComponentBatchExecutionMode::ParallelBlock,
                proposal_node_ids_by_component,
            )?;
        let state_processor_phase =
            VulkanResidentPlacedComponentBatchRunner::new_single_device_for_nodes(
                device,
                &device_slice,
                &format!("draft:{}:committed-context", model.id),
                1,
                VulkanComponentBatchExecutionMode::ParallelBlock,
                state_node_ids_by_component.clone(),
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
        let mut batch_source_taps = Vec::new();
        for port in graph
            .boundary
            .external_inputs
            .iter()
            .filter(|port| port.source_tap.is_some())
        {
            let tap = port.source_tap.as_ref().expect("filtered source tap");
            let source =
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
            if destination.byte_capacity != source.frame_byte_capacity {
                return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                    VulkanResidentTokenModelPackageError::new(format!(
                        "parallel speculative decoder {:?} source tap {}.{} has {} bytes, destination {:?} has {}",
                        model.id,
                        tap.component_id,
                        tap.port_id,
                        source.frame_byte_capacity,
                        port.id,
                        destination.byte_capacity
                    )),
                ));
            }
            source_taps.push(VulkanSpeculativeSourceTapTransfer::new(
                device_for(source.device_id)?,
                device,
                source.scalar_buffer,
                &destination.buffer,
                source.frame_byte_capacity,
            )?);
            if state_input_signal_ids.contains(&port.endpoint.port_id) {
                batch_source_taps.push(VulkanParallelSpeculativeSourceTapBatchBinding {
                    source_device_id: source.device_id.to_string(),
                    source_scalar_buffer: Arc::clone(&source.scalar_buffer_owner),
                    source_batch_signal_key: source.batch_signal_key,
                    destination_signal_id: port.id.clone(),
                    frame_byte_capacity: source.frame_byte_capacity,
                });
            }
        }
        if batch_source_taps.len() != state_input_signal_ids.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} mounted {} batched source taps for {} committed-context inputs",
                    model.id,
                    batch_source_taps.len(),
                    state_input_signal_ids.len(),
                )),
            ));
        }

        let processor_ids = processor_components
            .iter()
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut ingress_ranges = Vec::new();
        let mut state_ingress_ranges = Vec::new();
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
            if !source_is_processor && destination_is_processor {
                match edge.connection {
                    StreamCircuitConnection::ParallelBlockScatter { width } => {
                        let batch_edge =
                            processor_phase.single_device_edge_signal_buffer(edge_index)?;
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
                        let state_edge = state_processor_phase
                            .single_device_edge_signal_buffer(edge_index)?;
                        if mounted_edge.byte_capacity != state_edge.frame_byte_capacity {
                            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                                VulkanResidentTokenModelPackageError::new(format!(
                                    "parallel speculative decoder {:?} committed-context edge {edge_index} has incompatible state-ingestion storage",
                                    model.id
                                )),
                            ));
                        }
                        state_ingress_ranges.push(
                            VulkanResidentBufferRangeCopy::new(
                                &mounted_edge.buffer,
                                &state_edge.buffer,
                                0,
                                0,
                                state_edge.frame_byte_capacity,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
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
                let batch_edge = processor_phase.single_device_edge_signal_buffer(edge_index)?;
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
        if ingress_ranges.is_empty()
            || state_ingress_ranges.len() != committed_context_edges.len()
            || egress_ranges.len() != 1
        {
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
        let state_ingress_copies = device
            .create_resident_buffer_copy_batch(&state_ingress_ranges)
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
        let draft_tokens_output = device_slice
            .mounted
            .boundary_io
            .output_buffer(&draft_tokens_output_signal_id)
            .expect("validated parallel token output remains mounted");
        let confidence_output = device_slice
            .mounted
            .boundary_io
            .output_buffer(&confidence_output_signal_id)
            .expect("validated parallel confidence output remains mounted");
        let output_readback = device
            .create_resident_buffer_readback_binding(&[
                VulkanResidentBufferReadRange::new(
                    &draft_tokens_output.buffer,
                    0,
                    block_width * size_of::<u32>(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                VulkanResidentBufferReadRange::new(
                    &confidence_output.buffer,
                    0,
                    block_width * size_of::<f32>(),
                )
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            ])
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let state_transaction = VulkanResidentStateTransactionBank::new_transactional(
            device,
            &device_slice.mounted.buffers,
            1,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let minimum_draft_token_count = model.package.minimum_draft_tokens().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative decoder {:?} has no minimum draft width",
                    model.id
                )),
            )
        })?;

        Ok(Self {
            device_slice,
            input_phase,
            processor_phase,
            state_processor_phase,
            output_phase,
            source_taps,
            batch_source_taps,
            state_ingestion_node_ids_by_component,
            ingress_copies,
            state_ingress_copies,
            egress_copies,
            output_readback,
            anchor_input_signal_id: anchor_port.id.clone(),
            minimum_draft_token_count,
            block_width: *block_width,
            source_context_tick_offset: *source_context_tick_offset,
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

    fn run_state_step(
        &self,
        device: &VulkanComputeDevice,
        input_token_id: u32,
        stream_tick: u64,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let anchor = self
            .mounted()
            .boundary_io
            .input_buffer(&self.anchor_input_signal_id)
            .expect("validated parallel anchor remains mounted");
        anchor
            .buffer
            .write_bytes(&input_token_id.to_le_bytes())
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
            stream_tick,
            control_flags: 0,
            dynamic_state_capacity_activations,
        };
        self.input_phase
            .run_with_stream_control(device, control)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;
        self.state_ingress_copies
            .run()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.state_processor_phase.run_parallel_block_single_device(
            device,
            &self.device_slice.device_id,
            &[input_token_id],
            stream_tick,
            dynamic_state_capacity_activations,
        )
    }

    fn run_draft_window(
        &self,
        device: &VulkanComputeDevice,
        initial_token_id: u32,
        start_stream_tick: u64,
        draft_token_count: usize,
        confidence_threshold: f32,
    ) -> Result<Vec<u32>, VulkanResidentInProcessPlacedRuntimeError> {
        let draft_token_count = draft_token_count.min(self.block_width);
        if draft_token_count == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget);
        }
        if draft_token_count < self.minimum_draft_token_count {
            return Err(VulkanResidentInProcessPlacedRuntimeError::Package(
                VulkanResidentTokenModelPackageError::new(format!(
                    "parallel speculative draft width {draft_token_count} is below the compiled minimum {}",
                    self.minimum_draft_token_count,
                )),
            ));
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
        let source_context_tick = offset_stream_tick(
            start_stream_tick,
            self.source_context_tick_offset,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let control = VulkanMountedPlacedStreamControl {
            stream_tick: source_context_tick,
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
            source_context_tick,
            dynamic_state_capacity_activations,
        )?;
        self.egress_copies
            .run()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        self.output_phase
            .run_with_stream_control(device, control)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::ResidentDispatch)?;

        let output_readback = self
            .output_readback
            .run()
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut tokens = output_readback
            .range_bytes(0)
            .expect("mounted parallel token readback range exists")
            .chunks_exact(size_of::<u32>())
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("u32-sized token chunk")))
            .take(draft_token_count)
            .collect::<Vec<_>>();
        let confidence_logits = output_readback
            .range_bytes(1)
            .expect("mounted parallel confidence readback range exists")
            .chunks_exact(size_of::<f32>())
            .take(draft_token_count)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32-sized confidence chunk")))
            .collect::<Vec<_>>();
        let confident_prefix_len =
            speculative_confident_prefix_len(&confidence_logits, confidence_threshold)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        tokens.truncate(confident_prefix_len);
        Ok(tokens)
    }
}
