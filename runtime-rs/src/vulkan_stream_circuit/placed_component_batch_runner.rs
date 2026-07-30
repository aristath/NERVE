impl VulkanResidentPlacedComponentBatchRunner {
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        runtime_execution_identity: &str,
        quantum_calibrators: &BTreeMap<
            String,
            Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
        >,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        capture_causal_state_snapshots: bool,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        distributed_parameter_buffers: &VulkanDistributedParameterBuffers,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let lane_mounteds_by_slice = placed_slices
            .iter()
            .map(|slice| vec![&slice.mounted; lane_capacity])
            .collect::<Vec<_>>();
        Self::new_with_lane_mounteds(
            devices,
            placed_slices,
            runtime_execution_identity,
            &lane_mounteds_by_slice,
            quantum_calibrators,
            lane_capacity,
            execution_mode,
            capture_causal_state_snapshots,
            distributed_execution_plan,
            distributed_parameter_buffers,
        )
    }

    fn new_for_independent_streams(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        processors: &[&VulkanResidentInProcessPlacedStreamProcessor],
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let first = processors.first().copied().ok_or(
            VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget,
        )?;
        let lane_capacity = processors.len();
        for processor in processors.iter().copied().skip(1) {
            if processor.device_slices.len() != first.device_slices.len()
                || processor
                    .device_slices
                    .iter()
                    .zip(&first.device_slices)
                    .any(|(candidate, reference)| {
                        candidate.device_id != reference.device_id
                            || candidate.mounted_bound.dispatches.len()
                                != reference.mounted_bound.dispatches.len()
                    })
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "multi-stream component batch requires identical placed device slices"
                            .to_string(),
                    ),
                ));
            }
        }
        let lane_mounteds_by_slice = (0..first.device_slices.len())
            .map(|slice_index| {
                processors
                    .iter()
                    .map(|processor| &processor.device_slices[slice_index].mounted)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self::new_with_lane_mounteds(
            devices,
            &first.device_slices,
            &first.model.runtime_execution_identity,
            &lane_mounteds_by_slice,
            &first.execution_quantum_calibrators,
            lane_capacity,
            VulkanComponentBatchExecutionMode::IndependentStreams,
            false,
            &first.model.distributed_execution_plan,
            &first.model.distributed_parameter_buffers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_lane_mounteds(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        runtime_execution_identity: &str,
        lane_mounteds_by_slice: &[Vec<&VulkanMountedPlacedStreamCircuit>],
        quantum_calibrators: &BTreeMap<
            String,
            Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
        >,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        capture_causal_state_snapshots: bool,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        distributed_parameter_buffers: &VulkanDistributedParameterBuffers,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if lane_mounteds_by_slice.len() != placed_slices.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch has {} device slices but {} lane-mounted slice sets",
                    placed_slices.len(),
                    lane_mounteds_by_slice.len()
                )),
            ));
        }
        let slices = placed_slices
            .iter()
            .zip(lane_mounteds_by_slice)
            .map(|slice| {
                let (slice, lane_mounteds) = slice;
                let device = devices.get(&slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: slice.device_id.clone(),
                    }
                })?;
                let quantum_calibrator = quantum_calibrators
                    .get(&slice.device_id)
                    .cloned()
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: slice.device_id.clone(),
                        }
                    })?;
                VulkanResidentComponentBatchSliceRunner::new(
                    devices,
                    device,
                    slice,
                    runtime_execution_identity,
                    lane_mounteds,
                    lane_capacity,
                    execution_mode,
                    capture_causal_state_snapshots,
                    distributed_execution_plan,
                    quantum_calibrator,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distributed_dispatches = VulkanDistributedComponentBatchRunners::new(
            devices,
            placed_slices,
            &slices,
            distributed_execution_plan,
            distributed_parameter_buffers,
            lane_capacity,
            execution_mode,
        )?;
        let mut edge_transfers = Vec::new();
        for (source_device_index, placed_slice) in placed_slices.iter().enumerate() {
            for outgoing in &placed_slice.mounted.edge_io.outgoing_buffers {
                let destination_device_index = placed_slices
                    .iter()
                    .position(|candidate| {
                        candidate.device_id == outgoing.endpoint.remote_device_id
                            && candidate
                                .mounted
                                .edge_io
                                .incoming_buffer(outgoing.endpoint.edge_index)
                                .is_some()
                    })
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "component batch edge {} has no destination device {:?}",
                                outgoing.endpoint.edge_index, outgoing.endpoint.remote_device_id
                            ),
                        ))
                    })?;
                let source = slices[source_device_index].signal_buffer(
                    &VulkanComponentBatchSignalKey::OutgoingEdge(outgoing.endpoint.edge_index),
                )?;
                let destination = slices[destination_device_index].signal_buffer(
                    &VulkanComponentBatchSignalKey::IncomingEdge(outgoing.endpoint.edge_index),
                )?;
                if source.frame_byte_capacity != destination.frame_byte_capacity
                    || source.buffer.byte_capacity() != destination.buffer.byte_capacity()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "component batch edge {} source and destination capacities differ",
                            outgoing.endpoint.edge_index
                        )),
                    ));
                }
                let source_device = devices.get(&placed_slice.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: placed_slice.device_id.clone(),
                    }
                })?;
                let destination_device = devices
                    .get(&placed_slices[destination_device_index].device_id)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: placed_slices[destination_device_index].device_id.clone(),
                        }
                    })?;
                let byte_len = source.buffer.byte_capacity();
                let binding = if Rc::ptr_eq(source_device, destination_device) {
                    VulkanComponentBatchEdgeTransferBinding::Resident(Box::new(
                        source_device
                            .create_resident_buffer_copy(
                                &source.buffer,
                                &destination.buffer,
                                byte_len,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                    ))
                } else {
                    if !source_device.supports_shared_host_memory()
                        || !destination_device.supports_shared_host_memory()
                        || !source_device.supports_opaque_fd_timeline_semaphores()
                        || !destination_device.supports_opaque_fd_timeline_semaphores()
                    {
                        VulkanComponentBatchEdgeTransferBinding::HostStaging {
                            source: Arc::clone(&source.buffer),
                            destination: Arc::clone(&destination.buffer),
                            byte_len,
                        }
                    } else {
                        let staging_allocation = source_device
                            .create_shared_host_allocation(
                                &[destination_device.as_ref()],
                                byte_len,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                        let source_staging = Arc::new(
                            source_device
                                .import_shared_host_buffer(Arc::clone(&staging_allocation))
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                        let destination_staging = Arc::new(
                            destination_device
                                .import_shared_host_buffer(staging_allocation)
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                        let source_copy = Box::new(
                            source_device
                                .create_resident_buffer_copy(
                                    &source.buffer,
                                    &source_staging,
                                    byte_len,
                                )
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                        let destination_copy = Box::new(
                            destination_device
                                .create_resident_buffer_copy(
                                    &destination_staging,
                                    &destination.buffer,
                                    byte_len,
                                )
                                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                        );
                        let source_signal = source_device
                            .create_opaque_fd_exportable_timeline_semaphore(0)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                        let destination_wait = destination_device
                            .create_timeline_semaphore(0)
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                        destination_device
                            .import_timeline_semaphore_opaque_fd(
                                &destination_wait,
                                source_device
                                    .export_timeline_semaphore_opaque_fd(&source_signal)
                                    .map_err(
                                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                                    )?,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                        VulkanComponentBatchEdgeTransferBinding::DeviceLocalStaging {
                            source_device: Rc::clone(source_device),
                            destination_device: Rc::clone(destination_device),
                            source_copy,
                            destination_copy,
                            source_signal,
                            destination_wait,
                            next_value: Cell::new(1),
                            _source_staging: source_staging,
                            _destination_staging: destination_staging,
                        }
                    }
                };
                edge_transfers.push(VulkanComponentBatchEdgeTransfer {
                    source_device_index,
                    destination_device_index,
                    edge_index: outgoing.endpoint.edge_index,
                    binding,
                });
            }
        }
        Ok(Self {
            distributed_dispatches,
            lane_capacity,
            slices,
            edge_transfers,
        })
    }

    fn slice(
        &self,
        index: usize,
    ) -> Result<&VulkanResidentComponentBatchSliceRunner, VulkanResidentInProcessPlacedRuntimeError>
    {
        self.slices.get(index).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "placed component batch has no device slice {index}"
            )))
        })
    }

    fn commit_causal_state_prefix(
        &self,
        processed_tick_count: usize,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        if self
            .slices
            .iter()
            .any(|slice| !slice.can_commit_causal_state_prefix())
        {
            return Ok(false);
        }
        for slice in &self.slices {
            if !slice.commit_causal_state_prefix(processed_tick_count)? {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(
                        "causal state snapshot capability changed during prefix commit"
                            .to_string(),
                    ),
                ));
            }
        }
        Ok(true)
    }

    fn supports_deferred_completion(&self) -> bool {
        self.slices
            .iter()
            .all(VulkanResidentComponentBatchSliceRunner::supports_deferred_completion)
            && self
                .edge_transfers
                .iter()
                .all(VulkanComponentBatchEdgeTransfer::supports_deferred_completion)
    }

    fn transfer_edge(
        &self,
        source_device_index: usize,
        destination_device_index: usize,
        edge_index: usize,
    ) -> Result<VulkanPlacedEdgeTransferRoute, VulkanResidentInProcessPlacedRuntimeError> {
        self.edge_transfers
            .iter()
            .find(|transfer| {
                transfer.source_device_index == source_device_index
                    && transfer.destination_device_index == destination_device_index
                    && transfer.edge_index == edge_index
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no edge transfer {source_device_index}->{destination_device_index}:{edge_index}"
                )))
            })?
            .run()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_causal_sequence(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device_index: usize,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        completion_mode: VulkanComponentBatchCompletionMode,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(owner_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: owner_device_id.to_string(),
            }
        })?;
        self.slice(device_index)?.run_causal_sequence(
            devices,
            device,
            owner_device_id,
            &self.distributed_dispatches,
            mounted,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            completion_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_independent_streams(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device_index: usize,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(owner_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: owner_device_id.to_string(),
            }
        })?;
        self.slice(device_index)?.run_independent_streams(
            devices,
            device,
            owner_device_id,
            &self.distributed_dispatches,
            mounted,
            input_token_ids,
            stream_ticks,
            dynamic_state_capacity_activations,
        )
    }
}

fn component_batch_signal_target(
    descriptor: &VulkanMountedPlacedBoundDescriptor,
) -> Result<Option<(VulkanComponentBatchSignalKey, usize)>, VulkanResidentInProcessPlacedRuntimeError> {
    let target = match &descriptor.target {
        VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target:
                VulkanBoundDescriptorTarget::ActivationSlot {
                    component_id,
                    signal_id,
                    signal_byte_capacity,
                    ..
                },
        } => Some((
            VulkanComponentBatchSignalKey::Activation {
                component_id: component_id.clone(),
                signal_id: signal_id.clone(),
            },
            *signal_byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::Resident { .. } => None,
        VulkanMountedPlacedBoundDescriptorTarget::ModelInput { .. }
        | VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { .. } => None,
        VulkanMountedPlacedBoundDescriptorTarget::LocalEdgeInputBuffer { edge }
        | VulkanMountedPlacedBoundDescriptorTarget::LocalEdgeOutputBuffer { edge } => Some((
            VulkanComponentBatchSignalKey::LocalEdge(edge.edge.edge_index),
            edge.byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer { endpoint } => Some((
            VulkanComponentBatchSignalKey::IncomingEdge(endpoint.endpoint.edge_index),
            endpoint.byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::OutgoingEdgeBuffer { endpoint } => Some((
            VulkanComponentBatchSignalKey::OutgoingEdge(endpoint.endpoint.edge_index),
            endpoint.byte_capacity,
        )),
    };
    Ok(target)
}

fn component_batch_bindings<'a>(
    mounted: &'a VulkanMountedPlacedStreamCircuit,
    dispatch: &VulkanMountedPlacedBoundDispatch,
    signal_buffers: &'a [VulkanComponentBatchSignalBuffer],
    signal_buffer_indices: &BTreeMap<VulkanComponentBatchSignalKey, usize>,
    lane_index: Option<usize>,
    stream_control_buffer: Option<&'a VulkanResidentBuffer>,
) -> Result<Vec<VulkanResidentKernelBufferBinding<'a>>, VulkanResidentInProcessPlacedRuntimeError> {
    let mut bindings = Vec::with_capacity(
        dispatch.descriptors.len() + usize::from(stream_control_buffer.is_some()),
    );
    for descriptor in &dispatch.descriptors {
        let binding = u32::try_from(descriptor.binding).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "component batch descriptor binding exceeds u32".to_string(),
            ))
        })?;
        let access = match descriptor.usage {
            VulkanKernelDescriptorUsage::InputSignal
            | VulkanKernelDescriptorUsage::Parameter
            | VulkanKernelDescriptorUsage::DynamicResourceAddressTable
            | VulkanKernelDescriptorUsage::DynamicResourceParameterSlots
            | VulkanKernelDescriptorUsage::StateRead => VulkanResidentKernelBufferAccess::Read,
            VulkanKernelDescriptorUsage::OutputSignal | VulkanKernelDescriptorUsage::StateWrite => {
                VulkanResidentKernelBufferAccess::Write
            }
            VulkanKernelDescriptorUsage::StateView
            | VulkanKernelDescriptorUsage::SelectionTelemetry => {
                VulkanResidentKernelBufferAccess::ReadWrite
            }
        };
        if let Some((key, frame_byte_capacity)) =
            component_batch_signal_target_with_mounted(mounted, descriptor)?
        {
            let index = signal_buffer_indices.get(&key).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch descriptor {}.{} has no signal buffer {key:?}",
                    dispatch.component_id, dispatch.node_id
                )))
            })?;
            let allocation = &signal_buffers[*index];
            let (byte_offset, byte_len) = if let Some(lane_index) = lane_index {
                (
                    lane_index.checked_mul(frame_byte_capacity).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "component batch lane offset overflowed".to_string(),
                        ))
                    })?,
                    frame_byte_capacity,
                )
            } else {
                (0, allocation.buffer.byte_capacity())
            };
            bindings.push(
                VulkanResidentKernelBufferBinding::new(binding, &allocation.buffer, byte_len)
                    .with_byte_offset(byte_offset)
                    .with_access(access),
            );
            continue;
        }
        let (buffer, byte_len) = match &descriptor.target {
            VulkanMountedPlacedBoundDescriptorTarget::Resident { target } => match target {
                VulkanBoundDescriptorTarget::PermanentParameter { tensor, .. } => {
                    let parameter = mounted
                        .parameter_buffers
                        .parameter_buffer(tensor)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("component batch is missing parameter {tensor:?}"),
                            ))
                        })?;
                    (
                        parameter.buffer.as_ref(),
                        parameter.byte_capacity,
                    )
                }
                VulkanBoundDescriptorTarget::DynamicResourceAddressTable {
                    ..
                } => {
                    let resources =
                        mounted.dynamic_resource_buffers.as_ref().ok_or_else(
                            || {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                    VulkanError(
                                        "component batch has no dynamic resource buffers"
                                            .to_string(),
                                    ),
                                )
                            },
                        )?;
                    let buffer = resources.address_table();
                    (buffer, buffer.byte_capacity())
                }
                VulkanBoundDescriptorTarget::DynamicResourceParameterSlots {
                    component_id,
                    node_id,
                    selection_signal,
                    ..
                } => {
                    let resources =
                        mounted.dynamic_resource_buffers.as_ref().ok_or_else(
                            || {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                    VulkanError(
                                        "component batch has no dynamic resource buffers"
                                            .to_string(),
                                    ),
                                )
                            },
                        )?;
                    let buffer = resources
                        .parameter_slots(
                            component_id,
                            node_id,
                            selection_signal,
                        )
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                VulkanError(format!(
                                    "component batch has no dynamic resource parameter slots for {component_id}.{node_id} signal {selection_signal:?}"
                                )),
                            )
                        })?;
                    (buffer, buffer.byte_capacity())
                }
                VulkanBoundDescriptorTarget::StreamStateBuffer {
                    buffer_index,
                    byte_capacity,
                    ..
                }
                | VulkanBoundDescriptorTarget::StreamStateView {
                    buffer_index,
                    byte_capacity,
                    ..
                } => {
                    let state = mounted
                        .buffers
                        .state_buffers
                        .get(*buffer_index)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("component batch has no state buffer {buffer_index}"),
                            ))
                        })?;
                    (&state.buffer, *byte_capacity)
                }
                VulkanBoundDescriptorTarget::SelectionTelemetry {
                    buffer_index,
                    byte_capacity,
                    ..
                } => {
                    let telemetry = mounted
                        .buffers
                        .selection_telemetry_buffers
                        .get(*buffer_index)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "component batch has no selection telemetry buffer {buffer_index}"
                                ),
                            ))
                        })?;
                    (&telemetry.buffer, *byte_capacity)
                }
                VulkanBoundDescriptorTarget::BoundaryInput { .. }
                | VulkanBoundDescriptorTarget::BoundaryOutput { .. }
                | VulkanBoundDescriptorTarget::ActivationSlot { .. } => {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "component batch descriptor {}.{} has an unbound resident signal target",
                            dispatch.component_id, dispatch.node_id
                        )),
                    ));
                }
            },
            _ => unreachable!("signal targets were handled above"),
        };
        bindings.push(
            VulkanResidentKernelBufferBinding::new(binding, buffer, byte_len).with_access(access),
        );
    }
    if let Some(stream_control_buffer) = stream_control_buffer {
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(dispatch.descriptors.len()).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "component batch stream-control binding exceeds u32".to_string(),
                    ))
                })?,
                stream_control_buffer,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    Ok(bindings)
}

fn component_batch_signal_target_with_mounted(
    mounted: &VulkanMountedPlacedStreamCircuit,
    descriptor: &VulkanMountedPlacedBoundDescriptor,
) -> Result<Option<(VulkanComponentBatchSignalKey, usize)>, VulkanResidentInProcessPlacedRuntimeError> {
    match &descriptor.target {
        VulkanMountedPlacedBoundDescriptorTarget::ModelInput { signal_id } => {
            let allocation = mounted.boundary_io.input_buffer(signal_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no model input {signal_id:?}"
                )))
            })?;
            Ok(Some((
                VulkanComponentBatchSignalKey::ModelInput(signal_id.clone()),
                allocation.byte_capacity,
            )))
        }
        VulkanMountedPlacedBoundDescriptorTarget::ModelOutput { signal_id } => {
            let allocation = mounted
                .boundary_io
                .output_buffer(signal_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch has no model output {signal_id:?}"
                    )))
                })?;
            Ok(Some((
                VulkanComponentBatchSignalKey::ModelOutput(signal_id.clone()),
                allocation.byte_capacity,
            )))
        }
        _ => component_batch_signal_target(descriptor),
    }
}
