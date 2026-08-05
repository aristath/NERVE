impl VulkanResidentPlacedComponentBatchRunner {
    fn new_single_device(
        device: &VulkanComputeDevice,
        slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
        runtime_execution_identity: &str,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        Self::new_single_device_with_scope(
            device,
            slice,
            runtime_execution_identity,
            lane_capacity,
            execution_mode,
            VulkanComponentBatchExecutionScope::all(),
        )
    }

    fn new_single_device_for_nodes(
        device: &VulkanComputeDevice,
        slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
        runtime_execution_identity: &str,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        node_ids_by_component: BTreeMap<String, BTreeSet<String>>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let execution_scope = VulkanComponentBatchExecutionScope::nodes(node_ids_by_component)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        Self::new_single_device_with_scope(
            device,
            slice,
            runtime_execution_identity,
            lane_capacity,
            execution_mode,
            execution_scope,
        )
    }

    fn new_single_device_with_scope(
        device: &VulkanComputeDevice,
        slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
        runtime_execution_identity: &str,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        execution_scope: VulkanComponentBatchExecutionScope,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        let devices = BTreeMap::new();
        let distributed_execution_plan = VulkanDistributedExecutionPlan {
            device_ids: Vec::new(),
            storage_buffer_offset_alignment: device.min_storage_buffer_offset_alignment(),
            dispatches: Vec::new(),
            dispatch_groups: Vec::new(),
            shared_input_byte_capacity: 0,
            shared_output_byte_capacity: 0,
            distributed_parameter_byte_count: 0,
        };
        let distributed_parameter_buffers = VulkanDistributedParameterBuffers {
            plan: VulkanDistributedParameterAllocationPlan {
                allocations: Vec::new(),
                allocation_count: 0,
                tensor_count: 0,
                total_byte_capacity: 0,
            },
            buffers: Vec::new(),
            total_byte_capacity: 0,
        };
        let lane_mounteds = vec![&slice.mounted; lane_capacity];
        let pipeline_continuation_predicate = if execution_mode
            == VulkanComponentBatchExecutionMode::CausalSequence
            && slice.demand_residency_context.is_some()
        {
            let predicate = Arc::new(
                device
                    .create_conditional_resident_buffer(size_of::<u32>())
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
            );
            predicate
                .write_bytes(&1u32.to_le_bytes())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            Some(predicate)
        } else {
            None
        };
        let batch_slice = VulkanResidentComponentBatchSliceRunner::new(
            &devices,
            device,
            slice,
            runtime_execution_identity,
            &lane_mounteds,
            lane_capacity,
            execution_mode,
            &execution_scope,
            &BTreeSet::new(),
            false,
            &distributed_execution_plan,
            pipeline_continuation_predicate.clone(),
            Rc::new(RefCell::new(RuntimeExecutionQuantumCalibrator::default())),
        )?;
        let slices = vec![batch_slice];
        let distributed_dispatches = VulkanDistributedComponentBatchRunners::new(
            &devices,
            std::slice::from_ref(slice),
            &slices,
            &distributed_execution_plan,
            &distributed_parameter_buffers,
            lane_capacity,
            execution_mode,
        )?;
        Ok(Self {
            distributed_dispatches,
            lane_capacity,
            device_ids: vec![slice.device_id.clone()],
            slices,
            edge_transfers: Vec::new(),
            demand_pipeline_predicates: pipeline_continuation_predicate.map(|buffer| vec![buffer]),
        })
    }

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
        retained_signal_keys_by_device:
            &BTreeMap<String, BTreeSet<VulkanComponentBatchSignalKey>>,
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
            retained_signal_keys_by_device,
            capture_causal_state_snapshots,
            distributed_execution_plan,
            distributed_parameter_buffers,
            &VulkanComponentBatchExecutionScope::all(),
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
            &BTreeMap::new(),
            false,
            &first.model.distributed_execution_plan,
            &first.model.distributed_parameter_buffers,
            &VulkanComponentBatchExecutionScope::all(),
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
        retained_signal_keys_by_device:
            &BTreeMap<String, BTreeSet<VulkanComponentBatchSignalKey>>,
        capture_causal_state_snapshots: bool,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        distributed_parameter_buffers: &VulkanDistributedParameterBuffers,
        execution_scope: &VulkanComponentBatchExecutionScope,
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
        execution_scope
            .validate_component_ids(placed_slices.iter().flat_map(|slice| {
                slice
                    .mounted_bound
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.component_id.as_str())
            }))
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        execution_scope
            .validate_dispatch_ids(placed_slices.iter().flat_map(|slice| {
                slice.mounted_bound.dispatches.iter().map(|dispatch| {
                    (dispatch.component_id.as_str(), dispatch.node_id.as_str())
                })
            }))
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let phase_execution_plan = execution_scope
            .filter_distributed_plan(distributed_execution_plan)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let demand_pipeline_predicates = if execution_mode
            == VulkanComponentBatchExecutionMode::CausalSequence
            && !placed_slices.is_empty()
            && placed_slices
                .iter()
                .all(|slice| slice.demand_residency_context.is_some())
            && phase_execution_plan.dispatches.is_empty()
        {
            let slice_devices = placed_slices
                .iter()
                .map(|slice| {
                    devices
                        .get(&slice.device_id)
                        .map(|device| device.as_ref())
                        .ok_or_else(|| VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: slice.device_id.clone(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (owner, peers) = slice_devices
                .split_first()
                .expect("non-empty placed slices contain an owner device");
            let shared = owner
                .create_shared_conditional_resident_buffers(peers, size_of::<u32>())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            if shared.buffers.len() != placed_slices.len() {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    format!(
                        "shared demand predicate produced {} device views for {} placed slices",
                        shared.buffers.len(),
                        placed_slices.len(),
                    ),
                )));
            }
            shared
                .buffers
                .first()
                .expect("shared demand predicate contains its owner view")
                .write_bytes(&1u32.to_le_bytes())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            Some(shared.buffers)
        } else {
            None
        };
        let no_retained_signal_keys = BTreeSet::new();
        let slices = placed_slices
            .iter()
            .zip(lane_mounteds_by_slice)
            .enumerate()
            .map(|(slice_index, slice)| {
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
                    execution_scope,
                    retained_signal_keys_by_device
                        .get(&slice.device_id)
                        .unwrap_or(&no_retained_signal_keys),
                    capture_causal_state_snapshots,
                    &phase_execution_plan,
                    demand_pipeline_predicates
                        .as_ref()
                        .and_then(|predicates| predicates.get(slice_index))
                        .cloned(),
                    quantum_calibrator,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let distributed_dispatches = VulkanDistributedComponentBatchRunners::new(
            devices,
            placed_slices,
            &slices,
            &phase_execution_plan,
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
                );
                let destination = slices[destination_device_index].signal_buffer(
                    &VulkanComponentBatchSignalKey::IncomingEdge(outgoing.endpoint.edge_index),
                );
                let (Ok(source), Ok(destination)) = (source, destination) else {
                    continue;
                };
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
            device_ids: placed_slices
                .iter()
                .map(|slice| slice.device_id.clone())
                .collect(),
            slices,
            edge_transfers,
            demand_pipeline_predicates,
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

    fn single_device_edge_signal_buffer(
        &self,
        edge_index: usize,
    ) -> Result<&VulkanComponentBatchSignalBuffer, VulkanResidentInProcessPlacedRuntimeError> {
        if self.slices.len() != 1 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "single-device component batch edge requested from a distributed runner"
                        .to_string(),
                ),
            ));
        }
        self.slice(0)?
            .signal_buffer(&VulkanComponentBatchSignalKey::LocalEdge(edge_index))
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

    fn has_deferred_demand_pipeline(&self) -> bool {
        self.demand_pipeline_predicates
            .as_ref()
            .is_some_and(|predicates| {
                !self.slices.is_empty()
                    && predicates.len() == self.slices.len()
                    && self
                        .slices
                        .iter()
                        .all(VulkanResidentComponentBatchSliceRunner::has_pipeline_deferred_demand_segment)
            })
    }

    fn reset_deferred_demand_pipeline_predicate(
        &self,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let predicate = self
            .demand_pipeline_predicates
            .as_ref()
            .and_then(|predicates| predicates.first())
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "component batch has no shared demand-pipeline predicate".to_string(),
                ))
            })?;
        predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn begin_deferred_demand_pipeline_executions<'a>(
        &'a self,
        devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
        pipeline: &[usize],
    ) -> Result<
        Vec<VulkanCompiledResourceExecutionGuard<'a>>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if !self.has_deferred_demand_pipeline() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("component batch has no deferred demand pipeline".to_string()),
            ));
        }
        // Pressure must be resolved on every participating device before any
        // execution read guard is held. A reclaimer needs the corresponding
        // store's execution barrier exclusively, and acquiring guards while
        // still preflighting later devices would permit a cross-store deadlock.
        for device_index in pipeline.iter().copied() {
            let device_id = self.device_ids.get(device_index).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no device identity for slice {device_index}"
                )))
            })?;
            let device = devices.get(device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: device_id.clone(),
                }
            })?;
            device
                .ensure_device_local_memory_headroom()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        pipeline
            .iter()
            .copied()
            .map(|device_index| {
                self.slice(device_index)?
                    .begin_pipeline_demand_execution_after_headroom_check()
            })
            .collect()
    }

    fn complete_deferred_demand_pipeline_submissions(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        pipeline: &[usize],
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let (&last_device_index, preceding) = pipeline.split_last().ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "deferred demand pipeline is empty".to_string(),
            ))
        })?;
        let last_slice = self.slice(last_device_index)?;
        let last_device_id = self.device_ids.get(last_device_index).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "component batch has no device identity for slice {last_device_index}"
            )))
        })?;
        let last_device = devices.get(last_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: last_device_id.clone(),
            }
        })?;
        last_slice.wait_pipeline_demand_submission(last_device, batch_width)?;
        for device_index in preceding {
            self.slice(*device_index)?
                .mark_pipeline_demand_submission_completed(batch_width)?;
        }
        Ok(())
    }

    fn mark_deferred_demand_pipeline_submitted(
        &self,
        pipeline: &[usize],
        batch_width: usize,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        for device_index in pipeline {
            self.slice(*device_index)?
                .mark_pipeline_demand_submission_submitted(batch_width)?;
        }
        Ok(())
    }

    fn resolve_deferred_demand_pipeline_submissions(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        pipeline: &[usize],
        batch_width: usize,
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
    ) -> Result<Option<usize>, VulkanResidentInProcessPlacedRuntimeError> {
        for (pipeline_position, device_index) in pipeline.iter().copied().enumerate() {
            let slice = self.slice(device_index)?;
            let device_id = self.device_ids.get(device_index).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no device identity for slice {device_index}"
                )))
            })?;
            let device = devices.get(device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: device_id.clone(),
                }
            })?;
            if slice.resolve_pipeline_demand_submission(
                device,
                batch_width,
                stream_ticks,
                dynamic_state_capacity_activations,
            )? {
                return Ok(Some(pipeline_position));
            }
        }
        Ok(None)
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

    fn enqueue_deferred_edge<'a>(
        &'a self,
        source_device_index: usize,
        destination_device_index: usize,
        edge_index: usize,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
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
            .enqueue_deferred(submission_batch)
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_deferred_demand_causal_sequence<'a>(
        &'a self,
        devices: &'a BTreeMap<String, Rc<VulkanComputeDevice>>,
        device_index: usize,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        submission_batch: &VulkanResidentQueueSubmissionBatch<'a>,
        signal_completion: bool,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let device = devices.get(owner_device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: owner_device_id.to_string(),
            }
        })?;
        self.slice(device_index)?.enqueue_pipeline_demand_submission(
            devices,
            device,
            owner_device_id,
            &self.distributed_dispatches,
            mounted,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            submission_batch,
            signal_completion,
        )
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

    fn run_causal_sequence_single_device(
        &self,
        device: &VulkanComputeDevice,
        owner_device_id: &str,
        mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let devices = BTreeMap::new();
        self.slice(0)?.run_causal_sequence(
            &devices,
            device,
            owner_device_id,
            &self.distributed_dispatches,
            mounted,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
            VulkanComponentBatchCompletionMode::Blocking,
        )
    }

    fn run_parallel_block_single_device(
        &self,
        device: &VulkanComputeDevice,
        owner_device_id: &str,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        self.slice(0)?.run_parallel_block(
            device,
            owner_device_id,
            &self.distributed_dispatches,
            input_token_ids,
            start_stream_tick,
            dynamic_state_capacity_activations,
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
        VulkanMountedPlacedBoundDescriptorTarget::LocalEdgeInputBuffer { edge } => Some((
            produced_port_signal_key(
                &edge.edge.source_component_id,
                &edge.edge.source_port_id,
            ),
            component_batch_edge_frame_byte_capacity(
                &edge.edge.connection,
                edge.byte_capacity,
            )?,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::IncomingEdgeBuffer { endpoint } => Some((
            VulkanComponentBatchSignalKey::IncomingEdge(endpoint.endpoint.edge_index),
            endpoint.byte_capacity,
        )),
        VulkanMountedPlacedBoundDescriptorTarget::ProducedPortBuffer { port } => {
            let (component_id, port_id) = port.source().ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "component batch encountered an empty produced port".to_string(),
                ))
            })?;
            Some((
                produced_port_signal_key(component_id, port_id),
                component_batch_produced_port_frame_byte_capacity(port)?,
            ))
        }
    };
    Ok(target)
}

fn component_batch_produced_port_frame_byte_capacity(
    port: &VulkanPlacedProducedPortBufferBinding,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    let mut frame_capacities = port
        .local_edges
        .iter()
        .map(|edge| {
            component_batch_edge_frame_byte_capacity(
                &edge.edge.connection,
                edge.byte_capacity,
            )
        })
        .chain(port.outgoing_endpoints.iter().map(|endpoint| {
            component_batch_edge_frame_byte_capacity(
                &endpoint.endpoint.connection,
                endpoint.byte_capacity,
            )
        }));
    let frame_byte_capacity = frame_capacities.next().ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
            "component batch encountered an empty produced port".to_string(),
        ))
    })??;
    for candidate in frame_capacities {
        if candidate? != frame_byte_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "component batch produced-port consumers require incompatible frame capacities"
                        .to_string(),
                ),
            ));
        }
    }
    Ok(frame_byte_capacity)
}

fn component_batch_edge_frame_byte_capacity(
    connection: &StreamCircuitConnection,
    byte_capacity: usize,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    match connection {
        StreamCircuitConnection::ParallelBlockScatter { width }
        | StreamCircuitConnection::ParallelBlockGather { width } => {
            if *width == 0 || byte_capacity % width != 0 {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "parallel-block edge capacity {byte_capacity} is not divisible by width {width}"
                    )),
                ));
            }
            Ok(byte_capacity / width)
        }
        _ => Ok(byte_capacity),
    }
}

fn component_batch_bindings<'a>(
    mounted: &'a VulkanMountedPlacedStreamCircuit,
    dispatch: &VulkanMountedPlacedBoundDispatch,
    signal_buffers: &'a [VulkanComponentBatchSignalBuffer],
    signal_buffer_indices: &BTreeMap<VulkanComponentBatchSignalKey, usize>,
    lane_index: Option<usize>,
    runtime_control_buffer: Option<(&'a VulkanResidentBuffer, usize)>,
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
        if let VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target:
                VulkanBoundDescriptorTarget::RuntimeControl {
                    runtime_source,
                    byte_capacity,
                },
        } = &descriptor.target
        {
            if runtime_source != "input_token_id" || *byte_capacity != size_of::<u32>() {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "component batch descriptor {}.{} requests unsupported runtime control {runtime_source:?} with {byte_capacity} bytes",
                        dispatch.component_id, dispatch.node_id
                    )),
                ));
            }
            let (buffer, binding_byte_capacity) = runtime_control_buffer.ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch descriptor {}.{} has no runtime token-id buffer",
                    dispatch.component_id, dispatch.node_id
                )))
            })?;
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    binding,
                    buffer,
                    binding_byte_capacity,
                )
                .with_access(access),
            );
            continue;
        }
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
        let (buffer, byte_offset, byte_len) = match &descriptor.target {
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
                        parameter.byte_offset,
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
                    (buffer, 0, buffer.byte_capacity())
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
                    (buffer, 0, buffer.byte_capacity())
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
                    (&state.buffer, 0, *byte_capacity)
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
                    (&telemetry.buffer, 0, *byte_capacity)
                }
                VulkanBoundDescriptorTarget::BoundaryInput { .. }
                | VulkanBoundDescriptorTarget::BoundaryOutput { .. }
                | VulkanBoundDescriptorTarget::RuntimeControl { .. }
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
            VulkanResidentKernelBufferBinding::new(binding, buffer, byte_len)
                .with_byte_offset(byte_offset)
                .with_access(access),
        );
    }
    if let Some(stream_control_buffer) = stream_control_buffer {
        let binding = dispatch.stream_control_binding.ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "component batch dispatch {}.{} received a stream-control buffer without a compiled binding",
                dispatch.component_id, dispatch.node_id
            )))
        })?;
        if usize::try_from(binding).ok() != Some(dispatch.descriptors.len()) {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch dispatch {}.{} compiled stream-control binding {} disagrees with runtime descriptor count {}",
                    dispatch.component_id,
                    dispatch.node_id,
                    binding,
                    dispatch.descriptors.len()
                )),
            ));
        }
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                binding,
                stream_control_buffer,
                VULKAN_STREAM_CONTROL_BYTE_CAPACITY,
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    Ok(bindings)
}

fn component_batch_runtime_token_id_bytes(
    token_ids: &[u32],
) -> Result<Vec<u8>, VulkanError> {
    let mut bytes = Vec::with_capacity(
        token_ids
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| VulkanError(
                "component batch runtime token-id payload overflowed".to_string(),
            ))?,
    );
    for token_id in token_ids {
        bytes.extend_from_slice(&token_id.to_le_bytes());
    }
    Ok(bytes)
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
