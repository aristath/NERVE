struct VulkanResidentComponentBatchSliceRunner {
    runtime_execution_identity: String,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
    signal_buffers: Vec<VulkanComponentBatchSignalBuffer>,
    signal_buffer_indices: BTreeMap<VulkanComponentBatchSignalKey, usize>,
    stream_control_buffers: Vec<VulkanResidentBuffer>,
    batch_control_buffers:
        BTreeMap<VulkanResidentComponentBatchControlPayload, VulkanResidentBuffer>,
    steps: Vec<VulkanComponentBatchDispatchStep>,
    execution_units: Vec<VulkanComponentBatchExecutionUnit>,
    demand_residency:
        BTreeMap<usize, VulkanDemandResidencyBatchSegment>,
    submission_template_catalog:
        RefCell<BTreeMap<(usize, usize), VulkanResidentQueueSubmissionTemplate>>,
    execution_shape_class_catalog: RefCell<BTreeMap<usize, String>>,
    sequence_catalog: RefCell<BTreeMap<(usize, usize), VulkanResidentKernelSequence>>,
    causal_state_snapshots: VulkanCausalStateSnapshotBank,
    quantum_calibrator: Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionUnit {
    LocalComponent {
        component_id: String,
        step_start: usize,
        step_end: usize,
    },
    DistributedDispatch {
        dispatch_index: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanComponentBatchDispatchSpan {
    component_id: String,
    dispatch_index: usize,
    step_start: usize,
    step_end: usize,
    distributed: bool,
}

fn finish_component_batch_local_execution_unit(
    execution_units: &mut Vec<VulkanComponentBatchExecutionUnit>,
    component_id: &str,
    step_start: usize,
    step_end: usize,
) {
    if step_start < step_end {
        execution_units.push(VulkanComponentBatchExecutionUnit::LocalComponent {
            component_id: component_id.to_string(),
            step_start,
            step_end,
        });
    }
}

fn component_batch_execution_units(
    dispatch_spans: &[VulkanComponentBatchDispatchSpan],
) -> Result<Vec<VulkanComponentBatchExecutionUnit>, VulkanError> {
    let mut execution_units = Vec::new();
    let mut current_component_id = None::<&str>;
    let mut local_step_start = 0usize;
    let mut expected_step_start = 0usize;
    let mut previous_dispatch_index = None;
    for span in dispatch_spans {
        if span.step_start != expected_step_start {
            return Err(VulkanError(format!(
                "component batch dispatch {} starts at step {}, expected {expected_step_start}",
                span.dispatch_index, span.step_start
            )));
        }
        if previous_dispatch_index.is_some_and(|previous| previous >= span.dispatch_index) {
            return Err(VulkanError(format!(
                "component batch dispatch indices are not strictly increasing at {}",
                span.dispatch_index
            )));
        }
        if span.distributed && span.step_end != span.step_start {
            return Err(VulkanError(format!(
                "distributed component batch dispatch {} owns local steps {}..{}",
                span.dispatch_index, span.step_start, span.step_end
            )));
        }
        if !span.distributed && span.step_end <= span.step_start {
            return Err(VulkanError(format!(
                "local component batch dispatch {} has no executable steps",
                span.dispatch_index
            )));
        }
        previous_dispatch_index = Some(span.dispatch_index);
        expected_step_start = span.step_end;
        if span.distributed {
            if let Some(component_id) = current_component_id.take() {
                finish_component_batch_local_execution_unit(
                    &mut execution_units,
                    component_id,
                    local_step_start,
                    span.step_start,
                );
            }
            execution_units.push(VulkanComponentBatchExecutionUnit::DistributedDispatch {
                dispatch_index: span.dispatch_index,
            });
            continue;
        }
        if current_component_id != Some(span.component_id.as_str()) {
            if let Some(component_id) = current_component_id {
                finish_component_batch_local_execution_unit(
                    &mut execution_units,
                    component_id,
                    local_step_start,
                    span.step_start,
                );
            }
            current_component_id = Some(&span.component_id);
            local_step_start = span.step_start;
        }
    }
    if let (Some(component_id), Some(last_span)) = (current_component_id, dispatch_spans.last()) {
        finish_component_batch_local_execution_unit(
            &mut execution_units,
            component_id,
            local_step_start,
            last_span.step_end,
        );
    }
    Ok(execution_units)
}

fn component_batch_execution_units_for_distributed_groups(
    dispatch_spans: &[VulkanComponentBatchDispatchSpan],
    distributed_group_leaders: &BTreeSet<usize>,
) -> Result<Vec<VulkanComponentBatchExecutionUnit>, VulkanError> {
    let mut execution_units = component_batch_execution_units(dispatch_spans)?;
    execution_units.retain(|unit| match unit {
        VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index } => {
            distributed_group_leaders.contains(dispatch_index)
        }
        VulkanComponentBatchExecutionUnit::LocalComponent { .. } => true,
    });
    Ok(execution_units)
}

fn component_batch_static_state_write_indices(
    mounted: &VulkanMountedPlacedStreamCircuit,
    dispatch: &VulkanMountedPlacedBoundDispatch,
) -> Result<Vec<usize>, VulkanResidentInProcessPlacedRuntimeError> {
    let mut indices = BTreeSet::new();
    for descriptor in &dispatch.descriptors {
        if !matches!(
            descriptor.usage,
            VulkanKernelDescriptorUsage::StateWrite | VulkanKernelDescriptorUsage::StateView
        ) {
            continue;
        }
        let VulkanMountedPlacedBoundDescriptorTarget::Resident {
            target:
                VulkanBoundDescriptorTarget::StreamStateBuffer { buffer_index, .. }
                | VulkanBoundDescriptorTarget::StreamStateView { buffer_index, .. },
        } = &descriptor.target
        else {
            continue;
        };
        let state = mounted
            .buffers
            .state_buffers
            .get(*buffer_index)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch state writer references absent buffer {buffer_index}"
                )))
            })?;
        if state.layout.static_byte_capacity > 0 {
            indices.insert(*buffer_index);
        }
    }
    Ok(indices.into_iter().collect())
}

impl VulkanResidentComponentBatchSliceRunner {
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device: &VulkanComputeDevice,
        slice: &VulkanResidentInProcessPlacedStreamProcessorDevice,
        runtime_execution_identity: &str,
        lane_mounteds: &[&VulkanMountedPlacedStreamCircuit],
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
        capture_causal_state_snapshots: bool,
        distributed_execution_plan: &VulkanDistributedExecutionPlan,
        quantum_calibrator: Rc<RefCell<RuntimeExecutionQuantumCalibrator>>,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if lane_capacity == 0 {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError("component batch lane capacity is zero".to_string()),
            ));
        }
        if lane_mounteds.len() != lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch has {lane_capacity} lanes but {} mounted stream states",
                    lane_mounteds.len()
                )),
            ));
        }
        let (signal_buffer_indices, signal_buffer_plan) =
            component_batch_signal_buffer_plan(&slice.mounted, &slice.mounted_bound.dispatches)?;
        let private_distributed_activations =
            distributed_component_batch_private_activation_specs(distributed_execution_plan);
        let mut shared_device_ids_by_buffer = BTreeMap::<usize, BTreeSet<String>>::new();
        for dispatch in distributed_execution_plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.owner_device_id == slice.device_id)
        {
            for activation in std::iter::once(&dispatch.input_activation)
                .chain(&dispatch.auxiliary_input_activations)
                .chain(std::iter::once(&dispatch.output_activation))
            {
                if private_distributed_activations.contains_key(
                    &distributed_component_batch_activation_key(
                        &dispatch.owner_device_id,
                        activation,
                    ),
                ) {
                    continue;
                }
                let key = distributed_component_batch_signal_key(
                    activation,
                    &signal_buffer_indices,
                )?;
                let buffer_index = *signal_buffer_indices.get(&key).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed component batch has no signal buffer for {key:?}"
                    )))
                })?;
                shared_device_ids_by_buffer
                    .entry(buffer_index)
                    .or_default()
                    .extend(dispatch.shards.iter().map(|shard| shard.device_id.clone()));
            }
        }
        let mut signal_buffers = Vec::<VulkanComponentBatchSignalBuffer>::new();
        for (buffer_index, allocation) in signal_buffer_plan.into_iter().enumerate() {
            let byte_capacity = allocation
                .frame_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "component batch signal capacity overflowed".to_string(),
                    ))
                })?;
            let shared_device_ids = shared_device_ids_by_buffer.get(&buffer_index);
            let (mut buffer, shared_device_buffers) =
                if let Some(shared_device_ids) = shared_device_ids {
                    if !shared_device_ids.contains(&slice.device_id) {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "distributed component batch buffer {buffer_index} omits owner {:?}",
                                slice.device_id
                            )),
                        ));
                    }
                    let peers = shared_device_ids
                        .iter()
                        .filter(|device_id| *device_id != &slice.device_id)
                        .map(|device_id| {
                            devices
                                .get(device_id)
                                .map(|device| device.as_ref())
                                .ok_or_else(|| {
                                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                                        device_id: device_id.clone(),
                                    }
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let shared = device
                        .create_shared_resident_buffers(&peers, byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let mut shared_device_buffers = BTreeMap::new();
                    let mut shared_buffers = shared.buffers.into_iter();
                    shared_device_buffers.insert(
                        slice.device_id.clone(),
                        shared_buffers
                            .next()
                            .expect("shared batch activation contains its owner"),
                    );
                    for (device_id, buffer) in shared_device_ids
                        .iter()
                        .filter(|device_id| *device_id != &slice.device_id)
                        .zip(shared_buffers)
                    {
                        shared_device_buffers.insert(device_id.clone(), buffer);
                    }
                    let owner_buffer = Arc::clone(
                        shared_device_buffers
                            .get(&slice.device_id)
                            .expect("validated distributed batch owner was imported"),
                    );
                    (owner_buffer, shared_device_buffers)
                } else {
                    let buffer = if allocation.host_visible {
                        // Cross-device edges are the one place where the batch must be
                        // host-addressable. The edge still moves once per device boundary,
                        // as one contiguous frame batch.
                        device.create_host_visible_resident_buffer(byte_capacity)
                    } else {
                        device.create_resident_buffer(byte_capacity)
                    }
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    (Arc::new(buffer), BTreeMap::new())
                };
            if allocation.host_visible {
                Arc::get_mut(&mut buffer)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "host-visible component batch edge buffer is unexpectedly shared"
                                .to_string(),
                        ))
                    })?
                    .persistently_map()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            signal_buffers.push(VulkanComponentBatchSignalBuffer {
                frame_byte_capacity: allocation.frame_byte_capacity,
                buffer,
                shared_device_buffers,
            });
        }

        let stream_control_buffers = (0..lane_capacity)
            .map(|_| {
                let mut buffer = device
                    .create_host_visible_resident_buffer(VULKAN_STREAM_CONTROL_BYTE_CAPACITY)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                buffer
                    .persistently_map()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                Ok::<_, VulkanResidentInProcessPlacedRuntimeError>(buffer)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch_control_buffers = [
            VulkanResidentComponentBatchControlPayload::Width,
            VulkanResidentComponentBatchControlPayload::WidthStateSnapshots,
            VulkanResidentComponentBatchControlPayload::WidthExpertStart,
            VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect,
            VulkanResidentComponentBatchControlPayload::Temporal,
        ]
        .into_iter()
        .map(|payload| {
            let mut buffer = device
                .create_host_visible_resident_buffer(payload.byte_count() as usize)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            buffer
                .persistently_map()
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            Ok::<_, VulkanResidentInProcessPlacedRuntimeError>((payload, buffer))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
        let mut causal_state_snapshots = VulkanCausalStateSnapshotBank::new(
            device,
            lane_capacity,
            capture_causal_state_snapshots,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let mut steps = Vec::new();
        let mut dispatch_spans = Vec::with_capacity(slice.mounted_bound.dispatches.len());
        for dispatch in &slice.mounted_bound.dispatches {
            let dispatch_step_start = steps.len();
            let static_state_write_indices =
                component_batch_static_state_write_indices(&slice.mounted, dispatch)?;
            for state_buffer_index in &static_state_write_indices {
                causal_state_snapshots.require_state_buffer(*state_buffer_index);
            }
            let commits_state = component_batch_descriptors_commit_state(
                dispatch
                    .descriptors
                    .iter()
                    .map(|descriptor| &descriptor.usage),
            );
            if distributed_execution_plan
                .dispatches
                .iter()
                .any(|distributed| {
                    distributed.owner_device_id == slice.device_id
                        && distributed.dispatch_index == dispatch.dispatch_index
                })
            {
                dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                    component_id: dispatch.component_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    step_start: dispatch_step_start,
                    step_end: dispatch_step_start,
                    distributed: true,
                });
                continue;
            }
            let batch_artifact = select_component_batch_kernel_artifact(
                &slice.package_slice.batch_kernels,
                &dispatch.component_id,
                &dispatch.node_id,
                execution_mode,
                lane_capacity,
            )
            .filter(|artifact| {
                component_batch_stages_replace_push_constants(
                    &artifact.stages,
                    &dispatch.push_constants,
                )
            })
            .filter(|artifact| {
                execution_mode == VulkanComponentBatchExecutionMode::CausalSequence
                    || artifact.batch_mode != VulkanResidentComponentKernelBatchMode::WeightShared
                    || (!dispatch.uses_stream_tick
                        && !dispatch.descriptors.iter().any(|descriptor| {
                            matches!(
                                descriptor.usage,
                                VulkanKernelDescriptorUsage::StateRead
                                    | VulkanKernelDescriptorUsage::StateWrite
                                    | VulkanKernelDescriptorUsage::StateView
                            )
                        }))
            });
            if let Some(batch_artifact) = batch_artifact {
                if batch_artifact.batch_mode == VulkanResidentComponentKernelBatchMode::CausalScan
                    && lane_capacity > batch_artifact.lane_tile_width
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "causal scan kernel {}.{} cannot execute {lane_capacity} lanes with tile width {}",
                            dispatch.component_id, dispatch.node_id, batch_artifact.lane_tile_width
                        )),
                    ));
                }
                let workgroup_count_y = match batch_artifact.batch_mode {
                    VulkanResidentComponentKernelBatchMode::WeightShared => u32::try_from(
                        lane_capacity
                            .checked_add(batch_artifact.lane_tile_width - 1)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "component batch workgroup count overflowed".to_string(),
                                ))
                            })?
                            / batch_artifact.lane_tile_width,
                    )
                    .map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "component batch workgroup count exceeds u32".to_string(),
                        ))
                    })?,
                    VulkanResidentComponentKernelBatchMode::CausalScan => 1,
                    VulkanResidentComponentKernelBatchMode::SerialLanes => {
                        unreachable!("serial-lane kernels do not have component batch artifacts")
                    }
                };
                for stage in &batch_artifact.stages {
                    let parent_bindings = component_batch_bindings(
                        &slice.mounted,
                        dispatch,
                        &signal_buffers,
                        &signal_buffer_indices,
                        None,
                        None,
                    )?;
                    let (binding, byte_count, payload) = stage.control.storage_buffer();
                    let mut bindings = component_batch_stage_bindings(
                        &parent_bindings,
                        &stage.descriptor_bindings,
                        binding,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    if let Some(snapshot_binding) = stage.state_snapshot_binding {
                        if bindings
                            .iter()
                            .any(|binding| binding.binding == snapshot_binding)
                        {
                            return Err(
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                    VulkanError(format!(
                                        "component batch state snapshot binding {snapshot_binding} collides in stage {}",
                                        stage.shader_path,
                                    )),
                                ),
                            );
                        }
                        let [state_buffer_index] = static_state_write_indices.as_slice() else {
                            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                VulkanError(format!(
                                    "component batch stage {} snapshots {} static state writers; expected one",
                                    stage.shader_path,
                                    static_state_write_indices.len(),
                                )),
                            ));
                        };
                        let snapshot_buffer = causal_state_snapshots
                            .binding_buffer(
                                device,
                                &slice.mounted.buffers,
                                *state_buffer_index,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                        bindings.push(
                            VulkanResidentKernelBufferBinding::new(
                                snapshot_binding,
                                snapshot_buffer,
                                snapshot_buffer.byte_capacity(),
                            )
                            .with_access(VulkanResidentKernelBufferAccess::Write),
                        );
                    }
                    let control_buffer = batch_control_buffers.get(&payload).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "component batch stage {} has no {:?} control buffer",
                                stage.shader_path, payload
                            ),
                        ))
                    })?;
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            binding,
                            control_buffer,
                            byte_count as usize,
                        )
                        .with_access(component_batch_control_buffer_access(stage.control)),
                    );
                    let resident = device
                        .create_resident_kernel_dispatch_2d_labeled(
                            &stage.spirv_words,
                            &bindings,
                            stage.workgroup_count_x,
                            workgroup_count_y,
                            stage.local_size_x,
                            0,
                            Some(vulkan_dispatch_semantic_label(dispatch, Some("batch"))),
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    steps.push(VulkanComponentBatchDispatchStep {
                        dispatch: resident,
                        push_constants: Vec::new(),
                        lane_index: None,
                        commits_state,
                    });
                }
                dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                    component_id: dispatch.component_id.clone(),
                    dispatch_index: dispatch.dispatch_index,
                    step_start: dispatch_step_start,
                    step_end: steps.len(),
                    distributed: false,
                });
                continue;
            }

            let artifact = slice
                .package_slice
                .loaded_manifest
                .artifact(&dispatch.reusable_family_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch scalar kernel {}.{} has no loaded artifact",
                        dispatch.component_id, dispatch.node_id
                    )))
                })?;
            for (lane_index, stream_control_buffer) in stream_control_buffers.iter().enumerate() {
                let bindings = component_batch_bindings(
                    lane_mounteds[lane_index],
                    dispatch,
                    &signal_buffers,
                    &signal_buffer_indices,
                    Some(lane_index),
                    dispatch.uses_stream_tick.then_some(stream_control_buffer),
                )?;
                let resident = device
                    .create_resident_kernel_dispatch_labeled(
                        &artifact.words,
                        &bindings,
                        artifact.artifact.workgroup_count_x,
                        artifact.artifact.local_size_x,
                        push_constant_byte_count(&dispatch.push_constants).map_err(|error| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!("invalid component batch push constants: {error}"),
                            ))
                        })?,
                        Some(vulkan_dispatch_semantic_label(
                            dispatch,
                            Some(&format!("lane={lane_index}")),
                        )),
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                steps.push(VulkanComponentBatchDispatchStep {
                    dispatch: resident,
                    push_constants: dispatch.push_constants.clone(),
                    lane_index: Some(lane_index),
                    commits_state,
                });
            }
            dispatch_spans.push(VulkanComponentBatchDispatchSpan {
                component_id: dispatch.component_id.clone(),
                dispatch_index: dispatch.dispatch_index,
                step_start: dispatch_step_start,
                step_end: steps.len(),
                distributed: false,
            });
        }
        causal_state_snapshots
            .mount_commit_batches(device, &slice.mounted.buffers)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let distributed_group_leaders = distributed_execution_plan
            .dispatch_groups
            .iter()
            .filter(|group| group.owner_device_id == slice.device_id)
            .map(|group| group.leader().dispatch_index)
            .collect::<BTreeSet<_>>();
        let execution_units = component_batch_execution_units_for_distributed_groups(
            &dispatch_spans,
            &distributed_group_leaders,
        )
        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let demand_residency = match &slice.demand_residency_context {
            Some(context) => {
                let mut segments = BTreeMap::new();
                for (unit_index, unit) in execution_units.iter().enumerate() {
                    let VulkanComponentBatchExecutionUnit::LocalComponent {
                        step_start,
                        step_end,
                        ..
                    } = unit
                    else {
                        continue;
                    };
                    if let Some(segment) =
                        VulkanDemandResidencyBatchSegment::from_slice_steps(
                            &slice.mounted,
                            slice.package_slice.physical_residency_schedule(),
                            &dispatch_spans,
                            &signal_buffers,
                            &signal_buffer_indices,
                            *step_start,
                            *step_end,
                            lane_capacity,
                            context.clone(),
                        )?
                    {
                        segments.insert(unit_index, segment);
                    }
                }
                segments
            }
            None => BTreeMap::new(),
        };

        Ok(Self {
            runtime_execution_identity: runtime_execution_identity.to_string(),
            execution_mode,
            lane_capacity,
            signal_buffers,
            signal_buffer_indices,
            stream_control_buffers,
            batch_control_buffers,
            steps,
            execution_units,
            demand_residency,
            submission_template_catalog: RefCell::new(BTreeMap::new()),
            execution_shape_class_catalog: RefCell::new(BTreeMap::new()),
            sequence_catalog: RefCell::new(BTreeMap::new()),
            causal_state_snapshots,
            quantum_calibrator,
        })
    }

    fn execution_shape_class_id(&self, batch_width: usize) -> String {
        let execution_mode = match self.execution_mode {
            VulkanComponentBatchExecutionMode::IndependentStreams => "independent_streams",
            VulkanComponentBatchExecutionMode::CausalSequence => "causal_sequence",
        };
        format!(
            "{}:{execution_mode}:capacity={}:width={batch_width}",
            self.runtime_execution_identity, self.lane_capacity
        )
    }

    fn sequence_shape_keys(&self, batch_width: usize) -> Vec<usize> {
        self.execution_units
            .iter()
            .filter_map(|unit| match unit {
                VulkanComponentBatchExecutionUnit::LocalComponent {
                    step_start,
                    step_end,
                    ..
                } => Some(
                    if self.steps[*step_start..*step_end]
                        .iter()
                        .all(|step| step.lane_index.is_none())
                    {
                        self.lane_capacity
                    } else {
                        batch_width
                    },
                ),
                VulkanComponentBatchExecutionUnit::DistributedDispatch { .. } => None,
            })
            .collect()
    }

    fn commit_causal_state_prefix(
        &self,
        processed_tick_count: usize,
    ) -> Result<bool, VulkanResidentInProcessPlacedRuntimeError> {
        self.causal_state_snapshots
            .commit_prefix(processed_tick_count)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
    }

    fn can_commit_causal_state_prefix(&self) -> bool {
        self.causal_state_snapshots.can_commit_prefix()
    }

    fn ensure_sequence_shapes(
        &self,
        device: &VulkanComputeDevice,
        sequence_shape_keys: &[usize],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        for (sequence_index, shape_key) in sequence_shape_keys.iter().copied().enumerate() {
            if self
                .sequence_catalog
                .borrow()
                .contains_key(&(sequence_index, shape_key))
            {
                continue;
            }
            let sequence =
                device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            self.sequence_catalog
                .borrow_mut()
                .insert((sequence_index, shape_key), sequence);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_causal_sequence(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device: &VulkanComputeDevice,
        owner_device_id: &str,
        distributed_dispatches: &VulkanDistributedComponentBatchRunners,
        _mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let stream_ticks = consecutive_component_batch_stream_ticks(
            start_stream_tick,
            input_token_ids.len(),
        )?;
        self.run(
            device,
            input_token_ids,
            &stream_ticks,
            start_stream_tick,
            dynamic_state_capacity_activations,
            devices,
            owner_device_id,
            distributed_dispatches,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_independent_streams(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        device: &VulkanComputeDevice,
        owner_device_id: &str,
        distributed_dispatches: &VulkanDistributedComponentBatchRunners,
        _mounted: &VulkanMountedPlacedStreamCircuit,
        input_token_ids: &[u32],
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let start_stream_tick = stream_ticks.first().copied().ok_or(
            VulkanResidentInProcessPlacedRuntimeError::ZeroTickBudget,
        )?;
        self.run(
            device,
            input_token_ids,
            stream_ticks,
            start_stream_tick,
            dynamic_state_capacity_activations,
            devices,
            owner_device_id,
            distributed_dispatches,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &VulkanComputeDevice,
        input_token_ids: &[u32],
        stream_ticks: &[u64],
        start_stream_tick: u64,
        dynamic_state_capacity_activations: u32,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        owner_device_id: &str,
        distributed_dispatches: &VulkanDistributedComponentBatchRunners,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let batch_width = input_token_ids.len();
        if batch_width == 0 || batch_width > self.lane_capacity {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch tile width {} cannot execute {batch_width} lanes",
                    self.lane_capacity
                )),
            ));
        }
        if stream_ticks.len() != batch_width {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "component batch has {batch_width} lanes but {} stream ticks",
                    stream_ticks.len()
                )),
            ));
        }
        let lane_controls = component_batch_lane_stream_control_bytes_for_ticks(
            input_token_ids,
            stream_ticks,
            dynamic_state_capacity_activations,
        )?;
        for (stream_control_buffer, control_bytes) in
            self.stream_control_buffers.iter().zip(&lane_controls)
        {
            stream_control_buffer
                .write_bytes(control_bytes)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }

        let batch_width_u32 = u32::try_from(batch_width).map_err(|_| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "component batch width exceeds u32".to_string(),
            ))
        })?;
        let batch_control = component_batch_control_bytes(
            batch_width_u32,
            start_stream_tick,
            dynamic_state_capacity_activations,
        );
        for (payload, buffer) in &self.batch_control_buffers {
            buffer
                .write_bytes(&component_batch_control_payload_bytes(
                    *payload,
                    &batch_control,
                    self.causal_state_snapshots.enabled,
                ))
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        }
        let sequence_shape_keys = self.sequence_shape_keys(batch_width);
        self.ensure_sequence_shapes(device, &sequence_shape_keys)?;
        self.execution_shape_class_catalog
            .borrow_mut()
            .entry(batch_width)
            .or_insert_with(|| self.execution_shape_class_id(batch_width));
        let execution_shape_classes = self.execution_shape_class_catalog.borrow();
        let shape_class_id = execution_shape_classes
            .get(&batch_width)
            .expect("component batch execution shape was inserted");
        let shape_was_calibrated = self
            .quantum_calibrator
            .borrow()
            .has_observations(shape_class_id);
        let sequence_catalog = self.sequence_catalog.borrow();
        let mut sequence_index = 0usize;
        let mut local_group_index = 0usize;
        let mut measurements = Vec::new();
        let mut local_submission_batch = VulkanResidentQueueSubmissionBatch::new();
        let dependency_value = self
            .execution_units
            .iter()
            .any(|unit| {
                matches!(
                    unit,
                    VulkanComponentBatchExecutionUnit::DistributedDispatch { .. }
                )
            })
            .then(|| distributed_dispatches.reserve_dependency_value(owner_device_id))
            .transpose()?;
        let timeline_value_offset = dependency_value
            .map(|value| value - 1)
            .unwrap_or_default();
        let mut pending_owner_wait_points = Vec::new();
        for (unit_index, unit) in self.execution_units.iter().enumerate() {
            match unit {
                VulkanComponentBatchExecutionUnit::LocalComponent {
                    component_id,
                    step_start,
                    step_end,
                } => {
                    let flush_after_segment = self
                        .execution_units
                        .get(unit_index + 1)
                        .is_none_or(|next| {
                            !matches!(
                                next,
                                VulkanComponentBatchExecutionUnit::LocalComponent { .. }
                            )
                        });
                    let wait_points = if local_submission_batch.pending_submission_count() == 0 {
                        pending_owner_wait_points.as_slice()
                    } else {
                        &[]
                    };
                    let signal_points = if flush_after_segment {
                        match self.execution_units.get(unit_index + 1) {
                            Some(VulkanComponentBatchExecutionUnit::DistributedDispatch {
                                dispatch_index,
                            }) => distributed_dispatches.owner_ready_signal_points(
                                owner_device_id,
                                *dispatch_index,
                                1,
                            )?,
                            _ => Vec::new(),
                        }
                    } else {
                        Vec::new()
                    };
                    if let Some(demand_residency) =
                        self.demand_residency.get(&unit_index)
                    {
                        if local_submission_batch.pending_submission_count() > 0 {
                            self.submit_and_wait_local_batch(
                                std::mem::take(&mut local_submission_batch),
                                shape_class_id,
                                batch_width,
                                local_group_index,
                                shape_was_calibrated,
                                true,
                                timeline_value_offset,
                            )?
                            .into_iter()
                            .for_each(|measurement| {
                                measurements.push(measurement)
                            });
                            local_group_index += 1;
                        }
                        if !wait_points.is_empty() {
                            device
                                .submit_timeline_semaphore_bridge(
                                    wait_points,
                                    &[],
                                )
                                .map_err(
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                                )?;
                            pending_owner_wait_points.clear();
                        }
                        demand_residency.run(
                            device,
                            &self.steps,
                            batch_width,
                            stream_ticks,
                            dynamic_state_capacity_activations,
                        )?;
                        if !signal_points.is_empty() {
                            device
                                .submit_timeline_semaphore_bridge(
                                    &[],
                                    &signal_points,
                                )
                                .map_err(
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop,
                                )?;
                        }
                        sequence_index += 1;
                        continue;
                    }
                    self.run_segment(
                        device,
                        batch_width,
                        stream_ticks,
                        dynamic_state_capacity_activations,
                        component_id,
                        sequence_catalog
                            .get(&(sequence_index, sequence_shape_keys[sequence_index]))
                            .expect("component batch sequence shape was inserted"),
                        *step_start,
                        *step_end,
                        Some(&local_submission_batch),
                        wait_points,
                        &signal_points,
                        flush_after_segment,
                    )?;
                    if !wait_points.is_empty() {
                        pending_owner_wait_points.clear();
                    }
                    sequence_index += 1;
                    if flush_after_segment {
                        self.submit_and_wait_local_batch(
                            std::mem::replace(
                                &mut local_submission_batch,
                                VulkanResidentQueueSubmissionBatch::new(),
                            ),
                            shape_class_id,
                            batch_width,
                            local_group_index,
                            shape_was_calibrated,
                            true,
                            timeline_value_offset,
                        )?
                        .into_iter()
                        .for_each(|measurement| measurements.push(measurement));
                        local_group_index += 1;
                    }
                }
                VulkanComponentBatchExecutionUnit::DistributedDispatch { dispatch_index } => {
                    let consume_owner_ready_signal = matches!(
                        self.execution_units.get(unit_index.wrapping_sub(1)),
                        Some(VulkanComponentBatchExecutionUnit::LocalComponent { .. })
                    );
                    let prepare_owner_continuation = matches!(
                        self.execution_units.get(unit_index + 1),
                        Some(VulkanComponentBatchExecutionUnit::LocalComponent { .. })
                    );
                    distributed_dispatches.run_dispatch(
                        devices,
                        owner_device_id,
                        *dispatch_index,
                        &batch_control,
                        dependency_value.expect(
                            "a distributed execution unit reserved a dependency value",
                        ),
                        consume_owner_ready_signal,
                        prepare_owner_continuation,
                    )?;
                    if prepare_owner_continuation {
                        pending_owner_wait_points =
                            distributed_dispatches.owner_completion_wait_points(
                                owner_device_id,
                                *dispatch_index,
                                1,
                            )?;
                    }
                }
            }
        }
        if !pending_owner_wait_points.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "distributed component batch completed without consuming its owner continuation"
                        .to_string(),
                ),
            ));
        }
        self.submit_and_wait_local_batch(
            local_submission_batch,
            shape_class_id,
            batch_width,
            local_group_index,
            shape_was_calibrated,
            true,
            timeline_value_offset,
        )?
        .into_iter()
        .for_each(|measurement| measurements.push(measurement));
        debug_assert_eq!(sequence_index, sequence_shape_keys.len());
        let mut calibrator = self.quantum_calibrator.borrow_mut();
        for measurement in measurements {
            calibrator.observe_quantum(
                shape_class_id,
                measurement.cost,
                &measurement.kernel_families,
                measurement.duration_ns,
            );
            record_vulkan_execution_quantum_measurement(&measurement);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_segment<'a>(
        &self,
        device: &'a VulkanComputeDevice,
        batch_width: usize,
        stream_ticks: &[u64],
        dynamic_state_capacity_activations: u32,
        component_id: &str,
        sequence: &VulkanResidentKernelSequence,
        step_start: usize,
        step_end: usize,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_completion: bool,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let mut push_constant_storage = Vec::<Vec<u8>>::new();
        let mut active_steps = Vec::<&VulkanComponentBatchDispatchStep>::new();
        for step in &self.steps[step_start..step_end] {
            if step.lane_index.is_some_and(|lane| lane >= batch_width) {
                continue;
            }
            let push_constants = if let Some(lane_index) = step.lane_index {
                let stream_tick = *stream_ticks.get(lane_index).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "component batch has no stream tick for lane {lane_index}"
                    )))
                })?;
                stream_control_push_constant_bytes(
                    &step.push_constants,
                    VulkanMountedPlacedStreamControl {
                        stream_tick,
                        control_flags: 0,
                        dynamic_state_capacity_activations,
                    },
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "invalid component batch stream control: {error}"
                    )))
                })?
            } else {
                Vec::new()
            };
            push_constant_storage.push(push_constants);
            active_steps.push(step);
        }
        if active_steps.is_empty() {
            return Ok(());
        }
        let sequence_steps = active_steps
            .iter()
            .zip(&push_constant_storage)
            .map(|(step, push_constants)| {
                VulkanResidentKernelSequenceStep::new(&step.dispatch, push_constants)
            })
            .collect::<Vec<_>>();
        if let Some(submission_batch) = submission_batch {
            let execution_cost = RuntimeExecutionCost::new(
                active_steps.iter().fold(0u64, |total, step| {
                    total.saturating_add(step.dispatch.estimated_work_units())
                }),
                active_steps.iter().fold(0u64, |total, step| {
                    total.saturating_add(step.dispatch.estimated_memory_bytes())
                }),
                u64::try_from(active_steps.len()).unwrap_or(u64::MAX),
            );
            let mut execution_region = RuntimeExecutionRegion::new(
                format!("{component_id}:{step_start}..{step_end}"),
                component_id,
                execution_cost,
            );
            execution_region.kernel_families = active_steps
                .iter()
                .map(|step| step.dispatch.execution_family())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            execution_region.commits_state_after = active_steps
                .iter()
                .any(|step| step.commits_state);
            let record_result = if sequence.has_recorded_commands() {
                Ok(())
            } else {
                device.record_resident_kernel_sequence(sequence, &sequence_steps)
            };
            record_result
                .and_then(|_| {
                    submission_batch.enqueue_recorded_sequence_with_execution_region(
                        device,
                        sequence,
                        wait_points,
                        signal_points,
                        signal_completion,
                        Some(execution_region),
                    )
                })
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        } else {
            device
                .run_resident_kernel_sequence(sequence, &sequence_steps)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
        }
    }

    fn submit_and_wait_local_batch<'a>(
        &'a self,
        submission_batch: VulkanResidentQueueSubmissionBatch<'a>,
        shape_class_id: &str,
        batch_width: usize,
        local_group_index: usize,
        shape_was_calibrated: bool,
        reusable: bool,
        timeline_value_offset: u64,
    ) -> Result<
        Vec<VulkanResidentExecutionQuantumMeasurement>,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        if submission_batch.pending_submission_count() == 0 {
            return Ok(Vec::new());
        }
        let template_key = (batch_width, local_group_index);
        if reusable
            && let Some(template) = self
                .submission_template_catalog
                .borrow()
                .get(&template_key)
        {
            return template
                .submit_calibrated_quanta_and_wait(timeline_value_offset)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop);
        }
        let template = {
            let calibrator = self.quantum_calibrator.borrow();
            submission_batch
                .mount_calibrated(&calibrator, shape_class_id)
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?
        };
        let measurements = template
            .submit_calibrated_quanta_and_wait(timeline_value_offset)
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        if reusable && shape_was_calibrated {
            self.submission_template_catalog
                .borrow_mut()
                .insert(template_key, template);
        }
        Ok(measurements)
    }

    fn signal_buffer(
        &self,
        key: &VulkanComponentBatchSignalKey,
    ) -> Result<&VulkanComponentBatchSignalBuffer, VulkanResidentInProcessPlacedRuntimeError> {
        self.signal_buffer_indices
            .get(key)
            .and_then(|index| self.signal_buffers.get(*index))
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "component batch has no signal buffer {key:?}"
                )))
            })
    }

    fn distributed_signal_buffer(
        &self,
        key: &VulkanComponentBatchSignalKey,
        device_id: &str,
    ) -> Result<&Arc<VulkanResidentBuffer>, VulkanResidentInProcessPlacedRuntimeError> {
        let allocation = self.signal_buffer_indices.get(key).and_then(|index| {
            self.signal_buffers
                .get(*index)
                .and_then(|buffer| buffer.shared_device_buffers.get(device_id))
        });
        allocation.ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed component batch signal {key:?} is not imported on {device_id:?}"
            )))
        })
    }
}
