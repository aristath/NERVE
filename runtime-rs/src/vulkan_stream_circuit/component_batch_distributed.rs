struct VulkanResidentPlacedComponentBatchRunner {
    distributed_dispatches: VulkanDistributedComponentBatchRunners,
    lane_capacity: usize,
    device_ids: Vec<String>,
    slices: Vec<VulkanResidentComponentBatchSliceRunner>,
    edge_transfers: Vec<VulkanComponentBatchEdgeTransfer>,
    demand_pipeline_predicates: Option<Vec<Arc<VulkanResidentBuffer>>>,
}

struct VulkanDistributedComponentBatchRunners {
    dispatches: Vec<VulkanDistributedComponentBatchDispatchRunner>,
    dependency_clock: VulkanDistributedDependencyClock,
    _private_activation_buffers: BTreeMap<
        VulkanDistributedComponentBatchPrivateActivationBufferKey,
        Arc<VulkanResidentBuffer>,
    >,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedComponentBatchPrivateActivationKey {
    owner_device_id: String,
    component_id: String,
    signal_id: String,
    slot: usize,
    storage: VulkanDistributedActivationStorage,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanDistributedComponentBatchPrivateActivationBufferKey {
    activation: VulkanDistributedComponentBatchPrivateActivationKey,
    device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDistributedComponentBatchPrivateActivationSpec {
    signal_byte_capacity: usize,
    device_ids: BTreeSet<String>,
}

fn distributed_component_batch_activation_key(
    owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
) -> VulkanDistributedComponentBatchPrivateActivationKey {
    VulkanDistributedComponentBatchPrivateActivationKey {
        owner_device_id: owner_device_id.to_string(),
        component_id: activation.component_id.clone(),
        signal_id: activation.signal_id.clone(),
        slot: activation.slot,
        storage: activation.storage.clone(),
    }
}

fn distributed_component_batch_signal_key(
    activation: &VulkanDistributedActivationSlot,
    signal_buffer_indices: &BTreeMap<VulkanComponentBatchSignalKey, usize>,
) -> Result<VulkanComponentBatchSignalKey, VulkanResidentInProcessPlacedRuntimeError> {
    match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => {
            Ok(VulkanComponentBatchSignalKey::Activation {
                component_id: activation.component_id.clone(),
                signal_id: activation.signal_id.clone(),
            })
        }
        VulkanDistributedActivationStorage::BoundaryInput => Ok(
            VulkanComponentBatchSignalKey::ModelInput(activation.signal_id.clone()),
        ),
        VulkanDistributedActivationStorage::BoundaryOutput => Ok(
            VulkanComponentBatchSignalKey::ModelOutput(activation.signal_id.clone()),
        ),
        VulkanDistributedActivationStorage::Edge { edge_index, .. } => {
            let candidates = [
                VulkanComponentBatchSignalKey::LocalEdge(*edge_index),
                VulkanComponentBatchSignalKey::IncomingEdge(*edge_index),
                VulkanComponentBatchSignalKey::OutgoingEdge(*edge_index),
            ];
            let matching = candidates
                .into_iter()
                .filter(|key| signal_buffer_indices.contains_key(key))
                .collect::<Vec<_>>();
            let [key] = matching.as_slice() else {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "distributed component batch edge {edge_index} maps to {} signal buffers",
                        matching.len()
                    )),
                ));
            };
            Ok(key.clone())
        }
    }
}

fn distributed_component_batch_private_activation_specs(
    execution_plan: &VulkanDistributedExecutionPlan,
) -> BTreeMap<
    VulkanDistributedComponentBatchPrivateActivationKey,
    VulkanDistributedComponentBatchPrivateActivationSpec,
> {
    let mut specs = BTreeMap::new();
    for group in &execution_plan.execution_islands {
        for pair in group.dispatches.windows(2) {
            let producer = &pair[0];
            let consumer = &pair[1];
            if producer.output_activation.component_id
                != consumer.input_activation.component_id
                || producer.output_activation.slot != consumer.input_activation.slot
                || producer.output_activation.signal_id
                    != consumer.input_activation.signal_id
            {
                continue;
            }
            let key = distributed_component_batch_activation_key(
                &group.owner_device_id,
                &producer.output_activation,
            );
            let device_ids = producer
                .shards
                .iter()
                .map(|shard| shard.device_id.clone())
                .collect::<BTreeSet<_>>();
            specs
                .entry(key)
                .and_modify(
                    |existing: &mut VulkanDistributedComponentBatchPrivateActivationSpec| {
                        debug_assert_eq!(
                            existing.signal_byte_capacity,
                            producer.output_activation.signal_byte_capacity
                        );
                        existing.device_ids.extend(device_ids.iter().cloned());
                    },
                )
                .or_insert(VulkanDistributedComponentBatchPrivateActivationSpec {
                    signal_byte_capacity: producer.output_activation.signal_byte_capacity,
                    device_ids,
                });
        }
    }
    specs
}

struct VulkanDistributedComponentBatchDispatchRunner {
    planned: VulkanPhysicalExecutionIslandPlan,
    shards: Vec<VulkanDistributedComponentBatchShardRunner>,
    helper_synchronization: Vec<VulkanDistributedQueueSynchronization>,
}

struct VulkanDistributedComponentBatchShardRunner {
    device_id: String,
    dispatches: Vec<VulkanDistributedComponentBatchShardDispatch>,
    expert_start: u32,
    expert_count: u32,
    batch_control_buffer_sets:
        Vec<BTreeMap<VulkanResidentComponentBatchControlPayload, VulkanResidentBuffer>>,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
}

struct VulkanDistributedComponentBatchShardDispatch {
    dispatch: VulkanResidentKernelDispatch,
    control_buffer_set_index: usize,
    indirect_dispatch:
        Option<(VulkanResidentComponentBatchControlPayload, usize)>,
    dispatch_y_from_batch_width: bool,
}

impl VulkanDistributedComponentBatchShardRunner {
    fn append_group_member(
        &mut self,
        mut member: VulkanDistributedComponentBatchShardRunner,
    ) -> Result<(), VulkanError> {
        if self.device_id != member.device_id {
            return Err(VulkanError(format!(
                "distributed component batch group changes shard device from {:?} to {:?}",
                self.device_id, member.device_id
            )));
        }
        if self.expert_start != member.expert_start {
            return Err(VulkanError(format!(
                "distributed component batch group changes expert start from {} to {}",
                self.expert_start, member.expert_start
            )));
        }
        if self.expert_count != member.expert_count {
            return Err(VulkanError(format!(
                "distributed component batch group changes expert count from {} to {}",
                self.expert_count, member.expert_count
            )));
        }
        let control_buffer_set_offset = self.batch_control_buffer_sets.len();
        for dispatch in &mut member.dispatches {
            dispatch.control_buffer_set_index = dispatch
                .control_buffer_set_index
                .checked_add(control_buffer_set_offset)
                .ok_or_else(|| {
                    VulkanError(
                        "distributed component batch control-buffer index overflowed"
                            .to_string(),
                    )
                })?;
        }
        self.dispatches.extend(member.dispatches);
        // Every resident dispatch keeps descriptor references to the control
        // buffers with which it was created. Preserve each member's backing
        // buffers for exactly as long as the grouped sequence.
        self.batch_control_buffer_sets
            .extend(member.batch_control_buffer_sets);
        Ok(())
    }
}

impl VulkanDistributedComponentBatchRunners {
    fn resident_transient_bytes_by_device(&self) -> Result<BTreeMap<String, usize>, VulkanError> {
        let mut totals = BTreeMap::<String, usize>::new();
        for (key, buffer) in &self._private_activation_buffers {
            checked_add_device_bytes(&mut totals, &key.device_id, buffer.byte_capacity())?;
        }
        for dispatch in &self.dispatches {
            for shard in &dispatch.shards {
                for buffers in &shard.batch_control_buffer_sets {
                    for buffer in buffers.values() {
                        checked_add_device_bytes(
                            &mut totals,
                            &shard.device_id,
                            buffer.byte_capacity(),
                        )?;
                    }
                }
            }
        }
        Ok(totals)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
        batch_slices: &[VulkanResidentComponentBatchSliceRunner],
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        lane_capacity: usize,
        execution_mode: VulkanComponentBatchExecutionMode,
    ) -> Result<Self, VulkanResidentInProcessPlacedRuntimeError> {
        if let Some(planned) = execution_plan.dispatches.iter().find(|planned| {
            planned.distribution == VulkanDistributedDispatchDistribution::InputColumns
        }) {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "distributed component batch {}.{} requires partial-output collection",
                    planned.component_id, planned.node_id
                )),
            ));
        }
        let private_activation_specs =
            distributed_component_batch_private_activation_specs(execution_plan);
        let mut private_activation_buffers = BTreeMap::new();
        for (activation, spec) in &private_activation_specs {
            let byte_capacity = spec
                .signal_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "distributed private activation capacity overflowed".to_string(),
                    ))
                })?;
            for device_id in &spec.device_ids {
                let device = devices.get(device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: device_id.clone(),
                    }
                })?;
                let buffer = Arc::new(
                    device
                        .create_resident_buffer(byte_capacity)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
                );
                private_activation_buffers.insert(
                    VulkanDistributedComponentBatchPrivateActivationBufferKey {
                        activation: activation.clone(),
                        device_id: device_id.clone(),
                    },
                    buffer,
                );
            }
        }
        let mut dispatches = Vec::with_capacity(execution_plan.dispatches.len());
        for planned in &execution_plan.dispatches {
            for shard in &planned.shards {
                if !devices.contains_key(&shard.device_id) {
                    return Err(
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: shard.device_id.clone(),
                        },
                    );
                }
            }
            let owner_index = placed_slices
                .iter()
                .position(|slice| slice.device_id == planned.owner_device_id)
                .ok_or_else(
                    || VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: planned.owner_device_id.clone(),
                    },
                )?;
            let package_slice = &placed_slices[owner_index].package_slice;
            let batch_slice = &batch_slices[owner_index];
            let artifact = selected_distributed_component_batch_artifact(
                devices,
                package_slice,
                planned,
                execution_mode,
                lane_capacity,
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch {}.{} has no compatible batch artifact",
                    planned.component_id, planned.node_id
                )))
            })?;
            let input_key = distributed_component_batch_signal_key(
                &planned.input_activation,
                &batch_slice.signal_buffer_indices,
            )?;
            let auxiliary_input_keys = planned
                .auxiliary_input_activations
                .iter()
                .map(|activation| {
                    distributed_component_batch_signal_key(
                        activation,
                        &batch_slice.signal_buffer_indices,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_key = distributed_component_batch_signal_key(
                &planned.output_activation,
                &batch_slice.signal_buffer_indices,
            )?;
            let input_frame_capacity = batch_slice.signal_buffer(&input_key)?.frame_byte_capacity;
            let output_frame_capacity = batch_slice.signal_buffer(&output_key)?.frame_byte_capacity;
            if input_frame_capacity != planned.input_byte_capacity
                || output_frame_capacity != planned.output_byte_capacity
            {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "distributed component batch {}.{} signal capacities differ from its physical plan",
                        planned.component_id, planned.node_id
                    )),
                ));
            }
            for (activation, key) in planned
                .auxiliary_input_activations
                .iter()
                .zip(&auxiliary_input_keys)
            {
                if batch_slice.signal_buffer(key)?.frame_byte_capacity
                    != activation.signal_byte_capacity
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed component batch {}.{} auxiliary signal {} differs from its physical plan",
                            planned.component_id, planned.node_id, activation.signal_id
                        )),
                    ));
                }
            }
            let workgroup_count_y = u32::try_from(
                lane_capacity
                    .checked_add(artifact.lane_tile_width - 1)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed component batch lane count overflowed".to_string(),
                        ))
                    })?
                    / artifact.lane_tile_width,
            )
            .map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed component batch workgroup count exceeds u32".to_string(),
                ))
            })?;
            let mut shards = Vec::with_capacity(planned.shards.len());
            for shard in &planned.shards {
                if shard.auxiliary_input_ranges.len()
                    != planned.auxiliary_input_activations.len()
                {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed component batch {}.{} has {} auxiliary ranges for {} auxiliary inputs on {:?}",
                            planned.component_id,
                            planned.node_id,
                            shard.auxiliary_input_ranges.len(),
                            planned.auxiliary_input_activations.len(),
                            shard.device_id,
                        )),
                    ));
                }
                let device = devices.get(&shard.device_id).ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: shard.device_id.clone(),
                    }
                })?;
                let batch_control_payloads = artifact
                    .stages
                    .iter()
                    .map(|stage| stage.control.storage_buffer().2)
                    .collect::<BTreeSet<_>>();
                let batch_control_buffers = batch_control_payloads
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
                let input_private_key =
                    VulkanDistributedComponentBatchPrivateActivationBufferKey {
                        activation: distributed_component_batch_activation_key(
                            &planned.owner_device_id,
                            &planned.input_activation,
                        ),
                        device_id: shard.device_id.clone(),
                    };
                let output_private_key =
                    VulkanDistributedComponentBatchPrivateActivationBufferKey {
                        activation: distributed_component_batch_activation_key(
                            &planned.owner_device_id,
                            &planned.output_activation,
                        ),
                        device_id: shard.device_id.clone(),
                    };
                let input = if let Some(buffer) =
                    private_activation_buffers.get(&input_private_key)
                {
                    buffer
                } else {
                    batch_slice.distributed_signal_buffer(&input_key, &shard.device_id)?
                };
                let output = if let Some(buffer) =
                    private_activation_buffers.get(&output_private_key)
                {
                    buffer
                } else {
                    batch_slice.distributed_signal_buffer(&output_key, &shard.device_id)?
                };
                let (output_byte_offset, output_byte_capacity) = match planned.distribution {
                    VulkanDistributedDispatchDistribution::OutputRows => {
                        distributed_batch_shard_output_binding_range(
                            planned.output_byte_capacity,
                            lane_capacity,
                            shard.output_byte_offset,
                            shard.output_byte_count,
                        )?
                    }
                    VulkanDistributedDispatchDistribution::ExpertRange => (
                        0,
                        planned
                            .output_byte_capacity
                            .checked_mul(lane_capacity)
                            .ok_or_else(|| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed expert output capacity overflowed".to_string(),
                                ))
                            })?,
                    ),
                    VulkanDistributedDispatchDistribution::InputColumns => {
                        unreachable!("input-column component batches were rejected before allocation")
                    }
                };
                let mut bindings = Vec::with_capacity(
                    2 + planned.auxiliary_input_activations.len() + shard.parameters.len(),
                );
                let (input_byte_offset, input_byte_capacity) =
                    distributed_batch_shard_binding_range(
                        planned.input_byte_capacity,
                        lane_capacity,
                        &shard.input_range,
                    )?;
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        u32::try_from(planned.input_activation.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed component batch primary input binding exceeds u32"
                                    .to_string(),
                            ))
                        })?,
                        input,
                        input_byte_capacity,
                    )
                    .with_byte_offset(input_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
                for ((activation, key), range) in planned
                    .auxiliary_input_activations
                    .iter()
                    .zip(&auxiliary_input_keys)
                    .zip(&shard.auxiliary_input_ranges)
                {
                    let buffer = batch_slice.distributed_signal_buffer(key, &shard.device_id)?;
                    let (byte_offset, byte_capacity) =
                        distributed_batch_shard_binding_range(
                            activation.signal_byte_capacity,
                            lane_capacity,
                            range,
                        )?;
                    bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            u32::try_from(activation.binding).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed component batch auxiliary binding exceeds u32"
                                        .to_string(),
                                ))
                            })?,
                            buffer,
                            byte_capacity,
                        )
                        .with_byte_offset(byte_offset)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                bindings.push(
                    VulkanResidentKernelBufferBinding::new(
                        u32::try_from(planned.output_activation.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed component batch output binding exceeds u32".to_string(),
                            ))
                        })?,
                        output,
                        output_byte_capacity,
                    )
                    .with_byte_offset(output_byte_offset)
                    .with_access(VulkanResidentKernelBufferAccess::Write),
                );
                for fragment in &shard.parameters {
                    let allocation = parameter_buffers
                        .parameter_buffer(
                            &shard.device_id,
                            &fragment.tensor,
                            fragment.byte_offset,
                            fragment.byte_count,
                        )
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "distributed component batch {}.{} is missing tensor {:?} range {}..{} on {:?}",
                                    planned.component_id,
                                    planned.node_id,
                                    fragment.tensor,
                                    fragment.byte_offset,
                                    fragment.byte_offset + fragment.byte_count,
                                    shard.device_id
                                ),
                            ))
                        })?;
                    bindings.push(
                        allocation
                            .kernel_binding(u32::try_from(fragment.binding).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed component batch binding exceeds u32".to_string(),
                                ))
                            })?)
                            .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                let mut resident_dispatches = Vec::with_capacity(artifact.stages.len());
                for stage in &artifact.stages {
                    let (binding, byte_count, payload) = stage.control.storage_buffer();
                    let mut stage_bindings = component_batch_stage_bindings(
                        &bindings,
                        &stage.descriptor_bindings,
                        binding,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    let control_buffer = batch_control_buffers.get(&payload).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "distributed component batch stage {} has no {:?} control buffer",
                                stage.shader_path, payload
                            ),
                        ))
                    })?;
                    stage_bindings.push(
                        VulkanResidentKernelBufferBinding::new(
                            binding,
                            control_buffer,
                            byte_count as usize,
                        )
                        .with_access(component_batch_control_buffer_access(stage.control)),
                    );
                    let workgroup_count_x = match planned.distribution {
                        VulkanDistributedDispatchDistribution::ExpertRange => {
                            stage.workgroup_count_x
                        }
                        VulkanDistributedDispatchDistribution::OutputRows => {
                            let rows_per_workgroup = distributed_batch_rows_per_workgroup(
                                planned.output_rows,
                                stage.workgroup_count_x,
                                &planned.component_id,
                                &planned.node_id,
                            )?;
                            if !shard.row_start.is_multiple_of(rows_per_workgroup)
                                || !shard.row_count.is_multiple_of(rows_per_workgroup)
                            {
                                return Err(
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                        VulkanError(format!(
                                            "distributed component batch {}.{} shard rows {}..{} do not align to {rows_per_workgroup} rows per workgroup",
                                            planned.component_id,
                                            planned.node_id,
                                            shard.row_start,
                                            shard.row_start + shard.row_count
                                        )),
                                    ),
                                );
                            }
                            u32::try_from(shard.row_count / rows_per_workgroup).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed component batch shard workgroup count exceeds u32"
                                        .to_string(),
                                ))
                            })?
                        }
                        VulkanDistributedDispatchDistribution::InputColumns => {
                            unreachable!("input-column component batches were rejected before allocation")
                        }
                    };
                    // Expert range is a parameter-space offset, not an
                    // invocation-space offset. Batch kernels receive it through
                    // WidthExpertStart control; offsetting WorkGroupID.z as well
                    // changes their invocation domain and corrupts multi-lane
                    // helper execution.
                    let dispatch = device
                        .create_resident_kernel_dispatch_2d_labeled(
                            &stage.spirv_words,
                            &stage_bindings,
                            workgroup_count_x,
                            if stage.dispatch_y_from_batch_width {
                                u32::try_from(lane_capacity).map_err(|_| {
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                        VulkanError(
                                            "distributed component batch lane capacity exceeds u32"
                                                .to_string(),
                                        ),
                                    )
                                })?
                            } else {
                                workgroup_count_y
                            },
                            stage.local_size_x,
                            0,
                            Some(format!(
                                "component={} node={} distributed_batch=device:{} rows={}..{} expert_start={} distribution={:?}",
                                planned.component_id,
                                planned.node_id,
                                shard.device_id,
                                shard.row_start,
                                shard.row_start + shard.row_count,
                                shard.base_workgroup_z,
                                planned.distribution,
                            )),
                        )
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                    resident_dispatches.push(VulkanDistributedComponentBatchShardDispatch {
                        dispatch,
                        control_buffer_set_index: 0,
                        indirect_dispatch: stage
                            .indirect_dispatch_byte_offset
                            .map(|byte_offset| {
                                (
                                    payload,
                                    usize::try_from(byte_offset)
                                        .expect("u32 indirect offset fits usize"),
                                )
                            }),
                        dispatch_y_from_batch_width: stage
                            .dispatch_y_from_batch_width,
                    });
                }
                shards.push(VulkanDistributedComponentBatchShardRunner {
                    device_id: shard.device_id.clone(),
                    expert_start: u32::try_from(shard.row_start).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed component batch expert start exceeds u32".to_string(),
                        ))
                    })?,
                    expert_count: u32::try_from(shard.row_count).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed component batch expert count exceeds u32".to_string(),
                        ))
                    })?,
                    dispatches: resident_dispatches,
                    batch_control_buffer_sets: vec![batch_control_buffers],
                    sequence_catalog: RefCell::new(BTreeMap::new()),
                });
            }
            let planned_island =
                resolved_physical_execution_islands(
                    std::slice::from_ref(planned),
                    execution_plan.shared_activation_route,
                )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            error.to_string(),
                        ))
                    })?
                    .pop()
                    .expect("one distributed dispatch resolves to one physical island");
            dispatches.push(VulkanDistributedComponentBatchDispatchRunner {
                planned: planned_island,
                shards,
                helper_synchronization: Vec::new(),
            });
        }
        let mut dispatches_by_key = dispatches
            .into_iter()
            .map(|runner| {
                let leader = runner.planned.leader();
                (
                    (leader.owner_device_id.clone(), leader.dispatch_index),
                    runner,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut island_dispatches = Vec::with_capacity(execution_plan.execution_islands.len());
        for planned_island in &execution_plan.execution_islands {
            let mut members = planned_island
                .dispatches
                .iter()
                .map(|planned| {
                    dispatches_by_key
                        .remove(&(planned.owner_device_id.clone(), planned.dispatch_index))
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "distributed component batch has no physical dispatch {}.{}",
                                    planned.component_id, planned.node_id
                                ),
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut leader_runner = members.remove(0);
            for member in members {
                if member.shards.len() != leader_runner.shards.len() {
                    return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                        VulkanError(format!(
                            "distributed component batch group {}..{} changes shard count",
                            planned_island.leader().dispatch_index,
                            planned_island.tail().dispatch_index
                        )),
                    ));
                }
                for (leader_shard, member_shard) in
                    leader_runner.shards.iter_mut().zip(member.shards)
                {
                    leader_shard
                        .append_group_member(member_shard)
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                }
            }
            leader_runner.planned = planned_island.clone();
            let owner = devices
                .get(&planned_island.owner_device_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                        device_id: planned_island.owner_device_id.clone(),
                    }
                })?;
            leader_runner.helper_synchronization = leader_runner
                .shards
                .iter()
                .filter(|shard| shard.device_id != planned_island.owner_device_id)
                .map(|shard| {
                    let helper = devices.get(&shard.device_id).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                            device_id: shard.device_id.clone(),
                        }
                    })?;
                    VulkanDistributedQueueSynchronization::new(
                        owner,
                        helper,
                        &planned_island.owner_device_id,
                        &shard.device_id,
                        &format!(
                            "distributed component batch {}.{}",
                            planned_island.leader().component_id,
                            planned_island.leader().node_id
                        ),
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()?;
            island_dispatches.push(leader_runner);
        }
        if !dispatches_by_key.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "distributed component batch left ungrouped physical dispatches".to_string(),
                ),
            ));
        }
        Ok(Self {
            dispatches: island_dispatches,
            dependency_clock: VulkanDistributedDependencyClock::new(),
            _private_activation_buffers: private_activation_buffers,
        })
    }

    fn dispatch(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<
        &VulkanDistributedComponentBatchDispatchRunner,
        VulkanResidentInProcessPlacedRuntimeError,
    > {
        self.dispatches
            .iter()
            .find(|dispatch| {
                dispatch.planned.owner_device_id == owner_device_id
                    && dispatch.planned.leader().dispatch_index == dispatch_index
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch has no dispatch {dispatch_index} owned by {owner_device_id:?}"
                )))
            })
    }

    fn reserve_dependency_value(
        &self,
        owner_device_id: &str,
    ) -> Result<u64, VulkanResidentInProcessPlacedRuntimeError> {
        self.dependency_clock
            .reserve(owner_device_id, usize::MAX)
            .map_err(|error| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    error.to_string(),
                ))
            })
    }

    fn owner_ready_signal_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanResidentInProcessPlacedRuntimeError>
    {
        Ok(self
            .dispatch(owner_device_id, dispatch_index)?
            .helper_synchronization
            .iter()
            .map(|synchronization| synchronization.owner_ready(dependency_value))
            .collect())
    }

    fn owner_completion_wait_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanResidentInProcessPlacedRuntimeError>
    {
        Ok(self
            .dispatch(owner_device_id, dispatch_index)?
            .helper_synchronization
            .iter()
            .map(|synchronization| synchronization.owner_done(dependency_value))
            .collect())
    }

    fn run_dispatch(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        owner_device_id: &str,
        dispatch_index: usize,
        batch_control: &[u8],
        dependency_value: u64,
        consume_owner_ready_signal: bool,
        prepare_owner_continuation: bool,
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index)?;
        let batch_width = batch_control
            .get(..std::mem::size_of::<u32>())
            .and_then(|bytes| bytes.try_into().ok())
            .map(u32::from_le_bytes)
            .and_then(|width| usize::try_from(width).ok())
            .filter(|width| *width > 0)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed component batch control has no positive batch width".to_string(),
                ))
            })?;
        for shard in &dispatch.shards {
            let device = devices.get(&shard.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: shard.device_id.clone(),
                }
            })?;
            let batch_control: &[u8; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize] =
                batch_control.try_into().map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed component batch control has {} bytes",
                        batch_control.len()
                    )))
                })?;
            for control_buffers in &shard.batch_control_buffer_sets {
                for (payload, control_buffer) in control_buffers {
                    control_buffer
                        .write_bytes(&distributed_component_batch_control_payload_bytes(
                            *payload,
                            batch_control,
                            shard.expert_start,
                            shard.expert_count,
                        ))
                        .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                }
            }
            if !shard.sequence_catalog.borrow().contains_key(&batch_width) {
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                shard
                    .sequence_catalog
                    .borrow_mut()
                    .insert(batch_width, sequence);
            }
            let steps = shard
                .dispatches
                .iter()
                .map(|resident| {
                    let Some((payload, byte_offset)) = resident.indirect_dispatch else {
                        return if resident.dispatch_y_from_batch_width {
                            VulkanResidentKernelSequenceStep::new_direct_with_workgroup_count(
                                &resident.dispatch,
                                &[],
                                resident.dispatch.workgroup_count_x(),
                                u32::try_from(batch_width).map_err(|_| {
                                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                                        VulkanError(
                                            "distributed component batch width exceeds u32"
                                                .to_string(),
                                        ),
                                    )
                                })?,
                            )
                            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                        } else {
                            Ok(VulkanResidentKernelSequenceStep::new(
                                &resident.dispatch,
                                &[],
                            ))
                        };
                    };
                    let control_buffer = shard
                        .batch_control_buffer_sets
                        .get(resident.control_buffer_set_index)
                        .and_then(|buffers| buffers.get(&payload))
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                format!(
                                    "distributed indirect dispatch has no {:?} control buffer set {}",
                                    payload, resident.control_buffer_set_index
                                ),
                            ))
                        })?;
                    VulkanResidentKernelSequenceStep::new_indirect(
                        &resident.dispatch,
                        &[],
                        control_buffer,
                        byte_offset,
                    )
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let catalog = shard.sequence_catalog.borrow();
            let sequence = catalog.get(&batch_width).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch sequence for width {batch_width} was not retained on {:?}",
                    shard.device_id
                )))
            })?;
            if !sequence.has_recorded_commands() {
                device
                    .record_resident_kernel_sequence(sequence, &steps)
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
        }
        let mut submitted =
            Vec::<(&VulkanComputeDevice, &VulkanResidentKernelSequence)>::with_capacity(
                dispatch.shards.len(),
            );
        let sequence_catalogs = dispatch
            .shards
            .iter()
            .map(|shard| shard.sequence_catalog.borrow())
            .collect::<Vec<_>>();
        for (shard, sequence_catalog) in dispatch.shards.iter().zip(&sequence_catalogs) {
            let device = devices.get(&shard.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                    device_id: shard.device_id.clone(),
                }
            })?;
            let sequence = sequence_catalog.get(&batch_width).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch sequence for width {batch_width} is missing on {:?}",
                    shard.device_id
                )))
            })?;
            let synchronization = dispatch
                .helper_synchronization
                .iter()
                .find(|synchronization| synchronization.device_id == shard.device_id);
            let wait_points = synchronization
                .filter(|_| consume_owner_ready_signal)
                .map(|synchronization| {
                    vec![synchronization.helper_ready(dependency_value)]
                })
                .unwrap_or_default();
            let signal_points = synchronization
                .filter(|_| prepare_owner_continuation)
                .map(|synchronization| {
                    vec![synchronization.helper_done(dependency_value)]
                })
                .unwrap_or_default();
            if let Err(error) = device
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    sequence,
                    &wait_points,
                    &signal_points,
                )
            {
                for (submitted_device, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    error,
                ));
            }
            submitted.push((device.as_ref(), sequence));
        }
        if !prepare_owner_continuation {
            let mut first_error = None;
            for (device, sequence) in submitted {
                if let Err(error) = device.wait_resident_kernel_sequence(sequence)
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
            if let Some(error) = first_error {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    error,
                ));
            }
        }
        Ok(())
    }
}

fn checked_add_device_bytes(
    totals: &mut BTreeMap<String, usize>,
    device_id: &str,
    bytes: usize,
) -> Result<(), VulkanError> {
    let total = totals.entry(device_id.to_string()).or_default();
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| VulkanError("resident transient byte accounting overflowed".to_string()))?;
    Ok(())
}

fn selected_distributed_component_batch_artifact<'a>(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    package_slice: &'a VulkanResidentModelPackageDeviceSlice,
    planned: &VulkanDistributedDispatchPlan,
    execution_mode: VulkanComponentBatchExecutionMode,
    lane_capacity: usize,
) -> Option<&'a VulkanResidentComponentBatchKernelArtifact> {
    select_component_batch_kernel_artifact_where(
        &package_slice.batch_kernels,
        &planned.component_id,
        &planned.node_id,
        execution_mode,
        lane_capacity,
        |artifact| {
            artifact.batch_mode == VulkanResidentComponentKernelBatchMode::WeightShared
                && planned.shards.iter().all(|shard| {
                    devices
                        .get(&shard.device_id)
                        .is_some_and(|device| batch_kernel_artifact_is_supported(device, artifact))
                })
        },
    )
}

fn distributed_component_batch_control_payload_bytes(
    payload: VulkanResidentComponentBatchControlPayload,
    control: &[u8; VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY as usize],
    expert_start: u32,
    expert_count: u32,
) -> Vec<u8> {
    let mut bytes = component_batch_control_payload_bytes(payload, control, false);
    if matches!(
        payload,
        VulkanResidentComponentBatchControlPayload::WidthExpertStart
            | VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect
    ) {
        bytes[VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize
            ..2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize]
            .copy_from_slice(&expert_start.to_le_bytes());
    }
    if payload == VulkanResidentComponentBatchControlPayload::WidthExpertRangeIndirect {
        bytes[2 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize
            ..3 * VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY as usize]
            .copy_from_slice(&expert_count.to_le_bytes());
    }
    bytes
}

fn distributed_batch_shard_output_binding_range(
    frame_byte_capacity: usize,
    lane_capacity: usize,
    shard_byte_offset: usize,
    shard_byte_count: usize,
) -> Result<(usize, usize), VulkanResidentInProcessPlacedRuntimeError> {
    if lane_capacity == 0 || shard_byte_count == 0 {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError("distributed component batch output range is empty".to_string()),
        ));
    }
    let shard_end = shard_byte_offset
        .checked_add(shard_byte_count)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed component batch shard output end overflowed".to_string(),
            ))
        })?;
    if shard_end > frame_byte_capacity {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed component batch shard output range {shard_byte_offset}..{shard_end} exceeds frame capacity {frame_byte_capacity}"
            )),
        ));
    }
    let preceding_lanes = frame_byte_capacity
        .checked_mul(lane_capacity - 1)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed component batch output lane span overflowed".to_string(),
            ))
        })?;
    let binding_byte_capacity = preceding_lanes
        .checked_add(shard_byte_count)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed component batch output binding span overflowed".to_string(),
            ))
        })?;
    Ok((shard_byte_offset, binding_byte_capacity))
}

fn distributed_batch_shard_binding_range(
    frame_byte_capacity: usize,
    lane_capacity: usize,
    range: &VulkanDistributedActivationRange,
) -> Result<(usize, usize), VulkanResidentInProcessPlacedRuntimeError> {
    distributed_batch_shard_output_binding_range(
        frame_byte_capacity,
        lane_capacity,
        range.byte_offset,
        range.byte_count,
    )
}

fn distributed_batch_rows_per_workgroup(
    output_rows: usize,
    full_workgroup_count_x: u32,
    component_id: &str,
    node_id: &str,
) -> Result<usize, VulkanResidentInProcessPlacedRuntimeError> {
    let full_workgroup_count_x = usize::try_from(full_workgroup_count_x).map_err(|_| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
            "distributed component batch workgroup count exceeds usize".to_string(),
        ))
    })?;
    if full_workgroup_count_x == 0 || !output_rows.is_multiple_of(full_workgroup_count_x) {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed component batch {component_id}.{node_id} cannot partition {output_rows} rows across {full_workgroup_count_x} workgroups"
            )),
        ));
    }
    Ok(output_rows / full_workgroup_count_x)
}
