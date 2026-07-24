struct VulkanResidentPlacedComponentBatchRunner {
    distributed_dispatches: VulkanDistributedComponentBatchRunners,
    lane_capacity: usize,
    slices: Vec<VulkanResidentComponentBatchSliceRunner>,
    edge_transfers: Vec<VulkanComponentBatchEdgeTransfer>,
}

struct VulkanDistributedComponentBatchRunners {
    dispatches: Vec<VulkanDistributedComponentBatchDispatchRunner>,
}

struct VulkanDistributedComponentBatchDispatchRunner {
    planned: VulkanDistributedDispatchGroup,
    shards: Vec<VulkanDistributedComponentBatchShardRunner>,
}

struct VulkanDistributedComponentBatchShardRunner {
    device_id: String,
    dispatches: Vec<VulkanDistributedComponentBatchShardDispatch>,
    batch_control_buffers:
        BTreeMap<VulkanResidentComponentBatchControlPayload, VulkanResidentBuffer>,
    sequence_catalog: RefCell<BTreeMap<usize, VulkanResidentKernelSequence>>,
}

struct VulkanDistributedComponentBatchShardDispatch {
    dispatch: VulkanResidentKernelDispatch,
}

impl VulkanDistributedComponentBatchRunners {
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
            let artifact = select_component_batch_kernel_artifact_where(
                &package_slice.batch_kernels,
                &planned.component_id,
                &planned.node_id,
                execution_mode,
                lane_capacity,
                |artifact| {
                    planned.shards.iter().all(|shard| {
                        devices.get(&shard.device_id).is_some_and(|device| {
                            batch_kernel_artifact_is_supported(device, artifact)
                        })
                    })
                },
            )
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch {}.{} has no compatible batch artifact",
                    planned.component_id, planned.node_id
                )))
            })?;
            if artifact.batch_mode != VulkanResidentComponentKernelBatchMode::WeightShared {
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    VulkanError(format!(
                        "distributed component batch {}.{} requires a weight-shared artifact",
                        planned.component_id, planned.node_id
                    )),
                ));
            }
            let input_key = VulkanComponentBatchSignalKey::Activation {
                component_id: planned.input_activation.component_id.clone(),
                signal_id: planned.input_activation.signal_id.clone(),
            };
            let auxiliary_input_keys = planned
                .auxiliary_input_activations
                .iter()
                .map(|activation| VulkanComponentBatchSignalKey::Activation {
                    component_id: activation.component_id.clone(),
                    signal_id: activation.signal_id.clone(),
                })
                .collect::<Vec<_>>();
            let output_key = VulkanComponentBatchSignalKey::Activation {
                component_id: planned.output_activation.component_id.clone(),
                signal_id: planned.output_activation.signal_id.clone(),
            };
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
            let input_byte_capacity = planned
                .input_byte_capacity
                .checked_mul(lane_capacity)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "distributed component batch input capacity overflowed".to_string(),
                    ))
                })?;
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
                let input = batch_slice.distributed_signal_buffer(&input_key, &shard.device_id)?;
                let output =
                    batch_slice.distributed_signal_buffer(&output_key, &shard.device_id)?;
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
                };
                let mut bindings = Vec::with_capacity(
                    2 + planned.auxiliary_input_activations.len() + shard.parameters.len(),
                );
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
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                );
                for (activation, key) in planned
                    .auxiliary_input_activations
                    .iter()
                    .zip(&auxiliary_input_keys)
                {
                    let buffer = batch_slice.distributed_signal_buffer(key, &shard.device_id)?;
                    let byte_capacity = activation
                        .signal_byte_capacity
                        .checked_mul(lane_capacity)
                        .ok_or_else(|| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed component batch auxiliary input capacity overflowed"
                                    .to_string(),
                            ))
                        })?;
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
                        VulkanResidentKernelBufferBinding::new(
                            u32::try_from(fragment.binding).map_err(|_| {
                                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                    "distributed component batch binding exceeds u32".to_string(),
                                ))
                            })?,
                            &allocation.buffer,
                            fragment.byte_count,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    );
                }
                let mut resident_dispatches = Vec::with_capacity(artifact.stages.len());
                for stage in &artifact.stages {
                    let (binding, byte_count, payload) = stage.control.storage_buffer();
                    let control_buffer = batch_control_buffers.get(&payload).ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "distributed component batch stage {} has no {:?} control buffer",
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
                        .with_access(VulkanResidentKernelBufferAccess::Read),
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
                    };
                    let dispatch = device
                        .create_resident_kernel_dispatch_2d_with_base_z(
                            &stage.spirv_words,
                            &bindings,
                            workgroup_count_x,
                            workgroup_count_y,
                            shard.base_workgroup_z,
                            stage.local_size_x,
                            0,
                            Some(format!(
                                "component={} node={} distributed_batch=device:{} rows={}..{} base_z={} distribution={:?}",
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
                    resident_dispatches
                        .push(VulkanDistributedComponentBatchShardDispatch { dispatch });
                    bindings.pop();
                }
                shards.push(VulkanDistributedComponentBatchShardRunner {
                    device_id: shard.device_id.clone(),
                    dispatches: resident_dispatches,
                    batch_control_buffers,
                    sequence_catalog: RefCell::new(BTreeMap::new()),
                });
            }
            dispatches.push(VulkanDistributedComponentBatchDispatchRunner {
                planned: VulkanDistributedDispatchGroup {
                    owner_device_id: planned.owner_device_id.clone(),
                    dispatches: vec![planned.clone()],
                },
                shards,
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
        let mut grouped_dispatches = Vec::with_capacity(execution_plan.dispatch_groups.len());
        for planned_group in &execution_plan.dispatch_groups {
            let mut members = planned_group
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
                            planned_group.leader().dispatch_index,
                            planned_group.tail().dispatch_index
                        )),
                    ));
                }
                for (leader_shard, member_shard) in
                    leader_runner.shards.iter_mut().zip(member.shards)
                {
                    if leader_shard.device_id != member_shard.device_id {
                        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                            VulkanError(format!(
                                "distributed component batch group {}..{} changes shard device from {:?} to {:?}",
                                planned_group.leader().dispatch_index,
                                planned_group.tail().dispatch_index,
                                leader_shard.device_id,
                                member_shard.device_id
                            )),
                        ));
                    }
                    leader_shard.dispatches.extend(member_shard.dispatches);
                }
            }
            leader_runner.planned = planned_group.clone();
            grouped_dispatches.push(leader_runner);
        }
        if !dispatches_by_key.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(
                    "distributed component batch left ungrouped physical dispatches".to_string(),
                ),
            ));
        }
        Ok(Self {
            dispatches: grouped_dispatches,
        })
    }

    fn run_dispatch(
        &self,
        devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
        owner_device_id: &str,
        dispatch_index: usize,
        batch_control: &[u8],
    ) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
        let dispatch = self
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch.planned.owner_device_id == owner_device_id
                    && dispatch.planned.leader().dispatch_index == dispatch_index
            })
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch has no dispatch {dispatch_index} owned by {owner_device_id:?}"
                )))
        })?;
        let _batch_width = batch_control
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
            for (payload, control_buffer) in &shard.batch_control_buffers {
                control_buffer
                    .write_bytes(&component_batch_control_payload_bytes(
                        *payload,
                        batch_control,
                    ))
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
            }
            if !shard.sequence_catalog.borrow().contains_key(&0) {
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
                shard.sequence_catalog.borrow_mut().insert(0, sequence);
            }
            let steps = shard
                .dispatches
                .iter()
                .map(|resident| VulkanResidentKernelSequenceStep::new(&resident.dispatch, &[]))
                .collect::<Vec<_>>();
            let catalog = shard.sequence_catalog.borrow();
            let sequence = catalog
                .get(&0)
                .expect("distributed batch sequence shape was inserted");
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
            let sequence = sequence_catalog
                .get(&0)
                .expect("distributed batch sequence shape was inserted");
            if let Err(error) = device.submit_recorded_resident_kernel_sequence(sequence) {
                for (submitted_device, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                    error,
                ));
            }
            submitted.push((device.as_ref(), sequence));
        }
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
        Ok(())
    }
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
