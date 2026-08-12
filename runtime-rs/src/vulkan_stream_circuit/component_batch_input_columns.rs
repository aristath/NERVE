#[allow(clippy::too_many_arguments)]
fn create_distributed_input_column_component_batch_dispatch(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    batch_slices: &[VulkanResidentComponentBatchSliceRunner],
    planned: &VulkanDistributedDispatchPlan,
    parameter_buffers: &VulkanDistributedParameterBuffers,
    dynamic_resource_buffers: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    reduction_buffers: &[VulkanDistributedReductionBuffer],
    private_activation_buffers: &BTreeMap<
        VulkanDistributedComponentBatchPrivateActivationBufferKey,
        Arc<VulkanResidentBuffer>,
    >,
    lane_capacity: usize,
    shared_activation_route: VulkanSharedResidentBufferRoute,
) -> Result<
    VulkanDistributedComponentBatchDispatchRunner,
    VulkanResidentInProcessPlacedRuntimeError,
> {
    let reduction = planned.reduction.as_ref().ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
            "distributed input-column batch {}.{} has no reduction plan",
            planned.component_id, planned.node_id
        )))
    })?;
    let owner_index = placed_slices
        .iter()
        .position(|slice| slice.device_id == planned.owner_device_id)
        .ok_or_else(|| VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
            device_id: planned.owner_device_id.clone(),
        })?;
    let package_slice = &placed_slices[owner_index].package_slice;
    let batch_slice = &batch_slices[owner_index];
    let artifact = package_slice
        .loaded_manifest
        .physical_artifact(&planned.physical_artifact_id)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed input-column batch {}.{} is missing physical artifact {:?}",
                planned.component_id, planned.node_id, planned.physical_artifact_id
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
    if batch_slice.signal_buffer(&input_key)?.frame_byte_capacity
        != planned.input_byte_capacity
        || batch_slice.signal_buffer(&output_key)?.frame_byte_capacity
            != planned.output_byte_capacity
    {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed input-column batch {}.{} signal capacities differ from its physical plan",
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
                    "distributed input-column batch {}.{} auxiliary signal {} differs from its physical plan",
                    planned.component_id, planned.node_id, activation.signal_id
                )),
            ));
        }
    }
    let reduction_buffer = reduction_buffers
        .iter()
        .find(|buffer| {
            buffer.planned.owner_device_id == planned.owner_device_id
                && buffer.planned.dispatch_index == planned.dispatch_index
        })
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed input-column batch {}.{} has no partial-output allocation",
                planned.component_id, planned.node_id
            )))
        })?;
    let reduction_owner = reduction_buffer
        .device_buffers
        .get(&planned.owner_device_id)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed input-column batch {}.{} has no owner partial-output view",
                planned.component_id, planned.node_id
            )))
        })?;
    let output = &batch_slice.signal_buffer(&output_key)?.buffer;
    let residual = match &reduction.finalization {
        VulkanDistributedReductionFinalizationPlan::StoreF32
        | VulkanDistributedReductionFinalizationPlan::StoreF32ToBf16 => None,
        VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
            residual_input_index,
        } => {
            let key = if *residual_input_index == 0 {
                &input_key
            } else {
                auxiliary_input_keys
                    .get(*residual_input_index - 1)
                    .ok_or_else(|| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "distributed input-column batch {}.{} has no residual input {}",
                                planned.component_id,
                                planned.node_id,
                                residual_input_index
                            ),
                        ))
                    })?
            };
            Some(&batch_slice.signal_buffer(key)?.buffer)
        }
    };
    let owner_device = devices.get(&planned.owner_device_id).ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
            device_id: planned.owner_device_id.clone(),
        }
    })?;
    let reduction_runner = create_distributed_reduction_runner_for_buffers(
        owner_device,
        planned,
        lane_capacity,
        reduction_owner,
        output,
        residual,
        None,
        &[],
    )
    .map_err(|error| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(error.to_string()))
    })?;

    let mut shards = Vec::with_capacity(planned.shards.len());
    for (shard_index, shard) in planned.shards.iter().enumerate() {
        if shard.auxiliary_input_ranges.len() != planned.auxiliary_input_activations.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "distributed input-column batch {}.{} has {} auxiliary ranges for {} auxiliary inputs on {:?}",
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
        let input_private_key = VulkanDistributedComponentBatchPrivateActivationBufferKey {
            activation: distributed_component_batch_activation_key(
                &planned.owner_device_id,
                &planned.input_activation,
            ),
            device_id: shard.device_id.clone(),
        };
        let input = if let Some(buffer) = private_activation_buffers.get(&input_private_key) {
            buffer
        } else {
            batch_slice.distributed_signal_buffer(&input_key, &shard.device_id)?
        };
        let (input_byte_offset, input_byte_capacity) =
            distributed_batch_shard_binding_range(
                planned.input_byte_capacity,
                lane_capacity,
                &shard.input_range,
            )?;
        let partials = reduction_buffer
            .device_buffers
            .get(&shard.device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed input-column batch {}.{} has no partial-output view on {:?}",
                    planned.component_id, planned.node_id, shard.device_id
                )))
            })?;
        let (partial_byte_offset, partial_lane_byte_capacity) =
            distributed_batch_reduction_plane_binding_range(
                reduction.partial_byte_capacity,
                planned.shards.len(),
                lane_capacity,
                shard_index,
            )?;
        let mut bindings = Vec::with_capacity(
            2 + planned.auxiliary_input_activations.len()
                + shard.parameters.len()
                + 2 * planned.selected_resource_partitions.len(),
        );
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(planned.input_activation.binding).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "distributed input-column primary binding exceeds u32".to_string(),
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
            let (byte_offset, byte_capacity) = distributed_batch_shard_binding_range(
                activation.signal_byte_capacity,
                lane_capacity,
                range,
            )?;
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    u32::try_from(activation.binding).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed input-column auxiliary binding exceeds u32".to_string(),
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
                        "distributed input-column output binding exceeds u32".to_string(),
                    ))
                })?,
                partials,
                partial_lane_byte_capacity,
            )
            .with_byte_offset(partial_byte_offset)
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
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed input-column batch {}.{} is missing tensor {:?} range {}..{} on {:?}",
                        planned.component_id,
                        planned.node_id,
                        fragment.tensor,
                        fragment.byte_offset,
                        fragment.byte_offset + fragment.byte_count,
                        shard.device_id
                    )))
                })?;
            bindings.push(
                allocation
                    .kernel_binding_for_fragment(
                        u32::try_from(fragment.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "distributed input-column parameter binding exceeds u32"
                                    .to_string(),
                            ))
                        })?,
                        fragment.byte_offset,
                        fragment.byte_count,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!(
                                "failed to bind distributed input-column parameter: {error}"
                            ),
                        ))
                    })?
                    .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        for partition in &planned.selected_resource_partitions {
            let resources = dynamic_resource_buffers
                .get(&shard.device_id)
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed input-column batch {}.{} has no dynamic resource buffers on {:?}",
                        planned.component_id, planned.node_id, shard.device_id,
                    )))
                })?;
            let parameter_slots = resources
                .parameter_slots(
                    &planned.component_id,
                    &planned.node_id,
                    &partition.selection_signal,
                )
                .ok_or_else(|| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "distributed input-column batch {}.{} has no parameter slots for selector {:?} on {:?}",
                        planned.component_id,
                        planned.node_id,
                        partition.selector_id,
                        shard.device_id,
                    )))
                })?;
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    u32::try_from(partition.address_table_binding).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed input-column address-table binding exceeds u32"
                                .to_string(),
                        ))
                    })?,
                    resources.address_table(),
                    resources.address_table().byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::Read),
            );
            bindings.push(
                VulkanResidentKernelBufferBinding::new(
                    u32::try_from(partition.parameter_slots_binding).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed input-column parameter-slot binding exceeds u32"
                                .to_string(),
                        ))
                    })?,
                    parameter_slots,
                    parameter_slots.byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        let push_constants = distributed_shard_push_constants(planned, shard).map_err(|error| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(error.to_string()))
        })?;
        let dispatch = device
            .create_resident_kernel_dispatch_2d_labeled(
                &artifact.words,
                &bindings,
                shard.workgroup_count_x,
                u32::try_from(lane_capacity).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "distributed input-column lane capacity exceeds u32".to_string(),
                    ))
                })?,
                artifact.artifact.local_size_x,
                u32::try_from(push_constants.len()).expect("partition control is at most 8 bytes"),
                Some(format!(
                    "component={} node={} distributed_batch=device:{} columns={}..{} distribution=InputColumns",
                    planned.component_id,
                    planned.node_id,
                    shard.device_id,
                    shard.row_start,
                    shard.row_start + shard.row_count,
                )),
            )
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        shards.push(VulkanDistributedComponentBatchShardRunner {
            device_id: shard.device_id.clone(),
            expert_start: u32::try_from(shard.row_start).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed input-column start exceeds u32".to_string(),
                ))
            })?,
            expert_count: u32::try_from(shard.row_count).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "distributed input-column count exceeds u32".to_string(),
                ))
            })?,
            dispatches: vec![VulkanDistributedComponentBatchShardDispatch {
                dispatch,
                push_constants,
                control_buffer_set_index: 0,
                indirect_dispatch: None,
                dispatch_y_from_batch_width: true,
            }],
            selected_resource_gates: Vec::new(),
            batch_control_buffer_sets: Vec::new(),
            sequence_catalog: RefCell::new(BTreeMap::new()),
        });
    }
    let planned_island = resolved_physical_execution_islands(
        std::slice::from_ref(planned),
        shared_activation_route,
    )
    .map_err(|error| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(error.to_string()))
    })?
    .pop()
    .expect("one distributed dispatch resolves to one physical island");
    Ok(VulkanDistributedComponentBatchDispatchRunner {
        planned: planned_island,
        shards,
        helper_synchronization: Vec::new(),
        reduction: Some(reduction_runner),
    })
}

fn distributed_batch_reduction_plane_binding_range(
    plane_byte_capacity: usize,
    participant_count: usize,
    lane_capacity: usize,
    participant_index: usize,
) -> Result<(usize, usize), VulkanResidentInProcessPlacedRuntimeError> {
    if plane_byte_capacity == 0 || participant_count == 0 || lane_capacity == 0 {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError("distributed batch reduction geometry is empty".to_string()),
        ));
    }
    if participant_index >= participant_count {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "distributed batch reduction participant {participant_index} exceeds count {participant_count}"
            )),
        ));
    }
    let participant_byte_capacity = plane_byte_capacity
        .checked_mul(lane_capacity)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed batch reduction participant capacity overflowed".to_string(),
            ))
        })?;
    let byte_offset = participant_index
        .checked_mul(participant_byte_capacity)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed batch reduction participant offset overflowed".to_string(),
            ))
        })?;
    let total_byte_capacity = participant_byte_capacity
        .checked_mul(participant_count)
        .ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                "distributed batch reduction capacity overflowed".to_string(),
            ))
        })?;
    if byte_offset
        .checked_add(participant_byte_capacity)
        .is_none_or(|end| end > total_byte_capacity)
    {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError("distributed batch reduction binding exceeds its allocation".to_string()),
        ));
    }
    Ok((byte_offset, participant_byte_capacity))
}
