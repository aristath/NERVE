#[allow(clippy::too_many_arguments)]
fn create_distributed_output_row_physical_component_batch_dispatch(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    batch_slices: &[VulkanResidentComponentBatchSliceRunner],
    planned: &VulkanDistributedDispatchPlan,
    parameter_buffers: &VulkanDistributedParameterBuffers,
    dynamic_resource_buffers: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
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
    if planned.distribution != VulkanDistributedDispatchDistribution::OutputRows
        || planned.reduction.is_some()
    {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "physical output-row batch {}.{} has incompatible distribution or reduction",
                planned.component_id, planned.node_id,
            )),
        ));
    }
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
                "physical output-row batch {}.{} is missing artifact {:?}",
                planned.component_id, planned.node_id, planned.physical_artifact_id,
            )))
        })?;
    if !artifact.artifact.push_constants.is_empty() {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "physical output-row batch {}.{} unexpectedly requires push constants",
                planned.component_id, planned.node_id,
            )),
        ));
    }

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
                "physical output-row batch {}.{} signal capacities differ from its plan",
                planned.component_id, planned.node_id,
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
                    "physical output-row batch {}.{} auxiliary signal {} differs from its plan",
                    planned.component_id, planned.node_id, activation.signal_id,
                )),
            ));
        }
    }

    let mut shards = Vec::with_capacity(planned.shards.len());
    for shard in &planned.shards {
        if shard.auxiliary_input_ranges.len() != planned.auxiliary_input_activations.len() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "physical output-row batch {}.{} has {} auxiliary ranges for {} inputs on {:?}",
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
        let output_private_key = VulkanDistributedComponentBatchPrivateActivationBufferKey {
            activation: distributed_component_batch_activation_key(
                &planned.owner_device_id,
                &planned.output_activation,
            ),
            device_id: shard.device_id.clone(),
        };
        let private_input = private_activation_buffers.get(&input_private_key);
        let input = if let Some(buffer) = private_input {
            buffer
        } else {
            batch_slice.distributed_signal_buffer(&input_key, &shard.device_id)?
        };
        let private_output = private_activation_buffers.get(&output_private_key);
        let output = if let Some(buffer) = private_output {
            buffer
        } else {
            batch_slice.distributed_signal_buffer(&output_key, &shard.device_id)?
        };
        let (input_byte_offset, input_byte_capacity) = if private_input.is_some() {
            local_distributed_component_batch_binding_range(
                shard.input_range.byte_count,
                lane_capacity,
                "input",
            )?
        } else {
            distributed_batch_shard_binding_range(
                planned.input_byte_capacity,
                lane_capacity,
                &shard.input_range,
            )?
        };
        let (output_byte_offset, output_byte_capacity) = if private_output.is_some() {
            local_distributed_component_batch_binding_range(
                shard.output_byte_count,
                lane_capacity,
                "output",
            )?
        } else {
            distributed_batch_shard_output_binding_range(
                planned.output_byte_capacity,
                lane_capacity,
                shard.output_byte_offset,
                shard.output_byte_count,
            )?
        };
        let mut bindings = Vec::with_capacity(
            2 + planned.auxiliary_input_activations.len()
                + shard.parameters.len()
                + 2 * planned.selected_resource_partitions.len(),
        );
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(planned.input_activation.binding).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "physical output-row primary binding exceeds u32".to_string(),
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
                            "physical output-row auxiliary binding exceeds u32".to_string(),
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
                        "physical output-row output binding exceeds u32".to_string(),
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
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                        "physical output-row batch {}.{} is missing tensor {:?} range {}..{} on {:?}",
                        planned.component_id,
                        planned.node_id,
                        fragment.tensor,
                        fragment.byte_offset,
                        fragment.byte_offset + fragment.byte_count,
                        shard.device_id,
                    )))
                })?;
            bindings.push(
                allocation
                    .kernel_binding_for_fragment(
                        u32::try_from(fragment.binding).map_err(|_| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                "physical output-row parameter binding exceeds u32".to_string(),
                            ))
                        })?,
                        fragment.byte_offset,
                        fragment.byte_count,
                    )
                    .map_err(|error| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            format!("failed to bind physical output-row parameter: {error}"),
                        ))
                    })?
                    .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        for partition in &planned.selected_resource_partitions {
            let resources = dynamic_resource_buffers.get(&shard.device_id).ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "physical output-row batch {}.{} has no dynamic buffers on {:?}",
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
                        "physical output-row batch {}.{} has no parameter slots for {:?} on {:?}",
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
                            "physical output-row address-table binding exceeds u32".to_string(),
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
                            "physical output-row parameter-slot binding exceeds u32".to_string(),
                        ))
                    })?,
                    parameter_slots,
                    parameter_slots.byte_capacity(),
                )
                .with_access(VulkanResidentKernelBufferAccess::Read),
            );
        }
        let dispatch = device
            .create_resident_kernel_dispatch_2d_labeled(
                &artifact.words,
                &bindings,
                shard.workgroup_count_x,
                u32::try_from(lane_capacity).map_err(|_| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        "physical output-row lane capacity exceeds u32".to_string(),
                    ))
                })?,
                artifact.artifact.local_size_x,
                0,
                Some(format!(
                    "component={} node={} physical_distributed_batch=device:{} rows={}..{} distribution=OutputRows",
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
                    "physical output-row start exceeds u32".to_string(),
                ))
            })?,
            expert_count: u32::try_from(shard.row_count).map_err(|_| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                    "physical output-row count exceeds u32".to_string(),
                ))
            })?,
            dispatches: vec![VulkanDistributedComponentBatchShardDispatch {
                dispatch,
                push_constants: Vec::new(),
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
        reduction: None,
    })
}

fn local_distributed_component_batch_binding_range(
    frame_byte_capacity: usize,
    lane_capacity: usize,
    role: &str,
) -> Result<(usize, usize), VulkanResidentInProcessPlacedRuntimeError> {
    if frame_byte_capacity == 0 || lane_capacity == 0 {
        return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
            VulkanError(format!(
                "physical distributed private batch {role} range is empty"
            )),
        ));
    }
    Ok((
        0,
        frame_byte_capacity
            .checked_mul(lane_capacity)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "physical distributed private batch {role} capacity overflowed"
                )))
            })?,
    ))
}
