fn validate_device_pool(device_ids: &[String]) -> Result<(), VulkanDistributedPlanError> {
    if device_ids.is_empty() {
        return Err(VulkanDistributedPlanError(
            "distributed execution device pool must not be empty".to_string(),
        ));
    }
    let mut unique = BTreeSet::new();
    if let Some(device_id) = device_ids
        .iter()
        .find(|device_id| device_id.is_empty() || !unique.insert(device_id.as_str()))
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed execution device pool contains an empty or repeated device {device_id:?}"
        )));
    }
    Ok(())
}

fn accumulate_activation_allocation(
    allocations: &mut BTreeMap<
        VulkanDistributedActivationBufferAllocationKey,
        VulkanDistributedActivationBufferAllocation,
    >,
    owner_device_id: &str,
    activation: &VulkanDistributedActivationSlot,
    participant_device_ids: &BTreeSet<&str>,
    access: VulkanDistributedActivationAccess,
) -> Result<(), VulkanDistributedPlanError> {
    if activation.byte_capacity == 0 {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has zero capacity",
            activation.component_id, activation.slot
        )));
    }
    if activation.signal_byte_capacity == 0
        || activation.signal_byte_capacity > activation.byte_capacity
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has signal {:?} capacity {} outside its {}-byte slot",
            activation.component_id,
            activation.slot,
            activation.signal_id,
            activation.signal_byte_capacity,
            activation.byte_capacity
        )));
    }
    let key = distributed_activation_allocation_key(owner_device_id, activation);
    let allocation_owner_device_id = match &activation.storage {
        VulkanDistributedActivationStorage::ActivationSlot => owner_device_id,
        VulkanDistributedActivationStorage::Edge {
            owner_device_id, ..
        } => owner_device_id,
    };
    let allocation =
        allocations
            .entry(key)
            .or_insert_with(|| VulkanDistributedActivationBufferAllocation {
                storage: activation.storage.clone(),
                owner_device_id: allocation_owner_device_id.to_string(),
                component_id: activation.component_id.clone(),
                slot: activation.slot,
                byte_capacity: activation.byte_capacity,
                signal_ids: Vec::new(),
                device_ids: Vec::new(),
                input_use_count: 0,
                output_use_count: 0,
            });
    if allocation.storage != activation.storage
        || allocation.owner_device_id != allocation_owner_device_id
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} maps to conflicting storage",
            activation.component_id, activation.slot
        )));
    }
    if allocation.byte_capacity != activation.byte_capacity {
        return Err(VulkanDistributedPlanError(format!(
            "distributed activation {}.slot_{} has conflicting capacities {} and {}",
            activation.component_id,
            activation.slot,
            allocation.byte_capacity,
            activation.byte_capacity
        )));
    }
    if !allocation.signal_ids.contains(&activation.signal_id) {
        allocation.signal_ids.push(activation.signal_id.clone());
        allocation.signal_ids.sort();
    }
    for device_id in participant_device_ids {
        if !allocation
            .device_ids
            .iter()
            .any(|existing| existing == device_id)
        {
            allocation.device_ids.push((*device_id).to_string());
        }
    }
    if !allocation
        .device_ids
        .iter()
        .any(|existing| existing == allocation_owner_device_id)
    {
        allocation
            .device_ids
            .push(allocation_owner_device_id.to_string());
    }
    allocation.device_ids.sort();
    match access {
        VulkanDistributedActivationAccess::Input => {
            allocation.input_use_count =
                allocation.input_use_count.checked_add(1).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation input use count overflowed".to_string(),
                    )
                })?;
        }
        VulkanDistributedActivationAccess::Output => {
            allocation.output_use_count =
                allocation.output_use_count.checked_add(1).ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "distributed activation output use count overflowed".to_string(),
                    )
                })?;
        }
    }
    Ok(())
}

fn validate_tensor_partition_coverage<'a>(
    allocations: impl Iterator<Item = &'a VulkanDistributedParameterAllocation>,
    tensor_index: &TensorIndex,
) -> Result<(), VulkanDistributedPlanError> {
    let mut ranges_by_tensor = BTreeMap::<&str, BTreeSet<(usize, usize)>>::new();
    for allocation in allocations {
        ranges_by_tensor
            .entry(&allocation.tensor)
            .or_default()
            .insert((allocation.byte_offset, allocation.byte_count));
    }
    for (tensor, ranges) in ranges_by_tensor {
        let tensor_byte_count = tensor_index
            .tensors
            .get(tensor)
            .and_then(|metadata| metadata.byte_count)
            .ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} has no byte count"
                ))
            })?;
        let mut next_offset = 0usize;
        for (byte_offset, byte_count) in ranges {
            if byte_offset != next_offset {
                return Err(VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} has a gap or overlap at byte {next_offset}"
                )));
            }
            next_offset = next_offset.checked_add(byte_count).ok_or_else(|| {
                VulkanDistributedPlanError(format!(
                    "distributed parameter tensor {tensor:?} partition overflowed"
                ))
            })?;
        }
        if next_offset != tensor_byte_count {
            return Err(VulkanDistributedPlanError(format!(
                "distributed parameter tensor {tensor:?} partition covers {next_offset} of {tensor_byte_count} bytes"
            )));
        }
    }
    Ok(())
}

fn plan_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    edge_placements: &[ComponentEdgePlacement],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if DISTRIBUTABLE_SPARSE_EXPERT_OPS.contains(&dispatch.op.as_str()) {
        return plan_sparse_expert_dispatch(
            owner_device_id,
            dispatch,
            tensor_index,
            device_ids,
            artifact_workgroup_count_x,
            storage_buffer_offset_alignment,
        );
    }
    if dispatch.op == DISTRIBUTABLE_RESIDUAL_PROJECTION_OP {
        return plan_block_scaled_fp8_projection_dispatch(
            owner_device_id,
            dispatch,
            tensor_index,
            device_ids,
            edge_placements,
            artifact_workgroup_count_x,
            storage_buffer_offset_alignment,
            BlockScaledFp8ProjectionKind::Residual,
        );
    }
    plan_parallel_projection_dispatch(
        owner_device_id,
        dispatch,
        tensor_index,
        device_ids,
        edge_placements,
        artifact_workgroup_count_x,
        storage_buffer_offset_alignment,
    )
}

fn plan_parallel_projection_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    edge_placements: &[ComponentEdgePlacement],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if !dispatch.push_constants.is_empty() {
        return Ok(None);
    }
    let parameter_descriptors = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                Some((descriptor.binding, tensor.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if parameter_descriptors.len() == 4 {
        return plan_block_scaled_fp8_projection_dispatch(
            owner_device_id,
            dispatch,
            tensor_index,
            device_ids,
            edge_placements,
            artifact_workgroup_count_x,
            storage_buffer_offset_alignment,
            BlockScaledFp8ProjectionKind::ParallelSiluMultiply,
        );
    }
    let [
        (first_binding, first_tensor),
        (second_binding, second_tensor),
    ] = parameter_descriptors.as_slice()
    else {
        // A requested shard pool fails closed later when no dispatch in the
        // component exposes a supported physical distribution contract.
        return Ok(None);
    };
    let first = projection_metadata(tensor_index, dispatch, first_tensor)?;
    let second = projection_metadata(tensor_index, dispatch, second_tensor)?;
    if first.shape != second.shape {
        return Err(dispatch_error(
            dispatch,
            format!(
                "projection shapes {:?} and {:?} do not match",
                first.shape, second.shape
            ),
        ));
    }
    let output_rows = first.shape[0];
    let input_width = first.shape[1];
    let artifact_workgroup_count = usize::try_from(artifact_workgroup_count_x).map_err(|_| {
        dispatch_error(
            dispatch,
            "artifact workgroup count exceeds usize".to_string(),
        )
    })?;
    if artifact_workgroup_count == 0 || output_rows % artifact_workgroup_count != 0 {
        return Err(dispatch_error(
            dispatch,
            format!(
                "output row count {output_rows} is incompatible with artifact workgroup count {artifact_workgroup_count}"
            ),
        ));
    }
    let workgroup_row_count = output_rows / artifact_workgroup_count;
    let mut row_alignment = least_common_multiple(
        workgroup_row_count,
        storage_buffer_offset_alignment / BF16_BYTE_COUNT,
    )
    .ok_or_else(|| dispatch_error(dispatch, "row alignment overflowed".to_string()))?;
    if [first.layout.as_deref(), second.layout.as_deref()]
        .contains(&Some("vulkan_bf16_row_pair_u32"))
    {
        row_alignment = least_common_multiple(row_alignment, 2)
            .ok_or_else(|| dispatch_error(dispatch, "row alignment overflowed".to_string()))?;
    }
    let input_byte_capacity = input_width
        .checked_mul(BF16_BYTE_COUNT)
        .ok_or_else(|| dispatch_error(dispatch, "input byte capacity overflowed".to_string()))?;
    let output_byte_capacity = output_rows
        .checked_mul(BF16_BYTE_COUNT)
        .ok_or_else(|| dispatch_error(dispatch, "output byte capacity overflowed".to_string()))?;
    let input_activation = activation_slot(dispatch, 0, input_byte_capacity, "input")?;
    let output_activation = activation_slot(dispatch, 1, output_byte_capacity, "output")?;

    let raw_shards = distribute_rows(
        output_rows,
        device_ids.len(),
        workgroup_row_count,
        row_alignment,
    )
    .map_err(|error| dispatch_error(dispatch, error))?;
    if raw_shards.len() < 2 {
        return Ok(None);
    }
    let shard_device_ids = std::iter::once(owner_device_id)
        .chain(
            device_ids
                .iter()
                .map(String::as_str)
                .filter(|device_id| *device_id != owner_device_id),
        )
        .take(raw_shards.len())
        .collect::<Vec<_>>();
    let first_row_bytes = tensor_row_bytes(dispatch, first_tensor, first, output_rows)?;
    let second_row_bytes = tensor_row_bytes(dispatch, second_tensor, second, output_rows)?;
    let mut distributed_parameter_byte_count = 0usize;
    let shards = shard_device_ids
        .into_iter()
        .zip(raw_shards)
        .map(|(device_id, (row_start, row_count))| {
            let workgroup_count_x =
                u32::try_from(row_count / workgroup_row_count).map_err(|_| {
                    dispatch_error(dispatch, "shard workgroup count exceeds u32".to_string())
                })?;
            let first_fragment = parameter_fragment(
                *first_binding,
                first_tensor,
                first_row_bytes,
                row_start,
                row_count,
                dispatch,
            )?;
            let second_fragment = parameter_fragment(
                *second_binding,
                second_tensor,
                second_row_bytes,
                row_start,
                row_count,
                dispatch,
            )?;
            distributed_parameter_byte_count = distributed_parameter_byte_count
                .checked_add(first_fragment.byte_count)
                .and_then(|total| total.checked_add(second_fragment.byte_count))
                .ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        "shard parameter byte count overflowed".to_string(),
                    )
                })?;
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
                row_start,
                row_count,
                workgroup_count_x,
                base_workgroup_z: 0,
                input_range: VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: input_byte_capacity,
                },
                auxiliary_input_ranges: Vec::new(),
                output_byte_offset: row_start.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output offset overflowed".to_string())
                })?,
                output_byte_count: row_count.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output size overflowed".to_string())
                })?,
                parameters: vec![first_fragment, second_fragment],
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;

    Ok(Some(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        component_id: dispatch.component_id.clone(),
        node_id: dispatch.node_id.clone(),
        reusable_family_id: dispatch.reusable_family_id.clone(),
        input_byte_capacity,
        output_byte_capacity,
        output_rows,
        input_width,
        row_alignment,
        input_activation,
        auxiliary_input_activations: Vec::new(),
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::OutputRows,
        distributed_parameter_byte_count,
        shards,
    }))
}

#[derive(Clone, Copy)]
enum BlockScaledFp8ProjectionKind {
    ParallelSiluMultiply,
    Residual,
}

struct BlockScaledFp8Matrix {
    weight_binding: usize,
    weight_tensor: String,
    scale_binding: usize,
    scale_tensor: String,
    output_rows: usize,
    input_width: usize,
    scale_rows: usize,
    scale_columns: usize,
    block_rows: usize,
    block_columns: usize,
    weight_row_bytes: usize,
    scale_row_bytes: usize,
}

#[allow(clippy::too_many_arguments)]
fn plan_block_scaled_fp8_projection_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    edge_placements: &[ComponentEdgePlacement],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
    kind: BlockScaledFp8ProjectionKind,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    if !dispatch.push_constants.is_empty() {
        return Ok(None);
    }
    let parameters = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                Some((descriptor.binding, tensor.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let pairs = match kind {
        BlockScaledFp8ProjectionKind::ParallelSiluMultiply => {
            let [first_weight, first_scale, second_weight, second_scale] =
                parameters.as_slice()
            else {
                return Ok(None);
            };
            vec![(*first_weight, *first_scale), (*second_weight, *second_scale)]
        }
        BlockScaledFp8ProjectionKind::Residual => {
            let [weight, scale] = parameters.as_slice() else {
                return Ok(None);
            };
            vec![(*weight, *scale)]
        }
    };
    if pairs.iter().any(|((_, weight), (_, scale))| {
        tensor_index
            .tensors
            .get(*weight)
            .zip(tensor_index.tensors.get(*scale))
            .is_none_or(|(weight, scale)| weight.dtype != "F8_E4M3" || scale.dtype != "BF16")
    }) {
        return Ok(None);
    }
    let matrices = pairs
        .into_iter()
        .map(|(weight, scale)| {
            block_scaled_fp8_matrix(tensor_index, dispatch, weight, scale)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let leader = matrices
        .first()
        .expect("block-scaled projections contain at least one matrix");
    if matrices.iter().any(|matrix| {
        matrix.output_rows != leader.output_rows
            || matrix.input_width != leader.input_width
            || matrix.scale_rows != leader.scale_rows
            || matrix.scale_columns != leader.scale_columns
            || matrix.block_rows != leader.block_rows
            || matrix.block_columns != leader.block_columns
    }) {
        return Err(dispatch_error(
            dispatch,
            "block-scaled FP8 projection branches have incompatible shapes".to_string(),
        ));
    }
    let output_rows = leader.output_rows;
    let input_width = leader.input_width;
    let scale_columns = leader.scale_columns;
    let block_rows = leader.block_rows;
    let artifact_workgroup_count =
        usize::try_from(artifact_workgroup_count_x).map_err(|_| {
            dispatch_error(
                dispatch,
                "artifact workgroup count exceeds usize".to_string(),
            )
        })?;
    if artifact_workgroup_count == 0 || !output_rows.is_multiple_of(artifact_workgroup_count) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "output row count {output_rows} is incompatible with artifact workgroup count {artifact_workgroup_count}"
            ),
        ));
    }
    let workgroup_row_count = output_rows / artifact_workgroup_count;
    let output_row_alignment = storage_buffer_offset_alignment / BF16_BYTE_COUNT;
    let row_alignment = least_common_multiple(workgroup_row_count, block_rows)
        .and_then(|alignment| least_common_multiple(alignment, output_row_alignment))
        .ok_or_else(|| dispatch_error(dispatch, "FP8 row alignment overflowed".to_string()))?;
    let input_byte_capacity = input_width;
    let input_scale_byte_capacity = scale_columns
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| dispatch_error(dispatch, "FP8 input scale size overflowed".to_string()))?;
    let output_byte_capacity = output_rows
        .checked_mul(BF16_BYTE_COUNT)
        .ok_or_else(|| dispatch_error(dispatch, "output byte capacity overflowed".to_string()))?;
    let (input_activation, auxiliary_input_activations, output_activation) = match kind {
        BlockScaledFp8ProjectionKind::ParallelSiluMultiply => {
            let Some(input) = distributed_activation(
                dispatch,
                0,
                input_byte_capacity,
                "quantized input",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            let Some(scale) = distributed_activation(
                dispatch,
                1,
                input_scale_byte_capacity,
                "input scale",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            let Some(output) = distributed_activation(
                dispatch,
                2,
                output_byte_capacity,
                "output",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            (input, vec![scale], output)
        }
        BlockScaledFp8ProjectionKind::Residual => {
            let Some(input) = distributed_activation(
                dispatch,
                0,
                input_byte_capacity,
                "quantized input",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            let Some(scale) = distributed_activation(
                dispatch,
                1,
                input_scale_byte_capacity,
                "input scale",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            let Some(residual) = distributed_activation(
                dispatch,
                2,
                output_byte_capacity,
                "residual input",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            let Some(output) = distributed_activation(
                dispatch,
                3,
                output_byte_capacity,
                "output",
                edge_placements,
            )?
            else {
                return Ok(None);
            };
            (input, vec![scale, residual], output)
        }
    };
    let raw_shards = distribute_rows(
        output_rows,
        device_ids.len(),
        workgroup_row_count,
        row_alignment,
    )
    .map_err(|error| dispatch_error(dispatch, error))?;
    if raw_shards.len() < 2 {
        return Ok(None);
    }
    let shard_device_ids = std::iter::once(owner_device_id)
        .chain(
            device_ids
                .iter()
                .map(String::as_str)
                .filter(|device_id| *device_id != owner_device_id),
        )
        .take(raw_shards.len())
        .collect::<Vec<_>>();
    let mut distributed_parameter_byte_count = 0usize;
    let shards = shard_device_ids
        .into_iter()
        .zip(raw_shards)
        .map(|(device_id, (row_start, row_count))| {
            let scale_row_start = row_start / block_rows;
            let scale_row_count = row_count / block_rows;
            let parameters = matrices
                .iter()
                .flat_map(|matrix| {
                    [
                        parameter_fragment(
                            matrix.weight_binding,
                            &matrix.weight_tensor,
                            matrix.weight_row_bytes,
                            row_start,
                            row_count,
                            dispatch,
                        ),
                        parameter_fragment(
                            matrix.scale_binding,
                            &matrix.scale_tensor,
                            matrix.scale_row_bytes,
                            scale_row_start,
                            scale_row_count,
                            dispatch,
                        ),
                    ]
                })
                .collect::<Result<Vec<_>, _>>()?;
            distributed_parameter_byte_count = parameters.iter().try_fold(
                distributed_parameter_byte_count,
                |total, fragment| {
                    total.checked_add(fragment.byte_count).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "FP8 shard parameter byte count overflowed".to_string(),
                        )
                    })
                },
            )?;
            let mut auxiliary_input_ranges = vec![VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: input_scale_byte_capacity,
            }];
            if matches!(kind, BlockScaledFp8ProjectionKind::Residual) {
                auxiliary_input_ranges.push(VulkanDistributedActivationRange {
                    byte_offset: row_start.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                        dispatch_error(dispatch, "residual shard offset overflowed".to_string())
                    })?,
                    byte_count: row_count.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                        dispatch_error(dispatch, "residual shard size overflowed".to_string())
                    })?,
                });
            }
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
                row_start,
                row_count,
                workgroup_count_x: u32::try_from(row_count / workgroup_row_count).map_err(
                    |_| dispatch_error(dispatch, "shard workgroup count exceeds u32".to_string()),
                )?,
                base_workgroup_z: 0,
                input_range: VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: input_byte_capacity,
                },
                auxiliary_input_ranges,
                output_byte_offset: row_start.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output offset overflowed".to_string())
                })?,
                output_byte_count: row_count.checked_mul(BF16_BYTE_COUNT).ok_or_else(|| {
                    dispatch_error(dispatch, "shard output size overflowed".to_string())
                })?,
                parameters,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
    Ok(Some(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        component_id: dispatch.component_id.clone(),
        node_id: dispatch.node_id.clone(),
        reusable_family_id: dispatch.reusable_family_id.clone(),
        input_byte_capacity,
        output_byte_capacity,
        output_rows,
        input_width,
        row_alignment,
        input_activation,
        auxiliary_input_activations,
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::OutputRows,
        distributed_parameter_byte_count,
        shards,
    }))
}

fn block_scaled_fp8_matrix(
    tensor_index: &TensorIndex,
    dispatch: &VulkanPreparedDispatch,
    weight: (usize, &str),
    scale: (usize, &str),
) -> Result<BlockScaledFp8Matrix, VulkanDistributedPlanError> {
    let weight_metadata =
        block_scaled_fp8_matrix_metadata(tensor_index, dispatch, weight.1, "F8_E4M3")?;
    let scale_metadata =
        block_scaled_fp8_matrix_metadata(tensor_index, dispatch, scale.1, "BF16")?;
    let [output_rows, input_width] = weight_metadata.shape.as_slice() else {
        unreachable!("block-scaled FP8 weight metadata is rank two");
    };
    let [scale_rows, scale_columns] = scale_metadata.shape.as_slice() else {
        unreachable!("block-scaled FP8 scale metadata is rank two");
    };
    if *scale_rows == 0
        || *scale_columns == 0
        || !output_rows.is_multiple_of(*scale_rows)
        || !input_width.is_multiple_of(*scale_columns)
    {
        return Err(dispatch_error(
            dispatch,
            format!(
                "FP8 weight shape {:?} is incompatible with scale shape {:?}",
                weight_metadata.shape, scale_metadata.shape
            ),
        ));
    }
    Ok(BlockScaledFp8Matrix {
        weight_binding: weight.0,
        weight_tensor: weight.1.to_string(),
        scale_binding: scale.0,
        scale_tensor: scale.1.to_string(),
        output_rows: *output_rows,
        input_width: *input_width,
        scale_rows: *scale_rows,
        scale_columns: *scale_columns,
        block_rows: output_rows / scale_rows,
        block_columns: input_width / scale_columns,
        weight_row_bytes: exact_matrix_row_bytes(
            dispatch,
            weight.1,
            weight_metadata,
            1,
        )?,
        scale_row_bytes: exact_matrix_row_bytes(
            dispatch,
            scale.1,
            scale_metadata,
            BF16_BYTE_COUNT,
        )?,
    })
}

fn block_scaled_fp8_matrix_metadata<'a>(
    tensor_index: &'a TensorIndex,
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
    dtype: &str,
) -> Result<&'a TensorMetadata, VulkanDistributedPlanError> {
    let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
        dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
    })?;
    if metadata.dtype != dtype
        || metadata.shape.len() != 2
        || metadata.layout.as_deref() != Some("row_major")
    {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} must be a rank-2 row-major {dtype} matrix, found {} {:?} layout {:?}",
                metadata.dtype, metadata.shape, metadata.layout
            ),
        ));
    }
    Ok(metadata)
}

fn exact_matrix_row_bytes(
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
    metadata: &TensorMetadata,
    element_byte_count: usize,
) -> Result<usize, VulkanDistributedPlanError> {
    let expected = metadata
        .shape
        .iter()
        .try_fold(element_byte_count, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| {
            dispatch_error(dispatch, format!("tensor {tensor:?} byte count overflowed"))
        })?;
    let byte_count = metadata.byte_count.unwrap_or(expected);
    if byte_count != expected {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} byte count {byte_count} does not match {}-byte shape {:?}",
                element_byte_count, metadata.shape
            ),
        ));
    }
    Ok(byte_count / metadata.shape[0])
}

fn plan_sparse_expert_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    artifact_workgroup_count_x: u32,
    storage_buffer_offset_alignment: usize,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    let has_supported_expert_start = dispatch.push_constants.as_slice()
        == [VulkanKernelScalarBinding {
            name: "expert_start".to_string(),
            scalar_type: "u32".to_string(),
            source: VulkanKernelScalarSource::PushConstant,
        }];
    if !has_supported_expert_start || artifact_workgroup_count_x == 0 {
        return Ok(None);
    }
    let parameter_descriptors = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                Some((descriptor.binding, tensor.as_str()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if parameter_descriptors.is_empty() {
        return Ok(None);
    }

    let mut expert_count = None;
    let mut expert_alignment = 1usize;
    let mut parameters = Vec::with_capacity(parameter_descriptors.len());
    for (binding, tensor) in parameter_descriptors {
        let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
            dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
        })?;
        if metadata.shape.len() < 2
            || !matches!(
                metadata.layout.as_deref(),
                Some("row_major" | "vulkan_bf16_row_pair_u32")
            )
        {
            return Ok(None);
        }
        let tensor_expert_count = metadata.shape[0];
        if tensor_expert_count == 0
            || expert_count.is_some_and(|expected| expected != tensor_expert_count)
        {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "expert tensor {tensor:?} has incompatible leading dimension {}",
                    tensor_expert_count
                ),
            ));
        }
        expert_count = Some(tensor_expert_count);
        let tensor_byte_count = metadata.byte_count.ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("expert tensor {tensor:?} has no byte count"),
            )
        })?;
        if tensor_byte_count == 0 || !tensor_byte_count.is_multiple_of(tensor_expert_count) {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "expert tensor {tensor:?} byte count {tensor_byte_count} is not divisible by {tensor_expert_count} experts"
                ),
            ));
        }
        let bytes_per_expert = tensor_byte_count / tensor_expert_count;
        let tensor_expert_alignment = storage_buffer_offset_alignment
            / greatest_common_divisor(storage_buffer_offset_alignment, bytes_per_expert);
        expert_alignment = least_common_multiple(expert_alignment, tensor_expert_alignment)
            .ok_or_else(|| dispatch_error(dispatch, "expert alignment overflowed".to_string()))?;
        parameters.push((binding, tensor, bytes_per_expert));
    }
    let expert_count = expert_count.expect("non-empty expert parameter set has a leading size");
    let raw_shards = distribute_rows(expert_count, device_ids.len(), 1, expert_alignment)
        .map_err(|error| dispatch_error(dispatch, error))?;
    if raw_shards.len() < 2 {
        return Ok(None);
    }

    let mut input_activations = activation_slots_for_usage(
        dispatch,
        VulkanKernelDescriptorUsage::InputSignal,
        "input",
    )?;
    if input_activations.len() < 2 {
        return Err(dispatch_error(
            dispatch,
            "requires a primary expert input and route signal".to_string(),
        ));
    }
    let output_activations = activation_slots_for_usage(
        dispatch,
        VulkanKernelDescriptorUsage::OutputSignal,
        "output",
    )?;
    let [output_activation] = output_activations.as_slice() else {
        return Err(dispatch_error(
            dispatch,
            format!(
                "requires exactly one output activation, found {}",
                output_activations.len()
            ),
        ));
    };
    let input_activation = input_activations.remove(0);
    let auxiliary_input_activations = input_activations;
    let output_activation = output_activation.clone();
    let input_byte_capacity = input_activation.signal_byte_capacity;
    let output_byte_capacity = output_activation.signal_byte_capacity;
    let shard_device_ids = std::iter::once(owner_device_id)
        .chain(
            device_ids
                .iter()
                .map(String::as_str)
                .filter(|device_id| *device_id != owner_device_id),
        )
        .take(raw_shards.len())
        .collect::<Vec<_>>();
    let mut distributed_parameter_byte_count = 0usize;
    let shards = shard_device_ids
        .into_iter()
        .zip(raw_shards)
        .map(|(device_id, (expert_start, shard_expert_count))| {
            let parameters = parameters
                .iter()
                .map(|(binding, tensor, bytes_per_expert)| {
                    parameter_fragment(
                        *binding,
                        tensor,
                        *bytes_per_expert,
                        expert_start,
                        shard_expert_count,
                        dispatch,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            distributed_parameter_byte_count = parameters.iter().try_fold(
                distributed_parameter_byte_count,
                |total, fragment| {
                    total.checked_add(fragment.byte_count).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "expert shard parameter byte count overflowed".to_string(),
                        )
                    })
                },
            )?;
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
                row_start: expert_start,
                row_count: shard_expert_count,
                workgroup_count_x: artifact_workgroup_count_x,
                base_workgroup_z: u32::try_from(expert_start).map_err(|_| {
                    dispatch_error(dispatch, "expert start exceeds u32".to_string())
                })?,
                input_range: VulkanDistributedActivationRange {
                    byte_offset: 0,
                    byte_count: input_byte_capacity,
                },
                auxiliary_input_ranges: auxiliary_input_activations
                    .iter()
                    .map(|activation| VulkanDistributedActivationRange {
                        byte_offset: 0,
                        byte_count: activation.signal_byte_capacity,
                    })
                    .collect(),
                output_byte_offset: 0,
                output_byte_count: output_byte_capacity,
                parameters,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;

    Ok(Some(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        component_id: dispatch.component_id.clone(),
        node_id: dispatch.node_id.clone(),
        reusable_family_id: dispatch.reusable_family_id.clone(),
        input_byte_capacity,
        output_byte_capacity,
        output_rows: expert_count,
        input_width: input_byte_capacity / BF16_BYTE_COUNT,
        row_alignment: expert_alignment,
        input_activation,
        auxiliary_input_activations,
        output_activation,
        distribution: VulkanDistributedDispatchDistribution::ExpertRange,
        distributed_parameter_byte_count,
        shards,
    }))
}

fn projection_metadata<'a>(
    tensor_index: &'a TensorIndex,
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
) -> Result<&'a TensorMetadata, VulkanDistributedPlanError> {
    let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
        dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
    })?;
    if metadata.dtype != "BF16" || metadata.shape.len() != 2 {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} must be a rank-2 BF16 matrix, found {} {:?}",
                metadata.dtype, metadata.shape
            ),
        ));
    }
    if !matches!(
        metadata.layout.as_deref(),
        Some("row_major" | "vulkan_bf16_row_pair_u32")
    ) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} has non-shardable layout {:?}",
                metadata.layout
            ),
        ));
    }
    Ok(metadata)
}

fn tensor_row_bytes(
    dispatch: &VulkanPreparedDispatch,
    tensor: &str,
    metadata: &TensorMetadata,
    output_rows: usize,
) -> Result<usize, VulkanDistributedPlanError> {
    let expected = metadata
        .shape
        .iter()
        .try_fold(BF16_BYTE_COUNT, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        });
    let expected = expected.ok_or_else(|| {
        dispatch_error(dispatch, format!("tensor {tensor:?} byte count overflowed"))
    })?;
    let byte_count = metadata.byte_count.unwrap_or(expected);
    if byte_count != expected || !byte_count.is_multiple_of(output_rows) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "tensor {tensor:?} byte count {byte_count} does not match BF16 shape {:?}",
                metadata.shape
            ),
        ));
    }
    Ok(byte_count / output_rows)
}

fn activation_slot(
    dispatch: &VulkanPreparedDispatch,
    binding: usize,
    required: usize,
    role: &str,
) -> Result<VulkanDistributedActivationSlot, VulkanDistributedPlanError> {
    let activation = dispatch
        .descriptors
        .iter()
        .find(|descriptor| descriptor.binding == binding)
        .and_then(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::ActivationSlot {
                component_id,
                signal_id,
                slot,
                byte_capacity,
                signal_byte_capacity,
            } => Some(VulkanDistributedActivationSlot {
                binding,
                component_id: component_id.clone(),
                signal_id: signal_id.clone(),
                slot: *slot,
                byte_capacity: *byte_capacity,
                signal_byte_capacity: *signal_byte_capacity,
                storage: VulkanDistributedActivationStorage::ActivationSlot,
            }),
            _ => None,
        })
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("has no resident {role} activation at binding {binding}"),
            )
        })?;
    if activation.signal_byte_capacity < required {
        return Err(dispatch_error(
            dispatch,
            format!(
                "{role} signal has {} bytes but requires {required}",
                activation.signal_byte_capacity
            ),
        ));
    }
    Ok(activation)
}

fn distributed_activation(
    dispatch: &VulkanPreparedDispatch,
    binding: usize,
    required: usize,
    role: &str,
    edge_placements: &[ComponentEdgePlacement],
) -> Result<Option<VulkanDistributedActivationSlot>, VulkanDistributedPlanError> {
    let descriptor = dispatch
        .descriptors
        .iter()
        .find(|descriptor| descriptor.binding == binding)
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("has no resident {role} descriptor at binding {binding}"),
            )
        })?;
    let activation = match &descriptor.resource {
        VulkanDescriptorResourceAddress::ActivationSlot {
            component_id,
            signal_id,
            slot,
            byte_capacity,
            signal_byte_capacity,
        } => VulkanDistributedActivationSlot {
            binding,
            component_id: component_id.clone(),
            signal_id: signal_id.clone(),
            slot: *slot,
            byte_capacity: *byte_capacity,
            signal_byte_capacity: *signal_byte_capacity,
            storage: VulkanDistributedActivationStorage::ActivationSlot,
        },
        VulkanDescriptorResourceAddress::BoundaryInput { signal_id } => {
            let matching = edge_placements
                .iter()
                .filter(|edge| {
                    edge.destination_component_id == dispatch.component_id
                        && (edge.destination_port_id == *signal_id
                            || edge.destination_component_port.as_deref()
                                == Some(signal_id.as_str()))
                })
                .collect::<Vec<_>>();
            let [edge] = matching.as_slice() else {
                if matching.is_empty() {
                    return Ok(None);
                }
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "{role} boundary signal {signal_id:?} maps to {} runtime edges",
                        matching.len()
                    ),
                ));
            };
            VulkanDistributedActivationSlot {
                binding,
                component_id: dispatch.component_id.clone(),
                signal_id: signal_id.clone(),
                slot: edge.edge_index,
                byte_capacity: required,
                signal_byte_capacity: required,
                storage: VulkanDistributedActivationStorage::Edge {
                    edge_index: edge.edge_index,
                    owner_device_id: edge.source_device_id.clone(),
                },
            }
        }
        VulkanDescriptorResourceAddress::BoundaryOutput { signal_id } => {
            let matching = edge_placements
                .iter()
                .filter(|edge| {
                    edge.source_component_id == dispatch.component_id
                        && (edge.source_port_id == *signal_id
                            || edge.source_component_port.as_deref() == Some(signal_id.as_str()))
                })
                .collect::<Vec<_>>();
            let [edge] = matching.as_slice() else {
                if matching.is_empty() {
                    return Ok(None);
                }
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "{role} boundary signal {signal_id:?} maps to {} runtime edges",
                        matching.len()
                    ),
                ));
            };
            VulkanDistributedActivationSlot {
                binding,
                component_id: dispatch.component_id.clone(),
                signal_id: signal_id.clone(),
                slot: edge.edge_index,
                byte_capacity: required,
                signal_byte_capacity: required,
                storage: VulkanDistributedActivationStorage::Edge {
                    edge_index: edge.edge_index,
                    owner_device_id: edge.source_device_id.clone(),
                },
            }
        }
        _ => {
            return Err(dispatch_error(
                dispatch,
                format!("{role} binding {binding} is not a resident signal"),
            ));
        }
    };
    if activation.signal_byte_capacity < required {
        return Err(dispatch_error(
            dispatch,
            format!(
                "{role} signal has {} bytes but requires {required}",
                activation.signal_byte_capacity
            ),
        ));
    }
    Ok(Some(activation))
}

fn activation_slots_for_usage(
    dispatch: &VulkanPreparedDispatch,
    usage: VulkanKernelDescriptorUsage,
    role: &str,
) -> Result<Vec<VulkanDistributedActivationSlot>, VulkanDistributedPlanError> {
    let activations = dispatch
        .descriptors
        .iter()
        .filter(|descriptor| descriptor.usage == usage)
        .map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::ActivationSlot {
                component_id,
                signal_id,
                slot,
                byte_capacity,
                signal_byte_capacity,
            } => Ok(VulkanDistributedActivationSlot {
                binding: descriptor.binding,
                component_id: component_id.clone(),
                signal_id: signal_id.clone(),
                slot: *slot,
                byte_capacity: *byte_capacity,
                signal_byte_capacity: *signal_byte_capacity,
                storage: VulkanDistributedActivationStorage::ActivationSlot,
            }),
            _ => Err(dispatch_error(
                dispatch,
                format!(
                    "{role} descriptor {} is not a resident activation",
                    descriptor.binding
                ),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if activations.is_empty() {
        return Err(dispatch_error(
            dispatch,
            format!("has no resident {role} activations"),
        ));
    }
    Ok(activations)
}

fn distribute_rows(
    row_count: usize,
    requested_shards: usize,
    workgroup_row_count: usize,
    shard_boundary_row_alignment: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if row_count == 0
        || requested_shards == 0
        || workgroup_row_count == 0
        || shard_boundary_row_alignment == 0
    {
        return Err("row distribution dimensions must not be zero".to_string());
    }
    if !row_count.is_multiple_of(workgroup_row_count)
        || !shard_boundary_row_alignment.is_multiple_of(workgroup_row_count)
    {
        return Err(format!(
            "row count {row_count} and shard boundary {shard_boundary_row_alignment} are incompatible with workgroup width {workgroup_row_count}"
        ));
    }
    let aligned_groups = row_count / shard_boundary_row_alignment;
    let tail_rows = row_count % shard_boundary_row_alignment;
    let shard_count = requested_shards.min(aligned_groups + usize::from(tail_rows != 0));
    let groups_per_shard = aligned_groups / shard_count;
    let remainder = aligned_groups % shard_count;
    let mut row_start = 0usize;
    let mut shards = Vec::with_capacity(shard_count);
    for shard_index in 0..shard_count {
        let group_count = groups_per_shard + usize::from(shard_index < remainder);
        let shard_rows = group_count
            .checked_mul(shard_boundary_row_alignment)
            .and_then(|rows| {
                if shard_index + 1 == shard_count {
                    rows.checked_add(tail_rows)
                } else {
                    Some(rows)
                }
            })
            .ok_or_else(|| "row shard size overflowed".to_string())?;
        if shard_rows == 0 {
            return Err("row distribution produced an empty shard".to_string());
        }
        shards.push((row_start, shard_rows));
        row_start = row_start
            .checked_add(shard_rows)
            .ok_or_else(|| "row shard offset overflowed".to_string())?;
    }
    Ok(shards)
}

fn parameter_fragment(
    binding: usize,
    tensor: &str,
    row_bytes: usize,
    row_start: usize,
    row_count: usize,
    dispatch: &VulkanPreparedDispatch,
) -> Result<VulkanDistributedParameterFragment, VulkanDistributedPlanError> {
    Ok(VulkanDistributedParameterFragment {
        binding,
        tensor: tensor.to_string(),
        byte_offset: row_start.checked_mul(row_bytes).ok_or_else(|| {
            dispatch_error(dispatch, "parameter shard offset overflowed".to_string())
        })?,
        byte_count: row_count.checked_mul(row_bytes).ok_or_else(|| {
            dispatch_error(
                dispatch,
                "parameter shard byte count overflowed".to_string(),
            )
        })?,
    })
}

fn least_common_multiple(left: usize, right: usize) -> Option<usize> {
    left.checked_mul(right / greatest_common_divisor(left, right))
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn dispatch_error(
    dispatch: &VulkanPreparedDispatch,
    message: String,
) -> VulkanDistributedPlanError {
    VulkanDistributedPlanError(format!(
        "distributed dispatch {}.{} {message}",
        dispatch.component_id, dispatch.node_id
    ))
}
