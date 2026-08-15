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
        VulkanDistributedActivationStorage::ActivationSlot
        | VulkanDistributedActivationStorage::BoundaryInput
        | VulkanDistributedActivationStorage::BoundaryOutput => owner_device_id,
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

fn distributed_activation(
    dispatch: &VulkanPreparedDispatch,
    binding: usize,
    required: usize,
    role: &str,
    edge_placements: &[ComponentEdgePlacement],
    activation_element_bytes: usize,
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
                    return Ok(Some(VulkanDistributedActivationSlot {
                        binding,
                        component_id: dispatch.component_id.clone(),
                        signal_id: signal_id.clone(),
                        slot: binding,
                        byte_capacity: required,
                        signal_byte_capacity: required,
                        storage: VulkanDistributedActivationStorage::BoundaryInput,
                    }));
                }
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "{role} boundary signal {signal_id:?} maps to {} runtime edges",
                        matching.len()
                    ),
                ));
            };
            let byte_capacity = distributed_boundary_edge_byte_capacity(
                dispatch,
                edge,
                activation_element_bytes,
            )?;
            VulkanDistributedActivationSlot {
                binding,
                component_id: dispatch.component_id.clone(),
                signal_id: signal_id.clone(),
                slot: edge.edge_index,
                byte_capacity,
                signal_byte_capacity: byte_capacity,
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
                    return Ok(Some(VulkanDistributedActivationSlot {
                        binding,
                        component_id: dispatch.component_id.clone(),
                        signal_id: signal_id.clone(),
                        slot: binding,
                        byte_capacity: required,
                        signal_byte_capacity: required,
                        storage: VulkanDistributedActivationStorage::BoundaryOutput,
                    }));
                }
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "{role} boundary signal {signal_id:?} maps to {} runtime edges",
                        matching.len()
                    ),
                ));
            };
            let byte_capacity = distributed_boundary_edge_byte_capacity(
                dispatch,
                edge,
                activation_element_bytes,
            )?;
            VulkanDistributedActivationSlot {
                binding,
                component_id: dispatch.component_id.clone(),
                signal_id: signal_id.clone(),
                slot: edge.edge_index,
                byte_capacity,
                signal_byte_capacity: byte_capacity,
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

fn distributed_boundary_edge_byte_capacity(
    dispatch: &VulkanPreparedDispatch,
    edge: &ComponentEdgePlacement,
    activation_element_bytes: usize,
) -> Result<usize, VulkanDistributedPlanError> {
    if activation_element_bytes == 0 || edge.shape.is_empty() {
        return Err(dispatch_error(
            dispatch,
            "distributed boundary edge has no typed activation extent".to_string(),
        ));
    }
    edge.shape
        .iter()
        .try_fold(activation_element_bytes, |bytes, extent| {
            bytes.checked_mul(*extent).ok_or_else(|| {
                dispatch_error(
                    dispatch,
                    "distributed boundary edge byte capacity overflowed".to_string(),
                )
            })
        })
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
