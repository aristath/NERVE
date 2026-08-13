fn distributed_component_batch_demand_resolution_bound(
    resource_domain_counts: impl IntoIterator<Item = usize>,
) -> Result<usize, VulkanError> {
    let mut saw_domain = false;
    let mut bound = 1usize;
    for count in resource_domain_counts {
        if count == 0 {
            return Err(VulkanError(
                "distributed component batch residency has an empty resource domain".to_string(),
            ));
        }
        saw_domain = true;
        bound = bound.checked_add(count).ok_or_else(|| {
            VulkanError("distributed component batch residency bound overflowed".to_string())
        })?;
    }
    if !saw_domain {
        return Err(VulkanError(
            "distributed component batch residency has no resource domains".to_string(),
        ));
    }
    Ok(bound)
}

fn record_distributed_component_batch_demand_resolution(
    resolved: &mut BTreeMap<(usize, usize), BTreeSet<usize>>,
    checkpoint: (usize, usize),
    device_id: &str,
    resource_indices: &[usize],
) -> Result<(), VulkanError> {
    if resource_indices.is_empty() {
        return Err(VulkanError(format!(
            "distributed component batch residency checkpoint {checkpoint:?} on {device_id:?} resolved no resources"
        )));
    }
    let prior = resolved.entry(checkpoint).or_default();
    let repeated = resource_indices
        .iter()
        .copied()
        .filter(|resource_index| prior.contains(resource_index))
        .collect::<Vec<_>>();
    if !repeated.is_empty() {
        return Err(VulkanError(format!(
            "distributed component batch residency checkpoint {checkpoint:?} on {device_id:?} repeated loaded resources {repeated:?}"
        )));
    }
    prior.extend(resource_indices.iter().copied());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn mount_distributed_component_batch_selected_resource_gates(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    placed_slices: &[VulkanResidentInProcessPlacedStreamProcessorDevice],
    batch_slices: &[VulkanResidentComponentBatchSliceRunner],
    dynamic_resource_buffers: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    resource_stores: &BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
    lane_capacity: usize,
    planned_island: &VulkanPhysicalExecutionIslandPlan,
    runner: &mut VulkanDistributedComponentBatchDispatchRunner,
) -> Result<(), VulkanResidentInProcessPlacedRuntimeError> {
    let leader = planned_island.leader();
    if leader.selected_resource_partitions.is_empty() {
        return Ok(());
    }
    let owner_index = placed_slices
        .iter()
        .position(|slice| slice.device_id == planned_island.owner_device_id)
        .ok_or_else(|| VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
            device_id: planned_island.owner_device_id.clone(),
        })?;
    let batch_slice = batch_slices.get(owner_index).ok_or_else(|| {
        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
            "distributed component batch selected-resource owner has no batch slice".to_string(),
        ))
    })?;
    for (shard_index, (planned_shard, shard)) in
        leader.shards.iter().zip(&mut runner.shards).enumerate()
    {
        let store = resource_stores.get(&planned_shard.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                "distributed component batch selected-resource island has no resource store on {:?}",
                planned_shard.device_id,
            )))
        })?;
        if !store.residency_policy().is_demand_loaded() {
            continue;
        }
        let device = devices.get(&planned_shard.device_id).ok_or_else(|| {
            VulkanResidentInProcessPlacedRuntimeError::MissingBoundDevice {
                device_id: planned_shard.device_id.clone(),
            }
        })?;
        let dynamic_resources = dynamic_resource_buffers
            .get(&planned_shard.device_id)
            .ok_or_else(|| {
                VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(format!(
                    "distributed component batch selected-resource island has no dynamic buffers on {:?}",
                    planned_shard.device_id,
                )))
            })?;
        let predicate = Arc::new(
            device
                .create_conditional_resident_buffer(size_of::<u32>())
                .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?,
        );
        predicate
            .write_bytes(&1u32.to_le_bytes())
            .map_err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop)?;
        let gates = leader
            .selected_resource_partitions
            .iter()
            .enumerate()
            .map(|(partition_index, partition)| {
                let selection_activation =
                    selected_resource_activation(leader, &partition.selection_signal).map_err(
                        |error| {
                            VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                                error.to_string(),
                            ))
                        },
                    )?;
                let selection_key = distributed_component_batch_signal_key(
                    selection_activation,
                    &batch_slice.signal_buffer_indices,
                )?;
                let selection_buffer = batch_slice
                    .distributed_signal_buffer(&selection_key, &planned_shard.device_id)?
                    .clone();
                VulkanDistributedSelectedResourceGate::new(
                    device,
                    &planned_shard.device_id,
                    &partition.execution_scope,
                    leader,
                    partition,
                    selection_buffer,
                    selection_activation.signal_byte_capacity,
                    lane_capacity,
                    dynamic_resources,
                    Arc::clone(store),
                    Arc::clone(&predicate),
                    Arc::clone(&predicate),
                    u32::try_from(partition_index + 1).map_err(|_| {
                        VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                            "distributed component batch checkpoint tag exceeds u32".to_string(),
                        ))
                    })?,
                )
                .map_err(|error| {
                    VulkanResidentInProcessPlacedRuntimeError::BackendLoop(VulkanError(
                        error.to_string(),
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if gates.is_empty() {
            return Err(VulkanResidentInProcessPlacedRuntimeError::BackendLoop(
                VulkanError(format!(
                    "distributed component batch demand shard {shard_index} has no residency gates"
                )),
            ));
        }
        shard.selected_resource_gates = gates;
    }
    Ok(())
}
