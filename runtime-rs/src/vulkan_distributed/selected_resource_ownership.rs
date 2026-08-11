#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceStorePlan {
    pub devices: Vec<VulkanDistributedSelectedResourceDevicePlan>,
    pub device_count: usize,
    pub selector_count: usize,
    pub selector_placement_count: usize,
    pub unique_atomic_group_count: usize,
    pub total_addressable_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceDevicePlan {
    pub device_id: String,
    pub selectors: Vec<VulkanDistributedSelectedResourceOwnership>,
    pub unique_atomic_group_count: usize,
    pub maximum_atomic_group_bytes: usize,
    pub maximum_load_wave_bytes: usize,
    pub total_addressable_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedSelectedResourceOwnership {
    pub execution_scope: String,
    pub selector_id: String,
    pub component_id: String,
    pub selection_signal: String,
    pub resource_count: usize,
    pub selection_count_per_activation: usize,
    pub owned_resource_indices: Vec<usize>,
    pub atomic_group_ids: Vec<String>,
    pub atomic_group_byte_counts: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanDistributedSelectedResourceSelectorIdentity {
    execution_scope: String,
    component_id: String,
    selection_signal: String,
    resource_count: usize,
    selection_count_per_activation: usize,
    atomic_group_ids: Vec<String>,
    atomic_group_byte_counts: Vec<usize>,
}

impl VulkanDistributedSelectedResourceStorePlan {
    pub fn from_execution_plan_set(
        plans: &VulkanDistributedExecutionPlanSet,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let decode = Self::from_execution_plan(&plans.decode)?;
        for candidate in [&plans.decode_batch, &plans.prefill] {
            let candidate = Self::from_execution_plan(candidate)?;
            if candidate != decode {
                return Err(VulkanDistributedPlanError(
                    "single-lane decode, multi-lane decode, and multi-lane prefill assign different selected resources"
                        .to_string(),
                ));
            }
        }
        Ok(decode)
    }

    pub fn from_execution_plan(
        plan: &VulkanDistributedExecutionPlan,
    ) -> Result<Self, VulkanDistributedPlanError> {
        let execution_devices = plan
            .device_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let mut identities = BTreeMap::<
            String,
            VulkanDistributedSelectedResourceSelectorIdentity,
        >::new();
        let mut group_devices = BTreeMap::<(String, String), String>::new();
        let mut resources_by_device_selector =
            BTreeMap::<(String, String), BTreeSet<usize>>::new();

        for dispatch in &plan.dispatches {
            for partition in &dispatch.selected_resource_partitions {
                validate_selected_resource_partition(dispatch, partition)?;
                let identity = VulkanDistributedSelectedResourceSelectorIdentity {
                    execution_scope: partition.execution_scope.clone(),
                    component_id: dispatch.component_id.clone(),
                    selection_signal: partition.selection_signal.clone(),
                    resource_count: partition.resource_count,
                    selection_count_per_activation: partition.selection_count_per_activation,
                    atomic_group_ids: partition.atomic_group_ids.clone(),
                    atomic_group_byte_counts: partition.atomic_group_byte_counts.clone(),
                };
                if let Some(existing) = identities.insert(partition.selector_id.clone(), identity.clone())
                    && existing != identity
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} changes identity between distributed dispatches",
                        partition.selector_id
                    )));
                }

                let mut coverage = vec![0u8; partition.resource_count];
                for shard in &dispatch.shards {
                    if !execution_devices.contains(shard.device_id.as_str()) {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected resource shard for {}.{} uses device {:?} outside the execution pool",
                            dispatch.component_id, dispatch.node_id, shard.device_id
                        )));
                    }
                    let end = shard.row_start.checked_add(shard.row_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "selected resource shard range overflowed".to_string(),
                        )
                    })?;
                    if shard.row_count == 0 || end > partition.resource_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected resource shard {}..{} exceeds selector {:?} resource count {}",
                            shard.row_start, end, partition.selector_id, partition.resource_count
                        )));
                    }
                    let owned = resources_by_device_selector
                        .entry((shard.device_id.clone(), partition.selector_id.clone()))
                        .or_default();
                    for resource_index in shard.row_start..end {
                        coverage[resource_index] = coverage[resource_index]
                            .checked_add(1)
                            .ok_or_else(|| {
                                VulkanDistributedPlanError(
                                    "selected resource coverage count overflowed".to_string(),
                                )
                            })?;
                        let group_id = partition.atomic_group_ids[resource_index].clone();
                        let group_key = (partition.selector_id.clone(), group_id.clone());
                        if let Some(existing_device) =
                            group_devices.insert(group_key, shard.device_id.clone())
                            && existing_device != shard.device_id
                        {
                            return Err(VulkanDistributedPlanError(format!(
                                "selected atomic group {group_id:?} moves from device {existing_device:?} to {:?} between distributed dispatches",
                                shard.device_id
                            )));
                        }
                        owned.insert(resource_index);
                    }
                }
                if coverage.iter().any(|count| *count != 1) {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected resource selector {:?} is not partitioned exactly once across its shards",
                        partition.selector_id
                    )));
                }
            }
        }

        let mut device_selectors = BTreeMap::<
            String,
            Vec<VulkanDistributedSelectedResourceOwnership>,
        >::new();
        for ((device_id, selector_id), indices) in resources_by_device_selector {
            let identity = identities.get(&selector_id).expect(
                "selected resource ownership was created from a validated selector identity",
            );
            let owned_resource_indices = indices.into_iter().collect::<Vec<_>>();
            let atomic_group_ids = owned_resource_indices
                .iter()
                .map(|index| identity.atomic_group_ids[*index].clone())
                .collect::<Vec<_>>();
            let atomic_group_byte_counts = owned_resource_indices
                .iter()
                .map(|index| identity.atomic_group_byte_counts[*index])
                .collect::<Vec<_>>();
            device_selectors.entry(device_id).or_default().push(
                VulkanDistributedSelectedResourceOwnership {
                    execution_scope: identity.execution_scope.clone(),
                    selector_id,
                    component_id: identity.component_id.clone(),
                    selection_signal: identity.selection_signal.clone(),
                    resource_count: identity.resource_count,
                    selection_count_per_activation: identity.selection_count_per_activation,
                    owned_resource_indices,
                    atomic_group_ids,
                    atomic_group_byte_counts,
                },
            );
        }

        let mut devices = Vec::with_capacity(device_selectors.len());
        let mut global_groups = BTreeMap::<String, usize>::new();
        let selector_count = identities.len();
        let mut selector_placement_count = 0usize;
        let mut total_addressable_bytes = 0usize;
        for (device_id, mut selectors) in device_selectors {
            selectors.sort_by(|left, right| left.selector_id.cmp(&right.selector_id));
            selector_placement_count = selector_placement_count.checked_add(selectors.len()).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected resource selector count overflowed".to_string(),
                )
            })?;
            let mut device_groups = BTreeMap::<String, usize>::new();
            let mut maximum_load_wave_bytes = 0usize;
            for selector in &selectors {
                let mut selector_group_bytes = selector
                    .atomic_group_ids
                    .iter()
                    .cloned()
                    .zip(selector.atomic_group_byte_counts.iter().copied())
                    .collect::<Vec<_>>();
                selector_group_bytes.sort_by_key(|(_, bytes)| std::cmp::Reverse(*bytes));
                let selection_count = selector
                    .selection_count_per_activation
                    .min(selector_group_bytes.len());
                let load_wave_bytes = selector_group_bytes
                    .iter()
                    .take(selection_count)
                    .try_fold(0usize, |total, (_, bytes)| total.checked_add(*bytes))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "selected resource load-wave capacity overflowed".to_string(),
                        )
                    })?;
                maximum_load_wave_bytes = maximum_load_wave_bytes.max(load_wave_bytes);
                for (group_id, bytes) in selector_group_bytes {
                    insert_group_bytes(&mut device_groups, &group_id, bytes)?;
                    insert_group_bytes(&mut global_groups, &group_id, bytes)?;
                }
            }
            let device_addressable_bytes = device_groups.values().try_fold(
                0usize,
                |total, bytes| total.checked_add(*bytes),
            ).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected resource device capacity overflowed".to_string(),
                )
            })?;
            total_addressable_bytes = total_addressable_bytes
                .checked_add(device_addressable_bytes)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected resource total capacity overflowed".to_string(),
                    )
                })?;
            devices.push(VulkanDistributedSelectedResourceDevicePlan {
                device_id,
                unique_atomic_group_count: device_groups.len(),
                maximum_atomic_group_bytes: device_groups.values().copied().max().unwrap_or(0),
                maximum_load_wave_bytes,
                total_addressable_bytes: device_addressable_bytes,
                selectors,
            });
        }
        Ok(Self {
            device_count: devices.len(),
            selector_count,
            selector_placement_count,
            unique_atomic_group_count: global_groups.len(),
            total_addressable_bytes,
            devices,
        })
    }

    pub fn device(
        &self,
        device_id: &str,
    ) -> Option<&VulkanDistributedSelectedResourceDevicePlan> {
        self.devices.iter().find(|device| device.device_id == device_id)
    }
}

fn validate_selected_resource_partition(
    dispatch: &VulkanDistributedDispatchPlan,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
) -> Result<(), VulkanDistributedPlanError> {
    if partition.execution_scope.trim().is_empty()
        || partition.selector_id.trim().is_empty()
        || partition.selection_signal.trim().is_empty()
        || partition.resource_count == 0
        || partition.parameters_per_resource == 0
        || partition.selection_count_per_activation == 0
        || partition.selection_count_per_activation > partition.resource_count
        || partition.atomic_group_ids.len() != partition.resource_count
        || partition.atomic_group_byte_counts.len() != partition.resource_count
        || partition.atomic_group_ids.iter().any(|id| id.trim().is_empty())
        || partition.atomic_group_byte_counts.iter().any(|bytes| *bytes == 0)
    {
        return Err(VulkanDistributedPlanError(format!(
            "distributed dispatch {}.{} has an invalid selected resource partition",
            dispatch.component_id, dispatch.node_id
        )));
    }
    Ok(())
}

fn insert_group_bytes(
    groups: &mut BTreeMap<String, usize>,
    group_id: &str,
    bytes: usize,
) -> Result<(), VulkanDistributedPlanError> {
    if let Some(previous) = groups.insert(group_id.to_string(), bytes)
        && previous != bytes
    {
        return Err(VulkanDistributedPlanError(format!(
            "selected atomic group {group_id:?} has conflicting byte counts {previous} and {bytes}",
        )));
    }
    Ok(())
}
