#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourcePlacementDevice {
    pub device_id: String,
    pub physical_device_id: String,
    pub api_version: u32,
    pub driver_version: u32,
    /// Payload bytes that this device may safely keep resident after current
    /// reservations and fixed runtime allocations have been deducted.
    pub resident_payload_capacity_bytes: usize,
    /// Exact measured costs keyed by compiler-declared structural execution
    /// class. Missing classes are unavailable; costs are never inferred from
    /// another representation, device, or average component duration.
    pub measured_costs_by_execution_class:
        BTreeMap<String, VulkanSelectedResourceExecutionClassCost>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceExecutionClassCost {
    pub phase: nerve_execution_contracts::ExecutionPhase,
    pub complete_transaction: bool,
    pub output_valid: bool,
    pub warmup_call_count: usize,
    pub measured_call_count: usize,
    pub execution_duration_ns: u64,
    pub lazy_load_wave_duration_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceAssignment {
    pub resource_index: usize,
    pub device_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceDeviceLoad {
    pub device_id: String,
    /// Total immutable payload addressable through this device's selector
    /// tables. Demand-paged ownership may exceed resident capacity.
    pub addressable_bytes: usize,
    /// Conservative maximum payload required by one selector activation.
    pub maximum_load_wave_bytes: usize,
    pub first_moment_ns: u128,
    pub second_moment_ns2: u128,
    pub owned_resource_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourcePlacementPlan {
    pub selector_id: String,
    pub assignments: Vec<VulkanSelectedResourceAssignment>,
    pub device_loads: Vec<VulkanSelectedResourceDeviceLoad>,
    pub maximum_first_moment_ns: u128,
    pub maximum_second_moment_ns2: u128,
}

#[derive(Clone, Debug)]
struct VulkanSelectedResourceMutableDeviceLoad {
    device_id: String,
    resident_payload_capacity_bytes: usize,
    addressable_bytes: usize,
    maximum_load_wave_bytes: usize,
    first_moment_ns: u128,
    second_moment_ns2: u128,
    owned_resource_indices: Vec<usize>,
}

/// Places one selector's atomic resources using exact measured device costs.
///
/// The first moment is total expected serialized work on each device. The
/// second moment additionally uses joint selection counts, so resources that
/// commonly execute together are not treated as independent average load.
/// Minimizing the largest second moment is a deterministic, pairwise-complete
/// proxy for the critical path without inventing unobserved route sets.
pub fn plan_selected_resource_placement(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<VulkanSelectedResourcePlacementPlan, VulkanDistributedPlanError> {
    validate_selected_resource_placement_problem(
        component_id,
        partition,
        telemetry,
        devices,
        phase,
    )?;

    let joint_selection_totals = (0..partition.resource_count)
        .map(|resource_index| {
            (0..partition.resource_count)
                .filter(|other| *other != resource_index)
                .try_fold(0u64, |total, other| {
                    total.checked_add(
                        telemetry
                            .co_selection_count(resource_index, other)
                            .unwrap_or(0),
                    )
                })
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource joint selection total overflowed".to_string(),
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut resource_order = (0..partition.resource_count).collect::<Vec<_>>();
    resource_order.sort_by(|left, right| {
        telemetry.selection_counts[*right]
            .cmp(&telemetry.selection_counts[*left])
            .then_with(|| joint_selection_totals[*right].cmp(&joint_selection_totals[*left]))
            .then_with(|| {
                partition.atomic_group_byte_counts[*right]
                    .cmp(&partition.atomic_group_byte_counts[*left])
            })
            .then_with(|| left.cmp(right))
    });

    let mut loads = devices
        .iter()
        .map(|device| VulkanSelectedResourceMutableDeviceLoad {
            device_id: device.device_id.clone(),
            resident_payload_capacity_bytes: device.resident_payload_capacity_bytes,
            addressable_bytes: 0,
            maximum_load_wave_bytes: 0,
            first_moment_ns: 0,
            second_moment_ns2: 0,
            owned_resource_indices: Vec::new(),
        })
        .collect::<Vec<_>>();

    for resource_index in resource_order {
        let resource_bytes = partition.atomic_group_byte_counts[resource_index];
        let selection_count = u128::from(telemetry.selection_counts[resource_index]);
        let mut candidates = Vec::new();
        for (device_index, (device, load)) in devices.iter().zip(&loads).enumerate() {
            let Some(addressable_bytes) = load.addressable_bytes.checked_add(resource_bytes) else {
                continue;
            };
            let maximum_load_wave_bytes = selected_resource_maximum_load_wave_bytes(
                partition,
                load.owned_resource_indices
                    .iter()
                    .copied()
                    .chain(std::iter::once(resource_index)),
            )?;
            let required_resident_bytes = match residency_policy {
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager
                | crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained => {
                    addressable_bytes
                }
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged => {
                    maximum_load_wave_bytes
                }
            };
            if required_resident_bytes > load.resident_payload_capacity_bytes {
                continue;
            }
            let execution_class_id = &partition.resource_execution_class_ids[resource_index];
            let duration = u128::from(
                device.measured_costs_by_execution_class[execution_class_id]
                    .execution_duration_ns,
            );
            let first_contribution = selection_count.checked_mul(duration).ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected-resource first-moment cost overflowed".to_string(),
                )
            })?;
            let self_second = selection_count
                .checked_mul(duration)
                .and_then(|value| value.checked_mul(duration))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource second-moment cost overflowed".to_string(),
                    )
                })?;
            let pair_second = load.owned_resource_indices.iter().try_fold(
                0u128,
                |total, other_resource| {
                    let joint = u128::from(
                        telemetry
                            .co_selection_count(resource_index, *other_resource)
                            .unwrap_or(0),
                    );
                    let other_class_id =
                        &partition.resource_execution_class_ids[*other_resource];
                    let other_duration = u128::from(
                        device.measured_costs_by_execution_class[other_class_id]
                            .execution_duration_ns,
                    );
                    joint
                        .checked_mul(duration)
                        .and_then(|value| value.checked_mul(other_duration))
                        .and_then(|value| value.checked_mul(2))
                        .and_then(|value| total.checked_add(value))
                        .ok_or_else(|| {
                            VulkanDistributedPlanError(
                                "selected-resource joint second-moment cost overflowed"
                                    .to_string(),
                            )
                        })
                },
            )?;
            let projected_first = load
                .first_moment_ns
                .checked_add(first_contribution)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource projected first moment overflowed".to_string(),
                    )
                })?;
            let projected_second = load
                .second_moment_ns2
                .checked_add(self_second)
                .and_then(|value| value.checked_add(pair_second))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource projected second moment overflowed".to_string(),
                    )
                })?;
            let maximum_first = loads
                .iter()
                .enumerate()
                .map(|(index, load)| {
                    if index == device_index {
                        projected_first
                    } else {
                        load.first_moment_ns
                    }
                })
                .max()
                .unwrap_or(projected_first);
            let maximum_second = loads
                .iter()
                .enumerate()
                .map(|(index, load)| {
                    if index == device_index {
                        projected_second
                    } else {
                        load.second_moment_ns2
                    }
                })
                .max()
                .unwrap_or(projected_second);
            candidates.push((
                maximum_second,
                maximum_first,
                std::cmp::Reverse(
                    load.resident_payload_capacity_bytes - required_resident_bytes,
                ),
                device.device_id.as_str(),
                device_index,
                addressable_bytes,
                maximum_load_wave_bytes,
                projected_first,
                projected_second,
            ));
        }
        candidates.sort();
        let Some((
            _,
            _,
            _,
            _,
            device_index,
            addressable_bytes,
            maximum_load_wave_bytes,
            projected_first,
            projected_second,
        )) = candidates.into_iter().next()
        else {
            return Err(VulkanDistributedPlanError(format!(
                "selected resource {} for selector {:?} has no measured device whose resident payload capacity admits its {:?} residency requirement",
                resource_index, partition.selector_id, residency_policy,
            )));
        };
        let load = &mut loads[device_index];
        load.addressable_bytes = addressable_bytes;
        load.maximum_load_wave_bytes = maximum_load_wave_bytes;
        load.first_moment_ns = projected_first;
        load.second_moment_ns2 = projected_second;
        load.owned_resource_indices.push(resource_index);
    }

    for load in &mut loads {
        load.owned_resource_indices.sort_unstable();
    }
    let mut assignments = loads
        .iter()
        .flat_map(|load| {
            load.owned_resource_indices.iter().map(|resource_index| {
                VulkanSelectedResourceAssignment {
                    resource_index: *resource_index,
                    device_id: load.device_id.clone(),
                }
            })
        })
        .collect::<Vec<_>>();
    assignments.sort_by_key(|assignment| assignment.resource_index);
    let maximum_first_moment_ns = loads
        .iter()
        .map(|load| load.first_moment_ns)
        .max()
        .unwrap_or(0);
    let maximum_second_moment_ns2 = loads
        .iter()
        .map(|load| load.second_moment_ns2)
        .max()
        .unwrap_or(0);
    let device_loads = loads
        .into_iter()
        .map(|load| VulkanSelectedResourceDeviceLoad {
            device_id: load.device_id,
            addressable_bytes: load.addressable_bytes,
            maximum_load_wave_bytes: load.maximum_load_wave_bytes,
            first_moment_ns: load.first_moment_ns,
            second_moment_ns2: load.second_moment_ns2,
            owned_resource_indices: load.owned_resource_indices,
        })
        .collect();
    Ok(VulkanSelectedResourcePlacementPlan {
        selector_id: partition.selector_id.clone(),
        assignments,
        device_loads,
        maximum_first_moment_ns,
        maximum_second_moment_ns2,
    })
}

fn selected_resource_maximum_load_wave_bytes(
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    owned_resource_indices: impl IntoIterator<Item = usize>,
) -> Result<usize, VulkanDistributedPlanError> {
    let mut byte_counts = owned_resource_indices
        .into_iter()
        .map(|resource_index| partition.atomic_group_byte_counts[resource_index])
        .collect::<Vec<_>>();
    byte_counts.sort_unstable_by(|left, right| right.cmp(left));
    byte_counts
        .into_iter()
        .take(partition.selection_count_per_activation)
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| {
            VulkanDistributedPlanError(
                "selected-resource maximum load-wave bytes overflowed".to_string(),
            )
        })
}

impl VulkanDistributedExecutionPlanSet {
    /// Applies one measured ownership decision identically to decode and
    /// prefill. This rewires physical expert ownership only; logical selector
    /// identity and graph wiring remain unchanged.
    pub fn apply_selected_resource_placements(
        &mut self,
        placements: &[VulkanSelectedResourcePlacementPlan],
    ) -> Result<(), VulkanDistributedPlanError> {
        let placement_by_selector = canonical_selected_resource_placements(placements)?;
        apply_selected_resource_placements_to_phase(
            &mut self.decode,
            &placement_by_selector,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )?;
        apply_selected_resource_placements_to_phase(
            &mut self.decode_batch,
            &placement_by_selector,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )?;
        apply_selected_resource_placements_to_phase(
            &mut self.prefill,
            &placement_by_selector,
            nerve_execution_contracts::ExecutionPhase::Prefill,
        )?;
        VulkanDistributedSelectedResourceStorePlan::from_execution_plan_set(self)?;
        Ok(())
    }
}

fn canonical_selected_resource_placements<'a>(
    placements: &'a [VulkanSelectedResourcePlacementPlan],
) -> Result<BTreeMap<&'a str, &'a VulkanSelectedResourcePlacementPlan>, VulkanDistributedPlanError>
{
    let mut by_selector = BTreeMap::new();
    for placement in placements {
        if placement.selector_id.is_empty()
            || placement.assignments.is_empty()
            || by_selector
                .insert(placement.selector_id.as_str(), placement)
                .is_some()
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource placements require unique selector IDs and assignments"
                    .to_string(),
            ));
        }
        let mut resources = BTreeSet::new();
        if placement.assignments.iter().any(|assignment| {
            assignment.device_id.is_empty() || !resources.insert(assignment.resource_index)
        }) {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource placement {:?} repeats a resource or has an empty device",
                placement.selector_id,
            )));
        }
    }
    Ok(by_selector)
}

fn apply_selected_resource_placements_to_phase(
    execution_plan: &mut VulkanDistributedExecutionPlan,
    placements: &BTreeMap<&str, &VulkanSelectedResourcePlacementPlan>,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<(), VulkanDistributedPlanError> {
    let mut applied_selectors = BTreeSet::new();
    for dispatch in &mut execution_plan.dispatches {
        if dispatch.selected_resource_partitions.is_empty()
            || !dispatch
                .selected_resource_partitions
                .iter()
                .any(|partition| placements.contains_key(partition.selector_id.as_str()))
        {
            continue;
        }
        if dispatch.distribution != VulkanDistributedDispatchDistribution::ExpertRange
            || dispatch.shards.iter().any(|shard| !shard.parameters.is_empty())
        {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource placement cannot rewrite non-atomic dispatch {}.{}",
                dispatch.component_id, dispatch.node_id,
            )));
        }
        let original_devices = dispatch
            .shards
            .iter()
            .map(|shard| shard.device_id.as_str())
            .collect::<BTreeSet<_>>();
        let templates = dispatch
            .shards
            .iter()
            .map(|shard| (shard.device_id.as_str(), shard))
            .collect::<BTreeMap<_, _>>();
        let mut ownership_by_device = BTreeMap::<String, BTreeMap<String, Vec<usize>>>::new();
        for partition in &dispatch.selected_resource_partitions {
            let selected = if let Some(placement) = placements.get(partition.selector_id.as_str()) {
                applied_selectors.insert(partition.selector_id.as_str());
                if placement.assignments.len() != partition.resource_count
                    || placement
                        .assignments
                        .iter()
                        .enumerate()
                        .any(|(index, assignment)| assignment.resource_index != index)
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected-resource placement {:?} does not cover resources 0..{} exactly once",
                        partition.selector_id, partition.resource_count,
                    )));
                }
                placement
                    .assignments
                    .iter()
                    .map(|assignment| (assignment.resource_index, assignment.device_id.clone()))
                    .collect::<Vec<_>>()
            } else {
                dispatch
                    .shards
                    .iter()
                    .flat_map(|shard| {
                        shard
                            .selected_resource_indices
                            .get(&partition.selector_id)
                            .into_iter()
                            .flatten()
                            .map(|index| (*index, shard.device_id.clone()))
                    })
                    .collect::<Vec<_>>()
            };
            for (resource_index, device_id) in selected {
                if resource_index >= partition.resource_count
                    || !original_devices.contains(device_id.as_str())
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected-resource placement {:?} assigns resource {} outside its compiled participant pool",
                        partition.selector_id, resource_index,
                    )));
                }
                ownership_by_device
                    .entry(device_id)
                    .or_default()
                    .entry(partition.selector_id.clone())
                    .or_default()
                    .push(resource_index);
            }
        }
        if ownership_by_device.len() < 2 {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource placement for {}.{} leaves fewer than two distributed participants; select the single-device implementation instead",
                dispatch.component_id, dispatch.node_id,
            )));
        }
        let mut device_order = std::iter::once(dispatch.owner_device_id.as_str())
            .chain(
                dispatch
                    .shards
                    .iter()
                    .map(|shard| shard.device_id.as_str())
                    .filter(|device_id| *device_id != dispatch.owner_device_id),
            )
            .collect::<Vec<_>>();
        device_order.dedup();
        dispatch.shards = device_order
            .into_iter()
            .filter_map(|device_id| {
                let ownership = ownership_by_device.remove(device_id)?;
                let template = templates.get(device_id).copied().or_else(|| dispatch.shards.first())?;
                let mut shard = template.clone();
                shard.device_id = device_id.to_string();
                shard.selected_resource_indices = ownership;
                shard.row_start = 0;
                shard.row_count = dispatch.output_rows;
                shard.base_workgroup_z = 0;
                Some(shard)
            })
            .collect();
    }
    let missing = placements
        .keys()
        .copied()
        .filter(|selector_id| !applied_selectors.contains(selector_id))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(VulkanDistributedPlanError(format!(
            "selected-resource placements do not match this execution phase: {missing:?}",
        )));
    }
    execution_plan.device_ids = execution_plan
        .dispatches
        .iter()
        .flat_map(|dispatch| dispatch.shards.iter().map(|shard| shard.device_id.clone()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    execution_plan.execution_islands = resolved_physical_execution_islands_for_phase(
        &execution_plan.dispatches,
        execution_plan.shared_activation_route,
        phase,
    )?;
    Ok(())
}

fn validate_selected_resource_placement_problem(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<(), VulkanDistributedPlanError> {
    let expected_pair_count = if partition.selection_count_per_activation > 1 {
        partition
            .resource_count
            .checked_mul(partition.resource_count.saturating_sub(1))
            .and_then(|value| value.checked_div(2))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected-resource co-selection count overflowed".to_string(),
                )
            })?
    } else {
        0
    };
    if telemetry.execution_scope != partition.execution_scope
        || telemetry.component_id != component_id
        || telemetry.node_id != partition.node_id
        || telemetry.domain_id != partition.domain_id
        || telemetry.resource_count != partition.resource_count
        || telemetry.selection_counts.len() != partition.resource_count
        || telemetry.co_selection_counts.len() != expected_pair_count
    {
        return Err(VulkanDistributedPlanError(format!(
            "selection telemetry does not exactly match selector {:?}",
            partition.selector_id,
        )));
    }
    if partition.atomic_group_byte_counts.len() != partition.resource_count
        || partition.resource_execution_class_ids.len() != partition.resource_count
        || partition
            .resource_execution_class_ids
            .iter()
            .any(|class_id| !valid_selected_resource_execution_class_id(class_id))
        || partition
            .atomic_group_byte_counts
            .iter()
            .any(|bytes| *bytes == 0)
    {
        return Err(VulkanDistributedPlanError(format!(
            "selector {:?} has invalid resource execution classes or atomic byte counts",
            partition.selector_id,
        )));
    }
    if devices.is_empty() {
        return Err(VulkanDistributedPlanError(
            "selected-resource placement requires at least one device".to_string(),
        ));
    }
    let mut device_ids = BTreeSet::new();
    let mut physical_device_ids = BTreeSet::new();
    let required_classes = partition
        .resource_execution_class_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    for device in devices {
        if device.device_id.is_empty()
            || !device_ids.insert(device.device_id.as_str())
            || device.physical_device_id.is_empty()
            || !physical_device_ids.insert(device.physical_device_id.as_str())
            || device.api_version == 0
            || device.driver_version == 0
            || device.resident_payload_capacity_bytes == 0
            || device
                .measured_costs_by_execution_class
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>()
                != required_classes
            || device
                .measured_costs_by_execution_class
                .iter()
                .any(|(class_id, cost)| {
                    !valid_selected_resource_execution_class_id(class_id)
                        || cost.phase != phase
                        || !cost.complete_transaction
                        || !cost.output_valid
                        || cost.warmup_call_count == 0
                        || !(1..=2).contains(&cost.measured_call_count)
                        || cost.execution_duration_ns == 0
                        || cost.lazy_load_wave_duration_ns == 0
                })
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource placement devices require exact identities, positive capacity, and complete output-valid class measurements"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod selected_resource_placement_tests {
    use super::*;

    fn decode_cost(execution_duration_ns: u64) -> VulkanSelectedResourceExecutionClassCost {
        VulkanSelectedResourceExecutionClassCost {
            phase: nerve_execution_contracts::ExecutionPhase::Decode,
            complete_transaction: true,
            output_valid: true,
            warmup_call_count: 1,
            measured_call_count: 1,
            execution_duration_ns,
            lazy_load_wave_duration_ns: execution_duration_ns.saturating_mul(2),
        }
    }

    fn partition(resource_count: usize, selected: usize) -> VulkanDistributedSelectedResourcePartitionPlan {
        VulkanDistributedSelectedResourcePartitionPlan {
            execution_scope: "target".to_string(),
            selector_id: "selector".to_string(),
            node_id: "router".to_string(),
            domain_id: "experts".to_string(),
            selection_signal: "routes".to_string(),
            address_table_binding: 3,
            parameter_slots_binding: 4,
            resource_count,
            parameters_per_resource: 2,
            parameter_partitions: Vec::new(),
            selection_count_per_activation: selected,
            resource_execution_class_ids: vec![
                format!("sha256:{}", "a".repeat(64));
                resource_count
            ],
            atomic_group_ids: (0..resource_count)
                .map(|index| format!("expert_{index}"))
                .collect(),
            atomic_group_byte_counts: vec![10; resource_count],
            atomic_group_resource_ids: (0..resource_count)
                .map(|index| vec![format!("resource_{index}_0"), format!("resource_{index}_1")])
                .collect(),
            parameter_resource_ids: (0..resource_count)
                .map(|index| vec![format!("resource_{index}_0"), format!("resource_{index}_1")])
                .collect(),
            parameter_resource_byte_counts: vec![vec![5, 5]; resource_count],
        }
    }

    fn telemetry(
        selection_counts: Vec<u64>,
        co_selection_counts: Vec<u64>,
    ) -> crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot {
        crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot {
            execution_scope: "target".to_string(),
            component_id: "layer".to_string(),
            node_id: "router".to_string(),
            domain_id: "experts".to_string(),
            resource_count: selection_counts.len(),
            selection_counts,
            co_selection_counts,
        }
    }

    fn devices(_resource_count: usize, capacity: usize) -> Vec<VulkanSelectedResourcePlacementDevice> {
        ["a", "b"]
            .into_iter()
            .map(|device_id| VulkanSelectedResourcePlacementDevice {
                device_id: device_id.to_string(),
                physical_device_id: format!("physical-{device_id}"),
                api_version: 1,
                driver_version: 1,
                resident_payload_capacity_bytes: capacity,
                measured_costs_by_execution_class: BTreeMap::from([(
                    format!("sha256:{}", "a".repeat(64)),
                    decode_cost(10),
                )]),
            })
            .collect()
    }

    #[test]
    fn placement_separates_resources_that_are_selected_together() {
        let plan = plan_selected_resource_placement(
            "layer",
            &partition(4, 2),
            &telemetry(vec![100; 4], vec![100, 0, 0, 0, 0, 100]),
            &devices(4, 40),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap();
        let owner = |resource_index| {
            plan.assignments
                .iter()
                .find(|assignment| assignment.resource_index == resource_index)
                .unwrap()
                .device_id
                .as_str()
        };

        assert_ne!(owner(0), owner(1));
        assert_ne!(owner(2), owner(3));
        assert_eq!(plan.assignments.len(), 4);
        assert_eq!(plan.device_loads[0].addressable_bytes, 20);
        assert_eq!(plan.device_loads[0].maximum_load_wave_bytes, 20);
        assert_eq!(plan.device_loads[1].addressable_bytes, 20);
        assert_eq!(plan.device_loads[1].maximum_load_wave_bytes, 20);
    }

    #[test]
    fn placement_balances_hot_resources_before_cold_capacity_fill() {
        let plan = plan_selected_resource_placement(
            "layer",
            &partition(4, 2),
            &telemetry(vec![1_000, 900, 1, 1], vec![0; 6]),
            &devices(4, 30),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap();

        assert_ne!(
            plan.assignments[0].device_id,
            plan.assignments[1].device_id
        );
        assert!(plan.maximum_first_moment_ns >= 9_000);
        assert!(plan.maximum_second_moment_ns2 >= 90_000);
    }

    #[test]
    fn placement_uses_exact_class_costs_instead_of_resource_ordinals() {
        let mut partition = partition(2, 1);
        let class_a = format!("sha256:{}", "a".repeat(64));
        let class_b = format!("sha256:{}", "b".repeat(64));
        partition.resource_execution_class_ids = vec![class_a.clone(), class_b.clone()];
        let devices = vec![
            VulkanSelectedResourcePlacementDevice {
                device_id: "a".to_string(),
                physical_device_id: "physical-a".to_string(),
                api_version: 1,
                driver_version: 1,
                resident_payload_capacity_bytes: 20,
                measured_costs_by_execution_class: BTreeMap::from([
                    (class_a.clone(), decode_cost(1)),
                    (class_b.clone(), decode_cost(100)),
                ]),
            },
            VulkanSelectedResourcePlacementDevice {
                device_id: "b".to_string(),
                physical_device_id: "physical-b".to_string(),
                api_version: 1,
                driver_version: 1,
                resident_payload_capacity_bytes: 20,
                measured_costs_by_execution_class: BTreeMap::from([
                    (class_a, decode_cost(100)),
                    (class_b, decode_cost(1)),
                ]),
            },
        ];
        let plan = plan_selected_resource_placement(
            "layer",
            &partition,
            &telemetry(vec![1, 1], Vec::new()),
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(plan.assignments[0].device_id, "a");
        assert_eq!(plan.assignments[1].device_id, "b");
    }

    #[test]
    fn eager_placement_rejects_insufficient_aggregate_capacity() {
        let error = plan_selected_resource_placement(
            "layer",
            &partition(4, 2),
            &telemetry(vec![1; 4], vec![0; 6]),
            &devices(4, 15),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();

        assert!(error.0.contains("no measured device"));
    }

    #[test]
    fn demand_paged_placement_separates_addressable_bank_from_resident_load_wave() {
        let plan = plan_selected_resource_placement(
            "layer",
            &partition(6, 2),
            &telemetry(vec![1; 6], vec![0; 15]),
            &devices(6, 20)[..1],
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap();

        assert_eq!(plan.assignments.len(), 6);
        assert_eq!(plan.device_loads.len(), 1);
        assert_eq!(plan.device_loads[0].addressable_bytes, 60);
        assert_eq!(plan.device_loads[0].maximum_load_wave_bytes, 20);
    }

    #[test]
    fn demand_paged_placement_rejects_a_selection_wave_larger_than_resident_capacity() {
        let error = plan_selected_resource_placement(
            "layer",
            &partition(4, 2),
            &telemetry(vec![1; 4], vec![0; 6]),
            &devices(4, 15)[..1],
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();

        assert!(error.0.contains("residency requirement"));
    }

    #[test]
    fn demand_retained_placement_requires_eventual_full_residency() {
        let error = plan_selected_resource_placement(
            "layer",
            &partition(6, 2),
            &telemetry(vec![1; 6], vec![0; 15]),
            &devices(6, 20)[..1],
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandRetained,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();

        assert!(error.0.contains("residency requirement"));
    }

    #[test]
    fn placement_rejects_incomplete_measurements_and_malformed_joint_telemetry() {
        let mut incomplete_devices = devices(4, 40);
        incomplete_devices[0]
            .measured_costs_by_execution_class
            .clear();
        assert!(
            plan_selected_resource_placement(
                "layer",
                &partition(4, 2),
                &telemetry(vec![1; 4], vec![0; 6]),
                &incomplete_devices,
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .is_err()
        );
        let mut wrong_phase = devices(4, 40);
        wrong_phase[0]
            .measured_costs_by_execution_class
            .values_mut()
            .next()
            .unwrap()
            .phase = nerve_execution_contracts::ExecutionPhase::Prefill;
        assert!(
            plan_selected_resource_placement(
                "layer",
                &partition(4, 2),
                &telemetry(vec![1; 4], vec![0; 6]),
                &wrong_phase,
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .is_err()
        );
        assert!(
            plan_selected_resource_placement(
                "layer",
                &partition(4, 2),
                &telemetry(vec![1; 4], vec![0; 5]),
                &devices(4, 40),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .is_err()
        );
        let mut malformed_class = partition(4, 2);
        malformed_class.resource_execution_class_ids[2] = "expert-2".to_string();
        let error = plan_selected_resource_placement(
            "layer",
            &malformed_class,
            &telemetry(vec![1; 4], vec![0; 6]),
            &devices(4, 40),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();
        assert!(error.0.contains("resource execution classes"));
    }
}
