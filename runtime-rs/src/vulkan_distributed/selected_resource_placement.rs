#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceExecutionClassPlan {
    pub component_id: String,
    pub selector_id: String,
    pub resource_execution_class_ids: Vec<String>,
}

impl VulkanDistributedExecutionPlan {
    /// Combines every selected-resource operation in dispatch order into the
    /// complete locally executable expert transaction used by placement and
    /// calibration. Runtime instance names do not enter the class digest.
    pub fn selected_resource_execution_classes(
        &self,
        selector_id: &str,
    ) -> Result<VulkanSelectedResourceExecutionClassPlan, VulkanDistributedPlanError> {
        let operations = self
            .dispatches
            .iter()
            .flat_map(|dispatch| {
                dispatch
                    .selected_resource_partitions
                    .iter()
                    .filter(move |partition| partition.selector_id == selector_id)
                    .map(move |partition| (dispatch, partition))
            })
            .collect::<Vec<_>>();
        let Some((first_dispatch, first_partition)) = operations.first().copied() else {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource execution class has no operations for selector {selector_id:?}",
            )));
        };
        if operations.iter().any(|(dispatch, partition)| {
            dispatch.component_id != first_dispatch.component_id
                || partition.execution_scope != first_partition.execution_scope
                || partition.node_id != first_partition.node_id
                || partition.domain_id != first_partition.domain_id
                || partition.selection_signal != first_partition.selection_signal
                || partition.resource_count != first_partition.resource_count
                || partition.selection_count_per_activation
                    != first_partition.selection_count_per_activation
                || partition.atomic_group_ids != first_partition.atomic_group_ids
                || partition.atomic_group_byte_counts
                    != first_partition.atomic_group_byte_counts
                || partition.atomic_group_resource_ids
                    != first_partition.atomic_group_resource_ids
                || partition.resource_operation_class_ids.len() != partition.resource_count
                || partition
                    .resource_operation_class_ids
                    .iter()
                    .any(|class_id| !valid_selected_resource_execution_class_id(class_id))
        }) {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource operations disagree on transaction identity for selector {selector_id:?}",
            )));
        }
        let resource_execution_class_ids = (0..first_partition.resource_count)
            .map(|resource_index| {
                let operation_classes = operations
                    .iter()
                    .map(|(_, partition)| {
                        partition.resource_operation_class_ids[resource_index].as_str()
                    })
                    .collect::<Vec<_>>();
                let payload = serde_json::to_vec(&serde_json::json!({
                    "schema": "nerve.selected_resource_execution_class.v1",
                    "operation_classes": operation_classes,
                }))
                .map_err(|error| {
                    VulkanDistributedPlanError(format!(
                        "could not encode selected-resource transaction class: {error}",
                    ))
                })?;
                Ok(format!("sha256:{:x}", Sha256::digest(payload)))
            })
            .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;
        Ok(VulkanSelectedResourceExecutionClassPlan {
            component_id: first_dispatch.component_id.clone(),
            selector_id: selector_id.to_string(),
            resource_execution_class_ids,
        })
    }

    /// Projects the complete compiler-declared operation chain for exactly one
    /// selector resource onto one physical participant. The selector domain
    /// and width remain unchanged: only the resource address/parameter tables
    /// are narrowed, so valid non-local routes can still be present while
    /// exactly one route performs arithmetic locally.
    pub fn isolated_selected_resource_transaction(
        &self,
        selector_id: &str,
        resource_index: usize,
        device_id: &str,
        phase: nerve_execution_contracts::ExecutionPhase,
    ) -> Result<Self, VulkanDistributedPlanError> {
        if selector_id.trim().is_empty() || device_id.trim().is_empty() {
            return Err(VulkanDistributedPlanError(
                "isolated selected-resource execution requires a selector and target device"
                    .to_string(),
            ));
        }
        let classes = self.selected_resource_execution_classes(selector_id)?;
        if resource_index >= classes.resource_execution_class_ids.len() {
            return Err(VulkanDistributedPlanError(format!(
                "isolated selected-resource index {resource_index} exceeds selector {selector_id:?} domain {}",
                classes.resource_execution_class_ids.len(),
            )));
        }
        let mut dispatches = Vec::new();
        for dispatch in &self.dispatches {
            if !dispatch
                .selected_resource_partitions
                .iter()
                .any(|partition| partition.selector_id == selector_id)
            {
                continue;
            }
            if dispatch.selected_resource_partitions.len() != 1
                || dispatch.selected_resource_partitions[0].selector_id != selector_id
            {
                return Err(VulkanDistributedPlanError(format!(
                    "isolated selected-resource dispatch {}.{} couples selector {selector_id:?} to another selector",
                    dispatch.component_id, dispatch.node_id,
                )));
            }
            let partition = &dispatch.selected_resource_partitions[0];
            if resource_index >= partition.resource_count
                || partition.resource_operation_class_ids.len() != partition.resource_count
            {
                return Err(VulkanDistributedPlanError(format!(
                    "isolated selected-resource dispatch {}.{} has an incomplete selector domain",
                    dispatch.component_id, dispatch.node_id,
                )));
            }
            let mut shard = merged_distributed_dispatch_shard(dispatch)?;
            shard.device_id = device_id.to_string();
            if partition.parameter_partitions.is_empty() {
                shard.selected_resource_indices = BTreeMap::from([(
                    selector_id.to_string(),
                    vec![resource_index],
                )]);
                shard.selected_resource_fragments.clear();
            } else {
                shard.selected_resource_indices.clear();
                let fragments = shard
                    .selected_resource_fragments
                    .get_mut(selector_id)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(format!(
                            "isolated selected-resource dispatch {}.{} has no merged fragments for selector {selector_id:?}",
                            dispatch.component_id, dispatch.node_id,
                        ))
                    })?;
                fragments.retain(|fragment| fragment.resource_index == resource_index);
                if fragments.len() != 1 {
                    return Err(VulkanDistributedPlanError(format!(
                        "isolated selected-resource dispatch {}.{} does not reconstruct exactly one complete resource fragment",
                        dispatch.component_id, dispatch.node_id,
                    )));
                }
                shard
                    .selected_resource_fragments
                    .retain(|candidate, _| candidate == selector_id);
            }
            let mut isolated = dispatch.clone();
            isolated.owner_device_id = device_id.to_string();
            isolated.distributed_parameter_byte_count = shard
                .parameters
                .iter()
                .try_fold(0usize, |total, fragment| {
                    total.checked_add(fragment.byte_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "isolated selected-resource parameter bytes overflowed".to_string(),
                        )
                    })
                })?;
            isolated.shards = vec![shard];
            dispatches.push(isolated);
        }
        if dispatches.is_empty() {
            return Err(VulkanDistributedPlanError(format!(
                "isolated selected-resource selector {selector_id:?} has no executable dispatches",
            )));
        }
        let distributed_parameter_byte_count = dispatches.iter().try_fold(
            0usize,
            |total, dispatch| {
                total
                    .checked_add(dispatch.distributed_parameter_byte_count)
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "isolated selected-resource parameter total overflowed".to_string(),
                        )
                    })
            },
        )?;
        let execution_islands = resolved_physical_execution_islands_for_phase(
            &dispatches,
            self.shared_activation_route,
            phase,
        )?;
        Ok(Self {
            device_ids: vec![device_id.to_string()],
            storage_buffer_offset_alignment: self.storage_buffer_offset_alignment,
            shared_input_byte_capacity: dispatches
                .first()
                .expect("isolated dispatches were checked nonempty")
                .input_byte_capacity,
            shared_output_byte_capacity: dispatches
                .last()
                .expect("isolated dispatches were checked nonempty")
                .output_byte_capacity,
            dispatches,
            execution_islands,
            shared_activation_route: self.shared_activation_route,
            distributed_parameter_byte_count,
        })
    }
}

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

impl VulkanSelectedResourcePlacementPlan {
    pub(crate) fn execution_ownership_by_device(
        &self,
        resource_count: usize,
    ) -> Result<BTreeMap<String, BTreeSet<usize>>, VulkanDistributedPlanError> {
        if self.selector_id.trim().is_empty() || self.assignments.len() != resource_count {
            return Err(VulkanDistributedPlanError(
                "selected-resource placement does not cover its complete resource domain"
                    .to_string(),
            ));
        }
        let mut ownership = BTreeMap::<String, BTreeSet<usize>>::new();
        let mut covered = vec![false; resource_count];
        for assignment in &self.assignments {
            if assignment.device_id.trim().is_empty()
                || assignment.resource_index >= resource_count
                || std::mem::replace(&mut covered[assignment.resource_index], true)
            {
                return Err(VulkanDistributedPlanError(
                    "selected-resource placement repeats a resource or has an invalid owner"
                        .to_string(),
                ));
            }
            ownership
                .entry(assignment.device_id.clone())
                .or_default()
                .insert(assignment.resource_index);
        }
        if covered.iter().any(|covered| !covered) {
            return Err(VulkanDistributedPlanError(
                "selected-resource placement leaves an unowned resource".to_string(),
            ));
        }
        Ok(ownership)
    }
}

pub(crate) fn selected_resource_placements_from_execution_plan(
    execution_plan: &VulkanDistributedExecutionPlan,
) -> Result<Vec<VulkanSelectedResourcePlacementPlan>, VulkanDistributedPlanError> {
    let mut assignments_by_selector =
        BTreeMap::<String, (usize, Vec<VulkanSelectedResourceAssignment>)>::new();
    for dispatch in &execution_plan.dispatches {
        for partition in &dispatch.selected_resource_partitions {
            if !partition.parameter_partitions.is_empty() {
                continue;
            }
            let mut assignments = dispatch
                .shards
                .iter()
                .flat_map(|shard| {
                    shard
                        .selected_resource_indices
                        .get(&partition.selector_id)
                        .into_iter()
                        .flatten()
                        .map(|resource_index| VulkanSelectedResourceAssignment {
                            resource_index: *resource_index,
                            device_id: shard.device_id.clone(),
                        })
                })
                .collect::<Vec<_>>();
            assignments.sort_by_key(|assignment| assignment.resource_index);
            let candidate = VulkanSelectedResourcePlacementPlan {
                selector_id: partition.selector_id.clone(),
                assignments: assignments.clone(),
                device_loads: Vec::new(),
                maximum_first_moment_ns: 0,
                maximum_second_moment_ns2: 0,
            };
            candidate.execution_ownership_by_device(partition.resource_count)?;
            match assignments_by_selector.entry(partition.selector_id.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((partition.resource_count, assignments));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() != &(partition.resource_count, assignments) =>
                {
                    return Err(VulkanDistributedPlanError(format!(
                        "selected-resource selector {:?} changes ownership between connected dispatches",
                        partition.selector_id,
                    )));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(assignments_by_selector
        .into_iter()
        .map(|(selector_id, (_, assignments))| VulkanSelectedResourcePlacementPlan {
            selector_id,
            assignments,
            device_loads: Vec::new(),
            maximum_first_moment_ns: 0,
            maximum_second_moment_ns2: 0,
        })
        .collect())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourcePlacementMove {
    pub resource_index: usize,
    pub source_device_id: String,
    pub destination_device_id: String,
    pub payload_bytes: usize,
    pub destination_load_duration_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanSelectedResourceReconfigurationPlan {
    pub selector_id: String,
    pub observed_activation_count: u64,
    pub current_duration_ns_per_activation: u128,
    pub proposed_duration_ns_per_activation: u128,
    pub migration_critical_path_ns: u128,
    pub break_even_activation_count: u128,
    pub moves: Vec<VulkanSelectedResourcePlacementMove>,
    pub proposed: VulkanSelectedResourcePlacementPlan,
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
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<VulkanSelectedResourcePlacementPlan, VulkanDistributedPlanError> {
    try_plan_selected_resource_placement(
        component_id,
        partition,
        execution_classes,
        telemetry,
        devices,
        residency_policy,
        phase,
    )?
    .ok_or_else(|| {
        VulkanDistributedPlanError(format!(
            "selected-resource placement for selector {:?} has no measured device with sufficient resident payload capacity",
            partition.selector_id,
        ))
    })
}

/// Returns `None` when valid exact inputs have no feasible capacity assignment.
/// Invalid identities, telemetry, or measurements remain errors.
pub fn try_plan_selected_resource_placement(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
) -> Result<Option<VulkanSelectedResourcePlacementPlan>, VulkanDistributedPlanError> {
    validate_selected_resource_placement_problem(
        component_id,
        partition,
        execution_classes,
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
            let execution_class_id =
                &execution_classes.resource_execution_class_ids[resource_index];
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
                        &execution_classes.resource_execution_class_ids[*other_resource];
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
            return Ok(None);
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
    Ok(Some(VulkanSelectedResourcePlacementPlan {
        selector_id: partition.selector_id.clone(),
        assignments,
        device_loads,
        maximum_first_moment_ns,
        maximum_second_moment_ns2,
    }))
}

/// Plans a quiescent-boundary ownership change from exact warm selection
/// telemetry. This function never mutates a mounted package. It accepts a
/// candidate only when the same measured execution classes predict a strict
/// reduction in per-activation serialized device work, and exposes the exact
/// cold migration break-even point to the runtime scheduler.
#[allow(clippy::too_many_arguments)]
pub fn try_plan_warm_selected_resource_reconfiguration(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
    current: &VulkanSelectedResourcePlacementPlan,
) -> Result<Option<VulkanSelectedResourceReconfigurationPlan>, VulkanDistributedPlanError> {
    validate_selected_resource_placement_problem(
        component_id,
        partition,
        execution_classes,
        telemetry,
        devices,
        phase,
    )?;
    if current.selector_id != partition.selector_id {
        return Err(VulkanDistributedPlanError(format!(
            "current selected-resource placement belongs to selector {:?}, expected {:?}",
            current.selector_id, partition.selector_id,
        )));
    }
    let observed_selection_count = telemetry
        .selection_counts
        .iter()
        .try_fold(0u64, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            VulkanDistributedPlanError(
                "selected-resource observed selection count overflowed".to_string(),
            )
        })?;
    let selections_per_activation = u64::try_from(partition.selection_count_per_activation)
        .map_err(|_| {
            VulkanDistributedPlanError(
                "selected-resource selection width exceeds u64".to_string(),
            )
        })?;
    if selections_per_activation == 0
        || observed_selection_count == 0
        || observed_selection_count % selections_per_activation != 0
    {
        return Err(VulkanDistributedPlanError(format!(
            "selected-resource telemetry for selector {:?} does not contain a complete activation history",
            partition.selector_id,
        )));
    }
    let observed_activation_count = observed_selection_count / selections_per_activation;
    let current = score_selected_resource_assignments(
        component_id,
        partition,
        execution_classes,
        telemetry,
        devices,
        residency_policy,
        phase,
        &current.assignments,
    )?;
    let Some(proposed) = try_plan_selected_resource_placement(
        component_id,
        partition,
        execution_classes,
        telemetry,
        devices,
        residency_policy,
        phase,
    )?
    else {
        return Ok(None);
    };
    if current.assignments == proposed.assignments {
        return Ok(None);
    }
    let activations = u128::from(observed_activation_count);
    let current_duration_ns_per_activation = current.maximum_first_moment_ns.div_ceil(activations);
    let proposed_duration_ns_per_activation =
        proposed.maximum_first_moment_ns.div_ceil(activations);
    let Some(improvement_ns_per_activation) = current_duration_ns_per_activation
        .checked_sub(proposed_duration_ns_per_activation)
        .filter(|improvement| *improvement > 0)
    else {
        return Ok(None);
    };
    let current_owners = current
        .assignments
        .iter()
        .map(|assignment| (assignment.resource_index, assignment.device_id.as_str()))
        .collect::<BTreeMap<_, _>>();
    let devices_by_id = devices
        .iter()
        .map(|device| (device.device_id.as_str(), device))
        .collect::<BTreeMap<_, _>>();
    let mut moves = Vec::new();
    let mut migration_ns_by_destination = BTreeMap::<String, u128>::new();
    for assignment in &proposed.assignments {
        let source_device_id = current_owners
            .get(&assignment.resource_index)
            .copied()
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "current selected-resource placement does not cover every resource"
                        .to_string(),
                )
            })?;
        if source_device_id == assignment.device_id {
            continue;
        }
        let destination = devices_by_id
            .get(assignment.device_id.as_str())
            .copied()
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "proposed selected-resource placement references an unknown device"
                        .to_string(),
                )
            })?;
        let class_id = &execution_classes.resource_execution_class_ids[assignment.resource_index];
        let load_duration = destination.measured_costs_by_execution_class[class_id]
            .lazy_load_wave_duration_ns;
        let destination_total = migration_ns_by_destination
            .entry(assignment.device_id.clone())
            .or_default();
        *destination_total = destination_total
            .checked_add(u128::from(load_duration))
            .ok_or_else(|| {
                VulkanDistributedPlanError(
                    "selected-resource migration duration overflowed".to_string(),
                )
            })?;
        moves.push(VulkanSelectedResourcePlacementMove {
            resource_index: assignment.resource_index,
            source_device_id: source_device_id.to_string(),
            destination_device_id: assignment.device_id.clone(),
            payload_bytes: partition.atomic_group_byte_counts[assignment.resource_index],
            destination_load_duration_ns: load_duration,
        });
    }
    if moves.is_empty() {
        return Ok(None);
    }
    moves.sort_by_key(|movement| movement.resource_index);
    let migration_critical_path_ns = migration_ns_by_destination
        .values()
        .copied()
        .max()
        .unwrap_or(0);
    let break_even_activation_count =
        migration_critical_path_ns.div_ceil(improvement_ns_per_activation);
    Ok(Some(VulkanSelectedResourceReconfigurationPlan {
        selector_id: partition.selector_id.clone(),
        observed_activation_count,
        current_duration_ns_per_activation,
        proposed_duration_ns_per_activation,
        migration_critical_path_ns,
        break_even_activation_count,
        moves,
        proposed,
    }))
}

#[allow(clippy::too_many_arguments)]
fn score_selected_resource_assignments(
    component_id: &str,
    partition: &VulkanDistributedSelectedResourcePartitionPlan,
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
    telemetry: &crate::vulkan_stream_circuit::VulkanSelectionTelemetryDomainSnapshot,
    devices: &[VulkanSelectedResourcePlacementDevice],
    residency_policy: crate::vulkan_stream_circuit::ResourceResidencyPolicy,
    phase: nerve_execution_contracts::ExecutionPhase,
    assignments: &[VulkanSelectedResourceAssignment],
) -> Result<VulkanSelectedResourcePlacementPlan, VulkanDistributedPlanError> {
    validate_selected_resource_placement_problem(
        component_id,
        partition,
        execution_classes,
        telemetry,
        devices,
        phase,
    )?;
    if assignments.len() != partition.resource_count {
        return Err(VulkanDistributedPlanError(
            "selected-resource assignment does not cover the complete resource domain"
                .to_string(),
        ));
    }
    let devices_by_id = devices
        .iter()
        .map(|device| (device.device_id.as_str(), device))
        .collect::<BTreeMap<_, _>>();
    let mut owner_by_resource = vec![None; partition.resource_count];
    for assignment in assignments {
        if assignment.resource_index >= partition.resource_count
            || !devices_by_id.contains_key(assignment.device_id.as_str())
            || owner_by_resource[assignment.resource_index]
                .replace(assignment.device_id.as_str())
                .is_some()
        {
            return Err(VulkanDistributedPlanError(
                "selected-resource assignment repeats a resource or references an unknown device"
                    .to_string(),
            ));
        }
    }
    if owner_by_resource.iter().any(Option::is_none) {
        return Err(VulkanDistributedPlanError(
            "selected-resource assignment leaves an unowned resource".to_string(),
        ));
    }
    let mut loads = devices
        .iter()
        .map(|device| (device.device_id.as_str(), Vec::<usize>::new()))
        .collect::<BTreeMap<_, _>>();
    for (resource_index, owner) in owner_by_resource.iter().enumerate() {
        loads
            .get_mut(owner.expect("resource ownership was validated"))
            .expect("assignment owners were validated")
            .push(resource_index);
    }
    let mut device_loads = Vec::with_capacity(devices.len());
    for device in devices {
        let owned_resource_indices = loads.remove(device.device_id.as_str()).unwrap_or_default();
        let addressable_bytes = owned_resource_indices.iter().try_fold(0usize, |total, index| {
            total
                .checked_add(partition.atomic_group_byte_counts[*index])
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource addressable byte count overflowed".to_string(),
                    )
                })
        })?;
        let maximum_load_wave_bytes = selected_resource_maximum_load_wave_bytes(
            partition,
            owned_resource_indices.iter().copied(),
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
        if required_resident_bytes > device.resident_payload_capacity_bytes {
            return Err(VulkanDistributedPlanError(format!(
                "selected-resource assignment exceeds resident capacity on {:?}",
                device.device_id,
            )));
        }
        let first_moment_ns = owned_resource_indices.iter().try_fold(0u128, |total, index| {
            let class_id = &execution_classes.resource_execution_class_ids[*index];
            u128::from(telemetry.selection_counts[*index])
                .checked_mul(u128::from(
                    device.measured_costs_by_execution_class[class_id].execution_duration_ns,
                ))
                .and_then(|contribution| total.checked_add(contribution))
                .ok_or_else(|| {
                    VulkanDistributedPlanError(
                        "selected-resource first moment overflowed".to_string(),
                    )
                })
        })?;
        let mut second_moment_ns2 = owned_resource_indices.iter().try_fold(
            0u128,
            |total, index| {
                let class_id = &execution_classes.resource_execution_class_ids[*index];
                let duration = u128::from(
                    device.measured_costs_by_execution_class[class_id].execution_duration_ns,
                );
                u128::from(telemetry.selection_counts[*index])
                    .checked_mul(duration)
                    .and_then(|value| value.checked_mul(duration))
                    .and_then(|contribution| total.checked_add(contribution))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "selected-resource second moment overflowed".to_string(),
                        )
                    })
            },
        )?;
        for (offset, left) in owned_resource_indices.iter().enumerate() {
            let left_class = &execution_classes.resource_execution_class_ids[*left];
            let left_duration = u128::from(
                device.measured_costs_by_execution_class[left_class].execution_duration_ns,
            );
            for right in owned_resource_indices.iter().skip(offset + 1) {
                let right_class = &execution_classes.resource_execution_class_ids[*right];
                let right_duration = u128::from(
                    device.measured_costs_by_execution_class[right_class].execution_duration_ns,
                );
                let joint = u128::from(
                    telemetry
                        .co_selection_count(*left, *right)
                        .unwrap_or(0),
                );
                second_moment_ns2 = joint
                    .checked_mul(left_duration)
                    .and_then(|value| value.checked_mul(right_duration))
                    .and_then(|value| value.checked_mul(2))
                    .and_then(|contribution| second_moment_ns2.checked_add(contribution))
                    .ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "selected-resource joint second moment overflowed".to_string(),
                        )
                    })?;
            }
        }
        device_loads.push(VulkanSelectedResourceDeviceLoad {
            device_id: device.device_id.clone(),
            addressable_bytes,
            maximum_load_wave_bytes,
            first_moment_ns,
            second_moment_ns2,
            owned_resource_indices,
        });
    }
    let maximum_first_moment_ns = device_loads
        .iter()
        .map(|load| load.first_moment_ns)
        .max()
        .unwrap_or(0);
    let maximum_second_moment_ns2 = device_loads
        .iter()
        .map(|load| load.second_moment_ns2)
        .max()
        .unwrap_or(0);
    let mut assignments = assignments.to_vec();
    assignments.sort_by_key(|assignment| assignment.resource_index);
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

pub(crate) fn selected_resource_placements_fit_phase_participants(
    plans: &VulkanDistributedExecutionPlanSet,
    placements: &[VulkanSelectedResourcePlacementPlan],
) -> Result<bool, VulkanDistributedPlanError> {
    let placement_by_selector = canonical_selected_resource_placements(placements)?;
    for phase in plans.all() {
        for dispatch in &phase.dispatches {
            let matching_partitions = dispatch
                .selected_resource_partitions
                .iter()
                .filter_map(|partition| {
                    placement_by_selector
                        .get(partition.selector_id.as_str())
                        .map(|placement| (partition, *placement))
                })
                .collect::<Vec<_>>();
            if matching_partitions.is_empty() {
                continue;
            }
            let compiled_devices = dispatch
                .shards
                .iter()
                .map(|shard| shard.device_id.as_str())
                .collect::<BTreeSet<_>>();
            let assigned_devices = matching_partitions
                .iter()
                .flat_map(|(partition, placement)| {
                    placement.assignments.iter().map(move |assignment| {
                        (partition.resource_count, assignment)
                    })
                })
                .map(|(resource_count, assignment)| {
                    if assignment.resource_index >= resource_count {
                        return Err(VulkanDistributedPlanError(format!(
                            "selected-resource placement assigns resource {} outside 0..{resource_count}",
                            assignment.resource_index,
                        )));
                    }
                    Ok(assignment.device_id.as_str())
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            if assigned_devices.len() < 2
                || !assigned_devices.is_subset(&compiled_devices)
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
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
    let phase_selectors = execution_plan
        .dispatches
        .iter()
        .flat_map(|dispatch| {
            dispatch
                .selected_resource_partitions
                .iter()
                .map(|partition| partition.selector_id.clone())
        })
        .collect::<BTreeSet<_>>();
    if phase_selectors.is_empty() {
        return Ok(());
    }
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
        .filter(|selector_id| {
            phase_selectors.contains(*selector_id) && !applied_selectors.contains(selector_id)
        })
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
    execution_classes: &VulkanSelectedResourceExecutionClassPlan,
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
    if execution_classes.component_id != component_id
        || execution_classes.selector_id != partition.selector_id
        || execution_classes.resource_execution_class_ids.len() != partition.resource_count
        || execution_classes
            .resource_execution_class_ids
            .iter()
            .any(|class_id| !valid_selected_resource_execution_class_id(class_id))
    {
        return Err(VulkanDistributedPlanError(format!(
            "execution classes do not exactly match selector {:?}",
            partition.selector_id,
        )));
    }
    if partition.atomic_group_byte_counts.len() != partition.resource_count
        || partition.resource_operation_class_ids.len() != partition.resource_count
        || partition
            .resource_operation_class_ids
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
    let required_classes = execution_classes
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
            resource_operation_class_ids: vec![
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

    fn execution_classes(
        partition: &VulkanDistributedSelectedResourcePartitionPlan,
    ) -> VulkanSelectedResourceExecutionClassPlan {
        VulkanSelectedResourceExecutionClassPlan {
            component_id: "layer".to_string(),
            selector_id: partition.selector_id.clone(),
            resource_execution_class_ids: partition.resource_operation_class_ids.clone(),
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
        let partition = partition(4, 2);
        let plan = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(4, 2);
        let plan = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(2, 1);
        let class_a = format!("sha256:{}", "a".repeat(64));
        let class_b = format!("sha256:{}", "b".repeat(64));
        let classes = VulkanSelectedResourceExecutionClassPlan {
            component_id: "layer".to_string(),
            selector_id: partition.selector_id.clone(),
            resource_execution_class_ids: vec![class_a.clone(), class_b.clone()],
        };
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
            &classes,
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
        let partition = partition(4, 2);
        assert!(
            try_plan_selected_resource_placement(
                "layer",
                &partition,
                &execution_classes(&partition),
                &telemetry(vec![1; 4], vec![0; 6]),
                &devices(4, 15),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .unwrap()
            .is_none()
        );
        let error = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(6, 2);
        let plan = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(4, 2);
        let error = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(6, 2);
        let error = plan_selected_resource_placement(
            "layer",
            &partition,
            &execution_classes(&partition),
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
        let partition = partition(4, 2);
        let classes = execution_classes(&partition);
        let mut incomplete_devices = devices(4, 40);
        incomplete_devices[0]
            .measured_costs_by_execution_class
            .clear();
        assert!(
            plan_selected_resource_placement(
                "layer",
                &partition,
                &classes,
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
                &partition,
                &classes,
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
                &partition,
                &classes,
                &telemetry(vec![1; 4], vec![0; 5]),
                &devices(4, 40),
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
                nerve_execution_contracts::ExecutionPhase::Decode,
            )
            .is_err()
        );
        let mut malformed_classes = classes;
        malformed_classes.resource_execution_class_ids[2] = "expert-2".to_string();
        let error = plan_selected_resource_placement(
            "layer",
            &partition,
            &malformed_classes,
            &telemetry(vec![1; 4], vec![0; 6]),
            &devices(4, 40),
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap_err();
        assert!(error.0.contains("execution classes"));
    }

    #[test]
    fn warm_reconfiguration_reports_exact_moves_and_cold_break_even() {
        let partition = partition(4, 2);
        let classes = execution_classes(&partition);
        let telemetry = telemetry(vec![100, 100, 1, 1], vec![100, 0, 0, 0, 0, 1]);
        let devices = devices(4, 40);
        let current = score_selected_resource_assignments(
            "layer",
            &partition,
            &classes,
            &telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &[
                VulkanSelectedResourceAssignment {
                    resource_index: 0,
                    device_id: "a".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 1,
                    device_id: "a".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 2,
                    device_id: "b".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 3,
                    device_id: "b".to_string(),
                },
            ],
        )
        .unwrap();

        let plan = try_plan_warm_selected_resource_reconfiguration(
            "layer",
            &partition,
            &classes,
            &telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &current,
        )
        .unwrap()
        .expect("co-selected hot experts should move to different devices");

        assert_eq!(plan.observed_activation_count, 101);
        assert!(
            plan.proposed_duration_ns_per_activation
                < plan.current_duration_ns_per_activation
        );
        assert!(!plan.moves.is_empty());
        assert!(plan.moves.iter().all(|movement| {
            movement.source_device_id != movement.destination_device_id
                && movement.payload_bytes == 10
                && movement.destination_load_duration_ns == 20
        }));
        assert!(plan.migration_critical_path_ns > 0);
        assert!(plan.break_even_activation_count > 0);
        assert_ne!(plan.proposed.assignments, current.assignments);
    }

    #[test]
    fn warm_reconfiguration_rejects_noise_and_incomplete_activation_history() {
        let partition = partition(4, 2);
        let classes = execution_classes(&partition);
        let devices = devices(4, 40);
        let balanced_telemetry = telemetry(vec![100; 4], vec![100, 0, 0, 0, 0, 100]);
        let current = plan_selected_resource_placement(
            "layer",
            &partition,
            &classes,
            &balanced_telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
        )
        .unwrap();
        assert!(
            try_plan_warm_selected_resource_reconfiguration(
                "layer",
                &partition,
                &classes,
                &balanced_telemetry,
                &devices,
                crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
                nerve_execution_contracts::ExecutionPhase::Decode,
                &current,
            )
            .unwrap()
            .is_none()
        );

        let incomplete = telemetry(vec![2, 2, 2, 1], vec![0; 6]);
        let error = try_plan_warm_selected_resource_reconfiguration(
            "layer",
            &partition,
            &classes,
            &incomplete,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &current,
        )
        .unwrap_err();
        assert!(error.0.contains("complete activation history"));
    }

    #[test]
    fn warm_reconfiguration_rejects_invalid_current_ownership_and_capacity() {
        let partition = partition(4, 2);
        let classes = execution_classes(&partition);
        let telemetry = telemetry(vec![2; 4], vec![0; 6]);
        let devices = devices(4, 20);
        let invalid = VulkanSelectedResourcePlacementPlan {
            selector_id: partition.selector_id.clone(),
            assignments: vec![
                VulkanSelectedResourceAssignment {
                    resource_index: 0,
                    device_id: "a".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 1,
                    device_id: "a".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 2,
                    device_id: "a".to_string(),
                },
                VulkanSelectedResourceAssignment {
                    resource_index: 3,
                    device_id: "b".to_string(),
                },
            ],
            device_loads: Vec::new(),
            maximum_first_moment_ns: 0,
            maximum_second_moment_ns2: 0,
        };

        let error = try_plan_warm_selected_resource_reconfiguration(
            "layer",
            &partition,
            &classes,
            &telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::Eager,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &invalid,
        )
        .unwrap_err();
        assert!(error.0.contains("exceeds resident capacity"));

        let mut wrong_selector = invalid;
        wrong_selector.selector_id = "another-selector".to_string();
        let error = try_plan_warm_selected_resource_reconfiguration(
            "layer",
            &partition,
            &classes,
            &telemetry,
            &devices,
            crate::vulkan_stream_circuit::ResourceResidencyPolicy::DemandPaged,
            nerve_execution_contracts::ExecutionPhase::Decode,
            &wrong_selector,
        )
        .unwrap_err();
        assert!(error.0.contains("belongs to selector"));
    }
}
