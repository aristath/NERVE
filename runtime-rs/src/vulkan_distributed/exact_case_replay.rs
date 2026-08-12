use crate::vulkan_stream_circuit::{
    VulkanPlacementDeviceExecutionIdentity, VulkanPlacementExecutionCaseIdentity,
    VulkanPlacementExecutionStrategy,
};

impl VulkanDistributedExecutionPlanSet {
    pub fn apply_exact_execution_cases(
        &mut self,
        decode_cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
        decode_batch_cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
        prefill_cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
        device_execution_identity_by_logical_device: &BTreeMap<
            String,
            VulkanPlacementDeviceExecutionIdentity,
        >,
    ) -> Result<(), VulkanDistributedPlanError> {
        replay_exact_execution_cases_to_phase(
            &mut self.decode,
            decode_cases,
            ExecutionPhase::Decode,
            device_execution_identity_by_logical_device,
        )?;
        replay_exact_execution_cases_to_phase(
            &mut self.decode_batch,
            decode_batch_cases,
            ExecutionPhase::Decode,
            device_execution_identity_by_logical_device,
        )?;
        replay_exact_execution_cases_to_phase(
            &mut self.prefill,
            prefill_cases,
            ExecutionPhase::Prefill,
            device_execution_identity_by_logical_device,
        )?;
        VulkanDistributedSelectedResourceStorePlan::from_execution_plan_set(self)?;
        Ok(())
    }
}

fn replay_exact_execution_cases_to_phase(
    execution_plan: &mut VulkanDistributedExecutionPlan,
    cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    phase: ExecutionPhase,
    device_execution_identity_by_logical_device: &BTreeMap<
        String,
        VulkanPlacementDeviceExecutionIdentity,
    >,
) -> Result<(), VulkanDistributedPlanError> {
    if cases.is_empty() {
        return Ok(());
    }
    let dispatch_indices_by_component = execution_plan
        .dispatches
        .iter()
        .enumerate()
        .fold(BTreeMap::<String, Vec<usize>>::new(), |mut by_component, (index, dispatch)| {
            by_component
                .entry(dispatch.component_id.clone())
                .or_default()
                .push(index);
            by_component
        });

    for (component_id, case) in cases {
        if case.behavior.phase != phase {
            return exact_case_error(format!(
                "exact execution case for component {component_id:?} belongs to {:?}, not {phase:?}",
                case.behavior.phase,
            ));
        }
        let dispatch_indices = dispatch_indices_by_component
            .get(component_id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        match case.strategy {
            VulkanPlacementExecutionStrategy::SingleDevice => {
                if !dispatch_indices.is_empty() || !case.shards.is_empty() {
                    return exact_case_error(format!(
                        "single-device exact case for component {component_id:?} conflicts with a distributed runtime plan",
                    ));
                }
            }
            VulkanPlacementExecutionStrategy::Serialized => {
                return exact_case_error(format!(
                    "serialized exact case for component {component_id:?} cannot replay as one distributed component island",
                ));
            }
            _ => replay_exact_distributed_component_case(
                execution_plan,
                component_id,
                dispatch_indices,
                case,
                device_execution_identity_by_logical_device,
            )?,
        }
    }

    if let Some(component_id) = dispatch_indices_by_component
        .keys()
        .find(|component_id| !cases.contains_key(*component_id))
    {
        return exact_case_error(format!(
            "distributed runtime component {component_id:?} has no exact execution case for {phase:?}",
        ));
    }
    execution_plan.execution_islands = resolved_physical_execution_islands_for_phase(
        &execution_plan.dispatches,
        execution_plan.shared_activation_route,
        phase,
    )?;
    Ok(())
}

fn replay_exact_distributed_component_case(
    execution_plan: &mut VulkanDistributedExecutionPlan,
    component_id: &str,
    dispatch_indices: &[usize],
    case: &VulkanPlacementExecutionCaseIdentity,
    device_execution_identity_by_logical_device: &BTreeMap<
        String,
        VulkanPlacementDeviceExecutionIdentity,
    >,
) -> Result<(), VulkanDistributedPlanError> {
    if dispatch_indices.is_empty() || case.devices.len() < 2 || case.shards.is_empty() {
        return exact_case_error(format!(
            "distributed exact case for component {component_id:?} has no matching runtime dispatches or participants",
        ));
    }
    let mut runtime_contracts = BTreeMap::<&str, &str>::new();
    for dispatch_index in dispatch_indices {
        let dispatch = &execution_plan.dispatches[*dispatch_index];
        if runtime_contracts
            .insert(
                dispatch.physical_execution_contract_id.as_str(),
                dispatch.implementation_digest.as_str(),
            )
            .is_some_and(|existing| existing != dispatch.implementation_digest)
        {
            return exact_case_error(format!(
                "compiled component {component_id:?} repeats one physical contract with conflicting implementations",
            ));
        }
    }
    let runtime_contract_ids = runtime_contracts.keys().map(|id| (*id).to_string()).collect::<Vec<_>>();
    let runtime_implementation_digests = runtime_contracts
        .values()
        .map(|digest| (*digest).to_string())
        .collect::<Vec<_>>();
    if runtime_contract_ids != case.behavior.contract_ids
        || runtime_implementation_digests != case.behavior.implementation_digests
    {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with different physical execution contracts",
        ));
    }
    let runtime_strategy = vulkan_distributed_placement_strategy(
        case.devices.len(),
        dispatch_indices.iter().map(|dispatch_index| {
            execution_plan.dispatches[*dispatch_index].execution_strategy
        }),
    )?;
    if runtime_strategy != case.strategy {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with a different physical execution strategy",
        ));
    }
    let runtime_participants = dispatch_indices
        .iter()
        .flat_map(|dispatch_index| {
            execution_plan.dispatches[*dispatch_index]
                .shards
                .iter()
                .map(|shard| shard.device_id.as_str())
        })
        .collect::<BTreeSet<_>>();
    let runtime_device_identities = runtime_participants
        .iter()
        .map(|device_id| {
            device_execution_identity_by_logical_device
                .get(*device_id)
                .cloned()
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "exact case for component {component_id:?} references unbound logical participant {device_id:?}",
                    ))
                })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if runtime_device_identities != case.devices.iter().cloned().collect::<BTreeSet<_>>() {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured on different physical devices or drivers",
        ));
    }
    let owner_identity = device_execution_identity_by_logical_device
        .get(&execution_plan.dispatches[dispatch_indices[0]].owner_device_id)
        .ok_or_else(|| {
            VulkanDistributedPlanError(format!(
                "exact case for component {component_id:?} has an unbound logical owner",
            ))
        })?;
    if owner_identity.physical_device_id != case.owner_physical_device_id {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with a different physical owner",
        ));
    }
    validate_exact_case_transport(
        execution_plan,
        component_id,
        dispatch_indices,
        case,
        device_execution_identity_by_logical_device,
    )?;
    let mut consumed_exact_shards = 0usize;

    for (dispatch_ordinal, dispatch_index) in dispatch_indices.iter().enumerate() {
        let dispatch = &mut execution_plan.dispatches[*dispatch_index];
        let mut exact_shards = case
            .shards
            .iter()
            .filter(|shard| shard.dispatch_ordinal == dispatch_ordinal)
            .collect::<Vec<_>>();
        exact_shards.sort_by_key(|shard| shard.participant_ordinal);
        if exact_shards.len() != dispatch.shards.len()
            || exact_shards
                .iter()
                .map(|shard| shard.participant_ordinal)
                .ne(0..dispatch.shards.len())
        {
            return exact_case_error(format!(
                "exact case for {component_id:?} dispatch {dispatch_ordinal} does not cover every runtime participant ordinal exactly once",
            ));
        }
        let distribution = exact_distribution_name(dispatch.distribution);
        let selected_resource_partitions = dispatch.selected_resource_partitions.clone();
        for (runtime_shard, exact_shard) in dispatch.shards.iter_mut().zip(exact_shards) {
            let runtime_identity = device_execution_identity_by_logical_device
                .get(&runtime_shard.device_id)
                .expect("every runtime participant was resolved above");
            let parameter_bytes = runtime_shard.parameters.iter().try_fold(
                0usize,
                |total, parameter| {
                    total.checked_add(parameter.byte_count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "exact replay parameter byte count overflowed".to_string(),
                        )
                    })
                },
            )?;
            if exact_shard.physical_device_id != runtime_identity.physical_device_id
                || exact_shard.distribution != distribution
                || exact_shard.logical_start != runtime_shard.row_start
                || exact_shard.logical_count != runtime_shard.row_count
                || exact_shard.parameter_bytes != parameter_bytes
            {
                return exact_case_error(format!(
                    "exact case for {component_id:?} dispatch {dispatch_ordinal} does not match its compiled shard geometry",
                ));
            }
            let runtime_fragments = exact_runtime_selected_resource_fragments(
                &selected_resource_partitions,
                runtime_shard,
            )?;
            let mut exact_partition_ordinals = exact_shard
                .selected_resource_indices_by_partition
                .keys()
                .chain(exact_shard.selected_resource_fragments_by_partition.keys())
                .copied()
                .collect::<Vec<_>>();
            exact_partition_ordinals.sort_unstable();
            exact_partition_ordinals.dedup();
            if exact_partition_ordinals
                .into_iter()
                .ne(0..selected_resource_partitions.len())
                || exact_shard.selected_resource_fragments_by_partition != runtime_fragments
            {
                return exact_case_error(format!(
                    "exact case for {component_id:?} dispatch {dispatch_ordinal} does not match every selected-resource partition fragment",
                ));
            }
            runtime_shard.selected_resource_indices = selected_resource_partitions
                .iter()
                .enumerate()
                .filter_map(|(partition_ordinal, partition)| {
                    exact_shard
                        .selected_resource_indices_by_partition
                        .get(&partition_ordinal)
                        .map(|indices| (
                        partition.selector_id.clone(),
                        indices.clone(),
                    ))
                })
                .collect();
            consumed_exact_shards += 1;
        }
        validate_exact_selected_resource_coverage(dispatch, component_id, dispatch_ordinal)?;
    }
    if consumed_exact_shards != case.shards.len()
        || case
            .shards
            .iter()
            .any(|shard| shard.dispatch_ordinal >= dispatch_indices.len())
    {
        return exact_case_error(format!(
            "exact case for component {component_id:?} contains shards outside its runtime dispatch sequence",
        ));
    }
    Ok(())
}

fn validate_exact_selected_resource_coverage(
    dispatch: &VulkanDistributedDispatchPlan,
    component_id: &str,
    dispatch_ordinal: usize,
) -> Result<(), VulkanDistributedPlanError> {
    for partition in &dispatch.selected_resource_partitions {
        if !partition.parameter_partitions.is_empty() {
            if dispatch.shards.iter().any(|shard| {
                !shard.selected_resource_indices.is_empty()
                    || !shard
                        .selected_resource_fragments
                        .contains_key(&partition.selector_id)
            }) {
                return exact_case_error(format!(
                    "exact case for {component_id:?} dispatch {dispatch_ordinal} mixes whole-resource and fragment ownership for selector {:?}",
                    partition.selector_id,
                ));
            }
            for resource_index in 0..partition.resource_count {
                let mut logical_ranges = dispatch
                    .shards
                    .iter()
                    .filter_map(|shard| {
                        shard.selected_resource_fragments[&partition.selector_id]
                            .iter()
                            .find(|fragment| fragment.resource_index == resource_index)
                            .map(|fragment| (fragment.logical_start, fragment.logical_count))
                    })
                    .collect::<Vec<_>>();
                logical_ranges.sort_unstable();
                let mut frontier = 0usize;
                for (start, count) in logical_ranges {
                    if start != frontier {
                        return exact_case_error(format!(
                            "exact case for {component_id:?} dispatch {dispatch_ordinal} leaves a fragmented selector gap for resource {resource_index}",
                        ));
                    }
                    frontier = frontier.checked_add(count).ok_or_else(|| {
                        VulkanDistributedPlanError(
                            "exact selected-resource fragment coverage overflowed".to_string(),
                        )
                    })?;
                }
                if frontier != dispatch.output_rows {
                    return exact_case_error(format!(
                        "exact case for {component_id:?} dispatch {dispatch_ordinal} does not cover the full fragment extent for resource {resource_index}",
                    ));
                }
            }
            continue;
        }
        let mut owners = vec![0usize; partition.resource_count];
        for shard in &dispatch.shards {
            let indices = shard
                .selected_resource_indices
                .get(&partition.selector_id)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "exact case for {component_id:?} dispatch {dispatch_ordinal} omitted selector {:?} on participant {:?}",
                        partition.selector_id, shard.device_id,
                    ))
                })?;
            for resource_index in indices {
                let Some(owner_count) = owners.get_mut(*resource_index) else {
                    return exact_case_error(format!(
                        "exact case for {component_id:?} dispatch {dispatch_ordinal} assigns selector {:?} resource {resource_index} out of range",
                        partition.selector_id,
                    ));
                };
                *owner_count += 1;
            }
        }
        if owners.into_iter().any(|owner_count| owner_count != 1) {
            return exact_case_error(format!(
                "exact case for {component_id:?} dispatch {dispatch_ordinal} must assign every selector {:?} resource exactly once",
                partition.selector_id,
            ));
        }
    }
    Ok(())
}

fn exact_runtime_selected_resource_fragments(
    selected_resource_partitions: &[VulkanDistributedSelectedResourcePartitionPlan],
    shard: &VulkanDistributedDispatchShard,
) -> Result<
    BTreeMap<usize, Vec<crate::vulkan_stream_circuit::VulkanPlacementSelectedResourceFragmentIdentity>>,
    VulkanDistributedPlanError,
> {
    let known_selectors = selected_resource_partitions
        .iter()
        .map(|partition| partition.selector_id.as_str())
        .collect::<BTreeSet<_>>();
    if shard
        .selected_resource_fragments
        .keys()
        .any(|selector| !known_selectors.contains(selector.as_str()))
    {
        return exact_case_error("runtime shard references an unknown selected-resource fragment");
    }
    Ok(selected_resource_partitions
        .iter()
        .enumerate()
        .filter_map(|(ordinal, partition)| {
            shard
                .selected_resource_fragments
                .get(&partition.selector_id)
                .map(|fragments| {
                    (
                        ordinal,
                        fragments
                            .iter()
                            .map(|fragment| {
                                crate::vulkan_stream_circuit::VulkanPlacementSelectedResourceFragmentIdentity {
                                    resource_index: fragment.resource_index,
                                    atomic_group_id: fragment.atomic_group_id.clone(),
                                    logical_start: fragment.logical_start,
                                    logical_count: fragment.logical_count,
                                    parameters: fragment.parameters.iter().map(|parameter| {
                                        crate::vulkan_stream_circuit::VulkanPlacementSelectedResourceParameterFragmentIdentity {
                                            parameter_slot: parameter.parameter_slot,
                                            resource_id: parameter.resource_id.clone(),
                                            resource_byte_count: parameter.resource_byte_count,
                                            byte_offset: parameter.byte_offset,
                                            byte_count: parameter.byte_count,
                                        }
                                    }).collect(),
                                }
                            })
                            .collect(),
                    )
                })
        })
        .collect())
}

fn validate_exact_case_transport(
    execution_plan: &VulkanDistributedExecutionPlan,
    component_id: &str,
    dispatch_indices: &[usize],
    case: &VulkanPlacementExecutionCaseIdentity,
    device_execution_identity_by_logical_device: &BTreeMap<
        String,
        VulkanPlacementDeviceExecutionIdentity,
    >,
) -> Result<(), VulkanDistributedPlanError> {
    let route_name = match execution_plan.shared_activation_route {
        VulkanSharedResidentBufferRoute::ExternalDeviceLocal => "external_device_local",
        VulkanSharedResidentBufferRoute::SharedHost => "shared_host",
    };
    let dispatch_index_set = dispatch_indices
        .iter()
        .map(|index| execution_plan.dispatches[*index].dispatch_index)
        .collect::<BTreeSet<_>>();
    let runtime_transports = execution_plan
        .execution_islands
        .iter()
        .filter(|island| {
            island
                .dispatches
                .iter()
                .any(|dispatch| dispatch_index_set.contains(&dispatch.dispatch_index))
        })
        .flat_map(|island| &island.transport_routes)
        .map(|route| {
            let source = device_execution_identity_by_logical_device
                .get(&route.source_device_id)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "exact case for component {component_id:?} has an unbound transport source",
                    ))
                })?;
            let destination = device_execution_identity_by_logical_device
                .get(&route.destination_device_id)
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "exact case for component {component_id:?} has an unbound transport destination",
                    ))
                })?;
            Ok((
                source.physical_device_id.clone(),
                destination.physical_device_id.clone(),
                route.byte_capacity,
                route_name.to_string(),
            ))
        })
        .collect::<Result<BTreeSet<_>, VulkanDistributedPlanError>>()?;
    let exact_transports = case
        .transports
        .iter()
        .map(|route| {
            (
                route.source_physical_device_id.clone(),
                route.destination_physical_device_id.clone(),
                route.byte_capacity,
                route.route.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    if runtime_transports != exact_transports {
        return exact_case_error(format!(
            "exact case for component {component_id:?} requires different physical transport routes",
        ));
    }
    Ok(())
}

fn exact_distribution_name(distribution: VulkanDistributedDispatchDistribution) -> &'static str {
    match distribution {
        VulkanDistributedDispatchDistribution::OutputRows => "output_rows",
        VulkanDistributedDispatchDistribution::InputColumns => "input_columns",
        VulkanDistributedDispatchDistribution::ExpertRange => "expert_range",
    }
}

fn exact_case_error<T>(message: impl Into<String>) -> Result<T, VulkanDistributedPlanError> {
    Err(VulkanDistributedPlanError(message.into()))
}

#[cfg(test)]
mod exact_case_replay_tests {
    use super::*;
    use crate::vulkan_stream_circuit::{
        VulkanPlacementBehaviorIdentity, VulkanPlacementDeviceExecutionIdentity,
        VulkanPlacementDispatchGeometry, VulkanPlacementEquivalenceIdentity,
        VulkanPlacementOperationGeometry, VulkanPlacementShapeClass, VulkanPlacementShardIdentity,
    };

    fn activation(binding: usize, signal_id: &str) -> VulkanDistributedActivationSlot {
        VulkanDistributedActivationSlot {
            binding,
            component_id: "moe".to_string(),
            signal_id: signal_id.to_string(),
            slot: binding,
            byte_capacity: 16,
            signal_byte_capacity: 16,
            storage: VulkanDistributedActivationStorage::ActivationSlot,
        }
    }

    fn shard(device_id: &str, row_start: usize, resources: &[usize]) -> VulkanDistributedDispatchShard {
        VulkanDistributedDispatchShard {
            device_id: device_id.to_string(),
            selected_resource_indices: BTreeMap::from([(
                "selector".to_string(),
                resources.to_vec(),
            )]),
            selected_resource_fragments: BTreeMap::new(),
            row_start,
            row_count: 2,
            workgroup_count_x: 1,
            base_workgroup_z: u32::try_from(row_start).unwrap(),
            input_range: VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: 16,
            },
            auxiliary_input_ranges: Vec::new(),
            output_byte_offset: 0,
            output_byte_count: 16,
            parameters: Vec::new(),
        }
    }

    fn dispatch() -> VulkanDistributedDispatchPlan {
        VulkanDistributedDispatchPlan {
            owner_device_id: "owner".to_string(),
            dispatch_index: 7,
            component_id: "moe".to_string(),
            node_id: "down".to_string(),
            physical_artifact_id: "artifact".to_string(),
            physical_execution_contract_id: "contract".to_string(),
            implementation_digest: "implementation".to_string(),
            execution_strategy: nerve_execution_contracts::ExecutionStrategy::ExpertParallel,
            equivalence: VulkanDistributedEquivalencePlan {
                output: VulkanDistributedEquivalenceKind::BitExact,
                state: VulkanDistributedEquivalenceKind::BitExact,
                absolute_tolerance_bits: None,
                relative_tolerance_bits: None,
            },
            contract_member_node_ids: vec!["down".to_string()],
            local_intermediates: Vec::new(),
            has_lazy_resource_requirements: true,
            selected_resource_partitions: vec![VulkanDistributedSelectedResourcePartitionPlan {
                execution_scope: "target".to_string(),
                selector_id: "selector".to_string(),
                node_id: "router".to_string(),
                domain_id: "experts".to_string(),
                selection_signal: "routes".to_string(),
                address_table_binding: 3,
                parameter_slots_binding: 4,
                resource_count: 4,
                parameters_per_resource: 2,
                parameter_partitions: Vec::new(),
                selection_count_per_activation: 2,
                atomic_group_ids: (0..4).map(|index| format!("expert-{index}")).collect(),
                atomic_group_byte_counts: vec![8; 4],
                atomic_group_resource_ids: (0..4)
                    .map(|index| vec![format!("resource-{index}-0"), format!("resource-{index}-1")])
                    .collect(),
                parameter_resource_ids: (0..4)
                    .map(|index| vec![format!("resource-{index}-0"), format!("resource-{index}-1")])
                    .collect(),
                parameter_resource_byte_counts: vec![vec![4, 4]; 4],
            }],
            owner_residency_requirements: Vec::new(),
            input_byte_capacity: 16,
            output_byte_capacity: 16,
            output_rows: 4,
            input_width: 4,
            row_alignment: 1,
            input_activation: activation(0, "input"),
            input_distribution: InputDistribution::Routed,
            auxiliary_input_activations: Vec::new(),
            auxiliary_input_distributions: Vec::new(),
            output_activation: activation(1, "output"),
            output_collection: OutputCollection::Routed,
            reduction: None,
            distribution: VulkanDistributedDispatchDistribution::ExpertRange,
            distributed_parameter_byte_count: 0,
            shards: vec![shard("owner", 0, &[0, 1]), shard("helper", 2, &[2, 3])],
        }
    }

    fn exact_case(owner_resources: &[usize], helper_resources: &[usize]) -> VulkanPlacementExecutionCaseIdentity {
        let device = |physical_device_id: &str| VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: physical_device_id.to_string(),
            api_version: 1,
            driver_version: 2,
        };
        let exact_shard = |participant_ordinal, physical_device_id: &str, row_start, resources: &[usize]| VulkanPlacementShardIdentity {
            dispatch_ordinal: 0,
            participant_ordinal,
            physical_device_id: physical_device_id.to_string(),
            distribution: "expert_range".to_string(),
            logical_start: row_start,
            logical_count: 2,
            selected_resource_indices_by_partition: BTreeMap::from([(0, resources.to_vec())]),
            selected_resource_fragments_by_partition: BTreeMap::new(),
            parameter_bytes: 0,
        };
        VulkanPlacementExecutionCaseIdentity {
            behavior: VulkanPlacementBehaviorIdentity {
                compiled_execution_signature: "signature".to_string(),
                contract_ids: vec!["contract".to_string()],
                implementation_digests: vec!["implementation".to_string()],
                artifact_digest: "artifact".to_string(),
                execution_graph_digest: "graph".to_string(),
                runtime_implementation_fingerprint: "runtime".to_string(),
                phase: ExecutionPhase::Decode,
                shape: VulkanPlacementShapeClass {
                    activation_batch_width: 1,
                    input_byte_capacity: 16,
                    output_byte_capacity: 16,
                    operations: vec![VulkanPlacementOperationGeometry::Dispatch {
                        geometry: VulkanPlacementDispatchGeometry {
                            contract_id: "contract".to_string(),
                            logical_extent: 4,
                            sampled_extent: 4,
                            input_width: 4,
                            workgroup_count_x: 1,
                            local_size_x: 64,
                        },
                    }],
                },
                input_fixture_digest: "fixture".to_string(),
                equivalence: VulkanPlacementEquivalenceIdentity::bit_exact(),
            },
            strategy: VulkanPlacementExecutionStrategy::WholeExpertParallel,
            devices: vec![device("physical-owner"), device("physical-helper")],
            shards: vec![
                exact_shard(0, "physical-owner", 0, owner_resources),
                exact_shard(1, "physical-helper", 2, helper_resources),
            ],
            input_physical_device_id: "physical-owner".to_string(),
            output_physical_device_id: "physical-owner".to_string(),
            owner_physical_device_id: "physical-owner".to_string(),
            transports: Vec::new(),
        }
    }

    #[test]
    fn exact_replay_applies_partition_ordinal_expert_ownership() {
        let mut plan = VulkanDistributedExecutionPlan {
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            storage_buffer_offset_alignment: 4,
            dispatches: vec![dispatch()],
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 16,
            shared_output_byte_capacity: 16,
            distributed_parameter_byte_count: 0,
        };

        replay_exact_distributed_component_case(
            &mut plan,
            "moe",
            &[0],
            &exact_case(&[0, 2], &[1, 3]),
            &BTreeMap::from([
                (
                    "owner".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-owner".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
                (
                    "helper".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-helper".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
            ]),
        )
        .unwrap();

        assert_eq!(
            plan.dispatches[0].shards[0].selected_resource_indices["selector"],
            [0, 2]
        );
        assert_eq!(
            plan.dispatches[0].shards[1].selected_resource_indices["selector"],
            [1, 3]
        );
    }

    #[test]
    fn exact_replay_rejects_duplicate_or_missing_expert_ownership() {
        let mut plan = VulkanDistributedExecutionPlan {
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            storage_buffer_offset_alignment: 4,
            dispatches: vec![dispatch()],
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 16,
            shared_output_byte_capacity: 16,
            distributed_parameter_byte_count: 0,
        };

        assert!(
            replay_exact_distributed_component_case(
                &mut plan,
                "moe",
                &[0],
                &exact_case(&[0, 1], &[1, 2]),
                &BTreeMap::from([
                    (
                        "owner".to_string(),
                        VulkanPlacementDeviceExecutionIdentity {
                            physical_device_id: "physical-owner".to_string(),
                            api_version: 1,
                            driver_version: 2,
                        },
                    ),
                    (
                        "helper".to_string(),
                        VulkanPlacementDeviceExecutionIdentity {
                            physical_device_id: "physical-helper".to_string(),
                            api_version: 1,
                            driver_version: 2,
                        },
                    ),
                ]),
            )
            .unwrap_err()
            .to_string()
            .contains("exactly once")
        );
    }

    #[test]
    fn exact_replay_rejects_stale_physical_driver_identity() {
        let mut plan = VulkanDistributedExecutionPlan {
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            storage_buffer_offset_alignment: 4,
            dispatches: vec![dispatch()],
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 16,
            shared_output_byte_capacity: 16,
            distributed_parameter_byte_count: 0,
        };

        let error = replay_exact_distributed_component_case(
            &mut plan,
            "moe",
            &[0],
            &exact_case(&[0, 1], &[2, 3]),
            &BTreeMap::from([
                (
                    "owner".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-owner".to_string(),
                        api_version: 1,
                        driver_version: 999,
                    },
                ),
                (
                    "helper".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-helper".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
            ]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("different physical devices or drivers"));
    }

    #[test]
    fn exact_replay_rejects_stale_physical_execution_contracts() {
        let mut plan = VulkanDistributedExecutionPlan {
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            storage_buffer_offset_alignment: 4,
            dispatches: vec![dispatch()],
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 16,
            shared_output_byte_capacity: 16,
            distributed_parameter_byte_count: 0,
        };
        let mut stale = exact_case(&[0, 1], &[2, 3]);
        stale.behavior.implementation_digests = vec!["stale".to_string()];

        let error = replay_exact_distributed_component_case(
            &mut plan,
            "moe",
            &[0],
            &stale,
            &BTreeMap::from([
                (
                    "owner".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-owner".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
                (
                    "helper".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-helper".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
            ]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("different physical execution contracts"));
    }

    #[test]
    fn exact_replay_rejects_relabelled_physical_execution_strategy() {
        let mut plan = VulkanDistributedExecutionPlan {
            device_ids: vec!["helper".to_string(), "owner".to_string()],
            storage_buffer_offset_alignment: 4,
            dispatches: vec![dispatch()],
            execution_islands: Vec::new(),
            shared_activation_route: VulkanSharedResidentBufferRoute::SharedHost,
            shared_input_byte_capacity: 16,
            shared_output_byte_capacity: 16,
            distributed_parameter_byte_count: 0,
        };
        let mut relabelled = exact_case(&[0, 1], &[2, 3]);
        relabelled.strategy = VulkanPlacementExecutionStrategy::IntraExpertTensorParallel;

        let error = replay_exact_distributed_component_case(
            &mut plan,
            "moe",
            &[0],
            &relabelled,
            &BTreeMap::from([
                (
                    "owner".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-owner".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
                (
                    "helper".to_string(),
                    VulkanPlacementDeviceExecutionIdentity {
                        physical_device_id: "physical-helper".to_string(),
                        api_version: 1,
                        driver_version: 2,
                    },
                ),
            ]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("different physical execution strategy"));
    }

    #[test]
    fn exact_fragment_coverage_requires_every_resource_and_no_gaps() {
        let mut dispatch = dispatch();
        dispatch.execution_strategy = nerve_execution_contracts::ExecutionStrategy::TensorParallelExpert;
        dispatch.selected_resource_partitions[0].parameter_partitions = vec![
            VulkanDistributedSelectedResourceParameterPartitionPlan {
                parameter_slot: 0,
                dimension: 0,
                kind: nerve_execution_contracts::ParameterPartitionKind::Contiguous,
                alignment_elements: 1,
                logical_elements_per_index: 1,
            },
        ];
        dispatch.output_rows = 4;
        for shard in &mut dispatch.shards {
            shard.selected_resource_indices.clear();
            shard.selected_resource_fragments = BTreeMap::from([(
                "selector".to_string(),
                (0..4)
                    .map(|resource_index| VulkanDistributedSelectedResourceFragmentPlan {
                        resource_index,
                        atomic_group_id: format!("expert-{resource_index}"),
                        logical_start: shard.row_start,
                        logical_count: shard.row_count,
                        parameters: vec![
                            VulkanDistributedSelectedResourceParameterFragmentPlan {
                                parameter_slot: 0,
                                resource_id: format!("resource-{resource_index}-0"),
                                resource_byte_count: 4,
                                byte_offset: shard.row_start,
                                byte_count: shard.row_count,
                            },
                        ],
                    })
                    .collect(),
            )]);
        }
        validate_exact_selected_resource_coverage(&dispatch, "moe", 0).unwrap();

        dispatch.shards[1]
            .selected_resource_fragments
            .get_mut("selector")
            .unwrap()[0]
            .logical_start = 3;
        assert!(
            validate_exact_selected_resource_coverage(&dispatch, "moe", 0)
                .unwrap_err()
                .to_string()
                .contains("gap")
        );
    }
}
