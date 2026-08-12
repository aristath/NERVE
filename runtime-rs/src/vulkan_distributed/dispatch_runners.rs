pub struct VulkanDistributedDispatchRunners {
    pub dispatches: Vec<VulkanDistributedDispatchRunner>,
    pub dispatch_count: usize,
    pub shard_count: usize,
    transaction_predicates: BTreeMap<String, Arc<VulkanResidentBuffer>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VulkanDistributedResolvedResidencyMiss {
    pub shard_index: usize,
    pub gate_index: usize,
    pub selector_id: String,
    pub checkpoint_tag: u32,
    pub resource_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VulkanDistributedResolvedResidencyFault {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub misses: Vec<VulkanDistributedResolvedResidencyMiss>,
}

fn selected_resource_activation<'a>(
    dispatch: &'a VulkanDistributedDispatchPlan,
    selection_signal: &str,
) -> Result<&'a VulkanDistributedActivationSlot, VulkanDistributedDispatchRunnerError> {
    let matching = std::iter::once(&dispatch.input_activation)
        .chain(dispatch.auxiliary_input_activations.iter())
        .filter(|activation| {
            activation.component_id == dispatch.component_id
                && activation.signal_id == selection_signal
        })
        .collect::<Vec<_>>();
    let [activation] = matching.as_slice() else {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed selected-resource dispatch {}.{} resolves {} activation signals named {selection_signal:?}",
            dispatch.component_id,
            dispatch.node_id,
            matching.len()
        )));
    };
    Ok(*activation)
}

fn distributed_sequence_for_kind<'a, T>(
    direct: &'a T,
    feedback_indirect: Option<&'a T>,
    sequence_kind: VulkanDistributedDispatchSequenceKind,
    device_id: &str,
) -> Result<&'a T, VulkanDistributedDispatchRunnerError> {
    match sequence_kind {
        VulkanDistributedDispatchSequenceKind::Direct => Ok(direct),
        VulkanDistributedDispatchSequenceKind::FeedbackIndirect => {
            feedback_indirect.ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed feedback shard on {device_id:?} has no indirect sequence"
                ))
            })
        }
    }
}

fn physical_island_reduction_dispatch(
    island: &VulkanPhysicalExecutionIslandPlan,
) -> Result<Option<&VulkanDistributedDispatchPlan>, VulkanDistributedDispatchRunnerError> {
    let tail = island.tail();
    let reduced_dispatches = island
        .dispatches
        .iter()
        .filter(|dispatch| dispatch.reduction.is_some())
        .collect::<Vec<_>>();
    match reduced_dispatches.as_slice() {
        [] => Ok(None),
        [planned_dispatch] if planned_dispatch.dispatch_index == tail.dispatch_index => {
            Ok(Some(*planned_dispatch))
        }
        _ => Err(VulkanDistributedDispatchRunnerError(format!(
            "physical execution island {}..{} has {} reductions across {} dispatches; only one tail reduction is legal",
            island.leader().dispatch_index,
            tail.dispatch_index,
            reduced_dispatches.len(),
            island.dispatches.len()
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VulkanDistributedCoordinatorKind {
    None,
    NumericReduction,
    ResidencyCommit,
}

fn physical_island_coordinator_kind(
    island: &VulkanPhysicalExecutionIslandPlan,
    has_shard_residency_predicates: bool,
) -> Result<VulkanDistributedCoordinatorKind, VulkanDistributedDispatchRunnerError> {
    if physical_island_reduction_dispatch(island)?.is_some() {
        Ok(VulkanDistributedCoordinatorKind::NumericReduction)
    } else if has_shard_residency_predicates {
        Ok(VulkanDistributedCoordinatorKind::ResidencyCommit)
    } else {
        Ok(VulkanDistributedCoordinatorKind::None)
    }
}

pub(crate) fn distributed_shard_push_constants(
    planned_dispatch: &VulkanDistributedDispatchPlan,
    planned_shard: &VulkanDistributedDispatchShard,
) -> Result<Vec<u8>, VulkanDistributedDispatchRunnerError> {
    match planned_dispatch.distribution {
        VulkanDistributedDispatchDistribution::OutputRows => Ok(Vec::new()),
        VulkanDistributedDispatchDistribution::InputColumns
        | VulkanDistributedDispatchDistribution::ExpertRange => {
            let (partition_start, partition_count) = if planned_shard
                .selected_resource_indices
                .is_empty()
            {
                (planned_shard.base_workgroup_z, planned_shard.row_count)
            } else {
                let resource_counts = planned_dispatch
                    .selected_resource_partitions
                    .iter()
                    .map(|partition| partition.resource_count)
                    .collect::<BTreeSet<_>>();
                let [resource_count] = resource_counts.iter().copied().collect::<Vec<_>>()[..]
                else {
                    return Err(VulkanDistributedDispatchRunnerError(format!(
                        "distributed selected-resource dispatch {}.{} has incompatible selector extents",
                        planned_dispatch.component_id, planned_dispatch.node_id,
                    )));
                };
                if planned_dispatch.selected_resource_partitions.len()
                    != planned_shard.selected_resource_indices.len()
                    || planned_dispatch.selected_resource_partitions.iter().any(|partition| {
                        !planned_shard
                            .selected_resource_indices
                            .contains_key(&partition.selector_id)
                    })
                {
                    return Err(VulkanDistributedDispatchRunnerError(format!(
                        "distributed selected-resource dispatch {}.{} has incomplete shard ownership",
                        planned_dispatch.component_id, planned_dispatch.node_id,
                    )));
                }
                (0, resource_count)
            };
            let mut bytes = partition_start.to_le_bytes().to_vec();
            let partition_count = u32::try_from(partition_count).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed repeated partition count exceeds u32".to_string(),
                )
            })?;
            bytes.extend_from_slice(&partition_count.to_le_bytes());
            Ok(bytes)
        }
    }
}

fn create_distributed_resident_dispatch(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    planned_shard: &VulkanDistributedDispatchShard,
    shard_index: usize,
    parameter_buffers: &VulkanDistributedParameterBuffers,
    dynamic_resource_buffers: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
    activation_buffers: &VulkanDistributedActivationBuffers,
    artifact: &VulkanLoadedPhysicalKernelArtifact,
    private_input: Option<&Arc<VulkanResidentBuffer>>,
    private_output: Option<&Arc<VulkanResidentBuffer>>,
) -> Result<VulkanResidentKernelDispatch, VulkanDistributedDispatchRunnerError> {
    let (input, input_byte_offset) = if let Some(input) = private_input {
        (input, 0)
    } else {
        let input = activation_buffers
            .activation_buffer(
                &planned_dispatch.owner_device_id,
                &planned_dispatch.input_activation,
                &planned_shard.device_id,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no input activation on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    planned_shard.device_id
                ))
            })?;
        (input, planned_shard.input_range.byte_offset)
    };
    let (output, output_byte_offset, output_byte_count) =
        if let Some(reduction) = &planned_dispatch.reduction {
            if private_output.is_some() {
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "distributed reduction {}.{} cannot publish a private intermediate",
                    planned_dispatch.component_id, planned_dispatch.node_id
                )));
            }
            let output = activation_buffers
                .reduction_partial_buffer(
                    &planned_dispatch.owner_device_id,
                    planned_dispatch.dispatch_index,
                    &planned_shard.device_id,
                )
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} has no partial plane on {:?}",
                        planned_dispatch.component_id,
                        planned_dispatch.node_id,
                        planned_shard.device_id
                    ))
                })?;
            if planned_shard.output_byte_offset != 0
                || planned_shard.output_byte_count != reduction.partial_byte_capacity
            {
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} shard {} has an invalid partial-output range",
                    planned_dispatch.component_id, planned_dispatch.node_id, shard_index
                )));
            }
            let output_byte_offset = shard_index
                .checked_mul(reduction.partial_byte_capacity)
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed partial plane offset overflowed".to_string(),
                    )
                })?;
            (output, output_byte_offset, reduction.partial_byte_capacity)
        } else if let Some(output) = private_output {
            (output, 0, planned_shard.output_byte_count)
        } else {
            let output = activation_buffers
                .activation_buffer(
                    &planned_dispatch.owner_device_id,
                    &planned_dispatch.output_activation,
                    &planned_shard.device_id,
                )
                .ok_or_else(|| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "distributed dispatch {}.{} has no output activation on {:?}",
                        planned_dispatch.component_id,
                        planned_dispatch.node_id,
                        planned_shard.device_id
                    ))
                })?;
            (
                output,
                planned_shard.output_byte_offset,
                planned_shard.output_byte_count,
            )
        };
    let mut bindings = Vec::with_capacity(
        2 + planned_dispatch.auxiliary_input_activations.len()
            + planned_shard.parameters.len()
            + 2 * planned_dispatch.selected_resource_partitions.len(),
    );
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.input_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed primary input binding exceeds u32".to_string(),
                )
            })?,
            input,
            planned_shard.input_range.byte_count,
        )
        .with_byte_offset(input_byte_offset)
        .with_access(VulkanResidentKernelBufferAccess::Read),
    );
    if planned_shard.auxiliary_input_ranges.len()
        != planned_dispatch.auxiliary_input_activations.len()
    {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed dispatch {}.{} has {} auxiliary ranges for {} auxiliary inputs on {:?}",
            planned_dispatch.component_id,
            planned_dispatch.node_id,
            planned_shard.auxiliary_input_ranges.len(),
            planned_dispatch.auxiliary_input_activations.len(),
            planned_shard.device_id,
        )));
    }
    for (auxiliary, range) in planned_dispatch
        .auxiliary_input_activations
        .iter()
        .zip(&planned_shard.auxiliary_input_ranges)
    {
        let buffer = activation_buffers
            .activation_buffer(
                &planned_dispatch.owner_device_id,
                auxiliary,
                &planned_shard.device_id,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no auxiliary input {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    auxiliary.signal_id,
                    planned_shard.device_id
                ))
            })?;
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(auxiliary.binding).map_err(|_| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed auxiliary input binding exceeds u32".to_string(),
                    )
                })?,
                buffer,
                range.byte_count,
            )
            .with_byte_offset(range.byte_offset)
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    bindings.push(
        VulkanResidentKernelBufferBinding::new(
            u32::try_from(planned_dispatch.output_activation.binding).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed output binding exceeds u32".to_string(),
                )
            })?,
            output,
            output_byte_count,
        )
        .with_byte_offset(output_byte_offset)
        .with_access(VulkanResidentKernelBufferAccess::Write),
    );
    for fragment in &planned_shard.parameters {
        let allocation = parameter_buffers
            .parameter_buffer(
                &planned_shard.device_id,
                &fragment.tensor,
                fragment.byte_offset,
                fragment.byte_count,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no tensor {:?} range at byte {} with length {} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    fragment.tensor,
                    fragment.byte_offset,
                    fragment.byte_count,
                    planned_shard.device_id
                ))
            })?;
        let binding = u32::try_from(fragment.binding).map_err(|_| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed descriptor binding {} exceeds u32",
                fragment.binding
            ))
        })?;
        bindings.push(
            allocation
                .kernel_binding_for_fragment(
                    binding,
                    fragment.byte_offset,
                    fragment.byte_count,
                )
                .map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to bind distributed tensor fragment: {error}"
                    ))
                })?
                .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    for partition in &planned_dispatch.selected_resource_partitions {
        let resources = dynamic_resource_buffers
            .get(&planned_shard.device_id)
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no dynamic resource buffers on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    planned_shard.device_id
                ))
            })?;
        let parameter_slots = resources
            .parameter_slots(
                &planned_dispatch.component_id,
                &planned_dispatch.node_id,
                &partition.selection_signal,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no parameter slots for selector {:?} on {:?}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    partition.selector_id,
                    planned_shard.device_id
                ))
            })?;
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(partition.address_table_binding).map_err(|_| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed dynamic address-table binding exceeds u32".to_string(),
                    )
                })?,
                resources.address_table(),
                resources.address_table().byte_capacity(),
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
        bindings.push(
            VulkanResidentKernelBufferBinding::new(
                u32::try_from(partition.parameter_slots_binding).map_err(|_| {
                    VulkanDistributedDispatchRunnerError(
                        "distributed dynamic parameter-slot binding exceeds u32".to_string(),
                    )
                })?,
                parameter_slots,
                parameter_slots.byte_capacity(),
            )
            .with_access(VulkanResidentKernelBufferAccess::Read),
        );
    }
    let push_constant_bytes = distributed_shard_push_constants(planned_dispatch, planned_shard)?;
    device
        .create_resident_kernel_dispatch_2d_with_base_z(
            &artifact.words,
            &bindings,
            planned_shard.workgroup_count_x,
            1,
            0,
            artifact.artifact.local_size_x,
            u32::try_from(push_constant_bytes.len()).expect("push constant size is at most 8"),
            Some(format!(
                "component={} node={} distributed=device:{} rows={}..{} base_z={} distribution={:?}",
                planned_dispatch.component_id,
                planned_dispatch.node_id,
                planned_shard.device_id,
                planned_shard.row_start,
                planned_shard.row_start + planned_shard.row_count,
                planned_shard.base_workgroup_z,
                planned_dispatch.distribution,
            )),
        )
        .map_err(|error| {
            VulkanDistributedDispatchRunnerError(format!(
                "failed to create distributed dispatch {}.{} shard on {:?}: {error}",
                planned_dispatch.component_id, planned_dispatch.node_id, planned_shard.device_id
            ))
        })
}

impl VulkanDistributedDispatchRunners {
    pub fn create<'a, F, E>(
        execution_plan: &VulkanDistributedExecutionPlan,
        parameter_buffers: &VulkanDistributedParameterBuffers,
        dynamic_resource_buffers: &BTreeMap<String, Arc<VulkanDynamicResourceBuffers>>,
        resource_stores: &BTreeMap<String, Arc<VulkanCompiledResourceDeviceStore>>,
        transaction_predicates: Option<&BTreeMap<String, Arc<VulkanResidentBuffer>>>,
        execution_scope: &str,
        activation_buffers: &VulkanDistributedActivationBuffers,
        loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
        mut device_for: F,
    ) -> Result<Self, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let mut dispatches = Vec::with_capacity(execution_plan.execution_islands.len());
        let mut shard_count = 0usize;
        for planned_island in &execution_plan.execution_islands {
            let leader = planned_island.leader();
            let tail = planned_island.tail();
            let owner_device = device_for(&planned_island.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed owner device {:?}: {error}",
                    planned_island.owner_device_id
                ))
            })?;
            let mut shards = Vec::with_capacity(leader.shards.len());
            let mut owner_shard_residency_predicates = Vec::new();
            for shard_index in 0..leader.shards.len() {
                let leader_shard = &leader.shards[shard_index];
                let device = device_for(&leader_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed shard device {:?}: {error}",
                        leader_shard.device_id
                    ))
                })?;
                let shard_requires_demand_gate = planned_island.dispatches.iter().any(|dispatch| {
                    !dispatch.selected_resource_partitions.is_empty()
                        && resource_stores
                            .get(&leader_shard.device_id)
                            .is_some_and(|store| store.residency_policy().is_demand_loaded())
                });
                let shard_residency_predicate = if shard_requires_demand_gate {
                    let (owner_view, shard_view) = if owner_device
                        .shares_logical_device_with(device)
                    {
                        let predicate = Arc::new(
                            owner_device
                                .create_conditional_resident_buffer(size_of::<u32>())
                                .map_err(VulkanDistributedDispatchRunnerError::from)?,
                        );
                        (Arc::clone(&predicate), predicate)
                    } else {
                        let mut buffers = owner_device
                            .create_shared_conditional_resident_buffers(
                                &[device],
                                size_of::<u32>(),
                            )
                            .map_err(VulkanDistributedDispatchRunnerError::from)?
                            .buffers
                            .into_iter();
                        let owner_view = buffers.next().ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(
                                "distributed shard predicate has no owner view".to_string(),
                            )
                        })?;
                        let shard_view = buffers.next().ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(
                                "distributed shard predicate has no participant view".to_string(),
                            )
                        })?;
                        if buffers.next().is_some() {
                            return Err(VulkanDistributedDispatchRunnerError(
                                "distributed shard predicate produced unexpected views"
                                    .to_string(),
                            ));
                        }
                        (owner_view, shard_view)
                    };
                    owner_view
                        .write_bytes(&1u32.to_le_bytes())
                        .map_err(VulkanDistributedDispatchRunnerError::from)?;
                    owner_shard_residency_predicates.push(owner_view);
                    Some(shard_view)
                } else {
                    None
                };
                let mut resident_dispatches = Vec::with_capacity(planned_island.dispatches.len());
                let mut planned_shards = Vec::with_capacity(planned_island.dispatches.len());
                let mut selected_resource_gates =
                    Vec::with_capacity(planned_island.dispatches.len());
                for (dispatch_offset, planned_dispatch) in
                    planned_island.dispatches.iter().enumerate()
                {
                    let planned_shard =
                        planned_dispatch.shards.get(shard_index).ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "physical execution island {}..{} has no shard {shard_index} for {}.{}",
                                leader.dispatch_index,
                                tail.dispatch_index,
                                planned_dispatch.component_id,
                                planned_dispatch.node_id
                            ))
                        })?;
                    if planned_shard.device_id != leader_shard.device_id {
                        return Err(VulkanDistributedDispatchRunnerError(format!(
                            "physical execution island {}..{} changes shard {shard_index} device from {:?} to {:?}",
                            leader.dispatch_index,
                            tail.dispatch_index,
                            leader_shard.device_id,
                            planned_shard.device_id
                        )));
                    }
                    let artifact = loaded_manifest
                        .physical_artifact(&planned_dispatch.physical_artifact_id)
                        .ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(format!(
                                "distributed dispatch {}.{} is missing physical artifact {:?}",
                                planned_dispatch.component_id,
                                planned_dispatch.node_id,
                                planned_dispatch.physical_artifact_id
                            ))
                        })?;
                    let private_input = if dispatch_offset > 0
                        && local_shard_handoff(
                            &planned_island.dispatches[dispatch_offset - 1],
                            planned_dispatch,
                        )
                    {
                        Some(
                            activation_buffers
                                .private_intermediate_buffer(
                                    planned_island.dispatches[dispatch_offset - 1]
                                        .dispatch_index,
                                    planned_dispatch.dispatch_index,
                                    &planned_shard.device_id,
                                )
                                .ok_or_else(|| {
                                    VulkanDistributedDispatchRunnerError(format!(
                                        "physical execution island {}..{} has no private input for {}.{} on {:?}",
                                        leader.dispatch_index,
                                        tail.dispatch_index,
                                        planned_dispatch.component_id,
                                        planned_dispatch.node_id,
                                        planned_shard.device_id,
                                    ))
                                })?,
                        )
                    } else {
                        None
                    };
                    let private_output = if dispatch_offset + 1
                        < planned_island.dispatches.len()
                        && local_shard_handoff(
                            planned_dispatch,
                            &planned_island.dispatches[dispatch_offset + 1],
                        )
                    {
                        Some(
                            activation_buffers
                                .private_intermediate_buffer(
                                    planned_dispatch.dispatch_index,
                                    planned_island.dispatches[dispatch_offset + 1]
                                        .dispatch_index,
                                    &planned_shard.device_id,
                                )
                                .ok_or_else(|| {
                                    VulkanDistributedDispatchRunnerError(format!(
                                        "physical execution island {}..{} has no private output for {}.{} on {:?}",
                                        leader.dispatch_index,
                                        tail.dispatch_index,
                                        planned_dispatch.component_id,
                                        planned_dispatch.node_id,
                                        planned_shard.device_id,
                                    ))
                                })?,
                        )
                    } else {
                        None
                    };
                    resident_dispatches.push(create_distributed_resident_dispatch(
                        device,
                        planned_dispatch,
                        planned_shard,
                        shard_index,
                        parameter_buffers,
                        dynamic_resource_buffers,
                        activation_buffers,
                        artifact,
                        private_input,
                        private_output,
                    )?);
                    let requires_demand_gates = resource_stores
                        .get(&planned_shard.device_id)
                        .is_some_and(|store| store.residency_policy().is_demand_loaded())
                        && !planned_dispatch.selected_resource_partitions.is_empty();
                    // The island planner proves that every selected-resource
                    // member uses the same selector-owned atomic groups and
                    // exact shard ownership. One leading checkpoint therefore
                    // validates every dynamic address needed by the complete
                    // expert sequence; repeating the same gate before gate/up
                    // and down would add warm-path dispatches without adding a
                    // residency guarantee.
                    let gates = if requires_demand_gates && dispatch_offset == 0 {
                        let store = resource_stores
                            .get(&planned_shard.device_id)
                            .cloned()
                            .expect("demand-gated store was checked");
                        let dynamic_resources = dynamic_resource_buffers
                            .get(&planned_shard.device_id)
                            .ok_or_else(|| {
                                VulkanDistributedDispatchRunnerError(format!(
                                    "distributed selected-resource dispatch {}.{} has no dynamic buffers on {:?}",
                                    planned_dispatch.component_id,
                                    planned_dispatch.node_id,
                                    planned_shard.device_id
                                ))
                            })?;
                        let transaction_predicate = transaction_predicates
                            .and_then(|predicates| predicates.get(&planned_shard.device_id))
                            .cloned()
                            .ok_or_else(|| {
                                VulkanDistributedDispatchRunnerError(format!(
                                    "distributed selected-resource dispatch {}.{} has no transaction predicate on {:?}",
                                    planned_dispatch.component_id,
                                    planned_dispatch.node_id,
                                    planned_shard.device_id
                                ))
                            })?;
                        let local_predicate = shard_residency_predicate
                            .as_ref()
                            .cloned()
                            .ok_or_else(|| {
                                VulkanDistributedDispatchRunnerError(
                                    "distributed demand gate has no shard predicate".to_string(),
                                )
                            })?;
                        planned_dispatch
                            .selected_resource_partitions
                            .iter()
                            .enumerate()
                            .map(|(partition_index, partition)| {
                                let selection_activation = selected_resource_activation(
                                    planned_dispatch,
                                    &partition.selection_signal,
                                )?;
                                let selection_buffer = activation_buffers
                                    .activation_buffer(
                                        &planned_dispatch.owner_device_id,
                                        selection_activation,
                                        &planned_shard.device_id,
                                    )
                                    .cloned()
                                    .ok_or_else(|| {
                                        VulkanDistributedDispatchRunnerError(format!(
                                            "distributed selected-resource dispatch {}.{} has no selection buffer {:?} on {:?}",
                                            planned_dispatch.component_id,
                                            planned_dispatch.node_id,
                                            partition.selection_signal,
                                            planned_shard.device_id
                                        ))
                                    })?;
                                VulkanDistributedSelectedResourceGate::new(
                                    device,
                                    execution_scope,
                                    planned_dispatch,
                                    partition,
                                    selection_buffer,
                                    selection_activation.signal_byte_capacity,
                                    1,
                                    dynamic_resources,
                                    Arc::clone(&store),
                                    Arc::clone(&local_predicate),
                                    Arc::clone(&transaction_predicate),
                                    u32::try_from(partition_index + 1).map_err(|_| {
                                        VulkanDistributedDispatchRunnerError(
                                            "distributed selected-resource checkpoint tag exceeds u32"
                                                .to_string(),
                                        )
                                    })?,
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };
                    selected_resource_gates.push(gates);
                    planned_shards.push(planned_shard.clone());
                }
                let sequence = device.create_resident_kernel_sequence().map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to create distributed sequence {}..{} shard on {:?}: {error}",
                        leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                    ))
                })?;
                let push_constants = planned_shards
                    .iter()
                    .zip(&planned_island.dispatches)
                    .map(|(shard, dispatch)| distributed_shard_push_constants(dispatch, shard))
                    .collect::<Result<Vec<_>, _>>()?;
                let base_steps = resident_dispatches
                    .iter()
                    .zip(&push_constants)
                    .map(|(dispatch, push_constants)| {
                        VulkanResidentKernelSequenceStep::new(dispatch, push_constants)
                    })
                    .collect::<Vec<_>>();
                let mut steps = Vec::new();
                let residency_guard = selected_resource_gates.iter().flatten().next();
                for ((base_step, gates), planned_dispatch) in base_steps
                    .into_iter()
                    .zip(&selected_resource_gates)
                    .zip(&planned_island.dispatches)
                {
                    for gate in gates {
                        steps.push(gate.gate_step()?);
                    }
                    let step = match residency_guard {
                        Some(gate) => gate.guard_step(
                            base_step,
                            u32::try_from(planned_dispatch.dispatch_index).unwrap_or(u32::MAX),
                        )?,
                        None => base_step,
                    };
                    steps.push(step);
                }
                device
                    .record_resident_kernel_sequence(&sequence, &steps)
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to record distributed sequence {}..{} shard on {:?}: {error}",
                            leader.dispatch_index, tail.dispatch_index, leader_shard.device_id
                        ))
                    })?;
                shards.push(VulkanDistributedDispatchShardRunner {
                    device_id: leader_shard.device_id.clone(),
                    planned: planned_shards,
                    resident_dispatches,
                    selected_resource_gates,
                    sequence,
                    feedback_sequence: None,
                });
                shard_count = shard_count
                    .checked_add(planned_island.dispatches.len())
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(
                            "distributed dispatch shard count overflowed".to_string(),
                        )
                    })?;
            }
            let mut helper_synchronization = Vec::with_capacity(
                leader
                    .shards
                    .iter()
                    .filter(|shard| shard.device_id != planned_island.owner_device_id)
                    .count(),
            );
            for planned_shard in &leader.shards {
                if planned_shard.device_id == planned_island.owner_device_id {
                    continue;
                }
                let helper_device = device_for(&planned_shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve distributed helper device {:?}: {error}",
                        planned_shard.device_id
                    ))
                })?;
                helper_synchronization.push(
                    VulkanDistributedQueueSynchronization::new(
                        owner_device,
                        helper_device,
                        &planned_island.owner_device_id,
                        &planned_shard.device_id,
                        &format!(
                            "distributed dispatch {}.{}",
                            leader.component_id, leader.node_id
                        ),
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?,
                );
            }
            let coordinator_kind = physical_island_coordinator_kind(
                planned_island,
                !owner_shard_residency_predicates.is_empty(),
            )?;
            let reduction = match coordinator_kind {
                VulkanDistributedCoordinatorKind::NumericReduction => {
                    let planned_dispatch = physical_island_reduction_dispatch(planned_island)?
                        .expect("numeric coordinator kind has a reduction dispatch");
                    Some(create_distributed_reduction_runner(
                        owner_device,
                        planned_dispatch,
                        activation_buffers,
                        transaction_predicates.and_then(|predicates| {
                            predicates.get(&planned_island.owner_device_id)
                        }),
                        &owner_shard_residency_predicates,
                    )?)
                }
                VulkanDistributedCoordinatorKind::None
                | VulkanDistributedCoordinatorKind::ResidencyCommit => None,
            };
            let residency_commit = match coordinator_kind {
                VulkanDistributedCoordinatorKind::ResidencyCommit => {
                    let commit = create_distributed_residency_commit_runner(
                        owner_device,
                        &leader.component_id,
                        &tail.node_id,
                        transaction_predicates.and_then(|predicates| {
                            predicates.get(&planned_island.owner_device_id)
                        }),
                        &owner_shard_residency_predicates,
                    )?;
                    Some(commit)
                }
                VulkanDistributedCoordinatorKind::None
                | VulkanDistributedCoordinatorKind::NumericReduction => None,
            };
            dispatches.push(VulkanDistributedDispatchRunner {
                planned: planned_island.clone(),
                shards,
                helper_synchronization,
                reduction,
                residency_commit,
                dependency_clock: VulkanDistributedDependencyClock::new(),
            });
        }

        Ok(Self {
            dispatch_count: execution_plan.dispatches.len(),
            dispatches,
            shard_count,
            transaction_predicates: transaction_predicates.cloned().unwrap_or_default(),
        })
    }

    pub fn dispatch(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanDistributedDispatchRunner> {
        self.dispatches.iter().find(|dispatch| {
            dispatch.planned.owner_device_id == owner_device_id
                && dispatch.planned.leader().dispatch_index == dispatch_index
        })
    }

    pub fn execution_island(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<&VulkanPhysicalExecutionIslandPlan> {
        self.dispatches
            .iter()
            .find(|runner| {
                runner.planned.owner_device_id == owner_device_id
                    && runner.planned.contains_dispatch(dispatch_index)
            })
            .map(|runner| &runner.planned)
    }

    pub fn leader_dispatch_index(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Option<usize> {
        self.execution_island(owner_device_id, dispatch_index)
            .map(|group| group.leader().dispatch_index)
    }

    pub fn requires_residency_checkpoint(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> bool {
        self.dispatch(owner_device_id, dispatch_index).is_some_and(|dispatch| {
            dispatch.shards.iter().any(|shard| {
                shard
                    .selected_resource_gates
                    .iter()
                    .any(|gates| !gates.is_empty())
            })
        })
    }

    pub(crate) fn feedback_dispatch_count(
        &self,
    ) -> Result<usize, VulkanDistributedDispatchRunnerError> {
        self.dispatches
            .iter()
            .flat_map(|dispatch| &dispatch.shards)
            .try_fold(0usize, |total, shard| {
                let gate_count = shard
                    .selected_resource_gates
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                total
                    .checked_add(shard.resident_dispatches.len())
                    .and_then(|count| count.checked_add(gate_count))
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(
                            "distributed feedback dispatch count overflowed".to_string(),
                        )
                    })
            })
    }

    pub fn reserve_dependency_value(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
    ) -> Result<u64, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        dispatch
            .dependency_clock
            .reserve(owner_device_id, dispatch_index)
    }

    pub fn advance_replayed_dependency_values(
        &self,
        count: usize,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        let count = u64::try_from(count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed replay dependency count exceeds u64".to_string(),
            )
        })?;
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.validate_advance(
                count,
                &dispatch.planned.owner_device_id,
                dispatch.planned.leader().dispatch_index,
            )?;
        }
        for dispatch in &self.dispatches {
            dispatch.dependency_clock.advance(count);
        }
        Ok(())
    }

    pub fn capture_replay_timeline_state(
        &self,
        state: &mut VulkanTimelineSemaphoreReplayState,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        for dispatch in &self.dispatches {
            let next_value = dispatch.dependency_clock.next_value.get();
            for synchronization in &dispatch.helper_synchronization {
                state
                    .capture(&synchronization.ready_source, next_value)
                    .and_then(|_| state.capture(&synchronization.ready_wait, next_value))
                    .and_then(|_| state.capture(&synchronization.done_source, next_value))
                    .and_then(|_| state.capture(&synchronization.done_wait, next_value))
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
            }
        }
        Ok(())
    }

    pub(crate) fn configure_feedback_indirect_dispatches<'a, F, E>(
        &mut self,
        control: &mut VulkanResidentFeedbackControlPlane,
        mut device_for: F,
    ) -> Result<(), VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        for dispatch in &mut self.dispatches {
            for shard in &mut dispatch.shards {
                let device = device_for(&shard.device_id).map_err(|error| {
                    VulkanDistributedDispatchRunnerError(format!(
                        "failed to resolve feedback shard device {:?}: {error}",
                        shard.device_id
                    ))
                })?;
                let feedback_dispatches = shard
                    .resident_dispatches
                    .iter()
                    .zip(&shard.selected_resource_gates)
                    .flat_map(|(dispatch, gates)| {
                        gates
                            .iter()
                            .map(VulkanDistributedSelectedResourceGate::dispatch)
                            .chain(std::iter::once(dispatch))
                    })
                    .collect::<Vec<_>>();
                let indirect = control
                    .register_sequence(&shard.device_id, feedback_dispatches)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                let push_constants = shard
                    .planned
                    .iter()
                    .zip(&dispatch.planned.dispatches)
                    .map(|(planned, dispatch)| {
                        distributed_shard_push_constants(dispatch, planned)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut steps = Vec::new();
                let mut byte_offsets = indirect.byte_offsets.iter().copied();
                let residency_guard = shard.selected_resource_gates.iter().flatten().next();
                for (((resident_dispatch, push_constants), gates), planned_dispatch) in shard
                    .resident_dispatches
                    .iter()
                    .zip(&push_constants)
                    .zip(&shard.selected_resource_gates)
                    .zip(&dispatch.planned.dispatches)
                {
                    for gate in gates {
                        let byte_offset = byte_offsets.next().ok_or_else(|| {
                            VulkanDistributedDispatchRunnerError(
                                "distributed feedback gate has no indirect command".to_string(),
                            )
                        })?;
                        steps.push(gate.indirect_gate_step(&indirect.buffer, byte_offset)?);
                    }
                    let byte_offset = byte_offsets.next().ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(
                            "distributed feedback dispatch has no indirect command".to_string(),
                        )
                    })?;
                    let base_step = VulkanResidentKernelSequenceStep::new_indirect(
                        resident_dispatch,
                        push_constants,
                        &indirect.buffer,
                        byte_offset,
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                    let step = match residency_guard {
                        Some(gate) => gate.guard_step(
                            base_step,
                            u32::try_from(planned_dispatch.dispatch_index).unwrap_or(u32::MAX),
                        )?,
                        None => base_step,
                    };
                    steps.push(step);
                }
                if byte_offsets.next().is_some() {
                    return Err(VulkanDistributedDispatchRunnerError(
                        "distributed feedback indirect commands exceed recorded steps".to_string(),
                    ));
                }
                let sequence = device
                    .create_resident_kernel_sequence()
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                device
                    .record_resident_kernel_sequence(&sequence, &steps)
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
                shard.feedback_sequence = Some(sequence);
            }
        }
        Ok(())
    }

    pub fn owner_ready_signal_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| sync.owner_ready(dependency_value))
            .collect())
    }

    pub fn owner_completion_wait_points(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        dependency_value: u64,
    ) -> Result<Vec<VulkanTimelineSemaphorePoint<'_>>, VulkanDistributedDispatchRunnerError> {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        Ok(dispatch
            .helper_synchronization
            .iter()
            .map(|sync| sync.owner_done(dependency_value))
            .collect())
    }

    pub fn submit_dispatch_with_device_dependencies<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        submission: VulkanDistributedDispatchSubmission,
        submission_batch: Option<&VulkanResidentQueueSubmissionBatch<'a>>,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let VulkanDistributedDispatchSubmission {
            dependency_value,
            consume_owner_ready_signal,
            prepare_owner_continuation,
            signal_completion,
            sequence_kind,
        } = submission;
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let resolved_shards = resolved_shards
            .into_iter()
            .map(|(shard, device)| {
                distributed_sequence_for_kind(
                    &shard.sequence,
                    shard.feedback_sequence.as_ref(),
                    sequence_kind,
                    &shard.device_id,
                )
                .map(|sequence| (shard, device, sequence))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut submitted: Vec<(
            &VulkanComputeDevice,
            &VulkanDistributedDispatchShardRunner,
            &VulkanResidentKernelSequence,
        )> = Vec::with_capacity(dispatch.shards.len());
        for (shard, device, sequence) in resolved_shards {
            let synchronization = dispatch
                .helper_synchronization
                .iter()
                .find(|sync| sync.device_id == shard.device_id);
            let wait_points = synchronization
                .filter(|_| consume_owner_ready_signal)
                .map(|sync| {
                    vec![sync.helper_ready(dependency_value)]
                })
                .unwrap_or_default();
            let signal_points = synchronization
                .filter(|_| prepare_owner_continuation || dispatch.coordinator_sequence().is_some())
                .map(|sync| {
                    vec![sync.helper_done(dependency_value)]
                })
                .unwrap_or_default();
            let shard_signal_completion = signal_completion && dispatch.coordinator_sequence().is_none();
            let submission = if let Some(submission_batch) = submission_batch {
                submission_batch.enqueue_recorded_sequence(
                    device,
                    sequence,
                    &wait_points,
                    &signal_points,
                    shard_signal_completion,
                )
            } else if shard_signal_completion {
                device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    sequence,
                    &wait_points,
                    &signal_points,
                )
            } else {
                device.submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                    sequence,
                    &wait_points,
                    &signal_points,
                )
            };
            if let Err(error) = submission {
                for (submitted_device, _, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.leader().node_id,
                    shard.device_id
                )));
            }
            submitted.push((device, shard, sequence));
        }
        if let Some(coordinator_sequence) = dispatch.coordinator_sequence() {
            let owner_device = device_for(&dispatch.planned.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed coordinator owner {:?}: {error}",
                    dispatch.planned.owner_device_id
                ))
            })?;
            let wait_points = dispatch
                .helper_synchronization
                .iter()
                .map(|sync| sync.owner_done(dependency_value))
                .collect::<Vec<_>>();
            let coordinator_submission = if let Some(submission_batch) = submission_batch {
                submission_batch.enqueue_recorded_sequence(
                    owner_device,
                    coordinator_sequence,
                    &wait_points,
                    &[],
                    signal_completion,
                )
            } else if signal_completion {
                owner_device.submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    coordinator_sequence,
                    &wait_points,
                    &[],
                )
            } else {
                owner_device
                    .submit_recorded_resident_kernel_sequence_unfenced_with_timeline_semaphores(
                        coordinator_sequence,
                        &wait_points,
                        &[],
                    )
            };
            if let Err(error) = coordinator_submission {
                for (submitted_device, _, submitted_sequence) in &submitted {
                    let _ = submitted_device.wait_resident_kernel_sequence(submitted_sequence);
                }
                return Err(VulkanDistributedDispatchRunnerError(format!(
                    "failed to submit distributed coordinator {}.{} on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    dispatch.planned.owner_device_id
                )));
            }
        }

        Ok(VulkanDistributedDispatchRun {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            component_id: dispatch.planned.leader().component_id.clone(),
            node_id: dispatch.planned.tail().node_id.clone(),
            shard_count: dispatch.shards.len(),
            sequence_kind,
        })
    }

    pub(crate) fn pending_residency_dispatches(
        &self,
    ) -> Result<Vec<(String, usize)>, VulkanDistributedDispatchRunnerError> {
        let mut pending = Vec::new();
        for dispatch in &self.dispatches {
            let has_pending = dispatch
                .shards
                .iter()
                .flat_map(|shard| shard.selected_resource_gates.iter().flatten())
                .try_fold(false, |pending, gate| {
                    Ok::<_, VulkanDistributedDispatchRunnerError>(
                        pending
                            || gate.notification_epoch()? != gate.observed_notification_epoch(),
                    )
                })?;
            if has_pending {
                pending.push((
                    dispatch.planned.owner_device_id.clone(),
                    dispatch.planned.leader().dispatch_index,
                ));
            }
        }
        Ok(pending)
    }

    pub(crate) fn reset_residency_predicates(
        &self,
    ) -> Result<(), VulkanDistributedDispatchRunnerError> {
        for gate in self
            .dispatches
            .iter()
            .flat_map(|dispatch| &dispatch.shards)
            .flat_map(|shard| shard.selected_resource_gates.iter().flatten())
        {
            gate.reset_local_predicate()?;
        }
        Ok(())
    }

    pub(crate) fn resolve_completed_residency_fault<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        sequence_kind: VulkanDistributedDispatchSequenceKind,
        mut device_for: F,
    ) -> Result<Option<VulkanDistributedResolvedResidencyFault>, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut affected_shards = Vec::new();
        let mut misses = Vec::new();
        for (shard_index, (shard, device)) in resolved_shards.iter().enumerate() {
            let mut affected = false;
            for (gate_index, gate) in shard.selected_resource_gates.iter().flatten().enumerate() {
                if let Some(miss) = gate.resolve_completed_miss(device)? {
                    affected = true;
                    misses.push(VulkanDistributedResolvedResidencyMiss {
                        shard_index,
                        gate_index,
                        selector_id: miss.selector_id,
                        checkpoint_tag: miss.checkpoint_tag,
                        resource_indices: miss.resource_indices,
                    });
                }
            }
            if affected {
                affected_shards.push((*shard, *device));
            }
        }
        if affected_shards.is_empty() {
            return Ok(None);
        }
        let mut restored = BTreeSet::new();
        for predicate in self.transaction_predicates.values() {
            if restored.insert(Arc::as_ptr(predicate) as usize) {
                predicate
                    .write_bytes(&1u32.to_le_bytes())
                    .map_err(VulkanDistributedDispatchRunnerError::from)?;
            }
        }
        for (shard, device) in affected_shards {
            let sequence = distributed_sequence_for_kind(
                &shard.sequence,
                shard.feedback_sequence.as_ref(),
                sequence_kind,
                &shard.device_id,
            )?;
            device
                .run_recorded_resident_kernel_sequence(sequence)
                .map_err(VulkanDistributedDispatchRunnerError::from)?;
            for gate in shard.selected_resource_gates.iter().flatten() {
                if gate.notification_epoch()? != gate.observed_notification_epoch() {
                    return Err(VulkanDistributedDispatchRunnerError(format!(
                        "distributed selected-resource checkpoint on {:?} faulted again immediately after loading",
                        shard.device_id
                    )));
                }
            }
        }
        if let Some(coordinator_sequence) = dispatch.coordinator_sequence() {
            let owner_device = device_for(&dispatch.planned.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed coordinator owner {:?}: {error}",
                    dispatch.planned.owner_device_id
                ))
            })?;
            owner_device
                .run_recorded_resident_kernel_sequence(coordinator_sequence)
                .map_err(VulkanDistributedDispatchRunnerError::from)?;
        }
        Ok(Some(VulkanDistributedResolvedResidencyFault {
            owner_device_id: owner_device_id.to_string(),
            dispatch_index,
            misses,
        }))
    }

    pub fn wait_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        sequence_kind: VulkanDistributedDispatchSequenceKind,
        mut device_for: F,
    ) -> Result<(), VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dispatch = self.dispatch(owner_device_id, dispatch_index).ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed runner has no dispatch {dispatch_index} owned by {owner_device_id:?}"
            ))
        })?;
        let resolved_shards = dispatch
            .shards
            .iter()
            .map(|shard| {
                device_for(&shard.device_id)
                    .map(|device| (shard, device))
                    .map_err(|error| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "failed to resolve distributed shard device {:?}: {error}",
                            shard.device_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut first_error = None;
        for (shard, device) in &resolved_shards {
            let sequence = distributed_sequence_for_kind(
                &shard.sequence,
                shard.feedback_sequence.as_ref(),
                sequence_kind,
                &shard.device_id,
            )?;
            if let Err(error) = device.wait_resident_kernel_sequence(sequence)
                && first_error.is_none()
            {
                first_error = Some(format!(
                    "failed waiting for distributed dispatch {}.{} shard on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    shard.device_id
                ));
            }
        }
        if let Some(coordinator_sequence) = dispatch.coordinator_sequence() {
            let owner_device = device_for(&dispatch.planned.owner_device_id).map_err(|error| {
                VulkanDistributedDispatchRunnerError(format!(
                    "failed to resolve distributed coordinator owner {:?}: {error}",
                    dispatch.planned.owner_device_id
                ))
            })?;
            if let Err(error) = owner_device.wait_resident_kernel_sequence(coordinator_sequence)
                && first_error.is_none()
            {
                first_error = Some(format!(
                    "failed waiting for distributed coordinator {}.{} on {:?}: {error}",
                    dispatch.planned.leader().component_id,
                    dispatch.planned.tail().node_id,
                    dispatch.planned.owner_device_id
                ));
            }
        }
        if let Some(error) = first_error {
            return Err(VulkanDistributedDispatchRunnerError(error));
        }
        let _ = self.resolve_completed_residency_fault(
            owner_device_id,
            dispatch_index,
            sequence_kind,
            |device_id| device_for(device_id),
        )?;
        Ok(())
    }

    pub fn run_dispatch<'a, F, E>(
        &self,
        owner_device_id: &str,
        dispatch_index: usize,
        mut device_for: F,
    ) -> Result<VulkanDistributedDispatchRun, VulkanDistributedDispatchRunnerError>
    where
        F: FnMut(&str) -> Result<&'a VulkanComputeDevice, E>,
        E: Display,
    {
        let dependency_value = self.reserve_dependency_value(owner_device_id, dispatch_index)?;
        let run = self.submit_dispatch_with_device_dependencies(
            owner_device_id,
            dispatch_index,
            VulkanDistributedDispatchSubmission {
                dependency_value,
                consume_owner_ready_signal: false,
                prepare_owner_continuation: true,
                signal_completion: true,
                sequence_kind: VulkanDistributedDispatchSequenceKind::Direct,
            },
            None,
            |device_id| device_for(device_id),
        )?;
        self.wait_dispatch(
            owner_device_id,
            dispatch_index,
            run.sequence_kind,
            |device_id| device_for(device_id),
        )?;
        Ok(run)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRun {
    pub owner_device_id: String,
    pub dispatch_index: usize,
    pub component_id: String,
    pub node_id: String,
    pub shard_count: usize,
    pub sequence_kind: VulkanDistributedDispatchSequenceKind,
}

pub struct VulkanDistributedDispatchRunner {
    pub planned: VulkanPhysicalExecutionIslandPlan,
    pub shards: Vec<VulkanDistributedDispatchShardRunner>,
    helper_synchronization: Vec<VulkanDistributedQueueSynchronization>,
    reduction: Option<VulkanDistributedReductionRunner>,
    residency_commit: Option<VulkanDistributedResidencyCommitRunner>,
    dependency_clock: VulkanDistributedDependencyClock,
}

impl VulkanDistributedDispatchRunner {
    fn coordinator_sequence(&self) -> Option<&VulkanResidentKernelSequence> {
        self.reduction
            .as_ref()
            .map(|reduction| &reduction.sequence)
            .or_else(|| {
                self.residency_commit
                    .as_ref()
                    .map(|commit| &commit.sequence)
            })
    }
}

pub struct VulkanDistributedDispatchShardRunner {
    pub device_id: String,
    pub planned: Vec<VulkanDistributedDispatchShard>,
    pub resident_dispatches: Vec<VulkanResidentKernelDispatch>,
    pub(crate) selected_resource_gates:
        Vec<Vec<VulkanDistributedSelectedResourceGate>>,
    pub sequence: VulkanResidentKernelSequence,
    feedback_sequence: Option<VulkanResidentKernelSequence>,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDistributedDispatchRunnerError(pub String);

impl Display for VulkanDistributedDispatchRunnerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanDistributedDispatchRunnerError {}

impl From<VulkanError> for VulkanDistributedDispatchRunnerError {
    fn from(error: VulkanError) -> Self {
        Self(error.to_string())
    }
}
