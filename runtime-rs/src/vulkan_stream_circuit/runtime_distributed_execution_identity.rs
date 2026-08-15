/// Derives the executable binary identity for one exact distributed
/// component transaction. Semantic artifact IDs and source paths are omitted:
/// only the interface and SPIR-V bytes that will execute are identity.
pub(crate) fn vulkan_distributed_execution_artifact_digest(
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    execution_plan: &VulkanDistributedExecutionPlan,
    dispatch_indices: &[usize],
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if dispatch_indices.is_empty() {
        return distributed_execution_identity_error(
            "distributed execution artifact identity is empty",
        );
    }
    let mut digest = Sha256::new();
    digest.update(b"nerve.distributed_execution_artifacts.v3\0");
    for dispatch_index in dispatch_indices {
        let dispatch = execution_plan.dispatches.get(*dispatch_index).ok_or_else(|| {
            distributed_execution_identity_error_value(format!(
                "distributed execution artifact identity references dispatch index {dispatch_index} outside the runtime plan",
            ))
        })?;
        let artifact = loaded_manifest
            .physical_artifact(&dispatch.physical_artifact_id)
            .ok_or_else(|| {
                distributed_execution_identity_error_value(format!(
                    "distributed execution artifact identity is missing physical artifact {:?}",
                    dispatch.physical_artifact_id,
                ))
            })?;
        vulkan_distributed_execution_update_artifact_digest(&mut digest, artifact)?;
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

pub(crate) fn vulkan_distributed_execution_update_artifact_digest(
    digest: &mut Sha256,
    artifact: &VulkanLoadedPhysicalKernelArtifact,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let interface = serde_json::to_vec(&(
        artifact.artifact.entry_point.as_str(),
        artifact.artifact.local_size_x,
        artifact.artifact.workgroup_count_x,
        &artifact.artifact.descriptor_signature,
        &artifact.artifact.push_constants,
        artifact.artifact.stream_control_binding,
    ))
    .map_err(|error| {
        distributed_execution_identity_error_value(format!(
            "failed to encode distributed execution artifact interface: {error}",
        ))
    })?;
    digest.update((interface.len() as u64).to_le_bytes());
    digest.update(interface);
    digest.update((artifact.words.len() as u64).to_le_bytes());
    for word in &artifact.words {
        digest.update(word.to_le_bytes());
    }
    Ok(())
}

/// Derives the exact dispatch and reduction geometry for one distributed
/// component transaction. `row_extents` carries `(logical, executed)` rows so
/// the calibration producer can describe bounded samples; normal runtime
/// replay supplies the full extent for both values.
pub(crate) fn vulkan_distributed_execution_operations(
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
    execution_plan: &VulkanDistributedExecutionPlan,
    dispatch_indices: &[usize],
    row_extents: &[(usize, usize)],
) -> Result<Vec<VulkanPlacementOperationGeometry>, VulkanResidentTokenModelPackageError> {
    if dispatch_indices.is_empty() || dispatch_indices.len() != row_extents.len() {
        return distributed_execution_identity_error(
            "distributed execution operation identity has incomplete dispatch geometry",
        );
    }
    let mut operations = Vec::with_capacity(dispatch_indices.len());
    for (dispatch_index, (logical_extent, sampled_extent)) in
        dispatch_indices.iter().zip(row_extents)
    {
        let dispatch = execution_plan.dispatches.get(*dispatch_index).ok_or_else(|| {
            distributed_execution_identity_error_value(format!(
                "distributed execution operation identity references dispatch index {dispatch_index} outside the runtime plan",
            ))
        })?;
        if *logical_extent == 0 || *sampled_extent == 0 || sampled_extent > logical_extent {
            return distributed_execution_identity_error(
                "distributed execution operation identity has invalid logical or sampled rows",
            );
        }
        let artifact = loaded_manifest
            .physical_artifact(&dispatch.physical_artifact_id)
            .ok_or_else(|| {
                distributed_execution_identity_error_value(format!(
                    "distributed execution operation identity is missing physical artifact {:?}",
                    dispatch.physical_artifact_id,
                ))
            })?;
        let workgroup_count_x = dispatch.shards.iter().try_fold(0u32, |total, shard| {
            total.checked_add(shard.workgroup_count_x).ok_or_else(|| {
                distributed_execution_identity_error_value(
                    "distributed execution workgroup geometry overflowed",
                )
            })
        })?;
        operations.push(VulkanPlacementOperationGeometry::Dispatch {
            geometry: VulkanPlacementDispatchGeometry {
                contract_id: dispatch.physical_execution_contract_id.clone(),
                logical_extent: *logical_extent,
                sampled_extent: *sampled_extent,
                input_width: dispatch.input_width,
                workgroup_count_x,
                local_size_x: artifact.artifact.local_size_x,
            },
        });
        if let Some(reduction) = vulkan_distributed_execution_reduction_geometry(
            &dispatch.physical_execution_contract_id,
            dispatch.reduction.as_ref(),
            dispatch.shards.len(),
        )? {
            operations.push(reduction);
        }
    }
    Ok(operations)
}

pub(crate) fn vulkan_distributed_execution_equivalence(
    execution_plan: &VulkanDistributedExecutionPlan,
    dispatch_indices: &[usize],
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let dispatches = dispatch_indices
        .iter()
        .map(|dispatch_index| {
            execution_plan.dispatches.get(*dispatch_index).ok_or_else(|| {
                distributed_execution_identity_error_value(format!(
                    "distributed execution equivalence references dispatch index {dispatch_index} outside the runtime plan",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tolerant_intermediates_are_compiler_local = dispatches
        .windows(2)
        .all(|pair| {
            pair[0].equivalence.output == VulkanDistributedEquivalenceKind::BitExact
                || execution_plan.execution_islands.iter().any(|island| {
                    island.dispatches.windows(2).any(|island_pair| {
                        distributed_dispatch_identity_matches(&island_pair[0], pair[0])
                            && distributed_dispatch_identity_matches(&island_pair[1], pair[1])
                    })
                })
        });
    vulkan_distributed_execution_equivalence_from_dispatches(
        &dispatches,
        tolerant_intermediates_are_compiler_local,
    )
}

fn distributed_dispatch_identity_matches(
    left: &VulkanDistributedDispatchPlan,
    right: &VulkanDistributedDispatchPlan,
) -> bool {
    left.dispatch_index == right.dispatch_index
        && left.component_id == right.component_id
        && left.node_id == right.node_id
        && left.physical_execution_contract_id == right.physical_execution_contract_id
}

fn vulkan_distributed_execution_equivalence_from_dispatches(
    dispatches: &[&VulkanDistributedDispatchPlan],
    tolerant_intermediates_are_compiler_local: bool,
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let contracts = dispatches
        .iter()
        .map(|dispatch| {
            (
                dispatch.equivalence.clone(),
                dispatch
                    .reduction
                    .as_ref()
                    .map(|reduction| reduction.finalization.clone()),
            )
        })
        .collect::<Vec<_>>();
    vulkan_distributed_execution_equivalence_from_contracts(
        &contracts,
        tolerant_intermediates_are_compiler_local,
    )
}

fn vulkan_distributed_execution_equivalence_from_contracts(
    contracts: &[(
        crate::VulkanDistributedEquivalencePlan,
        Option<VulkanDistributedReductionFinalizationPlan>,
    )],
    tolerant_intermediates_are_compiler_local: bool,
) -> Result<VulkanPlacementEquivalenceIdentity, VulkanResidentTokenModelPackageError> {
    let Some((tail, tail_finalization)) = contracts.last() else {
        return distributed_execution_identity_error(
            "distributed execution equivalence requires an executable dispatch",
        );
    };
    if contracts[..contracts.len() - 1]
        .iter()
        .any(|(equivalence, _)| equivalence.output != VulkanDistributedEquivalenceKind::BitExact)
        && !tolerant_intermediates_are_compiler_local
    {
        return distributed_execution_identity_error(
            "distributed execution cannot compose a tolerant intermediate outside a compiler-declared local physical island",
        );
    }
    if contracts
        .iter()
        .any(|(equivalence, _)| equivalence.state != VulkanDistributedEquivalenceKind::BitExact)
    {
        return distributed_execution_identity_error(
            "distributed execution cannot validate tolerant state without a typed compiled state layout",
        );
    }
    let output = match tail.output {
        VulkanDistributedEquivalenceKind::BitExact => VulkanPlacementEquivalenceKind::BitExact,
        VulkanDistributedEquivalenceKind::AbsoluteRelativeTolerance => {
            VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance
        }
    };
    let output_scalar_format = match output {
        VulkanPlacementEquivalenceKind::BitExact => None,
        VulkanPlacementEquivalenceKind::AbsoluteRelativeTolerance => match tail_finalization {
            Some(VulkanDistributedReductionFinalizationPlan::StoreF32) => {
                Some(VulkanPlacementScalarFormat::F32)
            }
            Some(
                VulkanDistributedReductionFinalizationPlan::StoreF32ToBf16
                | VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 { .. }
                | VulkanDistributedReductionFinalizationPlan::ScaleByPackedBf16InputToBf16 { .. },
            ) => Some(VulkanPlacementScalarFormat::Bf16),
            None => {
                return distributed_execution_identity_error(
                    "tolerant distributed output has no typed reduction finalization",
                );
            }
        },
    };
    let equivalence = VulkanPlacementEquivalenceIdentity {
        output,
        state: VulkanPlacementEquivalenceKind::BitExact,
        absolute_tolerance_bits: tail.absolute_tolerance_bits,
        relative_tolerance_bits: tail.relative_tolerance_bits,
        output_scalar_format,
    };
    equivalence
        .validate()
        .map_err(|error| distributed_execution_identity_error_value(error.to_string()))?;
    Ok(equivalence)
}

pub(crate) fn vulkan_distributed_execution_reduction_geometry(
    contract_id: &str,
    reduction: Option<&VulkanDistributedReductionPlan>,
    participant_count: usize,
) -> Result<Option<VulkanPlacementOperationGeometry>, VulkanResidentTokenModelPackageError> {
    let Some(reduction) = reduction else {
        return Ok(None);
    };
    // A staged calibration first executes a partition contract on one device
    // to fix the exact workload later used by wider candidates. That
    // degenerate finalization is executable and has complete geometry. Catalog
    // validation separately requires at least two participants before a
    // distributed reduction observation can be published for replay.
    if contract_id.is_empty() || reduction.element_count == 0 || participant_count == 0 {
        return Err(distributed_execution_identity_error_value(format!(
            "distributed execution reduction geometry is incomplete: contract_id={contract_id:?}, element_count={}, participant_count={participant_count}",
            reduction.element_count,
        )));
    }
    Ok(Some(VulkanPlacementOperationGeometry::Reduction {
        contract_id: contract_id.to_string(),
        element_count: reduction.element_count,
        element_byte_count: size_of::<f32>(),
        participant_count,
    }))
}

/// Hashes the physical topology that connects the selected distributed
/// dispatches. Component, node, device, selector, and resource labels are
/// intentionally excluded so compiler-equivalent repeated layers can reuse a
/// measurement. Their executable signature, binaries, devices, shards, and
/// routes are validated independently.
pub(crate) fn vulkan_distributed_execution_graph_digest(
    compiled_execution_signature: &str,
    execution_plan: &VulkanDistributedExecutionPlan,
    dispatch_indices: &[usize],
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if compiled_execution_signature.is_empty() || dispatch_indices.is_empty() {
        return distributed_execution_identity_error(
            "distributed execution graph identity is incomplete",
        );
    }
    let selected_dispatch_ordinals = dispatch_indices
        .iter()
        .enumerate()
        .map(|(ordinal, dispatch_index)| (*dispatch_index, ordinal))
        .collect::<BTreeMap<_, _>>();
    if selected_dispatch_ordinals.len() != dispatch_indices.len() {
        return distributed_execution_identity_error(
            "distributed execution graph identity repeats a dispatch index",
        );
    }

    let dispatches = dispatch_indices
        .iter()
        .map(|dispatch_index| {
            let dispatch = execution_plan.dispatches.get(*dispatch_index).ok_or_else(|| {
                distributed_execution_identity_error_value(format!(
                    "distributed execution graph identity references dispatch index {dispatch_index} outside the runtime plan",
                ))
            })?;
            if dispatch.auxiliary_input_activations.len()
                != dispatch.auxiliary_input_distributions.len()
            {
                return distributed_execution_identity_error(
                    "distributed execution graph has incomplete auxiliary input topology",
                );
            }
            Ok(serde_json::json!({
                "implementation_digest": dispatch.implementation_digest,
                "strategy": distributed_execution_strategy_name(dispatch.execution_strategy),
                "contract_member_count": dispatch.contract_member_node_ids.len(),
                "local_intermediates": dispatch.local_intermediates.iter().map(|intermediate| serde_json::json!({
                    "producer_binding": intermediate.producer_binding,
                    "consumer_binding": intermediate.consumer_binding,
                    "format": intermediate.format,
                })).collect::<Vec<_>>(),
                "input_byte_capacity": dispatch.input_byte_capacity,
                "output_byte_capacity": dispatch.output_byte_capacity,
                "output_rows": dispatch.output_rows,
                "input_width": dispatch.input_width,
                "row_alignment": dispatch.row_alignment,
                "has_lazy_resource_requirements": dispatch.has_lazy_resource_requirements,
                "input_activation": distributed_execution_activation_graph_identity(&dispatch.input_activation),
                "input_distribution": distributed_input_distribution_name(dispatch.input_distribution),
                "auxiliary_inputs": dispatch.auxiliary_input_activations.iter().zip(&dispatch.auxiliary_input_distributions).map(|(activation, distribution)| serde_json::json!({
                    "activation": distributed_execution_activation_graph_identity(activation),
                    "distribution": distributed_input_distribution_name(*distribution),
                })).collect::<Vec<_>>(),
                "selected_resource_activations": dispatch.selected_resource_activations.iter()
                    .map(distributed_execution_activation_graph_identity)
                    .collect::<Vec<_>>(),
                "output_activation": distributed_execution_activation_graph_identity(&dispatch.output_activation),
                "output_collection": distributed_output_collection_name(dispatch.output_collection),
                "reduction": distributed_execution_reduction_graph_identity(dispatch.reduction.as_ref()),
                "distribution": distributed_dispatch_distribution_name(dispatch.distribution),
                "selected_resource_partitions": dispatch.selected_resource_partitions.iter().map(|partition| serde_json::json!({
                    "resource_count": partition.resource_count,
                    "parameters_per_resource": partition.parameters_per_resource,
                    "address_table_binding": partition.address_table_binding,
                    "parameter_slots_binding": partition.parameter_slots_binding,
                    "selection_count_per_activation": partition.selection_count_per_activation,
                    "parameter_partitions": partition.parameter_partitions.iter().map(|parameter| serde_json::json!({
                        "parameter_slot": parameter.parameter_slot,
                        "dimension": parameter.dimension,
                        "kind": distributed_parameter_partition_kind_name(parameter.kind),
                        "alignment_elements": parameter.alignment_elements,
                        "logical_elements_per_index": parameter.logical_elements_per_index,
                    })).collect::<Vec<_>>(),
                    "resource_operation_class_ids": partition.resource_operation_class_ids,
                    "atomic_group_byte_counts": partition.atomic_group_byte_counts,
                    "parameter_resource_byte_counts": partition.parameter_resource_byte_counts,
                })).collect::<Vec<_>>(),
            }))
        })
        .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?;

    let islands = execution_plan
        .execution_islands
        .iter()
        .filter_map(|island| {
            let ordinals = island
                .dispatches
                .iter()
                .filter_map(|dispatch| {
                    execution_plan
                        .dispatches
                        .iter()
                        .position(|candidate| {
                            candidate.dispatch_index == dispatch.dispatch_index
                                && candidate.component_id == dispatch.component_id
                                && candidate.node_id == dispatch.node_id
                        })
                        .and_then(|index| selected_dispatch_ordinals.get(&index).copied())
                })
                .collect::<Vec<_>>();
            (!ordinals.is_empty()).then_some(ordinals)
        })
        .collect::<Vec<_>>();
    if islands.is_empty() {
        return distributed_execution_identity_error(
            "distributed execution graph identity has no physical execution island",
        );
    }
    let mut covered_ordinals = islands.iter().flatten().copied().collect::<Vec<_>>();
    covered_ordinals.sort_unstable();
    if covered_ordinals.into_iter().ne(0..dispatch_indices.len()) {
        return distributed_execution_identity_error(
            "distributed execution graph identity does not cover every selected dispatch exactly once",
        );
    }

    let payload = serde_json::to_vec(&serde_json::json!({
        "schema": "nerve.distributed_execution_graph.v4",
        "compiled_execution_signature": compiled_execution_signature,
        "dispatches": dispatches,
        "islands": islands,
    }))
    .map_err(|error| {
        distributed_execution_identity_error_value(format!(
            "failed to encode distributed execution graph identity: {error}",
        ))
    })?;
    Ok(format!("sha256:{:x}", Sha256::digest(payload)))
}

fn distributed_execution_activation_graph_identity(
    activation: &VulkanDistributedActivationSlot,
) -> serde_json::Value {
    serde_json::json!({
        "binding": activation.binding,
        "slot": activation.slot,
        "byte_capacity": activation.byte_capacity,
        "signal_byte_capacity": activation.signal_byte_capacity,
        "storage": match activation.storage {
            VulkanDistributedActivationStorage::ActivationSlot => "activation_slot",
            VulkanDistributedActivationStorage::BoundaryInput => "boundary_input",
            VulkanDistributedActivationStorage::BoundaryOutput => "boundary_output",
            VulkanDistributedActivationStorage::Edge { .. } => "edge",
        },
    })
}

fn distributed_execution_reduction_graph_identity(
    reduction: Option<&VulkanDistributedReductionPlan>,
) -> serde_json::Value {
    let Some(reduction) = reduction else {
        return serde_json::Value::Null;
    };
    serde_json::json!({
        "operation": match reduction.operation {
            nerve_execution_contracts::ReductionOperation::SumF32 => "sum_f32",
        },
        "element_count": reduction.element_count,
        "partial_byte_capacity": reduction.partial_byte_capacity,
        "finalization": match reduction.finalization {
            VulkanDistributedReductionFinalizationPlan::StoreF32 => serde_json::json!({"kind": "store_f32"}),
            VulkanDistributedReductionFinalizationPlan::StoreF32ToBf16 => serde_json::json!({"kind": "store_f32_to_bf16"}),
            VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 { residual_input_index } => serde_json::json!({
                "kind": "add_bf16_residual_to_bf16",
                "residual_input_index": residual_input_index,
            }),
            VulkanDistributedReductionFinalizationPlan::ScaleByPackedBf16InputToBf16 {
                scale_input_index,
                elements_per_scale,
                scale_bit_offset,
            } => serde_json::json!({
                "kind": "scale_by_packed_bf16_input_to_bf16",
                "scale_input_index": scale_input_index,
                "elements_per_scale": elements_per_scale,
                "scale_bit_offset": scale_bit_offset,
            }),
        },
    })
}

fn distributed_execution_strategy_name(
    strategy: nerve_execution_contracts::ExecutionStrategy,
) -> &'static str {
    match strategy {
        nerve_execution_contracts::ExecutionStrategy::SingleDevice => "single_device",
        nerve_execution_contracts::ExecutionStrategy::TensorParallel => "tensor_parallel",
        nerve_execution_contracts::ExecutionStrategy::ExpertParallel => "expert_parallel",
        nerve_execution_contracts::ExecutionStrategy::TensorParallelExpert => {
            "tensor_parallel_expert"
        }
    }
}

fn distributed_input_distribution_name(
    distribution: nerve_execution_contracts::InputDistribution,
) -> &'static str {
    match distribution {
        nerve_execution_contracts::InputDistribution::Replicated => "replicated",
        nerve_execution_contracts::InputDistribution::Sharded => "sharded",
        nerve_execution_contracts::InputDistribution::Routed => "routed",
        nerve_execution_contracts::InputDistribution::Local => "local",
    }
}

fn distributed_output_collection_name(
    collection: nerve_execution_contracts::OutputCollection,
) -> &'static str {
    match collection {
        nerve_execution_contracts::OutputCollection::Local => "local",
        nerve_execution_contracts::OutputCollection::Concatenated => "concatenated",
        nerve_execution_contracts::OutputCollection::Reduced => "reduced",
        nerve_execution_contracts::OutputCollection::Routed => "routed",
        nerve_execution_contracts::OutputCollection::Retained => "retained",
    }
}

fn distributed_dispatch_distribution_name(
    distribution: VulkanDistributedDispatchDistribution,
) -> &'static str {
    match distribution {
        VulkanDistributedDispatchDistribution::OutputRows => "output_rows",
        VulkanDistributedDispatchDistribution::InputColumns => "input_columns",
        VulkanDistributedDispatchDistribution::ExpertRange => "expert_range",
    }
}

fn distributed_parameter_partition_kind_name(
    kind: nerve_execution_contracts::ParameterPartitionKind,
) -> &'static str {
    match kind {
        nerve_execution_contracts::ParameterPartitionKind::Contiguous => "contiguous",
        nerve_execution_contracts::ParameterPartitionKind::BlockCyclic => "block_cyclic",
        nerve_execution_contracts::ParameterPartitionKind::ExpertRange => "expert_range",
    }
}

fn distributed_execution_identity_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(distributed_execution_identity_error_value(message))
}

fn distributed_execution_identity_error_value(
    message: impl Into<String>,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(message.into())
}
