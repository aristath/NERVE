use crate::vulkan_stream_circuit::{
    VulkanPlacementDeviceExecutionIdentity, VulkanPlacementExecutionCaseIdentity,
    VulkanPlacementExecutionStrategy, VulkanPlacementOperationGeometry,
    VulkanPlacementSelectedResourceFragmentIdentity,
    vulkan_distributed_execution_artifact_digest, vulkan_distributed_execution_equivalence,
    vulkan_distributed_execution_graph_digest, vulkan_distributed_execution_operations,
};

/// Converts a measured structurally equivalent transaction into the exact
/// identity of the current component instance. Performance evidence may be
/// shared by repeated compiled layers, but execution never borrows another
/// layer's contract, operation, or selected-resource labels.
fn rebind_exact_execution_case_to_runtime_component(
    execution_plan: &VulkanDistributedExecutionPlan,
    component_id: &str,
    dispatch_indices: &[usize],
    measured_case: &VulkanPlacementExecutionCaseIdentity,
    loaded_manifest: &VulkanLoadedKernelArtifactCatalog,
) -> Result<VulkanPlacementExecutionCaseIdentity, VulkanDistributedPlanError> {
    let mut runtime_contracts = BTreeMap::<String, String>::new();
    for dispatch_index in dispatch_indices {
        let dispatch = execution_plan.dispatches.get(*dispatch_index).ok_or_else(|| {
            VulkanDistributedPlanError(format!(
                "exact case for component {component_id:?} references a runtime dispatch outside the plan",
            ))
        })?;
        if runtime_contracts
            .insert(
                dispatch.physical_execution_contract_id.clone(),
                dispatch.implementation_digest.clone(),
            )
            .is_some_and(|existing| existing != dispatch.implementation_digest)
        {
            return exact_case_error(format!(
                "compiled component {component_id:?} repeats one physical contract with conflicting implementations",
            ));
        }
    }
    let runtime_contract_ids = runtime_contracts.keys().cloned().collect::<Vec<_>>();
    let runtime_implementation_digests = runtime_contracts.values().cloned().collect::<Vec<_>>();
    let mut measured_implementation_digests = measured_case.implementation_digests.clone();
    let mut structural_runtime_implementation_digests = runtime_implementation_digests.clone();
    measured_implementation_digests.sort();
    structural_runtime_implementation_digests.sort();
    if measured_case.contract_ids.len() != runtime_contract_ids.len()
        || measured_implementation_digests != structural_runtime_implementation_digests
    {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with different physical execution contracts or implementations",
        ));
    }

    let runtime_artifact_digest = vulkan_distributed_execution_artifact_digest(
        loaded_manifest,
        execution_plan,
        dispatch_indices,
    )
    .map_err(|error| VulkanDistributedPlanError(error.to_string()))?;
    if runtime_artifact_digest != measured_case.artifact_digest {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with different executable artifacts",
        ));
    }
    let row_extents = dispatch_indices
        .iter()
        .map(|dispatch_index| {
            let rows = execution_plan.dispatches[*dispatch_index].output_rows;
            (rows, rows)
        })
        .collect::<Vec<_>>();
    let runtime_operations = vulkan_distributed_execution_operations(
        loaded_manifest,
        execution_plan,
        dispatch_indices,
        &row_extents,
    )
    .map_err(|error| VulkanDistributedPlanError(error.to_string()))?;
    if !exact_operations_are_structurally_equivalent(
        &measured_case.operations,
        &runtime_operations,
    ) {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with different operation geometry",
        ));
    }
    let runtime_graph_digest = vulkan_distributed_execution_graph_digest(
        &measured_case.behavior.compiled_execution_signature,
        execution_plan,
        dispatch_indices,
    )
    .map_err(|error| VulkanDistributedPlanError(error.to_string()))?;
    if runtime_graph_digest != measured_case.execution_graph_digest {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with a different distributed execution graph",
        ));
    }
    let runtime_equivalence =
        vulkan_distributed_execution_equivalence(execution_plan, dispatch_indices)
            .map_err(|error| VulkanDistributedPlanError(error.to_string()))?;
    if runtime_equivalence != measured_case.equivalence {
        return exact_case_error(format!(
            "exact case for component {component_id:?} was measured with different output or state equivalence",
        ));
    }

    let mut rebound = measured_case.clone();
    rebound.contract_ids = runtime_contract_ids;
    rebound.implementation_digests = runtime_implementation_digests;
    rebound.artifact_digest = runtime_artifact_digest;
    rebound.operations = runtime_operations;
    rebound.execution_graph_digest = runtime_graph_digest;
    rebound.equivalence = runtime_equivalence;
    for (dispatch_ordinal, dispatch_index) in dispatch_indices.iter().enumerate() {
        let dispatch = &execution_plan.dispatches[*dispatch_index];
        for (participant_ordinal, runtime_shard) in dispatch.shards.iter().enumerate() {
            let exact_shard = rebound
                .shards
                .iter_mut()
                .find(|shard| {
                    shard.dispatch_ordinal == dispatch_ordinal
                        && shard.participant_ordinal == participant_ordinal
                })
                .ok_or_else(|| {
                    VulkanDistributedPlanError(format!(
                        "exact case for {component_id:?} dispatch {dispatch_ordinal} omits participant ordinal {participant_ordinal}",
                    ))
                })?;
            let runtime_fragments = exact_runtime_selected_resource_fragments(
                &dispatch.selected_resource_partitions,
                runtime_shard,
            )?;
            if !exact_selected_resource_fragments_are_structurally_equivalent(
                &exact_shard.selected_resource_fragments_by_partition,
                &runtime_fragments,
            ) {
                return exact_case_error(format!(
                    "exact case for {component_id:?} dispatch {dispatch_ordinal} was measured with different selected-resource fragment geometry",
                ));
            }
            exact_shard.selected_resource_fragments_by_partition = runtime_fragments;
        }
    }
    Ok(rebound)
}

fn exact_operations_are_structurally_equivalent(
    measured: &[VulkanPlacementOperationGeometry],
    runtime: &[VulkanPlacementOperationGeometry],
) -> bool {
    measured.len() == runtime.len()
        && measured.iter().zip(runtime).all(|(measured, runtime)| {
            match (measured, runtime) {
                (
                    VulkanPlacementOperationGeometry::Dispatch {
                        geometry: measured,
                    },
                    VulkanPlacementOperationGeometry::Dispatch { geometry: runtime },
                ) => {
                    measured.logical_extent == runtime.logical_extent
                        && measured.sampled_extent == runtime.sampled_extent
                        && measured.input_width == runtime.input_width
                        && measured.workgroup_count_x == runtime.workgroup_count_x
                        && measured.local_size_x == runtime.local_size_x
                }
                (
                    VulkanPlacementOperationGeometry::DirectedTransfer {
                        byte_count: measured,
                        ..
                    },
                    VulkanPlacementOperationGeometry::DirectedTransfer {
                        byte_count: runtime,
                        ..
                    },
                ) => measured == runtime,
                (
                    VulkanPlacementOperationGeometry::Reduction {
                        element_count: measured_elements,
                        element_byte_count: measured_bytes,
                        participant_count: measured_participants,
                        ..
                    },
                    VulkanPlacementOperationGeometry::Reduction {
                        element_count: runtime_elements,
                        element_byte_count: runtime_bytes,
                        participant_count: runtime_participants,
                        ..
                    },
                ) => {
                    measured_elements == runtime_elements
                        && measured_bytes == runtime_bytes
                        && measured_participants == runtime_participants
                }
                (
                    VulkanPlacementOperationGeometry::LazyLoadWave {
                        resource_count: measured_resources,
                        byte_count: measured_bytes,
                        ..
                    },
                    VulkanPlacementOperationGeometry::LazyLoadWave {
                        resource_count: runtime_resources,
                        byte_count: runtime_bytes,
                        ..
                    },
                ) => measured_resources == runtime_resources && measured_bytes == runtime_bytes,
                (
                    VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                        resource_execution_class_id: measured_class,
                        selector_selection_count: measured_selections,
                        executed_resource_occurrence_count: measured_occurrences,
                        ..
                    },
                    VulkanPlacementOperationGeometry::SelectedResourceTransaction {
                        resource_execution_class_id: runtime_class,
                        selector_selection_count: runtime_selections,
                        executed_resource_occurrence_count: runtime_occurrences,
                        ..
                    },
                ) => {
                    measured_class == runtime_class
                        && measured_selections == runtime_selections
                        && measured_occurrences == runtime_occurrences
                }
                _ => false,
            }
        })
}

fn exact_selected_resource_fragments_are_structurally_equivalent(
    measured: &BTreeMap<usize, Vec<VulkanPlacementSelectedResourceFragmentIdentity>>,
    runtime: &BTreeMap<usize, Vec<VulkanPlacementSelectedResourceFragmentIdentity>>,
) -> bool {
    let structure = |fragments: &BTreeMap<
        usize,
        Vec<VulkanPlacementSelectedResourceFragmentIdentity>,
    >| {
        fragments
            .iter()
            .map(|(partition, fragments)| {
                (
                    *partition,
                    fragments
                        .iter()
                        .map(|fragment| {
                            (
                                fragment.resource_index,
                                fragment.logical_start,
                                fragment.logical_count,
                                fragment
                                    .parameters
                                    .iter()
                                    .map(|parameter| {
                                        (
                                            parameter.parameter_slot,
                                            parameter.resource_byte_count,
                                            parameter.byte_offset,
                                            parameter.byte_count,
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    structure(measured) == structure(runtime)
}
