use nerve_execution_contracts::{
    ArtifactRole, ExecutionForm, ExecutionPhase, ExecutionShape, InputDistribution,
    OutputCollection, ParameterPartitionKind, PartitionOrigin, PhysicalExecutionContract,
    ReductionFinalization, ReductionOperation, WorkgroupXMapping,
};

struct ContractParameterSlice<'a> {
    binding: usize,
    tensor: &'a str,
    bytes_per_physical_index: usize,
    logical_elements_per_index: usize,
}

fn resolved_owner_residency_requirements(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
) -> Result<Vec<VulkanPhysicalExecutionResidencyRequirement>, VulkanDistributedPlanError> {
    let mut requirements = BTreeMap::<
        (VulkanPhysicalExecutionResidencyKind, String),
        usize,
    >::new();
    for descriptor in &dispatch.descriptors {
        let resolved = match &descriptor.resource {
            VulkanDescriptorResourceAddress::RuntimeControl {
                runtime_source,
                byte_capacity,
            } => Some((
                VulkanPhysicalExecutionResidencyKind::OwnerControl,
                format!("runtime:{runtime_source}"),
                *byte_capacity,
            )),
            VulkanDescriptorResourceAddress::StateBuffer {
                component_id,
                state_id,
                byte_capacity,
                ..
            }
            | VulkanDescriptorResourceAddress::StateView {
                component_id,
                state_id,
                byte_capacity,
                ..
            } => Some((
                VulkanPhysicalExecutionResidencyKind::OwnerState,
                format!("state:{component_id}:{state_id}"),
                *byte_capacity,
            )),
            VulkanDescriptorResourceAddress::SelectionTelemetry {
                component_id,
                node_id,
                domain_id,
                byte_capacity,
                ..
            } => Some((
                VulkanPhysicalExecutionResidencyKind::OwnerSelectionTelemetry,
                format!("selection:{component_id}:{node_id}:{domain_id}"),
                *byte_capacity,
            )),
            VulkanDescriptorResourceAddress::BoundaryInput { .. }
            | VulkanDescriptorResourceAddress::BoundaryOutput { .. }
            | VulkanDescriptorResourceAddress::PermanentParameter { .. }
            | VulkanDescriptorResourceAddress::DynamicResourceAddressTable { .. }
            | VulkanDescriptorResourceAddress::DynamicResourceParameterSlots { .. }
            | VulkanDescriptorResourceAddress::ActivationSlot { .. } => None,
        };
        let Some((kind, resource_id, byte_capacity)) = resolved else {
            continue;
        };
        let key = (kind, resource_id.clone());
        if let Some(existing) = requirements.insert(key, byte_capacity)
            && existing != byte_capacity
        {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "resource {resource_id:?} has conflicting capacities {existing} and {byte_capacity}"
                ),
            ));
        }
    }
    Ok(requirements
        .into_iter()
        .map(|((kind, resource_id), byte_capacity)| {
            VulkanPhysicalExecutionResidencyRequirement {
                device_id: owner_device_id.to_string(),
                kind,
                resource_id,
                byte_capacity,
            }
        })
        .collect())
}

fn select_distributed_contract<'a, 'b>(
    dispatch: &'a VulkanPreparedDispatch,
    artifact_manifest: &'b VulkanPhysicalKernelArtifactManifest,
    phase: ExecutionPhase,
    execution_shape: ExecutionShape,
) -> Result<
    Option<(
        &'a PhysicalExecutionContract,
        &'b crate::vulkan_stream_circuit::VulkanPhysicalKernelArtifact,
    )>,
    VulkanDistributedPlanError,
> {
    let mut candidates = Vec::new();
    for contract in dispatch.physical_execution_contracts.iter().filter(|contract| {
        contract.strategy.is_distributed()
            && contract.phases.contains(&phase)
            && contract.execution_shape.supports(execution_shape)
            && contract.operation_family == dispatch.op
            && contract.member_node_ids.contains(&dispatch.node_id)
    }) {
        contract
            .validate()
            .map_err(|error| dispatch_error(dispatch, format!("has an invalid contract: {error}")))?;
        let primary_artifacts = contract
            .artifacts
            .iter()
            .enumerate()
            .filter(|(_, artifact)| artifact.role == ArtifactRole::Primary)
            .collect::<Vec<_>>();
        let [(artifact_index, identity)] = primary_artifacts.as_slice() else {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "physical contract {:?} requires exactly one primary artifact, found {}",
                    contract.contract_id, primary_artifacts.len()
                ),
            ));
        };
        let artifact_id = crate::vulkan_stream_circuit::physical_execution_artifact_id(
            &contract.contract_id,
            *artifact_index,
        );
        let artifact = artifact_manifest
            .artifact(&artifact_id)
            .ok_or_else(|| {
                dispatch_error(
                    dispatch,
                    format!(
                        "physical contract {:?} has no loaded artifact {:?}",
                        contract.contract_id, artifact_id
                    ),
                )
            })?;
        let local_size_x = contract.geometry.dimensions.get("local_size_x").copied();
        let workgroup_count_x = contract
            .geometry
            .dimensions
            .get("workgroup_count_x")
            .copied();
        if artifact.op != contract.operation_family
            || artifact.path != identity.path
            || artifact.entry_point != identity.entry_point
            || local_size_x != Some(u64::from(artifact.local_size_x))
            || workgroup_count_x != Some(u64::from(artifact.workgroup_count_x))
        {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "physical artifact {:?} disagrees with contract {:?}",
                    artifact.artifact_id, contract.contract_id
                ),
            ));
        }
        candidates.push((contract, artifact));
    }
    let [candidate] = candidates.as_slice() else {
        if candidates.is_empty() {
            return Ok(None);
        }
        return Err(dispatch_error(
            dispatch,
            format!(
                "has {} ambiguous {phase:?} distribution contracts for reusable artifact family {:?}",
                candidates.len(),
                dispatch.reusable_family_id
            ),
        ));
    };
    Ok(Some(*candidate))
}

#[allow(clippy::too_many_arguments)]
fn plan_contract_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    edge_placements: &[ComponentEdgePlacement],
    artifact: &crate::vulkan_stream_circuit::VulkanPhysicalKernelArtifact,
    contract: &PhysicalExecutionContract,
    storage_buffer_offset_alignment: usize,
    resource_context: Option<(&str, &CompiledResourceResidencyContract)>,
) -> Result<Option<VulkanDistributedDispatchPlan>, VulkanDistributedPlanError> {
    let extent = contract.partition_extent.as_ref().ok_or_else(|| {
        dispatch_error(
            dispatch,
            "distributed contract has no partition extent".to_string(),
        )
    })?;
    let launch = contract.partition_launch.as_ref().ok_or_else(|| {
        dispatch_error(
            dispatch,
            "distributed contract has no partition launch".to_string(),
        )
    })?;
    validate_contract_descriptor_coverage(dispatch, contract)?;
    let selected_resource_partitions = resolve_selected_resource_partitions(
        dispatch,
        contract,
        resource_context,
    )?;
    let logical_extent = usize::try_from(extent.elements)
        .map_err(|_| dispatch_error(dispatch, "partition extent exceeds usize".to_string()))?;
    let mut logical_alignment = usize::try_from(extent.alignment_elements)
        .map_err(|_| dispatch_error(dispatch, "partition alignment exceeds usize".to_string()))?;
    let contract_workgroup_count = contract
        .geometry
        .dimensions
        .get("workgroup_count_x")
        .copied()
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                "distributed contract omits workgroup_count_x geometry".to_string(),
            )
        })?;
    if contract_workgroup_count != u64::from(artifact.workgroup_count_x) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "contract workgroup count {contract_workgroup_count} disagrees with artifact count {}",
                artifact.workgroup_count_x
            ),
        ));
    }
    validate_partition_origin(dispatch, artifact, launch)?;

    let mut inputs = Vec::with_capacity(contract.inputs.len());
    for input in &contract.inputs {
        if input.distribution == InputDistribution::Local {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "distributed contract marks input binding {} as local",
                    input.binding
                ),
            ));
        }
        let binding = usize::try_from(input.binding)
            .map_err(|_| dispatch_error(dispatch, "input binding exceeds usize".to_string()))?;
        let activation = distributed_activation(
            dispatch,
            binding,
            1,
            "contract input",
            edge_placements,
        )?
        .ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("input binding {binding} cannot be distributed"),
            )
        })?;
        if input.distribution == InputDistribution::Sharded {
            logical_alignment = aligned_activation_partition(
                dispatch,
                logical_alignment,
                logical_extent,
                activation.signal_byte_capacity,
                storage_buffer_offset_alignment,
                "input",
            )?;
        }
        inputs.push((input, activation));
    }
    let Some((_, primary_input)) = inputs.first() else {
        return Err(dispatch_error(
            dispatch,
            "distributed contract has no primary input".to_string(),
        ));
    };

    let [output_contract] = contract.outputs.as_slice() else {
        return Err(dispatch_error(
            dispatch,
            format!(
                "distributed execution currently requires one output, contract declares {}",
                contract.outputs.len()
            ),
        ));
    };
    if matches!(output_contract.collection, OutputCollection::Local) {
        return Err(dispatch_error(
            dispatch,
            "distributed contract marks its output as local".to_string(),
        ));
    }
    let output_binding = usize::try_from(output_contract.binding)
        .map_err(|_| dispatch_error(dispatch, "output binding exceeds usize".to_string()))?;
    let output_activation = distributed_activation(
        dispatch,
        output_binding,
        1,
        "contract output",
        edge_placements,
    )?
    .ok_or_else(|| {
        dispatch_error(
            dispatch,
            format!("output binding {output_binding} cannot be distributed"),
        )
    })?;
    let reduction = output_contract
        .reduction
        .as_ref()
        .map(|reduction| {
            if contract.formats.accumulation != "f32" {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "reduced output requires f32 accumulation, contract declares {:?}",
                        contract.formats.accumulation
                    ),
                ));
            }
            let element_count = contract
                .geometry
                .dimensions
                .get(&reduction.dimension_name)
                .copied()
                .ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        format!(
                            "reduction dimension {:?} is not declared",
                            reduction.dimension_name
                        ),
                    )
                })
                .and_then(|elements| {
                    usize::try_from(elements).map_err(|_| {
                        dispatch_error(dispatch, "reduction element count exceeds usize".to_string())
                    })
                })?;
            let partial_byte_capacity = element_count.checked_mul(size_of::<f32>()).ok_or_else(|| {
                dispatch_error(dispatch, "reduction partial byte capacity overflowed".to_string())
            })?;
            let finalization = match &reduction.finalization {
                ReductionFinalization::StoreF32 => {
                    if output_activation.signal_byte_capacity != partial_byte_capacity {
                        return Err(dispatch_error(
                            dispatch,
                            format!(
                                "sum_f32 reduction produces {partial_byte_capacity} bytes but output signal {} has {} bytes",
                                output_activation.signal_id,
                                output_activation.signal_byte_capacity
                            ),
                        ));
                    }
                    VulkanDistributedReductionFinalizationPlan::StoreF32
                }
                ReductionFinalization::AddBf16ResidualToBf16 { residual_binding } => {
                    if !element_count.is_multiple_of(2) {
                        return Err(dispatch_error(
                            dispatch,
                            "BF16 reduction finalization requires an even element count"
                                .to_string(),
                        ));
                    }
                    let residual_input_index = contract
                        .inputs
                        .iter()
                        .position(|input| input.binding == *residual_binding)
                        .ok_or_else(|| {
                            dispatch_error(
                                dispatch,
                                "BF16 reduction residual binding is absent".to_string(),
                            )
                        })?;
                    let bf16_byte_capacity = element_count.checked_mul(2).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "BF16 reduction output capacity overflowed".to_string(),
                        )
                    })?;
                    let residual_activation = &inputs[residual_input_index].1;
                    if residual_activation.signal_byte_capacity != bf16_byte_capacity
                        || output_activation.signal_byte_capacity != bf16_byte_capacity
                    {
                        return Err(dispatch_error(
                            dispatch,
                            format!(
                                "BF16 reduction finalization requires {bf16_byte_capacity} residual and output bytes, found {} and {}",
                                residual_activation.signal_byte_capacity,
                                output_activation.signal_byte_capacity
                            ),
                        ));
                    }
                    VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
                        residual_input_index,
                    }
                }
            };
            Ok(VulkanDistributedReductionPlan {
                operation: reduction.operation,
                element_count,
                partial_byte_capacity,
                finalization,
            })
        })
        .transpose()?;
    if output_contract.collection == OutputCollection::Concatenated {
        logical_alignment = aligned_activation_partition(
            dispatch,
            logical_alignment,
            logical_extent,
            output_activation.signal_byte_capacity,
            storage_buffer_offset_alignment,
            "output",
        )?;
    }

    let parameter_slices = contract_parameter_slices(
        dispatch,
        tensor_index,
        contract,
        logical_extent,
        &mut logical_alignment,
    )?;
    let (distribution, workgroup_elements) = match contract.execution_form {
        ExecutionForm::ReplicatedInputPartitionedOutput => {
            let artifact_groups = usize::try_from(artifact.workgroup_count_x).map_err(|_| {
                dispatch_error(dispatch, "artifact workgroup count exceeds usize".to_string())
            })?;
            if artifact_groups == 0 || !logical_extent.is_multiple_of(artifact_groups) {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "partition extent {logical_extent} is incompatible with {} artifact workgroups",
                        artifact.workgroup_count_x
                    ),
                ));
            }
            let workgroup_elements = logical_extent / artifact_groups;
            logical_alignment = least_common_multiple(logical_alignment, workgroup_elements)
                .ok_or_else(|| {
                    dispatch_error(dispatch, "logical shard alignment overflowed".to_string())
                })?;
            (
                VulkanDistributedDispatchDistribution::OutputRows,
                workgroup_elements,
            )
        }
        ExecutionForm::PartitionedInputPartialOutput => (
            VulkanDistributedDispatchDistribution::InputColumns,
            1,
        ),
        ExecutionForm::WholeExpertOwnership => (
            VulkanDistributedDispatchDistribution::ExpertRange,
            1,
        ),
        ExecutionForm::Local => unreachable!("distributed contract validation rejects local form"),
    };
    let raw_shards = distribute_rows(
        logical_extent,
        device_ids.len(),
        workgroup_elements,
        logical_alignment,
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
        .map(|(device_id, (logical_start, logical_count))| {
            let parameters = parameter_slices
                .iter()
                .map(|parameter| {
                    if !logical_start.is_multiple_of(parameter.logical_elements_per_index)
                        || !logical_count.is_multiple_of(parameter.logical_elements_per_index)
                    {
                        return Err(dispatch_error(
                            dispatch,
                            format!(
                                "logical shard {logical_start}..{} does not align tensor {:?}",
                                logical_start + logical_count,
                                parameter.tensor
                            ),
                        ));
                    }
                    parameter_fragment(
                        parameter.binding,
                        parameter.tensor,
                        parameter.bytes_per_physical_index,
                        logical_start / parameter.logical_elements_per_index,
                        logical_count / parameter.logical_elements_per_index,
                        dispatch,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            distributed_parameter_byte_count = parameters.iter().try_fold(
                distributed_parameter_byte_count,
                |total, parameter| {
                    total.checked_add(parameter.byte_count).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "distributed parameter byte count overflowed".to_string(),
                        )
                    })
                },
            )?;
            let input_ranges = inputs
                .iter()
                .map(|(input, activation)| {
                    contract_activation_range(
                        dispatch,
                        input.distribution,
                        activation.signal_byte_capacity,
                        logical_extent,
                        logical_start,
                        logical_count,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let shard_output_byte_capacity = reduction
                .as_ref()
                .map(|reduction| reduction.partial_byte_capacity)
                .unwrap_or(output_activation.signal_byte_capacity);
            let output_range = contract_output_range(
                dispatch,
                output_contract.collection,
                shard_output_byte_capacity,
                logical_extent,
                logical_start,
                logical_count,
            )?;
            let workgroup_count_x = match distribution {
                VulkanDistributedDispatchDistribution::OutputRows => u32::try_from(
                    logical_count / workgroup_elements,
                )
                .map_err(|_| {
                    dispatch_error(dispatch, "shard workgroup count exceeds u32".to_string())
                })?,
                VulkanDistributedDispatchDistribution::InputColumns
                | VulkanDistributedDispatchDistribution::ExpertRange => {
                    artifact.workgroup_count_x
                }
            };
            let base_workgroup_z = match launch.origin {
                PartitionOrigin::LocalZero => 0,
                PartitionOrigin::PushConstantU32 => u32::try_from(logical_start).map_err(|_| {
                    dispatch_error(dispatch, "partition origin exceeds u32".to_string())
                })?,
            };
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
                selected_resource_indices: if distribution
                    == VulkanDistributedDispatchDistribution::ExpertRange
                {
                    selected_resource_partitions
                        .iter()
                        .map(|partition| {
                            (
                                partition.selector_id.clone(),
                                (logical_start..logical_start + logical_count).collect(),
                            )
                        })
                        .collect()
                } else {
                    BTreeMap::new()
                },
                row_start: logical_start,
                row_count: logical_count,
                workgroup_count_x,
                base_workgroup_z,
                input_range: input_ranges[0].clone(),
                auxiliary_input_ranges: input_ranges[1..].to_vec(),
                output_byte_offset: output_range.byte_offset,
                output_byte_count: output_range.byte_count,
                parameters,
            })
        })
        .collect::<Result<Vec<_>, VulkanDistributedPlanError>>()?;

    let input_byte_capacity = primary_input.signal_byte_capacity;
    let input_width = contract
        .geometry
        .dimensions
        .get("parameter_0_dimension_1")
        .and_then(|value| usize::try_from(*value).ok())
        .unwrap_or(input_byte_capacity);
    Ok(Some(VulkanDistributedDispatchPlan {
        owner_device_id: owner_device_id.to_string(),
        dispatch_index: dispatch.dispatch_index,
        component_id: dispatch.component_id.clone(),
        node_id: dispatch.node_id.clone(),
        physical_artifact_id: artifact.artifact_id.clone(),
        physical_execution_contract_id: contract.contract_id.clone(),
        implementation_digest: contract.implementation_digest.clone(),
        contract_member_node_ids: contract.member_node_ids.clone(),
        local_intermediates: contract.local_intermediates.clone(),
        has_lazy_resource_requirements: contract.resources.iter().any(|resource| {
            resource.kind == nerve_execution_contracts::ResourceKind::LazyResource
        }),
        selected_resource_partitions,
        owner_residency_requirements: resolved_owner_residency_requirements(
            owner_device_id,
            dispatch,
        )?,
        input_byte_capacity,
        output_byte_capacity: output_activation.signal_byte_capacity,
        output_rows: logical_extent,
        input_width,
        row_alignment: logical_alignment,
        input_activation: primary_input.clone(),
        input_distribution: contract.inputs[0].distribution,
        auxiliary_input_activations: inputs
            .into_iter()
            .skip(1)
            .map(|(_, activation)| activation)
            .collect(),
        auxiliary_input_distributions: contract
            .inputs
            .iter()
            .skip(1)
            .map(|input| input.distribution)
            .collect(),
        output_activation,
        output_collection: output_contract.collection,
        reduction,
        distribution,
        distributed_parameter_byte_count,
        shards,
    }))
}

fn resolve_selected_resource_partitions(
    dispatch: &VulkanPreparedDispatch,
    contract: &PhysicalExecutionContract,
    resource_context: Option<(&str, &CompiledResourceResidencyContract)>,
) -> Result<Vec<VulkanDistributedSelectedResourcePartitionPlan>, VulkanDistributedPlanError> {
    if contract.selected_resource_partitions.is_empty() {
        return Ok(Vec::new());
    }
    let Some((execution_scope, residency)) = resource_context else {
        return Err(dispatch_error(
            dispatch,
            "contains selected resources without a compiled atomic residency contract"
                .to_string(),
        ));
    };
    let groups = residency
        .atomic_groups
        .iter()
        .map(|group| (group.id.as_str(), group))
        .collect::<BTreeMap<_, _>>();
    let resources = residency
        .resources
        .iter()
        .map(|resource| (resource.id.as_str(), resource))
        .collect::<BTreeMap<_, _>>();
    contract
        .selected_resource_partitions
        .iter()
        .map(|partition| {
            let matching_selectors = residency
                .selectors
                .iter()
                .filter(|selector| {
                    selector.execution_scope == execution_scope
                        && selector.component_id == dispatch.component_id
                        && selector.selection_signal == partition.selection_signal
                })
                .collect::<Vec<_>>();
            let [selector] = matching_selectors.as_slice() else {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "selected resource partition {:?} resolves {} residency selectors",
                        partition.selection_signal,
                        matching_selectors.len(),
                    ),
                ));
            };
            if selector.resource_count != usize::try_from(partition.resource_count).unwrap_or(usize::MAX)
            {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "selected resource partition {:?} declares {} resources but selector {:?} declares {}",
                        partition.selection_signal,
                        partition.resource_count,
                        selector.id,
                        selector.resource_count,
                    ),
                ));
            }
            let CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } =
                &selector.mapping
            else {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "selected resource partition {:?} requires an atomic group table",
                        partition.selection_signal,
                    ),
                ));
            };
            if atomic_group_ids.len() != selector.resource_count {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "selector {:?} has {} atomic groups for {} resources",
                        selector.id,
                        atomic_group_ids.len(),
                        selector.resource_count,
                    ),
                ));
            }
            let parameters_per_resource = usize::try_from(partition.parameters_per_resource)
                .map_err(|_| {
                    dispatch_error(
                        dispatch,
                        "selected parameter count exceeds usize".to_string(),
                    )
                })?;
            let mut bindings = BTreeMap::<(usize, usize), (&str, &str)>::new();
            for binding in &residency.bindings {
                let CompiledResourceBindingMapping::SelectedAtomicGroup {
                    atomic_group_id,
                    resource_id,
                    selection_signal,
                    selector_index,
                    parameter_slot,
                } = &binding.mapping
                else {
                    continue;
                };
                if binding.execution_scope != execution_scope
                    || binding.component_id != dispatch.component_id
                    || binding.node_id != dispatch.node_id
                    || selection_signal != &partition.selection_signal
                {
                    continue;
                }
                if bindings
                    .insert(
                        (*selector_index, *parameter_slot),
                        (atomic_group_id.as_str(), resource_id.as_str()),
                    )
                    .is_some()
                {
                    return Err(dispatch_error(
                        dispatch,
                        "selected residency repeats a selector parameter slot".to_string(),
                    ));
                }
            }
            let expected_binding_count = selector
                .resource_count
                .checked_mul(parameters_per_resource)
                .ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        "selected residency binding count overflowed".to_string(),
                    )
                })?;
            if bindings.len() != expected_binding_count {
                return Err(dispatch_error(
                    dispatch,
                    format!(
                        "selected resource partition {:?} resolves {} parameter slots, expected {expected_binding_count}",
                        partition.selection_signal,
                        bindings.len(),
                    ),
                ));
            }
            let mut atomic_group_byte_counts = Vec::with_capacity(atomic_group_ids.len());
            for (selector_index, group_id) in atomic_group_ids.iter().enumerate() {
                let group = groups.get(group_id.as_str()).ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        format!("selector {:?} references missing atomic group {group_id:?}", selector.id),
                    )
                })?;
                if group.lifetime != CompiledResourceLifetime::Dynamic {
                    return Err(dispatch_error(
                        dispatch,
                        format!("selected atomic group {group_id:?} is not dynamic"),
                    ));
                }
                let group_resources = group.resource_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
                for parameter_slot in 0..parameters_per_resource {
                    let Some((binding_group_id, resource_id)) =
                        bindings.get(&(selector_index, parameter_slot))
                    else {
                        return Err(dispatch_error(
                            dispatch,
                            "selected residency parameter slots are incomplete".to_string(),
                        ));
                    };
                    if *binding_group_id != group_id
                        || !group_resources.contains(resource_id)
                    {
                        return Err(dispatch_error(
                            dispatch,
                            format!(
                                "selected residency slot {selector_index}:{parameter_slot} escapes atomic group {group_id:?}",
                            ),
                        ));
                    }
                }
                let group_bytes = group.resource_ids.iter().try_fold(0usize, |total, resource_id| {
                    let resource = resources.get(resource_id.as_str()).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            format!("selected atomic group references missing resource {resource_id:?}"),
                        )
                    })?;
                    let bytes = resource.source_byte_count().map_err(|error| {
                        dispatch_error(dispatch, error.to_string())
                    })?;
                    total.checked_add(bytes).ok_or_else(|| {
                        dispatch_error(
                            dispatch,
                            "selected atomic group byte count overflowed".to_string(),
                        )
                    })
                })?;
                atomic_group_byte_counts.push(group_bytes);
            }
            Ok(VulkanDistributedSelectedResourcePartitionPlan {
                execution_scope: execution_scope.to_string(),
                selector_id: selector.id.clone(),
                node_id: selector.node_id.clone(),
                domain_id: selector.domain_id.clone(),
                selection_signal: partition.selection_signal.clone(),
                address_table_binding: usize::try_from(partition.address_table_binding)
                    .map_err(|_| dispatch_error(dispatch, "dynamic binding exceeds usize".to_string()))?,
                parameter_slots_binding: usize::try_from(partition.parameter_slots_binding)
                    .map_err(|_| dispatch_error(dispatch, "dynamic binding exceeds usize".to_string()))?,
                resource_count: selector.resource_count,
                parameters_per_resource,
                selection_count_per_activation: selector
                    .encoding
                    .selection_count_per_activation,
                atomic_group_ids: atomic_group_ids.clone(),
                atomic_group_byte_counts,
            })
        })
        .collect()
}

fn validate_contract_descriptor_coverage(
    dispatch: &VulkanPreparedDispatch,
    contract: &PhysicalExecutionContract,
) -> Result<(), VulkanDistributedPlanError> {
    let descriptor_bindings = |usage| {
        dispatch
            .descriptors
            .iter()
            .filter(|descriptor| descriptor.usage == usage)
            .map(|descriptor| descriptor.binding)
            .collect::<BTreeSet<_>>()
    };
    let contract_inputs = contract
        .inputs
        .iter()
        .map(|input| usize::try_from(input.binding))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|_| dispatch_error(dispatch, "input binding exceeds usize".to_string()))?;
    let contract_outputs = contract
        .outputs
        .iter()
        .map(|output| usize::try_from(output.binding))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|_| dispatch_error(dispatch, "output binding exceeds usize".to_string()))?;
    let contract_parameters = contract
        .parameter_partitions
        .iter()
        .map(|partition| usize::try_from(partition.binding))
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|_| dispatch_error(dispatch, "parameter binding exceeds usize".to_string()))?;
    let mut contract_dynamic_addresses = BTreeMap::new();
    let mut contract_dynamic_slots = BTreeMap::new();
    for partition in &contract.selected_resource_partitions {
        let address_binding = usize::try_from(partition.address_table_binding).map_err(|_| {
            dispatch_error(
                dispatch,
                "dynamic address-table binding exceeds usize".to_string(),
            )
        })?;
        let slots_binding = usize::try_from(partition.parameter_slots_binding).map_err(|_| {
            dispatch_error(
                dispatch,
                "dynamic parameter-slot binding exceeds usize".to_string(),
            )
        })?;
        contract_dynamic_addresses.insert(address_binding, partition);
        contract_dynamic_slots.insert(slots_binding, partition);
    }
    let descriptor_parameters = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| {
            matches!(
                descriptor.resource,
                VulkanDescriptorResourceAddress::PermanentParameter { .. }
            )
            .then_some(descriptor.binding)
        })
        .collect::<BTreeSet<_>>();
    let descriptor_dynamic_addresses = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::DynamicResourceAddressTable {
                component_id,
                node_id,
                selection_signal,
            } => Some((
                descriptor.binding,
                (component_id.as_str(), node_id.as_str(), selection_signal.as_str()),
            )),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let descriptor_dynamic_slots = dispatch
        .descriptors
        .iter()
        .filter_map(|descriptor| match &descriptor.resource {
            VulkanDescriptorResourceAddress::DynamicResourceParameterSlots {
                component_id,
                node_id,
                selection_signal,
                parameter_ids,
            } => Some((
                descriptor.binding,
                (
                    component_id.as_str(),
                    node_id.as_str(),
                    selection_signal.as_str(),
                    parameter_ids.len(),
                ),
            )),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let descriptor_inputs = descriptor_bindings(VulkanKernelDescriptorUsage::InputSignal);
    let descriptor_outputs = descriptor_bindings(VulkanKernelDescriptorUsage::OutputSignal);
    let dynamic_resources_match = contract_dynamic_addresses.iter().all(
        |(binding, partition)| {
            descriptor_dynamic_addresses.get(binding).is_some_and(
                |(component_id, node_id, selection_signal)| {
                    *component_id == dispatch.component_id
                        && *node_id == dispatch.node_id
                        && *selection_signal == partition.selection_signal
                },
            )
        },
    ) && contract_dynamic_slots.iter().all(|(binding, partition)| {
        let expected_parameter_count = partition
            .resource_count
            .checked_mul(partition.parameters_per_resource)
            .and_then(|count| usize::try_from(count).ok());
        descriptor_dynamic_slots.get(binding).is_some_and(
            |(component_id, node_id, selection_signal, parameter_count)| {
                *component_id == dispatch.component_id
                    && *node_id == dispatch.node_id
                    && *selection_signal == partition.selection_signal
                    && Some(*parameter_count) == expected_parameter_count
            },
        )
    }) && contract_dynamic_addresses.len() == descriptor_dynamic_addresses.len()
        && contract_dynamic_slots.len() == descriptor_dynamic_slots.len();
    if contract_inputs != descriptor_inputs
        || contract_outputs != descriptor_outputs
        || contract_parameters != descriptor_parameters
        || !dynamic_resources_match
    {
        return Err(dispatch_error(
            dispatch,
            format!(
                "contract bindings inputs={contract_inputs:?} outputs={contract_outputs:?} parameters={contract_parameters:?} dynamic_addresses={:?} dynamic_slots={:?} disagree with artifact ABI inputs={descriptor_inputs:?} outputs={descriptor_outputs:?} parameters={descriptor_parameters:?} dynamic_addresses={descriptor_dynamic_addresses:?} dynamic_slots={descriptor_dynamic_slots:?}",
                contract_dynamic_addresses.keys().collect::<Vec<_>>(),
                contract_dynamic_slots.keys().collect::<Vec<_>>(),
            ),
        ));
    }
    Ok(())
}

fn validate_partition_origin(
    dispatch: &VulkanPreparedDispatch,
    artifact: &crate::vulkan_stream_circuit::VulkanPhysicalKernelArtifact,
    launch: &nerve_execution_contracts::PartitionLaunch,
) -> Result<(), VulkanDistributedPlanError> {
    match launch.origin {
        PartitionOrigin::LocalZero if artifact.push_constants.is_empty() => Ok(()),
        PartitionOrigin::PushConstantU32 => {
            let Some(name) = launch.origin_push_constant.as_deref() else {
                return Err(dispatch_error(
                    dispatch,
                    "partition-origin push constant is unnamed".to_string(),
                ));
            };
            let mut expected = vec![VulkanKernelScalarBinding {
                    name: name.to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
            }];
            if launch.workgroup_x == WorkgroupXMapping::Repeated {
                let count_name = launch.count_push_constant.as_deref().ok_or_else(|| {
                    dispatch_error(
                        dispatch,
                        "repeated partition count push constant is unnamed".to_string(),
                    )
                })?;
                expected.push(VulkanKernelScalarBinding {
                    name: count_name.to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                });
            }
            if artifact.push_constants == expected {
                Ok(())
            } else {
                Err(dispatch_error(
                    dispatch,
                    format!(
                        "partition contract requires exact push-constant ABI {expected:?}"
                    ),
                ))
            }
        }
        PartitionOrigin::LocalZero => Err(dispatch_error(
            dispatch,
            "local-zero partition contract forbids push constants".to_string(),
        )),
    }
}

fn contract_parameter_slices<'a>(
    dispatch: &'a VulkanPreparedDispatch,
    tensor_index: &'a TensorIndex,
    contract: &'a PhysicalExecutionContract,
    logical_extent: usize,
    logical_alignment: &mut usize,
) -> Result<Vec<ContractParameterSlice<'a>>, VulkanDistributedPlanError> {
    let mut slices = Vec::with_capacity(contract.parameter_partitions.len());
    for partition in &contract.parameter_partitions {
        if partition.kind == ParameterPartitionKind::BlockCyclic {
            return Err(dispatch_error(
                dispatch,
                "block-cyclic parameter partitioning is not implemented".to_string(),
            ));
        }
        if partition.dimension != 0 {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "contiguous parameter binding {} partitions unsupported dimension {}",
                    partition.binding, partition.dimension
                ),
            ));
        }
        let binding = usize::try_from(partition.binding).map_err(|_| {
            dispatch_error(dispatch, "parameter binding exceeds usize".to_string())
        })?;
        let tensor = partition.resource.as_str();
        let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
            dispatch_error(dispatch, format!("has no tensor metadata for {tensor:?}"))
        })?;
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
        let logical_elements_per_index =
            usize::try_from(partition.logical_elements_per_index).map_err(|_| {
                dispatch_error(
                    dispatch,
                    format!("tensor {tensor:?} logical index ratio exceeds usize"),
                )
            })?;
        if !logical_extent.is_multiple_of(logical_elements_per_index) {
            return Err(dispatch_error(
                dispatch,
                format!("tensor {tensor:?} does not divide the logical extent"),
            ));
        }
        let physical_extent = logical_extent / logical_elements_per_index;
        if metadata.shape.first().copied() != Some(physical_extent) {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "tensor {tensor:?} leading dimension {:?} disagrees with contract extent {physical_extent}",
                    metadata.shape.first()
                ),
            ));
        }
        let byte_count = metadata.byte_count.ok_or_else(|| {
            dispatch_error(
                dispatch,
                format!("parameter tensor {tensor:?} has no byte count"),
            )
        })?;
        if byte_count == 0 || !byte_count.is_multiple_of(physical_extent) {
            return Err(dispatch_error(
                dispatch,
                format!(
                    "tensor {tensor:?} byte count {byte_count} is not divisible by {physical_extent} physical indices"
                ),
            ));
        }
        let physical_alignment = usize::try_from(partition.alignment_elements).map_err(|_| {
            dispatch_error(
                dispatch,
                format!("tensor {tensor:?} partition alignment exceeds usize"),
            )
        })?;
        let mut tensor_logical_alignment = physical_alignment
            .checked_mul(logical_elements_per_index)
            .ok_or_else(|| {
                dispatch_error(
                    dispatch,
                    format!("tensor {tensor:?} partition alignment overflowed"),
                )
            })?;
        if metadata.layout.as_deref() == Some("vulkan_bf16_row_pair_u32") {
            tensor_logical_alignment = least_common_multiple(
                tensor_logical_alignment,
                2usize.checked_mul(logical_elements_per_index).ok_or_else(|| {
                    dispatch_error(dispatch, "packed row alignment overflowed".to_string())
                })?,
            )
            .ok_or_else(|| dispatch_error(dispatch, "packed row alignment overflowed".to_string()))?;
        }
        *logical_alignment = least_common_multiple(*logical_alignment, tensor_logical_alignment)
            .ok_or_else(|| {
                dispatch_error(dispatch, "parameter shard alignment overflowed".to_string())
            })?;
        slices.push(ContractParameterSlice {
            binding,
            tensor,
            bytes_per_physical_index: byte_count / physical_extent,
            logical_elements_per_index,
        });
    }
    Ok(slices)
}

fn aligned_activation_partition(
    dispatch: &VulkanPreparedDispatch,
    current_alignment: usize,
    logical_extent: usize,
    byte_capacity: usize,
    storage_buffer_offset_alignment: usize,
    role: &str,
) -> Result<usize, VulkanDistributedPlanError> {
    if !byte_capacity.is_multiple_of(logical_extent) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "{role} activation capacity {byte_capacity} is not divisible by logical extent {logical_extent}"
            ),
        ));
    }
    let bytes_per_element = byte_capacity / logical_extent;
    let offset_alignment = storage_buffer_offset_alignment
        / greatest_common_divisor(storage_buffer_offset_alignment, bytes_per_element);
    least_common_multiple(current_alignment, offset_alignment).ok_or_else(|| {
        dispatch_error(dispatch, format!("{role} activation alignment overflowed"))
    })
}

fn contract_activation_range(
    dispatch: &VulkanPreparedDispatch,
    distribution: InputDistribution,
    byte_capacity: usize,
    logical_extent: usize,
    logical_start: usize,
    logical_count: usize,
) -> Result<VulkanDistributedActivationRange, VulkanDistributedPlanError> {
    match distribution {
        InputDistribution::Replicated | InputDistribution::Routed => {
            Ok(VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: byte_capacity,
            })
        }
        InputDistribution::Sharded => proportional_activation_range(
            dispatch,
            byte_capacity,
            logical_extent,
            logical_start,
            logical_count,
            "input",
        ),
        InputDistribution::Local => unreachable!("local input was rejected before planning"),
    }
}

fn contract_output_range(
    dispatch: &VulkanPreparedDispatch,
    collection: OutputCollection,
    byte_capacity: usize,
    logical_extent: usize,
    logical_start: usize,
    logical_count: usize,
) -> Result<VulkanDistributedActivationRange, VulkanDistributedPlanError> {
    match collection {
        OutputCollection::Concatenated => proportional_activation_range(
            dispatch,
            byte_capacity,
            logical_extent,
            logical_start,
            logical_count,
            "output",
        ),
        OutputCollection::Reduced | OutputCollection::Routed | OutputCollection::Retained => {
            Ok(VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: byte_capacity,
            })
        }
        OutputCollection::Local => {
            unreachable!("unsupported output collection was rejected before planning")
        }
    }
}

fn proportional_activation_range(
    dispatch: &VulkanPreparedDispatch,
    byte_capacity: usize,
    logical_extent: usize,
    logical_start: usize,
    logical_count: usize,
    role: &str,
) -> Result<VulkanDistributedActivationRange, VulkanDistributedPlanError> {
    if !byte_capacity.is_multiple_of(logical_extent) {
        return Err(dispatch_error(
            dispatch,
            format!(
                "{role} activation capacity {byte_capacity} is not divisible by logical extent {logical_extent}"
            ),
        ));
    }
    let bytes_per_element = byte_capacity / logical_extent;
    Ok(VulkanDistributedActivationRange {
        byte_offset: logical_start.checked_mul(bytes_per_element).ok_or_else(|| {
            dispatch_error(dispatch, format!("{role} activation offset overflowed"))
        })?,
        byte_count: logical_count.checked_mul(bytes_per_element).ok_or_else(|| {
            dispatch_error(dispatch, format!("{role} activation size overflowed"))
        })?,
    })
}
