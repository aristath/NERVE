use nerve_execution_contracts::{
    ExecutionPhase, InputDistribution, OutputCollection, ParameterPartitionKind,
    PartitionOrigin, PhysicalExecutionContract, WorkgroupXMapping,
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

fn select_distributed_contract<'a>(
    dispatch: &'a VulkanPreparedDispatch,
    artifact: &crate::vulkan_stream_circuit::VulkanReusableKernelArtifact,
) -> Result<Option<&'a PhysicalExecutionContract>, VulkanDistributedPlanError> {
    let candidates = dispatch
        .physical_execution_contracts
        .iter()
        .filter(|contract| {
            contract.strategy.is_distributed()
                && contract.phases.contains(&ExecutionPhase::Decode)
                && contract.operation_family == dispatch.op
                && contract.member_node_ids.contains(&dispatch.node_id)
                && contract.artifacts.iter().any(|identity| {
                    identity.path == artifact.path && identity.entry_point == artifact.entry_point
                })
        })
        .collect::<Vec<_>>();
    let [contract] = candidates.as_slice() else {
        if candidates.is_empty() {
            return Ok(None);
        }
        return Err(dispatch_error(
            dispatch,
            format!(
                "has {} ambiguous decode distribution contracts for reusable artifact family {:?}",
                candidates.len(),
                artifact.family_id
            ),
        ));
    };
    contract
        .validate()
        .map_err(|error| dispatch_error(dispatch, format!("has an invalid contract: {error}")))?;
    Ok(Some(*contract))
}

#[allow(clippy::too_many_arguments)]
fn plan_contract_dispatch(
    owner_device_id: &str,
    dispatch: &VulkanPreparedDispatch,
    tensor_index: &TensorIndex,
    device_ids: &[String],
    edge_placements: &[ComponentEdgePlacement],
    artifact: &crate::vulkan_stream_circuit::VulkanReusableKernelArtifact,
    contract: &PhysicalExecutionContract,
    storage_buffer_offset_alignment: usize,
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
    validate_partition_origin(dispatch, launch)?;

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
    if output_contract.collection == OutputCollection::Reduced {
        return Err(dispatch_error(
            dispatch,
            "partial-output reduction is not implemented for this contract".to_string(),
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
    let (distribution, workgroup_elements) = match launch.workgroup_x {
        WorkgroupXMapping::Proportional => {
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
        WorkgroupXMapping::Repeated => (
            VulkanDistributedDispatchDistribution::ExpertRange,
            1,
        ),
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
            let output_range = contract_output_range(
                dispatch,
                output_contract.collection,
                output_activation.signal_byte_capacity,
                logical_extent,
                logical_start,
                logical_count,
            )?;
            let workgroup_count_x = match launch.workgroup_x {
                WorkgroupXMapping::Proportional => u32::try_from(
                    logical_count / workgroup_elements,
                )
                .map_err(|_| {
                    dispatch_error(dispatch, "shard workgroup count exceeds u32".to_string())
                })?,
                WorkgroupXMapping::Repeated => artifact.workgroup_count_x,
            };
            let base_workgroup_z = match launch.origin {
                PartitionOrigin::LocalZero => 0,
                PartitionOrigin::PushConstantU32 => u32::try_from(logical_start).map_err(|_| {
                    dispatch_error(dispatch, "partition origin exceeds u32".to_string())
                })?,
            };
            Ok(VulkanDistributedDispatchShard {
                device_id: device_id.to_string(),
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
        reusable_family_id: dispatch.reusable_family_id.clone(),
        physical_execution_contract_id: contract.contract_id.clone(),
        implementation_digest: contract.implementation_digest.clone(),
        contract_member_node_ids: contract.member_node_ids.clone(),
        has_lazy_resource_requirements: contract.resources.iter().any(|resource| {
            resource.kind == nerve_execution_contracts::ResourceKind::LazyResource
        }),
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
        auxiliary_input_activations: inputs
            .into_iter()
            .skip(1)
            .map(|(_, activation)| activation)
            .collect(),
        output_activation,
        distribution,
        distributed_parameter_byte_count,
        shards,
    }))
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
    let descriptor_inputs = descriptor_bindings(VulkanKernelDescriptorUsage::InputSignal);
    let descriptor_outputs = descriptor_bindings(VulkanKernelDescriptorUsage::OutputSignal);
    if contract_inputs != descriptor_inputs
        || contract_outputs != descriptor_outputs
        || contract_parameters != descriptor_parameters
    {
        return Err(dispatch_error(
            dispatch,
            format!(
                "contract bindings inputs={contract_inputs:?} outputs={contract_outputs:?} parameters={contract_parameters:?} disagree with artifact ABI inputs={descriptor_inputs:?} outputs={descriptor_outputs:?} parameters={descriptor_parameters:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_partition_origin(
    dispatch: &VulkanPreparedDispatch,
    launch: &nerve_execution_contracts::PartitionLaunch,
) -> Result<(), VulkanDistributedPlanError> {
    match launch.origin {
        PartitionOrigin::LocalZero if dispatch.push_constants.is_empty() => Ok(()),
        PartitionOrigin::PushConstantU32 => {
            let Some(name) = launch.origin_push_constant.as_deref() else {
                return Err(dispatch_error(
                    dispatch,
                    "partition-origin push constant is unnamed".to_string(),
                ));
            };
            if dispatch.push_constants.as_slice()
                == [VulkanKernelScalarBinding {
                    name: name.to_string(),
                    scalar_type: "u32".to_string(),
                    source: VulkanKernelScalarSource::PushConstant,
                }]
            {
                Ok(())
            } else {
                Err(dispatch_error(
                    dispatch,
                    format!(
                        "partition contract requires the sole u32 push constant {name:?}"
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
    contract: &PhysicalExecutionContract,
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
        let tensor = dispatch
            .descriptors
            .iter()
            .find(|descriptor| descriptor.binding == binding)
            .and_then(|descriptor| match &descriptor.resource {
                VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                    Some(tensor.as_str())
                }
                _ => None,
            })
            .ok_or_else(|| {
                dispatch_error(
                    dispatch,
                    format!(
                        "contract parameter binding {binding} is not a permanent parameter"
                    ),
                )
            })?;
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
        OutputCollection::Routed | OutputCollection::Retained => {
            Ok(VulkanDistributedActivationRange {
                byte_offset: 0,
                byte_count: byte_capacity,
            })
        }
        OutputCollection::Local | OutputCollection::Reduced => {
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
