#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanHybridPreparedDispatchIdentity {
    logical_device_id: String,
    dispatch_index: usize,
    component_id: String,
    node_id: String,
}

pub(crate) struct VulkanHybridDispatchParameterRequirements {
    pub requirements_by_component: BTreeMap<String, Vec<VulkanHybridSharedRangeRequirement>>,
    prepared_parameter_tensors: BTreeSet<(String, String)>,
}

/// Reconstructs exact immutable parameter byte ranges for every mounted
/// component across all physical execution phases.
///
/// A dispatch that remains local contributes its canonical full tensors. A
/// dispatch replaced by an exact physical execution case contributes only its
/// replayed shard fragments. Taking the union across phases deliberately keeps
/// a canonical tensor whenever any mounted phase still needs it. Compiled
/// resource identities, rather than source tensor names, preserve aliases such
/// as tied embeddings and shared normalization allocations.
pub fn vulkan_runtime_hybrid_parameter_resources_by_component(
    runtime_model: &VulkanResidentRuntimeModel,
    prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
    execution_plans: &VulkanDistributedExecutionPlanSet,
    tensor_index: &TensorIndex,
    resource_contract: &CompiledResourceResidencyContract,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
) -> Result<BTreeMap<String, VulkanHybridCandidateResources>, VulkanHybridResourceError> {
    let VulkanHybridDispatchParameterRequirements {
        mut requirements_by_component,
        prepared_parameter_tensors,
    } = vulkan_hybrid_dispatch_parameter_requirements_by_component(
        prepared_plans,
        execution_plans,
        tensor_index,
        identity_by_logical_device,
        |prepared, parameter_id, actual_tensor| {
            let prepared_tensor = prepared
                .descriptors
                .iter()
                .find_map(|descriptor| match &descriptor.resource {
                    VulkanDescriptorResourceAddress::PermanentParameter {
                        param_id,
                        tensor,
                        ..
                    } if param_id == parameter_id => Some(tensor.as_str()),
                    _ => None,
                })
                .ok_or_else(|| {
                    VulkanHybridResourceError(format!(
                        "exact hybrid prepared parameter {}.{parameter_id} has no descriptor",
                        prepared.component_id,
                    ))
                })?;
            if actual_tensor == prepared_tensor {
                exact_vulkan_hybrid_fixed_resource_identity(
                    resource_contract,
                    &runtime_model.execution_scope,
                    &prepared.component_id,
                    Some(&prepared.node_id),
                    parameter_id,
                )
            } else {
                vulkan_hybrid_physical_tensor_resource_identity(tensor_index, actual_tensor)
            }
        },
    )?;

    // Transducers and other host-orchestrated components may retain parameters
    // without owning a Vulkan dispatch. Add only graph parameters absent from
    // the prepared descriptor set; dispatch-backed parameters were already
    // resolved above and may have been physically replaced by exact shards.
    for component in &runtime_model.circuit_graph.components {
        let logical_device_id = runtime_model
            .placement
            .device_for_component(&component.component_id);
        let physical_identity = identity_by_logical_device
            .get(logical_device_id)
            .ok_or_else(|| {
                VulkanHybridResourceError(format!(
                    "exact hybrid graph parameter has no physical identity for logical device {logical_device_id:?}",
                ))
            })?;
        for (parameter_id, tensor) in component
            .params
            .refs
            .iter()
            .filter_map(|(parameter_id, parameter)| {
                parameter
                    .tensor
                    .as_deref()
                    .map(|tensor| (parameter_id.as_str(), tensor))
            })
            .collect::<BTreeSet<_>>()
        {
            if prepared_parameter_tensors
                .contains(&(component.component_id.clone(), tensor.to_string()))
            {
                continue;
            }
            let tensor_byte_count = vulkan_hybrid_tensor_byte_count(tensor_index, tensor)?;
            requirements_by_component
                .entry(component.component_id.clone())
                .or_default()
                .push(VulkanHybridSharedRangeRequirement::device_parameter(
                    exact_vulkan_hybrid_fixed_resource_identity(
                        resource_contract,
                        &runtime_model.execution_scope,
                        &component.component_id,
                        None,
                        parameter_id,
                    )?,
                    physical_identity.clone(),
                    0,
                    tensor_byte_count,
                ));
        }
    }
    canonical_vulkan_hybrid_shared_range_resources(&requirements_by_component)
}

pub(crate) fn vulkan_hybrid_dispatch_parameter_requirements_by_component<F>(
    prepared_plans: &[(&str, &VulkanPreparedDispatchPlan)],
    execution_plans: &VulkanDistributedExecutionPlanSet,
    tensor_index: &TensorIndex,
    identity_by_logical_device: &BTreeMap<String, VulkanPlacementDeviceExecutionIdentity>,
    mut resource_identity: F,
) -> Result<VulkanHybridDispatchParameterRequirements, VulkanHybridResourceError>
where
    F: FnMut(&VulkanPreparedDispatch, &str, &str) -> Result<String, VulkanHybridResourceError>,
{
    let mut prepared_by_identity = BTreeMap::new();
    let mut prepared_parameter_tensors = BTreeSet::new();
    for (logical_device_id, prepared_plan) in prepared_plans {
        let physical_identity = identity_by_logical_device
            .get(*logical_device_id)
            .ok_or_else(|| {
                VulkanHybridResourceError(format!(
                    "exact hybrid parameter planning has no physical identity for logical device {logical_device_id:?}",
                ))
            })?;
        for dispatch in &prepared_plan.dispatches {
            prepared_parameter_tensors.extend(dispatch.descriptors.iter().filter_map(
                |descriptor| match &descriptor.resource {
                    VulkanDescriptorResourceAddress::PermanentParameter { tensor, .. } => {
                        Some((dispatch.component_id.clone(), tensor.clone()))
                    }
                    _ => None,
                },
            ));
            let identity = VulkanHybridPreparedDispatchIdentity {
                logical_device_id: (*logical_device_id).to_string(),
                dispatch_index: dispatch.dispatch_index,
                component_id: dispatch.component_id.clone(),
                node_id: dispatch.node_id.clone(),
            };
            if prepared_by_identity
                .insert(identity, (physical_identity, dispatch))
                .is_some()
            {
                return Err(VulkanHybridResourceError(
                    "exact hybrid parameter planning found a duplicate prepared dispatch identity"
                        .to_string(),
                ));
            }
        }
    }
    if prepared_by_identity.is_empty() || execution_plans.all().is_empty() {
        return Err(VulkanHybridResourceError(
            "exact hybrid parameter planning requires prepared dispatches and physical phase plans"
                .to_string(),
        ));
    }

    let mut requirements_by_component =
        BTreeMap::<String, Vec<VulkanHybridSharedRangeRequirement>>::new();
    for execution_plan in execution_plans.all() {
        let mut distributed_by_identity = BTreeMap::new();
        for dispatch in &execution_plan.dispatches {
            let identity = VulkanHybridPreparedDispatchIdentity {
                logical_device_id: dispatch.owner_device_id.clone(),
                dispatch_index: dispatch.dispatch_index,
                component_id: dispatch.component_id.clone(),
                node_id: dispatch.node_id.clone(),
            };
            if !prepared_by_identity.contains_key(&identity) {
                return Err(VulkanHybridResourceError(format!(
                    "exact hybrid parameter plan replaces unknown prepared dispatch {}.{} at index {} on {:?}",
                    dispatch.component_id,
                    dispatch.node_id,
                    dispatch.dispatch_index,
                    dispatch.owner_device_id,
                )));
            }
            if distributed_by_identity.insert(identity, dispatch).is_some() {
                return Err(VulkanHybridResourceError(
                    "exact hybrid parameter plan repeats a distributed dispatch identity"
                        .to_string(),
                ));
            }
        }

        for (identity, (owner_physical_identity, prepared)) in &prepared_by_identity {
            if let Some(distributed) = distributed_by_identity.get(identity) {
                for shard in &distributed.shards {
                    let physical_identity = identity_by_logical_device
                        .get(&shard.device_id)
                        .ok_or_else(|| {
                            VulkanHybridResourceError(format!(
                                "exact hybrid parameter shard has no physical identity for logical device {:?}",
                                shard.device_id,
                            ))
                        })?;
                    for fragment in &shard.parameters {
                        let descriptor = prepared
                            .descriptors
                            .iter()
                            .find(|descriptor| descriptor.binding == fragment.binding)
                            .ok_or_else(|| {
                                VulkanHybridResourceError(format!(
                                    "exact hybrid parameter fragment uses absent binding {} on {}.{}",
                                    fragment.binding, prepared.component_id, prepared.node_id,
                                ))
                            })?;
                        let VulkanDescriptorResourceAddress::PermanentParameter {
                            param_id,
                            tensor,
                            ..
                        } = &descriptor.resource
                        else {
                            return Err(VulkanHybridResourceError(format!(
                                "exact hybrid parameter fragment binding {} on {}.{} is not a permanent parameter",
                                fragment.binding, prepared.component_id, prepared.node_id,
                            )));
                        };
                        if tensor != &fragment.tensor {
                            return Err(VulkanHybridResourceError(format!(
                                "exact hybrid parameter fragment tensor {:?} differs from prepared binding tensor {tensor:?}",
                                fragment.tensor,
                            )));
                        }
                        validate_vulkan_hybrid_parameter_range(
                            tensor_index,
                            &fragment.tensor,
                            fragment.byte_offset,
                            fragment.byte_count,
                        )?;
                        requirements_by_component
                            .entry(distributed.component_id.clone())
                            .or_default()
                            .push(VulkanHybridSharedRangeRequirement::device_parameter(
                                resource_identity(prepared, param_id, &fragment.tensor)?,
                                physical_identity.clone(),
                                fragment.byte_offset,
                                fragment.byte_count,
                            ));
                    }
                }
                continue;
            }

            for descriptor in &prepared.descriptors {
                let VulkanDescriptorResourceAddress::PermanentParameter {
                    param_id,
                    tensor,
                    byte_count,
                } = &descriptor.resource
                else {
                    continue;
                };
                let tensor_byte_count = vulkan_hybrid_tensor_byte_count(tensor_index, tensor)?;
                if byte_count.is_some_and(|declared| declared != tensor_byte_count) {
                    return Err(VulkanHybridResourceError(format!(
                        "exact hybrid local parameter {tensor:?} declares {byte_count:?} bytes but the tensor index requires {tensor_byte_count}",
                    )));
                }
                requirements_by_component
                    .entry(prepared.component_id.clone())
                    .or_default()
                    .push(VulkanHybridSharedRangeRequirement::device_parameter(
                        resource_identity(prepared, param_id, tensor)?,
                        (*owner_physical_identity).clone(),
                        0,
                        tensor_byte_count,
                    ));
            }
        }
    }
    Ok(VulkanHybridDispatchParameterRequirements {
        requirements_by_component,
        prepared_parameter_tensors,
    })
}

fn exact_vulkan_hybrid_fixed_resource_identity(
    resource_contract: &CompiledResourceResidencyContract,
    execution_scope: &str,
    component_id: &str,
    node_id: Option<&str>,
    parameter_id: &str,
) -> Result<String, VulkanHybridResourceError> {
    let identities = resource_contract
        .bindings
        .iter()
        .filter(|binding| {
            binding.execution_scope == execution_scope
                && binding.component_id == component_id
                && binding.parameter_id == parameter_id
                && node_id.is_none_or(|node_id| binding.node_id == node_id)
        })
        .map(|binding| match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup { resource_id, .. } => {
                Ok(resource_id.clone())
            }
            CompiledResourceBindingMapping::SelectedAtomicGroup { .. }
            | CompiledResourceBindingMapping::PartitionTemplateMember { .. } => {
                Err(VulkanHybridResourceError(format!(
                    "exact fixed parameter {}.{parameter_id} resolves to a selected resource mapping",
                    component_id,
                )))
            }
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if identities.len() != 1 {
        return Err(VulkanHybridResourceError(format!(
            "exact fixed parameter {}.{parameter_id} resolves to {} compiled resource identities",
            component_id,
            identities.len(),
        )));
    }
    Ok(identities
        .into_iter()
        .next()
        .expect("one exact resource identity was proved"))
}

fn vulkan_hybrid_tensor_byte_count(
    tensor_index: &TensorIndex,
    tensor: &str,
) -> Result<usize, VulkanHybridResourceError> {
    let byte_count = tensor_index
        .tensors
        .get(tensor)
        .and_then(|metadata| metadata.byte_count)
        .ok_or_else(|| {
            VulkanHybridResourceError(format!(
                "exact hybrid parameter range references tensor {tensor:?} without a byte count",
            ))
        })?;
    if byte_count == 0 {
        return Err(VulkanHybridResourceError(format!(
            "exact hybrid parameter tensor {tensor:?} is empty",
        )));
    }
    Ok(byte_count)
}

fn vulkan_hybrid_physical_tensor_resource_identity(
    tensor_index: &TensorIndex,
    tensor: &str,
) -> Result<String, VulkanHybridResourceError> {
    let metadata = tensor_index.tensors.get(tensor).ok_or_else(|| {
        VulkanHybridResourceError(format!(
            "exact hybrid physical tensor {tensor:?} has no tensor metadata",
        ))
    })?;
    let digest = metadata
        .immutable_content_identity(tensor)
        .map_err(|error| {
            VulkanHybridResourceError(format!(
                "exact hybrid physical tensor identity is invalid: {error}",
            ))
        })?;
    let source_file = metadata.source_file.as_deref().ok_or_else(|| {
        VulkanHybridResourceError(format!(
            "exact hybrid physical tensor {tensor:?} has no source file identity",
        ))
    })?;
    let offsets = metadata.data_offsets.as_deref().ok_or_else(|| {
        VulkanHybridResourceError(format!(
            "exact hybrid physical tensor {tensor:?} has no source byte range",
        ))
    })?;
    if offsets.len() != 2 || offsets[0] >= offsets[1] {
        return Err(VulkanHybridResourceError(format!(
            "exact hybrid physical tensor {tensor:?} has an invalid source byte range",
        )));
    }
    Ok(format!(
        "physical-tensor:{source_file}:{}:{}:{digest}",
        offsets[0], offsets[1],
    ))
}

fn validate_vulkan_hybrid_parameter_range(
    tensor_index: &TensorIndex,
    tensor: &str,
    byte_offset: usize,
    byte_count: usize,
) -> Result<(), VulkanHybridResourceError> {
    let tensor_byte_count = vulkan_hybrid_tensor_byte_count(tensor_index, tensor)?;
    let end = byte_offset.checked_add(byte_count).ok_or_else(|| {
        VulkanHybridResourceError(format!(
            "exact hybrid parameter range for {tensor:?} overflows",
        ))
    })?;
    if byte_count == 0 || end > tensor_byte_count {
        return Err(VulkanHybridResourceError(format!(
            "exact hybrid parameter range for {tensor:?} is empty or ends at {end} beyond {tensor_byte_count}",
        )));
    }
    Ok(())
}
