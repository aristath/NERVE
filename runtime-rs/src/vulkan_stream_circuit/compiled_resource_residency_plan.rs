#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeParameterResidencyBytes {
    pub always_resident_bytes: usize,
    pub initial_dynamic_bytes: usize,
    pub current_resident_bytes: usize,
    pub maximum_addressable_bytes: usize,
    pub staging_headroom_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VulkanRuntimeResidencyGrowthAdmission {
    pub current_resident_parameter_bytes: usize,
    pub requested_growth_bytes: usize,
    pub projected_resident_parameter_bytes: usize,
    pub projected_device_resident_bytes: usize,
    pub safe_device_capacity_bytes: usize,
}

pub fn admit_vulkan_runtime_initial_residency_by_physical_device(
    plan: &VulkanRuntimeResidencyPlan,
    physical_device_by_logical_device: &BTreeMap<String, String>,
    safe_capacity_by_physical_device: &BTreeMap<String, usize>,
) -> Result<BTreeMap<String, usize>, VulkanRuntimeResidencyPlanError> {
    let mut admitted = BTreeMap::<String, usize>::new();
    for device_plan in &plan.device_plans {
        let physical_device_id = physical_device_by_logical_device
            .get(&device_plan.device_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "runtime residency device {:?} has no physical-device binding",
                    device_plan.device_id
                ))
            })?;
        let total = admitted.entry(physical_device_id.clone()).or_default();
        *total = checked_residency_add(
            *total,
            device_plan.initial_device_resident_bytes,
            "physical initial device residency",
        )?;
    }
    for (physical_device_id, initial_bytes) in &admitted {
        let safe_capacity = safe_capacity_by_physical_device
            .get(physical_device_id)
            .copied()
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "physical device {physical_device_id:?} has no stable capacity budget"
                ))
            })?;
        if *initial_bytes > safe_capacity {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "physical device {physical_device_id:?} needs {initial_bytes} initial device bytes but its stable safe capacity is {safe_capacity}"
            )));
        }
    }
    Ok(admitted)
}

pub fn admit_vulkan_runtime_residency_growth(
    device_plan: &VulkanRuntimeDeviceResidencyPlan,
    current_resident_parameter_bytes: usize,
    requested_atomic_group_bytes: usize,
    safe_device_capacity_bytes: usize,
) -> Result<
    VulkanRuntimeResidencyGrowthAdmission,
    VulkanRuntimeResidencyPlanError,
> {
    let parameters = &device_plan.parameter_residency;
    if current_resident_parameter_bytes < parameters.current_resident_bytes
        || current_resident_parameter_bytes
            > parameters.maximum_addressable_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "current parameter residency is outside the planned range"
                .to_string(),
        ));
    }
    if requested_atomic_group_bytes == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "residency growth request must contain a missing atomic group"
                .to_string(),
        ));
    }
    if requested_atomic_group_bytes > parameters.staging_headroom_bytes {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "atomic residency growth of {requested_atomic_group_bytes} bytes exceeds \
             planned staging headroom {}",
            parameters.staging_headroom_bytes
        )));
    }
    let projected_resident_parameter_bytes = checked_residency_add(
        current_resident_parameter_bytes,
        requested_atomic_group_bytes,
        "projected parameter residency",
    )?;
    if projected_resident_parameter_bytes
        > parameters.maximum_addressable_bytes
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "atomic residency growth would exceed the maximum addressable \
             parameter bytes {}",
            parameters.maximum_addressable_bytes
        )));
    }
    let fixed_residency_bytes = [
        parameters.staging_headroom_bytes,
        device_plan.working_set.transient_state_bytes,
        device_plan.working_set.activation_headroom_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        checked_residency_add(total, bytes, "fixed residency headroom")
    })?;
    let projected_device_resident_bytes = checked_residency_add(
        projected_resident_parameter_bytes,
        fixed_residency_bytes,
        "projected device residency",
    )?;
    if projected_device_resident_bytes > safe_device_capacity_bytes {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "atomic residency growth requires {projected_device_resident_bytes} \
             device bytes but the safe capacity is {safe_device_capacity_bytes}"
        )));
    }
    Ok(VulkanRuntimeResidencyGrowthAdmission {
        current_resident_parameter_bytes,
        requested_growth_bytes: requested_atomic_group_bytes,
        projected_resident_parameter_bytes,
        projected_device_resident_bytes,
        safe_device_capacity_bytes,
    })
}

#[derive(Default)]
struct DeviceResourceSelection {
    concrete_always: BTreeSet<String>,
    concrete_dynamic: BTreeSet<String>,
    partition_templates: BTreeSet<String>,
}

fn plan_compiled_parameter_residency(
    runtime_model: &VulkanResidentRuntimeModel,
    contract: &CompiledResourceResidencyContract,
    input_device_id: &str,
    output_device_id: &str,
    owner_device_ids: &[String],
    mount_speculative_decoders: bool,
    policy: ResourceResidencyPolicy,
) -> Result<
    BTreeMap<String, VulkanRuntimeParameterResidencyBytes>,
    VulkanRuntimeResidencyPlanError,
> {
    if !contract
        .supported_policies
        .contains(&policy.required_compiled_loading_policy())
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled package does not support {policy:?} residency"
        )));
    }

    let contract_index = CompiledResourceContractIndex::new(contract)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let mut selected = owner_device_ids
        .iter()
        .map(|device_id| (device_id.clone(), DeviceResourceSelection::default()))
        .collect::<BTreeMap<_, _>>();

    for binding in &contract.bindings {
        let Some(device_id) = resource_binding_device(
            runtime_model,
            binding,
            input_device_id,
            output_device_id,
            mount_speculative_decoders,
        )?
        else {
            continue;
        };
        let device = selected.get_mut(device_id).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "resource binding resolved to non-owner device {device_id:?}"
            ))
        })?;
        accumulate_compiled_resource_binding(device, binding, contract, &contract_index)?;
    }

    selected
        .into_iter()
        .map(|(device_id, selection)| {
            compiled_parameter_residency_bytes(contract, &contract_index, &selection, policy)
                .map(|bytes| (device_id, bytes))
        })
        .collect()
}

fn plan_compiled_parameter_residency_for_device_set(
    runtime_model: &VulkanResidentRuntimeModel,
    contract: &CompiledResourceResidencyContract,
    input_device_id: &str,
    output_device_id: &str,
    device_ids: &BTreeSet<String>,
    mount_speculative_decoders: bool,
    policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeParameterResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let contract_index = CompiledResourceContractIndex::new(contract)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let mut selected = DeviceResourceSelection::default();
    for binding in &contract.bindings {
        let Some(device_id) = resource_binding_device(
            runtime_model,
            binding,
            input_device_id,
            output_device_id,
            mount_speculative_decoders,
        )?
        else {
            continue;
        };
        if !device_ids.contains(device_id) {
            continue;
        }
        accumulate_compiled_resource_binding(
            &mut selected,
            binding,
            contract,
            &contract_index,
        )?;
    }
    compiled_parameter_residency_bytes(contract, &contract_index, &selected, policy)
}

fn compiled_resource_selector_ids_for_device_set(
    runtime_model: &VulkanResidentRuntimeModel,
    contract: &CompiledResourceResidencyContract,
    input_device_id: &str,
    output_device_id: &str,
    device_ids: &BTreeSet<String>,
    mount_speculative_decoders: bool,
) -> Result<BTreeSet<String>, VulkanRuntimeResidencyPlanError> {
    let mut selector_ids = BTreeSet::new();
    for selector in &contract.selectors {
        let Some(device_id) = resource_component_device(
            runtime_model,
            &selector.execution_scope,
            &selector.component_id,
            input_device_id,
            output_device_id,
            mount_speculative_decoders,
        )?
        else {
            continue;
        };
        if device_ids.contains(device_id) {
            selector_ids.insert(selector.id.clone());
        }
    }
    Ok(selector_ids)
}

fn accumulate_compiled_resource_binding(
    selection: &mut DeviceResourceSelection,
    binding: &CompiledResourceBinding,
    contract: &CompiledResourceResidencyContract,
    contract_index: &CompiledResourceContractIndex,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    match &binding.mapping {
        CompiledResourceBindingMapping::AtomicGroup {
            atomic_group_id,
            resource_id,
        } => {
            let group = contract_index.atomic_group(contract, atomic_group_id).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "resource binding references unknown group {atomic_group_id:?}"
                ))
            })?;
            if group.lifetime == CompiledResourceLifetime::Dynamic {
                selection
                    .concrete_dynamic
                    .extend(group.resource_ids.iter().cloned());
            } else {
                selection.concrete_always.insert(resource_id.clone());
            }
        }
        CompiledResourceBindingMapping::SelectedAtomicGroup {
            atomic_group_id,
            ..
        } => {
            let group = contract_index.atomic_group(contract, atomic_group_id).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "selected resource binding references unknown group {atomic_group_id:?}"
                ))
            })?;
            selection
                .concrete_dynamic
                .extend(group.resource_ids.iter().cloned());
        }
        CompiledResourceBindingMapping::PartitionTemplateMember {
            partition_template_id,
            ..
        } => {
            selection
                .partition_templates
                .insert(partition_template_id.clone());
        }
    }
    Ok(())
}

fn compiled_parameter_residency_bytes(
    contract: &CompiledResourceResidencyContract,
    contract_index: &CompiledResourceContractIndex,
    selection: &DeviceResourceSelection,
    policy: ResourceResidencyPolicy,
) -> Result<VulkanRuntimeParameterResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let mut always_resident_bytes = 0usize;
    for resource_id in &selection.concrete_always {
        let resource = contract_index.resource(contract, resource_id).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "residency plan references unknown resource {resource_id:?}"
            ))
        })?;
        always_resident_bytes = checked_residency_add(
            always_resident_bytes,
            compiled_resource_bytes(resource)?,
            "always-resident parameter bytes",
        )?;
    }

    let mut maximum_dynamic_bytes = 0usize;
    let mut staging_headroom_bytes = 0usize;
    let mut staged_concrete_groups = BTreeSet::new();
    for resource_id in &selection.concrete_dynamic {
        let resource = contract_index.resource(contract, resource_id).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "residency plan references unknown dynamic resource {resource_id:?}"
            ))
        })?;
        maximum_dynamic_bytes = checked_residency_add(
            maximum_dynamic_bytes,
            compiled_resource_bytes(resource)?,
            "maximum dynamic parameter bytes",
        )?;
        for group_index in contract_index.atomic_group_indices_for_resource(resource_id) {
            let group = &contract.atomic_groups[*group_index];
            if group.lifetime == CompiledResourceLifetime::Dynamic {
                staged_concrete_groups.insert(group.id.as_str());
            }
        }
    }
    for group_id in staged_concrete_groups {
        let group = contract_index
            .atomic_group(contract, group_id)
            .expect("selected group was indexed above");
        let group_bytes =
            group
                .resource_ids
                .iter()
                .try_fold(0usize, |total, resource_id| {
                    let resource = contract_index
                        .resource(contract, resource_id)
                        .ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(format!(
                                "dynamic group references unknown resource {resource_id:?}"
                            ))
                        })?;
                    checked_residency_add(
                        total,
                        compiled_resource_bytes(resource)?,
                        "dynamic atomic group bytes",
                    )
                })?;
        staging_headroom_bytes = staging_headroom_bytes.max(group_bytes);
    }

    let mut dynamic_members = BTreeMap::<String, (usize, usize)>::new();
    for template_id in &selection.partition_templates {
        let template = contract_index.partition_template(contract, template_id).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "residency plan references unknown partition template {template_id:?}"
            ))
        })?;
        let mut group_bytes = 0usize;
        for member in &template.member_templates {
            let member_bytes =
                member
                    .range_templates
                    .iter()
                    .try_fold(0usize, |total, range| {
                        checked_residency_add(
                            total,
                            range.byte_count,
                            "partition member bytes",
                        )
                    })?;
            group_bytes = checked_residency_add(
                group_bytes,
                member_bytes,
                "partition atomic group bytes",
            )?;
            match dynamic_members.entry(member.resource_identity_seed.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((template.partition_count, member_bytes));
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if *entry.get() != (template.partition_count, member_bytes) =>
                {
                    return Err(VulkanRuntimeResidencyPlanError(
                        "shared partition resource has conflicting byte accounting"
                            .to_string(),
                    ));
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
        staging_headroom_bytes = staging_headroom_bytes.max(group_bytes);
    }
    for (partition_count, member_bytes) in dynamic_members.into_values() {
        maximum_dynamic_bytes = checked_residency_add(
            maximum_dynamic_bytes,
            checked_residency_mul(
                partition_count,
                member_bytes,
                "maximum partition resource bytes",
            )?,
            "maximum dynamic parameter bytes",
        )?;
    }

    let initial_dynamic_bytes = match policy {
        ResourceResidencyPolicy::DemandPaged | ResourceResidencyPolicy::DemandRetained => 0,
        ResourceResidencyPolicy::Eager => maximum_dynamic_bytes,
    };
    let current_resident_bytes = checked_residency_add(
        always_resident_bytes,
        initial_dynamic_bytes,
        "initial current parameter bytes",
    )?;
    let maximum_addressable_bytes = checked_residency_add(
        always_resident_bytes,
        maximum_dynamic_bytes,
        "maximum addressable parameter bytes",
    )?;
    Ok(VulkanRuntimeParameterResidencyBytes {
        always_resident_bytes,
        initial_dynamic_bytes,
        current_resident_bytes,
        maximum_addressable_bytes,
        staging_headroom_bytes,
    })
}

fn resource_binding_device<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    binding: &CompiledResourceBinding,
    input_device_id: &'a str,
    output_device_id: &'a str,
    mount_speculative_decoders: bool,
) -> Result<Option<&'a str>, VulkanRuntimeResidencyPlanError> {
    resource_component_device(
        runtime_model,
        &binding.execution_scope,
        &binding.component_id,
        input_device_id,
        output_device_id,
        mount_speculative_decoders,
    )
}

fn resource_component_device<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    execution_scope: &str,
    component_id: &str,
    input_device_id: &'a str,
    output_device_id: &'a str,
    mount_speculative_decoders: bool,
) -> Result<Option<&'a str>, VulkanRuntimeResidencyPlanError> {
    if execution_scope == "target" {
        let component = runtime_model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == component_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "resource binding references unknown target component {:?}",
                    component_id
                ))
            })?;
        return Ok(Some(match component.runtime_role {
            CircuitRuntimeRole::SignalProcessor => runtime_model
                .placement
                .device_for_component(&component.component_id),
            CircuitRuntimeRole::InputTransducer => input_device_id,
            CircuitRuntimeRole::OutputTransducer | CircuitRuntimeRole::Sampler => {
                output_device_id
            }
            CircuitRuntimeRole::DraftProcessor
            | CircuitRuntimeRole::DraftInputAdapter
            | CircuitRuntimeRole::DraftOutputTransducer => {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "target resource binding uses draft component {:?}",
                    component.component_id
                )));
            }
        }));
    }

    let decoder_id =
        execution_scope
            .strip_prefix("draft:")
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "resource binding has unsupported execution scope {:?}",
                    execution_scope
                ))
            })?;
    let decoder = runtime_model
        .package
        .speculative_decoders
        .iter()
        .find(|decoder| decoder.id == decoder_id)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "resource binding references unknown decoder {decoder_id:?}"
            ))
        })?;
    if !decoder
        .circuit_graph
        .components
        .iter()
        .any(|component| component.component_id == component_id)
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "resource binding references unknown draft component {:?}",
            component_id
        )));
    }
    Ok(mount_speculative_decoders.then_some(output_device_id))
}

fn compiled_resource_bytes(
    resource: &CompiledImmutableResource,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    resource.ranges.iter().try_fold(0usize, |total, range| {
        checked_residency_add(
            total,
            range.byte_count,
            "compiled resource byte count",
        )
    })
}
