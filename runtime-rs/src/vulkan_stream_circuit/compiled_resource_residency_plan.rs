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
    input_device_id: &str,
    output_device_id: &str,
    owner_device_ids: &[String],
    mount_speculative_decoders: bool,
    policy: ResourceResidencyPolicy,
) -> Result<
    BTreeMap<String, VulkanRuntimeParameterResidencyBytes>,
    VulkanRuntimeResidencyPlanError,
> {
    let contract = &runtime_model.package.resource_residency;
    if !contract.supported_policies.contains(&policy) {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled package does not support {policy:?} residency"
        )));
    }

    let resources = contract
        .resources
        .iter()
        .map(|resource| (resource.id.as_str(), resource))
        .collect::<BTreeMap<_, _>>();
    let groups = contract
        .atomic_groups
        .iter()
        .map(|group| (group.id.as_str(), group))
        .collect::<BTreeMap<_, _>>();
    let templates = contract
        .partition_templates
        .iter()
        .map(|template| (template.id.as_str(), template))
        .collect::<BTreeMap<_, _>>();
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
        match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup {
                atomic_group_id,
                resource_id,
            } => {
                let group = groups.get(atomic_group_id.as_str()).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "resource binding references unknown group {atomic_group_id:?}"
                    ))
                })?;
                if group.lifetime == CompiledResourceLifetime::Dynamic {
                    device
                        .concrete_dynamic
                        .extend(group.resource_ids.iter().cloned());
                } else {
                    device.concrete_always.insert(resource_id.clone());
                }
            }
            CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id,
                ..
            } => {
                device
                    .partition_templates
                    .insert(partition_template_id.clone());
            }
        }
    }

    selected
        .into_iter()
        .map(|(device_id, selection)| {
            let mut always_resident_bytes = 0usize;
            for resource_id in &selection.concrete_always {
                let resource = resources.get(resource_id.as_str()).ok_or_else(|| {
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
                let resource = resources.get(resource_id.as_str()).ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "residency plan references unknown dynamic resource {resource_id:?}"
                    ))
                })?;
                maximum_dynamic_bytes = checked_residency_add(
                    maximum_dynamic_bytes,
                    compiled_resource_bytes(resource)?,
                    "maximum dynamic parameter bytes",
                )?;
                for group in contract.atomic_groups.iter().filter(|group| {
                    group.lifetime == CompiledResourceLifetime::Dynamic
                        && group.resource_ids.contains(resource_id)
                }) {
                    staged_concrete_groups.insert(group.id.as_str());
                }
            }
            for group_id in staged_concrete_groups {
                let group = groups
                    .get(group_id)
                    .expect("selected group was indexed above");
                let group_bytes = group.resource_ids.iter().try_fold(
                    0usize,
                    |total, resource_id| {
                        let resource =
                            resources.get(resource_id.as_str()).ok_or_else(|| {
                                VulkanRuntimeResidencyPlanError(format!(
                                    "dynamic group references unknown resource {resource_id:?}"
                                ))
                            })?;
                        checked_residency_add(
                            total,
                            compiled_resource_bytes(resource)?,
                            "dynamic atomic group bytes",
                        )
                    },
                )?;
                staging_headroom_bytes =
                    staging_headroom_bytes.max(group_bytes);
            }

            let mut dynamic_members =
                BTreeMap::<String, (usize, usize)>::new();
            for template_id in &selection.partition_templates {
                let template =
                    templates.get(template_id.as_str()).ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "residency plan references unknown partition template {template_id:?}"
                        ))
                    })?;
                let mut group_bytes = 0usize;
                for member in &template.member_templates {
                    let member_bytes = member.range_templates.iter().try_fold(
                        0usize,
                        |total, range| {
                            checked_residency_add(
                                total,
                                range.byte_count,
                                "partition member bytes",
                            )
                        },
                    )?;
                    group_bytes = checked_residency_add(
                        group_bytes,
                        member_bytes,
                        "partition atomic group bytes",
                    )?;
                    match dynamic_members
                        .entry(member.resource_identity_seed.clone())
                    {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert((template.partition_count, member_bytes));
                        }
                        std::collections::btree_map::Entry::Occupied(entry)
                            if *entry.get()
                                != (template.partition_count, member_bytes) =>
                        {
                            return Err(VulkanRuntimeResidencyPlanError(
                                "shared partition resource has conflicting byte accounting"
                                    .to_string(),
                            ));
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {}
                    }
                }
                staging_headroom_bytes =
                    staging_headroom_bytes.max(group_bytes);
            }
            for (partition_count, member_bytes) in
                dynamic_members.into_values()
            {
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
                ResourceResidencyPolicy::DemandRetained => 0,
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
            Ok((
                device_id,
                VulkanRuntimeParameterResidencyBytes {
                    always_resident_bytes,
                    initial_dynamic_bytes,
                    current_resident_bytes,
                    maximum_addressable_bytes,
                    staging_headroom_bytes,
                },
            ))
        })
        .collect()
}

fn resource_binding_device<'a>(
    runtime_model: &'a VulkanResidentRuntimeModel,
    binding: &CompiledResourceBinding,
    input_device_id: &'a str,
    output_device_id: &'a str,
    mount_speculative_decoders: bool,
) -> Result<Option<&'a str>, VulkanRuntimeResidencyPlanError> {
    if binding.execution_scope == "target" {
        let component = runtime_model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == binding.component_id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "resource binding references unknown target component {:?}",
                    binding.component_id
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
        binding
            .execution_scope
            .strip_prefix("draft:")
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "resource binding has unsupported execution scope {:?}",
                    binding.execution_scope
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
        .any(|component| component.component_id == binding.component_id)
    {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "resource binding references unknown draft component {:?}",
            binding.component_id
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
