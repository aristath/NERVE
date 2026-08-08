#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeParameterResidentDerivation {
    node_id: String,
    parameter_id: String,
    derivation: CompiledResourceResidentDerivation,
}

fn apply_runtime_component_resident_derivations(
    runtime_model: &mut VulkanResidentRuntimeModel,
    package_root: &Path,
    source_component_id: &str,
    target_execution: &VulkanResidentComponentExecutionSpec,
    requests: &[VulkanRuntimeParameterResidentDerivation],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let request_keys = requests
        .iter()
        .map(|request| (request.node_id.as_str(), request.parameter_id.as_str()))
        .collect::<Vec<_>>();
    if request_keys.is_empty()
        || request_keys
            .iter()
            .any(|(node_id, parameter_id)| node_id.is_empty() || parameter_id.is_empty())
        || request_keys.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return runtime_resident_derivation_error(
            "runtime resident derivations must be non-empty, sorted, and unique by node and parameter",
        );
    }
    validate_runtime_mxfp4_derivation_targets(
        runtime_model,
        source_component_id,
        target_execution,
        requests,
    )?;

    let source = &runtime_model.package.resource_residency;
    let bindings = source
        .bindings
        .iter()
        .map(|binding| {
            (
                (
                    binding.component_id.as_str(),
                    binding.node_id.as_str(),
                    binding.parameter_id.as_str(),
                ),
                binding,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let resources = source
        .resources
        .iter()
        .map(|resource| (resource.id.as_str(), resource))
        .collect::<BTreeMap<_, _>>();
    let mut derivations_by_resource = BTreeMap::<String, CompiledResourceResidentDerivation>::new();
    for request in requests {
        let binding = bindings
            .get(&(
                source_component_id,
                request.node_id.as_str(),
                request.parameter_id.as_str(),
            ))
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime resident derivation cannot find source binding {source_component_id:?}.{:?}.{:?}",
                    request.node_id, request.parameter_id,
                ))
            })?;
        let resource_id = match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup { resource_id, .. }
            | CompiledResourceBindingMapping::SelectedAtomicGroup { resource_id, .. } => {
                resource_id
            }
            CompiledResourceBindingMapping::PartitionTemplateMember { .. } => {
                return runtime_resident_derivation_error(format!(
                    "runtime resident derivation for {source_component_id:?}.{:?}.{:?} targets a partition template instead of an independently addressable resource",
                    request.node_id, request.parameter_id,
                ));
            }
        };
        let resource = resources.get(resource_id.as_str()).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime resident derivation references unknown resource {resource_id:?}",
            ))
        })?;
        let source_byte_count = resource.ranges.iter().try_fold(0usize, |total, range| {
            total.checked_add(range.byte_count).ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "runtime resident derivation source size overflowed",
                )
            })
        })?;
        request
            .derivation
            .validate_for_source_byte_count(source_byte_count)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        match derivations_by_resource.entry(resource_id.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(request.derivation.clone());
            }
            std::collections::btree_map::Entry::Occupied(entry)
                if entry.get() == &request.derivation => {}
            std::collections::btree_map::Entry::Occupied(_) => {
                return runtime_resident_derivation_error(format!(
                    "runtime resource {resource_id:?} received conflicting resident derivations",
                ));
            }
        }
    }

    for binding in &source.bindings {
        let resource_id = match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup { resource_id, .. }
            | CompiledResourceBindingMapping::SelectedAtomicGroup { resource_id, .. } => {
                Some(resource_id)
            }
            CompiledResourceBindingMapping::PartitionTemplateMember { .. } => None,
        };
        if resource_id.is_some_and(|id| derivations_by_resource.contains_key(id))
            && binding.component_id != source_component_id
        {
            return runtime_resident_derivation_error(format!(
                "runtime component {source_component_id:?} cannot change a resource shared with component {:?}",
                binding.component_id,
            ));
        }
    }

    let (rewritten_resources, resource_ids) = rewrite_resident_derivation_resources(
        &source.resources,
        &derivations_by_resource,
    )?;
    let (rewritten_groups, group_ids) = rewrite_resident_derivation_groups(
        &source.atomic_groups,
        &resource_ids,
    )?;
    let (rewritten_templates, template_ids) = rewrite_resident_derivation_templates(
        &source.partition_templates,
        &group_ids,
    )?;
    let rewritten_bindings = source
        .bindings
        .iter()
        .cloned()
        .map(|mut binding| {
            rewrite_resident_derivation_binding(
                &mut binding,
                &resource_ids,
                &group_ids,
                &template_ids,
            );
            binding
        })
        .collect::<Vec<_>>();
    let (rewritten_selectors, selector_ids) = rewrite_resident_derivation_selectors(
        &source.selectors,
        &group_ids,
        &template_ids,
    )?;
    let rewritten_checkpoints = rewrite_resident_derivation_checkpoints(
        &source.checkpoints,
        &selector_ids,
    )?;
    let candidate = CompiledResourceResidencyContract {
        schema: source.schema.clone(),
        identity_algorithm: source.identity_algorithm.clone(),
        state_machine_schema: source.state_machine_schema.clone(),
        supported_policies: source.supported_policies.clone(),
        resources: rewritten_resources,
        atomic_groups: rewritten_groups,
        partition_templates: rewritten_templates,
        bindings: rewritten_bindings,
        selectors: rewritten_selectors,
        checkpoints: rewritten_checkpoints,
    };

    let original = std::mem::replace(
        &mut runtime_model.package.resource_residency,
        candidate,
    );
    if let Err(error) = package::validate_compiled_resource_residency(
        package_root,
        &runtime_model.package,
    ) {
        runtime_model.package.resource_residency = original;
        return runtime_resident_derivation_error(format!(
            "runtime resident derivation produced an invalid resource graph: {error}",
        ));
    }
    Ok(())
}

fn validate_runtime_mxfp4_derivation_targets(
    runtime_model: &VulkanResidentRuntimeModel,
    source_component_id: &str,
    target_execution: &VulkanResidentComponentExecutionSpec,
    requests: &[VulkanRuntimeParameterResidentDerivation],
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let component = runtime_model
        .package
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == source_component_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime resident derivation references unknown source component {source_component_id:?}",
            ))
        })?;
    let execution = runtime_model
        .package
        .component_executions
        .iter()
        .find(|execution| execution.component_id == source_component_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime resident derivation cannot find source execution {source_component_id:?}",
            ))
        })?;
    let requested_parameters_by_node = requests.iter().fold(
        BTreeMap::<&str, BTreeSet<&str>>::new(),
        |mut by_node, request| {
            by_node
                .entry(request.node_id.as_str())
                .or_default()
                .insert(request.parameter_id.as_str());
            by_node
        },
    );
    for request in requests {
        let node = component
            .circuit
            .nodes
            .iter()
            .find(|node| node.id == request.node_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime resident derivation references unknown node {:?}",
                    request.node_id,
                ))
            })?;
        let parameter_stride = match node.op.as_str() {
            "independent_sparse_moe_gate_up" => 4,
            "independent_sparse_moe_down" => 2,
            _ => {
                return runtime_resident_derivation_error(format!(
                    "runtime resident derivation node {:?} is not an independently addressable MXFP4 expert projection",
                    node.id,
                ));
            }
        };
        let accesses = node
            .attrs
            .get("selected_parameter_accesses")
            .and_then(Value::as_array)
            .filter(|accesses| accesses.len() == 1)
            .and_then(|accesses| accesses[0].get("mapping"))
            .and_then(Value::as_array)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime resident derivation node {:?} has no selector-ordered expert mapping",
                    node.id,
                ))
            })?;
        let weight_parameters = accesses
            .iter()
            .flat_map(|mapping| {
                mapping
                    .get("parameter_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                    .filter_map(|(index, value)| {
                        (index.is_multiple_of(2))
                            .then(|| value.as_str())
                            .flatten()
                    })
            })
            .collect::<BTreeSet<_>>();
        if accesses.iter().any(|mapping| {
            mapping
                .get("parameter_ids")
                .and_then(Value::as_array)
                .is_none_or(|parameters| parameters.len() != parameter_stride)
        }) || !weight_parameters.contains(request.parameter_id.as_str())
        {
            return runtime_resident_derivation_error(format!(
                "runtime resident derivation parameter {:?} is not an MXFP4 weight in node {:?}",
                request.parameter_id, node.id,
            ));
        }
        if requested_parameters_by_node[node.id.as_str()] != weight_parameters {
            return runtime_resident_derivation_error(format!(
                "runtime resident derivation must replace every MXFP4 weight consumed by node {:?} as one representation boundary",
                node.id,
            ));
        }
        let source_kernel = execution
            .kernels
            .iter()
            .find(|kernel| kernel.node_id == node.id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime resident derivation node {:?} has no source kernel",
                    node.id,
                ))
            })?;
        if source_kernel
            .resource_representation_dispatch
            .as_ref()
            .is_none_or(|contract| !contract.is_exact_mxfp4_source())
        {
            return runtime_resident_derivation_error(format!(
                "runtime resident derivation node {:?} is not backed by an explicit exact compact MXFP4 resource-representation contract",
                node.id,
            ));
        }
        let target_kernel = target_execution
            .kernels
            .iter()
            .find(|kernel| kernel.node_id == node.id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime resident derivation node {:?} has no target kernel",
                    node.id,
                ))
            })?;
        if target_kernel
            .resource_representation_dispatch
            .as_ref()
            .is_none_or(|contract| {
                !contract.selects_resident_derivation(request.derivation.kind)
            })
        {
            return runtime_resident_derivation_error(format!(
                "runtime resident derivation node {:?} does not declare address-tag selection of the matching resident representation",
                node.id,
            ));
        }
    }
    Ok(())
}

fn rewrite_resident_derivation_resources(
    resources: &[CompiledImmutableResource],
    derivations: &BTreeMap<String, CompiledResourceResidentDerivation>,
) -> Result<(Vec<CompiledImmutableResource>, BTreeMap<String, String>), VulkanResidentTokenModelPackageError> {
    let by_id = resources
        .iter()
        .map(|resource| (resource.id.clone(), resource.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut rewritten = BTreeMap::<String, CompiledImmutableResource>::new();
    let mut ids = BTreeMap::<String, String>::new();
    let mut visiting = BTreeSet::new();
    for resource_id in by_id.keys() {
        rewrite_resident_derivation_resource(
            resource_id,
            &by_id,
            derivations,
            &mut rewritten,
            &mut ids,
            &mut visiting,
        )?;
    }
    if ids.len() != resources.len() || rewritten.len() != resources.len() {
        return runtime_resident_derivation_error(
            "runtime resident derivation caused resource identity collisions",
        );
    }
    Ok((rewritten.into_values().collect(), ids))
}

fn rewrite_resident_derivation_resource(
    resource_id: &str,
    resources: &BTreeMap<String, CompiledImmutableResource>,
    derivations: &BTreeMap<String, CompiledResourceResidentDerivation>,
    rewritten: &mut BTreeMap<String, CompiledImmutableResource>,
    ids: &mut BTreeMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if let Some(id) = ids.get(resource_id) {
        return Ok(id.clone());
    }
    if !visiting.insert(resource_id.to_string()) {
        return runtime_resident_derivation_error(
            "runtime resident derivation encountered a resource dependency cycle",
        );
    }
    let mut resource = resources.get(resource_id).cloned().ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(format!(
            "runtime resident derivation references unknown resource dependency {resource_id:?}",
        ))
    })?;
    resource.dependencies = resource
        .dependencies
        .iter()
        .map(|dependency| {
            rewrite_resident_derivation_resource(
                dependency,
                resources,
                derivations,
                rewritten,
                ids,
                visiting,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    resource.dependencies.sort();
    if let Some(derivation) = derivations.get(resource_id) {
        if resource.resident_derivation.is_some() {
            return runtime_resident_derivation_error(format!(
                "runtime resource {resource_id:?} already has a resident derivation",
            ));
        }
        resource.resident_derivation = Some(derivation.clone());
        resource.compatibility.required_features.extend(
            derivation.required_features.iter().cloned(),
        );
        resource.compatibility.required_features.sort();
        resource.compatibility.required_features.dedup();
    }
    resource.id = package::compiled_resource_identity(&resource)
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
    visiting.remove(resource_id);
    let target_id = resource.id.clone();
    if rewritten.insert(target_id.clone(), resource).is_some() {
        return runtime_resident_derivation_error(
            "runtime resident derivation caused a resource identity collision",
        );
    }
    ids.insert(resource_id.to_string(), target_id.clone());
    Ok(target_id)
}

fn rewrite_resident_derivation_groups(
    groups: &[CompiledAtomicResidencyGroup],
    resource_ids: &BTreeMap<String, String>,
) -> Result<(Vec<CompiledAtomicResidencyGroup>, BTreeMap<String, String>), VulkanResidentTokenModelPackageError> {
    let by_id = groups
        .iter()
        .map(|group| (group.id.clone(), group.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut rewritten = BTreeMap::<String, CompiledAtomicResidencyGroup>::new();
    let mut ids = BTreeMap::<String, String>::new();
    let mut visiting = BTreeSet::new();
    for group_id in by_id.keys() {
        rewrite_resident_derivation_group(
            group_id,
            &by_id,
            resource_ids,
            &mut rewritten,
            &mut ids,
            &mut visiting,
        )?;
    }
    if ids.len() != groups.len() || rewritten.len() != groups.len() {
        return runtime_resident_derivation_error(
            "runtime resident derivation caused atomic-group identity collisions",
        );
    }
    Ok((rewritten.into_values().collect(), ids))
}

fn rewrite_resident_derivation_group(
    group_id: &str,
    groups: &BTreeMap<String, CompiledAtomicResidencyGroup>,
    resource_ids: &BTreeMap<String, String>,
    rewritten: &mut BTreeMap<String, CompiledAtomicResidencyGroup>,
    ids: &mut BTreeMap<String, String>,
    visiting: &mut BTreeSet<String>,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if let Some(id) = ids.get(group_id) {
        return Ok(id.clone());
    }
    if !visiting.insert(group_id.to_string()) {
        return runtime_resident_derivation_error(
            "runtime resident derivation encountered an atomic-group dependency cycle",
        );
    }
    let mut group = groups.get(group_id).cloned().ok_or_else(|| {
        VulkanResidentTokenModelPackageError::new(format!(
            "runtime resident derivation references unknown atomic group {group_id:?}",
        ))
    })?;
    group.resource_ids = group
        .resource_ids
        .iter()
        .map(|id| resource_ids.get(id).cloned().unwrap_or_else(|| id.clone()))
        .collect();
    group.resource_ids.sort();
    group.dependencies = group
        .dependencies
        .iter()
        .map(|dependency| {
            rewrite_resident_derivation_group(
                dependency,
                groups,
                resource_ids,
                rewritten,
                ids,
                visiting,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    group.dependencies.sort();
    group.id = package::compiled_atomic_group_identity(&group)
        .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
    visiting.remove(group_id);
    let target_id = group.id.clone();
    if rewritten.insert(target_id.clone(), group).is_some() {
        return runtime_resident_derivation_error(
            "runtime resident derivation caused an atomic-group identity collision",
        );
    }
    ids.insert(group_id.to_string(), target_id.clone());
    Ok(target_id)
}

fn rewrite_resident_derivation_templates(
    templates: &[CompiledPartitionTemplate],
    group_ids: &BTreeMap<String, String>,
) -> Result<(Vec<CompiledPartitionTemplate>, BTreeMap<String, String>), VulkanResidentTokenModelPackageError> {
    let mut rewritten = BTreeMap::new();
    let mut ids = BTreeMap::new();
    for source in templates {
        let mut target = source.clone();
        target.dependencies = target
            .dependencies
            .iter()
            .map(|id| group_ids.get(id).cloned().unwrap_or_else(|| id.clone()))
            .collect();
        target.dependencies.sort();
        target.id = package::compiled_partition_template_identity(&target)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        if rewritten.insert(target.id.clone(), target.clone()).is_some() {
            return runtime_resident_derivation_error(
                "runtime resident derivation caused a partition-template identity collision",
            );
        }
        ids.insert(source.id.clone(), target.id);
    }
    Ok((rewritten.into_values().collect(), ids))
}

fn rewrite_resident_derivation_binding(
    binding: &mut CompiledResourceBinding,
    resource_ids: &BTreeMap<String, String>,
    group_ids: &BTreeMap<String, String>,
    template_ids: &BTreeMap<String, String>,
) {
    match &mut binding.mapping {
        CompiledResourceBindingMapping::AtomicGroup {
            atomic_group_id,
            resource_id,
        }
        | CompiledResourceBindingMapping::SelectedAtomicGroup {
            atomic_group_id,
            resource_id,
            ..
        } => {
            *atomic_group_id = group_ids
                .get(atomic_group_id)
                .cloned()
                .unwrap_or_else(|| atomic_group_id.clone());
            *resource_id = resource_ids
                .get(resource_id)
                .cloned()
                .unwrap_or_else(|| resource_id.clone());
        }
        CompiledResourceBindingMapping::PartitionTemplateMember {
            partition_template_id,
            ..
        } => {
            *partition_template_id = template_ids
                .get(partition_template_id)
                .cloned()
                .unwrap_or_else(|| partition_template_id.clone());
        }
    }
}

fn rewrite_resident_derivation_selectors(
    selectors: &[CompiledResourceSelector],
    group_ids: &BTreeMap<String, String>,
    template_ids: &BTreeMap<String, String>,
) -> Result<(Vec<CompiledResourceSelector>, BTreeMap<String, String>), VulkanResidentTokenModelPackageError> {
    let mut rewritten = BTreeMap::new();
    let mut ids = BTreeMap::new();
    for source in selectors {
        let mut target = source.clone();
        match &mut target.mapping {
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                for id in atomic_group_ids {
                    *id = group_ids.get(id).cloned().unwrap_or_else(|| id.clone());
                }
            }
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => {
                *partition_template_id = template_ids
                    .get(partition_template_id)
                    .cloned()
                    .unwrap_or_else(|| partition_template_id.clone());
            }
        }
        target.id = package::compiled_selector_identity(&target)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        if rewritten.insert(target.id.clone(), target.clone()).is_some() {
            return runtime_resident_derivation_error(
                "runtime resident derivation caused a selector identity collision",
            );
        }
        ids.insert(source.id.clone(), target.id);
    }
    Ok((rewritten.into_values().collect(), ids))
}

fn rewrite_resident_derivation_checkpoints(
    checkpoints: &[CompiledResidencyCheckpoint],
    selector_ids: &BTreeMap<String, String>,
) -> Result<Vec<CompiledResidencyCheckpoint>, VulkanResidentTokenModelPackageError> {
    let mut rewritten = BTreeMap::new();
    for source in checkpoints {
        let mut target = source.clone();
        target.selector_ids = target
            .selector_ids
            .iter()
            .map(|id| selector_ids.get(id).cloned().unwrap_or_else(|| id.clone()))
            .collect();
        target.selector_ids.sort();
        target.id = package::compiled_checkpoint_identity(&target)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        if rewritten.insert(target.id.clone(), target).is_some() {
            return runtime_resident_derivation_error(
                "runtime resident derivation caused a checkpoint identity collision",
            );
        }
    }
    Ok(rewritten.into_values().collect())
}

fn runtime_resident_derivation_error<T>(
    message: impl Into<String>,
) -> Result<T, VulkanResidentTokenModelPackageError> {
    Err(VulkanResidentTokenModelPackageError::new(message))
}

#[cfg(test)]
mod runtime_resident_derivation_tests {
    use super::*;

    fn resource(digest_byte: char, dependencies: Vec<String>) -> CompiledImmutableResource {
        let mut resource = CompiledImmutableResource {
            id: String::new(),
            lifetime: CompiledResourceLifetime::Dynamic,
            ranges: vec![CompiledResourceByteRange {
                artifact_path: format!("weights/{digest_byte}.bin"),
                byte_offset: 0,
                byte_count: 4,
                alignment_bytes: 1,
                integrity: CompiledResourceRangeIntegrity {
                    algorithm: "sha256".to_string(),
                    digest: digest_byte.to_string().repeat(64),
                },
            }],
            dependencies,
            compatibility: CompiledResourceCompatibility {
                device_api: "vulkan".to_string(),
                storage_class: "storage_buffer".to_string(),
                read_only: true,
                required_features: Vec::new(),
            },
            resident_derivation: None,
        };
        resource.id = package::compiled_resource_identity(&resource).unwrap();
        resource
    }

    fn derivation() -> CompiledResourceResidentDerivation {
        CompiledResourceResidentDerivation {
            schema: RESIDENT_DERIVATION_SCHEMA.to_string(),
            kind: CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
            source_byte_count: 4,
            resident_byte_count: 8,
            required_features: vec![
                "shader_float8".to_string(),
                "shader_int8".to_string(),
                "shader_mixed_float_dot_product_float8_acc_float32".to_string(),
            ],
        }
    }

    fn dispatch(
        resident_derivation: Option<CompiledResourceResidentDerivationKind>,
        selection: VulkanResidentKernelResourceRepresentationSelection,
    ) -> VulkanResidentKernelResourceRepresentationDispatchSpec {
        VulkanResidentKernelResourceRepresentationDispatchSpec {
            schema: KERNEL_RESOURCE_REPRESENTATION_DISPATCH_SCHEMA.to_string(),
            source_representation:
                VulkanResidentKernelSourceResourceRepresentation::Mxfp4E2m1G32,
            resident_derivation,
            selection,
        }
    }

    #[test]
    fn kernel_resource_representation_contract_distinguishes_fixed_and_adaptive_dispatch() {
        let source = dispatch(
            None,
            VulkanResidentKernelResourceRepresentationSelection::FixedSource,
        );
        let adaptive = dispatch(
            Some(CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3),
            VulkanResidentKernelResourceRepresentationSelection::ResourceAddressTag,
        );

        assert!(source.is_exact_mxfp4_source());
        assert!(!source.selects_resident_derivation(
            CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
        ));
        assert!(!adaptive.is_exact_mxfp4_source());
        assert!(adaptive.selects_resident_derivation(
            CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
        ));

        let malformed = dispatch(
            Some(CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3),
            VulkanResidentKernelResourceRepresentationSelection::FixedSource,
        );
        assert!(!malformed.is_exact_mxfp4_source());
        assert!(!malformed.selects_resident_derivation(
            CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
        ));
    }

    #[test]
    fn resident_derivation_rewrites_every_transitive_content_identity() {
        let weight = resource('1', Vec::new());
        let dependent = resource('2', vec![weight.id.clone()]);
        let (resources, resource_ids) = rewrite_resident_derivation_resources(
            &[weight.clone(), dependent.clone()],
            &BTreeMap::from([(weight.id.clone(), derivation())]),
        )
        .unwrap();
        assert_ne!(resource_ids[&weight.id], weight.id);
        assert_ne!(resource_ids[&dependent.id], dependent.id);
        let derived = resources
            .iter()
            .find(|resource| resource.id == resource_ids[&weight.id])
            .unwrap();
        assert_eq!(derived.resident_derivation, Some(derivation()));
        assert_eq!(
            derived.compatibility.required_features,
            derivation().required_features,
        );
        let rewritten_dependent = resources
            .iter()
            .find(|resource| resource.id == resource_ids[&dependent.id])
            .unwrap();
        assert_eq!(
            rewritten_dependent.dependencies,
            vec![resource_ids[&weight.id].clone()],
        );

        let mut base_group = CompiledAtomicResidencyGroup {
            id: String::new(),
            lifetime: CompiledResourceLifetime::Dynamic,
            resource_ids: vec![weight.id.clone()],
            dependencies: Vec::new(),
        };
        base_group.id = package::compiled_atomic_group_identity(&base_group).unwrap();
        let mut dependent_group = CompiledAtomicResidencyGroup {
            id: String::new(),
            lifetime: CompiledResourceLifetime::Dynamic,
            resource_ids: vec![dependent.id.clone()],
            dependencies: vec![base_group.id.clone()],
        };
        dependent_group.id = package::compiled_atomic_group_identity(&dependent_group).unwrap();
        let (groups, group_ids) = rewrite_resident_derivation_groups(
            &[base_group.clone(), dependent_group.clone()],
            &resource_ids,
        )
        .unwrap();
        assert_ne!(group_ids[&base_group.id], base_group.id);
        assert_ne!(group_ids[&dependent_group.id], dependent_group.id);
        let rewritten_group = groups
            .iter()
            .find(|group| group.id == group_ids[&dependent_group.id])
            .unwrap();
        assert_eq!(
            rewritten_group.dependencies,
            vec![group_ids[&base_group.id].clone()],
        );
        assert_eq!(
            rewritten_group.resource_ids,
            vec![resource_ids[&dependent.id].clone()],
        );

        let mut template = CompiledPartitionTemplate {
            id: String::new(),
            partition_count: 2,
            lifetime: CompiledResourceLifetime::Dynamic,
            group_identity_seed: "group_seed".to_string(),
            member_templates: Vec::new(),
            dependencies: vec![dependent_group.id.clone()],
        };
        template.id = package::compiled_partition_template_identity(&template).unwrap();
        let (templates, template_ids) = rewrite_resident_derivation_templates(
            &[template.clone()],
            &group_ids,
        )
        .unwrap();
        assert_ne!(template_ids[&template.id], template.id);
        assert_eq!(
            templates[0].dependencies,
            vec![group_ids[&dependent_group.id].clone()],
        );

        let mut group_selector = CompiledResourceSelector {
            id: String::new(),
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "node".to_string(),
            domain_id: "domain".to_string(),
            resource_count: 2,
            selection_signal: "selection".to_string(),
            encoding: CompiledResourceSelectionEncoding {
                element_type: CompiledResourceSelectionElementType::U32,
                selection_count_per_activation: 1,
                index_shift: 0,
                index_mask: 1,
            },
            mapping: CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids: vec![
                    dependent_group.id.clone(),
                    base_group.id.clone(),
                ],
            },
        };
        group_selector.id = package::compiled_selector_identity(&group_selector).unwrap();
        let mut template_selector = group_selector.clone();
        template_selector.node_id = "partition_node".to_string();
        template_selector.mapping = CompiledResourceSelectorMapping::PartitionTemplate {
            partition_template_id: template.id.clone(),
        };
        template_selector.id = package::compiled_selector_identity(&template_selector).unwrap();
        let (selectors, selector_ids) = rewrite_resident_derivation_selectors(
            &[group_selector.clone(), template_selector.clone()],
            &group_ids,
            &template_ids,
        )
        .unwrap();
        let rewritten_group_selector = selectors
            .iter()
            .find(|selector| selector.id == selector_ids[&group_selector.id])
            .unwrap();
        assert_eq!(
            rewritten_group_selector.mapping,
            CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids: vec![
                    group_ids[&dependent_group.id].clone(),
                    group_ids[&base_group.id].clone(),
                ],
            },
        );
        let rewritten_template_selector = selectors
            .iter()
            .find(|selector| selector.id == selector_ids[&template_selector.id])
            .unwrap();
        assert_eq!(
            rewritten_template_selector.mapping,
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id: template_ids[&template.id].clone(),
            },
        );

        let mut selector_source_ids = vec![
            group_selector.id.clone(),
            template_selector.id.clone(),
        ];
        selector_source_ids.sort();
        let mut checkpoint = CompiledResidencyCheckpoint {
            id: String::new(),
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            after_node_id: "router".to_string(),
            resume_node_id: "node".to_string(),
            selector_ids: selector_source_ids,
        };
        checkpoint.id = package::compiled_checkpoint_identity(&checkpoint).unwrap();
        let checkpoints = rewrite_resident_derivation_checkpoints(
            &[checkpoint.clone()],
            &selector_ids,
        )
        .unwrap();
        assert_ne!(checkpoints[0].id, checkpoint.id);
        let mut expected_selector_ids = vec![
            selector_ids[&group_selector.id].clone(),
            selector_ids[&template_selector.id].clone(),
        ];
        expected_selector_ids.sort();
        assert_eq!(checkpoints[0].selector_ids, expected_selector_ids);

        let mut binding = CompiledResourceBinding {
            execution_scope: "target".to_string(),
            component_id: "component".to_string(),
            node_id: "node".to_string(),
            parameter_id: "weight".to_string(),
            mapping: CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id: dependent_group.id.clone(),
                resource_id: dependent.id.clone(),
                selection_signal: "selection".to_string(),
                selector_index: 0,
                parameter_slot: 0,
            },
        };
        rewrite_resident_derivation_binding(
            &mut binding,
            &resource_ids,
            &group_ids,
            &template_ids,
        );
        assert_eq!(
            binding.mapping,
            CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id: group_ids[&dependent_group.id].clone(),
                resource_id: resource_ids[&dependent.id].clone(),
                selection_signal: "selection".to_string(),
                selector_index: 0,
                parameter_slot: 0,
            },
        );
    }

    #[test]
    fn resident_derivation_rejects_resource_dependency_cycles() {
        let mut first = resource('3', Vec::new());
        let mut second = resource('4', Vec::new());
        first.dependencies = vec![second.id.clone()];
        second.dependencies = vec![first.id.clone()];

        let error = rewrite_resident_derivation_resources(
            &[first.clone(), second],
            &BTreeMap::from([(first.id, derivation())]),
        )
        .unwrap_err();

        assert!(error.to_string().contains("dependency cycle"));
    }
}
