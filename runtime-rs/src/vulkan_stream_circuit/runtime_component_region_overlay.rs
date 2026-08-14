#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeComponentRegionOverlay {
    schema: String,
    source_component_id: String,
    source: VulkanRuntimeComponentRegion,
    replacement: VulkanRuntimeComponentRegion,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VulkanRuntimeComponentRegion {
    nodes: Vec<crate::stream_circuit::CircuitNode>,
    kernels: Vec<VulkanResidentComponentKernelSpec>,
    parameter_refs:
        BTreeMap<String, crate::stream_circuit::ParameterRef>,
}

fn validate_runtime_component_region_overlay(
    overlay: &VulkanRuntimeComponentRegionOverlay,
    source: &VulkanResidentPackageComponentCircuit,
    source_execution: &VulkanResidentComponentExecutionSpec,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if overlay.schema != crate::VULKAN_COMPONENT_REGION_OVERLAY_SCHEMA
        || overlay.source_component_id != source.component_id
        || source_execution.component_id != source.component_id
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region overlay for {:?} changes its logical source identity",
            source.component_id
        )));
    }
    let source_node_ids = component_region_node_ids(
        &overlay.source.nodes,
        "component-region source nodes",
    )?;
    let source_kernel_ids = component_region_kernel_ids(
        &overlay.source.kernels,
        "component-region source kernels",
    )?;
    let replacement_node_ids = component_region_node_ids(
        &overlay.replacement.nodes,
        "component-region replacement nodes",
    )?;
    let replacement_kernel_ids = component_region_kernel_ids(
        &overlay.replacement.kernels,
        "component-region replacement kernels",
    )?;
    validate_component_region_node_kernel_pairs(
        &overlay.source,
        "component-region source",
    )?;
    validate_component_region_node_kernel_pairs(
        &overlay.replacement,
        "component-region replacement",
    )?;
    if source_node_ids != source_kernel_ids
        || replacement_node_ids != replacement_kernel_ids
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime component-region nodes and kernels cover different identities",
        ));
    }
    validate_component_region_parameter_scope(
        &overlay.source,
        "component-region source",
    )?;
    validate_component_region_parameter_scope(
        &overlay.replacement,
        "component-region replacement",
    )?;
    for node in &overlay.source.nodes {
        if source
            .circuit
            .nodes
            .iter()
            .find(|candidate| candidate.id == node.id)
            != Some(node)
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source node {:?} is stale",
                node.id
            )));
        }
    }
    for kernel in &overlay.source.kernels {
        if source_execution
            .kernels
            .iter()
            .find(|candidate| candidate.node_id == kernel.node_id)
            != Some(kernel)
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source kernel {:?} is stale",
                kernel.node_id
            )));
        }
    }
    let mut candidate_component = source.clone();
    apply_component_region_parameter_refs(
        &mut candidate_component,
        &source_node_ids,
        &overlay.source.parameter_refs,
        &overlay.replacement.parameter_refs,
    )?;
    candidate_component.circuit.nodes = replace_component_region_records(
        candidate_component.circuit.nodes,
        &source_node_ids,
        overlay.replacement.nodes.clone(),
        |node| node.id.as_str(),
    )?;
    let mut candidate_execution = source_execution.clone();
    candidate_execution.kernels = replace_component_region_records(
        candidate_execution.kernels,
        &source_kernel_ids,
        overlay.replacement.kernels.clone(),
        |kernel| kernel.node_id.as_str(),
    )?;
    for (execution_index, kernel) in
        candidate_execution.kernels.iter_mut().enumerate()
    {
        kernel.execution_index = execution_index;
    }
    validate_runtime_component_region_result(
        &candidate_component,
        &candidate_execution,
    )?;
    Ok(())
}

fn component_region_node_ids(
    nodes: &[crate::stream_circuit::CircuitNode],
    label: &str,
) -> Result<BTreeSet<String>, VulkanResidentTokenModelPackageError> {
    let ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    if nodes.is_empty() || ids.len() != nodes.len() || ids.contains("") {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} must be non-empty and uniquely identified",
        )));
    }
    Ok(ids)
}

fn component_region_kernel_ids(
    kernels: &[VulkanResidentComponentKernelSpec],
    label: &str,
) -> Result<BTreeSet<String>, VulkanResidentTokenModelPackageError> {
    let ids = kernels
        .iter()
        .map(|kernel| kernel.node_id.clone())
        .collect::<BTreeSet<_>>();
    if kernels.is_empty() || ids.len() != kernels.len() || ids.contains("") {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} must be non-empty and uniquely identified",
        )));
    }
    Ok(ids)
}

fn component_region_parameter_ref_ids(
    refs: &BTreeMap<String, crate::stream_circuit::ParameterRef>,
    label: &str,
) -> Result<BTreeSet<String>, VulkanResidentTokenModelPackageError> {
    let ids = refs.keys().cloned().collect::<BTreeSet<_>>();
    if ids.contains("") {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} must use non-empty parameter identities",
        )));
    }
    Ok(ids)
}

fn component_region_node_parameter_ids(
    nodes: &[crate::stream_circuit::CircuitNode],
) -> BTreeSet<String> {
    nodes
        .iter()
        .flat_map(|node| node.params.iter().cloned())
        .collect()
}

fn validate_component_region_node_kernel_pairs(
    region: &VulkanRuntimeComponentRegion,
    label: &str,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if region.nodes.len() != region.kernels.len()
        || region
            .nodes
            .iter()
            .zip(&region.kernels)
            .any(|(node, kernel)| {
                node.id != kernel.node_id || node.op != kernel.op
            })
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} nodes and kernels must have identical ordered identities and operations",
        )));
    }
    Ok(())
}

fn validate_component_region_parameter_scope(
    region: &VulkanRuntimeComponentRegion,
    label: &str,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let declared = component_region_parameter_ref_ids(
        &region.parameter_refs,
        &format!("{label} parameter refs"),
    )?;
    let used = component_region_node_parameter_ids(&region.nodes);
    if !declared.is_subset(&used) {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "{label} declares parameter refs that its nodes do not use: {:?}",
            declared.difference(&used).collect::<Vec<_>>(),
        )));
    }
    Ok(())
}

fn apply_component_region_parameter_refs(
    component: &mut VulkanResidentPackageComponentCircuit,
    source_node_ids: &BTreeSet<String>,
    source_refs: &BTreeMap<String, crate::stream_circuit::ParameterRef>,
    replacement_refs: &BTreeMap<String, crate::stream_circuit::ParameterRef>,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if component.circuit.parameters.refs != component.params.refs {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region component {:?} has inconsistent parameter tables",
            component.component_id,
        )));
    }
    let source_ref_ids = component_region_parameter_ref_ids(
        source_refs,
        "component-region source parameter refs",
    )?;
    let replacement_ref_ids = component_region_parameter_ref_ids(
        replacement_refs,
        "component-region replacement parameter refs",
    )?;
    for (parameter_id, source_ref) in source_refs {
        if component.params.refs.get(parameter_id) != Some(source_ref) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source parameter ref {parameter_id:?} is stale",
            )));
        }
    }
    if let Some(collision) = replacement_ref_ids
        .difference(&source_ref_ids)
        .find(|parameter_id| component.params.refs.contains_key(*parameter_id))
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region replacement parameter ref {collision:?} collides with an unrelated binding",
        )));
    }
    for parameter_id in &source_ref_ids {
        if replacement_refs.get(parameter_id) == source_refs.get(parameter_id)
        {
            continue;
        }
        if component.circuit.nodes.iter().any(|node| {
            !source_node_ids.contains(&node.id)
                && node.params.contains(parameter_id)
        }) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source parameter ref {parameter_id:?} is still used outside the replaced region",
            )));
        }
    }
    for parameter_id in &source_ref_ids {
        component.circuit.parameters.refs.remove(parameter_id);
        component.params.refs.remove(parameter_id);
    }
    for (parameter_id, parameter_ref) in replacement_refs {
        component
            .circuit
            .parameters
            .refs
            .insert(parameter_id.clone(), parameter_ref.clone());
        component
            .params
            .refs
            .insert(parameter_id.clone(), parameter_ref.clone());
    }
    Ok(())
}

fn validate_runtime_component_region_result(
    component: &VulkanResidentPackageComponentCircuit,
    execution: &VulkanResidentComponentExecutionSpec,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    component.circuit.validate_contract().map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region replacement for {:?} is invalid: {error}",
            component.component_id,
        ))
    })?;
    if component.circuit.parameters.refs != component.params.refs {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region replacement for {:?} produced inconsistent parameter tables",
            component.component_id,
        )));
    }
    validate_component_executions(
        "runtime component-region replacement",
        std::slice::from_ref(execution),
    )?;
    if component.component_id != execution.component_id
        || component.operator_type != execution.operator_type
        || component.implementation != execution.implementation
        || component.circuit.nodes.len() != execution.kernels.len()
        || component
            .circuit
            .nodes
            .iter()
            .zip(&execution.kernels)
            .enumerate()
            .any(|(index, (node, kernel))| {
                let source_node_ids = semantic_source_node_ids(node);
                let semantic_module_ids = component
                    .circuit
                    .semantic_module_tree
                    .as_ref()
                    .map(|tree| {
                        tree.modules
                            .iter()
                            .filter(|module| {
                                module
                                    .source_node_ids
                                    .iter()
                                    .any(|node_id| {
                                        source_node_ids.contains(node_id)
                                    })
                            })
                            .map(|module| module.id.clone())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                kernel.execution_index != index
                    || node.id != kernel.node_id
                    || node.op != kernel.op
                    || kernel.source_node_ids != source_node_ids
                    || kernel.semantic_module_ids != semantic_module_ids
            })
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "runtime component-region replacement for {:?} produced inconsistent node and kernel execution",
            component.component_id,
        )));
    }
    Ok(())
}

fn rebase_component_region_shader_paths(
    overlay: &mut VulkanRuntimeComponentRegionOverlay,
    package_root: &Path,
    candidate_root: &Path,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    for kernel in &mut overlay.replacement.kernels {
        kernel.shader_path = rebase_overlay_shader_path(
            &kernel.shader_path,
            None,
            package_root,
            candidate_root,
            "runtime component-region shader",
        )?;
        for implementation in &mut kernel.batch_implementations {
            for stage in &mut implementation.stages {
                stage.shader_path = rebase_overlay_shader_path(
                    &stage.shader_path,
                    None,
                    package_root,
                    candidate_root,
                    "runtime component-region batch shader",
                )?;
            }
        }
    }
    Ok(())
}

fn mount_runtime_component_region_overlay(
    runtime_model: &mut VulkanResidentRuntimeModel,
    runtime_instance_id: &str,
    mut overlay: VulkanRuntimeComponentRegionOverlay,
    package_root: &Path,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let (source_nodes, replacement_nodes) =
        executable_component_region_nodes(runtime_model, &overlay)?;
    overlay.source.nodes = source_nodes;
    overlay.replacement.nodes = replacement_nodes;
    let component_index = runtime_model
        .circuit_graph
        .components
        .iter()
        .position(|component| component.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region implementation cannot find component {runtime_instance_id:?}"
            ))
        })?;
    let execution_index = runtime_model
        .component_executions
        .iter()
        .position(|execution| execution.component_id == runtime_instance_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region implementation cannot find execution for {runtime_instance_id:?}"
            ))
        })?;
    let mut component = runtime_model.circuit_graph.components[component_index].clone();
    let mut execution = runtime_model.component_executions[execution_index].clone();

    let source_node_ids = component_region_node_ids(
        &overlay.source.nodes,
        "component-region source nodes",
    )?;
    let replacement_node_ids = component_region_node_ids(
        &overlay.replacement.nodes,
        "component-region replacement nodes",
    )?;
    let source_kernel_ids = component_region_kernel_ids(
        &overlay.source.kernels,
        "component-region source kernels",
    )?;
    let replacement_kernel_ids = component_region_kernel_ids(
        &overlay.replacement.kernels,
        "component-region replacement kernels",
    )?;
    validate_component_region_node_kernel_pairs(
        &overlay.source,
        "component-region source",
    )?;
    validate_component_region_node_kernel_pairs(
        &overlay.replacement,
        "component-region replacement",
    )?;
    if source_node_ids != source_kernel_ids
        || replacement_node_ids != replacement_kernel_ids
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime component-region nodes and kernels cover different identities",
        ));
    }
    validate_component_region_parameter_scope(
        &overlay.source,
        "component-region source",
    )?;
    validate_component_region_parameter_scope(
        &overlay.replacement,
        "component-region replacement",
    )?;
    if component.circuit.nodes.iter().any(|node| {
        replacement_node_ids.contains(&node.id)
            && !source_node_ids.contains(&node.id)
    }) || execution.kernels.iter().any(|kernel| {
        replacement_kernel_ids.contains(&kernel.node_id)
            && !source_kernel_ids.contains(&kernel.node_id)
    }) {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime component-region replacement collides with an unrelated node",
        ));
    }
    for source in &overlay.source.nodes {
        if component
            .circuit
            .nodes
            .iter()
            .find(|node| node.id == source.id)
            != Some(source)
        {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source node {:?} changed before composition",
                source.id
            )));
        }
    }
    for source in &overlay.source.kernels {
        let current = execution
            .kernels
            .iter()
            .find(|kernel| kernel.node_id == source.node_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "runtime component-region source kernel {:?} is missing before composition",
                    source.node_id
                ))
            })?;
        if !runtime_kernel_matches_region_source(
            current,
            source,
            package_root,
        )? {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source kernel {:?} changed before composition",
                source.node_id
            )));
        }
    }

    apply_component_region_parameter_refs(
        &mut component,
        &source_node_ids,
        &overlay.source.parameter_refs,
        &overlay.replacement.parameter_refs,
    )?;
    component.circuit.nodes = replace_component_region_records(
        component.circuit.nodes,
        &source_node_ids,
        overlay.replacement.nodes,
        |node| node.id.as_str(),
    )?;
    execution.kernels = replace_component_region_records(
        execution.kernels,
        &source_kernel_ids,
        overlay.replacement.kernels,
        |kernel| kernel.node_id.as_str(),
    )?;
    for (execution_index, kernel) in
        execution.kernels.iter_mut().enumerate()
    {
        kernel.execution_index = execution_index;
    }
    validate_runtime_component_region_result(&component, &execution)?;
    runtime_model.circuit_graph.components[component_index] = component;
    runtime_model.component_executions[execution_index] = execution;
    Ok(())
}

fn executable_component_region_nodes(
    runtime_model: &VulkanResidentRuntimeModel,
    overlay: &VulkanRuntimeComponentRegionOverlay,
) -> Result<
    (
        Vec<crate::stream_circuit::CircuitNode>,
        Vec<crate::stream_circuit::CircuitNode>,
    ),
    VulkanResidentTokenModelPackageError,
> {
    let source_execution = runtime_model
        .package
        .component_executions
        .iter()
        .find(|execution| execution.component_id == overlay.source_component_id)
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "runtime component-region source execution {:?} is unavailable",
                overlay.source_component_id
            ))
        })?;
    let mut source_nodes = overlay.source.nodes.clone();
    package::attach_node_kernel_runtime_contracts(
        &overlay.source_component_id,
        &mut source_nodes,
        &source_execution.kernels,
    )?;
    let mut replacement_nodes = overlay.replacement.nodes.clone();
    package::attach_node_kernel_runtime_contracts(
        &overlay.source_component_id,
        &mut replacement_nodes,
        &overlay.replacement.kernels,
    )?;
    Ok((source_nodes, replacement_nodes))
}

fn runtime_kernel_matches_region_source(
    current: &VulkanResidentComponentKernelSpec,
    source: &VulkanResidentComponentKernelSpec,
    package_root: &Path,
) -> Result<bool, VulkanResidentTokenModelPackageError> {
    let mut normalized = current.clone();
    normalized.execution_index = source.execution_index;
    if !runtime_shader_reference_matches_source(
        &normalized.shader_path,
        &source.shader_path,
        package_root,
    )? {
        return Ok(false);
    }
    normalized.shader_path = source.shader_path.clone();
    if normalized.batch_implementations.len()
        != source.batch_implementations.len()
    {
        return Ok(false);
    }
    for (current_implementation, source_implementation) in normalized
        .batch_implementations
        .iter_mut()
        .zip(&source.batch_implementations)
    {
        if current_implementation.stages.len()
            != source_implementation.stages.len()
        {
            return Ok(false);
        }
        for (current_stage, source_stage) in current_implementation
            .stages
            .iter_mut()
            .zip(&source_implementation.stages)
        {
            if !runtime_shader_reference_matches_source(
                &current_stage.shader_path,
                &source_stage.shader_path,
                package_root,
            )? {
                return Ok(false);
            }
            current_stage.shader_path = source_stage.shader_path.clone();
        }
    }
    Ok(&normalized == source)
}

fn runtime_shader_reference_matches_source(
    current: &str,
    source: &str,
    package_root: &Path,
) -> Result<bool, VulkanResidentTokenModelPackageError> {
    if current == source {
        return Ok(true);
    }
    Ok(Path::new(current)
        == contained_package_artifact(
            package_root,
            source,
            "runtime component-region source shader",
        )?)
}

fn replace_component_region_records<T, F>(
    current: Vec<T>,
    source_ids: &BTreeSet<String>,
    replacement: Vec<T>,
    id: F,
) -> Result<Vec<T>, VulkanResidentTokenModelPackageError>
where
    F: Fn(&T) -> &str,
{
    let mut replacement = Some(replacement);
    let mut replaced = BTreeSet::new();
    let mut output = Vec::with_capacity(current.len());
    for record in current {
        let record_id = id(&record);
        if source_ids.contains(record_id) {
            if replacement.is_some() {
                output.extend(
                    replacement
                        .take()
                        .expect("checked component-region replacement"),
                );
            }
            replaced.insert(record_id.to_string());
        } else {
            output.push(record);
        }
    }
    if replaced != *source_ids {
        return Err(VulkanResidentTokenModelPackageError::new(
            "runtime component-region did not replace its exact source set",
        ));
    }
    Ok(output)
}
