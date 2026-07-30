#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimeResourceContractError(String);

impl VulkanRuntimeResourceContractError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for VulkanRuntimeResourceContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanRuntimeResourceContractError {}

/// Instantiates source-component residency semantics for the effective runtime
/// graph. Immutable resources, groups, and partition templates retain their
/// compiled content identities; only bindings, selectors, and physical
/// checkpoints acquire runtime-instance identities.
fn instantiate_runtime_resource_contract(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<CompiledResourceResidencyContract, VulkanRuntimeResourceContractError> {
    let source = &runtime_model.package.resource_residency;
    if runtime_model.execution_scope != "target" {
        return Ok(source.clone());
    }

    let mounted_component_ids = runtime_model
        .circuit_graph
        .components
        .iter()
        .map(|component| component.component_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut runtime_instances_by_source = BTreeMap::<String, Vec<String>>::new();
    for instance in runtime_model.runtime_graph.instances.iter().filter(|instance| {
        instance.enabled && mounted_component_ids.contains(instance.instance_id.as_str())
    }) {
        runtime_instances_by_source
            .entry(instance.source_component_id.clone())
            .or_default()
            .push(instance.instance_id.clone());
    }
    for instances in runtime_instances_by_source.values_mut() {
        instances.sort_unstable();
        instances.dedup();
    }

    let runtime_instances_for = |execution_scope: &str, component_id: &str| {
        if execution_scope == "target" {
            runtime_instances_by_source
                .get(component_id)
                .cloned()
                .unwrap_or_default()
        } else {
            vec![component_id.to_string()]
        }
    };

    let mut bindings = Vec::new();
    for binding in &source.bindings {
        for runtime_component_id in
            runtime_instances_for(&binding.execution_scope, &binding.component_id)
        {
            let mut runtime_binding = binding.clone();
            runtime_binding.component_id = runtime_component_id;
            bindings.push(runtime_binding);
        }
    }
    bindings.sort_by(|left, right| {
        (
            left.execution_scope.as_str(),
            left.component_id.as_str(),
            left.node_id.as_str(),
            left.parameter_id.as_str(),
        )
            .cmp(&(
                right.execution_scope.as_str(),
                right.component_id.as_str(),
                right.node_id.as_str(),
                right.parameter_id.as_str(),
            ))
    });

    let mut selectors = Vec::new();
    let mut runtime_selector_ids =
        BTreeMap::<(String, String), String>::new();
    for selector in &source.selectors {
        for runtime_component_id in
            runtime_instances_for(&selector.execution_scope, &selector.component_id)
        {
            let mut runtime_selector = selector.clone();
            runtime_selector.component_id = runtime_component_id.clone();
            runtime_selector.id =
                package::compiled_selector_identity(&runtime_selector).map_err(|error| {
                    VulkanRuntimeResourceContractError::new(format!(
                        "failed to derive runtime selector identity for {}.{}: {error}",
                        runtime_component_id, selector.node_id
                    ))
                })?;
            if runtime_selector_ids
                .insert(
                    (selector.id.clone(), runtime_component_id.clone()),
                    runtime_selector.id.clone(),
                )
                .is_some()
            {
                return Err(VulkanRuntimeResourceContractError::new(format!(
                    "runtime resource selector {:?} was instantiated twice for component {runtime_component_id:?}",
                    selector.id
                )));
            }
            selectors.push(runtime_selector);
        }
    }
    selectors.sort_by(|left, right| left.id.cmp(&right.id));
    if selectors
        .windows(2)
        .any(|pair| pair[0].id == pair[1].id)
    {
        return Err(VulkanRuntimeResourceContractError::new(
            "runtime resource selector identities collided",
        ));
    }

    let mut checkpoints = Vec::new();
    for checkpoint in &source.checkpoints {
        for runtime_component_id in runtime_instances_for(
            &checkpoint.execution_scope,
            &checkpoint.component_id,
        ) {
            let mut runtime_checkpoint = checkpoint.clone();
            runtime_checkpoint.component_id = runtime_component_id.clone();
            runtime_checkpoint.selector_ids = checkpoint
                .selector_ids
                .iter()
                .map(|selector_id| {
                    runtime_selector_ids
                        .get(&(selector_id.clone(), runtime_component_id.clone()))
                        .cloned()
                        .ok_or_else(|| {
                            VulkanRuntimeResourceContractError::new(format!(
                                "runtime checkpoint {:?} has no selector {:?} for component {runtime_component_id:?}",
                                checkpoint.id, selector_id
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            runtime_checkpoint.selector_ids.sort();
            runtime_checkpoint.id =
                package::compiled_checkpoint_identity(&runtime_checkpoint).map_err(
                    |error| {
                        VulkanRuntimeResourceContractError::new(format!(
                            "failed to derive runtime checkpoint identity for component {runtime_component_id:?}: {error}"
                        ))
                    },
                )?;
            checkpoints.push(runtime_checkpoint);
        }
    }
    checkpoints.sort_by(|left, right| left.id.cmp(&right.id));
    if checkpoints
        .windows(2)
        .any(|pair| pair[0].id == pair[1].id)
    {
        return Err(VulkanRuntimeResourceContractError::new(
            "runtime resource checkpoint identities collided",
        ));
    }

    Ok(CompiledResourceResidencyContract {
        schema: source.schema.clone(),
        identity_algorithm: source.identity_algorithm.clone(),
        state_machine_schema: source.state_machine_schema.clone(),
        supported_policies: source.supported_policies.clone(),
        resources: source.resources.clone(),
        atomic_groups: source.atomic_groups.clone(),
        partition_templates: source.partition_templates.clone(),
        bindings,
        selectors,
        checkpoints,
    })
}
