fn canonical_runtime_execution_identity(
    runtime_model: &VulkanResidentRuntimeModel,
    physical_execution_plan: &VulkanRuntimePhysicalExecutionPlan,
    dynamic_state_capacity_activations: usize,
    speculative_decoders_enabled: bool,
    resource_residency_policy: ResourceResidencyPolicy,
) -> Result<String, VulkanResidentTokenModelPackageError> {
    let mut instances = runtime_model.runtime_graph.instances.clone();
    instances.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    let mut edges = runtime_model.runtime_graph.edges.clone();
    edges.sort_by(|left, right| {
        (
            left.source.component_id.as_str(),
            left.source.port_id.as_str(),
            left.destination.component_id.as_str(),
            left.destination.port_id.as_str(),
            left.id.as_str(),
        )
            .cmp(&(
                right.source.component_id.as_str(),
                right.source.port_id.as_str(),
                right.destination.component_id.as_str(),
                right.destination.port_id.as_str(),
                right.id.as_str(),
            ))
    });
    let mut external_inputs = runtime_model.runtime_graph.boundary.external_inputs.clone();
    external_inputs.sort_by(|left, right| left.id.cmp(&right.id));
    let mut public_outputs = runtime_model.runtime_graph.boundary.public_outputs.clone();
    public_outputs.sort_by(|left, right| left.id.cmp(&right.id));
    let mut component_executions = runtime_model.component_executions.as_ref().clone();
    component_executions.sort_by(|left, right| left.component_id.cmp(&right.component_id));

    let identity = serde_json::json!({
        "schema": "nerve.runtime_execution_identity.v2",
        "package": {
            "id": runtime_model.package.package_id,
            "schema": runtime_model.package.schema,
            "artifact_integrity": runtime_model.package.artifact_integrity,
        },
        "graph": {
            "schema": runtime_model.runtime_graph.schema,
            "topology": runtime_model.runtime_graph.topology,
            "default_device_id": runtime_model.runtime_graph.default_device_id,
            "instances": instances,
            "edges": edges,
            "boundary": {
                "external_inputs": external_inputs,
                "public_outputs": public_outputs,
            },
        },
        "component_executions": component_executions,
        "execution_scope": runtime_model.execution_scope,
        "implementation_selection": runtime_model.implementation_selection,
        "physical_execution_plan": {
            "prefill_activation_batch_width": physical_execution_plan.prefill_activation_batch_width,
            "component_device_pools": {
                "decode": physical_execution_plan.component_device_pools.decode,
                "decode_batch": physical_execution_plan.component_device_pools.decode_batch,
                "prefill": physical_execution_plan.component_device_pools.prefill,
            },
            "decode_execution_cases_by_component": physical_execution_plan.decode_execution_cases_by_component,
            "decode_batch_execution_cases_by_component": physical_execution_plan.decode_batch_execution_cases_by_component,
            "prefill_execution_cases_by_component": physical_execution_plan.prefill_execution_cases_by_component,
            "decode_contract_ids_by_component": physical_execution_plan.decode_contract_ids_by_component,
            "decode_batch_contract_ids_by_component": physical_execution_plan.decode_batch_contract_ids_by_component,
            "prefill_contract_ids_by_component": physical_execution_plan.prefill_contract_ids_by_component,
        },
        "state_capacity_activations": dynamic_state_capacity_activations,
        "speculative_decoders_enabled": speculative_decoders_enabled,
        "resource_residency_policy": resource_residency_policy,
    });
    let bytes = serde_json::to_vec(&identity).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to serialize canonical runtime execution identity: {error}"
        ))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn canonical_mounted_runtime_execution_identity(
    base_identity: &str,
    selected_resource_placements: &[VulkanSelectedResourcePlacementPlan],
) -> Result<String, VulkanResidentTokenModelPackageError> {
    if base_identity.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "mounted runtime execution identity requires a base identity",
        ));
    }
    if selected_resource_placements.is_empty() {
        return Ok(base_identity.to_string());
    }
    let mut placements = selected_resource_placements
        .iter()
        .map(|placement| {
            let mut assignments = placement
                .assignments
                .iter()
                .map(|assignment| {
                    serde_json::json!({
                        "resource_index": assignment.resource_index,
                        "device_id": assignment.device_id,
                    })
                })
                .collect::<Vec<_>>();
            assignments.sort_by(|left, right| left.to_string().cmp(&right.to_string()));
            serde_json::json!({
                "selector_id": placement.selector_id,
                "assignments": assignments,
            })
        })
        .collect::<Vec<_>>();
    placements.sort_by(|left, right| left.to_string().cmp(&right.to_string()));
    let identity = serde_json::json!({
        "schema": "nerve.mounted_runtime_execution_identity.v1",
        "base_identity": base_identity,
        "selected_resource_placements": placements,
    });
    let bytes = serde_json::to_vec(&identity).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to serialize mounted runtime execution identity: {error}",
        ))
    })?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod runtime_execution_identity_tests {
    use super::*;

    #[test]
    fn canonical_execution_identity_ignores_non_semantic_graph_storage_order() {
        let mut left = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let mut right = left.clone();
        right.runtime_graph.instances.reverse();
        right.runtime_graph.edges.reverse();
        right.runtime_graph.boundary.external_inputs.reverse();
        right.runtime_graph.boundary.public_outputs.reverse();
        right.component_executions.reverse();

        assert_eq!(
            canonical_runtime_execution_identity(
                &left,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&left),
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap(),
            canonical_runtime_execution_identity(
                &right,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&right),
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );

        left.runtime_graph.instances[0].device_id = "gpu1".to_string();
        assert_ne!(
            canonical_runtime_execution_identity(
                &left,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&left),
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap(),
            canonical_runtime_execution_identity(
                &right,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&right),
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );
    }

    #[test]
    fn canonical_execution_identity_includes_capacity_and_kernel_selection() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let base = canonical_runtime_execution_identity(
            &model,
            &VulkanRuntimePhysicalExecutionPlan::uniform(&model),
            4096,
            false,
            ResourceResidencyPolicy::Eager,
        )
        .unwrap();
        assert_ne!(
            base,
            canonical_runtime_execution_identity(
                &model,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&model),
                8192,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );

        let mut changed_kernel = model.clone();
        changed_kernel.component_executions[0].kernels[0].local_size_x += 1;
        assert_ne!(
            base,
            canonical_runtime_execution_identity(
                &changed_kernel,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&changed_kernel),
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );
        assert_ne!(
            base,
            canonical_runtime_execution_identity(
                &model,
                &VulkanRuntimePhysicalExecutionPlan::uniform(&model),
                4096,
                false,
                ResourceResidencyPolicy::DemandRetained,
            )
            .unwrap()
        );

        let mut explicit_contract = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
        explicit_contract.decode_contract_ids_by_component.insert(
            model.circuit_graph.components[0].component_id.clone(),
            BTreeSet::from(["sha256:manual-contract".to_string()]),
        );
        assert_ne!(
            base,
            canonical_runtime_execution_identity(
                &model,
                &explicit_contract,
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );
    }

    #[test]
    fn canonical_execution_identity_includes_phase_local_physical_plan() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let uniform = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
        let base = canonical_runtime_execution_identity(
            &model,
            &uniform,
            4096,
            false,
            ResourceResidencyPolicy::Eager,
        )
        .unwrap();
        let component_id = model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.runtime_role.is_signal_processor())
            .unwrap()
            .component_id
            .clone();
        let mut split_decode = uniform;
        split_decode.component_device_pools.decode.insert(
            component_id,
            vec!["gpu0".to_string(), "gpu1".to_string()],
        );

        assert_ne!(
            base,
            canonical_runtime_execution_identity(
                &model,
                &split_decode,
                4096,
                false,
                ResourceResidencyPolicy::Eager,
            )
            .unwrap()
        );
    }

    #[test]
    fn mounted_execution_identity_includes_selected_resource_ownership() {
        let placement = |device_id: &str| VulkanSelectedResourcePlacementPlan {
            selector_id: "experts".to_string(),
            assignments: vec![crate::vulkan_distributed::VulkanSelectedResourceAssignment {
                resource_index: 0,
                device_id: device_id.to_string(),
            }],
            device_loads: Vec::new(),
            maximum_first_moment_ns: 0,
            maximum_second_moment_ns2: 0,
        };
        let base = canonical_mounted_runtime_execution_identity("base", &[]).unwrap();
        let owner = canonical_mounted_runtime_execution_identity("base", &[placement("gpu0")])
            .unwrap();
        let helper = canonical_mounted_runtime_execution_identity("base", &[placement("gpu1")])
            .unwrap();

        assert_eq!(base, "base");
        assert_ne!(owner, helper);
        assert_eq!(owner.len(), 64);
    }
}
