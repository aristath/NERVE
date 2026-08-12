#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanSparseMoeExecutionContract {
    component_count: usize,
    declared_experts_per_activation: usize,
    selected_routes_per_activation: usize,
    submitted_route_slots_per_activation: usize,
}

impl VulkanResidentRuntimeModel {
    pub fn sparse_moe_execution_contract(
        &self,
    ) -> Result<VulkanSparseMoeExecutionContract, VulkanResidentTokenModelPackageError> {
        let mut contract = VulkanSparseMoeExecutionContract::default();
        for component in &self.circuit_graph.components {
            for node in &component.circuit.nodes {
                let Some((declared, selected)) = sparse_moe_geometry(node)? else {
                    continue;
                };
                if selected == 0 || selected > declared {
                    return Err(VulkanResidentTokenModelPackageError::new(format!(
                        "sparse MoE node {}.{} selects {selected} routes from {declared} experts",
                        component.component_id, node.id
                    )));
                }
                validate_sparse_moe_execution_chain(&component.circuit.nodes, node)?;
                contract.component_count = contract.component_count.saturating_add(1);
                contract.declared_experts_per_activation = contract
                    .declared_experts_per_activation
                    .saturating_add(declared);
                contract.selected_routes_per_activation = contract
                    .selected_routes_per_activation
                    .saturating_add(selected);
                contract.submitted_route_slots_per_activation = contract
                    .submitted_route_slots_per_activation
                    .saturating_add(selected);
            }
        }
        Ok(contract)
    }
}

fn sparse_moe_geometry(
    node: &CircuitNode,
) -> Result<Option<(usize, usize)>, VulkanResidentTokenModelPackageError> {
    let selected = match node.op.as_str() {
        "sparse_moe_gate_up" | "independent_sparse_moe_gate_up" => {
            sparse_moe_attr_usize(node, "experts_per_token")?
        }
        _ => return Ok(None),
    };
    let declared = if node.op == "sparse_moe_gate_up" {
        sparse_moe_attr_usize(node, "num_experts")?
    } else {
        let accesses = node
            .attrs
            .get("selected_parameter_accesses")
            .and_then(Value::as_array)
            .ok_or_else(|| sparse_moe_error(node, "has no selected-parameter access"))?;
        let [access] = accesses.as_slice() else {
            return Err(sparse_moe_error(
                node,
                "requires exactly one selected-parameter access",
            ));
        };
        let mapping = access
            .get("mapping")
            .and_then(Value::as_array)
            .ok_or_else(|| sparse_moe_error(node, "has no selected-resource mapping"))?;
        if mapping.is_empty()
            || mapping.iter().enumerate().any(|(selector, entry)| {
                entry.get("selector").and_then(Value::as_u64)
                    != u64::try_from(selector).ok()
                    || entry
                        .get("parameter_ids")
                        .and_then(Value::as_array)
                        .is_none_or(Vec::is_empty)
            })
        {
            return Err(sparse_moe_error(
                node,
                "does not map every selected resource exactly once",
            ));
        }
        mapping.len()
    };
    Ok(Some((declared, selected)))
}

fn validate_sparse_moe_execution_chain(
    nodes: &[CircuitNode],
    gate_up: &CircuitNode,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    let route_signal = if gate_up.op == "independent_sparse_moe_gate_up" {
        gate_up
            .attrs
            .get("selected_parameter_accesses")
            .and_then(Value::as_array)
            .and_then(|accesses| accesses.first())
            .and_then(|access| access.get("execution_signal"))
            .and_then(Value::as_str)
    } else {
        gate_up.inputs.get(1).map(String::as_str)
    }
    .filter(|signal| gate_up.inputs.iter().any(|input| input == signal))
    .ok_or_else(|| sparse_moe_error(gate_up, "has no routed execution signal"))?;
    let [gate_output] = gate_up.outputs.as_slice() else {
        return Err(sparse_moe_error(
            gate_up,
            "must publish exactly one expert intermediate",
        ));
    };
    require_one_sparse_moe_node(
        nodes,
        gate_up,
        "router",
        |candidate| {
            matches!(candidate.op.as_str(), "moe_route" | "moe_topk")
                && candidate
                    .outputs
                    .iter()
                    .any(|output| output == route_signal)
        },
    )?;
    let expected_down_op = if gate_up.op == "independent_sparse_moe_gate_up" {
        "independent_sparse_moe_down"
    } else {
        "sparse_moe_down"
    };
    let down = require_one_sparse_moe_node(nodes, gate_up, "down projection", |candidate| {
        candidate.op == expected_down_op
            && candidate.inputs.iter().any(|input| input == gate_output)
            && candidate.inputs.iter().any(|input| input == route_signal)
    })?;
    let [down_output] = down.outputs.as_slice() else {
        return Err(sparse_moe_error(
            down,
            "must publish exactly one routed expert output",
        ));
    };
    require_one_sparse_moe_node(nodes, gate_up, "reduction", |candidate| {
        candidate.op == "moe_reduce"
            && candidate.inputs.iter().any(|input| input == down_output)
    })?;
    Ok(())
}

fn require_one_sparse_moe_node<'a>(
    nodes: &'a [CircuitNode],
    gate_up: &CircuitNode,
    role: &str,
    predicate: impl Fn(&CircuitNode) -> bool,
) -> Result<&'a CircuitNode, VulkanResidentTokenModelPackageError> {
    let matching = nodes
        .iter()
        .filter(|candidate| predicate(candidate))
        .collect::<Vec<_>>();
    let [matching] = matching.as_slice() else {
        return Err(sparse_moe_error(
            gate_up,
            &format!("resolves {} {role} nodes", matching.len()),
        ));
    };
    Ok(*matching)
}

fn sparse_moe_error(
    node: &CircuitNode,
    message: &str,
) -> VulkanResidentTokenModelPackageError {
    VulkanResidentTokenModelPackageError::new(format!(
        "sparse MoE node {:?} {message}",
        node.id
    ))
}

impl VulkanSparseMoeExecutionContract {
    pub fn work_report(
        &self,
        prefill_activation_count: usize,
        decode_activation_count: usize,
    ) -> RuntimeSparseMoeWorkReport {
        if self.component_count == 0 {
            return RuntimeSparseMoeWorkReport::default();
        }
        let activation_count =
            prefill_activation_count.saturating_add(decode_activation_count);
        let declared_expert_slots = activation_count
            .saturating_mul(self.declared_experts_per_activation);
        let selected_expert_routes =
            activation_count.saturating_mul(self.selected_routes_per_activation);
        let submitted_expert_route_slots = activation_count
            .saturating_mul(self.submitted_route_slots_per_activation);
        RuntimeSparseMoeWorkReport {
            component_count: self.component_count,
            activation_count,
            declared_expert_slots,
            selected_expert_routes,
            submitted_expert_route_slots,
            grouped_prefill_routes: prefill_activation_count
                .saturating_mul(self.selected_routes_per_activation),
            skipped_dense_expert_slots: declared_expert_slots
                .saturating_sub(selected_expert_routes),
            empty_shard_route_checks: submitted_expert_route_slots
                .saturating_sub(selected_expert_routes),
            route_weights_device_resident: self.component_count > 0,
            reduction_device_resident: self.component_count > 0,
        }
    }
}

fn sparse_moe_attr_usize(
    node: &CircuitNode,
    attr: &str,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    node.attrs
        .get(attr)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "sparse MoE node {:?} has no valid {attr:?} attribute",
                node.id
            ))
        })
}

#[cfg(test)]
mod sparse_moe_execution_tests {
    use super::*;

    fn independent_node(id: &str, op: &str, inputs: &[&str], outputs: &[&str]) -> CircuitNode {
        CircuitNode {
            id: id.to_string(),
            op: op.to_string(),
            inputs: inputs.iter().map(|value| (*value).to_string()).collect(),
            outputs: outputs.iter().map(|value| (*value).to_string()).collect(),
            params: Vec::new(),
            state_reads: Vec::new(),
            state_writes: Vec::new(),
            attrs: Value::Object(Default::default()),
        }
    }

    #[test]
    fn route_work_reports_only_device_compacted_selected_routes() {
        let contract = VulkanSparseMoeExecutionContract {
            component_count: 40,
            declared_experts_per_activation: 40 * 256,
            selected_routes_per_activation: 40 * 8,
            submitted_route_slots_per_activation: 40 * 8,
        };

        let report = contract.work_report(3, 2);

        assert_eq!(report.activation_count, 5);
        assert_eq!(report.declared_expert_slots, 51_200);
        assert_eq!(report.selected_expert_routes, 1_600);
        assert_eq!(report.submitted_expert_route_slots, 1_600);
        assert_eq!(report.grouped_prefill_routes, 960);
        assert_eq!(report.skipped_dense_expert_slots, 49_600);
        assert_eq!(report.empty_shard_route_checks, 0);
        assert!(report.route_weights_device_resident);
        assert!(report.reduction_device_resident);
    }

    #[test]
    fn dense_models_do_not_claim_sparse_device_work() {
        let report = VulkanSparseMoeExecutionContract::default().work_report(9, 7);

        assert_eq!(report, RuntimeSparseMoeWorkReport::default());
    }

    #[test]
    fn independent_sparse_geometry_counts_routed_and_always_selected_resources() {
        let mut gate_up = independent_node(
            "gate-up",
            "independent_sparse_moe_gate_up",
            &["hidden", "routes"],
            &["intermediates"],
        );
        gate_up.attrs = serde_json::json!({
            "experts_per_token": 7,
            "selected_parameter_accesses": [{
                "execution_signal": "routes",
                "mapping": (0..257).map(|selector| serde_json::json!({
                    "selector": selector,
                    "parameter_ids": [format!("expert_{selector}_gate"), format!("expert_{selector}_up")],
                })).collect::<Vec<_>>(),
            }],
        });

        assert_eq!(sparse_moe_geometry(&gate_up).unwrap(), Some((257, 7)));
    }

    #[test]
    fn independent_sparse_geometry_rejects_noncanonical_resource_mapping() {
        let mut gate_up = independent_node(
            "gate-up",
            "independent_sparse_moe_gate_up",
            &["hidden", "routes"],
            &["intermediates"],
        );
        gate_up.attrs = serde_json::json!({
            "experts_per_token": 2,
            "selected_parameter_accesses": [{
                "execution_signal": "routes",
                "mapping": [
                    {"selector": 0, "parameter_ids": ["expert_0"]},
                    {"selector": 0, "parameter_ids": ["expert_1"]},
                ],
            }],
        });

        let error = sparse_moe_geometry(&gate_up).unwrap_err();
        assert!(error.to_string().contains("exactly once"));
    }

    #[test]
    fn independent_sparse_chain_requires_one_router_down_and_reduction() {
        let mut gate_up = independent_node(
            "gate-up",
            "independent_sparse_moe_gate_up",
            &["hidden", "routes"],
            &["intermediates"],
        );
        gate_up.attrs = serde_json::json!({
            "experts_per_token": 2,
            "selected_parameter_accesses": [{
                "execution_signal": "routes",
                "mapping": [
                    {"selector": 0, "parameter_ids": ["expert_0"]},
                    {"selector": 1, "parameter_ids": ["shared_expert"]},
                ],
            }],
        });
        let nodes = vec![
            independent_node("router", "moe_route", &["logits"], &["routes"]),
            gate_up.clone(),
            independent_node(
                "down",
                "independent_sparse_moe_down",
                &["intermediates", "routes"],
                &["expert_outputs"],
            ),
            independent_node(
                "reduce",
                "moe_reduce",
                &["expert_outputs"],
                &["ffn_output"],
            ),
        ];

        validate_sparse_moe_execution_chain(&nodes, &gate_up).unwrap();

        let mut ambiguous = nodes;
        ambiguous.push(independent_node(
            "second-reduce",
            "moe_reduce",
            &["expert_outputs"],
            &["second_output"],
        ));
        let error = validate_sparse_moe_execution_chain(&ambiguous, &gate_up).unwrap_err();
        assert!(error.to_string().contains("resolves 2 reduction nodes"));
    }

    #[test]
    fn packed_sparse_chain_accepts_the_structural_topk_router_form() {
        let mut gate_up = independent_node(
            "gate-up",
            "sparse_moe_gate_up",
            &["hidden", "routes"],
            &["intermediates"],
        );
        gate_up.attrs = serde_json::json!({
            "num_experts": 64,
            "experts_per_token": 6,
        });
        let nodes = vec![
            independent_node("router", "moe_topk", &["logits"], &["routes"]),
            gate_up.clone(),
            independent_node(
                "down",
                "sparse_moe_down",
                &["intermediates", "routes"],
                &["expert_outputs"],
            ),
            independent_node(
                "reduce",
                "moe_reduce",
                &["expert_outputs"],
                &["ffn_output"],
            ),
        ];

        assert_eq!(sparse_moe_geometry(&gate_up).unwrap(), Some((64, 6)));
        validate_sparse_moe_execution_chain(&nodes, &gate_up).unwrap();
    }
}
