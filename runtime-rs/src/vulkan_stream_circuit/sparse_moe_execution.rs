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
            let shard_count = self
                .placement
                .component_shard_devices
                .get(&component.component_id)
                .map_or(1, Vec::len);
            for node in &component.circuit.nodes {
                if node.op != "sparse_moe_gate_up" {
                    continue;
                }
                let declared = sparse_moe_attr_usize(node, "num_experts")?;
                let selected = sparse_moe_attr_usize(node, "experts_per_token")?;
                if selected == 0 || selected > declared {
                    return Err(VulkanResidentTokenModelPackageError::new(format!(
                        "sparse MoE node {}.{} selects {selected} routes from {declared} experts",
                        component.component_id, node.id
                    )));
                }
                contract.component_count = contract.component_count.saturating_add(1);
                contract.declared_experts_per_activation = contract
                    .declared_experts_per_activation
                    .saturating_add(declared);
                contract.selected_routes_per_activation = contract
                    .selected_routes_per_activation
                    .saturating_add(selected);
                contract.submitted_route_slots_per_activation = contract
                    .submitted_route_slots_per_activation
                    .saturating_add(selected.saturating_mul(shard_count));
            }
        }
        Ok(contract)
    }
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

    #[test]
    fn route_work_reports_selected_scaling_and_shard_predication() {
        let contract = VulkanSparseMoeExecutionContract {
            component_count: 40,
            declared_experts_per_activation: 40 * 256,
            selected_routes_per_activation: 40 * 8,
            submitted_route_slots_per_activation: 40 * 8 * 2,
        };

        let report = contract.work_report(3, 2);

        assert_eq!(report.activation_count, 5);
        assert_eq!(report.declared_expert_slots, 51_200);
        assert_eq!(report.selected_expert_routes, 1_600);
        assert_eq!(report.submitted_expert_route_slots, 3_200);
        assert_eq!(report.grouped_prefill_routes, 960);
        assert_eq!(report.skipped_dense_expert_slots, 49_600);
        assert_eq!(report.empty_shard_route_checks, 1_600);
        assert!(report.route_weights_device_resident);
        assert!(report.reduction_device_resident);
    }

    #[test]
    fn dense_models_do_not_claim_sparse_device_work() {
        let report = VulkanSparseMoeExecutionContract::default().work_report(9, 7);

        assert_eq!(report, RuntimeSparseMoeWorkReport::default());
    }
}
