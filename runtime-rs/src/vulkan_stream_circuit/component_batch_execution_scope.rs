#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionScope {
    All,
    Components(BTreeSet<String>),
    Nodes(BTreeMap<String, BTreeSet<String>>),
}

impl VulkanComponentBatchExecutionScope {
    fn all() -> Self {
        Self::All
    }

    fn nodes(
        node_ids_by_component: BTreeMap<String, BTreeSet<String>>,
    ) -> Result<Self, VulkanError> {
        if node_ids_by_component.is_empty()
            || node_ids_by_component.iter().any(|(component_id, node_ids)| {
                component_id.is_empty()
                    || node_ids.is_empty()
                    || node_ids.iter().any(String::is_empty)
            })
        {
            return Err(VulkanError(
                "component batch execution scope requires non-empty component and node ids"
                    .to_string(),
            ));
        }
        Ok(Self::Nodes(node_ids_by_component))
    }

    fn components(component_ids: BTreeSet<String>) -> Result<Self, VulkanError> {
        if component_ids.is_empty() || component_ids.iter().any(String::is_empty) {
            return Err(VulkanError(
                "component batch execution scope requires non-empty component ids".to_string(),
            ));
        }
        Ok(Self::Components(component_ids))
    }

    fn includes_component(&self, component_id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Components(component_ids) => component_ids.contains(component_id),
            Self::Nodes(node_ids_by_component) => {
                node_ids_by_component.contains_key(component_id)
            }
        }
    }

    fn allows_open_boundaries(&self) -> bool {
        !matches!(self, Self::All)
    }

    fn includes_dispatch(&self, component_id: &str, node_id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Components(component_ids) => component_ids.contains(component_id),
            Self::Nodes(node_ids_by_component) => node_ids_by_component
                .get(component_id)
                .is_some_and(|node_ids| node_ids.contains(node_id)),
        }
    }

    fn validate_component_ids<'a>(
        &self,
        available: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), VulkanError> {
        let available = available.into_iter().collect::<BTreeSet<_>>();
        match self {
            Self::All => return Ok(()),
            Self::Components(requested) => {
                let missing = requested
                    .iter()
                    .filter(|component_id| !available.contains(component_id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                return if missing.is_empty() {
                    Ok(())
                } else {
                    Err(VulkanError(format!(
                        "component batch execution scope references absent components {missing:?}"
                    )))
                };
            }
            Self::Nodes(requested) => {
                let missing = requested
                    .keys()
                    .filter(|component_id| !available.contains(component_id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                return if missing.is_empty() {
                    Ok(())
                } else {
                    Err(VulkanError(format!(
                        "component batch execution scope references absent components {missing:?}"
                    )))
                };
            }
        }
    }

    fn validate_dispatch_ids<'a>(
        &self,
        available: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<(), VulkanError> {
        let Self::Nodes(requested) = self else {
            return Ok(());
        };
        let available = available.into_iter().collect::<BTreeSet<_>>();
        let mut missing = Vec::new();
        for (component_id, node_ids) in requested {
            for node_id in node_ids {
                if !available.contains(&(component_id.as_str(), node_id.as_str())) {
                    missing.push(format!("{component_id}.{node_id}"));
                }
            }
        }
        if missing.is_empty() {
            Ok(())
        } else {
            Err(VulkanError(format!(
                "component batch execution scope references absent dispatches {missing:?}"
            )))
        }
    }

    fn filter_distributed_plan(
        &self,
        plan: &VulkanDistributedExecutionPlan,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanError> {
        if matches!(self, Self::Nodes(_)) && !plan.dispatches.is_empty() {
            return Err(VulkanError(
                "node-scoped component batch execution cannot split distributed dispatches"
                    .to_string(),
            ));
        }
        let mut filtered = plan.clone();
        filtered
            .dispatches
            .retain(|dispatch| self.includes_component(&dispatch.component_id));
        filtered.execution_islands.retain(|group| {
            let included = group
                .dispatches
                .iter()
                .filter(|dispatch| self.includes_component(&dispatch.component_id))
                .count();
            included == group.dispatches.len()
        });
        if plan.execution_islands.iter().any(|group| {
            let included = group
                .dispatches
                .iter()
                .filter(|dispatch| self.includes_component(&dispatch.component_id))
                .count();
            included > 0 && included < group.dispatches.len()
        }) {
            return Err(VulkanError(
                "component batch execution scope splits a distributed dispatch group"
                    .to_string(),
            ));
        }
        filtered.device_ids = filtered
            .dispatches
            .iter()
            .flat_map(|dispatch| dispatch.shards.iter().map(|shard| shard.device_id.clone()))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        filtered.distributed_parameter_byte_count = filtered.dispatches.iter().try_fold(
            0usize,
            |total, dispatch| {
                total
                    .checked_add(dispatch.distributed_parameter_byte_count)
                    .ok_or_else(|| {
                        VulkanError(
                            "component batch distributed parameter bytes overflowed"
                                .to_string(),
                        )
                    })
            },
        )?;
        filtered.shared_input_byte_capacity = filtered
            .dispatches
            .first()
            .map(|dispatch| dispatch.input_byte_capacity)
            .unwrap_or(0);
        filtered.shared_output_byte_capacity = filtered
            .dispatches
            .last()
            .map(|dispatch| dispatch.output_byte_capacity)
            .unwrap_or(0);
        Ok(filtered)
    }
}

#[cfg(test)]
mod component_batch_execution_scope_tests {
    use super::*;

    #[test]
    fn complete_component_scope_selects_every_node_in_requested_components() {
        let scope = VulkanComponentBatchExecutionScope::components(BTreeSet::from([
            "layer_1".to_string(),
            "layer_2".to_string(),
        ]))
        .unwrap();

        assert!(scope.includes_dispatch("layer_1", "attention"));
        assert!(scope.includes_dispatch("layer_1", "mlp"));
        assert!(scope.includes_dispatch("layer_2", "attention"));
        assert!(!scope.includes_dispatch("layer_0", "mlp"));
        assert!(
            scope
                .validate_component_ids(["layer_0", "layer_1", "layer_2"])
                .is_ok()
        );
        assert!(
            scope
                .validate_dispatch_ids([
                    ("layer_1", "attention"),
                    ("layer_2", "attention"),
                ])
                .is_ok()
        );
    }

    #[test]
    fn complete_component_scope_rejects_empty_or_absent_components() {
        assert!(VulkanComponentBatchExecutionScope::components(BTreeSet::new()).is_err());
        assert!(VulkanComponentBatchExecutionScope::components(BTreeSet::from([
            String::new()
        ]))
        .is_err());

        let scope = VulkanComponentBatchExecutionScope::components(BTreeSet::from([
            "layer_1".to_string(),
            "layer_2".to_string(),
        ]))
        .unwrap();
        let error = scope
            .validate_component_ids(["layer_0", "layer_1"])
            .unwrap_err();

        assert!(error.0.contains("layer_2"));
    }

    #[test]
    fn node_scope_remains_distinct_from_complete_component_scope() {
        let scope = VulkanComponentBatchExecutionScope::nodes(BTreeMap::from([(
            "layer_1".to_string(),
            BTreeSet::from(["attention".to_string()]),
        )]))
        .unwrap();

        assert!(scope.includes_dispatch("layer_1", "attention"));
        assert!(!scope.includes_dispatch("layer_1", "mlp"));
    }
}
