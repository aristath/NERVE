#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionScope {
    All,
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

    fn includes_component(&self, component_id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Nodes(node_ids_by_component) => {
                node_ids_by_component.contains_key(component_id)
            }
        }
    }

    fn includes_dispatch(&self, component_id: &str, node_id: &str) -> bool {
        match self {
            Self::All => true,
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
        Ok(filtered)
    }
}
