#[derive(Clone, Debug, PartialEq, Eq)]
enum VulkanComponentBatchExecutionScope {
    All,
    Components(BTreeSet<String>),
}

impl VulkanComponentBatchExecutionScope {
    fn all() -> Self {
        Self::All
    }

    fn components(
        component_ids: impl IntoIterator<Item = String>,
    ) -> Result<Self, VulkanError> {
        let component_ids = component_ids.into_iter().collect::<BTreeSet<_>>();
        if component_ids.is_empty() || component_ids.iter().any(|id| id.is_empty()) {
            return Err(VulkanError(
                "component batch execution scope requires non-empty component ids".to_string(),
            ));
        }
        Ok(Self::Components(component_ids))
    }

    fn includes(&self, component_id: &str) -> bool {
        match self {
            Self::All => true,
            Self::Components(component_ids) => component_ids.contains(component_id),
        }
    }

    fn validate_component_ids<'a>(
        &self,
        available: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), VulkanError> {
        let Self::Components(requested) = self else {
            return Ok(());
        };
        let available = available.into_iter().collect::<BTreeSet<_>>();
        let missing = requested
            .iter()
            .filter(|component_id| !available.contains(component_id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(VulkanError(format!(
                "component batch execution scope references absent components {missing:?}"
            )))
        }
    }

    fn filter_distributed_plan(
        &self,
        plan: &VulkanDistributedExecutionPlan,
    ) -> Result<VulkanDistributedExecutionPlan, VulkanError> {
        let mut filtered = plan.clone();
        filtered
            .dispatches
            .retain(|dispatch| self.includes(&dispatch.component_id));
        filtered.dispatch_groups.retain(|group| {
            let included = group
                .dispatches
                .iter()
                .filter(|dispatch| self.includes(&dispatch.component_id))
                .count();
            included == group.dispatches.len()
        });
        if plan.dispatch_groups.iter().any(|group| {
            let included = group
                .dispatches
                .iter()
                .filter(|dispatch| self.includes(&dispatch.component_id))
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
