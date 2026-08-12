#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRuntimePhysicalExecutionPlan {
    pub component_device_pools: VulkanDistributedPhaseComponentDevicePools,
    pub decode_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub decode_batch_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    pub prefill_execution_cases_by_component:
        BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
}

impl VulkanRuntimePhysicalExecutionPlan {
    pub fn uniform(runtime_model: &VulkanResidentRuntimeModel) -> Self {
        let signal_processor_placement = runtime_model
            .circuit_graph
            .signal_processor_placement(&runtime_model.placement);
        Self {
            component_device_pools: VulkanDistributedPhaseComponentDevicePools::uniform(
                &signal_processor_placement.component_shard_devices,
            ),
            ..Self::default()
        }
    }

    pub fn validate(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        let component_ids = runtime_model
            .circuit_graph
            .components
            .iter()
            .filter(|component| component.runtime_role.is_signal_processor())
            .map(|component| component.component_id.as_str())
            .collect::<BTreeSet<_>>();
        if component_ids.is_empty() {
            return runtime_hybrid_error(
                "physical execution plan requires at least one signal processor",
            );
        }

        self.validate_phase_pools(
            runtime_model,
            "decode",
            &self.component_device_pools.decode,
            &component_ids,
        )?;
        self.validate_phase_pools(
            runtime_model,
            "decode_batch",
            &self.component_device_pools.decode_batch,
            &component_ids,
        )?;
        self.validate_phase_pools(
            runtime_model,
            "prefill",
            &self.component_device_pools.prefill,
            &component_ids,
        )?;
        self.validate_exact_phase_cases(
            "decode",
            nerve_execution_contracts::ExecutionPhase::Decode,
            Some(1),
            &component_ids,
            &self.component_device_pools.decode,
            &self.decode_execution_cases_by_component,
        )?;
        self.validate_exact_phase_cases(
            "decode_batch",
            nerve_execution_contracts::ExecutionPhase::Decode,
            None,
            &component_ids,
            &self.component_device_pools.decode_batch,
            &self.decode_batch_execution_cases_by_component,
        )?;
        self.validate_exact_phase_cases(
            "prefill",
            nerve_execution_contracts::ExecutionPhase::Prefill,
            None,
            &component_ids,
            &self.component_device_pools.prefill,
            &self.prefill_execution_cases_by_component,
        )?;
        Ok(())
    }

    pub fn device_ids(&self, runtime_model: &VulkanResidentRuntimeModel) -> Vec<String> {
        runtime_model
            .placement_device_ids()
            .into_iter()
            .chain(
                [
                    &self.component_device_pools.decode,
                    &self.component_device_pools.decode_batch,
                    &self.component_device_pools.prefill,
                ]
                .into_iter()
                .flat_map(|pools| pools.values().flatten().cloned()),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn validate_phase_pools(
        &self,
        runtime_model: &VulkanResidentRuntimeModel,
        phase: &str,
        pools: &BTreeMap<String, Vec<String>>,
        component_ids: &BTreeSet<&str>,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        for (component_id, device_ids) in pools {
            if !component_ids.contains(component_id.as_str()) {
                return runtime_hybrid_error(format!(
                    "physical {phase} plan references non-signal-processor component {component_id:?}",
                ));
            }
            if device_ids.len() < 2
                || device_ids.iter().any(String::is_empty)
                || device_ids.iter().collect::<BTreeSet<_>>().len() != device_ids.len()
            {
                return runtime_hybrid_error(format!(
                    "physical {phase} shard pool for component {component_id:?} requires at least two distinct nonempty devices",
                ));
            }
            let owner = runtime_model.placement.device_for_component(component_id);
            if device_ids.first().map(String::as_str) != Some(owner) {
                return runtime_hybrid_error(format!(
                    "physical {phase} shard pool for component {component_id:?} must begin with stable owner {owner:?}",
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_exact_phase_cases(
        &self,
        phase_name: &str,
        execution_phase: nerve_execution_contracts::ExecutionPhase,
        exact_batch_width: Option<usize>,
        component_ids: &BTreeSet<&str>,
        pools: &BTreeMap<String, Vec<String>>,
        cases: &BTreeMap<String, VulkanPlacementExecutionCaseIdentity>,
    ) -> Result<(), VulkanRuntimeHybridPlacementError> {
        if cases.is_empty() {
            return Ok(());
        }
        if cases.keys().map(String::as_str).collect::<BTreeSet<_>>() != *component_ids {
            return runtime_hybrid_error(format!(
                "exact physical {phase_name} plan must cover every signal processor exactly once",
            ));
        }
        for (component_id, case) in cases {
            let batch_width = case.behavior.shape.activation_batch_width;
            if case.behavior.phase != execution_phase
                || exact_batch_width.is_some_and(|expected| batch_width != expected)
                || exact_batch_width.is_none() && batch_width < 2
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} has incompatible phase geometry",
                ));
            }
            let distributed = case.strategy != VulkanPlacementExecutionStrategy::SingleDevice;
            if case.strategy == VulkanPlacementExecutionStrategy::Serialized
                || distributed != pools.contains_key(component_id)
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} disagrees with its phase-local shard pool",
                ));
            }
            if let Some(pool) = pools.get(component_id)
                && pool.len() != case.devices.len()
            {
                return runtime_hybrid_error(format!(
                    "exact physical {phase_name} case for component {component_id:?} has {} participants but its logical pool has {}",
                    case.devices.len(),
                    pool.len(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod runtime_physical_execution_plan_tests {
    use super::*;

    #[test]
    fn uniform_physical_plan_preserves_manual_shards_for_every_phase() {
        let canonical = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let component_id = canonical
            .circuit_graph
            .components
            .iter()
            .find(|component| component.runtime_role.is_signal_processor())
            .unwrap()
            .component_id
            .clone();
        let model = canonical
            .with_component_shard_devices(
                &component_id,
                vec!["gpu0".to_string(), "gpu1".to_string()],
            )
            .unwrap();
        let plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model);

        plan.validate(&model).unwrap();
        assert_eq!(
            plan.component_device_pools.decode,
            plan.component_device_pools.prefill
        );
        assert_eq!(plan.device_ids(&model), ["gpu0", "gpu1"]);
    }

    #[test]
    fn physical_plan_rejects_pool_that_is_not_rooted_at_stable_owner() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let component_id = model
            .circuit_graph
            .components
            .iter()
            .find(|component| component.runtime_role.is_signal_processor())
            .unwrap()
            .component_id
            .clone();
        let mut plan = VulkanRuntimePhysicalExecutionPlan::uniform(&model);
        plan.component_device_pools.decode.insert(
            component_id,
            vec!["gpu1".to_string(), "gpu0".to_string()],
        );

        assert!(
            plan.validate(&model)
                .unwrap_err()
                .0
                .contains("must begin with stable owner")
        );
    }
}
