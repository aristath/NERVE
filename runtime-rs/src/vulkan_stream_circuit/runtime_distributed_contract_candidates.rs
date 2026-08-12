#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanRuntimeDistributedContractCandidate {
    pub contract_ids: BTreeSet<String>,
}

pub fn vulkan_runtime_distributed_contract_candidates(
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<Vec<VulkanRuntimeDistributedContractCandidate>, VulkanRuntimeResidencyPlanError> {
    let execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == target.component_id)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "distributed contract discovery found no execution for component {:?}",
                target.component_id,
            ))
        })?;
    let (execution_phase, execution_shape) = match phase {
        VulkanTargetedComponentExecutionPhase::Decode => (
            nerve_execution_contracts::ExecutionPhase::Decode,
            nerve_execution_contracts::ExecutionShape::SingleLane,
        ),
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        } => {
            if activation_batch_width == 0 {
                return Err(VulkanRuntimeResidencyPlanError(
                    "distributed contract discovery requires a positive prefill batch width"
                        .to_string(),
                ));
            }
            (
                nerve_execution_contracts::ExecutionPhase::Prefill,
                nerve_execution_contracts::ExecutionShape::MultiLane,
            )
        }
    };
    let mut alternatives = execution
        .kernels
        .iter()
        .filter_map(|kernel| {
            let mut contracts = kernel
                .physical_execution_contracts
                .iter()
                .filter(|contract| {
                    contract.strategy.is_distributed()
                        && contract.phases.contains(&execution_phase)
                        && contract.execution_shape.supports(execution_shape)
                        && contract.operation_family == kernel.op
                        && contract.member_node_ids.contains(&kernel.node_id)
                })
                .collect::<Vec<_>>();
            contracts.sort_by(|left, right| left.contract_id.cmp(&right.contract_id));
            (!contracts.is_empty()).then_some(contracts)
        })
        .collect::<Vec<_>>();
    if alternatives.is_empty() {
        return Ok(Vec::new());
    }
    alternatives.sort_by(|left, right| {
        left[0]
            .member_node_ids
            .cmp(&right[0].member_node_ids)
            .then_with(|| left[0].operation_family.cmp(&right[0].operation_family))
    });

    let mut selected = Vec::with_capacity(alternatives.len());
    let mut candidates = BTreeSet::new();
    enumerate_distributed_contract_candidates(
        &alternatives,
        0,
        &mut selected,
        &mut candidates,
    )?;
    Ok(candidates
        .into_iter()
        .map(|contract_ids| VulkanRuntimeDistributedContractCandidate { contract_ids })
        .collect())
}

fn enumerate_distributed_contract_candidates<'a>(
    alternatives: &[Vec<&'a nerve_execution_contracts::PhysicalExecutionContract>],
    index: usize,
    selected: &mut Vec<&'a nerve_execution_contracts::PhysicalExecutionContract>,
    candidates: &mut BTreeSet<BTreeSet<String>>,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if index == alternatives.len() {
        if !selected.is_empty() && selected_contracts_have_complete_local_handoffs(selected)? {
            candidates.insert(
                selected
                    .iter()
                    .map(|contract| contract.contract_id.clone())
                    .collect(),
            );
        }
        return Ok(());
    }
    // Leaving this kernel on its canonical implementation is a first-class
    // hybrid choice. Without this branch candidate discovery can only measure
    // the product of every distributable kernel in the component, which turns
    // a local physical option into an accidental component-wide TP switch.
    enumerate_distributed_contract_candidates(
        alternatives,
        index + 1,
        selected,
        candidates,
    )?;
    for contract in &alternatives[index] {
        contract.validate().map_err(|error| {
            VulkanRuntimeResidencyPlanError(format!(
                "distributed contract candidate {:?} is invalid: {error}",
                contract.contract_id,
            ))
        })?;
        selected.push(contract);
        enumerate_distributed_contract_candidates(
            alternatives,
            index + 1,
            selected,
            candidates,
        )?;
        selected.pop();
    }
    Ok(())
}

fn selected_contracts_have_complete_local_handoffs(
    selected: &[&nerve_execution_contracts::PhysicalExecutionContract],
) -> Result<bool, VulkanRuntimeResidencyPlanError> {
    let mut declarations = BTreeMap::<
        (String, u32, u32, String),
        usize,
    >::new();
    for contract in selected {
        for local in &contract.local_intermediates {
            let key = (
                local.signal.clone(),
                local.producer_binding,
                local.consumer_binding,
                local.format.clone(),
            );
            let count = declarations.entry(key).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(
                    "distributed local-handoff declaration count overflowed".to_string(),
                )
            })?;
        }
    }
    Ok(declarations.values().all(|count| *count == 2))
}

#[cfg(test)]
mod runtime_distributed_contract_candidate_tests {
    use super::*;

    #[test]
    fn discovers_only_complete_compiler_contract_choices() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let target = vulkan_runtime_placement_calibration_targets(&model)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let candidates = vulkan_runtime_distributed_contract_candidates(
            &model,
            &target,
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();

        assert!(!candidates.is_empty());
        assert!(
            candidates
                .iter()
                .all(|candidate| !candidate.contract_ids.is_empty())
        );
        assert_eq!(
            candidates.iter().collect::<BTreeSet<_>>().len(),
            candidates.len(),
        );
    }

    #[test]
    fn rejects_zero_width_prefill_before_enumerating_contracts() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let target = vulkan_runtime_placement_calibration_targets(&model)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        assert!(
            vulkan_runtime_distributed_contract_candidates(
                &model,
                &target,
                VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width: 0,
                },
            )
            .unwrap_err()
            .0
            .contains("positive prefill batch width")
        );
    }

    #[test]
    fn local_physical_layout_requires_both_handoff_contracts() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let mut producer = model
            .component_executions
            .iter()
            .flat_map(|execution| &execution.kernels)
            .flat_map(|kernel| &kernel.physical_execution_contracts)
            .find(|contract| contract.strategy.is_distributed())
            .expect("fixture has a distributed contract")
            .clone();
        producer.local_intermediates = vec![
            nerve_execution_contracts::LocalIntermediateContract {
                signal: "private-shard".to_string(),
                producer_binding: 1,
                consumer_binding: 0,
                format: "bf16:private-layout".to_string(),
            },
        ];
        let mut consumer = producer.clone();
        consumer.contract_id = format!("sha256:{}", "f".repeat(64));

        assert!(!selected_contracts_have_complete_local_handoffs(&[&producer]).unwrap());
        assert!(
            selected_contracts_have_complete_local_handoffs(&[&producer, &consumer]).unwrap()
        );
    }

    #[test]
    fn enumerates_selective_and_combined_distributed_islands() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let contracts = model
            .component_executions
            .iter()
            .flat_map(|execution| &execution.kernels)
            .flat_map(|kernel| &kernel.physical_execution_contracts)
            .filter(|contract| contract.strategy.is_distributed())
            .take(2)
            .collect::<Vec<_>>();
        let [first, second] = contracts.as_slice() else {
            panic!("fixture must expose two independent distributed alternatives")
        };
        assert!(first.local_intermediates.is_empty());
        assert!(second.local_intermediates.is_empty());

        let alternatives = vec![vec![*first], vec![*second]];
        let mut selected = Vec::new();
        let mut candidates = BTreeSet::new();
        enumerate_distributed_contract_candidates(
            &alternatives,
            0,
            &mut selected,
            &mut candidates,
        )
        .unwrap();

        assert_eq!(
            candidates,
            BTreeSet::from([
                BTreeSet::from([first.contract_id.clone()]),
                BTreeSet::from([second.contract_id.clone()]),
                BTreeSet::from([first.contract_id.clone(), second.contract_id.clone()]),
            ]),
        );
        assert!(!candidates.contains(&BTreeSet::new()));
    }
}
