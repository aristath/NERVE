#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanRuntimeDistributedContractCandidate {
    pub contract_ids: BTreeSet<String>,
}

#[derive(Clone, Copy)]
struct VulkanRuntimeDistributedContractChoice<'a> {
    contract: &'a nerve_execution_contracts::PhysicalExecutionContract,
    execution_index: usize,
    input_signals: &'a [String],
    output_signals: &'a [String],
}

pub fn vulkan_runtime_distributed_contract_candidates(
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<Vec<VulkanRuntimeDistributedContractCandidate>, VulkanRuntimeResidencyPlanError> {
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
    vulkan_runtime_distributed_contract_candidates_for_execution(
        runtime_model,
        &target.component_id,
        execution_phase,
        execution_shape,
    )
}

pub fn vulkan_runtime_distributed_contract_candidates_for_execution(
    runtime_model: &VulkanResidentRuntimeModel,
    component_id: &str,
    execution_phase: nerve_execution_contracts::ExecutionPhase,
    execution_shape: nerve_execution_contracts::ExecutionShape,
) -> Result<Vec<VulkanRuntimeDistributedContractCandidate>, VulkanRuntimeResidencyPlanError> {
    if component_id.is_empty() {
        return Err(VulkanRuntimeResidencyPlanError(
            "distributed contract discovery requires a component ID".to_string(),
        ));
    }
    let execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == component_id)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "distributed contract discovery found no execution for component {:?}",
                component_id,
            ))
        })?;
    let component = runtime_model
        .circuit_graph
        .components
        .iter()
        .find(|component| component.component_id == component_id)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "distributed contract discovery found no circuit for component {:?}",
                component_id,
            ))
        })?;
    let mut alternatives = execution
        .kernels
        .iter()
        .map(|kernel| {
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
            if contracts.is_empty() {
                return Ok(None);
            }
            let node = component
                .circuit
                .nodes
                .iter()
                .find(|node| node.id == kernel.node_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "distributed contract discovery found no circuit node for kernel {}.{}",
                        component_id, kernel.node_id,
                    ))
                })?;
            Ok::<_, VulkanRuntimeResidencyPlanError>(Some(
                contracts
                    .into_iter()
                    .map(|contract| VulkanRuntimeDistributedContractChoice {
                        contract,
                        execution_index: kernel.execution_index,
                        input_signals: &node.inputs,
                        output_signals: &node.outputs,
                    })
                    .collect::<Vec<_>>(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if alternatives.is_empty() {
        return Ok(Vec::new());
    }
    alternatives.sort_by(|left, right| {
        left[0]
            .contract
            .member_node_ids
            .cmp(&right[0].contract.member_node_ids)
            .then_with(|| {
                left[0]
                    .contract
                    .operation_family
                    .cmp(&right[0].contract.operation_family)
            })
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
    alternatives: &[Vec<VulkanRuntimeDistributedContractChoice<'a>>],
    index: usize,
    selected: &mut Vec<VulkanRuntimeDistributedContractChoice<'a>>,
    candidates: &mut BTreeSet<BTreeSet<String>>,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    if index == alternatives.len() {
        if !selected.is_empty() && selected_contracts_have_complete_local_handoffs(selected)? {
            candidates.insert(
                selected
                    .iter()
                    .map(|choice| choice.contract.contract_id.clone())
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
    for choice in &alternatives[index] {
        choice.contract.validate().map_err(|error| {
            VulkanRuntimeResidencyPlanError(format!(
                "distributed contract candidate {:?} is invalid: {error}",
                choice.contract.contract_id,
            ))
        })?;
        selected.push(*choice);
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
    selected: &[VulkanRuntimeDistributedContractChoice<'_>],
) -> Result<bool, VulkanRuntimeResidencyPlanError> {
    let mut declarations = BTreeMap::<
        (String, u32, u32, String),
        (Vec<(usize, usize)>, Vec<(usize, usize)>),
    >::new();
    for (choice_index, choice) in selected.iter().enumerate() {
        for local in &choice.contract.local_intermediates {
            let key = (
                local.signal.clone(),
                local.producer_binding,
                local.consumer_binding,
                local.format.clone(),
            );
            let is_producer = choice.output_signals.contains(&local.signal)
                && choice
                    .contract
                    .outputs
                    .iter()
                    .any(|output| output.binding == local.producer_binding);
            let is_consumer = choice.input_signals.contains(&local.signal)
                && choice
                    .contract
                    .inputs
                    .iter()
                    .any(|input| input.binding == local.consumer_binding);
            if !is_producer && !is_consumer {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed contract {:?} declares local signal {:?} without producing or consuming its typed binding",
                    choice.contract.contract_id, local.signal,
                )));
            }
            let (producers, consumers) = declarations.entry(key).or_default();
            if is_producer {
                producers.push((choice_index, choice.execution_index));
            }
            if is_consumer {
                consumers.push((choice_index, choice.execution_index));
            }
        }
    }
    Ok(declarations.values().all(|(producers, consumers)| {
        matches!(
            (producers.as_slice(), consumers.as_slice()),
            ([(producer_choice, producer_index)], [(consumer_choice, consumer_index)])
                if producer_choice != consumer_choice
                    && producer_index.checked_add(1) == Some(*consumer_index)
        )
    }))
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
    fn compiled_dense_ffn_handoff_is_an_atomic_decode_and_prefill_candidate() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let target = vulkan_runtime_placement_calibration_targets(&model)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let contracts = model
            .component_executions
            .iter()
            .flat_map(|execution| &execution.kernels)
            .flat_map(|kernel| &kernel.physical_execution_contracts)
            .filter(|contract| {
                contract.strategy.is_distributed()
                    && contract
                        .local_intermediates
                        .iter()
                        .any(|local| local.signal == "ffn_hidden" && local.format == "bf16")
            })
            .collect::<Vec<_>>();
        let gate_contract_ids = contracts
            .iter()
            .filter(|contract| {
                contract.operation_family == "parallel_linear_silu_multiply"
            })
            .map(|contract| contract.contract_id.clone())
            .collect::<BTreeSet<_>>();
        let down_contract_ids = contracts
            .iter()
            .filter(|contract| contract.operation_family == "linear_residual")
            .map(|contract| contract.contract_id.clone())
            .collect::<BTreeSet<_>>();

        assert!(!gate_contract_ids.is_empty());
        assert_eq!(down_contract_ids.len(), 1);
        let down = contracts
            .iter()
            .find(|contract| contract.operation_family == "linear_residual")
            .unwrap();
        assert!(matches!(
            down.execution_form,
            nerve_execution_contracts::ExecutionForm::PartitionedInputPartialOutput
        ));
        assert_eq!(down.formats.accumulation, "f32");
        let reduction = down.outputs[0]
            .reduction
            .as_ref()
            .expect("dense down contract must publish F32 partials");
        assert!(matches!(
            reduction.operation,
            nerve_execution_contracts::ReductionOperation::SumF32
        ));
        assert!(matches!(
            reduction.finalization,
            nerve_execution_contracts::ReductionFinalization::AddBf16ResidualToBf16 {
                residual_binding: 1
            }
        ));

        for phase in [
            VulkanTargetedComponentExecutionPhase::Decode,
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 4,
            },
        ] {
            let candidates = vulkan_runtime_distributed_contract_candidates(
                &model,
                &target,
                phase,
            )
            .unwrap();
            let mut complete_island_found = false;
            for candidate in candidates {
                let has_gate = candidate
                    .contract_ids
                    .iter()
                    .any(|contract_id| gate_contract_ids.contains(contract_id));
                let has_down = candidate
                    .contract_ids
                    .iter()
                    .any(|contract_id| down_contract_ids.contains(contract_id));
                assert_eq!(
                    has_gate, has_down,
                    "private dense FFN handoff cannot select only one side"
                );
                complete_island_found |= has_gate;
            }
            assert!(
                complete_island_found,
                "compiled package must expose the complete dense FFN island for {phase:?}"
            );
        }
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
        let producer_inputs = vec!["normalized".to_string()];
        let producer_outputs = vec!["private-shard".to_string()];
        let consumer_inputs = vec!["private-shard".to_string()];
        let consumer_outputs = vec!["hidden".to_string()];
        let producer = VulkanRuntimeDistributedContractChoice {
            contract: &producer,
            execution_index: 4,
            input_signals: &producer_inputs,
            output_signals: &producer_outputs,
        };
        let consumer = VulkanRuntimeDistributedContractChoice {
            contract: &consumer,
            execution_index: 5,
            input_signals: &consumer_inputs,
            output_signals: &consumer_outputs,
        };

        assert!(!selected_contracts_have_complete_local_handoffs(&[producer]).unwrap());
        assert!(
            selected_contracts_have_complete_local_handoffs(&[producer, consumer]).unwrap()
        );
    }

    #[test]
    fn local_handoff_rejects_duplicate_producers_and_nonadjacent_consumers() {
        let model = tests::tiny_fixture_model_runtime_model_with_placement(
            StreamCircuitPlacementSpec::new("gpu0"),
        );
        let mut first = model
            .component_executions
            .iter()
            .flat_map(|execution| &execution.kernels)
            .flat_map(|kernel| &kernel.physical_execution_contracts)
            .find(|contract| contract.strategy.is_distributed())
            .expect("fixture has a distributed contract")
            .clone();
        first.local_intermediates = vec![
            nerve_execution_contracts::LocalIntermediateContract {
                signal: "private-shard".to_string(),
                producer_binding: 1,
                consumer_binding: 0,
                format: "bf16:private-layout".to_string(),
            },
        ];
        let mut second = first.clone();
        second.contract_id = format!("sha256:{}", "e".repeat(64));
        let producer_inputs = vec!["normalized".to_string()];
        let producer_outputs = vec!["private-shard".to_string()];
        let consumer_inputs = vec!["private-shard".to_string()];
        let consumer_outputs = vec!["hidden".to_string()];
        let producer = VulkanRuntimeDistributedContractChoice {
            contract: &first,
            execution_index: 4,
            input_signals: &producer_inputs,
            output_signals: &producer_outputs,
        };
        let duplicate_producer = VulkanRuntimeDistributedContractChoice {
            contract: &second,
            execution_index: 5,
            input_signals: &producer_inputs,
            output_signals: &producer_outputs,
        };
        let nonadjacent_consumer = VulkanRuntimeDistributedContractChoice {
            contract: &second,
            execution_index: 6,
            input_signals: &consumer_inputs,
            output_signals: &consumer_outputs,
        };

        assert!(
            !selected_contracts_have_complete_local_handoffs(&[
                producer,
                duplicate_producer,
            ])
            .unwrap()
        );
        assert!(
            !selected_contracts_have_complete_local_handoffs(&[
                producer,
                nonadjacent_consumer,
            ])
            .unwrap()
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

        let inputs = Vec::<String>::new();
        let outputs = Vec::<String>::new();
        let alternatives = vec![
            vec![VulkanRuntimeDistributedContractChoice {
                contract: first,
                execution_index: 0,
                input_signals: &inputs,
                output_signals: &outputs,
            }],
            vec![VulkanRuntimeDistributedContractChoice {
                contract: second,
                execution_index: 1,
                input_signals: &inputs,
                output_signals: &outputs,
            }],
        ];
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
