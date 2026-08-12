use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::Path;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanPlacementExecutionStrategy,
    VulkanRuntimeDistributedPlacementCalibrationReport,
};

use crate::boundary_calibration::measure_boundary_candidate;
use crate::calibration_package::CalibrationPackage;
use crate::calibration_suite_plan::{expand_target_orders, plan_calibration_suite};
use crate::load_wave_calibration::measure_load_wave_candidate;
use crate::output::write_atomic;
use crate::package_calibration::measure_package_candidates;

#[derive(Clone, Debug, PartialEq, Eq)]
struct MeasuredTargetOrder {
    order: Vec<String>,
    duration_ns: u64,
    owner_target_id: String,
    output_target_id: String,
    resident_bytes: BTreeMap<String, usize>,
    transient_bytes: BTreeMap<String, usize>,
    host_transient_bytes: usize,
    contract_ids: Vec<String>,
    strategy: VulkanPlacementExecutionStrategy,
}

pub fn run_calibration_suite(
    package_path: &Path,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package_path)?;
    package.reject_output_collision(output)?;
    let plan = plan_calibration_suite(
        package.runtime_model(),
        target_ids,
        prefill_widths,
        maximum_group_size,
    )?;
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut measured_component_candidates = 0usize;
    let mut unavailable_component_candidates = 0usize;

    for case in &plan.component_cases {
        let mut current_width_measurements = Vec::new();
        for order in &plan.initial_target_orders {
            let measurement =
                measure_package_candidates(&package, &case.target, case.phase, order)?;
            if measurement.reports.is_empty() {
                unavailable_component_candidates += 1;
                continue;
            }
            catalog.merge(&measurement.catalog)?;
            measured_component_candidates += measurement.reports.len();
            if order.len() == 2 {
                current_width_measurements
                    .extend(measurement.reports.iter().map(measured_target_order));
            }
        }

        let mut width = 2usize;
        while width < plan.maximum_group_size && !current_width_measurements.is_empty() {
            let promising = non_dominated_target_orders(&current_width_measurements);
            let expanded = expand_target_orders(&promising, target_ids, plan.maximum_group_size)?;
            let mut next_width_measurements = Vec::new();
            for order in expanded {
                let measurement =
                    measure_package_candidates(&package, &case.target, case.phase, &order)?;
                if measurement.reports.is_empty() {
                    unavailable_component_candidates += 1;
                    continue;
                }
                catalog.merge(&measurement.catalog)?;
                measured_component_candidates += measurement.reports.len();
                next_width_measurements
                    .extend(measurement.reports.iter().map(measured_target_order));
            }
            current_width_measurements = next_width_measurements;
            width += 1;
        }
    }

    for case in &plan.boundary_cases {
        let measured = measure_boundary_candidate(
            &package,
            case.phase,
            &case.source_target_id,
            &case.destination_target_id,
        )?;
        catalog.merge(&measured)?;
    }

    for case in &plan.load_wave_cases {
        for target_id in target_ids {
            let measured = measure_load_wave_candidate(
                &package,
                &case.component_id,
                &case.selector_id,
                case.phase,
                &case.resource_indices,
                target_id,
            )?;
            catalog.merge(&measured.catalog)?;
        }
    }

    catalog.validate()?;
    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package suite: package={}, targets={}, component_cases={}, measured_component_candidates={}, unavailable_component_candidates={}, boundary_cases={}, load_wave_cases={}, references={}, observations={}, output={}",
        package.source_path().display(),
        target_ids.len(),
        plan.component_cases.len(),
        measured_component_candidates,
        unavailable_component_candidates,
        plan.boundary_cases.len(),
        plan.load_wave_cases.len().saturating_mul(target_ids.len()),
        catalog.reference_count(),
        catalog.observation_count(),
        output.display(),
    );
    Ok(())
}

fn measured_target_order(
    report: &VulkanRuntimeDistributedPlacementCalibrationReport,
) -> MeasuredTargetOrder {
    MeasuredTargetOrder {
        order: report.physical_device_ids.clone(),
        duration_ns: report.measured_execution_ns,
        owner_target_id: report.execution_case.owner_physical_device_id.clone(),
        output_target_id: report.execution_case.output_physical_device_id.clone(),
        resident_bytes: report.resident_parameter_bytes_by_device.clone(),
        transient_bytes: report.resident_transient_bytes_by_device.clone(),
        host_transient_bytes: report.resident_host_transient_bytes,
        contract_ids: report.execution_case.behavior.contract_ids.clone(),
        strategy: report.execution_case.strategy,
    }
}

fn non_dominated_target_orders(measurements: &[MeasuredTargetOrder]) -> Vec<Vec<String>> {
    let mut retained = measurements
        .iter()
        .enumerate()
        .filter(|(candidate_index, candidate)| {
            !measurements.iter().enumerate().any(|(other_index, other)| {
                candidate_index != &other_index
                    && same_future_state(candidate, other)
                    && dominates(other, candidate)
            })
        })
        .map(|(_, measurement)| measurement.order.clone())
        .collect::<Vec<_>>();
    retained.sort();
    retained.dedup();
    retained
}

fn same_future_state(left: &MeasuredTargetOrder, right: &MeasuredTargetOrder) -> bool {
    left.owner_target_id == right.owner_target_id
        && left.output_target_id == right.output_target_id
        && left.order.iter().collect::<BTreeSet<_>>() == right.order.iter().collect::<BTreeSet<_>>()
        && left.contract_ids == right.contract_ids
        && left.strategy == right.strategy
}

fn dominates(left: &MeasuredTargetOrder, right: &MeasuredTargetOrder) -> bool {
    let duration_better = left.duration_ns <= right.duration_ns;
    let host_better = left.host_transient_bytes <= right.host_transient_bytes;
    let resident_better = byte_vector_is_no_larger(&left.resident_bytes, &right.resident_bytes);
    let transient_better = byte_vector_is_no_larger(&left.transient_bytes, &right.transient_bytes);
    let strictly_better = left.duration_ns < right.duration_ns
        || left.host_transient_bytes < right.host_transient_bytes
        || left.resident_bytes != right.resident_bytes
        || left.transient_bytes != right.transient_bytes;
    duration_better && host_better && resident_better && transient_better && strictly_better
}

fn byte_vector_is_no_larger(
    left: &BTreeMap<String, usize>,
    right: &BTreeMap<String, usize>,
) -> bool {
    let devices = left.keys().chain(right.keys()).collect::<BTreeSet<_>>();
    devices.into_iter().all(|device| {
        left.get(device).copied().unwrap_or(0) <= right.get(device).copied().unwrap_or(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured(
        order: &[&str],
        duration_ns: u64,
        resident: &[(&str, usize)],
        transient: &[(&str, usize)],
    ) -> MeasuredTargetOrder {
        MeasuredTargetOrder {
            order: order.iter().map(|target| (*target).to_string()).collect(),
            duration_ns,
            owner_target_id: order[0].to_string(),
            output_target_id: order[0].to_string(),
            resident_bytes: resident
                .iter()
                .map(|(target, bytes)| ((*target).to_string(), *bytes))
                .collect(),
            transient_bytes: transient
                .iter()
                .map(|(target, bytes)| ((*target).to_string(), *bytes))
                .collect(),
            host_transient_bytes: 0,
            contract_ids: vec!["contract".to_string()],
            strategy: VulkanPlacementExecutionStrategy::TensorParallel,
        }
    }

    #[test]
    fn pruning_only_compares_candidates_with_the_same_future_state() {
        let measurements = vec![
            measured(&["a", "b", "c"], 10, &[("a", 5)], &[("a", 2)]),
            measured(&["a", "c", "b"], 20, &[("a", 5)], &[("a", 2)]),
            measured(&["a", "b", "d"], 30, &[("a", 5)], &[("a", 2)]),
            measured(&["b", "a", "c"], 40, &[("b", 5)], &[("b", 2)]),
        ];
        assert_eq!(
            non_dominated_target_orders(&measurements),
            vec![
                vec!["a", "b", "c"],
                vec!["a", "b", "d"],
                vec!["b", "a", "c"],
            ]
            .into_iter()
            .map(|order| order.into_iter().map(str::to_string).collect())
            .collect::<Vec<Vec<String>>>(),
        );
    }

    #[test]
    fn duration_cannot_dominate_a_different_resource_tradeoff() {
        let fast_large = measured(&["a", "b", "c"], 10, &[("a", 20)], &[("a", 2)]);
        let slow_small = measured(&["a", "c", "b"], 20, &[("a", 5)], &[("a", 2)]);
        assert_eq!(
            non_dominated_target_orders(&[fast_large.clone(), slow_small.clone()]),
            vec![fast_large.order, slow_small.order],
        );
    }

    #[test]
    fn equal_candidates_are_both_preserved_as_valid_evidence() {
        let first = measured(&["a", "b", "c"], 10, &[("a", 5)], &[("a", 2)]);
        let second = measured(&["a", "c", "b"], 10, &[("a", 5)], &[("a", 2)]);
        assert_eq!(
            non_dominated_target_orders(&[first.clone(), second.clone()]),
            vec![first.order, second.order],
        );
    }

    #[test]
    fn pruning_preserves_distinct_physical_strategies() {
        let fast_tp = measured(&["a", "b", "c"], 10, &[("a", 5)], &[("a", 2)]);
        let mut slower_expert = measured(&["a", "c", "b"], 20, &[("a", 5)], &[("a", 2)]);
        slower_expert.strategy = VulkanPlacementExecutionStrategy::IntraExpertTensorParallel;
        slower_expert.contract_ids = vec!["expert-contract".to_string()];

        assert_eq!(
            non_dominated_target_orders(&[fast_tp, slower_expert]),
            vec![
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                vec!["a".to_string(), "c".to_string(), "b".to_string()],
            ],
        );
    }
}
