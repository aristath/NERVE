use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io;
use std::path::Path;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanPlacementExecutionStrategy,
    VulkanRuntimeDistributedPlacementCalibrationReport, VulkanRuntimePlacementCalibrationTarget,
    vulkan_runtime_placement_transfer_byte_counts,
};

use crate::boundary_calibration::measure_boundary_candidate_for_byte_counts;
use crate::calibration_device_state::discover_calibration_hardware_profiles;
use crate::calibration_package::CalibrationPackage;
use crate::calibration_package::CalibrationRuntimeConfig;
use crate::calibration_suite_plan::{expand_target_orders, plan_calibration_suite};
use crate::load_wave_calibration::measure_load_wave_candidate_for_runtime_model;
use crate::output::write_atomic;
use crate::package_calibration::measure_package_candidates_for_runtime_model;

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

#[derive(Clone, Debug)]
struct SelectedComponentCalibrationCase {
    phase: crate::cli::PackageCalibrationPhase,
    owner_target_id: String,
    runtime_variant_index: usize,
    target: VulkanRuntimePlacementCalibrationTarget,
}

#[derive(Clone, Debug)]
struct SelectedLoadWaveCalibrationCase {
    phase: crate::cli::PackageCalibrationPhase,
    owner_target_id: String,
    runtime_variant_index: usize,
    component_id: String,
    selector_id: String,
    resource_indices: Vec<usize>,
}

pub fn run_calibration_suite(
    package_path: &Path,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package_path)?;
    package.reject_output_collision(output)?;
    let hardware_profiles = discover_calibration_hardware_profiles(target_ids)?;
    let runtime_models = target_ids
        .iter()
        .map(|owner_target_id| {
            let profile = hardware_profiles
                .get(owner_target_id)
                .expect("every requested calibration target has a profile");
            package
                .runtime_models_for_owner(owner_target_id, profile, runtime)
                .map(|models| (owner_target_id.clone(), models))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let plans = runtime_models
        .iter()
        .flat_map(|(owner_target_id, runtime_models)| {
            runtime_models
                .iter()
                .enumerate()
                .map(move |(runtime_variant_index, runtime_model)| {
                    plan_calibration_suite(
                        runtime_model,
                        target_ids,
                        prefill_widths,
                        maximum_group_size,
                    )
                    .map(|plan| ((owner_target_id.clone(), runtime_variant_index), plan))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let reference_plan = plans.values().next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration suite has no target plan",
        )
    })?;
    let component_cases = selected_component_calibration_cases(&plans);
    let load_wave_cases = selected_load_wave_calibration_cases(&plans);
    let boundary_frame_byte_counts = runtime_models
        .values()
        .flatten()
        .map(vulkan_runtime_placement_transfer_byte_counts)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut measured_component_candidates = 0usize;
    let mut unavailable_component_candidates = 0usize;

    for case in &component_cases {
        let mut current_width_measurements = Vec::new();
        for order in reference_plan
            .initial_target_orders
            .iter()
            .filter(|order| order.first() == Some(&case.owner_target_id))
        {
            let measurement = measure_package_candidates_for_runtime_model(
                &package,
                &runtime_models[&case.owner_target_id][case.runtime_variant_index],
                &case.target,
                case.phase,
                order,
            )?;
            if measurement.catalog.observation_count() == 0 {
                unavailable_component_candidates += 1;
                continue;
            }
            catalog.merge(&measurement.catalog)?;
            measured_component_candidates += measurement.catalog.observation_count();
            if order.len() == 2 {
                current_width_measurements
                    .extend(measurement.reports.iter().map(measured_target_order));
            }
        }

        let mut width = 2usize;
        while width < reference_plan.maximum_group_size && !current_width_measurements.is_empty() {
            let promising = non_dominated_target_orders(&current_width_measurements);
            let expanded =
                expand_target_orders(&promising, target_ids, reference_plan.maximum_group_size)?;
            let mut next_width_measurements = Vec::new();
            for order in expanded {
                let measurement = measure_package_candidates_for_runtime_model(
                    &package,
                    &runtime_models[&case.owner_target_id][case.runtime_variant_index],
                    &case.target,
                    case.phase,
                    &order,
                )?;
                if measurement.catalog.observation_count() == 0 {
                    unavailable_component_candidates += 1;
                    continue;
                }
                catalog.merge(&measurement.catalog)?;
                measured_component_candidates += measurement.catalog.observation_count();
                next_width_measurements
                    .extend(measurement.reports.iter().map(measured_target_order));
            }
            current_width_measurements = next_width_measurements;
            width += 1;
        }
    }

    for case in &reference_plan.boundary_cases {
        let measured = measure_boundary_candidate_for_byte_counts(
            case.phase,
            &case.source_target_id,
            &case.destination_target_id,
            &boundary_frame_byte_counts,
        )?;
        catalog.merge(&measured)?;
    }

    for case in &load_wave_cases {
        let measured = measure_load_wave_candidate_for_runtime_model(
            &package,
            &runtime_models[&case.owner_target_id][case.runtime_variant_index],
            &case.component_id,
            &case.selector_id,
            case.phase,
            &case.resource_indices,
            &case.owner_target_id,
        )?;
        catalog.merge(&measured.catalog)?;
    }

    catalog.validate()?;
    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package suite: package={}, targets={}, component_cases={}, measured_component_candidates={}, unavailable_component_candidates={}, boundary_cases={}, load_wave_cases={}, references={}, observations={}, output={}",
        package.source_path().display(),
        target_ids.len(),
        component_cases.len(),
        measured_component_candidates,
        unavailable_component_candidates,
        reference_plan.boundary_cases.len(),
        load_wave_cases.len(),
        catalog.reference_count(),
        catalog.observation_count(),
        output.display(),
    );
    Ok(())
}

fn selected_component_calibration_cases(
    plans: &BTreeMap<(String, usize), crate::calibration_suite_plan::CalibrationSuitePlan>,
) -> Vec<SelectedComponentCalibrationCase> {
    let mut selected = BTreeMap::new();
    for ((owner_target_id, runtime_variant_index), plan) in plans {
        for case in &plan.component_cases {
            selected
                .entry((
                    phase_key(case.phase),
                    owner_target_id.clone(),
                    case.target.signature_id.clone(),
                ))
                .or_insert_with(|| SelectedComponentCalibrationCase {
                    phase: case.phase,
                    owner_target_id: owner_target_id.clone(),
                    runtime_variant_index: *runtime_variant_index,
                    target: case.target.clone(),
                });
        }
    }
    selected.into_values().collect()
}

fn selected_load_wave_calibration_cases(
    plans: &BTreeMap<(String, usize), crate::calibration_suite_plan::CalibrationSuitePlan>,
) -> Vec<SelectedLoadWaveCalibrationCase> {
    let mut selected = BTreeMap::new();
    for ((owner_target_id, runtime_variant_index), plan) in plans {
        for case in &plan.load_wave_cases {
            selected
                .entry((
                    phase_key(case.phase),
                    owner_target_id.clone(),
                    case.component_id.clone(),
                    case.selector_id.clone(),
                    case.resource_indices.clone(),
                ))
                .or_insert_with(|| SelectedLoadWaveCalibrationCase {
                    phase: case.phase,
                    owner_target_id: owner_target_id.clone(),
                    runtime_variant_index: *runtime_variant_index,
                    component_id: case.component_id.clone(),
                    selector_id: case.selector_id.clone(),
                    resource_indices: case.resource_indices.clone(),
                });
        }
    }
    selected.into_values().collect()
}

fn phase_key(phase: crate::cli::PackageCalibrationPhase) -> (&'static str, usize) {
    match phase {
        crate::cli::PackageCalibrationPhase::Decode => ("decode", 1),
        crate::cli::PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => ("prefill", activation_batch_width),
    }
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
        contract_ids: report.execution_case.contract_ids.clone(),
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
        // Participant order is physical state, not presentation. Contract
        // lowering assigns shard ordinals, tensor ranges, and whole experts
        // from this order. Two permutations of the same target set can
        // therefore expose different work and transports when expanded.
        && left.order == right.order
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

    fn tiny_runtime_model() -> nerve_runtime::VulkanResidentRuntimeModel {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../runtime-rs/test-fixtures/tiny_model/vulkan_resident_package.json");
        nerve_runtime::VulkanResidentModelPackageManifest::from_json_file(&path)
            .unwrap()
            .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
            .unwrap()
    }

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
    fn selected_component_cases_remain_owner_and_representation_specific() {
        let model = tiny_runtime_model();
        let targets = ["owner-a".to_string(), "owner-b".to_string()];
        let plan = plan_calibration_suite(&model, &targets, &[4], Some(2)).unwrap();
        assert!(
            plan.component_cases
                .iter()
                .all(|case| case.phase == crate::cli::PackageCalibrationPhase::Decode)
        );
        assert!(
            plan.boundary_cases
                .iter()
                .all(|case| case.phase == crate::cli::PackageCalibrationPhase::Decode)
        );
        let plans = BTreeMap::from([
            ((targets[0].clone(), 0), plan.clone()),
            ((targets[0].clone(), 1), plan.clone()),
            ((targets[1].clone(), 0), plan.clone()),
        ]);
        let cases = selected_component_calibration_cases(&plans);
        let owners = cases
            .iter()
            .map(|case| case.owner_target_id.as_str())
            .collect::<BTreeSet<_>>();

        assert_eq!(owners, BTreeSet::from(["owner-a", "owner-b"]));
        assert_eq!(cases.len(), plan.component_cases.len() * owners.len());
        assert!(
            cases.iter().all(|case| case.runtime_variant_index == 0),
            "an identical representation must be calibrated once per owner"
        );
        assert!(
            cases
                .iter()
                .all(|case| !case.target.signature_id.is_empty())
        );
    }

    #[test]
    fn pruning_preserves_distinct_participant_orders() {
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
                vec!["a", "c", "b"],
                vec!["b", "a", "c"],
            ]
            .into_iter()
            .map(|order| order.into_iter().map(str::to_string).collect())
            .collect::<Vec<Vec<String>>>(),
        );
    }

    #[test]
    fn slower_expert_ownership_permutation_remains_expandable() {
        let fast_contiguous = measured(
            &["owner", "helper-a", "helper-b"],
            10,
            &[("owner", 5)],
            &[("owner", 2)],
        );
        let mut hot_expert_friendly = measured(
            &["owner", "helper-b", "helper-a"],
            20,
            &[("owner", 5)],
            &[("owner", 2)],
        );
        hot_expert_friendly.strategy = VulkanPlacementExecutionStrategy::WholeExpertParallel;
        let mut fast_contiguous = fast_contiguous;
        fast_contiguous.strategy = VulkanPlacementExecutionStrategy::WholeExpertParallel;

        assert_eq!(
            non_dominated_target_orders(&[fast_contiguous.clone(), hot_expert_friendly.clone(),]),
            vec![fast_contiguous.order, hot_expert_friendly.order],
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
