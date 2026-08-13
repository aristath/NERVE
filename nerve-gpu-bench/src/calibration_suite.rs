use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io;
use std::path::Path;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanPlacementExecutionStrategy,
    VulkanRuntimeDistributedPlacementCalibrationReport, VulkanRuntimePlacementCalibrationTarget,
    VulkanTargetedComponentExecutionPhase, vulkan_runtime_distributed_contract_candidates,
    vulkan_runtime_placement_calibration_target_for_component,
    vulkan_runtime_placement_transfer_byte_counts,
};
use serde::Serialize;

use crate::boundary_calibration::measure_boundary_candidate_for_byte_counts;
use crate::calibration_device_state::discover_calibration_hardware_profiles;
use crate::calibration_package::CalibrationPackage;
use crate::calibration_package::CalibrationRuntimeConfig;
use crate::calibration_suite_plan::{expand_target_orders, plan_calibration_suite};
use crate::output::write_atomic;
use crate::package_calibration::measure_package_candidates_for_runtime_model;
use crate::region_calibration::measure_region_candidates_for_runtime_model;
use crate::selected_resource_calibration::measure_selected_resource_classes_for_runtime_model;

pub const CALIBRATION_SUITE_DRY_PLAN_SCHEMA: &str = "nerve.calibration_suite_dry_plan.v2";

struct PreparedCalibrationSuite {
    package: CalibrationPackage,
    runtime_models: BTreeMap<String, Vec<nerve_runtime::VulkanResidentRuntimeModel>>,
    plans: BTreeMap<(String, usize), crate::calibration_suite_plan::CalibrationSuitePlan>,
}

#[derive(Debug, Serialize)]
struct CalibrationSuiteDryPlanReport {
    schema: &'static str,
    executes_workloads: bool,
    opens_compute_devices: bool,
    package: String,
    package_id: String,
    target_ids: Vec<String>,
    context_size: usize,
    speculative_draft_tokens: usize,
    residency_policy: String,
    requested_prefill_widths: Vec<usize>,
    unsupported_requested_prefill_widths: Vec<usize>,
    phase_component_case_counts: BTreeMap<String, usize>,
    maximum_group_size: usize,
    initial_target_orders: Vec<Vec<String>>,
    component_cases: Vec<CalibrationSuiteDryPlanComponentCase>,
    component_case_count: usize,
    component_occurrence_count: usize,
    distributed_candidate_count: usize,
    distributed_contract_count: usize,
    candidate_strategy_counts: BTreeMap<String, usize>,
    boundary_case_count: usize,
    representation_variant_count: usize,
    runtime_variant_equivalence_class_count: usize,
    adaptive_expansion: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct CalibrationSuiteDryPlanComponentCase {
    owner_target_id: String,
    runtime_variant_index: usize,
    phase: &'static str,
    activation_batch_width: usize,
    signature_id: String,
    representative_component_id: String,
    occurrence_count: usize,
    distributed_candidates: Vec<CalibrationSuiteDryPlanContractCandidate>,
}

#[derive(Clone, Debug, Serialize)]
struct CalibrationSuiteDryPlanContractCandidate {
    contract_ids: Vec<String>,
    strategies: Vec<String>,
}

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
struct SelectedRegionCalibrationCase {
    phase: crate::cli::PackageCalibrationPhase,
    owner_target_id: String,
    runtime_variant_index: usize,
}

pub fn run_calibration_suite(
    package_path: &Path,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let PreparedCalibrationSuite {
        package,
        runtime_models,
        plans,
    } = prepare_calibration_suite(
        package_path,
        target_ids,
        prefill_widths,
        maximum_group_size,
        runtime,
    )?;
    package.reject_output_collision(output)?;
    let reference_plan = plans.values().next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration suite has no target plan",
        )
    })?;
    let component_cases = selected_component_calibration_cases(&plans);
    let region_cases = selected_region_calibration_cases(&runtime_models, &plans)?;
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
    let mut selected_resource_cases = 0usize;
    let mut measured_selected_resource_cases = 0usize;
    let mut unavailable_selected_resource_cases = 0usize;
    let mut planned_region_cases = 0usize;
    let mut measured_region_cases = 0usize;
    let mut unavailable_region_cases = 0usize;

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

    for case in &component_cases {
        let measured = measure_selected_resource_classes_for_runtime_model(
            &package,
            &runtime_models[&case.owner_target_id][case.runtime_variant_index],
            &case.target,
            case.phase,
            &case.owner_target_id,
        )?;
        selected_resource_cases += measured.planned_case_count;
        measured_selected_resource_cases += measured.measured_case_count;
        unavailable_selected_resource_cases += measured.unavailable_case_count;
        catalog.merge(&measured.catalog)?;
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

    for case in &region_cases {
        let measured = measure_region_candidates_for_runtime_model(
            &package,
            &runtime_models[&case.owner_target_id][case.runtime_variant_index],
            case.phase,
            target_ids,
            &catalog,
            runtime.residency_policy,
        )?;
        planned_region_cases += measured.planned_case_count;
        measured_region_cases += measured.measured_case_count;
        unavailable_region_cases += measured.unavailable_case_count;
        catalog.merge(&measured.catalog)?;
    }

    catalog.validate()?;
    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package suite: package={}, targets={}, component_cases={}, measured_component_candidates={}, unavailable_component_candidates={}, selected_resource_cases={}, measured_selected_resource_cases={}, unavailable_selected_resource_cases={}, boundary_cases={}, region_plans={}, measured_regions={}, unavailable_regions={}, references={}, observations={}, selected_resource_classes={}, region_executions={}, output={}",
        package.source_path().display(),
        target_ids.len(),
        component_cases.len(),
        measured_component_candidates,
        unavailable_component_candidates,
        selected_resource_cases,
        measured_selected_resource_cases,
        unavailable_selected_resource_cases,
        reference_plan.boundary_cases.len(),
        planned_region_cases,
        measured_region_cases,
        unavailable_region_cases,
        catalog.reference_count(),
        catalog.observation_count(),
        catalog.selected_resource_execution_class_count(),
        catalog.region_execution_count(),
        output.display(),
    );
    Ok(())
}

fn prepare_calibration_suite(
    package_path: &Path,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
    runtime: CalibrationRuntimeConfig,
) -> Result<PreparedCalibrationSuite, Box<dyn Error>> {
    let package = CalibrationPackage::load(package_path)?;
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
    Ok(PreparedCalibrationSuite {
        package,
        runtime_models,
        plans,
    })
}

pub fn dry_plan_calibration_suite(
    package_path: &Path,
    target_ids: &[String],
    prefill_widths: &[usize],
    maximum_group_size: Option<usize>,
    runtime: CalibrationRuntimeConfig,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let package = CalibrationPackage::load(package_path)?;
    let package_source_path = package.source_path().to_path_buf();
    let envelope = package.execution_envelope(runtime)?;
    let profiles = discover_calibration_hardware_profiles(target_ids)?;
    let profile_specific_variants = package.has_runtime_implementation_alternatives()?;
    let mut capability_group_index = BTreeMap::<String, usize>::new();
    let mut capability_groups = Vec::<Vec<String>>::new();
    for target_id in target_ids {
        let profile = profiles.get(target_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("calibration dry plan target {target_id:?} has no profile"),
            )
        })?;
        let variant_equivalence_key = if profile_specific_variants {
            profile.profile_id.as_str()
        } else {
            "exact_baseline_without_alternatives"
        };
        let group_index = if let Some(index) = capability_group_index.get(variant_equivalence_key) {
            *index
        } else {
            let index = capability_groups.len();
            capability_group_index.insert(variant_equivalence_key.to_string(), index);
            capability_groups.push(Vec::new());
            index
        };
        capability_groups[group_index].push(target_id.clone());
    }

    let mut package = Some(package);
    let mut package_id = None;
    let mut reference_plan = None;
    let mut component_cases = Vec::new();
    let mut representation_variant_count = 0usize;
    for group in &capability_groups {
        let representative = group
            .first()
            .expect("capability groups are created from concrete targets");
        let profile = profiles
            .get(representative)
            .expect("representative target has a discovered profile");
        let runtime_models = if profile_specific_variants {
            package
                .as_ref()
                .expect("profile-specific planning retains its source package")
                .runtime_models_for_owner(representative, profile, runtime)?
        } else {
            package
                .take()
                .expect("exact-baseline planning consumes its package once")
                .into_runtime_models_for_owner(representative, profile, runtime)?
        };
        let plans = runtime_models
            .iter()
            .enumerate()
            .map(|(runtime_variant_index, runtime_model)| {
                plan_calibration_suite(
                    runtime_model,
                    target_ids,
                    prefill_widths,
                    maximum_group_size,
                )
                .map(|plan| ((representative.clone(), runtime_variant_index), plan))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let first_plan = plans.values().next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "calibration dry plan capability class has no runtime variant",
            )
        })?;
        reference_plan.get_or_insert_with(|| first_plan.clone());
        let variant_count = runtime_models.len();
        package_id.get_or_insert_with(|| runtime_models[0].package.package_id.clone());
        let runtime_models = BTreeMap::from([(representative.clone(), runtime_models)]);
        let local_cases = calibration_suite_dry_plan_component_cases(&runtime_models, &plans)?;
        for owner_target_id in group {
            component_cases.extend(local_cases.iter().cloned().map(|mut case| {
                case.owner_target_id = owner_target_id.clone();
                case
            }));
        }
        representation_variant_count = representation_variant_count
            .checked_add(variant_count.checked_mul(group.len()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "calibration dry plan representation count overflowed",
                )
            })?)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "calibration dry plan representation count overflowed",
                )
            })?;
    }
    let report = finalize_calibration_suite_dry_plan_report(
        &package_source_path,
        package_id
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no runtime variants"))?,
        target_ids.to_vec(),
        envelope.context_activations.maximum,
        envelope.speculative_draft_tokens,
        envelope.residency_policy,
        prefill_widths,
        reference_plan.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "calibration suite has no target plan",
            )
        })?,
        component_cases,
        representation_variant_count,
        capability_groups.len(),
    )?;
    Ok(serde_json::to_vec_pretty(&report)?)
}

#[cfg(test)]
fn calibration_suite_dry_plan_report(
    prepared: &PreparedCalibrationSuite,
    context_size: usize,
    speculative_draft_tokens: usize,
    residency_policy: String,
    requested_prefill_widths: &[usize],
) -> Result<CalibrationSuiteDryPlanReport, Box<dyn Error>> {
    let reference_plan = prepared.plans.values().next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration suite has no target plan",
        )
    })?;
    let component_cases =
        calibration_suite_dry_plan_component_cases(&prepared.runtime_models, &prepared.plans)?;
    Ok(finalize_calibration_suite_dry_plan_report(
        prepared.package.source_path(),
        prepared
            .runtime_models
            .values()
            .flatten()
            .next()
            .map(|model| model.package.package_id.clone())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no runtime variants"))?,
        prepared.runtime_models.keys().cloned().collect(),
        context_size,
        speculative_draft_tokens,
        residency_policy,
        requested_prefill_widths,
        reference_plan.clone(),
        component_cases,
        prepared.runtime_models.values().map(Vec::len).sum(),
        prepared.runtime_models.len(),
    )?)
}

fn calibration_suite_dry_plan_component_cases(
    runtime_models: &BTreeMap<String, Vec<nerve_runtime::VulkanResidentRuntimeModel>>,
    plans: &BTreeMap<(String, usize), crate::calibration_suite_plan::CalibrationSuitePlan>,
) -> Result<Vec<CalibrationSuiteDryPlanComponentCase>, Box<dyn Error>> {
    let selected_cases = selected_component_calibration_cases(plans);
    let mut component_cases = Vec::with_capacity(selected_cases.len());

    for case in selected_cases {
        let runtime_model = runtime_models
            .get(&case.owner_target_id)
            .and_then(|variants| variants.get(case.runtime_variant_index))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "calibration dry plan references absent runtime variant {} for owner {:?}",
                        case.runtime_variant_index, case.owner_target_id,
                    ),
                )
            })?;
        let execution = runtime_model
            .component_executions
            .iter()
            .find(|execution| execution.component_id == case.target.component_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "calibration dry plan found no execution for component {:?}",
                        case.target.component_id,
                    ),
                )
            })?;
        let strategies_by_contract = execution
            .kernels
            .iter()
            .flat_map(|kernel| &kernel.physical_execution_contracts)
            .map(|contract| (contract.contract_id.as_str(), contract.strategy))
            .collect::<BTreeMap<_, _>>();
        let runtime_phase = match case.phase {
            crate::cli::PackageCalibrationPhase::Decode => {
                VulkanTargetedComponentExecutionPhase::Decode
            }
            crate::cli::PackageCalibrationPhase::Prefill {
                activation_batch_width,
            } => VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width,
            },
        };
        let distributed_candidates = vulkan_runtime_distributed_contract_candidates(
            runtime_model,
            &case.target,
            runtime_phase,
        )?
        .into_iter()
        .map(|candidate| {
            let mut strategies = BTreeSet::new();
            for contract_id in &candidate.contract_ids {
                let strategy = strategies_by_contract.get(contract_id.as_str()).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "calibration dry plan candidate references unknown contract {contract_id:?}",
                        ),
                    )
                })?;
                strategies.insert(execution_strategy_name(*strategy).to_string());
            }
            Ok(CalibrationSuiteDryPlanContractCandidate {
                contract_ids: candidate.contract_ids.into_iter().collect(),
                strategies: strategies.into_iter().collect(),
            })
        })
        .collect::<Result<Vec<_>, io::Error>>()?;
        let (phase, activation_batch_width) = phase_key(case.phase);
        component_cases.push(CalibrationSuiteDryPlanComponentCase {
            owner_target_id: case.owner_target_id,
            runtime_variant_index: case.runtime_variant_index,
            phase,
            activation_batch_width,
            signature_id: case.target.signature_id,
            representative_component_id: case.target.component_id,
            occurrence_count: case.target.component_ids.len(),
            distributed_candidates,
        });
    }
    Ok(component_cases)
}

#[allow(clippy::too_many_arguments)]
fn finalize_calibration_suite_dry_plan_report(
    package_path: &Path,
    package_id: String,
    target_ids: Vec<String>,
    context_size: usize,
    speculative_draft_tokens: usize,
    residency_policy: String,
    requested_prefill_widths: &[usize],
    reference_plan: crate::calibration_suite_plan::CalibrationSuitePlan,
    component_cases: Vec<CalibrationSuiteDryPlanComponentCase>,
    representation_variant_count: usize,
    runtime_variant_equivalence_class_count: usize,
) -> Result<CalibrationSuiteDryPlanReport, io::Error> {
    let mut all_contract_ids = BTreeSet::new();
    let mut candidate_strategy_counts = BTreeMap::<String, usize>::new();
    let mut distributed_candidate_count = 0usize;
    let mut component_occurrence_count = 0usize;
    let mut phase_component_case_counts = BTreeMap::<String, usize>::new();
    for case in &component_cases {
        let phase_key = format!("{}:{}", case.phase, case.activation_batch_width);
        *phase_component_case_counts.entry(phase_key).or_default() += 1;
        component_occurrence_count = component_occurrence_count
            .checked_add(case.occurrence_count)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "component count overflow")
            })?;
        for candidate in &case.distributed_candidates {
            distributed_candidate_count =
                distributed_candidate_count.checked_add(1).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "candidate count overflow")
                })?;
            all_contract_ids.extend(candidate.contract_ids.iter().cloned());
            for strategy in &candidate.strategies {
                *candidate_strategy_counts
                    .entry(strategy.clone())
                    .or_default() += 1;
            }
        }
    }
    let mut requested_prefill_widths = requested_prefill_widths.to_vec();
    requested_prefill_widths.sort_unstable();
    requested_prefill_widths.dedup();
    let unsupported_requested_prefill_widths = requested_prefill_widths
        .iter()
        .copied()
        .filter(|width| !phase_component_case_counts.contains_key(&format!("prefill:{width}")))
        .collect();
    Ok(CalibrationSuiteDryPlanReport {
        schema: CALIBRATION_SUITE_DRY_PLAN_SCHEMA,
        executes_workloads: false,
        opens_compute_devices: false,
        package: package_path.display().to_string(),
        package_id,
        target_ids,
        context_size,
        speculative_draft_tokens,
        residency_policy,
        requested_prefill_widths,
        unsupported_requested_prefill_widths,
        phase_component_case_counts,
        maximum_group_size: reference_plan.maximum_group_size,
        initial_target_orders: reference_plan.initial_target_orders,
        component_case_count: component_cases.len(),
        component_occurrence_count,
        distributed_candidate_count,
        distributed_contract_count: all_contract_ids.len(),
        candidate_strategy_counts,
        boundary_case_count: reference_plan.boundary_cases.len(),
        representation_variant_count,
        runtime_variant_equivalence_class_count,
        adaptive_expansion: "measurement_driven_after_singletons_and_directed_pairs",
        component_cases,
    })
}

fn execution_strategy_name(strategy: nerve_execution_contracts::ExecutionStrategy) -> &'static str {
    match strategy {
        nerve_execution_contracts::ExecutionStrategy::SingleDevice => "single_device",
        nerve_execution_contracts::ExecutionStrategy::TensorParallel => "tensor_parallel",
        nerve_execution_contracts::ExecutionStrategy::ExpertParallel => "expert_parallel",
        nerve_execution_contracts::ExecutionStrategy::TensorParallelExpert => {
            "tensor_parallel_expert"
        }
    }
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

fn selected_region_calibration_cases(
    runtime_models: &BTreeMap<String, Vec<nerve_runtime::VulkanResidentRuntimeModel>>,
    plans: &BTreeMap<(String, usize), crate::calibration_suite_plan::CalibrationSuitePlan>,
) -> Result<Vec<SelectedRegionCalibrationCase>, Box<dyn Error>> {
    let mut selected = BTreeMap::new();
    for ((owner_target_id, runtime_variant_index), plan) in plans {
        let runtime_model = runtime_models
            .get(owner_target_id)
            .and_then(|variants| variants.get(*runtime_variant_index))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "region calibration plan references absent runtime variant {} for owner {:?}",
                        runtime_variant_index, owner_target_id,
                    ),
                )
            })?;
        let phases = plan
            .component_cases
            .iter()
            .map(|case| (phase_key(case.phase), case.phase))
            .collect::<BTreeMap<_, _>>();
        for phase in phases.into_values() {
            let runtime_phase = match phase {
                crate::cli::PackageCalibrationPhase::Decode => {
                    VulkanTargetedComponentExecutionPhase::Decode
                }
                crate::cli::PackageCalibrationPhase::Prefill {
                    activation_batch_width,
                } => VulkanTargetedComponentExecutionPhase::Prefill {
                    activation_batch_width,
                },
            };
            let ordered_signatures = runtime_model
                .circuit_graph
                .components
                .iter()
                .filter(|component| component.runtime_role.is_signal_processor())
                .map(|component| {
                    vulkan_runtime_placement_calibration_target_for_component(
                        runtime_model,
                        &component.component_id,
                        runtime_phase,
                    )
                    .map(|target| target.signature_id)
                    .map_err(|error| -> Box<dyn Error> { Box::new(error) })
                })
                .collect::<Result<Vec<_>, _>>()?;
            selected
                .entry((phase_key(phase), ordered_signatures))
                .or_insert_with(|| SelectedRegionCalibrationCase {
                    phase,
                    owner_target_id: owner_target_id.clone(),
                    runtime_variant_index: *runtime_variant_index,
                });
        }
    }
    Ok(selected.into_values().collect())
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

        let runtime_models = BTreeMap::from([
            (targets[0].clone(), vec![model.clone(), model.clone()]),
            (targets[1].clone(), vec![model]),
        ]);
        let region_cases = selected_region_calibration_cases(&runtime_models, &plans).unwrap();
        assert_eq!(region_cases.len(), 1);
        assert_eq!(
            region_cases[0].phase,
            crate::cli::PackageCalibrationPhase::Decode
        );
    }

    #[test]
    fn dry_plan_reports_exact_distributed_candidates_without_workloads() {
        let package_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../runtime-rs/test-fixtures/tiny_model/vulkan_resident_package.json");
        let package = CalibrationPackage::load(&package_path).unwrap();
        let model = tiny_runtime_model();
        let targets = ["owner-a".to_string(), "owner-b".to_string()];
        let plan = plan_calibration_suite(&model, &targets, &[4], Some(2)).unwrap();
        let prepared = PreparedCalibrationSuite {
            package,
            runtime_models: BTreeMap::from([
                (targets[0].clone(), vec![model.clone()]),
                (targets[1].clone(), vec![model]),
            ]),
            plans: BTreeMap::from([
                ((targets[0].clone(), 0), plan.clone()),
                ((targets[1].clone(), 0), plan),
            ]),
        };

        let report =
            calibration_suite_dry_plan_report(&prepared, 128, 0, "demand_paged".to_string(), &[4])
                .unwrap();

        assert_eq!(report.schema, CALIBRATION_SUITE_DRY_PLAN_SCHEMA);
        assert!(!report.executes_workloads);
        assert!(!report.opens_compute_devices);
        assert_eq!(report.target_ids, targets);
        assert_eq!(report.maximum_group_size, 2);
        assert_eq!(report.requested_prefill_widths, [4]);
        assert!(report.unsupported_requested_prefill_widths.is_empty());
        assert!(report.phase_component_case_counts.contains_key("decode:1"));
        assert!(report.phase_component_case_counts.contains_key("prefill:4"));
        assert_eq!(report.representation_variant_count, 2);
        assert_eq!(report.component_case_count, report.component_cases.len());
        assert!(report.component_occurrence_count > 0);
        assert!(report.distributed_candidate_count > 0);
        assert!(report.distributed_contract_count > 0);
        assert!(
            report
                .candidate_strategy_counts
                .contains_key("tensor_parallel")
        );
        assert!(report.component_cases.iter().any(|case| {
            case.distributed_candidates.iter().any(|candidate| {
                !candidate.contract_ids.is_empty()
                    && candidate.strategies == ["tensor_parallel".to_string()]
            })
        }));
        let encoded = serde_json::to_value(&report).unwrap();
        assert_eq!(encoded["executes_workloads"], false);
        assert_eq!(encoded["opens_compute_devices"], false);
        assert!(
            encoded
                .get("selected_resource_load_wave_case_count")
                .is_none(),
            "the suite must not schedule the redundant standalone load-wave pass",
        );
    }

    #[test]
    fn region_case_selection_rejects_a_plan_without_its_runtime_variant() {
        let model = tiny_runtime_model();
        let target = "owner-a".to_string();
        let plan =
            plan_calibration_suite(&model, std::slice::from_ref(&target), &[4], Some(1)).unwrap();
        let plans = BTreeMap::from([((target.clone(), 1), plan)]);
        let runtime_models = BTreeMap::from([(target, vec![model])]);

        let error = selected_region_calibration_cases(&runtime_models, &plans).unwrap_err();

        assert!(error.to_string().contains("absent runtime variant 1"));
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
