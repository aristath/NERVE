use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::error::Error;
use std::io;
use std::path::Path;

use nerve_execution_contracts::ExecutionStrategy;
use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanRuntimeDistributedPlacementCalibrationReport,
    VulkanRuntimePlacementCalibrationPolicy, VulkanRuntimePlacementCalibrationTarget,
    VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_canonical_placement_candidate_with_policy,
    calibrate_vulkan_runtime_staged_contract_candidate_with_policy,
    record_vulkan_runtime_canonical_placement_calibration,
    vulkan_runtime_distributed_contract_candidates,
    vulkan_runtime_placement_calibration_target_for_component,
};

use crate::calibration_device_state::{
    capture_device_snapshots, close_and_verify_device_snapshots,
    discover_calibration_hardware_profiles, open_calibration_targets, print_device_snapshots,
};
use crate::calibration_package::{CalibrationPackage, CalibrationRuntimeConfig};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;
use crate::selected_resource_calibration::measure_selected_resource_classes_for_runtime_model;

pub fn run_package_calibration(
    package: &Path,
    component: &str,
    phase: PackageCalibrationPhase,
    strategy: Option<ExecutionStrategy>,
    contract_ids: &[String],
    ordered_target_ids: &[String],
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package)?;
    package.reject_output_collision(output)?;
    let measurement = measure_package_candidates(
        &package,
        component,
        phase,
        strategy,
        contract_ids,
        ordered_target_ids,
        runtime,
        true,
    )?;
    if !requested_component_candidate_available(
        measurement.catalog.observation_count(),
        measurement.reports.len(),
        ordered_target_ids.len(),
    ) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the requested package placement candidate is unavailable",
        )
        .into());
    }
    validate_complete_selected_resource_evidence(&measurement)?;
    let payload = measurement.catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package={} signature={} representative={} requested_component={} phase={} batch_width={} targets={:?} observations={} contract_candidates={} selected_resource_cases={} measured_selected_resource_cases={} unavailable_selected_resource_cases={} selected_resource_classes={} sampled={} best_measured_ns={} output={}",
        package.source_path().display(),
        measurement
            .targets
            .iter()
            .map(|target| target.signature_id.as_str())
            .collect::<Vec<_>>()
            .join(","),
        measurement
            .targets
            .iter()
            .map(|target| target.component_id.as_str())
            .collect::<Vec<_>>()
            .join(","),
        component,
        match phase {
            PackageCalibrationPhase::Decode => "decode",
            PackageCalibrationPhase::Prefill { .. } => "prefill",
        },
        phase.activation_batch_width(),
        ordered_target_ids,
        measurement.catalog.observation_count(),
        measurement.reports.len(),
        measurement.planned_selected_resource_case_count,
        measurement.measured_selected_resource_case_count,
        measurement.unavailable_selected_resource_case_count,
        measurement
            .catalog
            .selected_resource_execution_class_count(),
        measurement
            .reports
            .iter()
            .any(|report| report.sampled_workload),
        measurement
            .reports
            .iter()
            .map(|report| report.measured_execution_ns)
            .min()
            .unwrap_or(0),
        output.display(),
    );
    Ok(())
}

pub fn run_placement_calibration(
    package: &Path,
    component: &str,
    phase: PackageCalibrationPhase,
    strategy: Option<ExecutionStrategy>,
    contract_ids: &[String],
    ordered_target_ids: &[String],
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package)?;
    package.reject_output_collision(output)?;
    let measurement = measure_package_candidates(
        &package,
        component,
        phase,
        strategy,
        contract_ids,
        ordered_target_ids,
        runtime,
        false,
    )?;
    if !requested_component_candidate_available(
        measurement.catalog.observation_count(),
        measurement.reports.len(),
        ordered_target_ids.len(),
    ) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the requested component placement candidate is unavailable",
        )
        .into());
    }
    let payload = measurement.catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated placement package={} requested_component={} phase={} batch_width={} targets={:?} observations={} distributed_reports={} best_measured_ns={} output={}",
        package.source_path().display(),
        component,
        match phase {
            PackageCalibrationPhase::Decode => "decode",
            PackageCalibrationPhase::Prefill { .. } => "prefill",
        },
        phase.activation_batch_width(),
        ordered_target_ids,
        measurement.catalog.observation_count(),
        measurement.reports.len(),
        measurement
            .reports
            .iter()
            .map(|report| report.measured_execution_ns)
            .min()
            .unwrap_or(0),
        output.display(),
    );
    Ok(())
}

pub struct PackageCalibrationMeasurement {
    pub targets: Vec<VulkanRuntimePlacementCalibrationTarget>,
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub reports: Vec<VulkanRuntimeDistributedPlacementCalibrationReport>,
    pub planned_selected_resource_case_count: usize,
    pub measured_selected_resource_case_count: usize,
    pub unavailable_selected_resource_case_count: usize,
}

fn validate_complete_selected_resource_evidence(
    measurement: &PackageCalibrationMeasurement,
) -> Result<(), io::Error> {
    if measurement.planned_selected_resource_case_count
        != measurement.measured_selected_resource_case_count
        || measurement.unavailable_selected_resource_case_count != 0
    {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "requested component has incomplete selected-resource evidence: planned={}, measured={}, unavailable={}",
                measurement.planned_selected_resource_case_count,
                measurement.measured_selected_resource_case_count,
                measurement.unavailable_selected_resource_case_count,
            ),
        ));
    }
    Ok(())
}

pub fn measure_package_candidates(
    package: &CalibrationPackage,
    component: &str,
    phase: PackageCalibrationPhase,
    strategy: Option<ExecutionStrategy>,
    contract_ids: &[String],
    ordered_target_ids: &[String],
    runtime: CalibrationRuntimeConfig,
    include_selected_resource_evidence: bool,
) -> Result<PackageCalibrationMeasurement, Box<dyn Error>> {
    let owner_id = ordered_target_ids.first().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "package calibration requires at least one ordered target",
        )
    })?;
    let profiles = discover_calibration_hardware_profiles(ordered_target_ids)?;
    let owner_profile = profiles.get(owner_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "package calibration owner has no hardware-process profile",
        )
    })?;
    let runtime_models = package.runtime_models_for_owner(owner_id, owner_profile, runtime)?;
    let execution_phase = match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    };
    let mut signatures = BTreeSet::new();
    let mut targets = Vec::new();
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let mut reports = Vec::new();
    let mut planned_selected_resource_case_count = 0usize;
    let mut measured_selected_resource_case_count = 0usize;
    let mut unavailable_selected_resource_case_count = 0usize;
    for runtime_model in &runtime_models {
        let target = vulkan_runtime_placement_calibration_target_for_component(
            runtime_model,
            component,
            execution_phase,
        )?;
        if !signatures.insert(target.signature_id.clone()) {
            continue;
        }
        let measurement = measure_package_candidates_for_runtime_model(
            package,
            runtime_model,
            &target,
            phase,
            strategy,
            contract_ids,
            ordered_target_ids,
        )?;
        let component_candidate_available = requested_component_candidate_available(
            measurement.catalog.observation_count(),
            measurement.reports.len(),
            ordered_target_ids.len(),
        );
        let selected_resource_participants = selected_resource_calibration_participants(
            include_selected_resource_evidence,
            component_candidate_available,
            ordered_target_ids,
        );
        targets.extend(measurement.targets);
        catalog.merge(&measurement.catalog)?;
        reports.extend(measurement.reports);
        for target_id in selected_resource_participants {
            let selected_resources = measure_selected_resource_classes_for_runtime_model(
                package,
                runtime_model,
                &target,
                phase,
                target_id,
            )?;
            planned_selected_resource_case_count = planned_selected_resource_case_count
                .checked_add(selected_resources.planned_case_count)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "selected-resource planned-case count overflowed",
                    )
                })?;
            measured_selected_resource_case_count = measured_selected_resource_case_count
                .checked_add(selected_resources.measured_case_count)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "selected-resource measured-case count overflowed",
                    )
                })?;
            unavailable_selected_resource_case_count = unavailable_selected_resource_case_count
                .checked_add(selected_resources.unavailable_case_count)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "selected-resource unavailable-case count overflowed",
                    )
                })?;
            catalog.merge(&selected_resources.catalog)?;
        }
    }
    Ok(PackageCalibrationMeasurement {
        targets,
        catalog,
        reports,
        planned_selected_resource_case_count,
        measured_selected_resource_case_count,
        unavailable_selected_resource_case_count,
    })
}

fn selected_resource_calibration_participants<'a>(
    include_selected_resource_evidence: bool,
    component_candidate_available: bool,
    ordered_target_ids: &'a [String],
) -> &'a [String] {
    if include_selected_resource_evidence && component_candidate_available {
        ordered_target_ids
    } else {
        &[]
    }
}

fn requested_component_candidate_available(
    observation_count: usize,
    distributed_report_count: usize,
    requested_target_count: usize,
) -> bool {
    match requested_target_count {
        0 => false,
        1 => observation_count > 0,
        _ => distributed_report_count > 0,
    }
}

pub fn measure_package_candidates_for_runtime_model(
    package: &CalibrationPackage,
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: PackageCalibrationPhase,
    strategy: Option<ExecutionStrategy>,
    requested_contract_ids: &[String],
    ordered_target_ids: &[String],
) -> Result<PackageCalibrationMeasurement, Box<dyn Error>> {
    let execution_phase = match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    };
    let devices = open_calibration_targets(ordered_target_ids)?;

    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before", &before);
    let maximum_resident_parameter_bytes_by_physical_device = before
        .iter()
        .map(|snapshot| {
            usize::try_from(snapshot.memory_accounting.admissible_remaining_bytes)
                .ok()
                .filter(|bytes| *bytes > 0)
                .map(|bytes| (snapshot.physical_device_id.clone(), bytes))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::OutOfMemory,
                        format!(
                            "selected target {:?} has no safe remaining parameter capacity",
                            snapshot.physical_device_id,
                        ),
                    )
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let maximum_total_resident_parameter_bytes =
        maximum_resident_parameter_bytes_by_physical_device
            .values()
            .try_fold(0usize, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "selected target parameter capacities overflow usize",
                )
            })?;
    let policy = VulkanRuntimePlacementCalibrationPolicy {
        maximum_total_resident_parameter_bytes,
        maximum_resident_parameter_bytes_by_physical_device,
        ..VulkanRuntimePlacementCalibrationPolicy::default()
    };
    let candidates =
        vulkan_runtime_distributed_contract_candidates(runtime_model, target, execution_phase)?;
    let candidates = candidates
        .into_iter()
        .filter_map(|candidate| {
            if !candidate_matches_requested_contracts(
                &candidate.contract_ids,
                requested_contract_ids,
            ) {
                return None;
            }
            match candidate_matches_requested_strategy(
                runtime_model,
                &target.component_id,
                &candidate.contract_ids,
                strategy,
            ) {
                Ok(true) => Some(Ok(candidate)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect::<Result<Vec<_>, io::Error>>()?;
    if (strategy.is_some() || !requested_contract_ids.is_empty()) && candidates.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "component {:?} has no complete candidate for {execution_phase:?} matching strategy={strategy:?} contracts={requested_contract_ids:?}",
                target.component_id,
            ),
        )
        .into());
    }
    let calibration_result = (|| {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let mut reports = Vec::new();
        for candidate in candidates {
            let report = calibrate_vulkan_runtime_staged_contract_candidate_with_policy(
                &devices,
                package.manifest_dir(),
                runtime_model,
                target,
                execution_phase,
                &candidate.contract_ids,
                &mut catalog,
                policy.clone(),
            )?;
            if let Some(report) = report {
                reports.push(report);
            }
            if ordered_target_ids.len() == 1 && catalog.observation_count() > 0 {
                break;
            }
        }
        if ordered_target_ids.len() == 1 && catalog.observation_count() == 0 {
            for (physical_device_id, device) in &devices {
                if let Some(canonical) =
                    calibrate_vulkan_runtime_canonical_placement_candidate_with_policy(
                        physical_device_id,
                        device.clone(),
                        package.manifest_dir(),
                        runtime_model,
                        target,
                        execution_phase,
                        policy.clone(),
                    )?
                {
                    record_vulkan_runtime_canonical_placement_calibration(&mut catalog, canonical)
                        .map_err(|error| {
                            nerve_runtime::VulkanResidentTokenModelPackageError::new(
                                error.to_string(),
                            )
                        })?;
                }
            }
        }
        Ok::<_, nerve_runtime::VulkanResidentTokenModelPackageError>((catalog, reports))
    })();

    let restoration_result = close_and_verify_device_snapshots(devices, &before);

    let (catalog, reports) = match (calibration_result, restoration_result) {
        (Ok(measured), Ok(())) => measured,
        (Err(error), Ok(())) => return Err(error.into()),
        (Ok(_), Err(restoration_error)) => {
            return Err(io::Error::other(restoration_error).into());
        }
        (Err(error), Err(restoration_error)) => {
            return Err(io::Error::other(format!(
                "{error}; teardown proof also failed: {restoration_error}"
            ))
            .into());
        }
    };
    Ok(PackageCalibrationMeasurement {
        targets: vec![target.clone()],
        catalog,
        reports,
        planned_selected_resource_case_count: 0,
        measured_selected_resource_case_count: 0,
        unavailable_selected_resource_case_count: 0,
    })
}

fn candidate_matches_requested_contracts(
    candidate: &BTreeSet<String>,
    requested: &[String],
) -> bool {
    requested.is_empty()
        || (candidate.len() == requested.len()
            && requested
                .iter()
                .all(|contract_id| candidate.contains(contract_id)))
}

fn candidate_matches_requested_strategy(
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    component_id: &str,
    contract_ids: &BTreeSet<String>,
    requested: Option<ExecutionStrategy>,
) -> Result<bool, io::Error> {
    let Some(requested) = requested else {
        return Ok(true);
    };
    let execution = runtime_model
        .component_executions
        .iter()
        .find(|execution| execution.component_id == component_id)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("component {component_id:?} has no execution catalog"),
            )
        })?;
    let matching = execution
        .kernels
        .iter()
        .flat_map(|kernel| &kernel.physical_execution_contracts)
        .filter(|contract| contract_ids.contains(&contract.contract_id))
        .collect::<Vec<_>>();
    if matching.len() != contract_ids.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("component {component_id:?} candidate references unknown physical contracts"),
        ));
    }
    Ok(matching
        .iter()
        .all(|contract| contract.strategy == requested))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_contract_selection_never_accepts_a_subset_or_superset() {
        let candidate = ["gate".to_string(), "down".to_string()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        assert!(candidate_matches_requested_contracts(&candidate, &[]));
        assert!(candidate_matches_requested_contracts(
            &candidate,
            &["down".to_string(), "gate".to_string()]
        ));
        assert!(!candidate_matches_requested_contracts(
            &candidate,
            &["gate".to_string()]
        ));
        assert!(!candidate_matches_requested_contracts(
            &candidate,
            &["down".to_string(), "gate".to_string(), "other".to_string()]
        ));
    }

    fn measurement_with_selected_resource_counts(
        planned: usize,
        measured: usize,
        unavailable: usize,
    ) -> PackageCalibrationMeasurement {
        PackageCalibrationMeasurement {
            targets: Vec::new(),
            catalog: VulkanPlacementCalibrationCatalog::default(),
            reports: Vec::new(),
            planned_selected_resource_case_count: planned,
            measured_selected_resource_case_count: measured,
            unavailable_selected_resource_case_count: unavailable,
        }
    }

    #[test]
    fn unavailable_component_skips_all_selected_resource_workloads() {
        let targets = ["owner".to_string(), "helper".to_string()];

        assert!(
            selected_resource_calibration_participants(true, false, &targets).is_empty()
        );
    }

    #[test]
    fn focused_placement_skips_selected_resource_workloads() {
        let targets = ["owner".to_string(), "helper".to_string()];

        assert!(
            selected_resource_calibration_participants(false, true, &targets).is_empty()
        );
    }

    #[test]
    fn available_component_calibrates_selected_resources_on_every_participant() {
        let targets = [
            "owner".to_string(),
            "helper-a".to_string(),
            "helper-b".to_string(),
        ];

        assert_eq!(
            selected_resource_calibration_participants(true, true, &targets),
            targets,
        );
    }

    #[test]
    fn local_reference_cannot_masquerade_as_a_distributed_candidate() {
        assert!(requested_component_candidate_available(1, 0, 1));
        assert!(!requested_component_candidate_available(1, 0, 2));
        assert!(requested_component_candidate_available(2, 1, 2));
        assert!(!requested_component_candidate_available(1, 1, 0));
    }

    #[test]
    fn component_catalog_rejects_partial_selected_resource_evidence() {
        validate_complete_selected_resource_evidence(&measurement_with_selected_resource_counts(
            2, 2, 0,
        ))
        .unwrap();
        validate_complete_selected_resource_evidence(&measurement_with_selected_resource_counts(
            0, 0, 0,
        ))
        .unwrap();

        let missing = validate_complete_selected_resource_evidence(
            &measurement_with_selected_resource_counts(2, 1, 1),
        )
        .unwrap_err();
        assert_eq!(missing.kind(), io::ErrorKind::Unsupported);
        assert!(
            missing
                .to_string()
                .contains("planned=2, measured=1, unavailable=1")
        );

        assert!(
            validate_complete_selected_resource_evidence(
                &measurement_with_selected_resource_counts(1, 2, 0),
            )
            .is_err(),
            "over-recorded evidence is also inconsistent and must not publish",
        );
    }
}
