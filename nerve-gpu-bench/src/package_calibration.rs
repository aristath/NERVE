use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::Path;

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
    capture_device_snapshots, discover_calibration_hardware_profiles, open_calibration_targets,
    print_device_snapshots, quiesce_and_verify_device_snapshots,
};
use crate::calibration_package::{CalibrationPackage, CalibrationRuntimeConfig};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;

pub fn run_package_calibration(
    package: &Path,
    component: &str,
    phase: PackageCalibrationPhase,
    ordered_target_ids: &[String],
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package)?;
    package.reject_output_collision(output)?;
    let measurement =
        measure_package_candidates(&package, component, phase, ordered_target_ids, runtime)?;
    if measurement.catalog.observation_count() == 0 {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the requested package placement candidate is unavailable",
        )
        .into());
    }
    let payload = measurement.catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package={} signature={} representative={} requested_component={} phase={} batch_width={} targets={:?} observations={} contract_candidates={} sampled={} best_measured_ns={} output={}",
        package.source_path().display(),
        measurement.target.signature_id,
        measurement.target.component_id,
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

pub struct PackageCalibrationMeasurement {
    pub target: VulkanRuntimePlacementCalibrationTarget,
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub reports: Vec<VulkanRuntimeDistributedPlacementCalibrationReport>,
}

pub fn measure_package_candidates(
    package: &CalibrationPackage,
    component: &str,
    phase: PackageCalibrationPhase,
    ordered_target_ids: &[String],
    runtime: CalibrationRuntimeConfig,
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
    let runtime_model = package.runtime_model_for_owner(owner_id, owner_profile, runtime)?;
    let execution_phase = match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    };
    let target = vulkan_runtime_placement_calibration_target_for_component(
        &runtime_model,
        component,
        execution_phase,
    )?;
    measure_package_candidates_for_runtime_model(
        package,
        &runtime_model,
        &target,
        phase,
        ordered_target_ids,
    )
}

pub fn measure_package_candidates_for_runtime_model(
    package: &CalibrationPackage,
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: PackageCalibrationPhase,
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
    let opened = open_calibration_targets(ordered_target_ids)?;
    let devices = opened.devices;

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

    let restoration_result = quiesce_and_verify_device_snapshots(&devices, &before);

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
        target: target.clone(),
        catalog,
        reports,
    })
}
