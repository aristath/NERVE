use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::Path;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanRuntimeDistributedPlacementCalibrationReport,
    VulkanRuntimePlacementCalibrationPolicy, VulkanRuntimePlacementCalibrationTarget,
    VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_staged_placement_candidate_with_policy,
    calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy,
    vulkan_runtime_placement_calibration_target_for_component,
};

use crate::calibration_device_state::{
    capture_device_snapshots, open_calibration_devices, print_device_snapshots,
    quiesce_and_verify_device_snapshots,
};
use crate::calibration_package::CalibrationPackage;
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;

pub fn run_package_calibration(
    package: &Path,
    component: &str,
    phase: PackageCalibrationPhase,
    ordered_target_ids: &[String],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package)?;
    package.reject_output_collision(output)?;
    let execution_phase = match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    };
    let target = vulkan_runtime_placement_calibration_target_for_component(
        package.runtime_model(),
        component,
        execution_phase,
    )?;

    let Some(measurement) =
        measure_package_candidate(&package, &target, phase, ordered_target_ids)?
    else {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the requested package placement candidate is unavailable",
        )
        .into());
    };
    let payload = measurement.catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package={} signature={} representative={} requested_component={} phase={} batch_width={} targets={:?} observations={} sampled={} measured_ns={} output={}",
        package.source_path().display(),
        target.signature_id,
        target.component_id,
        component,
        measurement.report.phase,
        measurement.report.activation_batch_width,
        measurement.report.physical_device_ids,
        measurement.catalog.observation_count(),
        measurement.report.sampled_workload,
        measurement.report.measured_execution_ns,
        output.display(),
    );
    Ok(())
}

pub struct PackageCalibrationMeasurement {
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub report: VulkanRuntimeDistributedPlacementCalibrationReport,
}

pub fn measure_package_candidate(
    package: &CalibrationPackage,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: PackageCalibrationPhase,
    ordered_target_ids: &[String],
) -> Result<Option<PackageCalibrationMeasurement>, Box<dyn Error>> {
    let devices = open_calibration_devices(ordered_target_ids)?;

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
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let calibration_result = match phase {
        PackageCalibrationPhase::Decode => {
            calibrate_vulkan_runtime_staged_placement_candidate_with_policy(
                &devices,
                package.manifest_dir(),
                package.runtime_model(),
                target,
                &mut catalog,
                policy,
            )
        }
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy(
            &devices,
            package.manifest_dir(),
            package.runtime_model(),
            target,
            activation_batch_width,
            &mut catalog,
            policy,
        ),
    };

    let restoration_result = quiesce_and_verify_device_snapshots(&devices, &before);

    let report = match (calibration_result, restoration_result) {
        (Ok(Some(report)), Ok(())) => report,
        (Ok(None), Ok(())) => return Ok(None),
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
    Ok(Some(PackageCalibrationMeasurement { catalog, report }))
}
