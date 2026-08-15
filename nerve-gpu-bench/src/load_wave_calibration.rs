use std::error::Error;
use std::io;
use std::path::Path;
use std::rc::Rc;

use crate::calibration_package::{CalibrationPackage, CalibrationRuntimeConfig};
use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanRuntimeLoadWaveCalibrationTarget,
    calibrate_vulkan_runtime_load_wave, record_vulkan_runtime_load_wave_calibration_report,
};

use crate::calibration_device_state::{
    capture_device_snapshots, close_and_verify_device_snapshots,
    discover_calibration_hardware_profiles, open_calibration_targets, print_device_snapshots,
};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;

#[allow(clippy::too_many_arguments)]
pub fn run_load_wave_calibration(
    package: &Path,
    component: &str,
    selector: &str,
    phase: PackageCalibrationPhase,
    resource_indices: &[usize],
    target_id: &str,
    runtime: CalibrationRuntimeConfig,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let package = CalibrationPackage::load(package)?;
    package.reject_output_collision(output)?;
    let measurement = measure_load_wave_candidate(
        &package,
        component,
        selector,
        phase,
        resource_indices,
        target_id,
        runtime,
    )?;
    let payload = measurement.catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated load wave package={} component={} selector={} phase={:?} batch_width={} target={} requested_resources={} loaded_groups={} loaded_resources={} loaded_bytes={} warmup_ns={} measured_ns={} output={}",
        package.source_path().display(),
        component,
        selector,
        phase.execution_phase(),
        phase.activation_batch_width(),
        target_id,
        resource_indices.len(),
        measurement.report.loaded_group_count,
        measurement.report.loaded_resource_count,
        measurement.report.loaded_byte_count,
        measurement.report.warmup_ns,
        measurement.report.measured_ns,
        output.display(),
    );
    Ok(())
}

pub struct LoadWaveCalibrationMeasurement {
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub report: nerve_runtime::VulkanRuntimeLoadWaveCalibrationReport,
}

pub fn measure_load_wave_candidate(
    package: &CalibrationPackage,
    component: &str,
    selector: &str,
    phase: PackageCalibrationPhase,
    resource_indices: &[usize],
    target_id: &str,
    runtime: CalibrationRuntimeConfig,
) -> Result<LoadWaveCalibrationMeasurement, Box<dyn Error>> {
    let profiles = discover_calibration_hardware_profiles(&[target_id.to_string()])?;
    let owner_profile = profiles.get(target_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "load-wave calibration target has no hardware-process profile",
        )
    })?;
    let runtime_model = package.runtime_model_for_owner(target_id, owner_profile, runtime)?;
    measure_load_wave_candidate_for_runtime_model(
        package,
        &runtime_model,
        component,
        selector,
        phase,
        resource_indices,
        target_id,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn measure_load_wave_candidate_for_runtime_model(
    package: &CalibrationPackage,
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    component: &str,
    selector: &str,
    phase: PackageCalibrationPhase,
    resource_indices: &[usize],
    target_id: &str,
) -> Result<LoadWaveCalibrationMeasurement, Box<dyn Error>> {
    let execution_phase = phase.execution_phase();
    let activation_batch_width = phase.activation_batch_width();
    let devices = open_calibration_targets(&[target_id.to_string()])?;
    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before", &before);
    let calibration_result = calibrate_vulkan_runtime_load_wave(
        target_id,
        Rc::clone(&devices[0].1),
        package.manifest_dir(),
        runtime_model,
        &VulkanRuntimeLoadWaveCalibrationTarget {
            component_id: component.to_string(),
            selector_id: selector.to_string(),
            resource_indices: resource_indices.to_vec(),
            phase: execution_phase,
            activation_batch_width,
        },
    );
    let restoration_result = close_and_verify_device_snapshots(devices, &before);
    let report = match (calibration_result, restoration_result) {
        (Ok(report), Ok(())) => report,
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
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    record_vulkan_runtime_load_wave_calibration_report(&mut catalog, &report)?;
    Ok(LoadWaveCalibrationMeasurement { catalog, report })
}
