use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::Path;
use std::rc::Rc;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanResidentModelPackageManifest,
    VulkanRuntimeLoadWaveCalibrationTarget, calibrate_vulkan_runtime_load_wave,
    record_vulkan_runtime_load_wave_calibration_report,
};

use crate::calibration_device_state::{
    capture_device_snapshots, open_calibration_devices, print_device_snapshots,
    quiesce_and_verify_device_snapshots,
};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;
use crate::package_calibration::reject_package_output_collision;

#[allow(clippy::too_many_arguments)]
pub fn run_load_wave_calibration(
    package: &Path,
    component: &str,
    selector: &str,
    phase: PackageCalibrationPhase,
    resource_indices: &[usize],
    target_id: &str,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    reject_package_output_collision(package, output)?;
    let manifest =
        VulkanResidentModelPackageManifest::from_json_file(package).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to load compiled package {}: {error}",
                    package.display()
                ),
            )
        })?;
    let manifest_dir = package.parent().unwrap_or_else(|| Path::new("."));
    let runtime_model = manifest.mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)?;
    let (execution_phase, activation_batch_width) = match phase {
        PackageCalibrationPhase::Decode => (
            nerve_runtime::execution_contracts::ExecutionPhase::Decode,
            1,
        ),
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => (
            nerve_runtime::execution_contracts::ExecutionPhase::Prefill,
            activation_batch_width,
        ),
    };
    let devices = open_calibration_devices(&[target_id.to_string()])?;
    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before", &before);
    let calibration_result = calibrate_vulkan_runtime_load_wave(
        target_id,
        Rc::clone(&devices[0].1),
        manifest_dir,
        &runtime_model,
        &VulkanRuntimeLoadWaveCalibrationTarget {
            component_id: component.to_string(),
            selector_id: selector.to_string(),
            resource_indices: resource_indices.to_vec(),
            phase: execution_phase,
            activation_batch_width,
        },
    );
    let restoration_result = quiesce_and_verify_device_snapshots(&devices, &before);
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
    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated load wave package={} component={} selector={} phase={:?} batch_width={} target={} requested_resources={} loaded_groups={} loaded_resources={} loaded_bytes={} warmup_ns={} measured_ns={} output={}",
        package.display(),
        component,
        selector,
        execution_phase,
        activation_batch_width,
        target_id,
        resource_indices.len(),
        report.loaded_group_count,
        report.loaded_resource_count,
        report.loaded_byte_count,
        report.warmup_ns,
        report.measured_ns,
        output.display(),
    );
    Ok(())
}
