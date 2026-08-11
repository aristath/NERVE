use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::Path;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanResidentModelPackageManifest,
    calibrate_vulkan_runtime_placement_phase_transfers,
    record_vulkan_runtime_transfer_calibration_report,
    vulkan_runtime_placement_transfer_byte_counts,
};

use crate::calibration_device_state::{
    capture_device_snapshots, open_calibration_devices, print_device_snapshots,
    quiesce_and_verify_device_snapshots,
};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;
use crate::package_calibration::reject_package_output_collision;

pub fn run_boundary_calibration(
    package: &Path,
    phase: PackageCalibrationPhase,
    source_id: &str,
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
    let runtime_model = manifest.mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)?;
    let frame_byte_counts = vulkan_runtime_placement_transfer_byte_counts(&runtime_model)?;
    if frame_byte_counts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "compiled package has no cross-component activation boundaries",
        )
        .into());
    }
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
    let devices = open_calibration_devices(&[source_id.to_string(), target_id.to_string()])?;
    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before", &before);
    let source = &devices[0].1;
    let target = &devices[1].1;
    let calibration_result = calibrate_vulkan_runtime_placement_phase_transfers(
        source_id,
        source,
        target_id,
        target,
        execution_phase,
        activation_batch_width,
        &frame_byte_counts,
    );
    let restoration_result = quiesce_and_verify_device_snapshots(&devices, &before);
    let reports = match (calibration_result, restoration_result) {
        (Ok(reports), Ok(())) => reports,
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
    for report in &reports {
        record_vulkan_runtime_transfer_calibration_report(&mut catalog, report)?;
    }
    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated boundaries package={} phase={:?} batch_width={} source={} target={} payload_sizes={} observations={} output={}",
        package.display(),
        execution_phase,
        activation_batch_width,
        source_id,
        target_id,
        reports.len(),
        catalog.observation_count(),
        output.display(),
    );
    Ok(())
}
