use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::rc::Rc;

use nerve_runtime::{
    VulkanComputeDeviceCatalog, VulkanPlacementCalibrationCatalog,
    VulkanResidentModelPackageManifest, VulkanRuntimePlacementCalibrationPolicy,
    VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_staged_placement_candidate_with_policy,
    calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy,
    vulkan_runtime_placement_calibration_target_for_component,
};

use crate::calibration_device_state::{
    capture_device_snapshots, print_device_snapshots, verify_device_snapshots_restored,
};
use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;

pub fn run_package_calibration(
    package: &Path,
    component: &str,
    phase: PackageCalibrationPhase,
    ordered_target_ids: &[String],
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

    let allowed_target_ids = ordered_target_ids.iter().cloned().collect::<BTreeSet<_>>();
    if allowed_target_ids.len() != ordered_target_ids.len() || allowed_target_ids.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "package calibration requires distinct ordered target identities",
        )
        .into());
    }
    let device_catalog =
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowed_target_ids)?;
    let available_by_id = device_catalog
        .available_compute_devices()
        .iter()
        .map(|device| (device.physical_device_id.clone(), device.clone()))
        .collect::<BTreeMap<_, _>>();
    let devices = ordered_target_ids
        .iter()
        .map(|physical_device_id| {
            let info = available_by_id.get(physical_device_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("selected target {physical_device_id:?} is unavailable"),
                )
            })?;
            let device = Rc::new(device_catalog.open_device_uuid(info.device_uuid)?);
            if device.physical_device_id() != physical_device_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "opened target {:?} instead of requested target {physical_device_id:?}",
                        device.physical_device_id()
                    ),
                )
                .into());
            }
            Ok((physical_device_id.clone(), device))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before", &before);
    let maximum_resident_parameter_bytes = before
        .iter()
        .map(|snapshot| snapshot.memory_accounting.remaining_bytes)
        .min()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "selected targets have no common safe remaining parameter capacity",
            )
        })?;
    let policy = VulkanRuntimePlacementCalibrationPolicy {
        maximum_resident_parameter_bytes,
        ..VulkanRuntimePlacementCalibrationPolicy::default()
    };
    let mut catalog = VulkanPlacementCalibrationCatalog::default();
    let calibration_result = match phase {
        PackageCalibrationPhase::Decode => {
            calibrate_vulkan_runtime_staged_placement_candidate_with_policy(
                &devices,
                manifest_dir,
                &runtime_model,
                &target,
                &mut catalog,
                policy,
            )
        }
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy(
            &devices,
            manifest_dir,
            &runtime_model,
            &target,
            activation_batch_width,
            &mut catalog,
            policy,
        ),
    };

    let quiesce_result = devices.iter().try_for_each(|(_, device)| device.quiesce());
    let after_result = capture_device_snapshots(&devices);
    let restoration_result = match (quiesce_result, after_result) {
        (Ok(()), Ok(after)) => {
            print_device_snapshots("after", &after);
            verify_device_snapshots_restored(&before, &after)
        }
        (Err(error), _) => Err(format!(
            "package calibration could not quiesce selected targets before teardown proof: {error}"
        )),
        (Ok(()), Err(error)) => Err(format!(
            "package calibration could not capture post-workload target state: {error}"
        )),
    };

    let report = match (calibration_result, restoration_result) {
        (Ok(Some(report)), Ok(())) => report,
        (Ok(None), Ok(())) => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "the requested package placement candidate is unavailable",
            )
            .into());
        }
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

    let payload = catalog.to_json_bytes()?;
    write_atomic(output, &payload)?;
    println!(
        "calibrated package={} signature={} representative={} requested_component={} phase={} batch_width={} targets={:?} observations={} sampled={} measured_ns={} output={}",
        package.display(),
        target.signature_id,
        target.component_id,
        component,
        report.phase,
        report.activation_batch_width,
        report.physical_device_ids,
        catalog.observation_count(),
        report.sampled_workload,
        report.measured_execution_ns,
        output.display(),
    );
    Ok(())
}

fn reject_package_output_collision(package: &Path, output: &Path) -> Result<(), io::Error> {
    if output.exists() && fs::canonicalize(package)? == fs::canonicalize(output)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "package calibration output must not replace the compiled package manifest",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temporary_test_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-gpu-bench-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn package_catalog_cannot_overwrite_its_source_manifest() {
        let directory = temporary_test_directory("source-collision");
        fs::create_dir_all(&directory).unwrap();
        let package = directory.join("package.json");
        fs::write(&package, b"source").unwrap();
        let alias = directory.join(".").join("package.json");

        let error = reject_package_output_collision(&package, &alias).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("must not replace"));
        fs::remove_dir_all(directory).unwrap();
    }
}
