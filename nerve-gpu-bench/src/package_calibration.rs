use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use nerve_runtime::{
    VulkanComputeDevice, VulkanComputeDeviceCatalog, VulkanDeviceLocalMemoryAccounting,
    VulkanDeviceLocalMemoryBudget, VulkanDeviceLocalMemoryPressure,
    VulkanPlacementCalibrationCatalog, VulkanResidentModelPackageManifest,
    VulkanRuntimePlacementCalibrationPolicy, VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_staged_placement_candidate_with_policy,
    calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy,
    vulkan_runtime_placement_calibration_target_for_component,
};

use crate::cli::PackageCalibrationPhase;
use crate::output::write_atomic;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceCalibrationSnapshot {
    physical_device_id: String,
    device_name: String,
    pci_address: Option<String>,
    memory_budget: VulkanDeviceLocalMemoryBudget,
    memory_accounting: VulkanDeviceLocalMemoryAccounting,
    memory_pressure: VulkanDeviceLocalMemoryPressure,
    activity: DeviceActivityObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceActivityObservation {
    gpu_busy_percent: Option<u64>,
    runtime_power_status: Option<String>,
}

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

fn capture_device_snapshots(
    devices: &[(String, Rc<VulkanComputeDevice>)],
) -> Result<Vec<DeviceCalibrationSnapshot>, Box<dyn Error>> {
    devices
        .iter()
        .map(|(physical_device_id, device)| {
            if device.physical_device_id() != physical_device_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "package calibration device identity changed",
                )
                .into());
            }
            Ok(DeviceCalibrationSnapshot {
                physical_device_id: physical_device_id.clone(),
                device_name: device.device_name().to_string(),
                pci_address: device.pci_address().map(str::to_string),
                memory_budget: device.device_local_memory_budget(),
                memory_accounting: device.device_local_memory_accounting()?,
                memory_pressure: device.device_local_memory_pressure()?,
                activity: observe_device_activity(device.pci_address()),
            })
        })
        .collect()
}

fn observe_device_activity(pci_address: Option<&str>) -> DeviceActivityObservation {
    let device_path =
        pci_address.map(|address| PathBuf::from("/sys/bus/pci/devices").join(address));
    DeviceActivityObservation {
        gpu_busy_percent: device_path
            .as_ref()
            .and_then(|path| fs::read_to_string(path.join("gpu_busy_percent")).ok())
            .and_then(|value| value.trim().parse::<u64>().ok()),
        runtime_power_status: device_path
            .as_ref()
            .and_then(|path| fs::read_to_string(path.join("power/runtime_status")).ok())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    }
}

fn print_device_snapshots(stage: &str, snapshots: &[DeviceCalibrationSnapshot]) {
    for snapshot in snapshots {
        eprintln!(
            "nerve package calibration target: stage={stage}, device={}, name={:?}, pci={}, available_bytes={}, reservable_bytes={}, remaining_bytes={}, admissible_remaining_bytes={}, tracked_bytes={}, pending_bytes={}, untracked_bytes={}, pressure_active={}, pressure_episode={}, gpu_busy_percent={}, runtime_power_status={}",
            snapshot.physical_device_id,
            snapshot.device_name,
            snapshot.pci_address.as_deref().unwrap_or("unavailable"),
            snapshot.memory_accounting.currently_available_bytes,
            snapshot.memory_accounting.reservable_bytes,
            snapshot.memory_accounting.remaining_bytes,
            snapshot.memory_accounting.admissible_remaining_bytes,
            snapshot.memory_accounting.tracked_allocation_bytes,
            snapshot.memory_accounting.pending_reservation_bytes,
            snapshot.memory_accounting.untracked_acquired_bytes,
            snapshot.memory_pressure.active,
            snapshot.memory_pressure.episode,
            snapshot
                .activity
                .gpu_busy_percent
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            snapshot
                .activity
                .runtime_power_status
                .as_deref()
                .unwrap_or("unavailable"),
        );
    }
}

fn verify_device_snapshots_restored(
    before: &[DeviceCalibrationSnapshot],
    after: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    if before.len() != after.len() {
        return Err("package calibration target set changed during execution".to_string());
    }
    for (before, after) in before.iter().zip(after) {
        if before.physical_device_id != after.physical_device_id
            || before.memory_budget != after.memory_budget
        {
            return Err(format!(
                "package calibration target {:?} changed identity or memory budget",
                before.physical_device_id
            ));
        }
        let tolerance = before.memory_budget.counter_tolerance_bytes;
        let accounting = [
            (
                "tracked allocation",
                before.memory_accounting.tracked_allocation_bytes,
                after.memory_accounting.tracked_allocation_bytes,
                0,
            ),
            (
                "pending reservation",
                before.memory_accounting.pending_reservation_bytes,
                after.memory_accounting.pending_reservation_bytes,
                0,
            ),
            (
                "untracked acquired",
                before.memory_accounting.untracked_acquired_bytes,
                after.memory_accounting.untracked_acquired_bytes,
                tolerance,
            ),
            (
                "available device-local",
                before.memory_accounting.currently_available_bytes,
                after.memory_accounting.currently_available_bytes,
                tolerance,
            ),
        ];
        for (name, before_bytes, after_bytes, allowed_difference) in accounting {
            if before_bytes.abs_diff(after_bytes) > allowed_difference {
                return Err(format!(
                    "package calibration target {:?} did not restore {name} bytes: before={before_bytes}, after={after_bytes}, tolerance={allowed_difference}",
                    before.physical_device_id
                ));
            }
        }
        if after.memory_pressure.episode != before.memory_pressure.episode
            || after.memory_pressure.active != before.memory_pressure.active
        {
            return Err(format!(
                "package calibration target {:?} entered a new memory-pressure episode",
                before.physical_device_id
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn snapshot(
        physical_device_id: &str,
        available_bytes: u64,
        tracked_bytes: u64,
        pending_bytes: u64,
        untracked_bytes: u64,
        pressure_episode: u64,
    ) -> DeviceCalibrationSnapshot {
        DeviceCalibrationSnapshot {
            physical_device_id: physical_device_id.to_string(),
            device_name: "test".to_string(),
            pci_address: None,
            memory_budget: VulkanDeviceLocalMemoryBudget {
                baseline_available_bytes: 1_000,
                reservable_bytes: 800,
                protected_headroom_bytes: 200,
                counter_tolerance_bytes: 16,
            },
            memory_accounting: VulkanDeviceLocalMemoryAccounting {
                baseline_available_bytes: 1_000,
                currently_available_bytes: available_bytes,
                reservable_bytes: 800,
                tracked_allocation_bytes: tracked_bytes,
                pending_reservation_bytes: pending_bytes,
                untracked_acquired_bytes: untracked_bytes,
                remaining_bytes: 700,
                admissible_remaining_bytes: 716,
            },
            memory_pressure: VulkanDeviceLocalMemoryPressure {
                active: false,
                episode: pressure_episode,
                observed_available_bytes: available_bytes,
                current_deficit_bytes: 0,
                peak_deficit_bytes: 0,
            },
            activity: DeviceActivityObservation {
                gpu_busy_percent: None,
                runtime_power_status: None,
            },
        }
    }

    #[test]
    fn teardown_proof_accepts_only_counter_noise_within_tolerance() {
        let before = vec![snapshot("a", 900, 5, 7, 9, 2)];
        let after = vec![snapshot("a", 884, 5, 7, 25, 2)];
        verify_device_snapshots_restored(&before, &after).unwrap();
    }

    #[test]
    fn teardown_proof_rejects_retained_owned_allocations() {
        let before = vec![snapshot("a", 900, 5, 7, 9, 2)];
        let after = vec![snapshot("a", 900, 6, 7, 9, 2)];
        let error = verify_device_snapshots_restored(&before, &after).unwrap_err();
        assert!(error.contains("tracked allocation"));
        assert!(error.contains("tolerance=0"));
    }

    #[test]
    fn teardown_proof_rejects_external_reservation_or_pressure_changes() {
        let before = vec![snapshot("a", 900, 5, 7, 9, 2)];
        let mut changed_reservation = snapshot("a", 883, 5, 7, 26, 2);
        assert!(
            verify_device_snapshots_restored(&before, &[changed_reservation.clone()])
                .unwrap_err()
                .contains("untracked acquired")
        );
        changed_reservation = snapshot("a", 900, 5, 7, 9, 3);
        assert!(
            verify_device_snapshots_restored(&before, &[changed_reservation])
                .unwrap_err()
                .contains("memory-pressure episode")
        );
    }

    #[test]
    fn teardown_proof_is_ordered_and_target_exact() {
        let before = vec![
            snapshot("owner", 900, 0, 0, 0, 0),
            snapshot("worker", 900, 0, 0, 0, 0),
        ];
        let after = vec![
            snapshot("worker", 900, 0, 0, 0, 0),
            snapshot("owner", 900, 0, 0, 0, 0),
        ];
        assert!(
            verify_device_snapshots_restored(&before, &after)
                .unwrap_err()
                .contains("changed identity")
        );
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
