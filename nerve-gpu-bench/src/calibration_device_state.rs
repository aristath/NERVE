use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;

use nerve_runtime::{
    HardwareProcessProfile, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanDeviceLocalMemoryAccounting, VulkanDeviceLocalMemoryBudget,
    VulkanDeviceLocalMemoryPressure, VulkanDeviceLocalMemoryRestorationSnapshot,
    verify_vulkan_device_local_memory_restoration,
};

pub fn discover_calibration_hardware_profiles(
    ordered_target_ids: &[String],
) -> Result<BTreeMap<String, HardwareProcessProfile>, Box<dyn Error>> {
    let allowed_target_ids = validate_calibration_target_ids(ordered_target_ids)?;
    let device_catalog =
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowed_target_ids)?;
    let profiles = device_catalog.available_hardware_profiles()?;
    hardware_profiles_for_target_ids(ordered_target_ids, &profiles)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCalibrationSnapshot {
    pub physical_device_id: String,
    pub device_name: String,
    pub pci_address: Option<String>,
    pub memory_budget: VulkanDeviceLocalMemoryBudget,
    pub memory_accounting: VulkanDeviceLocalMemoryAccounting,
    pub memory_pressure: VulkanDeviceLocalMemoryPressure,
    activity: DeviceActivityObservation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceActivityObservation {
    gpu_busy_percent: Option<u64>,
    runtime_power_status: Option<String>,
}

pub fn open_calibration_targets(
    ordered_target_ids: &[String],
) -> Result<Vec<(String, Rc<VulkanComputeDevice>)>, Box<dyn Error>> {
    let allowed_target_ids = validate_calibration_target_ids(ordered_target_ids)?;
    let device_catalog =
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowed_target_ids)?;
    let available_by_id = device_catalog
        .available_compute_devices()
        .iter()
        .map(|device| (device.physical_device_id.clone(), device.clone()))
        .collect::<BTreeMap<_, _>>();
    ordered_target_ids
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
        .collect()
}

fn validate_calibration_target_ids(
    ordered_target_ids: &[String],
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let allowed_target_ids = ordered_target_ids.iter().cloned().collect::<BTreeSet<_>>();
    if allowed_target_ids.len() != ordered_target_ids.len() || allowed_target_ids.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration requires distinct ordered target identities",
        )
        .into());
    }
    Ok(allowed_target_ids)
}

fn hardware_profiles_for_target_ids(
    ordered_target_ids: &[String],
    profiles: &[HardwareProcessProfile],
) -> Result<BTreeMap<String, HardwareProcessProfile>, Box<dyn Error>> {
    ordered_target_ids
        .iter()
        .map(|physical_device_id| {
            profiles
                .iter()
                .find(|profile| profile.hardware_identity.stable_device_id == *physical_device_id)
                .cloned()
                .map(|profile| (physical_device_id.clone(), profile))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "selected target {physical_device_id:?} has no hardware-process profile"
                        ),
                    )
                    .into()
                })
        })
        .collect()
}

pub fn capture_device_snapshots(
    devices: &[(String, Rc<VulkanComputeDevice>)],
) -> Result<Vec<DeviceCalibrationSnapshot>, Box<dyn Error>> {
    devices
        .iter()
        .map(|(physical_device_id, device)| {
            if device.physical_device_id() != physical_device_id {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "calibration device identity changed",
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

pub fn print_device_snapshots(stage: &str, snapshots: &[DeviceCalibrationSnapshot]) {
    for snapshot in snapshots {
        eprintln!(
            "nerve calibration target: stage={stage}, device={}, name={:?}, pci={}, available_bytes={}, reservable_bytes={}, remaining_bytes={}, admissible_remaining_bytes={}, tracked_bytes={}, pending_bytes={}, untracked_bytes={}, pressure_active={}, pressure_episode={}, gpu_busy_percent={}, runtime_power_status={}",
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

pub fn verify_device_snapshots_restored(
    before: &[DeviceCalibrationSnapshot],
    after: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    let before = before
        .iter()
        .map(device_calibration_restoration_snapshot)
        .collect::<Vec<_>>();
    let after = after
        .iter()
        .map(device_calibration_restoration_snapshot)
        .collect::<Vec<_>>();
    let report = verify_vulkan_device_local_memory_restoration(&before, &after);
    if report.complete {
        return Ok(());
    }
    Err(report.errors.join("; "))
}

fn device_calibration_restoration_snapshot(
    snapshot: &DeviceCalibrationSnapshot,
) -> VulkanDeviceLocalMemoryRestorationSnapshot {
    VulkanDeviceLocalMemoryRestorationSnapshot {
        physical_device_id: snapshot.physical_device_id.clone(),
        device_name: snapshot.device_name.clone(),
        pci_address: snapshot.pci_address.clone(),
        api_version: 0,
        driver_version: 0,
        memory_budget: snapshot.memory_budget,
        memory_accounting: snapshot.memory_accounting,
        memory_pressure: snapshot.memory_pressure,
    }
}

pub fn quiesce_and_verify_device_snapshots(
    devices: &[(String, Rc<VulkanComputeDevice>)],
    before: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    let quiesce_result = devices.iter().try_for_each(|(_, device)| device.quiesce());
    let after_result = capture_device_snapshots(devices);
    match (quiesce_result, after_result) {
        (Ok(()), Ok(after)) => {
            print_device_snapshots("after", &after);
            verify_device_snapshots_restored(before, &after)
        }
        (Err(error), _) => Err(format!(
            "calibration could not quiesce selected targets before teardown proof: {error}"
        )),
        (Ok(()), Err(error)) => Err(format!(
            "calibration could not capture post-workload target state: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                .contains("memory-pressure state")
        );
    }

    #[test]
    fn teardown_proof_is_target_exact_but_order_independent() {
        let before = vec![
            snapshot("owner", 900, 0, 0, 0, 0),
            snapshot("worker", 900, 0, 0, 0, 0),
        ];
        let after = vec![
            snapshot("worker", 900, 0, 0, 0, 0),
            snapshot("owner", 900, 0, 0, 0, 0),
        ];
        verify_device_snapshots_restored(&before, &after).unwrap();

        let changed = vec![snapshot("other", 900, 0, 0, 0, 0)];
        assert!(
            verify_device_snapshots_restored(&before, &changed)
                .unwrap_err()
                .contains("set changed")
        );
    }
}
