use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::rc::Rc;

use nerve_runtime::{
    HardwareProcessProfile, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanDeviceLocalMemoryAccounting, VulkanDeviceLocalMemoryBudget,
    VulkanDeviceLocalMemoryPressure, VulkanPhysicalDeviceMemoryObservation,
    capture_vulkan_physical_device_memory_observations,
    verify_vulkan_physical_device_memory_restoration,
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
    physical_memory: VulkanPhysicalDeviceMemoryObservation,
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
            device.initialize_execution_context_memory_floor()?;
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
    let physical_device_ids = devices
        .iter()
        .map(|(physical_device_id, _)| physical_device_id.clone())
        .collect::<BTreeSet<_>>();
    let catalog =
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&physical_device_ids)?;
    let physical_memory_by_id =
        capture_vulkan_physical_device_memory_observations(&catalog, &physical_device_ids)?
            .into_iter()
            .map(|observation| (observation.physical_device_id.clone(), observation))
            .collect::<BTreeMap<_, _>>();
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
            let physical_memory = physical_memory_by_id
                .get(physical_device_id)
                .cloned()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "selected target {physical_device_id:?} has no physical memory observation"
                        ),
                    )
                })?;
            Ok(DeviceCalibrationSnapshot {
                physical_device_id: physical_device_id.clone(),
                device_name: device.device_name().to_string(),
                pci_address: device.pci_address().map(str::to_string),
                memory_budget: device.device_local_memory_budget(),
                memory_accounting: device.device_local_memory_accounting()?,
                memory_pressure: device.device_local_memory_pressure()?,
                physical_memory,
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
            "nerve calibration target: stage={stage}, device={}, name={:?}, pci={}, available_bytes={}, reservable_bytes={}, remaining_bytes={}, admissible_remaining_bytes={}, tracked_bytes={}, pending_bytes={}, untracked_bytes={}, physical_usage_bytes={}, physical_available_bytes={}, pressure_active={}, pressure_episode={}, gpu_busy_percent={}, runtime_power_status={}",
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
            snapshot
                .physical_memory
                .usage_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            snapshot
                .physical_memory
                .available_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
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

#[cfg(test)]
pub fn verify_device_snapshots_restored(
    before: &[DeviceCalibrationSnapshot],
    after: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    let before = before
        .iter()
        .map(|snapshot| snapshot.physical_memory.clone())
        .collect::<Vec<_>>();
    let after = after
        .iter()
        .map(|snapshot| snapshot.physical_memory.clone())
        .collect::<Vec<_>>();
    let report = verify_vulkan_physical_device_memory_restoration(&before, &after);
    if report.complete {
        return Ok(());
    }
    Err(report.errors.join("; "))
}

/// Closes every calibration-owned logical device before proving that physical
/// device-local memory returned to its pre-workload state. Reopening clean
/// contexts for the post-state snapshot keeps the before/after observations at
/// the same initialized-context boundary while excluding driver allocations
/// whose lifetime is the Vulkan logical device itself.
pub fn close_and_verify_device_snapshots(
    devices: Vec<(String, Rc<VulkanComputeDevice>)>,
    before: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    let target_ids = devices
        .iter()
        .map(|(physical_device_id, _)| physical_device_id.clone())
        .collect::<Vec<_>>();
    quiesce_and_release_calibration_devices(&devices)?;
    let logical_after = capture_device_snapshots(&devices).map_err(|error| {
        format!("calibration could not capture its released logical state: {error}")
    })?;
    let logical_release = verify_logical_calibration_release(before, &logical_after);
    let retained_devices = devices
        .iter()
        .filter_map(|(physical_device_id, device)| {
            let strong_count = Rc::strong_count(device);
            (strong_count != 1).then(|| format!("{physical_device_id:?} ({strong_count} owners)"))
        })
        .collect::<Vec<_>>();
    let ownership_release = if retained_devices.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "calibration teardown found retained NERVE device owners: {}",
            retained_devices.join(", ")
        ))
    };
    drop(devices);

    let allowed_target_ids = target_ids.iter().cloned().collect::<BTreeSet<_>>();
    let catalog =
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowed_target_ids)
            .map_err(|error| {
                format!("calibration could not rediscover targets for teardown proof: {error}")
            })?;
    let physical_after = capture_vulkan_physical_device_memory_observations(
        &catalog,
        &allowed_target_ids,
    )
    .map_err(|error| {
        format!("calibration could not capture post-workload physical memory state: {error}")
    })?;
    print_physical_memory_observations("after-close", &physical_after);
    let physical_before = before
        .iter()
        .map(|snapshot| snapshot.physical_memory.clone())
        .collect::<Vec<_>>();
    let physical_report =
        verify_vulkan_physical_device_memory_restoration(&physical_before, &physical_after);
    let physical_release = if physical_report.complete {
        Ok(())
    } else {
        Err(physical_report.errors.join("; "))
    };
    combine_teardown_proofs(logical_release, ownership_release, physical_release)
}

fn print_physical_memory_observations(
    stage: &str,
    observations: &[VulkanPhysicalDeviceMemoryObservation],
) {
    for observation in observations {
        eprintln!(
            "nerve calibration physical target: stage={stage}, device={}, name={:?}, pci={}, heap_bytes={}, budget_bytes={}, usage_bytes={}, available_bytes={}",
            observation.physical_device_id,
            observation.device_name,
            observation.pci_address.as_deref().unwrap_or("unavailable"),
            observation.physical_heap_bytes,
            observation
                .budget_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            observation
                .usage_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
            observation
                .available_bytes
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "unavailable".to_string()),
        );
    }
}

fn verify_logical_calibration_release(
    before: &[DeviceCalibrationSnapshot],
    after: &[DeviceCalibrationSnapshot],
) -> Result<(), String> {
    let before_by_id = before
        .iter()
        .map(|snapshot| (snapshot.physical_device_id.as_str(), snapshot))
        .collect::<BTreeMap<_, _>>();
    let after_by_id = after
        .iter()
        .map(|snapshot| (snapshot.physical_device_id.as_str(), snapshot))
        .collect::<BTreeMap<_, _>>();
    let mut errors = Vec::new();
    if before.is_empty() || before_by_id.len() != before.len() || after_by_id.len() != after.len() {
        errors.push("logical teardown proof requires distinct selected targets".to_string());
    }
    if before_by_id.keys().collect::<BTreeSet<_>>() != after_by_id.keys().collect::<BTreeSet<_>>() {
        errors.push("logical teardown target set changed".to_string());
    }
    for (physical_device_id, before) in before_by_id {
        let Some(after) = after_by_id.get(physical_device_id) else {
            continue;
        };
        if before.device_name != after.device_name
            || before.pci_address != after.pci_address
            || before.memory_budget != after.memory_budget
        {
            errors.push(format!(
                "logical target {physical_device_id:?} changed identity or admission budget"
            ));
        }
        for (name, before_bytes, after_bytes) in [
            (
                "tracked allocation",
                before.memory_accounting.tracked_allocation_bytes,
                after.memory_accounting.tracked_allocation_bytes,
            ),
            (
                "pending reservation",
                before.memory_accounting.pending_reservation_bytes,
                after.memory_accounting.pending_reservation_bytes,
            ),
        ] {
            if before_bytes != after_bytes {
                errors.push(format!(
                    "logical target {physical_device_id:?} did not restore {name} bytes: before={before_bytes}, after={after_bytes}, tolerance=0"
                ));
            }
        }
        if before.memory_pressure.active != after.memory_pressure.active
            || before.memory_pressure.episode != after.memory_pressure.episode
        {
            errors.push(format!(
                "logical target {physical_device_id:?} memory-pressure state changed: active={}->{}, episode={}->{}",
                before.memory_pressure.active,
                after.memory_pressure.active,
                before.memory_pressure.episode,
                after.memory_pressure.episode,
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn combine_teardown_proofs(
    logical_release: Result<(), String>,
    ownership_release: Result<(), String>,
    physical_release: Result<(), String>,
) -> Result<(), String> {
    let errors = [logical_release, ownership_release, physical_release]
        .into_iter()
        .filter_map(Result::err)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn quiesce_and_release_calibration_devices(
    devices: &[(String, Rc<VulkanComputeDevice>)],
) -> Result<(), String> {
    devices
        .iter()
        .try_for_each(|(_, device)| {
        device.quiesce()?;
        device.release_cached_execution_resources_after_quiescence()?;
        Ok::<(), nerve_runtime::VulkanError>(())
        })
        .map_err(|error| {
            format!(
            "calibration could not quiesce selected targets and release execution caches before teardown proof: {error}"
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        physical_device_id: &str,
        physical_usage_bytes: u64,
        tracked_bytes: u64,
        pending_bytes: u64,
        pressure_episode: u64,
    ) -> DeviceCalibrationSnapshot {
        let physical_budget_bytes = 30 * 1024 * 1024 * 1024;
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
                currently_available_bytes: 900,
                reservable_bytes: 800,
                tracked_allocation_bytes: tracked_bytes,
                pending_reservation_bytes: pending_bytes,
                untracked_acquired_bytes: 9,
                remaining_bytes: 700,
                admissible_remaining_bytes: 716,
            },
            memory_pressure: VulkanDeviceLocalMemoryPressure {
                active: false,
                episode: pressure_episode,
                observed_available_bytes: 900,
                current_deficit_bytes: 0,
                peak_deficit_bytes: 0,
            },
            physical_memory: VulkanPhysicalDeviceMemoryObservation {
                physical_device_id: physical_device_id.to_string(),
                device_name: "test".to_string(),
                pci_address: None,
                api_version: 1,
                driver_version: 2,
                heap_index: 0,
                physical_heap_bytes: 32 * 1024 * 1024 * 1024,
                memory_budget_supported: true,
                budget_bytes: Some(physical_budget_bytes),
                usage_bytes: Some(physical_usage_bytes),
                available_bytes: Some(physical_budget_bytes.saturating_sub(physical_usage_bytes)),
            },
            activity: DeviceActivityObservation {
                gpu_busy_percent: None,
                runtime_power_status: None,
            },
        }
    }

    #[test]
    fn teardown_proof_accepts_only_counter_noise_within_tolerance() {
        let before = vec![snapshot("a", 256 * 1024 * 1024, 5, 7, 2)];
        let after = vec![snapshot("a", 264 * 1024 * 1024, 5, 7, 2)];
        verify_device_snapshots_restored(&before, &after).unwrap();
    }

    #[test]
    fn teardown_proof_rejects_retained_owned_allocations() {
        let before = vec![snapshot("a", 256 * 1024 * 1024, 5, 7, 2)];
        let after = vec![snapshot("a", 256 * 1024 * 1024, 6, 7, 2)];
        let error = verify_logical_calibration_release(&before, &after).unwrap_err();
        assert!(error.contains("tracked allocation"));
        assert!(error.contains("tolerance=0"));
    }

    #[test]
    fn teardown_proof_rejects_external_reservation_or_pressure_changes() {
        let before = vec![snapshot("a", 256 * 1024 * 1024, 5, 7, 2)];
        let changed_reservation = snapshot("a", 320 * 1024 * 1024, 5, 7, 2);
        assert!(
            verify_device_snapshots_restored(&before, &[changed_reservation])
                .unwrap_err()
                .contains("usage bytes")
        );
        let changed_pressure = snapshot("a", 256 * 1024 * 1024, 5, 7, 3);
        assert!(
            verify_logical_calibration_release(&before, &[changed_pressure])
                .unwrap_err()
                .contains("memory-pressure state")
        );
    }

    #[test]
    fn teardown_proof_is_target_exact_but_order_independent() {
        let before = vec![
            snapshot("owner", 256 * 1024 * 1024, 0, 0, 0),
            snapshot("worker", 256 * 1024 * 1024, 0, 0, 0),
        ];
        let after = vec![
            snapshot("worker", 256 * 1024 * 1024, 0, 0, 0),
            snapshot("owner", 256 * 1024 * 1024, 0, 0, 0),
        ];
        verify_device_snapshots_restored(&before, &after).unwrap();

        let changed = vec![snapshot("other", 256 * 1024 * 1024, 0, 0, 0)];
        assert!(
            verify_device_snapshots_restored(&before, &changed)
                .unwrap_err()
                .contains("set changed")
        );
    }
}
