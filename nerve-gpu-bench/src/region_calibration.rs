use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io;
use std::rc::Rc;

use nerve_runtime::{
    ResourceResidencyPolicy, VulkanComputeDevice, VulkanPlacementCalibrationCatalog,
    VulkanPlacementCapacityEnvelope, VulkanPlacementDeviceExecutionIdentity,
    VulkanRuntimePlacementCalibrationPolicy, VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_region_placement_with_policy,
    record_vulkan_runtime_region_placement_calibration_report,
    try_plan_vulkan_runtime_region_placement_calibration,
    try_plan_vulkan_runtime_serialized_region_placement_calibration,
    vulkan_safe_host_available_bytes,
};

use crate::calibration_device_state::{
    capture_device_snapshots, open_calibration_targets, print_device_snapshots,
    quiesce_and_verify_device_snapshots,
};
use crate::calibration_package::CalibrationPackage;
use crate::cli::PackageCalibrationPhase;

pub struct RegionCalibrationMeasurement {
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub planned_case_count: usize,
    pub measured_case_count: usize,
    pub unavailable_case_count: usize,
}

pub fn measure_region_candidates_for_runtime_model(
    package: &CalibrationPackage,
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    phase: PackageCalibrationPhase,
    target_ids: &[String],
    source_catalog: &VulkanPlacementCalibrationCatalog,
    resource_residency_policy: ResourceResidencyPolicy,
) -> Result<RegionCalibrationMeasurement, Box<dyn Error>> {
    let opened = open_calibration_targets(target_ids)?;
    let before = capture_device_snapshots(&opened)?;
    print_device_snapshots("before-region", &before);
    let measurement = (|| {
        let devices = opened.iter().cloned().collect::<BTreeMap<_, _>>();
        let capacity = region_capacity_envelope(&devices, &before)?;
        if capacity.available_bytes_by_device.is_empty() {
            return Ok(RegionCalibrationMeasurement {
                catalog: VulkanPlacementCalibrationCatalog::default(),
                planned_case_count: 2,
                measured_case_count: 0,
                unavailable_case_count: 2,
            });
        }
        let logical_device_id_by_physical_device = devices
            .iter()
            .map(|(logical_id, device)| {
                (device.physical_device_id().to_string(), logical_id.clone())
            })
            .collect::<BTreeMap<_, _>>();
        if logical_device_id_by_physical_device.len() != devices.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "region calibration requires one logical binding per physical device",
            )
            .into());
        }
        let runtime_phase = targeted_phase(phase);
        let serialized = try_plan_vulkan_runtime_serialized_region_placement_calibration(
            runtime_model,
            source_catalog,
            &capacity,
            runtime_phase,
            &logical_device_id_by_physical_device,
        )?;
        let hybrid = try_plan_vulkan_runtime_region_placement_calibration(
            runtime_model,
            source_catalog,
            &capacity,
            runtime_phase,
            &logical_device_id_by_physical_device,
        )?;
        let (serialized, hybrid) = match (serialized, hybrid) {
            (Some(serialized), Some(hybrid)) => (serialized, hybrid),
            (None, _) | (_, None) => {
                return Ok(RegionCalibrationMeasurement {
                    catalog: VulkanPlacementCalibrationCatalog::default(),
                    planned_case_count: 2,
                    measured_case_count: 0,
                    unavailable_case_count: 2,
                });
            }
        };
        let policy = region_calibration_policy(&capacity)?;
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let serialized_report = calibrate_vulkan_runtime_region_placement_with_policy(
            &devices,
            package.manifest_dir(),
            &serialized,
            source_catalog,
            resource_residency_policy,
            policy.clone(),
        )?;
        record_vulkan_runtime_region_placement_calibration_report(
            &mut catalog,
            &serialized_report,
        )?;
        let mut measured_case_count = 1usize;
        let mut planned_case_count = 1usize;
        if hybrid.target.execution_case != serialized.target.execution_case {
            planned_case_count += 1;
            let hybrid_report = calibrate_vulkan_runtime_region_placement_with_policy(
                &devices,
                package.manifest_dir(),
                &hybrid,
                source_catalog,
                resource_residency_policy,
                policy,
            )?;
            record_vulkan_runtime_region_placement_calibration_report(
                &mut catalog,
                &hybrid_report,
            )?;
            measured_case_count += 1;
        }
        Ok::<_, Box<dyn Error>>(RegionCalibrationMeasurement {
            catalog,
            planned_case_count,
            measured_case_count,
            unavailable_case_count: 0,
        })
    })();
    let restoration = quiesce_and_verify_device_snapshots(&opened, &before);
    match (measurement, restoration) {
        (Ok(measurement), Ok(())) => Ok(measurement),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(io::Error::other(error).into()),
        (Err(error), Err(restoration)) => Err(io::Error::other(format!(
            "{error}; region teardown proof also failed: {restoration}",
        ))
        .into()),
    }
}

fn targeted_phase(phase: PackageCalibrationPhase) -> VulkanTargetedComponentExecutionPhase {
    match phase {
        PackageCalibrationPhase::Decode => VulkanTargetedComponentExecutionPhase::Decode,
        PackageCalibrationPhase::Prefill {
            activation_batch_width,
        } => VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
    }
}

fn region_capacity_envelope(
    devices: &BTreeMap<String, Rc<VulkanComputeDevice>>,
    snapshots: &[crate::calibration_device_state::DeviceCalibrationSnapshot],
) -> Result<VulkanPlacementCapacityEnvelope, Box<dyn Error>> {
    let physical_device_ids = devices
        .values()
        .map(|device| device.physical_device_id())
        .collect::<BTreeSet<_>>();
    if physical_device_ids.len() != devices.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "region calibration requires one logical binding per physical device",
        )
        .into());
    }
    let snapshot_by_physical_id = snapshots
        .iter()
        .map(|snapshot| (snapshot.physical_device_id.as_str(), snapshot))
        .collect::<BTreeMap<_, _>>();
    if snapshot_by_physical_id.len() != snapshots.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "region calibration snapshots repeat a physical device",
        )
        .into());
    }
    if snapshot_by_physical_id.len() != devices.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "region calibration snapshots do not exactly cover its opened targets",
        )
        .into());
    }
    let available_bytes_by_device = devices
        .values()
        .map(|device| {
            let physical_id = device.physical_device_id();
            let snapshot = snapshot_by_physical_id.get(physical_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "region calibration device has no pre-workload snapshot",
                )
            })?;
            let Some(available) = usable_region_capacity(
                physical_id,
                snapshot.memory_accounting.admissible_remaining_bytes,
            )?
            else {
                return Ok(None);
            };
            Ok(Some((
                VulkanPlacementDeviceExecutionIdentity {
                    physical_device_id: physical_id.to_string(),
                    api_version: device.api_version(),
                    driver_version: device.driver_version(),
                },
                available,
            )))
        })
        .collect::<Result<Vec<_>, io::Error>>()?
        .into_iter()
        .flatten()
        .collect::<BTreeMap<_, _>>();
    Ok(VulkanPlacementCapacityEnvelope {
        available_bytes_by_device,
        host_available_bytes: vulkan_safe_host_available_bytes()?,
    })
}

fn usable_region_capacity(
    physical_device_id: &str,
    admissible_remaining_bytes: u64,
) -> Result<Option<usize>, io::Error> {
    if physical_device_id.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "region calibration capacity has an empty physical device id",
        ));
    }
    if admissible_remaining_bytes == 0 {
        return Ok(None);
    }
    usize::try_from(admissible_remaining_bytes)
        .map(Some)
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "region calibration capacity exceeds usize",
            )
        })
}

fn region_calibration_policy(
    capacity: &VulkanPlacementCapacityEnvelope,
) -> Result<VulkanRuntimePlacementCalibrationPolicy, io::Error> {
    let mut maximum_by_physical_device = BTreeMap::new();
    let mut maximum_total = 0usize;
    for (device, bytes) in &capacity.available_bytes_by_device {
        if device.physical_device_id.is_empty()
            || *bytes == 0
            || maximum_by_physical_device
                .insert(device.physical_device_id.clone(), *bytes)
                .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "region calibration capacity requires unique physical devices with positive capacity",
            ));
        }
        maximum_total = maximum_total.checked_add(*bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "region calibration total capacity overflowed",
            )
        })?;
    }
    if maximum_total == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "region calibration has no usable target capacity",
        ));
    }
    Ok(VulkanRuntimePlacementCalibrationPolicy {
        maximum_total_resident_parameter_bytes: maximum_total,
        maximum_resident_parameter_bytes_by_physical_device: maximum_by_physical_device,
        ..VulkanRuntimePlacementCalibrationPolicy::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(id: &str, driver_version: u32) -> VulkanPlacementDeviceExecutionIdentity {
        VulkanPlacementDeviceExecutionIdentity {
            physical_device_id: id.to_string(),
            api_version: 1,
            driver_version,
        }
    }

    #[test]
    fn fully_reserved_target_remains_selected_but_is_not_given_work() {
        assert_eq!(usable_region_capacity("gpu0", 0).unwrap(), None);
        assert_eq!(usable_region_capacity("gpu0", 128).unwrap(), Some(128));
        assert!(usable_region_capacity("", 128).is_err());
    }

    #[test]
    fn region_policy_rejects_duplicate_physical_identity_or_empty_capacity() {
        let duplicate = VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([
                (device("gpu0", 1), 64),
                (device("gpu0", 2), 64),
            ]),
            host_available_bytes: 0,
        };
        assert!(region_calibration_policy(&duplicate).is_err());

        let empty = VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::new(),
            host_available_bytes: 0,
        };
        assert!(region_calibration_policy(&empty).is_err());
    }

    #[test]
    fn region_policy_preserves_exact_positive_capacity_vector() {
        let capacity = VulkanPlacementCapacityEnvelope {
            available_bytes_by_device: BTreeMap::from([
                (device("gpu0", 1), 64),
                (device("gpu1", 1), 96),
            ]),
            host_available_bytes: 32,
        };

        let policy = region_calibration_policy(&capacity).unwrap();

        assert_eq!(policy.maximum_total_resident_parameter_bytes, 160);
        assert_eq!(
            policy.maximum_resident_parameter_bytes_by_physical_device,
            BTreeMap::from([("gpu0".to_string(), 64), ("gpu1".to_string(), 96)])
        );
    }
}
