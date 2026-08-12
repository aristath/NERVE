use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::io;
use std::rc::Rc;

use nerve_runtime::{
    VulkanPlacementCalibrationCatalog, VulkanRuntimePlacementCalibrationPolicy,
    VulkanRuntimePlacementCalibrationTarget,
    VulkanRuntimeSelectedResourceExecutionCalibrationTarget, VulkanTargetedComponentExecutionPhase,
    calibrate_vulkan_runtime_load_wave, calibrate_vulkan_runtime_selected_resource_execution,
    instantiate_runtime_resource_contract, record_vulkan_runtime_load_wave_calibration_report,
    record_vulkan_runtime_selected_resource_execution_calibration_report,
    vulkan_runtime_distributed_contract_candidates,
    vulkan_runtime_selected_resource_execution_calibration_targets,
};

use crate::calibration_device_state::{
    capture_device_snapshots, open_calibration_targets, print_device_snapshots,
    quiesce_and_verify_device_snapshots,
};
use crate::calibration_package::CalibrationPackage;
use crate::cli::PackageCalibrationPhase;

pub struct SelectedResourceCalibrationMeasurement {
    pub catalog: VulkanPlacementCalibrationCatalog,
    pub planned_case_count: usize,
    pub measured_case_count: usize,
    pub unavailable_case_count: usize,
}

/// Measures every exact selected-resource execution class exposed by one
/// representative compiled component on one physical target. Discovery and
/// execution use the same open device and contract lowering. Every execution
/// report is paired with its exact singleton lazy-load wave before the class is
/// published; partial evidence is never returned.
pub fn measure_selected_resource_classes_for_runtime_model(
    package: &CalibrationPackage,
    runtime_model: &nerve_runtime::VulkanResidentRuntimeModel,
    component: &VulkanRuntimePlacementCalibrationTarget,
    phase: PackageCalibrationPhase,
    target_id: &str,
) -> Result<SelectedResourceCalibrationMeasurement, Box<dyn Error>> {
    let contract = instantiate_runtime_resource_contract(runtime_model)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    let selector_ids = contract
        .selectors
        .iter()
        .filter(|selector| {
            selector.execution_scope == runtime_model.execution_scope
                && selector.component_id == component.component_id
        })
        .map(|selector| selector.id.clone())
        .collect::<BTreeSet<_>>();
    let execution_phase = targeted_phase(phase);
    let contract_candidates =
        vulkan_runtime_distributed_contract_candidates(runtime_model, component, execution_phase)?;
    if selector_ids.is_empty() || contract_candidates.is_empty() {
        return Ok(SelectedResourceCalibrationMeasurement {
            catalog: VulkanPlacementCalibrationCatalog::default(),
            planned_case_count: 0,
            measured_case_count: 0,
            unavailable_case_count: 0,
        });
    }

    let devices = open_calibration_targets(&[target_id.to_string()])?;
    let before = capture_device_snapshots(&devices)?;
    print_device_snapshots("before selected-resource classes", &before);
    let capacity = usize::try_from(before[0].memory_accounting.admissible_remaining_bytes)
        .ok()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("selected target {target_id:?} has no safe remaining parameter capacity"),
            )
        })?;
    let policy = VulkanRuntimePlacementCalibrationPolicy {
        maximum_total_resident_parameter_bytes: capacity,
        maximum_resident_parameter_bytes_by_physical_device: BTreeMap::from([(
            target_id.to_string(),
            capacity,
        )]),
        ..VulkanRuntimePlacementCalibrationPolicy::default()
    };
    let device = Rc::clone(&devices[0].1);

    let calibration_result = (|| {
        let mut catalog = VulkanPlacementCalibrationCatalog::default();
        let mut seen = BTreeSet::new();
        let mut planned_case_count = 0usize;
        let mut measured_case_count = 0usize;
        let mut unavailable_case_count = 0usize;
        for candidate in &contract_candidates {
            for selector_id in &selector_ids {
                let targets = vulkan_runtime_selected_resource_execution_calibration_targets(
                    &device,
                    package.manifest_dir(),
                    runtime_model,
                    component,
                    selector_id,
                    execution_phase,
                    &candidate.contract_ids,
                )?;
                for target in targets {
                    if !seen.insert(selected_resource_target_key(&target)) {
                        continue;
                    }
                    planned_case_count += 1;
                    let Some(execution_report) =
                        calibrate_vulkan_runtime_selected_resource_execution(
                            target_id,
                            Rc::clone(&device),
                            package.manifest_dir(),
                            runtime_model,
                            &target,
                            policy.clone(),
                        )?
                    else {
                        unavailable_case_count += 1;
                        continue;
                    };
                    let load_wave_report = calibrate_vulkan_runtime_load_wave(
                        target_id,
                        Rc::clone(&device),
                        package.manifest_dir(),
                        runtime_model,
                        &nerve_runtime::VulkanRuntimeLoadWaveCalibrationTarget {
                            component_id: target.component.component_id.clone(),
                            selector_id: target.selector_id.clone(),
                            resource_indices: vec![target.resource_index],
                            phase: phase.execution_phase(),
                            activation_batch_width: phase.activation_batch_width(),
                        },
                    )?;
                    let mut paired = VulkanPlacementCalibrationCatalog::default();
                    record_vulkan_runtime_selected_resource_execution_calibration_report(
                        &mut paired,
                        &execution_report,
                    )
                    .map_err(|error| {
                        nerve_runtime::VulkanResidentTokenModelPackageError::new(error.to_string())
                    })?;
                    record_vulkan_runtime_load_wave_calibration_report(
                        &mut paired,
                        &load_wave_report,
                    )
                    .map_err(|error| {
                        nerve_runtime::VulkanResidentTokenModelPackageError::new(error.to_string())
                    })?;
                    let class = execution_report
                        .execution_class_calibration(&load_wave_report)
                        .map_err(|error| {
                            nerve_runtime::VulkanResidentTokenModelPackageError::new(
                                error.to_string(),
                            )
                        })?;
                    paired
                        .record_selected_resource_execution_class(class)
                        .map_err(|error| {
                            nerve_runtime::VulkanResidentTokenModelPackageError::new(
                                error.to_string(),
                            )
                        })?;
                    catalog.merge(&paired).map_err(|error| {
                        nerve_runtime::VulkanResidentTokenModelPackageError::new(error.to_string())
                    })?;
                    measured_case_count += 1;
                }
            }
        }
        Ok::<_, nerve_runtime::VulkanResidentTokenModelPackageError>(
            SelectedResourceCalibrationMeasurement {
                catalog,
                planned_case_count,
                measured_case_count,
                unavailable_case_count,
            },
        )
    })();
    let restoration_result = quiesce_and_verify_device_snapshots(&devices, &before);
    match (calibration_result, restoration_result) {
        (Ok(measurement), Ok(())) => Ok(measurement),
        (Err(error), Ok(())) => Err(error.into()),
        (Ok(_), Err(restoration_error)) => Err(io::Error::other(restoration_error).into()),
        (Err(error), Err(restoration_error)) => Err(io::Error::other(format!(
            "{error}; teardown proof also failed: {restoration_error}",
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

fn selected_resource_target_key(
    target: &VulkanRuntimeSelectedResourceExecutionCalibrationTarget,
) -> (
    String,
    String,
    usize,
    String,
    Vec<String>,
    &'static str,
    usize,
) {
    let (phase, batch_width) = match target.phase {
        VulkanTargetedComponentExecutionPhase::Decode => ("decode", 1),
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        } => ("prefill", activation_batch_width),
    };
    (
        target.component.signature_id.clone(),
        target.selector_id.clone(),
        target.resource_index,
        target.resource_execution_class_id.clone(),
        target.selected_contract_ids.iter().cloned().collect(),
        phase,
        batch_width,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn target(
        contract: &str,
        phase: VulkanTargetedComponentExecutionPhase,
    ) -> VulkanRuntimeSelectedResourceExecutionCalibrationTarget {
        VulkanRuntimeSelectedResourceExecutionCalibrationTarget {
            component: VulkanRuntimePlacementCalibrationTarget {
                signature_id: digest('a'),
                component_id: "block".to_string(),
                component_ids: vec!["block".to_string()],
                terminal_node_id: "down".to_string(),
                implementation: "sparse-ffn".to_string(),
                planned_resident_parameter_bytes: 1,
            },
            selector_id: "experts".to_string(),
            resource_index: 2,
            resource_execution_class_id: digest('b'),
            phase,
            selected_contract_ids: BTreeSet::from([contract.to_string()]),
        }
    }

    #[test]
    fn target_identity_keeps_contract_and_phase_specific_evidence_separate() {
        let decode_a = target("contract-a", VulkanTargetedComponentExecutionPhase::Decode);
        let decode_b = target("contract-b", VulkanTargetedComponentExecutionPhase::Decode);
        let prefill_a = target(
            "contract-a",
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 8,
            },
        );

        assert_ne!(
            selected_resource_target_key(&decode_a),
            selected_resource_target_key(&decode_b)
        );
        assert_ne!(
            selected_resource_target_key(&decode_a),
            selected_resource_target_key(&prefill_a)
        );
        assert_eq!(
            selected_resource_target_key(&decode_a),
            selected_resource_target_key(&decode_a.clone())
        );
    }
}
