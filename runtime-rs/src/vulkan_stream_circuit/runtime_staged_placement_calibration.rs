/// Measures one planner-requested physical candidate through its exact
/// prefixes. Every participant first establishes a canonical single-device
/// reference. The requested owner/worker order is then measured as a pair and
/// expanded one participant at a time. An unavailable or invalid prefix makes
/// the larger candidate unavailable; it is never assigned an inferred cost.
pub fn calibrate_vulkan_runtime_staged_placement_candidate_with_policy(
    devices: &[(String, Rc<VulkanComputeDevice>)],
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    catalog: &mut VulkanPlacementCalibrationCatalog,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    calibrate_vulkan_runtime_staged_placement_phase_candidate_with_policy(
        devices,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        VulkanTargetedComponentExecutionPhase::Decode,
        catalog,
        policy,
    )
}

/// Prefill counterpart to the staged decode calibration. It exercises the
/// same compiler-emitted component and physical contracts at the requested
/// batch width; the width is therefore part of the catalog's exact behavior
/// and shape identity rather than an inferred scaling factor.
pub fn calibrate_vulkan_runtime_staged_prefill_placement_candidate_with_policy(
    devices: &[(String, Rc<VulkanComputeDevice>)],
    manifest_dir: impl AsRef<Path>,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    activation_batch_width: usize,
    catalog: &mut VulkanPlacementCalibrationCatalog,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    calibrate_vulkan_runtime_staged_placement_phase_candidate_with_policy(
        devices,
        manifest_dir.as_ref(),
        runtime_model,
        target,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width,
        },
        catalog,
        policy,
    )
}

fn calibrate_vulkan_runtime_staged_placement_phase_candidate_with_policy(
    devices: &[(String, Rc<VulkanComputeDevice>)],
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
    target: &VulkanRuntimePlacementCalibrationTarget,
    phase: VulkanTargetedComponentExecutionPhase,
    catalog: &mut VulkanPlacementCalibrationCatalog,
    policy: VulkanRuntimePlacementCalibrationPolicy,
) -> Result<
    Option<VulkanRuntimeDistributedPlacementCalibrationReport>,
    VulkanResidentTokenModelPackageError,
> {
    validate_vulkan_runtime_staged_placement_phase(phase)?;
    let stages = vulkan_runtime_staged_placement_device_groups(devices)?;
    let common_parameter_budget = vulkan_runtime_staged_common_parameter_budget(
        devices.iter().map(|(physical_id, _)| physical_id.as_str()),
        &policy,
    )?;
    let started = Instant::now();
    let mut final_report = None;
    let mut sample_fraction_millionths = None;
    for stage in stages {
        let remaining = policy
            .maximum_duration
            .checked_sub(started.elapsed())
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "staged runtime placement calibration exceeded its complete-chain duration",
                )
            })?;
        let stage_policy = VulkanRuntimePlacementCalibrationPolicy {
            maximum_duration: remaining,
            maximum_total_resident_parameter_bytes: common_parameter_budget,
            ..policy.clone()
        };
        let expected_ids = stage
            .iter()
            .map(|(physical_id, _)| physical_id.clone())
            .collect::<Vec<_>>();
        let report = calibrate_vulkan_runtime_distributed_placement_phase_candidate_with_policy(
            stage,
            manifest_dir,
            runtime_model,
            target,
            phase,
            sample_fraction_millionths,
            stage_policy,
        )?;
        let Some(report) = report else {
            return Ok(None);
        };
        if report.physical_device_ids != expected_ids
            || report.execution_case.owner_physical_device_id != expected_ids[0]
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "staged runtime placement calibration changed its requested participants or owner",
            ));
        }
        if sample_fraction_millionths
            .is_some_and(|expected| expected != report.sample_fraction_millionths)
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "staged runtime placement calibration changed its fixed sampled workload",
            ));
        }
        sample_fraction_millionths = Some(report.sample_fraction_millionths);
        record_vulkan_runtime_distributed_calibration_report(catalog, &report)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        final_report = Some(report);
    }
    Ok(final_report)
}

fn vulkan_runtime_staged_common_parameter_budget<'a>(
    physical_device_ids: impl IntoIterator<Item = &'a str>,
    policy: &VulkanRuntimePlacementCalibrationPolicy,
) -> Result<usize, VulkanResidentTokenModelPackageError> {
    let common = physical_device_ids
        .into_iter()
        .map(|physical_id| policy.parameter_capacity_for_physical_device(physical_id))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap_or(policy.maximum_total_resident_parameter_bytes)
        .min(policy.maximum_total_resident_parameter_bytes);
    if common == 0 {
        return Err(VulkanResidentTokenModelPackageError::new(
            "staged runtime placement calibration has no common positive parameter budget",
        ));
    }
    Ok(common)
}

fn validate_vulkan_runtime_staged_placement_phase(
    phase: VulkanTargetedComponentExecutionPhase,
) -> Result<(), VulkanResidentTokenModelPackageError> {
    if matches!(
        phase,
        VulkanTargetedComponentExecutionPhase::Prefill {
            activation_batch_width: 0
        }
    ) {
        return Err(VulkanResidentTokenModelPackageError::new(
            "staged prefill calibration requires a positive activation batch width",
        ));
    }
    Ok(())
}

fn vulkan_runtime_staged_placement_device_groups(
    devices: &[(String, Rc<VulkanComputeDevice>)],
) -> Result<Vec<Vec<(String, Rc<VulkanComputeDevice>)>>, VulkanResidentTokenModelPackageError> {
    let ids = devices
        .iter()
        .map(|(physical_id, _)| physical_id.clone())
        .collect::<Vec<_>>();
    vulkan_runtime_staged_placement_indices(&ids).map(|stages| {
        stages
            .into_iter()
            .map(|stage| {
                stage
                    .into_iter()
                    .map(|index| devices[index].clone())
                    .collect()
            })
            .collect()
    })
}

fn vulkan_runtime_staged_placement_indices(
    physical_device_ids: &[String],
) -> Result<Vec<Vec<usize>>, VulkanResidentTokenModelPackageError> {
    if physical_device_ids.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(
            "staged runtime placement calibration requires at least one physical device",
        ));
    }
    let mut ids = BTreeSet::new();
    if physical_device_ids
        .iter()
        .any(|physical_id| physical_id.is_empty() || !ids.insert(physical_id.as_str()))
    {
        return Err(VulkanResidentTokenModelPackageError::new(
            "staged runtime placement calibration requires distinct nonempty physical device IDs",
        ));
    }
    let mut stages = (0..physical_device_ids.len())
        .map(|index| vec![index])
        .collect::<Vec<_>>();
    stages.extend((2..=physical_device_ids.len()).map(|count| (0..count).collect()));
    Ok(stages)
}

#[cfg(test)]
mod runtime_staged_placement_calibration_tests {
    use super::*;

    fn staged_id_groups(ids: &[&str]) -> Result<Vec<Vec<String>>, String> {
        let ids = ids.iter().map(|id| (*id).to_string()).collect::<Vec<_>>();
        vulkan_runtime_staged_placement_indices(&ids)
            .map(|stages| {
                stages
                    .into_iter()
                    .map(|stage| stage.into_iter().map(|index| ids[index].clone()).collect())
                    .collect()
            })
            .map_err(|error| error.to_string())
    }

    #[test]
    fn stages_every_single_then_directly_validates_each_requested_prefix() {
        assert_eq!(
            staged_id_groups(&["owner", "worker-b", "worker-a", "worker-c"]).unwrap(),
            [
                vec!["owner"],
                vec!["worker-b"],
                vec!["worker-a"],
                vec!["worker-c"],
                vec!["owner", "worker-b"],
                vec!["owner", "worker-b", "worker-a"],
                vec!["owner", "worker-b", "worker-a", "worker-c"],
            ]
            .map(|group| group.into_iter().map(str::to_string).collect::<Vec<_>>()),
        );
    }

    #[test]
    fn rejects_missing_repeated_or_empty_participants() {
        assert!(staged_id_groups(&[]).is_err());
        assert!(staged_id_groups(&["owner", ""]).is_err());
        assert!(staged_id_groups(&["owner", "owner"]).is_err());
    }

    #[test]
    fn rejects_zero_width_prefill_before_opening_a_calibration_session() {
        let error = validate_vulkan_runtime_staged_placement_phase(
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 0,
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("positive activation batch width")
        );
        validate_vulkan_runtime_staged_placement_phase(
            VulkanTargetedComponentExecutionPhase::Decode,
        )
        .unwrap();
        validate_vulkan_runtime_staged_placement_phase(
            VulkanTargetedComponentExecutionPhase::Prefill {
                activation_batch_width: 64,
            },
        )
        .unwrap();
    }

    #[test]
    fn common_sampling_budget_uses_the_smallest_exact_participant_capacity() {
        let policy = VulkanRuntimePlacementCalibrationPolicy {
            maximum_total_resident_parameter_bytes: 1_000,
            maximum_resident_parameter_bytes_by_physical_device: BTreeMap::from([
                ("gpu-a".to_string(), 800),
                ("gpu-b".to_string(), 300),
                ("gpu-c".to_string(), 600),
            ]),
            ..VulkanRuntimePlacementCalibrationPolicy::default()
        };

        assert_eq!(
            vulkan_runtime_staged_common_parameter_budget(["gpu-a", "gpu-b", "gpu-c"], &policy,)
                .unwrap(),
            300,
        );
        assert!(
            vulkan_runtime_staged_common_parameter_budget(["gpu-a", "missing"], &policy)
                .unwrap_err()
                .to_string()
                .contains("no positive parameter capacity")
        );
    }
}
