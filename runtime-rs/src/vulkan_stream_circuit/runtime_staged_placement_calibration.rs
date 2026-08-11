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
    let stages = vulkan_runtime_staged_placement_device_groups(devices)?;
    let started = Instant::now();
    let mut final_report = None;
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
            ..policy
        };
        let expected_ids = stage
            .iter()
            .map(|(physical_id, _)| physical_id.clone())
            .collect::<Vec<_>>();
        let Some(report) = calibrate_vulkan_runtime_distributed_placement_candidate_with_policy(
            stage,
            manifest_dir.as_ref(),
            runtime_model,
            target,
            stage_policy,
        )?
        else {
            return Ok(None);
        };
        if report.physical_device_ids != expected_ids
            || report.execution_case.owner_physical_device_id != expected_ids[0]
        {
            return Err(VulkanResidentTokenModelPackageError::new(
                "staged runtime placement calibration changed its requested participants or owner",
            ));
        }
        record_vulkan_runtime_distributed_calibration_report(catalog, &report)
            .map_err(|error| VulkanResidentTokenModelPackageError::new(error.to_string()))?;
        final_report = Some(report);
    }
    Ok(final_report)
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
}
