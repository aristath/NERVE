#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanRuntimePlacementTransferCalibrationReport {
    pub source_device_id: String,
    pub target_device_id: String,
    pub byte_count: usize,
    pub route: VulkanSharedResidentBufferRoute,
    pub warmup_ns: u64,
    pub measured_ns: u64,
}

pub fn vulkan_runtime_placement_transfer_byte_counts(
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<Vec<usize>, VulkanRuntimeResidencyPlanError> {
    Ok(vulkan_runtime_placement_boundary_byte_counts(runtime_model)?
        .into_iter()
        .collect())
}

/// Measures the exact activation payload sizes and physical route that the
/// mounted cross-device circuit will use. One cold replay is discarded and
/// two device-timestamped replays answer the deliberately binary placement
/// question without turning startup into a benchmark suite.
pub fn calibrate_vulkan_runtime_placement_transfers(
    source_device_id: &str,
    source: &VulkanComputeDevice,
    target_device_id: &str,
    target: &VulkanComputeDevice,
    byte_counts: &[usize],
) -> Result<Vec<VulkanRuntimePlacementTransferCalibrationReport>, VulkanError> {
    if source_device_id.is_empty()
        || target_device_id.is_empty()
        || source_device_id == target_device_id
        || source.shares_logical_device_with(target)
    {
        return Err(VulkanError(
            "runtime transfer calibration requires two distinct named physical devices"
                .to_string(),
        ));
    }
    let unique_byte_counts = byte_counts.iter().copied().collect::<BTreeSet<_>>();
    if unique_byte_counts.len() != byte_counts.len()
        || unique_byte_counts.iter().any(|byte_count| *byte_count == 0)
    {
        return Err(VulkanError(
            "runtime transfer calibration requires unique positive payload sizes".to_string(),
        ));
    }
    unique_byte_counts
        .into_iter()
        .map(|byte_count| {
            calibrate_vulkan_runtime_placement_transfer(
                source_device_id,
                source,
                target_device_id,
                target,
                byte_count,
            )
        })
        .collect()
}

fn calibrate_vulkan_runtime_placement_transfer(
    source_device_id: &str,
    source: &VulkanComputeDevice,
    target_device_id: &str,
    target: &VulkanComputeDevice,
    byte_count: usize,
) -> Result<VulkanRuntimePlacementTransferCalibrationReport, VulkanError> {
    let source_buffer = source.create_resident_buffer(byte_count)?;
    source_buffer.write_bytes(&vec![0x5a; byte_count])?;
    let target_buffer = target.create_resident_buffer(byte_count)?;
    let shared = if source.supports_opaque_fd_timeline_semaphores()
        && target.supports_opaque_fd_timeline_semaphores()
    {
        source.create_shared_resident_buffers(&[target], byte_count)?
    } else {
        let allocation = source.create_shared_host_allocation(&[target], byte_count)?;
        VulkanSharedResidentBufferSet {
            route: VulkanSharedResidentBufferRoute::SharedHost,
            buffers: vec![
                Arc::new(source.import_shared_host_buffer(Arc::clone(&allocation))?),
                Arc::new(target.import_shared_host_buffer(allocation)?),
            ],
            external_device_local_error: Some(
                "cross-device timeline semaphores are unavailable".to_string(),
            ),
        }
    };
    let source_shared = shared
        .buffers
        .first()
        .ok_or_else(|| VulkanError("transfer calibration omitted its source view".to_string()))?;
    let target_shared = shared
        .buffers
        .get(1)
        .ok_or_else(|| VulkanError("transfer calibration omitted its target view".to_string()))?;
    let source_copy = source.create_timestamped_resident_buffer_copy(
        &source_buffer,
        source_shared,
        byte_count,
    )?;
    let target_copy = target.create_timestamped_resident_buffer_copy(
        target_shared,
        &target_buffer,
        byte_count,
    )?;
    let measure = || -> Result<u64, VulkanError> {
        source_copy
            .run_with_device_duration(byte_count)?
            .checked_add(target_copy.run_with_device_duration(byte_count)?)
            .ok_or_else(|| VulkanError("runtime transfer calibration time overflowed".to_string()))
    };
    let warmup_ns = measure()?;
    let measured_ns = measure()?.min(measure()?).max(1);
    Ok(VulkanRuntimePlacementTransferCalibrationReport {
        source_device_id: source_device_id.to_string(),
        target_device_id: target_device_id.to_string(),
        byte_count,
        route: shared.route,
        warmup_ns,
        measured_ns,
    })
}
