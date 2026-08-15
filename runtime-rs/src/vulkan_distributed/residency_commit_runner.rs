pub(crate) struct VulkanDistributedResidencyCommitRunner {
    _predicate_commit_dispatches: Vec<VulkanResidentKernelDispatch>,
    pub(crate) sequence: VulkanResidentKernelSequence,
    // The recorded sequence stores raw Vulkan buffer handles. Retain the
    // corresponding buffer objects until after the sequence is destroyed.
    _transaction_predicate: Arc<VulkanResidentBuffer>,
    _shard_residency_predicates: Vec<Arc<VulkanResidentBuffer>>,
}

fn distributed_commit_residency_fault_spirv_words(
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    embedded_distributed_reduction_spirv_words(
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/distributed_commit_residency_fault.spv"
        )),
        "commit_residency_fault",
    )
}

fn create_distributed_residency_commit_runner(
    device: &VulkanComputeDevice,
    component_id: &str,
    node_id: &str,
    transaction_predicate: Option<&Arc<VulkanResidentBuffer>>,
    shard_residency_predicates: &[Arc<VulkanResidentBuffer>],
) -> Result<VulkanDistributedResidencyCommitRunner, VulkanDistributedDispatchRunnerError> {
    if shard_residency_predicates.is_empty() {
        return Err(VulkanDistributedDispatchRunnerError(
            "distributed residency commit requires at least one shard predicate".to_string(),
        ));
    }
    let transaction_predicate = transaction_predicate.ok_or_else(|| {
        VulkanDistributedDispatchRunnerError(
            "distributed shard residency commit requires a transaction predicate".to_string(),
        )
    })?;
    let predicate_commit_dispatches = create_distributed_residency_fault_commit_dispatches(
        device,
        component_id,
        node_id,
        transaction_predicate,
        shard_residency_predicates,
    )?;
    let steps = predicate_commit_dispatches
        .iter()
        .zip(shard_residency_predicates)
        .enumerate()
        .map(|(predicate_index, (dispatch, predicate))| {
            VulkanResidentKernelSequenceStep::new(dispatch, &[])
                .with_condition(
                    predicate,
                    0,
                    true,
                    u32::try_from(predicate_index + 1).unwrap_or(u32::MAX),
                )
                .map_err(VulkanDistributedDispatchRunnerError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sequence = device
        .create_resident_kernel_sequence()
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    device
        .record_resident_kernel_sequence(&sequence, &steps)
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    Ok(VulkanDistributedResidencyCommitRunner {
        _predicate_commit_dispatches: predicate_commit_dispatches,
        sequence,
        _transaction_predicate: Arc::clone(transaction_predicate),
        _shard_residency_predicates: shard_residency_predicates.to_vec(),
    })
}

fn create_distributed_residency_fault_commit_dispatches(
    device: &VulkanComputeDevice,
    component_id: &str,
    node_id: &str,
    transaction_predicate: &Arc<VulkanResidentBuffer>,
    shard_residency_predicates: &[Arc<VulkanResidentBuffer>],
) -> Result<Vec<VulkanResidentKernelDispatch>, VulkanDistributedDispatchRunnerError> {
    if transaction_predicate.byte_capacity()
        < VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY
        || shard_residency_predicates.iter().any(|predicate| {
            predicate.byte_capacity() < VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY
        })
    {
        return Err(VulkanDistributedDispatchRunnerError(
            "distributed residency fault publication requires two-word predicates"
                .to_string(),
        ));
    }
    let spirv = distributed_commit_residency_fault_spirv_words()?;
    shard_residency_predicates
        .iter()
        .enumerate()
        .map(|(predicate_index, predicate)| {
            device
                .create_resident_kernel_dispatch_2d_labeled(
                    &spirv,
                    &[
                        VulkanResidentKernelBufferBinding::new(
                            0,
                            transaction_predicate,
                            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::ReadWrite),
                        VulkanResidentKernelBufferBinding::new(
                            1,
                            predicate,
                            VULKAN_DEMAND_FEEDBACK_PREDICATE_BYTE_CAPACITY,
                        )
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    ],
                    1,
                    1,
                    1,
                    0,
                    Some(format!(
                        "component={component_id} node={node_id} distributed=commit_residency shard={predicate_index}",
                    )),
                )
                .map_err(VulkanDistributedDispatchRunnerError::from)
        })
        .collect()
}
