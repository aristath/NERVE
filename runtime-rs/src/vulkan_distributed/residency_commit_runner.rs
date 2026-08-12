pub(crate) struct VulkanDistributedResidencyCommitRunner {
    _predicate_clear_dispatch: VulkanResidentKernelDispatch,
    pub(crate) sequence: VulkanResidentKernelSequence,
}

fn distributed_clear_predicate_spirv_words(
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    embedded_distributed_reduction_spirv_words(
        include_bytes!(concat!(env!("OUT_DIR"), "/distributed_clear_predicate.spv")),
        "clear_predicate",
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
    let predicate_clear_dispatch = device
        .create_resident_kernel_dispatch_2d_labeled(
            &distributed_clear_predicate_spirv_words()?,
            &[VulkanResidentKernelBufferBinding::new(
                0,
                transaction_predicate,
                size_of::<u32>(),
            )
            .with_access(VulkanResidentKernelBufferAccess::Write)],
            1,
            1,
            1,
            0,
            Some(format!(
                "component={component_id} node={node_id} distributed=commit_residency",
            )),
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    let steps = shard_residency_predicates
        .iter()
        .enumerate()
        .map(|(predicate_index, predicate)| {
            VulkanResidentKernelSequenceStep::new(&predicate_clear_dispatch, &[])
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
        _predicate_clear_dispatch: predicate_clear_dispatch,
        sequence,
    })
}
