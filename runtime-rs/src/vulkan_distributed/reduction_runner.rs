const DISTRIBUTED_SUM_F32_LOCAL_SIZE_X: usize = 64;
const DISTRIBUTED_SUM_F32_PUSH_CONSTANT_BYTE_COUNT: u32 = 12;

struct VulkanDistributedReductionRunner {
    _resident_dispatch: VulkanResidentKernelDispatch,
    sequence: VulkanResidentKernelSequence,
}

fn distributed_sum_f32_spirv_words(
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/distributed_sum_f32.spv"));
    if bytes.is_empty() || !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(VulkanDistributedDispatchRunnerError(
            "embedded distributed sum_f32 SPIR-V is empty or misaligned".to_string(),
        ));
    }
    Ok(bytes
        .chunks_exact(size_of::<u32>())
        .map(|word| u32::from_le_bytes(word.try_into().expect("SPIR-V word is four bytes")))
        .collect())
}

fn distributed_sum_f32_push_constants(
    reduction: &VulkanDistributedReductionPlan,
    participant_count: usize,
    lane_count: usize,
) -> Result<Vec<u8>, VulkanDistributedDispatchRunnerError> {
    let fields = [
        u32::try_from(reduction.element_count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction element count exceeds u32".to_string(),
            )
        })?,
        u32::try_from(participant_count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction participant count exceeds u32".to_string(),
            )
        })?,
        u32::try_from(lane_count).map_err(|_| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction lane count exceeds u32".to_string(),
            )
        })?,
    ];
    Ok(fields.into_iter().flat_map(u32::to_le_bytes).collect())
}

fn create_distributed_reduction_runner(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    activation_buffers: &VulkanDistributedActivationBuffers,
) -> Result<VulkanDistributedReductionRunner, VulkanDistributedDispatchRunnerError> {
    let reduction = planned_dispatch.reduction.as_ref().ok_or_else(|| {
        VulkanDistributedDispatchRunnerError(format!(
            "distributed dispatch {}.{} has no reduction plan",
            planned_dispatch.component_id, planned_dispatch.node_id
        ))
    })?;
    if reduction.operation != ReductionOperation::SumF32 {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "distributed dispatch {}.{} has unsupported reduction {:?}",
            planned_dispatch.component_id, planned_dispatch.node_id, reduction.operation
        )));
    }
    if activation_buffers.lane_capacity != 1 {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "scalar distributed reduction requires one lane, activation buffers have {}",
            activation_buffers.lane_capacity
        )));
    }
    let partials = activation_buffers
        .reduction_partial_buffer(
            &planned_dispatch.owner_device_id,
            planned_dispatch.dispatch_index,
            &planned_dispatch.owner_device_id,
        )
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {}.{} has no owner reduction-plane view",
                planned_dispatch.component_id, planned_dispatch.node_id
            ))
        })?;
    let output = activation_buffers
        .activation_buffer(
            &planned_dispatch.owner_device_id,
            &planned_dispatch.output_activation,
            &planned_dispatch.owner_device_id,
        )
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(format!(
                "distributed dispatch {}.{} has no owner reduction output",
                planned_dispatch.component_id, planned_dispatch.node_id
            ))
        })?;
    let partial_byte_capacity = reduction
        .partial_byte_capacity
        .checked_mul(planned_dispatch.shards.len())
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction plane capacity overflowed".to_string(),
            )
        })?;
    let workgroup_count_x = reduction
        .element_count
        .checked_add(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X - 1)
        .and_then(|count| count.checked_div(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X))
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction workgroup count overflowed".to_string(),
            )
        })?;
    let spirv = distributed_sum_f32_spirv_words()?;
    let resident_dispatch = device
        .create_resident_kernel_dispatch_labeled(
            &spirv,
            &[
                VulkanResidentKernelBufferBinding::new(0, partials, partial_byte_capacity)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(
                    1,
                    output,
                    reduction.partial_byte_capacity,
                )
                .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
            workgroup_count_x,
            u32::try_from(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X)
                .expect("distributed reduction local size fits u32"),
            DISTRIBUTED_SUM_F32_PUSH_CONSTANT_BYTE_COUNT,
            Some(format!(
                "component={} node={} distributed=sum_f32 participants={}",
                planned_dispatch.component_id,
                planned_dispatch.node_id,
                planned_dispatch.shards.len()
            )),
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    let push_constants =
        distributed_sum_f32_push_constants(reduction, planned_dispatch.shards.len(), 1)?;
    let sequence = device
        .create_resident_kernel_sequence()
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    device
        .record_resident_kernel_sequence(
            &sequence,
            &[VulkanResidentKernelSequenceStep::new(
                &resident_dispatch,
                &push_constants,
            )],
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    Ok(VulkanDistributedReductionRunner {
        _resident_dispatch: resident_dispatch,
        sequence,
    })
}
