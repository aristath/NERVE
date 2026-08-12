const DISTRIBUTED_SUM_F32_LOCAL_SIZE_X: usize = 64;
const DISTRIBUTED_SUM_F32_PUSH_CONSTANT_BYTE_COUNT: u32 = 12;

pub(crate) struct VulkanDistributedReductionRunner {
    _predicate_clear_dispatch: Option<VulkanResidentKernelDispatch>,
    _resident_dispatch: VulkanResidentKernelDispatch,
    pub(crate) sequence: VulkanResidentKernelSequence,
}

fn embedded_distributed_reduction_spirv_words(
    bytes: &[u8],
    label: &str,
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(VulkanDistributedDispatchRunnerError(format!(
            "embedded distributed {label} SPIR-V is empty or misaligned"
        )));
    }
    Ok(bytes
        .chunks_exact(size_of::<u32>())
        .map(|word| u32::from_le_bytes(word.try_into().expect("SPIR-V word is four bytes")))
        .collect())
}

fn distributed_sum_f32_spirv_words(
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    embedded_distributed_reduction_spirv_words(
        include_bytes!(concat!(env!("OUT_DIR"), "/distributed_sum_f32.spv")),
        "sum_f32",
    )
}

fn distributed_sum_f32_add_bf16_residual_spirv_words(
) -> Result<Vec<u32>, VulkanDistributedDispatchRunnerError> {
    embedded_distributed_reduction_spirv_words(
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/distributed_sum_f32_add_bf16_residual.spv"
        )),
        "sum_f32_add_bf16_residual",
    )
}

fn distributed_reduction_input_activation(
    planned_dispatch: &VulkanDistributedDispatchPlan,
    input_index: usize,
) -> Option<&VulkanDistributedActivationSlot> {
    if input_index == 0 {
        Some(&planned_dispatch.input_activation)
    } else {
        planned_dispatch
            .auxiliary_input_activations
            .get(input_index - 1)
    }
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

fn distributed_reduction_buffer_capacities(
    reduction: &VulkanDistributedReductionPlan,
    participant_count: usize,
    lane_count: usize,
) -> Result<(usize, usize), VulkanDistributedDispatchRunnerError> {
    if participant_count == 0 || lane_count == 0 {
        return Err(VulkanDistributedDispatchRunnerError(
            "distributed reduction requires at least one participant and lane".to_string(),
        ));
    }
    if matches!(
        reduction.finalization,
        VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 { .. }
    ) && !reduction.element_count.is_multiple_of(2)
    {
        return Err(VulkanDistributedDispatchRunnerError(
            "distributed BF16 residual reduction requires an even element count".to_string(),
        ));
    }
    let partial_byte_capacity = reduction
        .partial_byte_capacity
        .checked_mul(participant_count)
        .and_then(|bytes| bytes.checked_mul(lane_count))
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction plane capacity overflowed".to_string(),
            )
        })?;
    let output_frame_byte_capacity = match &reduction.finalization {
        VulkanDistributedReductionFinalizationPlan::StoreF32 => {
            reduction.partial_byte_capacity
        }
        VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 { .. } => {
            reduction.element_count.checked_mul(2).ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(
                    "distributed BF16 reduction capacity overflowed".to_string(),
                )
            })?
        }
    };
    let output_byte_capacity = output_frame_byte_capacity
        .checked_mul(lane_count)
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction output capacity overflowed".to_string(),
            )
        })?;
    Ok((partial_byte_capacity, output_byte_capacity))
}

fn create_distributed_reduction_runner(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    activation_buffers: &VulkanDistributedActivationBuffers,
    transaction_predicate: Option<&Arc<VulkanResidentBuffer>>,
    shard_residency_predicates: &[Arc<VulkanResidentBuffer>],
) -> Result<VulkanDistributedReductionRunner, VulkanDistributedDispatchRunnerError> {
    let lane_count = activation_buffers.lane_capacity;
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
    let residual = match planned_dispatch.reduction.as_ref().map(|plan| &plan.finalization) {
        Some(VulkanDistributedReductionFinalizationPlan::StoreF32) => None,
        Some(VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
            residual_input_index,
        }) => {
            let residual_activation = distributed_reduction_input_activation(
                planned_dispatch,
                *residual_input_index,
            )
            .ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no residual input {}",
                    planned_dispatch.component_id,
                    planned_dispatch.node_id,
                    residual_input_index
                ))
            })?;
            Some(
                activation_buffers
                    .activation_buffer(
                        &planned_dispatch.owner_device_id,
                        residual_activation,
                        &planned_dispatch.owner_device_id,
                    )
                    .ok_or_else(|| {
                        VulkanDistributedDispatchRunnerError(format!(
                            "distributed dispatch {}.{} has no owner residual input {}",
                            planned_dispatch.component_id,
                            planned_dispatch.node_id,
                            residual_activation.signal_id
                        ))
                    })?,
            )
        }
        None => None,
    };
    create_distributed_reduction_runner_for_buffers(
        device,
        planned_dispatch,
        lane_count,
        partials,
        output,
        residual,
        transaction_predicate,
        shard_residency_predicates,
    )
}

pub(crate) fn create_distributed_reduction_runner_for_buffers(
    device: &VulkanComputeDevice,
    planned_dispatch: &VulkanDistributedDispatchPlan,
    lane_count: usize,
    partials: &Arc<VulkanResidentBuffer>,
    output: &Arc<VulkanResidentBuffer>,
    residual: Option<&Arc<VulkanResidentBuffer>>,
    transaction_predicate: Option<&Arc<VulkanResidentBuffer>>,
    shard_residency_predicates: &[Arc<VulkanResidentBuffer>],
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
    let (partial_byte_capacity, output_byte_capacity) =
        distributed_reduction_buffer_capacities(
            reduction,
            planned_dispatch.shards.len(),
            lane_count,
        )?;
    let (spirv, work_item_count, bindings) = match &reduction.finalization {
        VulkanDistributedReductionFinalizationPlan::StoreF32 => (
            distributed_sum_f32_spirv_words()?,
            reduction.element_count,
            vec![
                VulkanResidentKernelBufferBinding::new(0, partials, partial_byte_capacity)
                    .with_access(VulkanResidentKernelBufferAccess::Read),
                VulkanResidentKernelBufferBinding::new(
                    1,
                    output,
                    output_byte_capacity,
                )
                .with_access(VulkanResidentKernelBufferAccess::Write),
            ],
        ),
        VulkanDistributedReductionFinalizationPlan::AddBf16ResidualToBf16 {
            residual_input_index: _,
        } => {
            let residual = residual.ok_or_else(|| {
                VulkanDistributedDispatchRunnerError(format!(
                    "distributed dispatch {}.{} has no owner residual buffer",
                    planned_dispatch.component_id, planned_dispatch.node_id,
                ))
            })?;
            (
                distributed_sum_f32_add_bf16_residual_spirv_words()?,
                reduction.element_count / 2,
                vec![
                    VulkanResidentKernelBufferBinding::new(0, partials, partial_byte_capacity)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(1, residual, output_byte_capacity)
                        .with_access(VulkanResidentKernelBufferAccess::Read),
                    VulkanResidentKernelBufferBinding::new(2, output, output_byte_capacity)
                        .with_access(VulkanResidentKernelBufferAccess::Write),
                ],
            )
        }
    };
    let workgroup_count_x = work_item_count
        .checked_add(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X - 1)
        .and_then(|count| count.checked_div(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X))
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| {
            VulkanDistributedDispatchRunnerError(
                "distributed reduction workgroup count overflowed".to_string(),
            )
        })?;
    let resident_dispatch = device
        .create_resident_kernel_dispatch_2d_labeled(
            &spirv,
            &bindings,
            workgroup_count_x,
            u32::try_from(lane_count).map_err(|_| {
                VulkanDistributedDispatchRunnerError(
                    "distributed reduction lane count exceeds u32".to_string(),
                )
            })?,
            u32::try_from(DISTRIBUTED_SUM_F32_LOCAL_SIZE_X)
                .expect("distributed reduction local size fits u32"),
            DISTRIBUTED_SUM_F32_PUSH_CONSTANT_BYTE_COUNT,
            Some(format!(
                "component={} node={} distributed=sum_f32 participants={} lanes={}",
                planned_dispatch.component_id,
                planned_dispatch.node_id,
                planned_dispatch.shards.len(),
                lane_count,
            )),
        )
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    let push_constants = distributed_sum_f32_push_constants(
        reduction,
        planned_dispatch.shards.len(),
        lane_count,
    )?;
    let sequence = device
        .create_resident_kernel_sequence()
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    let predicate_clear_dispatch = match (
        transaction_predicate,
        shard_residency_predicates.is_empty(),
    ) {
        (_, true) => None,
        (None, false) => {
            return Err(VulkanDistributedDispatchRunnerError(
                "distributed residency predicates require a transaction predicate".to_string(),
            ));
        }
        (Some(predicate), false) => Some(
            device
                .create_resident_kernel_dispatch_2d_labeled(
                    &distributed_clear_predicate_spirv_words()?,
                    &[VulkanResidentKernelBufferBinding::new(
                        0,
                        predicate,
                        size_of::<u32>(),
                    )
                    .with_access(VulkanResidentKernelBufferAccess::Write)],
                    1,
                    1,
                    1,
                    0,
                    Some(format!(
                        "component={} node={} distributed=commit_residency",
                        planned_dispatch.component_id, planned_dispatch.node_id,
                    )),
                )
                .map_err(VulkanDistributedDispatchRunnerError::from)?,
        ),
    };
    let mut steps = Vec::with_capacity(shard_residency_predicates.len() + 1);
    if let Some(clear) = &predicate_clear_dispatch {
        for (predicate_index, predicate) in shard_residency_predicates.iter().enumerate() {
            steps.push(
                VulkanResidentKernelSequenceStep::new(clear, &[])
                    .with_condition(
                        predicate,
                        0,
                        true,
                        u32::try_from(predicate_index + 1).unwrap_or(u32::MAX),
                    )
                    .map_err(VulkanDistributedDispatchRunnerError::from)?,
            );
        }
    }
    let reduction_step = VulkanResidentKernelSequenceStep::new(&resident_dispatch, &push_constants);
    let reduction_step = match transaction_predicate {
        Some(predicate) => reduction_step
            .with_condition(predicate, 0, false, 0)
            .map_err(VulkanDistributedDispatchRunnerError::from)?,
        None => reduction_step,
    };
    steps.push(reduction_step);
    device
        .record_resident_kernel_sequence(&sequence, &steps)
        .map_err(VulkanDistributedDispatchRunnerError::from)?;
    Ok(VulkanDistributedReductionRunner {
        _predicate_clear_dispatch: predicate_clear_dispatch,
        _resident_dispatch: resident_dispatch,
        sequence,
    })
}
