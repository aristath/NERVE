fn load_resident_package_transducer_parameter_buffers(
    device: &VulkanComputeDevice,
    device_id: &str,
    resource_plan: &StreamCircuitResourcePlan,
    tensor_index: &TensorIndex,
    parameter_pool: Option<&VulkanResidentBufferPool>,
) -> Result<VulkanPermanentParameterBuffers, VulkanResidentTokenModelPackageError> {
    let transducer_parameter_plan = VulkanPermanentParameterBufferPlan::from_transducer_parameters(
        device_id,
        resource_plan,
        Some(tensor_index),
    )
    .map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to create transducer parameter plan: {error}"
        ))
    })?;
    let transducer_parameter_buffers = match parameter_pool {
        Some(pool) => transducer_parameter_plan
            .allocate_and_load_from_pool(tensor_index, pool)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to acquire pooled transducer parameters: {error}"
                ))
            })?,
        None => {
            let buffers = transducer_parameter_plan
                .allocate_buffers(device)
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to allocate transducer parameter buffers: {error}"
                    ))
                })?;
            buffers
                .load_from_tensor_index(tensor_index)
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load transducer parameters: {error}"
                    ))
                })?;
            buffers
        }
    };
    Ok(transducer_parameter_buffers)
}

fn load_resident_package_transducer_parameter_buffers_for(
    device: &VulkanComputeDevice,
    device_id: &str,
    resource_plan: &StreamCircuitResourcePlan,
    tensor_index: &TensorIndex,
    transducer_id: &str,
    parameter_pool: Option<&VulkanResidentBufferPool>,
) -> Result<VulkanPermanentParameterBuffers, VulkanResidentTokenModelPackageError> {
    let transducer_parameter_plan =
        VulkanPermanentParameterBufferPlan::from_transducer_parameters_for(
            device_id,
            resource_plan,
            Some(tensor_index),
            transducer_id,
        )
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to create {transducer_id} parameter plan: {error}"
            ))
        })?;
    let transducer_parameter_buffers = match parameter_pool {
        Some(pool) => transducer_parameter_plan
            .allocate_and_load_from_pool(tensor_index, pool)
            .map_err(|error| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "failed to acquire pooled {transducer_id} parameters: {error}"
                ))
            })?,
        None => {
            let buffers = transducer_parameter_plan
                .allocate_buffers(device)
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to allocate {transducer_id} parameter buffers: {error}"
                    ))
                })?;
            buffers
                .load_from_tensor_index(tensor_index)
                .map_err(|error| {
                    VulkanResidentTokenModelPackageError::new(format!(
                        "failed to load {transducer_id} parameters: {error}"
                    ))
                })?;
            buffers
        }
    };
    Ok(transducer_parameter_buffers)
}

fn load_resident_package_parameter_buffers_for_tensors(
    device: &VulkanComputeDevice,
    device_id: &str,
    tensor_index: &TensorIndex,
    tensors: &[&str],
) -> Result<VulkanPermanentParameterBuffers, VulkanResidentTokenModelPackageError> {
    let mut parameters = Vec::with_capacity(tensors.len());
    let mut total_byte_capacity = 0usize;
    let mut unique = BTreeSet::new();
    for tensor in tensors {
        if !unique.insert(*tensor) {
            return Err(VulkanResidentTokenModelPackageError::new(format!(
                "resident parameter selection contains duplicate tensor {tensor:?}"
            )));
        }
        let metadata = tensor_index.tensors.get(*tensor).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package tensor index has no tensor {tensor:?}"
            ))
        })?;
        let byte_capacity = metadata.byte_count.ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(format!(
                "resident package tensor {tensor:?} has no byte count"
            ))
        })?;
        total_byte_capacity = total_byte_capacity
            .checked_add(byte_capacity)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "selected resident parameter byte count overflowed",
                )
            })?;
        parameters.push(VulkanPermanentParameterBuffer {
            buffer_index: parameters.len(),
            tensor: (*tensor).to_string(),
            dtype: Some(metadata.dtype.clone()),
            shape: Some(metadata.shape.clone()),
            byte_capacity: Some(byte_capacity),
            use_count: 1,
        });
    }
    let plan = VulkanPermanentParameterBufferPlan {
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        device_id: device_id.to_string(),
        parameter_count: parameters.len(),
        parameters,
        total_byte_capacity: Some(total_byte_capacity),
        unresolved_tensors: Vec::new(),
    };
    let buffers = plan.allocate_buffers(device).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to allocate selected resident parameter buffers on {device_id:?}: {error}"
        ))
    })?;
    buffers
        .load_from_tensor_index(tensor_index)
        .map_err(|error| {
            VulkanResidentTokenModelPackageError::new(format!(
                "failed to load selected resident parameter buffers on {device_id:?}: {error}"
            ))
        })?;
    Ok(buffers)
}
