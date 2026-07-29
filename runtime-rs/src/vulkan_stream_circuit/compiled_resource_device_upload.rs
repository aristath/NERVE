#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanResidentCompiledResourceRange {
    pub byte_offset: usize,
    pub byte_count: usize,
}

pub struct VulkanResidentCompiledResource {
    buffer: Arc<VulkanResidentBuffer>,
    ranges: Vec<VulkanResidentCompiledResourceRange>,
    byte_count: usize,
}

impl VulkanResidentCompiledResource {
    pub fn buffer(&self) -> &VulkanResidentBuffer {
        &self.buffer
    }

    pub fn ranges(&self) -> &[VulkanResidentCompiledResourceRange] {
        &self.ranges
    }
}

impl DeviceResidentResourcePayload for VulkanResidentCompiledResource {
    fn byte_count(&self) -> usize {
        self.byte_count
    }
}

pub fn upload_loaded_compiled_resource_group(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    descriptor: &DeviceResourceGroupDescriptor,
    loaded: &LoadedCompiledResourceGroup,
) -> Result<
    DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    VulkanError,
> {
    let loaded_descriptor =
        device_resource_descriptor_from_loaded_group(loaded).map_err(|error| {
            VulkanError(format!(
                "loaded compiled resource group is invalid: {error}"
            ))
        })?;
    if loaded_descriptor != *descriptor {
        return Err(VulkanError(
            "loaded compiled resources do not match the reserved device group"
                .to_string(),
        ));
    }

    let mut allocations = Vec::with_capacity(loaded.resources.len());
    for (resource, resource_descriptor) in
        loaded.resources.iter().zip(&descriptor.resources)
    {
        let buffer = Arc::new(
            device.create_resident_buffer(resource_descriptor.byte_count)?,
        );
        let mut byte_offset = 0usize;
        let mut ranges = Vec::with_capacity(resource.ranges.len());
        for range in &resource.ranges {
            ranges.push(VulkanResidentCompiledResourceRange {
                byte_offset,
                byte_count: range.bytes.len(),
            });
            byte_offset =
                byte_offset.checked_add(range.bytes.len()).ok_or_else(|| {
                    VulkanError(
                        "compiled resource upload byte offset overflowed"
                            .to_string(),
                    )
                })?;
        }
        if byte_offset != resource_descriptor.byte_count {
            return Err(VulkanError(
                "compiled resource upload size does not match its descriptor"
                    .to_string(),
            ));
        }
        allocations.push((buffer, ranges));
    }

    let mut writes = Vec::new();
    for ((resource, _), (buffer, ranges)) in loaded
        .resources
        .iter()
        .zip(&descriptor.resources)
        .zip(&allocations)
    {
        for (range, placement) in resource.ranges.iter().zip(ranges) {
            writes.push(VulkanResidentBufferWriteRange::new(
                buffer,
                placement.byte_offset,
                &range.bytes,
            )?);
        }
    }
    let ticket = transfer.submit(&writes)?;
    transfer.wait(&ticket)?;

    let resources = descriptor
        .resources
        .iter()
        .cloned()
        .zip(allocations)
        .map(|(resource_descriptor, (buffer, ranges))| {
            let byte_count = resource_descriptor.byte_count;
            DeviceResidentResource::new(
                resource_descriptor,
                VulkanResidentCompiledResource {
                    buffer,
                    ranges,
                    byte_count,
                },
            )
        })
        .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()
        .map_err(|error| {
            VulkanError(format!(
                "uploaded compiled resource payload is invalid: {error}"
            ))
        })?;
    DeviceResidentResourceGroup::new(descriptor.clone(), resources).map_err(
        |error| {
            VulkanError(format!(
                "uploaded compiled resource group cannot be published: {error}"
            ))
        },
    )
}

fn device_resource_descriptor_from_loaded_group(
    loaded: &LoadedCompiledResourceGroup,
) -> Result<DeviceResourceGroupDescriptor, DeviceResourceResidencyError> {
    let resources = loaded
        .resources
        .iter()
        .map(|resource| {
            let byte_count =
                resource.ranges.iter().try_fold(0usize, |total, range| {
                    total.checked_add(range.bytes.len()).ok_or_else(|| {
                        DeviceResourceResidencyError::invalid_descriptor(
                            "loaded compiled resource byte count overflowed",
                        )
                    })
                })?;
            Ok(DeviceResourceDescriptor {
                id: resource.id.clone(),
                byte_count,
                compatibility: resource.compatibility.clone(),
            })
        })
        .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()?;
    DeviceResourceGroupDescriptor::new(
        loaded.id.clone(),
        loaded.resource_ids.clone(),
        loaded.dependencies.clone(),
        resources,
    )
}
