struct VulkanRawExternalResidentBuffer<'a> {
    device: &'a VulkanComputeDevice,
    buffer: vk::Buffer,
    requirements: vk::MemoryRequirements,
    requires_dedicated: bool,
}

fn external_resident_buffer_allocation_size(
    raw_buffers: &[VulkanRawExternalResidentBuffer<'_>],
) -> Result<vk::DeviceSize, VulkanError> {
    let first = raw_buffers.first().ok_or_else(|| {
        VulkanError("external resident buffer participant set is empty".to_string())
    })?;
    if raw_buffers.iter().any(|raw| raw.requires_dedicated) {
        let required_size = first.requirements.size;
        if raw_buffers
            .iter()
            .any(|raw| raw.requirements.size != required_size)
        {
            return Err(VulkanError(
                "cross-device dedicated buffer requirements disagree on allocation size"
                    .to_string(),
            ));
        }
        Ok(required_size)
    } else {
        Ok(raw_buffers
            .iter()
            .map(|raw| raw.requirements.size)
            .max()
            .expect("validated external resident buffers are nonempty"))
    }
}

impl VulkanComputeDevice {
    fn create_raw_external_resident_buffers<'a>(
        &'a self,
        peer_devices: &[&'a VulkanComputeDevice],
        byte_capacity: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Vec<VulkanRawExternalResidentBuffer<'a>>, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "shared device-local allocation capacity must not be zero".to_string(),
            ));
        }
        let devices = std::iter::once(self)
            .chain(peer_devices.iter().copied())
            .collect::<Vec<_>>();
        if devices
            .iter()
            .any(|device| !device.supports_shared_device_memory())
        {
            return Err(VulkanError(format!(
                "shared device-local memory is not supported by every participating device: {:?}",
                devices
                    .iter()
                    .filter(|device| !device.supports_shared_device_memory())
                    .map(|device| device.device_name())
                    .collect::<Vec<_>>()
            )));
        }
        for (index, device) in devices.iter().enumerate() {
            if devices[..index]
                .iter()
                .any(|existing| existing.shares_logical_device_with(device))
            {
                return Err(VulkanError(format!(
                    "shared device-local allocation repeats logical device {:?}",
                    device.device_name()
                )));
            }
        }

        let mut raw_buffers =
            Vec::<VulkanRawExternalResidentBuffer<'_>>::with_capacity(devices.len());
        let create_result = (|| {
            for device in &devices {
                unsafe {
                    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                        .handle_types(VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE);
                    let buffer_info = vk::BufferCreateInfo::default()
                        .size(byte_capacity as vk::DeviceSize)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .push_next(&mut external);
                    let buffer = device.device.create_buffer(&buffer_info, None).map_err(
                        |error| {
                            VulkanError(format!(
                                "failed to create external device-local buffer on {:?}: {error:?}",
                                device.device_name
                            ))
                        },
                    )?;
                    let mut dedicated = vk::MemoryDedicatedRequirements::default();
                    let mut requirements =
                        vk::MemoryRequirements2::default().push_next(&mut dedicated);
                    let requirements_info =
                        vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
                    device
                        .device
                        .get_buffer_memory_requirements2(&requirements_info, &mut requirements);
                    raw_buffers.push(VulkanRawExternalResidentBuffer {
                        device,
                        buffer,
                        requirements: requirements.memory_requirements,
                        requires_dedicated: dedicated.requires_dedicated_allocation == vk::TRUE,
                    });
                }
            }
            Ok::<(), VulkanError>(())
        })();
        if let Err(error) = create_result {
            unsafe {
                for raw in raw_buffers {
                    raw.device.device.destroy_buffer(raw.buffer, None);
                }
            }
            return Err(error);
        }
        Ok(raw_buffers)
    }

    pub(crate) fn shared_device_resident_buffer_memory_requirement_bytes(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<usize, VulkanError> {
        let raw_buffers = self.create_raw_external_resident_buffers(
            peer_devices,
            byte_capacity,
            resident_buffer_usage(),
        )?;
        let requirement = external_resident_buffer_allocation_size(&raw_buffers)
            .and_then(|bytes| {
                usize::try_from(bytes).map_err(|_| {
                    VulkanError(
                        "shared device-local allocation requirement exceeds usize".to_string(),
                    )
                })
            });
        unsafe {
            for raw in raw_buffers {
                raw.device.device.destroy_buffer(raw.buffer, None);
            }
        }
        requirement
    }
}
