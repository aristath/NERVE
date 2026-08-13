impl VulkanComputeDevice {
    pub(crate) fn shared_device_resident_buffer_memory_requirement_bytes(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<usize, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "shared device-local requirement capacity must not be zero".to_string(),
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
                "shared device-local memory is not supported by every requirement participant: {:?}",
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
                    "shared device-local requirement repeats logical device {:?}",
                    device.device_name(),
                )));
            }
        }

        let mut raw_buffers = Vec::<(
            &VulkanComputeDevice,
            vk::Buffer,
            vk::MemoryRequirements,
            bool,
        )>::with_capacity(devices.len());
        let create_result = (|| {
            for device in &devices {
                unsafe {
                    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                        .handle_types(VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE);
                    let buffer_info = vk::BufferCreateInfo::default()
                        .size(byte_capacity as vk::DeviceSize)
                        .usage(resident_buffer_usage())
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .push_next(&mut external);
                    let buffer = device.device.create_buffer(&buffer_info, None).map_err(
                        |error| {
                            VulkanError(format!(
                                "failed to query external device-local buffer on {:?}: {error:?}",
                                device.device_name(),
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
                    raw_buffers.push((
                        device,
                        buffer,
                        requirements.memory_requirements,
                        dedicated.requires_dedicated_allocation == vk::TRUE,
                    ));
                }
            }
            Ok::<(), VulkanError>(())
        })();
        if let Err(error) = create_result {
            unsafe {
                for (device, buffer, _, _) in raw_buffers {
                    device.device.destroy_buffer(buffer, None);
                }
            }
            return Err(error);
        }

        let dedicated = raw_buffers.iter().any(|(_, _, _, dedicated)| *dedicated);
        let allocation_size = if dedicated {
            let required_size = raw_buffers[0].2.size;
            if raw_buffers
                .iter()
                .any(|(_, _, requirements, _)| requirements.size != required_size)
            {
                unsafe {
                    for (device, buffer, _, _) in raw_buffers {
                        device.device.destroy_buffer(buffer, None);
                    }
                }
                return Err(VulkanError(
                    "cross-device dedicated buffer requirements disagree on allocation size"
                        .to_string(),
                ));
            }
            required_size
        } else {
            raw_buffers
                .iter()
                .map(|(_, _, requirements, _)| requirements.size)
                .max()
                .expect("at least one external requirement buffer exists")
        };
        unsafe {
            for (device, buffer, _, _) in raw_buffers {
                device.device.destroy_buffer(buffer, None);
            }
        }
        usize::try_from(allocation_size).map_err(|_| {
            VulkanError(
                "shared device-local allocation requirement exceeds usize".to_string(),
            )
        })
    }
}
