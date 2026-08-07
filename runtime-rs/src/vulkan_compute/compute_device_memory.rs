#[derive(Clone, Copy)]
struct VulkanResidentBufferCopyVisibility {
    producer_stages: vk::PipelineStageFlags,
    producer_access: vk::AccessFlags,
    copy_read_stage: vk::PipelineStageFlags,
    copy_read_access: vk::AccessFlags,
    copy_write_stage: vk::PipelineStageFlags,
    copy_write_access: vk::AccessFlags,
    consumer_stages: vk::PipelineStageFlags,
    consumer_access: vk::AccessFlags,
}

fn resident_buffer_copy_visibility() -> VulkanResidentBufferCopyVisibility {
    VulkanResidentBufferCopyVisibility {
        producer_stages: vk::PipelineStageFlags::HOST
            | vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::TRANSFER,
        producer_access: vk::AccessFlags::HOST_WRITE
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::TRANSFER_WRITE,
        copy_read_stage: vk::PipelineStageFlags::TRANSFER,
        copy_read_access: vk::AccessFlags::TRANSFER_READ,
        copy_write_stage: vk::PipelineStageFlags::TRANSFER,
        copy_write_access: vk::AccessFlags::TRANSFER_WRITE,
        consumer_stages: vk::PipelineStageFlags::HOST
            | vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::TRANSFER
            | vk::PipelineStageFlags::DRAW_INDIRECT
            | vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT,
        consumer_access: vk::AccessFlags::HOST_READ
            | vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE
            | vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT,
    }
}

impl VulkanComputeDevice {
    pub fn create_shared_host_allocation(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<Arc<VulkanSharedHostAllocation>, VulkanError> {
        self.create_shared_host_allocation_with_usage(
            peer_devices,
            byte_capacity,
            resident_buffer_usage(),
        )
    }

    fn create_shared_host_allocation_with_usage(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Arc<VulkanSharedHostAllocation>, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "shared host allocation capacity must not be zero".to_string(),
            ));
        }
        let mut alignment = 1usize;
        let mut required_size = byte_capacity;
        for device in std::iter::once(self).chain(peer_devices.iter().copied()) {
            alignment = alignment.max(device.shared_host_memory_alignment.ok_or_else(|| {
                VulkanError(format!(
                    "Vulkan device {:?} cannot import shared host memory",
                    device.device_name
                ))
            })?);
            let requirements =
                device.shared_host_buffer_memory_requirements(byte_capacity, usage)?;
            alignment =
                alignment.max(usize::try_from(requirements.alignment).map_err(|_| {
                    VulkanError("shared buffer alignment exceeds usize".to_string())
                })?);
            required_size =
                required_size.max(usize::try_from(requirements.size).map_err(|_| {
                    VulkanError("shared buffer allocation size exceeds usize".to_string())
                })?);
        }
        if !alignment.is_power_of_two() {
            return Err(VulkanError(format!(
                "shared buffer alignment {alignment} is not a power of two"
            )));
        }
        let allocation_size = required_size
            .checked_add(alignment - 1)
            .map(|size| size & !(alignment - 1))
            .ok_or_else(|| VulkanError("shared host allocation size overflowed".to_string()))?;
        let layout = Layout::from_size_align(allocation_size, alignment).map_err(|error| {
            VulkanError(format!("invalid shared host allocation layout: {error}"))
        })?;
        let pointer = unsafe { alloc_zeroed(layout) };
        if pointer.is_null() {
            return Err(VulkanError(format!(
                "failed to allocate {allocation_size} bytes of aligned shared host memory"
            )));
        }
        Ok(Arc::new(VulkanSharedHostAllocation {
            address: pointer as usize,
            layout,
            byte_capacity,
        }))
    }

    fn shared_host_buffer_memory_requirements(
        &self,
        byte_capacity: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<vk::MemoryRequirements, VulkanError> {
        unsafe {
            let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity as vk::DeviceSize)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buffer_info);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to query shared host-backed buffer requirements: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            self.device.destroy_buffer(buffer, None);
            Ok(requirements)
        }
    }

    pub fn import_shared_host_buffer(
        &self,
        allocation: Arc<VulkanSharedHostAllocation>,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        self.import_shared_host_buffer_with_usage(allocation, resident_buffer_usage())
    }

    fn import_shared_host_buffer_with_usage(
        &self,
        allocation: Arc<VulkanSharedHostAllocation>,
        usage: vk::BufferUsageFlags,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if self.shared_host_memory_alignment.is_none() {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot import shared host memory",
                self.device_name
            )));
        }
        let loader =
            ash::ext::external_memory_host::Device::new(&self.context.instance, &self.device);
        let mut host_properties = vk::MemoryHostPointerPropertiesEXT::default();
        let result = unsafe {
            (loader.fp().get_memory_host_pointer_properties_ext)(
                loader.device(),
                VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE,
                allocation.address as *const c_void,
                &mut host_properties,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(VulkanError(format!(
                "failed to query shared host-pointer memory types: {result:?}"
            )));
        }

        unsafe {
            let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE);
            let buffer_info = vk::BufferCreateInfo::default()
                .size(allocation.byte_capacity as vk::DeviceSize)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buffer_info);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create shared host-backed storage buffer: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            if requirements.size > allocation.layout.size() as vk::DeviceSize {
                self.device.destroy_buffer(buffer, None);
                return Err(VulkanError(format!(
                    "shared host allocation has {} bytes but Vulkan requires {}",
                    allocation.layout.size(),
                    requirements.size
                )));
            }
            let compatible_memory_types =
                requirements.memory_type_bits & host_properties.memory_type_bits;
            let memory_type_index = match find_memory_type(
                &self.context.instance,
                self.physical_device,
                compatible_memory_types,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
                vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
            ) {
                Some(index) => index,
                None => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(format!(
                        "no host-visible memory type can import the shared allocation (buffer types {:#010x}, host types {:#010x})",
                        requirements.memory_type_bits, host_properties.memory_type_bits
                    )));
                }
            };
            let memory_access = match self.resident_memory_access(memory_type_index) {
                Ok(access) => access,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(error);
                }
            };
            let mut import_info = vk::ImportMemoryHostPointerInfoEXT::default()
                .handle_type(VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE)
                .host_pointer(allocation.address as *mut c_void);
            let memory_info = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation.layout.size() as vk::DeviceSize)
                .memory_type_index(memory_type_index)
                .push_next(&mut import_info);
            let memory = match self.device.allocate_memory(&memory_info, None) {
                Ok(memory) => memory,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(format!(
                        "failed to import shared host allocation: {error:?}"
                    )));
                }
            };
            if let Err(error) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(VulkanError(format!(
                    "failed to bind shared host allocation: {error:?}"
                )));
            }
            Ok(VulkanResidentBuffer {
                device: self.device.clone(),
                buffer,
                memory: Some(memory),
                memory_access,
                byte_capacity: allocation.byte_capacity as vk::DeviceSize,
                device_address: None,
                device_address_registry: None,
                persistent_mapping: Some(allocation.address),
                persistent_mapping_requires_unmap: false,
                _shared_host_allocation: Some(allocation),
                _shared_device_memory_identity: None,
                _device_local_memory_reservation: None,
            })
        }
    }

    /// Creates one byte-identical resident buffer view on every participating
    /// logical device. Device-local DMA-BUF storage is preferred. Shared host
    /// memory is an explicit capability fallback, not the normal route.
    pub fn create_shared_resident_buffers(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<VulkanSharedResidentBufferSet, VulkanError> {
        self.create_shared_resident_buffers_with_usage(
            peer_devices,
            byte_capacity,
            resident_buffer_usage(),
        )
    }

    pub fn create_shared_conditional_resident_buffers(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
    ) -> Result<VulkanSharedResidentBufferSet, VulkanError> {
        let devices = std::iter::once(self)
            .chain(peer_devices.iter().copied())
            .collect::<Vec<_>>();
        if let Some(device) = devices
            .iter()
            .find(|device| !device.supports_conditional_compute_dispatch())
        {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot share a conditional compute predicate",
                device.device_name()
            )));
        }
        self.create_shared_resident_buffers_with_usage(
            peer_devices,
            byte_capacity,
            resident_buffer_usage() | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT,
        )
    }

    fn create_shared_resident_buffers_with_usage(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<VulkanSharedResidentBufferSet, VulkanError> {
        let external_device_local_error =
            match self.create_shared_device_resident_buffers(peer_devices, byte_capacity, usage) {
                Ok(buffers) => {
                    return Ok(VulkanSharedResidentBufferSet {
                        route: VulkanSharedResidentBufferRoute::ExternalDeviceLocal,
                        buffers,
                        external_device_local_error: None,
                    });
                }
                Err(error) => error.to_string(),
            };
        let allocation = self
            .create_shared_host_allocation_with_usage(
                peer_devices,
                byte_capacity,
                usage,
            )
            .map_err(|host_error| {
                VulkanError(format!(
                    "device-local external memory is unavailable ({external_device_local_error}); shared-host fallback is unavailable ({host_error})"
                ))
            })?;
        let buffers = std::iter::once(self)
            .chain(peer_devices.iter().copied())
            .map(|device| {
                device
                    .import_shared_host_buffer_with_usage(Arc::clone(&allocation), usage)
                    .map(Arc::new)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VulkanSharedResidentBufferSet {
            route: VulkanSharedResidentBufferRoute::SharedHost,
            buffers,
            external_device_local_error: Some(external_device_local_error),
        })
    }

    fn create_shared_device_resident_buffers(
        &self,
        peer_devices: &[&VulkanComputeDevice],
        byte_capacity: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Vec<Arc<VulkanResidentBuffer>>, VulkanError> {
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

        struct RawExternalBuffer<'a> {
            device: &'a VulkanComputeDevice,
            buffer: vk::Buffer,
            requirements: vk::MemoryRequirements,
            requires_dedicated: bool,
        }

        let mut raw_buffers = Vec::<RawExternalBuffer<'_>>::with_capacity(devices.len());
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
                    raw_buffers.push(RawExternalBuffer {
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

        let dedicated = raw_buffers.iter().any(|raw| raw.requires_dedicated);
        let allocation_size = if dedicated {
            let required_size = raw_buffers[0].requirements.size;
            if raw_buffers
                .iter()
                .any(|raw| raw.requirements.size != required_size)
            {
                unsafe {
                    for raw in raw_buffers {
                        raw.device.device.destroy_buffer(raw.buffer, None);
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
                .map(|raw| raw.requirements.size)
                .max()
                .expect("at least the owner external buffer exists")
        };
        let device_local_memory_reservation =
            match self.reserve_device_local_memory(allocation_size) {
                Ok(reservation) => reservation,
                Err(error) => {
                    unsafe {
                        for raw in raw_buffers {
                            raw.device.device.destroy_buffer(raw.buffer, None);
                        }
                    }
                    return Err(error);
                }
            };
        let mut allocated_memories = vec![None; raw_buffers.len()];
        let mut allocated_memory_type_indices = vec![None; raw_buffers.len()];
        let allocation_result = (|| {
            let owner = &raw_buffers[0];
            let owner_memory_type = unsafe {
                find_memory_type(
                    &owner.device.context.instance,
                    owner.device.physical_device,
                    owner.requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::empty(),
                )
            }
            .ok_or_else(|| {
                VulkanError(format!(
                    "no device-local exportable memory type on {:?}",
                    owner.device.device_name
                ))
            })?;
            allocated_memory_type_indices[0] = Some(owner_memory_type);
            let mut export = vk::ExportMemoryAllocateInfo::default()
                .handle_types(VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE);
            let mut owner_dedicated =
                vk::MemoryDedicatedAllocateInfo::default().buffer(owner.buffer);
            let mut owner_allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(owner_memory_type)
                .push_next(&mut export);
            if dedicated {
                owner_allocate = owner_allocate.push_next(&mut owner_dedicated);
            }
            let owner_memory = unsafe {
                owner
                    .device
                    .device
                    .allocate_memory(&owner_allocate, None)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to allocate exportable device-local memory on {:?}: {error:?}",
                            owner.device.device_name
                        ))
                    })?
            };
            allocated_memories[0] = Some(owner_memory);
            unsafe {
                owner
                    .device
                    .device
                    .bind_buffer_memory(owner.buffer, owner_memory, 0)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to bind exportable device-local memory on {:?}: {error:?}",
                            owner.device.device_name
                        ))
                    })?;
            }

            let owner_fd_loader = ash::khr::external_memory_fd::Device::new(
                &owner.device.context.instance,
                &owner.device.device,
            );
            for (index, raw) in raw_buffers.iter().enumerate().skip(1) {
                let fd = unsafe {
                    owner_fd_loader
                        .get_memory_fd(
                            &vk::MemoryGetFdInfoKHR::default()
                                .memory(owner_memory)
                                .handle_type(VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE),
                        )
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to export device-local DMA-BUF from {:?}: {error:?}",
                                owner.device.device_name
                            ))
                        })?
                };
                let fd = unsafe { OwnedFd::from_raw_fd(fd) };
                let peer_fd_loader = ash::khr::external_memory_fd::Device::new(
                    &raw.device.context.instance,
                    &raw.device.device,
                );
                let mut fd_properties = vk::MemoryFdPropertiesKHR::default();
                unsafe {
                    peer_fd_loader
                        .get_memory_fd_properties(
                            VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE,
                            fd.as_raw_fd(),
                            &mut fd_properties,
                        )
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to inspect imported DMA-BUF on {:?}: {error:?}",
                                raw.device.device_name
                            ))
                        })?;
                }
                let compatible_memory_types =
                    raw.requirements.memory_type_bits & fd_properties.memory_type_bits;
                let memory_type = unsafe {
                    find_memory_type(
                        &raw.device.context.instance,
                        raw.device.physical_device,
                        compatible_memory_types,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        vk::MemoryPropertyFlags::empty(),
                    )
                }
                .ok_or_else(|| {
                    VulkanError(format!(
                        "DMA-BUF from {:?} has no device-local import type on {:?}",
                        owner.device.device_name, raw.device.device_name
                    ))
                })?;
                allocated_memory_type_indices[index] = Some(memory_type);
                let mut import = vk::ImportMemoryFdInfoKHR::default()
                    .handle_type(VULKAN_SHARED_DEVICE_MEMORY_HANDLE_TYPE)
                    .fd(fd.as_raw_fd());
                let mut peer_dedicated =
                    vk::MemoryDedicatedAllocateInfo::default().buffer(raw.buffer);
                let mut peer_allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(allocation_size)
                    .memory_type_index(memory_type)
                    .push_next(&mut import);
                if dedicated {
                    peer_allocate = peer_allocate.push_next(&mut peer_dedicated);
                }
                let memory = unsafe {
                    raw.device
                        .device
                        .allocate_memory(&peer_allocate, None)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to import device-local DMA-BUF on {:?}: {error:?}",
                                raw.device.device_name
                            ))
                        })?
                };
                let _ = fd.into_raw_fd();
                allocated_memories[index] = Some(memory);
                unsafe {
                    raw.device
                        .device
                        .bind_buffer_memory(raw.buffer, memory, 0)
                        .map_err(|error| {
                            VulkanError(format!(
                                "failed to bind imported device-local DMA-BUF on {:?}: {error:?}",
                                raw.device.device_name
                            ))
                        })?;
                }
            }
            Ok::<(), VulkanError>(())
        })();
        if let Err(error) = allocation_result {
            unsafe {
                for (raw, memory) in raw_buffers.iter().zip(&allocated_memories) {
                    raw.device.device.destroy_buffer(raw.buffer, None);
                    if let Some(memory) = memory {
                        raw.device.device.free_memory(*memory, None);
                    }
                }
            }
            return Err(error);
        }

        let identity = Arc::new(VulkanSharedDeviceMemoryIdentity);
        Ok(raw_buffers
            .into_iter()
            .zip(
                allocated_memories
                    .into_iter()
                    .zip(allocated_memory_type_indices),
            )
            .map(|(raw, (memory, memory_type_index))| {
                let memory = memory.expect("every external buffer was allocated");
                let memory_type_index =
                    memory_type_index.expect("every external memory type was selected");
                let memory_access = raw
                    .device
                    .resident_memory_access(memory_type_index)
                    .expect("the selected external memory type has resident access");
                Arc::new(VulkanResidentBuffer {
                    device: raw.device.device.clone(),
                    buffer: raw.buffer,
                    memory: Some(memory),
                    memory_access,
                    byte_capacity: byte_capacity as vk::DeviceSize,
                    device_address: None,
                    device_address_registry: None,
                    persistent_mapping: None,
                    persistent_mapping_requires_unmap: false,
                    _shared_host_allocation: None,
                    _shared_device_memory_identity: Some(Arc::clone(&identity)),
                    _device_local_memory_reservation: Some(Arc::clone(
                        &device_local_memory_reservation,
                    )),
                })
            })
            .collect())
    }

    pub fn create_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        self.create_timeline_semaphore_with_opaque_fd_export(initial_value, false)
    }

    pub fn create_opaque_fd_exportable_timeline_semaphore(
        &self,
        initial_value: u64,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        if !self.opaque_fd_timeline_semaphore_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot export persistent opaque-file timeline semaphores",
                self.device_name
            )));
        }
        self.create_timeline_semaphore_with_opaque_fd_export(initial_value, true)
    }

    pub fn wait_timeline_semaphore_value(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        value: u64,
    ) -> Result<(), VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        wait_for_vulkan_timeline_points_with_progress_watchdog(
            &self.device,
            &[semaphore.semaphore],
            &[value],
            false,
            &self.device_health,
            "timeline semaphore wait",
            |error| self.vulkan_operation_error("failed to wait for timeline semaphore", error),
        )
    }

    pub fn wait_timeline_semaphore_value_for(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        value: u64,
        timeout_ns: u64,
    ) -> Result<bool, VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        if timeout_ns == u64::MAX {
            self.wait_timeline_semaphore_value(semaphore, value)?;
            return Ok(true);
        }
        let semaphores = [semaphore.semaphore];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        match unsafe { self.device.wait_semaphores(&wait_info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(VulkanError(format!(
                "failed to wait for timeline semaphore value {value}: {error:?}"
            ))),
        }
    }

    pub fn wait_any_timeline_semaphore_points_for(
        &self,
        points: &[VulkanTimelineSemaphorePoint<'_>],
        timeout_ns: u64,
    ) -> Result<bool, VulkanError> {
        if points.is_empty() {
            return Err(VulkanError(
                "cannot wait on an empty timeline semaphore set".to_string(),
            ));
        }
        for point in points {
            self.validate_local_timeline_semaphore(point.semaphore)?;
        }
        let semaphores = points
            .iter()
            .map(|point| point.semaphore.semaphore)
            .collect::<Vec<_>>();
        let values = points.iter().map(|point| point.value).collect::<Vec<_>>();
        if timeout_ns == u64::MAX {
            wait_for_vulkan_timeline_points_with_progress_watchdog(
                &self.device,
                &semaphores,
                &values,
                true,
                &self.device_health,
                "timeline semaphore set wait",
                |error| {
                    self.vulkan_operation_error(
                        "failed to wait for timeline semaphore set",
                        error,
                    )
                },
            )?;
            return Ok(true);
        }
        let wait_info = vk::SemaphoreWaitInfo::default()
            .flags(vk::SemaphoreWaitFlags::ANY)
            .semaphores(&semaphores)
            .values(&values);
        match unsafe { self.device.wait_semaphores(&wait_info, timeout_ns) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(error) => Err(VulkanError(format!(
                "failed to wait for any of {} timeline semaphore points: {error:?}",
                points.len()
            ))),
        }
    }

    pub fn timeline_semaphore_point_is_complete(
        &self,
        point: VulkanTimelineSemaphorePoint<'_>,
    ) -> Result<bool, VulkanError> {
        Ok(self.timeline_semaphore_value(point.semaphore)? >= point.value)
    }

    pub fn timeline_semaphore_value(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<u64, VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        unsafe { self.device.get_semaphore_counter_value(semaphore.semaphore) }.map_err(|error| {
            VulkanError(format!(
                "failed to read timeline semaphore counter: {error:?}"
            ))
        })
    }

    fn create_timeline_semaphore_with_opaque_fd_export(
        &self,
        initial_value: u64,
        opaque_fd_exportable: bool,
    ) -> Result<VulkanTimelineSemaphore, VulkanError> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore = if opaque_fd_exportable {
            let mut export_info = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .push_next(&mut timeline_info)
                .push_next(&mut export_info);
            unsafe { self.device.create_semaphore(&create_info, None) }
        } else {
            let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
            unsafe { self.device.create_semaphore(&create_info, None) }
        }
        .map_err(|error| VulkanError(format!("failed to create timeline semaphore: {error:?}")))?;
        Ok(VulkanTimelineSemaphore {
            device: self.device.clone(),
            device_handle: self.device.handle(),
            semaphore,
            opaque_fd_exportable,
            permanent_opaque_fd_imported: Cell::new(false),
        })
    }

    pub fn export_timeline_semaphore_opaque_fd(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<OwnedFd, VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        if !semaphore.opaque_fd_exportable {
            return Err(VulkanError(
                "timeline semaphore was not created for persistent opaque-file export".to_string(),
            ));
        }
        let loader =
            ash::khr::external_semaphore_fd::Device::new(&self.context.instance, &self.device);
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore.semaphore)
            .handle_type(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE);
        let fd = unsafe { loader.get_semaphore_fd(&get_info) }.map_err(|error| {
            VulkanError(format!(
                "failed to export timeline semaphore as persistent opaque file: {error:?}"
            ))
        })?;
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    pub fn import_timeline_semaphore_opaque_fd(
        &self,
        semaphore: &VulkanTimelineSemaphore,
        fd: OwnedFd,
    ) -> Result<(), VulkanError> {
        self.validate_local_timeline_semaphore(semaphore)?;
        if !self.opaque_fd_timeline_semaphore_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} cannot import persistent opaque-file timeline semaphores",
                self.device_name
            )));
        }
        if semaphore.permanent_opaque_fd_imported.get() {
            return Err(VulkanError(
                "timeline semaphore already has a permanently imported opaque-file payload"
                    .to_string(),
            ));
        }
        let import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore.semaphore)
            .flags(vk::SemaphoreImportFlags::empty())
            .handle_type(VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE)
            .fd(fd.as_raw_fd());
        let loader =
            ash::khr::external_semaphore_fd::Device::new(&self.context.instance, &self.device);
        unsafe { loader.import_semaphore_fd(&import_info) }.map_err(|error| {
            VulkanError(format!(
                "failed to import timeline semaphore persistent opaque file: {error:?}"
            ))
        })?;
        let _fd_owned_by_vulkan = fd.into_raw_fd();
        semaphore.permanent_opaque_fd_imported.set(true);
        Ok(())
    }

    fn validate_local_timeline_semaphore(
        &self,
        semaphore: &VulkanTimelineSemaphore,
    ) -> Result<(), VulkanError> {
        if semaphore.device_handle != self.device.handle() {
            return Err(VulkanError(
                "timeline semaphore belongs to a different Vulkan logical device".to_string(),
            ));
        }
        Ok(())
    }

    fn resident_memory_access(
        &self,
        memory_type_index: u32,
    ) -> Result<VulkanResidentMemoryAccess, VulkanError> {
        let memory_properties = unsafe {
            self.context
                .instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let property_flags =
            memory_properties.memory_types[memory_type_index as usize].property_flags;
        let directly_mappable = property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let staging_memory_type_index = if directly_mappable {
            None
        } else {
            Some(
                unsafe {
                    find_memory_type(
                        &self.context.instance,
                        self.physical_device,
                        u32::MAX,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                        vk::MemoryPropertyFlags::empty(),
                    )
                }
                .ok_or_else(|| {
                    VulkanError(
                        "no host-visible coherent memory type for resident staging transfers"
                            .to_string(),
                    )
                })?,
            )
        };
        Ok(VulkanResidentMemoryAccess {
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            device_health: self.device_health.clone(),
            property_flags,
            staging_memory_type_index,
        })
    }

    pub fn create_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "resident byte buffer capacity must not be zero".to_string(),
            ));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) =
            self.create_resident_storage_buffer(
                byte_capacity,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: None,
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        })
    }

    pub fn create_conditional_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "conditional resident buffer capacity must not be zero".to_string(),
            ));
        }
        if self.conditional_rendering.is_none() {
            return Err(VulkanError(format!(
                "Vulkan device {:?} does not support conditional compute dispatch",
                self.device_name
            )));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) = self
            .create_resident_storage_buffer_with_usage(
                byte_capacity,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                resident_buffer_usage() | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT,
                false,
                None,
            )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: None,
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        })
    }

    pub fn create_addressable_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        self.create_addressable_resident_buffer_internal(byte_capacity, None)
    }

    pub fn create_addressable_resident_buffer_with_capacity_permit(
        &self,
        byte_capacity: usize,
        capacity_permit: VulkanDeviceLocalMemoryPermit,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        self.create_addressable_resident_buffer_internal(
            byte_capacity,
            Some(capacity_permit),
        )
    }

    pub fn addressable_resident_buffer_memory_requirement_bytes(
        &self,
        byte_capacity: usize,
    ) -> Result<usize, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "addressable resident buffer requirement capacity must not be zero"
                    .to_string(),
            ));
        }
        if !self.buffer_device_address_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} does not support buffer device addresses",
                self.device_name
            )));
        }
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity as vk::DeviceSize)
                .usage(resident_buffer_usage() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to query addressable resident buffer requirements: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            self.device.destroy_buffer(buffer, None);
            usize::try_from(requirements.size).map_err(|_| {
                VulkanError(
                    "addressable resident buffer memory requirement exceeds usize"
                        .to_string(),
                )
            })
        }
    }

    fn create_addressable_resident_buffer_internal(
        &self,
        byte_capacity: usize,
        capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "addressable resident buffer capacity must not be zero".to_string(),
            ));
        }
        if !self.buffer_device_address_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} does not support buffer device addresses",
                self.device_name
            )));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) = self
            .create_resident_storage_buffer_with_addressability(
                byte_capacity,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                true,
                capacity_permit,
            )?;
        let device_address = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        if device_address == 0 {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            return Err(VulkanError(
                "Vulkan returned a null device address for an addressable buffer".to_string(),
            ));
        }
        self.track_addressable_buffer(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: Some(device_address),
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        }, "device-local addressable buffer")
    }

    pub fn create_host_visible_addressable_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "host-visible addressable buffer capacity must not be zero".to_string(),
            ));
        }
        if !self.buffer_device_address_supported {
            return Err(VulkanError(format!(
                "Vulkan device {:?} does not support buffer device addresses",
                self.device_name
            )));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) = self
            .create_resident_storage_buffer_with_addressability(
                byte_capacity,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_CACHED,
                true,
                None,
            )?;
        let device_address = unsafe {
            self.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        if device_address == 0 {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            return Err(VulkanError(
                "Vulkan returned a null device address for a host-visible addressable buffer"
                    .to_string(),
            ));
        }
        self.track_addressable_buffer(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: Some(device_address),
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        }, "host-visible addressable buffer")
    }

    pub fn create_host_visible_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "resident byte buffer capacity must not be zero".to_string(),
            ));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) =
            self.create_resident_storage_buffer(
                byte_capacity,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: None,
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        })
    }

    fn create_host_readback_resident_buffer(
        &self,
        byte_capacity: usize,
    ) -> Result<VulkanResidentBuffer, VulkanError> {
        if byte_capacity == 0 {
            return Err(VulkanError(
                "resident readback buffer capacity must not be zero".to_string(),
            ));
        }
        let (buffer, memory, byte_capacity, memory_access, device_local_memory_reservation) =
            self.create_resident_storage_buffer(
                byte_capacity,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::HOST_CACHED,
            )?;
        Ok(VulkanResidentBuffer {
            device: self.device.clone(),
            buffer,
            memory: Some(memory),
            memory_access,
            byte_capacity,
            device_address: None,
            device_address_registry: None,
            persistent_mapping: None,
            persistent_mapping_requires_unmap: false,
            _shared_host_allocation: None,
            _shared_device_memory_identity: None,
            _device_local_memory_reservation: device_local_memory_reservation,
        })
    }

    fn create_resident_storage_buffer(
        &self,
        byte_capacity: usize,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
    ) -> Result<
        (
            vk::Buffer,
            vk::DeviceMemory,
            vk::DeviceSize,
            VulkanResidentMemoryAccess,
            Option<Arc<VulkanDeviceLocalMemoryReservation>>,
        ),
        VulkanError,
    > {
        self.create_resident_storage_buffer_with_addressability(
            byte_capacity,
            required_memory_flags,
            preferred_memory_flags,
            false,
            None,
        )
    }

    fn create_resident_storage_buffer_with_addressability(
        &self,
        byte_capacity: usize,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
        addressable: bool,
        capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
    ) -> Result<
        (
            vk::Buffer,
            vk::DeviceMemory,
            vk::DeviceSize,
            VulkanResidentMemoryAccess,
            Option<Arc<VulkanDeviceLocalMemoryReservation>>,
        ),
        VulkanError,
    > {
        let usage = if addressable {
            resident_buffer_usage() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        } else {
            resident_buffer_usage()
        };
        self.create_resident_storage_buffer_with_usage(
            byte_capacity,
            required_memory_flags,
            preferred_memory_flags,
            usage,
            addressable,
            capacity_permit,
        )
    }

    fn create_resident_storage_buffer_with_usage(
        &self,
        byte_capacity: usize,
        required_memory_flags: vk::MemoryPropertyFlags,
        preferred_memory_flags: vk::MemoryPropertyFlags,
        usage: vk::BufferUsageFlags,
        addressable: bool,
        capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
    ) -> Result<
        (
            vk::Buffer,
            vk::DeviceMemory,
            vk::DeviceSize,
            VulkanResidentMemoryAccess,
            Option<Arc<VulkanDeviceLocalMemoryReservation>>,
        ),
        VulkanError,
    > {
        let byte_capacity = byte_capacity as vk::DeviceSize;
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(byte_capacity)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = self
                .device
                .create_buffer(&buffer_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident storage buffer: {error:?}"
                    ))
                })?;
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            let memory_type_index = find_memory_type(
                &self.context.instance,
                self.physical_device,
                requirements.memory_type_bits,
                required_memory_flags,
                preferred_memory_flags,
            )
            .ok_or_else(|| {
                self.device.destroy_buffer(buffer, None);
                VulkanError(format!(
                    "no memory type with required flags {required_memory_flags:?} for resident storage buffer"
                ))
            })?;
            let memory_properties = self
                .context
                .instance
                .get_physical_device_memory_properties(self.physical_device);
            let property_flags =
                memory_properties.memory_types[memory_type_index as usize].property_flags;
            let device_local_memory_reservation =
                if property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
                    let reservation = match capacity_permit {
                        Some(permit) => {
                            self.commit_device_local_memory_capacity(permit, requirements.size)
                        }
                        None => self.reserve_device_local_memory(requirements.size),
                    };
                    match reservation {
                    Ok(reservation) => Some(reservation),
                    Err(error) => {
                        self.device.destroy_buffer(buffer, None);
                        return Err(error);
                    }
                    }
                } else if capacity_permit.is_some() {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(
                        "device-local capacity permit was supplied for host memory".to_string(),
                    ));
                } else {
                    None
                };
            let directly_mappable = property_flags.contains(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let staging_memory_type_index = if directly_mappable {
                None
            } else {
                match find_memory_type(
                    &self.context.instance,
                    self.physical_device,
                    u32::MAX,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    vk::MemoryPropertyFlags::empty(),
                ) {
                    Some(memory_type_index) => Some(memory_type_index),
                    None => {
                        self.device.destroy_buffer(buffer, None);
                        return Err(VulkanError(
                            "no host-visible coherent memory type for resident staging transfers"
                                .to_string(),
                        ));
                    }
                }
            };
            let mut address_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut memory_info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index);
            if addressable {
                memory_info = memory_info.push_next(&mut address_flags);
            }
            let memory = match self.device.allocate_memory(&memory_info, None) {
                Ok(memory) => memory,
                Err(error) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(VulkanError(format!(
                        "failed to allocate resident storage buffer memory: {error:?}"
                    )));
                }
            };
            if let Err(error) = self.device.bind_buffer_memory(buffer, memory, 0) {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
                return Err(VulkanError(format!(
                    "failed to bind resident storage buffer memory: {error:?}"
                )));
            }
            Ok((
                buffer,
                memory,
                byte_capacity,
                VulkanResidentMemoryAccess {
                    queue: self.queue,
                    queue_family_index: self.queue_family_index,
                    device_health: self.device_health.clone(),
                    property_flags,
                    staging_memory_type_index,
                },
                device_local_memory_reservation,
            ))
        }
    }

    pub fn copy_resident_buffer_bytes(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<(), VulkanError> {
        let binding = self.create_resident_buffer_copy(source, destination, len)?;
        self.run_resident_buffer_copy(&binding, len)
    }

    pub fn create_resident_buffer_copy(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<VulkanResidentBufferCopy, VulkanError> {
        self.create_resident_buffer_copy_internal(source, destination, len, false)
    }

    pub fn create_timestamped_resident_buffer_copy(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<VulkanResidentBufferCopy, VulkanError> {
        self.create_resident_buffer_copy_internal(source, destination, len, true)
    }

    fn create_resident_buffer_copy_internal(
        &self,
        source: &VulkanResidentBuffer,
        destination: &VulkanResidentBuffer,
        len: usize,
        timestamped: bool,
    ) -> Result<VulkanResidentBufferCopy, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte copy binding length must not be zero".to_string(),
            ));
        }
        let byte_len = source.byte_len(len)?;
        destination.byte_len(len)?;

        unsafe {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident byte copy binding command pool: {error:?}"
                    ))
                })?;
            let command_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = self
                .device
                .allocate_command_buffers(&command_alloc_info)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to allocate resident byte copy binding command buffer: {error:?}"
                    ))
                })?
                .remove(0);
            let timestamp_query_pool = if timestamped {
                match self.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(2),
                    None,
                ) {
                    Ok(query_pool) => Some(query_pool),
                    Err(error) => {
                        self.device.destroy_command_pool(command_pool, None);
                        return Err(VulkanError(format!(
                            "failed to create resident byte copy timestamp pool: {error:?}"
                        )));
                    }
                }
            } else {
                None
            };

            let command_begin = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &command_begin)
                .map_err(|error| {
                    if let Some(query_pool) = timestamp_query_pool {
                        self.device.destroy_query_pool(query_pool, None);
                    }
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to begin resident byte copy binding command buffer: {error:?}"
                    ))
                })?;
            if let Some(query_pool) = timestamp_query_pool {
                self.device
                    .cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
                self.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    query_pool,
                    0,
                );
            }
            // A resident copy is an independently submitted circuit boundary.
            // Its source can be produced by a shader, a transfer, or the host
            // in an earlier submission. Queue submission order alone is not a
            // Vulkan memory dependency, so make those writes visible before
            // the transfer reads them.
            let visibility = resident_buffer_copy_visibility();
            let source_visibility = [vk::BufferMemoryBarrier::default()
                .src_access_mask(visibility.producer_access)
                .dst_access_mask(visibility.copy_read_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(source.buffer)
                .offset(0)
                .size(byte_len)];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                visibility.producer_stages,
                visibility.copy_read_stage,
                vk::DependencyFlags::empty(),
                &[],
                &source_visibility,
                &[],
            );
            let copy_regions = [vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: byte_len,
            }];
            self.device.cmd_copy_buffer(
                command_buffer,
                source.buffer,
                destination.buffer,
                &copy_regions,
            );
            // The destination can feed another transfer, a shader, an
            // indirect/conditional dispatch, or a host read. Publish the copy
            // to every supported consumer before a completion semaphore can
            // make the containing submission externally observable.
            let destination_visibility = [vk::BufferMemoryBarrier::default()
                .src_access_mask(visibility.copy_write_access)
                .dst_access_mask(visibility.consumer_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(destination.buffer)
                .offset(0)
                .size(byte_len)];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                visibility.copy_write_stage,
                visibility.consumer_stages,
                vk::DependencyFlags::empty(),
                &[],
                &destination_visibility,
                &[],
            );
            if let Some(query_pool) = timestamp_query_pool {
                self.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    1,
                );
            }
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| {
                    if let Some(query_pool) = timestamp_query_pool {
                        self.device.destroy_query_pool(query_pool, None);
                    }
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to end resident byte copy binding command buffer: {error:?}"
                    ))
                })?;
            let completion_fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| {
                    if let Some(query_pool) = timestamp_query_pool {
                        self.device.destroy_query_pool(query_pool, None);
                    }
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to create resident byte copy completion fence: {error:?}"
                    ))
                })?;

            Ok(VulkanResidentBufferCopy {
                device: self.device.clone(),
                device_health: self.device_health.clone(),
                device_fault: self.device_fault.clone(),
                device_address_registry: Arc::clone(&self.device_address_registry),
                queue: self.queue,
                command_pool,
                command_buffer,
                source: source.buffer,
                destination: destination.buffer,
                byte_len,
                completion_fence,
                timestamp_query_pool,
                timestamp_period_ns: self.timestamp_period_ns,
            })
        }
    }

    pub fn submit_resident_buffer_copy_with_timeline_semaphores(
        &self,
        binding: &VulkanResidentBufferCopy,
        wait_points: &[VulkanTimelineSemaphorePoint<'_>],
        signal_points: &[VulkanTimelineSemaphorePoint<'_>],
    ) -> Result<(), VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        if binding.device.handle() != self.device.handle() {
            return Err(VulkanError(
                "resident buffer copy belongs to another logical device".to_string(),
            ));
        }
        self.submit_command_buffer_with_timeline_semaphores(
            binding.command_buffer,
            Some(binding.completion_fence),
            wait_points,
            signal_points,
            "resident buffer copy",
            false,
        )?;
        Ok(())
    }

    pub fn wait_resident_buffer_copy(
        &self,
        binding: &VulkanResidentBufferCopy,
    ) -> Result<(), VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        if binding.device.handle() != self.device.handle() {
            return Err(VulkanError(
                "resident buffer copy belongs to another logical device".to_string(),
            ));
        }
        wait_for_vulkan_fences_with_progress_watchdog(
            &self.device,
            &[binding.completion_fence],
            true,
            &self.device_health,
            "resident buffer copy",
            |error| {
                self.vulkan_operation_error(
                    "failed waiting for resident buffer copy completion",
                    error,
                )
            },
        )?;
        RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn wait_resident_buffer_copy_batch(
        &self,
        binding: &VulkanResidentBufferCopyBatch,
    ) -> Result<(), VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        if binding.device.handle() != self.device.handle() {
            return Err(VulkanError(
                "resident buffer copy batch belongs to another logical device".to_string(),
            ));
        }
        wait_for_vulkan_fences_with_progress_watchdog(
            &self.device,
            &[binding.completion_fence],
            true,
            &self.device_health,
            "resident buffer copy batch",
            |error| {
                self.vulkan_operation_error(
                    "failed waiting for resident buffer copy batch completion",
                    error,
                )
            },
        )?;
        RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn submit_resident_buffer_copy_batch(
        &self,
        binding: &VulkanResidentBufferCopyBatch,
    ) -> Result<(), VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        self.require_device_healthy()?;
        if binding.device.handle() != self.device.handle() {
            return Err(VulkanError(
                "resident buffer copy batch belongs to another logical device".to_string(),
            ));
        }
        unsafe {
            self.device
                .reset_fences(&[binding.completion_fence])
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident buffer copy batch fence: {error:?}"
                    ))
                })?;
            let command_buffers = [binding.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, binding.completion_fence)
                .map_err(|error| {
                    vulkan_error_with_device_quarantine(
                        &self.device_health,
                        error,
                        self.vulkan_operation_error(
                            "failed to submit resident buffer copy batch",
                            error,
                        ),
                    )
                })?;
        }
        RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub fn create_resident_buffer_copy_batch(
        &self,
        copies: &[VulkanResidentBufferRangeCopy<'_>],
    ) -> Result<VulkanResidentBufferCopyBatch, VulkanError> {
        if copies.is_empty() {
            return Err(VulkanError(
                "resident buffer copy batch must contain at least one copy".to_string(),
            ));
        }
        unsafe {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(self.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let command_pool = self
                .device
                .create_command_pool(&command_pool_info, None)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident buffer copy batch command pool: {error:?}"
                    ))
                })?;
            let command_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = self
                .device
                .allocate_command_buffers(&command_alloc_info)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to allocate resident buffer copy batch command buffer: {error:?}"
                    ))
                })?
                .remove(0);
            self.device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to begin resident buffer copy batch command buffer: {error:?}"
                    ))
                })?;
            for copy in copies {
                let regions = [vk::BufferCopy {
                    src_offset: copy.source_offset,
                    dst_offset: copy.destination_offset,
                    size: copy.byte_len,
                }];
                self.device.cmd_copy_buffer(
                    command_buffer,
                    copy.source.buffer,
                    copy.destination.buffer,
                    &regions,
                );
            }
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to end resident buffer copy batch command buffer: {error:?}"
                    ))
                })?;
            let completion_fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| {
                    self.device.destroy_command_pool(command_pool, None);
                    VulkanError(format!(
                        "failed to create resident buffer copy batch fence: {error:?}"
                    ))
                })?;
            Ok(VulkanResidentBufferCopyBatch {
                device: self.device.clone(),
                device_health: self.device_health.clone(),
                device_fault: self.device_fault.clone(),
                device_address_registry: Arc::clone(&self.device_address_registry),
                queue: self.queue,
                command_pool,
                command_buffer,
                completion_fence,
                copy_count: copies.len(),
            })
        }
    }

    pub fn read_resident_buffer_ranges(
        &self,
        ranges: &[VulkanResidentBufferReadRange<'_>],
    ) -> Result<VulkanResidentBufferReadback, VulkanError> {
        if ranges.is_empty() {
            return Ok(VulkanResidentBufferReadback {
                bytes: Vec::new(),
                ranges: Vec::new(),
            });
        }
        let mut packed_ranges = Vec::with_capacity(ranges.len());
        let total_byte_count = ranges.iter().try_fold(0usize, |offset, range| {
            let end = offset.checked_add(range.byte_len).ok_or_else(|| {
                VulkanError("resident buffer readback capacity overflowed".to_string())
            })?;
            packed_ranges.push(offset..end);
            Ok(end)
        })?;
        let staging = self.create_host_readback_resident_buffer(total_byte_count)?;
        let copies = ranges
            .iter()
            .zip(&packed_ranges)
            .map(|(range, destination)| {
                VulkanResidentBufferRangeCopy::new(
                    range.source,
                    &staging,
                    range.source_offset,
                    destination.start,
                    range.byte_len,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.create_resident_buffer_copy_batch(&copies)?.run()?;
        Ok(VulkanResidentBufferReadback {
            bytes: staging.read_bytes(total_byte_count)?,
            ranges: packed_ranges,
        })
    }

    pub fn write_resident_buffer_ranges(
        &self,
        ranges: &[VulkanResidentBufferWriteRange<'_>],
    ) -> Result<usize, VulkanError> {
        if ranges.is_empty() {
            return Ok(0);
        }
        let mut packed_ranges = Vec::with_capacity(ranges.len());
        let total_byte_count = ranges.iter().try_fold(0usize, |offset, range| {
            let end = offset.checked_add(range.bytes.len()).ok_or_else(|| {
                VulkanError("resident buffer upload capacity overflowed".to_string())
            })?;
            packed_ranges.push(offset..end);
            Ok(end)
        })?;
        let staging = self.create_host_visible_resident_buffer(total_byte_count)?;
        let mut packed_bytes = Vec::with_capacity(total_byte_count);
        for range in ranges {
            packed_bytes.extend_from_slice(range.bytes);
        }
        staging.write_bytes(&packed_bytes)?;
        let copies = ranges
            .iter()
            .zip(&packed_ranges)
            .map(|(range, source)| {
                VulkanResidentBufferRangeCopy::new(
                    &staging,
                    range.destination,
                    source.start,
                    range.destination_offset,
                    range.bytes.len(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.create_resident_buffer_copy_batch(&copies)?.run()?;
        Ok(total_byte_count)
    }

    pub fn run_resident_buffer_copy(
        &self,
        binding: &VulkanResidentBufferCopy,
        len: usize,
    ) -> Result<(), VulkanError> {
        binding.run(len)
    }
}
