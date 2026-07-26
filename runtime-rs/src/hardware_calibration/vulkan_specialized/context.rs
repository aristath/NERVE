use ash::{Entry, vk};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SpecializedVulkanRequirements {
    pub compute: bool,
    pub graphics: bool,
    pub ray_query: bool,
    pub device_generated_commands: bool,
}

pub(crate) struct SpecializedVulkanContext {
    _entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queues: BTreeMap<u32, vk::Queue>,
    graphics_queue_family: Option<u32>,
    compute_queue_family: Option<u32>,
    timestamp_period_ns: f32,
    pci_address: Option<String>,
    acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
}

pub(crate) struct SpecializedBuffer {
    device: ash::Device,
    pub(in crate::hardware_calibration) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(in crate::hardware_calibration) size: vk::DeviceSize,
    mapped: Option<usize>,
}

pub(crate) struct SpecializedImage {
    device: ash::Device,
    pub(in crate::hardware_calibration) image: vk::Image,
    memory: vk::DeviceMemory,
    pub(in crate::hardware_calibration) view: vk::ImageView,
}

pub(crate) struct SpecializedVulkanResources {
    context: Rc<SpecializedVulkanContext>,
    pub(in crate::hardware_calibration) command_pool: vk::CommandPool,
    pub(in crate::hardware_calibration) command_buffer: vk::CommandBuffer,
    pub(in crate::hardware_calibration) fence: vk::Fence,
    pub(in crate::hardware_calibration) query_pool: vk::QueryPool,
    queue: vk::Queue,
}

impl SpecializedVulkanContext {
    pub(in crate::hardware_calibration) fn new(
        physical_device_index: usize,
        requirements: SpecializedVulkanRequirements,
    ) -> Result<Rc<Self>, String> {
        let entry =
            unsafe { Entry::load() }.map_err(|error| format!("could not load Vulkan: {error}"))?;
        let app_name = CString::new("nerve-hardware-calibrator").expect("static string has no nul");
        let engine_name = CString::new("nerve").expect("static string has no nul");
        let application = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::make_api_version(0, 1, 4, 0));
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&application),
                None,
            )
        }
        .map_err(|error| format!("could not create calibration Vulkan instance: {error:?}"))?;
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| format!("could not enumerate calibration devices: {error:?}"))?;
        let physical_device = *physical_devices.get(physical_device_index).ok_or_else(|| {
            format!("Vulkan physical device index {physical_device_index} was not discovered")
        })?;
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        if properties.api_version < vk::make_api_version(0, 1, 4, 0) {
            unsafe { instance.destroy_instance(None) };
            return Err(format!(
                "specialized calibration requires Vulkan 1.4, device reports {}.{}.{}",
                vk::api_version_major(properties.api_version),
                vk::api_version_minor(properties.api_version),
                vk::api_version_patch(properties.api_version)
            ));
        }
        let queue_properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let graphics_queue_family = requirements
            .graphics
            .then(|| queue_family(&queue_properties, vk::QueueFlags::GRAPHICS))
            .transpose()?;
        let compute_queue_family = (requirements.compute
            || requirements.ray_query
            || requirements.device_generated_commands)
            .then(|| queue_family(&queue_properties, vk::QueueFlags::COMPUTE))
            .transpose()?;
        let family_indices = [graphics_queue_family, compute_queue_family]
            .into_iter()
            .flatten()
            .collect::<BTreeSet<_>>();
        if family_indices.is_empty() {
            unsafe { instance.destroy_instance(None) };
            return Err("specialized Vulkan calibration requested no queue family".to_string());
        }
        let queue_priority = [1.0f32];
        let queue_infos = family_indices
            .iter()
            .map(|family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*family)
                    .queue_priorities(&queue_priority)
            })
            .collect::<Vec<_>>();
        let supported_extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device) }
                .map_err(|error| format!("could not enumerate calibration extensions: {error:?}"))?
                .into_iter()
                .map(|extension| {
                    unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) }.to_owned()
                })
                .collect::<BTreeSet<_>>();
        let mut extension_names = Vec::new();
        if requirements.ray_query {
            require_extension(
                &supported_extensions,
                ash::khr::deferred_host_operations::NAME,
                &mut extension_names,
            )?;
            require_extension(
                &supported_extensions,
                ash::khr::acceleration_structure::NAME,
                &mut extension_names,
            )?;
            require_extension(
                &supported_extensions,
                ash::khr::ray_query::NAME,
                &mut extension_names,
            )?;
        }
        if requirements.device_generated_commands {
            require_extension(
                &supported_extensions,
                c"VK_EXT_device_generated_commands",
                &mut extension_names,
            )?;
        }
        let mut synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut timeline =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut buffer_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(requirements.ray_query);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(requirements.ray_query);
        let mut ray_query_features =
            vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(requirements.ray_query);
        let mut device_generated_commands_features =
            PhysicalDeviceDeviceGeneratedCommandsFeaturesExt {
                s_type: 1_000_572_000,
                p_next: std::ptr::null_mut(),
                device_generated_commands: u32::from(requirements.device_generated_commands),
                dynamic_generated_pipeline_layout: 0,
            };
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut synchronization2)
            .push_next(&mut timeline)
            .push_next(&mut buffer_address)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_query_features);
        if requirements.device_generated_commands {
            device_generated_commands_features.p_next = device_info.p_next.cast_mut();
            device_info.p_next = std::ptr::from_ref(&device_generated_commands_features).cast();
        }
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .map_err(|error| {
                unsafe { instance.destroy_instance(None) };
                format!("could not create specialized calibration device: {error:?}")
            })?;
        let queues = family_indices
            .into_iter()
            .map(|family| (family, unsafe { device.get_device_queue(family, 0) }))
            .collect();
        let acceleration_structure = requirements
            .ray_query
            .then(|| ash::khr::acceleration_structure::Device::new(&instance, &device));
        let pci_address = physical_device_pci_address(&instance, physical_device);
        Ok(Rc::new(Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            queues,
            graphics_queue_family,
            compute_queue_family,
            timestamp_period_ns: properties.limits.timestamp_period,
            pci_address,
            acceleration_structure,
        }))
    }

    pub(in crate::hardware_calibration) fn device(&self) -> &ash::Device {
        &self.device
    }

    pub(in crate::hardware_calibration) fn timestamp_period_ns(&self) -> f32 {
        self.timestamp_period_ns
    }

    pub(in crate::hardware_calibration) fn pci_address(&self) -> Option<&str> {
        self.pci_address.as_deref()
    }

    pub(in crate::hardware_calibration) fn graphics_queue_family(&self) -> Result<u32, String> {
        self.graphics_queue_family
            .ok_or_else(|| "specialized device has no graphics queue".to_string())
    }

    pub(in crate::hardware_calibration) fn compute_queue_family(&self) -> Result<u32, String> {
        self.compute_queue_family
            .ok_or_else(|| "specialized device has no compute queue".to_string())
    }

    pub(in crate::hardware_calibration) fn queue(&self, family: u32) -> Result<vk::Queue, String> {
        self.queues
            .get(&family)
            .copied()
            .ok_or_else(|| format!("queue family {family} was not opened"))
    }

    pub(in crate::hardware_calibration) fn device_proc_address(
        &self,
        name: &'static CStr,
    ) -> *const std::ffi::c_void {
        unsafe {
            self.instance
                .get_device_proc_addr(self.device.handle(), name.as_ptr())
                .map(|function| function as *const std::ffi::c_void)
                .unwrap_or(std::ptr::null())
        }
    }

    pub(in crate::hardware_calibration) fn acceleration_structure(
        &self,
    ) -> Result<&ash::khr::acceleration_structure::Device, String> {
        self.acceleration_structure
            .as_ref()
            .ok_or_else(|| "acceleration-structure extension was not enabled".to_string())
    }

    pub(in crate::hardware_calibration) fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory: vk::MemoryPropertyFlags,
        mapped: bool,
    ) -> Result<SpecializedBuffer, String> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }
        .map_err(|error| format!("could not create calibration buffer: {error:?}"))?;
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = self
            .find_memory_type(requirements.memory_type_bits, required_memory)
            .ok_or_else(|| {
                unsafe { self.device.destroy_buffer(buffer, None) };
                format!("no memory type satisfies {required_memory:?}")
            })?;
        let mut allocate_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            allocate_info = allocate_info.push_next(&mut allocate_flags);
        }
        let memory =
            unsafe { self.device.allocate_memory(&allocate_info, None) }.map_err(|error| {
                unsafe { self.device.destroy_buffer(buffer, None) };
                format!("could not allocate calibration buffer memory: {error:?}")
            })?;
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|error| {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_buffer(buffer, None);
            }
            format!("could not bind calibration buffer: {error:?}")
        })?;
        let mapped_address = if mapped {
            Some(
                unsafe {
                    self.device
                        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                }
                .map_err(|error| {
                    unsafe {
                        self.device.free_memory(memory, None);
                        self.device.destroy_buffer(buffer, None);
                    }
                    format!("could not map calibration buffer: {error:?}")
                })? as usize,
            )
        } else {
            None
        };
        Ok(SpecializedBuffer {
            device: self.device.clone(),
            buffer,
            memory,
            size,
            mapped: mapped_address,
        })
    }

    pub(in crate::hardware_calibration) fn create_image(
        &self,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Result<SpecializedImage, String> {
        let image = unsafe {
            self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
        }
        .map_err(|error| format!("could not create calibration image: {error:?}"))?;
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_type = self
            .find_memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                unsafe { self.device.destroy_image(image, None) };
                "no device-local image memory type".to_string()
            })?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .map_err(|error| {
            unsafe { self.device.destroy_image(image, None) };
            format!("could not allocate calibration image memory: {error:?}")
        })?;
        unsafe { self.device.bind_image_memory(image, memory, 0) }.map_err(|error| {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
            }
            format!("could not bind calibration image memory: {error:?}")
        })?;
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
                None,
            )
        }
        .map_err(|error| {
            unsafe {
                self.device.free_memory(memory, None);
                self.device.destroy_image(image, None);
            }
            format!("could not create calibration image view: {error:?}")
        })?;
        Ok(SpecializedImage {
            device: self.device.clone(),
            image,
            memory,
            view,
        })
    }

    pub(in crate::hardware_calibration) fn create_initialized_device_buffer(
        self: &Rc<Self>,
        bytes: &[u8],
        usage: vk::BufferUsageFlags,
    ) -> Result<SpecializedBuffer, String> {
        let size = u64::try_from(bytes.len())
            .map_err(|_| "calibration buffer size exceeds Vulkan limits".to_string())?;
        if let Ok(buffer) = self.create_buffer(
            size,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
                | vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        ) {
            buffer.write(bytes)?;
            return Ok(buffer);
        }
        let staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            true,
        )?;
        staging.write(bytes)?;
        let destination = self.create_buffer(
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            false,
        )?;
        let queue_family = self.compute_queue_family()?;
        let resources = SpecializedVulkanResources::new(Rc::clone(self), queue_family)?;
        resources.begin()?;
        unsafe {
            self.device.cmd_copy_buffer(
                resources.command_buffer,
                staging.buffer,
                destination.buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
            self.device.cmd_pipeline_barrier2(
                resources.command_buffer,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                                | vk::PipelineStageFlags2::COMPUTE_SHADER,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                                | vk::AccessFlags2::SHADER_READ,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(destination.buffer)
                        .offset(0)
                        .size(size),
                ]),
            );
        }
        resources.finish_recording()?;
        resources.run(1_000_000_000)?;
        Ok(destination)
    }

    fn find_memory_type(
        &self,
        memory_type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        (0..properties.memory_type_count).find(|index| {
            (memory_type_bits & (1 << index)) != 0
                && properties.memory_types[*index as usize]
                    .property_flags
                    .contains(required)
        })
    }
}

#[repr(C)]
struct PhysicalDeviceDeviceGeneratedCommandsFeaturesExt {
    s_type: i32,
    p_next: *mut std::ffi::c_void,
    device_generated_commands: u32,
    dynamic_generated_pipeline_layout: u32,
}

impl SpecializedBuffer {
    pub(in crate::hardware_calibration) fn write(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() > self.size as usize {
            return Err(format!(
                "write of {} bytes exceeds calibration buffer size {}",
                bytes.len(),
                self.size
            ));
        }
        let mapped = self
            .mapped
            .ok_or_else(|| "calibration buffer is not host mapped".to_string())?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
        }
        Ok(())
    }

    pub(in crate::hardware_calibration) fn read(
        &self,
        byte_count: usize,
    ) -> Result<Vec<u8>, String> {
        if byte_count > self.size as usize {
            return Err(format!(
                "read of {byte_count} bytes exceeds calibration buffer size {}",
                self.size
            ));
        }
        let mapped = self
            .mapped
            .ok_or_else(|| "calibration buffer is not host mapped".to_string())?;
        let mut output = vec![0u8; byte_count];
        unsafe {
            std::ptr::copy_nonoverlapping(mapped as *const u8, output.as_mut_ptr(), byte_count);
        }
        Ok(output)
    }

    pub(in crate::hardware_calibration) fn device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(self.buffer),
            )
        }
    }
}

impl Drop for SpecializedBuffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for SpecializedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl SpecializedVulkanResources {
    pub(in crate::hardware_calibration) fn new(
        context: Rc<SpecializedVulkanContext>,
        queue_family: u32,
    ) -> Result<Self, String> {
        let device = context.device();
        let queue = context.queue(queue_family)?;
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|error| format!("could not create calibration command pool: {error:?}"))?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| format!("could not allocate calibration command buffer: {error:?}"))?
        .remove(0);
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|error| format!("could not create calibration fence: {error:?}"))?;
        let query_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )
        }
        .map_err(|error| format!("could not create calibration timestamp pool: {error:?}"))?;
        Ok(Self {
            context,
            command_pool,
            command_buffer,
            fence,
            query_pool,
            queue,
        })
    }

    pub(in crate::hardware_calibration) fn begin(&self) -> Result<(), String> {
        let device = self.context.device();
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|error| format!("could not reset calibration commands: {error:?}"))?;
            device
                .begin_command_buffer(
                    self.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| format!("could not begin calibration commands: {error:?}"))?;
            device.cmd_reset_query_pool(self.command_buffer, self.query_pool, 0, 2);
            device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.query_pool,
                0,
            );
        }
        Ok(())
    }

    pub(in crate::hardware_calibration) fn finish_recording(&self) -> Result<(), String> {
        let device = self.context.device();
        unsafe {
            device.cmd_write_timestamp2(
                self.command_buffer,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.query_pool,
                1,
            );
            device
                .end_command_buffer(self.command_buffer)
                .map_err(|error| format!("could not end calibration commands: {error:?}"))
        }
    }

    pub(in crate::hardware_calibration) fn run(&self, timeout_ns: u64) -> Result<u64, String> {
        let device = self.context.device();
        unsafe {
            device
                .reset_fences(&[self.fence])
                .map_err(|error| format!("could not reset calibration fence: {error:?}"))?;
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)],
                    self.fence,
                )
                .map_err(|error| format!("could not submit calibration commands: {error:?}"))?;
            match device.wait_for_fences(&[self.fence], true, timeout_ns) {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => {
                    return Err(format!(
                        "calibration command exceeded bounded wait of {timeout_ns} ns"
                    ));
                }
                Err(error) => {
                    return Err(format!(
                        "could not wait for calibration commands: {error:?}"
                    ));
                }
            }
            let mut timestamps = [0u64; 2];
            device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| format!("could not read calibration timestamps: {error:?}"))?;
            let ticks = timestamps[1].wrapping_sub(timestamps[0]);
            Ok(
                (ticks as f64 * f64::from(self.context.timestamp_period_ns()))
                    .round()
                    .clamp(0.0, u64::MAX as f64) as u64,
            )
        }
    }

    pub(in crate::hardware_calibration) fn run_with_timeline(
        &self,
        semaphore: vk::Semaphore,
        signal_value: u64,
        timeout_ns: u64,
    ) -> Result<u64, String> {
        let device = self.context.device();
        unsafe {
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            let signals = [vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&command_buffers)
                        .signal_semaphore_infos(&signals)],
                    vk::Fence::null(),
                )
                .map_err(|error| {
                    format!("could not submit timeline calibration command: {error:?}")
                })?;
            match device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[semaphore])
                    .values(&[signal_value]),
                timeout_ns,
            ) {
                Ok(()) => {}
                Err(vk::Result::TIMEOUT) => {
                    return Err(format!(
                        "timeline calibration exceeded bounded wait of {timeout_ns} ns"
                    ));
                }
                Err(error) => {
                    return Err(format!(
                        "could not wait for calibration timeline: {error:?}"
                    ));
                }
            }
            let mut timestamps = [0u64; 2];
            device
                .get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
                .map_err(|error| {
                    format!("could not read timeline calibration timestamps: {error:?}")
                })?;
            let ticks = timestamps[1].wrapping_sub(timestamps[0]);
            Ok(
                (ticks as f64 * f64::from(self.context.timestamp_period_ns()))
                    .round()
                    .clamp(0.0, u64::MAX as f64) as u64,
            )
        }
    }
}

impl Drop for SpecializedVulkanResources {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();
            device.destroy_query_pool(self.query_pool, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl Drop for SpecializedVulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn queue_family(
    families: &[vk::QueueFamilyProperties],
    required: vk::QueueFlags,
) -> Result<u32, String> {
    families
        .iter()
        .enumerate()
        .filter(|(_, family)| family.queue_count > 0 && family.queue_flags.contains(required))
        .min_by_key(|(_, family)| family.queue_flags.as_raw().count_ones())
        .map(|(index, _)| index as u32)
        .ok_or_else(|| format!("physical device has no queue supporting {required:?}"))
}

fn require_extension(
    supported: &BTreeSet<std::ffi::CString>,
    name: &'static CStr,
    enabled: &mut Vec<*const i8>,
) -> Result<(), String> {
    if !supported.contains(name) {
        return Err(format!(
            "physical device does not expose required extension {}",
            name.to_string_lossy()
        ));
    }
    enabled.push(name.as_ptr());
    Ok(())
}

fn physical_device_pci_address(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<String> {
    let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties);
    }
    (pci.pci_domain != 0 || pci.pci_bus != 0 || pci.pci_device != 0 || pci.pci_function != 0).then(
        || {
            format!(
                "{:04x}:{:02x}:{:02x}.{}",
                pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function
            )
        },
    )
}
