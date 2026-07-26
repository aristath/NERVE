pub struct VulkanTextureCalibration {
    device: ash::Device,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion_fence: vk::Fence,
    queue: vk::Queue,
    _source: VulkanResidentBuffer,
    output: VulkanResidentBuffer,
}

impl VulkanComputeDevice {
    pub fn create_texture_calibration(
        &self,
        spirv_words: &[u32],
        linear_filter: bool,
        image_width: u32,
        image_height: u32,
        output_count: u32,
    ) -> Result<VulkanTextureCalibration, VulkanError> {
        if spirv_words.is_empty()
            || image_width == 0
            || image_height == 0
            || output_count == 0
        {
            return Err(VulkanError(
                "texture calibration dimensions and shader must be nonzero".to_string(),
            ));
        }
        validate_spirv_device_contract(
            spirv_words,
            &self.enabled_shader_features,
            self.subgroup_supported_stages,
            self.subgroup_supported_operations,
        )?;
        let image_texels = usize::try_from(image_width)
            .ok()
            .and_then(|width| {
                usize::try_from(image_height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| VulkanError("texture calibration image size overflowed".to_string()))?;
        let image_bytes = image_texels
            .checked_mul(8)
            .ok_or_else(|| VulkanError("texture calibration image bytes overflowed".to_string()))?;
        let output_bytes = usize::try_from(output_count)
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| VulkanError("texture calibration output bytes overflowed".to_string()))?;
        let source = self.create_resident_buffer(image_bytes)?;
        source.write_bytes(&texture_calibration_input(image_texels))?;
        let output = self.create_resident_buffer(output_bytes)?;
        output.write_bytes(&vec![0; output_bytes])?;

        unsafe {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .extent(vk::Extent3D {
                    width: image_width,
                    height: image_height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = self
                .device
                .create_image(&image_info, None)
                .map_err(|error| VulkanError(format!("failed to create sampled image: {error:?}")))?;
            let requirements = self.device.get_image_memory_requirements(image);
            let memory_type_index = find_memory_type(
                &self.context.instance,
                self.physical_device,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            )
            .ok_or_else(|| VulkanError("no device-local sampled-image memory type".to_string()))?;
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(memory_type_index),
                    None,
                )
                .map_err(|error| {
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to allocate sampled-image memory: {error:?}"))
                })?;
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|error| {
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to bind sampled-image memory: {error:?}"))
                })?;
            let image_view = self
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R16G16B16A16_SFLOAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .map_err(|error| {
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to create sampled-image view: {error:?}"))
                })?;
            let filter = if linear_filter {
                vk::Filter::LINEAR
            } else {
                vk::Filter::NEAREST
            };
            let sampler = self
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(filter)
                        .min_filter(filter)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .max_lod(0.0),
                    None,
                )
                .map_err(|error| {
                    self.device.destroy_image_view(image_view, None);
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to create texture sampler: {error:?}"))
                })?;
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let descriptor_set_layout = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|error| {
                    self.device.destroy_sampler(sampler, None);
                    self.device.destroy_image_view(image_view, None);
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to create texture descriptor layout: {error:?}"))
                })?;
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(4)];
            let set_layouts = [descriptor_set_layout];
            let pipeline_layout = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push_range),
                    None,
                )
                .map_err(|error| {
                    self.device
                        .destroy_descriptor_set_layout(descriptor_set_layout, None);
                    self.device.destroy_sampler(sampler, None);
                    self.device.destroy_image_view(image_view, None);
                    self.device.free_memory(memory, None);
                    self.device.destroy_image(image, None);
                    VulkanError(format!("failed to create texture pipeline layout: {error:?}"))
                })?;
            let shader_module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv_words), None)
                .map_err(|error| VulkanError(format!("failed to create texture shader: {error:?}")))?;
            let entry = c"main";
            let shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(entry);
            let pipeline = self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(shader_stage)
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, error)| {
                    VulkanError(format!("failed to create texture pipeline: {error:?}"))
                })?
                .remove(0);
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                },
            ];
            let descriptor_pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(|error| {
                    VulkanError(format!("failed to create texture descriptor pool: {error:?}"))
                })?;
            let descriptor_set = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
                .map_err(|error| {
                    VulkanError(format!("failed to allocate texture descriptor set: {error:?}"))
                })?
                .remove(0);
            let image_descriptor = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(image_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let output_descriptor = [output.descriptor_buffer(0, output_bytes)?];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_descriptor),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&output_descriptor),
            ];
            self.device.update_descriptor_sets(&writes, &[]);
            let command_pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_index),
                    None,
                )
                .map_err(|error| VulkanError(format!("failed to create texture command pool: {error:?}")))?;
            let command_buffer = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|error| VulkanError(format!("failed to allocate texture command buffer: {error:?}")))?
                .remove(0);
            let completion_fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|error| VulkanError(format!("failed to create texture fence: {error:?}")))?;
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                )
                .map_err(|error| VulkanError(format!("failed to begin texture commands: {error:?}")))?;
            let to_transfer = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_transfer,
            );
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                source.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: image_width,
                        height: image_height,
                        depth: 1,
                    })],
            );
            let to_sampled = [vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(to_transfer[0].subresource_range)];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &to_sampled,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &output_count.to_le_bytes(),
            );
            self.device
                .cmd_dispatch(command_buffer, output_count.div_ceil(256), 1, 1);
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
            self.device
                .end_command_buffer(command_buffer)
                .map_err(|error| VulkanError(format!("failed to end texture commands: {error:?}")))?;
            Ok(VulkanTextureCalibration {
                device: self.device.clone(),
                image,
                image_memory: memory,
                image_view,
                sampler,
                descriptor_pool,
                descriptor_set_layout,
                pipeline_layout,
                shader_module,
                pipeline,
                command_pool,
                command_buffer,
                completion_fence,
                queue: self.queue,
                _source: source,
                output,
            })
        }
    }
}

impl VulkanTextureCalibration {
    pub fn run_for(&self, timeout: Duration) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .reset_fences(&[self.completion_fence])
                .map_err(|error| VulkanError(format!("failed to reset texture fence: {error:?}")))?;
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            self.device
                .queue_submit2(
                    self.queue(),
                    &[vk::SubmitInfo2::default().command_buffer_infos(&command_buffers)],
                    self.completion_fence,
                )
                .map_err(|error| VulkanError(format!("failed to submit texture commands: {error:?}")))?;
            let timeout_ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
            self.device
                .wait_for_fences(&[self.completion_fence], true, timeout_ns)
                .map_err(|error| {
                    VulkanError(format!(
                        "texture calibration did not complete within {timeout_ns} ns: {error:?}"
                    ))
                })
        }
    }

    fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn output_bytes(&self, byte_count: usize) -> Result<Vec<u8>, VulkanError> {
        self.output.read_bytes(byte_count)
    }
}

impl Drop for VulkanTextureCalibration {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.image_memory, None);
        }
    }
}

fn texture_calibration_input(texel_count: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(texel_count * 8);
    for index in 0..texel_count {
        for channel in 0..4 {
            let value = (((index * 17 + channel * 31) & 0x3ff) as u16) | 0x3c00;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}
