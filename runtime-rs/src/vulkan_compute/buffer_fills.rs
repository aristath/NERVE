#[derive(Clone, Copy)]
pub struct VulkanResidentBufferFillRange<'a> {
    destination: &'a VulkanResidentBuffer,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
    value: u32,
}

impl<'a> VulkanResidentBufferFillRange<'a> {
    pub fn new(
        destination: &'a VulkanResidentBuffer,
        destination_offset: usize,
        byte_len: usize,
        value: u32,
    ) -> Result<Self, VulkanError> {
        validate_resident_transfer_range(destination_offset, byte_len)?;
        destination.byte_range(destination_offset, byte_len)?;
        Ok(Self {
            destination,
            destination_offset: destination_offset as vk::DeviceSize,
            byte_len: byte_len as vk::DeviceSize,
            value,
        })
    }
}

pub struct VulkanResidentBufferFillBatch {
    device: ash::Device,
    device_health: VulkanDeviceHealth,
    device_fault: Option<ash::ext::device_fault::Device>,
    device_address_registry: Arc<Mutex<VulkanDeviceAddressRegistry>>,
    queue_submission: VulkanQueueSubmissionGate,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion: Rc<VulkanMonotonicQueueCompletion>,
    total_byte_count: usize,
}

impl VulkanResidentBufferFillBatch {
    pub fn total_byte_count(&self) -> usize {
        self.total_byte_count
    }

    pub fn run(&self) -> Result<(), VulkanError> {
        self.device_health.require_healthy()?;
        let completion_value = self.completion.reserve("resident buffer fill batch")?;
        unsafe {
            let command_buffers =
                [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
            let completion_signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.completion.semaphore())
                .value(completion_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let submit_info = [vk::SubmitInfo2::default()
                .command_buffer_infos(&command_buffers)
                .signal_semaphore_infos(&completion_signal)];
            if let Err(error) =
                self.queue_submission
                    .submit2(&self.device, &submit_info, vk::Fence::null())
            {
                self.completion.cancel(completion_value);
                let mapped = vulkan_operation_error_with_device_fault(
                    "failed to submit resident buffer fill batch",
                    error,
                    self.device_fault.as_ref(),
                    &self.device_address_registry,
                );
                return Err(vulkan_error_with_device_quarantine(
                    &self.device_health,
                    error,
                    mapped,
                ));
            }
            let mut progress_points = vec![(self.completion.semaphore(), completion_value)];
            if let Some(progress) = self.queue_submission.latest_progress_point() {
                progress_points.push(progress);
            }
            wait_for_vulkan_timeline_points_with_progress_sources(
                &self.device,
                &[self.completion.semaphore()],
                &[completion_value],
                false,
                &self.device_health,
                "resident buffer fill batch",
                VulkanQueueProgressSources {
                    timeline_points: &progress_points,
                    timestamp_query_pool: None,
                },
                |error| {
                    vulkan_operation_error_with_device_fault(
                        "failed waiting for resident buffer fill batch",
                        error,
                        self.device_fault.as_ref(),
                        &self.device_address_registry,
                    )
                },
            )?;
            self.completion.complete(completion_value)?;
        }
        self.device_health.require_healthy()
    }
}

impl Drop for VulkanResidentBufferFillBatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl VulkanComputeDevice {
    pub fn create_resident_buffer_fill_batch(
        &self,
        fills: &[VulkanResidentBufferFillRange<'_>],
    ) -> Result<VulkanResidentBufferFillBatch, VulkanError> {
        if fills.is_empty() {
            return Err(VulkanError(
                "resident buffer fill batch must contain at least one range".to_string(),
            ));
        }
        let mut total_byte_count = 0usize;
        for (index, fill) in fills.iter().enumerate() {
            if fill.destination.device.handle() != self.device.handle() {
                return Err(VulkanError(format!(
                    "resident buffer fill range {index} belongs to another logical device"
                )));
            }
            total_byte_count = total_byte_count
                .checked_add(fill.byte_len as usize)
                .ok_or_else(|| VulkanError("resident buffer fill capacity overflowed".to_string()))?;
            let fill_start = fill.destination_offset;
            let fill_end = fill_start + fill.byte_len;
            if fills[..index].iter().any(|earlier| {
                earlier.destination.buffer == fill.destination.buffer
                    && fill_start < earlier.destination_offset + earlier.byte_len
                    && earlier.destination_offset < fill_end
            }) {
                return Err(VulkanError(format!(
                    "resident buffer fill range {index} overlaps an earlier range"
                )));
            }
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
                        "failed to create resident buffer fill command pool: {error:?}"
                    ))
                })?;
            let result = (|| {
                let command_buffer = self
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to allocate resident buffer fill command buffer: {error:?}"
                        ))
                    })?
                    .remove(0);
                self.device
                    .begin_command_buffer(
                        command_buffer,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE),
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to begin resident buffer fill command buffer: {error:?}"
                        ))
                    })?;

                let predecessor_barriers = fills
                    .iter()
                    .map(|fill| {
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(
                                vk::PipelineStageFlags2::ALL_COMMANDS
                                    | vk::PipelineStageFlags2::HOST,
                            )
                            .src_access_mask(
                                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                            )
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(fill.destination.buffer)
                            .offset(fill.destination_offset)
                            .size(fill.byte_len)
                    })
                    .collect::<Vec<_>>();
                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default()
                        .buffer_memory_barriers(&predecessor_barriers),
                );
                for fill in fills {
                    self.device.cmd_fill_buffer(
                        command_buffer,
                        fill.destination.buffer,
                        fill.destination_offset,
                        fill.byte_len,
                        fill.value,
                    );
                }
                let consumer_barriers = fills
                    .iter()
                    .map(|fill| {
                        vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(
                                vk::PipelineStageFlags2::ALL_COMMANDS
                                    | vk::PipelineStageFlags2::HOST,
                            )
                            .dst_access_mask(
                                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                            )
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .buffer(fill.destination.buffer)
                            .offset(fill.destination_offset)
                            .size(fill.byte_len)
                    })
                    .collect::<Vec<_>>();
                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&consumer_barriers),
                );
                self.device
                    .end_command_buffer(command_buffer)
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to end resident buffer fill command buffer: {error:?}"
                        ))
                    })?;
                let completion = Rc::new(VulkanMonotonicQueueCompletion::new(
                    self.create_timeline_semaphore(0).map_err(|error| {
                        VulkanError(format!(
                            "failed to create resident buffer fill completion timeline: {error}"
                        ))
                    })?,
                    self.device_health.clone(),
                ));
                Ok(VulkanResidentBufferFillBatch {
                    device: self.device.clone(),
                    device_health: self.device_health.clone(),
                    device_fault: self.device_fault.clone(),
                    device_address_registry: Arc::clone(&self.device_address_registry),
                    queue_submission: self.compute_queue_submission.clone(),
                    command_pool,
                    command_buffer,
                    completion,
                    total_byte_count,
                })
            })();
            if result.is_err() {
                self.device.destroy_command_pool(command_pool, None);
            }
            result
        }
    }

    pub fn fill_resident_buffer_ranges(
        &self,
        fills: &[VulkanResidentBufferFillRange<'_>],
    ) -> Result<usize, VulkanError> {
        if fills.is_empty() {
            return Ok(0);
        }
        let batch = self.create_resident_buffer_fill_batch(fills)?;
        batch.run()?;
        Ok(batch.total_byte_count())
    }
}
