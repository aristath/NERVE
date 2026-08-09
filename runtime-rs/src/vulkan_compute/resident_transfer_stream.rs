pub struct VulkanResidentTransferStream {
    device: ash::Device,
    device_health: VulkanDeviceHealth,
    device_fault: Option<ash::ext::device_fault::Device>,
    device_address_registry: Arc<Mutex<VulkanDeviceAddressRegistry>>,
    queue_submission: VulkanQueueSubmissionGate,
    consumer_queue_submission: VulkanQueueSubmissionGate,
    queue_is_distinct_from_consumer: bool,
    command_pool: vk::CommandPool,
    slots: Vec<VulkanResidentTransferSlot>,
    timeline: VulkanTimelineSemaphore,
    consumer_completion: VulkanMonotonicQueueCompletion,
    next_timeline_value: u64,
    next_slot_index: usize,
    staging_byte_capacity: usize,
}

struct VulkanResidentTransferSlot {
    staging: VulkanResidentBuffer,
    command_buffer: vk::CommandBuffer,
    pending_timeline_value: u64,
}

struct VulkanResidentConsumerWriteFailure {
    error: VulkanError,
    submission_accepted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanResidentTransferTicket {
    timeline_identity: u64,
    timeline_value: u64,
    uploaded_bytes: usize,
    copy_count: usize,
}

impl VulkanResidentTransferTicket {
    pub fn timeline_value(&self) -> u64 {
        self.timeline_value
    }

    pub fn uploaded_bytes(&self) -> usize {
        self.uploaded_bytes
    }

    pub fn copy_count(&self) -> usize {
        self.copy_count
    }
}

impl VulkanComputeDevice {
    pub fn create_resident_transfer_stream(
        &self,
        staging_slot_count: usize,
        staging_byte_capacity: usize,
    ) -> Result<VulkanResidentTransferStream, VulkanError> {
        if staging_slot_count == 0 {
            return Err(VulkanError(
                "resident transfer stream must have at least one staging slot".to_string(),
            ));
        }
        if staging_byte_capacity == 0 {
            return Err(VulkanError(
                "resident transfer staging capacity must not be zero".to_string(),
            ));
        }
        let command_buffer_count = u32::try_from(staging_slot_count).map_err(|_| {
            VulkanError("resident transfer staging slot count exceeds u32".to_string())
        })?;
        unsafe {
            let command_pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_index)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to create resident transfer command pool: {error:?}"
                    ))
                })?;
            let command_buffers = match self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(command_buffer_count),
            ) {
                Ok(command_buffers) => command_buffers,
                Err(error) => {
                    self.device.destroy_command_pool(command_pool, None);
                    return Err(VulkanError(format!(
                        "failed to allocate resident transfer command buffers: {error:?}"
                    )));
                }
            };
            let timeline = match self.create_timeline_semaphore(0) {
                Ok(timeline) => timeline,
                Err(error) => {
                    self.device.destroy_command_pool(command_pool, None);
                    return Err(error);
                }
            };
            let consumer_completion = match self.create_timeline_semaphore(0) {
                Ok(timeline) => VulkanMonotonicQueueCompletion::new(
                    timeline,
                    self.device_health.clone(),
                ),
                Err(error) => {
                    self.device.destroy_command_pool(command_pool, None);
                    return Err(error);
                }
            };
            let mut slots = Vec::with_capacity(staging_slot_count);
            for command_buffer in command_buffers {
                let mut staging =
                    match self.create_host_visible_resident_buffer(staging_byte_capacity) {
                        Ok(staging) => staging,
                        Err(error) => {
                            self.device.destroy_command_pool(command_pool, None);
                            return Err(error);
                        }
                    };
                if let Err(error) = staging.persistently_map() {
                    self.device.destroy_command_pool(command_pool, None);
                    return Err(error);
                }
                slots.push(VulkanResidentTransferSlot {
                    staging,
                    command_buffer,
                    pending_timeline_value: 0,
                });
            }
            Ok(VulkanResidentTransferStream {
                device: self.device.clone(),
                device_health: self.device_health.clone(),
                device_fault: self.device_fault.clone(),
                device_address_registry: Arc::clone(&self.device_address_registry),
                queue_submission: self.transfer_queue_submission.clone(),
                consumer_queue_submission: self.compute_queue_submission.clone(),
                queue_is_distinct_from_consumer:
                    self.transfer_queue_is_distinct,
                command_pool,
                slots,
                timeline,
                consumer_completion,
                next_timeline_value: 0,
                next_slot_index: 0,
                staging_byte_capacity,
            })
        }
    }
}

impl VulkanResidentTransferStream {
    pub fn staging_slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn staging_byte_capacity(&self) -> usize {
        self.staging_byte_capacity
    }

    pub fn outstanding_transfer_count(&self) -> Result<usize, VulkanError> {
        let completed = self.timeline_value()?;
        Ok(self
            .slots
            .iter()
            .filter(|slot| slot.pending_timeline_value > completed)
            .count())
    }

    pub fn submit(
        &mut self,
        writes: &[VulkanResidentBufferWriteRange<'_>],
    ) -> Result<VulkanResidentTransferTicket, VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        self.device_health.require_healthy()?;
        if writes.is_empty() {
            return Err(VulkanError(
                "resident transfer submission must contain at least one write".to_string(),
            ));
        }
        let mut packed_offsets = Vec::with_capacity(writes.len());
        let packed_byte_count = writes.iter().try_fold(0usize, |offset, write| {
            if write.destination.device.handle() != self.device.handle() {
                return Err(VulkanError(
                    "resident transfer destination belongs to another logical device".to_string(),
                ));
            }
            validate_resident_transfer_range(write.destination_offset, write.bytes.len())?;
            write
                .destination
                .byte_range(write.destination_offset, write.bytes.len())?;
            let end = offset.checked_add(write.bytes.len()).ok_or_else(|| {
                VulkanError("resident transfer packed byte count overflowed".to_string())
            })?;
            packed_offsets.push(offset);
            Ok(end)
        })?;
        if packed_byte_count > self.staging_byte_capacity {
            return Err(VulkanError(format!(
                "resident transfer needs {packed_byte_count} staging bytes but the bounded slot capacity is {}",
                self.staging_byte_capacity
            )));
        }

        let slot_index = self.next_slot_index;
        let pending_value = self.slots[slot_index].pending_timeline_value;
        if pending_value != 0 {
            self.wait_timeline_value(pending_value)?;
        }
        let timeline_value = self
            .next_timeline_value
            .checked_add(1)
            .ok_or_else(|| VulkanError("resident transfer timeline exhausted".to_string()))?;

        let slot = &mut self.slots[slot_index];
        for (write, source_offset) in writes.iter().zip(&packed_offsets) {
            slot.staging.write_bytes_at(*source_offset, write.bytes)?;
        }

        unsafe {
            self.device
                .reset_command_buffer(
                    slot.command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident transfer command buffer: {error:?}"
                    ))
                })?;
            self.device
                .begin_command_buffer(
                    slot.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to begin resident transfer command buffer: {error:?}"
                    ))
                })?;
            for (write, source_offset) in writes.iter().zip(&packed_offsets) {
                self.device.cmd_copy_buffer(
                    slot.command_buffer,
                    slot.staging.buffer,
                    write.destination.buffer,
                    &[vk::BufferCopy {
                        src_offset: *source_offset as vk::DeviceSize,
                        dst_offset: write.destination_offset as vk::DeviceSize,
                        size: write.bytes.len() as vk::DeviceSize,
                    }],
                );
            }
            let visibility_barriers =
                resident_transfer_visibility_barriers(writes);
            self.device.cmd_pipeline_barrier2(
                slot.command_buffer,
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&visibility_barriers),
            );
            self.device
                .end_command_buffer(slot.command_buffer)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to end resident transfer command buffer: {error:?}"
                    ))
                })?;
            let command_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(slot.command_buffer)];
            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.semaphore)
                .value(timeline_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            self.queue_submission
                .submit2(
                    &self.device,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&command_infos)
                        .signal_semaphore_infos(&signal_infos)],
                    vk::Fence::null(),
                )
                .map_err(|error| {
                    let mapped = vulkan_operation_error_with_device_fault(
                        "failed to submit resident transfer",
                        error,
                        self.device_fault.as_ref(),
                        &self.device_address_registry,
                    );
                    vulkan_error_with_device_quarantine(&self.device_health, error, mapped)
                })?;
        }
        RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
        slot.pending_timeline_value = timeline_value;
        self.next_timeline_value = timeline_value;
        self.next_slot_index = (slot_index + 1) % self.slots.len();
        Ok(VulkanResidentTransferTicket {
            timeline_identity: self.timeline.semaphore.as_raw(),
            timeline_value,
            uploaded_bytes: packed_byte_count,
            copy_count: writes.len(),
        })
    }

    fn submit_consumer_serialized(
        &mut self,
        writes: &[VulkanResidentBufferWriteRange<'_>],
    ) -> Result<(), VulkanResidentConsumerWriteFailure> {
        let result = self.submit_consumer_serialized_inner(writes);
        result.map_err(|(error, submission_accepted)| {
            VulkanResidentConsumerWriteFailure {
                error,
                submission_accepted,
            }
        })
    }

    fn submit_consumer_serialized_inner(
        &mut self,
        writes: &[VulkanResidentBufferWriteRange<'_>],
    ) -> Result<(), (VulkanError, bool)> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        self.device_health
            .require_healthy()
            .map_err(|error| (error, false))?;
        if writes.is_empty() {
            return Err((
                VulkanError(
                    "consumer-serialized transfer must contain at least one write"
                        .to_string(),
                ),
                false,
            ));
        }
        let mut packed_offsets = Vec::with_capacity(writes.len());
        let packed_byte_count = writes.iter().try_fold(
            0usize,
            |offset, write| -> Result<usize, VulkanError> {
                if write.destination.device.handle() != self.device.handle() {
                    return Err(VulkanError(
                        "consumer-serialized transfer destination belongs to another logical device"
                            .to_string(),
                    ));
                }
                validate_resident_transfer_range(
                    write.destination_offset,
                    write.bytes.len(),
                )?;
                write.destination.byte_range(
                    write.destination_offset,
                    write.bytes.len(),
                )?;
                let end = offset.checked_add(write.bytes.len()).ok_or_else(|| {
                    VulkanError(
                        "consumer-serialized transfer byte count overflowed"
                            .to_string(),
                    )
                })?;
                packed_offsets.push(offset);
                Ok(end)
            },
        )
        .map_err(|error| (error, false))?;
        if packed_byte_count > self.staging_byte_capacity {
            return Err((
                VulkanError(format!(
                    "consumer-serialized transfer needs {packed_byte_count} staging bytes but the bounded slot capacity is {}",
                    self.staging_byte_capacity
                )),
                false,
            ));
        }

        let slot_index = self.next_slot_index;
        let pending_value = self.slots[slot_index].pending_timeline_value;
        if pending_value != 0 {
            self.wait_timeline_value(pending_value)
                .map_err(|error| (error, false))?;
        }
        let slot = &mut self.slots[slot_index];
        for (write, source_offset) in writes.iter().zip(&packed_offsets) {
            slot.staging
                .write_bytes_at(*source_offset, write.bytes)
                .map_err(|error| (error, false))?;
        }

        unsafe {
            self.device
                .reset_command_buffer(
                    slot.command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|error| {
                    (
                        VulkanError(format!(
                            "failed to reset consumer-serialized transfer command buffer: {error:?}"
                        )),
                        false,
                    )
                })?;
            self.device
                .begin_command_buffer(
                    slot.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| {
                    (
                        VulkanError(format!(
                            "failed to begin consumer-serialized transfer command buffer: {error:?}"
                        )),
                        false,
                    )
                })?;
            for (write, source_offset) in writes.iter().zip(&packed_offsets) {
                self.device.cmd_copy_buffer(
                    slot.command_buffer,
                    slot.staging.buffer,
                    write.destination.buffer,
                    &[vk::BufferCopy {
                        src_offset: *source_offset as vk::DeviceSize,
                        dst_offset: write.destination_offset as vk::DeviceSize,
                        size: write.bytes.len() as vk::DeviceSize,
                    }],
                );
            }
            let visibility_barriers =
                resident_transfer_visibility_barriers(writes);
            self.device.cmd_pipeline_barrier2(
                slot.command_buffer,
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&visibility_barriers),
            );
            self.device
                .end_command_buffer(slot.command_buffer)
                .map_err(|error| {
                    (
                        VulkanError(format!(
                            "failed to end consumer-serialized transfer command buffer: {error:?}"
                        )),
                        false,
                    )
                })?;
            let completion_value = self
                .consumer_completion
                .reserve("consumer-serialized resident transfer")
                .map_err(|error| (error, false))?;
            let command_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(slot.command_buffer)];
            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.consumer_completion.semaphore())
                .value(completion_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            if let Err(error) = self.consumer_queue_submission.submit2(
                &self.device,
                &[vk::SubmitInfo2::default()
                    .command_buffer_infos(&command_infos)
                    .signal_semaphore_infos(&signal_infos)],
                vk::Fence::null(),
            ) {
                self.consumer_completion.cancel(completion_value);
                let mapped = vulkan_operation_error_with_device_fault(
                    "failed to submit consumer-serialized resident transfer",
                    error,
                    self.device_fault.as_ref(),
                    &self.device_address_registry,
                );
                return Err((
                    vulkan_error_with_device_quarantine(&self.device_health, error, mapped),
                    false,
                ));
            }
            let wait_result = wait_for_vulkan_timeline_points_with_progress_watchdog(
                &self.device,
                &[self.consumer_completion.semaphore()],
                &[completion_value],
                false,
                &self.device_health,
                "consumer-serialized resident transfer",
                |error| {
                    vulkan_operation_error_with_device_fault(
                        "failed waiting for consumer-serialized resident transfer",
                        error,
                        self.device_fault.as_ref(),
                        &self.device_address_registry,
                    )
                },
            );
            if let Err(error) = wait_result {
                return Err((
                    error,
                    true,
                ));
            }
            self.consumer_completion
                .complete(completion_value)
                .map_err(|error| (error, true))?;
        }
        RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
        RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        slot.pending_timeline_value = 0;
        self.next_slot_index = (slot_index + 1) % self.slots.len();
        self.device_health
            .require_healthy()
            .map_err(|error| (error, true))
    }

    pub fn completion_point<'a>(
        &'a self,
        ticket: &VulkanResidentTransferTicket,
    ) -> Result<VulkanTimelineSemaphorePoint<'a>, VulkanError> {
        self.validate_ticket(ticket)?;
        Ok(VulkanTimelineSemaphorePoint::new(
            &self.timeline,
            ticket.timeline_value,
        ))
    }

    pub fn is_complete(
        &self,
        ticket: &VulkanResidentTransferTicket,
    ) -> Result<bool, VulkanError> {
        self.validate_ticket(ticket)?;
        Ok(self.timeline_value()? >= ticket.timeline_value)
    }

    pub fn wait(
        &self,
        ticket: &VulkanResidentTransferTicket,
    ) -> Result<(), VulkanError> {
        let _transfer = runtime_critical_path_span(RuntimeCriticalPathPhase::CrossDeviceTransfer);
        self.validate_ticket(ticket)?;
        if self.queue_is_distinct_from_consumer {
            self.wait_timeline_value_on_consumer_queue(ticket.timeline_value)
        } else {
            self.wait_timeline_value(ticket.timeline_value)
        }
    }

    fn validate_ticket(
        &self,
        ticket: &VulkanResidentTransferTicket,
    ) -> Result<(), VulkanError> {
        if ticket.timeline_identity != self.timeline.semaphore.as_raw()
            || ticket.timeline_value == 0
            || ticket.timeline_value > self.next_timeline_value
        {
            return Err(VulkanError(
                "resident transfer ticket does not belong to this stream".to_string(),
            ));
        }
        Ok(())
    }

    fn timeline_value(&self) -> Result<u64, VulkanError> {
        self.device_health.require_healthy()?;
        let value = unsafe { self.device.get_semaphore_counter_value(self.timeline.semaphore) }
            .map_err(|error| {
                VulkanError(format!(
                    "failed to read resident transfer timeline: {error:?}"
                ))
            })?;
        self.device_health.require_healthy()?;
        Ok(value)
    }

    fn wait_timeline_value(&self, value: u64) -> Result<(), VulkanError> {
        let _wait = runtime_critical_path_span(RuntimeCriticalPathPhase::HostSynchronization);
        wait_for_vulkan_timeline_points_with_progress_watchdog(
            &self.device,
            &[self.timeline.semaphore],
            &[value],
            false,
            &self.device_health,
            "resident transfer timeline",
            |error| {
                vulkan_operation_error_with_device_fault(
                    &format!("failed waiting for resident transfer timeline value {value}"),
                    error,
                    self.device_fault.as_ref(),
                    &self.device_address_registry,
                )
            },
        )?;
        RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn wait_timeline_value_on_consumer_queue(
        &self,
        value: u64,
    ) -> Result<(), VulkanError> {
        let _wait = runtime_critical_path_span(RuntimeCriticalPathPhase::HostSynchronization);
        self.device_health.require_healthy()?;
        unsafe {
            let completion_value = self
                .consumer_completion
                .reserve("resident transfer compute-queue bridge")?;
            let wait_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let signal_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.consumer_completion.semaphore())
                .value(completion_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let submit_result = self.consumer_queue_submission.submit2(
                &self.device,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait_infos)
                    .signal_semaphore_infos(&signal_infos)],
                vk::Fence::null(),
            );
            if let Err(error) = submit_result {
                self.consumer_completion.cancel(completion_value);
                let mapped = vulkan_operation_error_with_device_fault(
                    &format!("failed to bridge resident transfer timeline value {value} to the compute queue"),
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
            wait_for_vulkan_timeline_points_with_progress_watchdog(
                &self.device,
                &[self.consumer_completion.semaphore()],
                &[completion_value],
                false,
                &self.device_health,
                "resident transfer compute-queue bridge",
                |error| {
                    vulkan_operation_error_with_device_fault(
                        &format!("failed waiting for resident transfer timeline value {value} on the compute queue"),
                        error,
                        self.device_fault.as_ref(),
                        &self.device_address_registry,
                    )
                },
            )?;
            self.consumer_completion.complete(completion_value)?;
        }
        RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        self.device_health.require_healthy()
    }
}

fn resident_transfer_visibility_barriers(
    writes: &[VulkanResidentBufferWriteRange<'_>],
) -> Vec<vk::BufferMemoryBarrier2<'static>> {
    writes
        .iter()
        .map(|write| {
            vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::COPY
                        | vk::PipelineStageFlags2::DRAW_INDIRECT,
                )
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE
                        | vk::AccessFlags2::TRANSFER_READ
                        | vk::AccessFlags2::TRANSFER_WRITE
                        | vk::AccessFlags2::INDIRECT_COMMAND_READ,
                )
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(write.destination.buffer)
                .offset(write.destination_offset as vk::DeviceSize)
                .size(write.bytes.len() as vk::DeviceSize)
        })
        .collect()
}

impl Drop for VulkanResidentTransferStream {
    fn drop(&mut self) {
        if self.next_timeline_value != 0 {
            let _ = self.wait_timeline_value(self.next_timeline_value);
        }
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
