pub struct VulkanResidentBufferCopy {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    source: vk::Buffer,
    destination: vk::Buffer,
    byte_len: vk::DeviceSize,
    completion_fence: vk::Fence,
    timestamp_query_pool: Option<vk::QueryPool>,
    timestamp_period_ns: f32,
}

pub struct VulkanResidentBufferCopyBatch {
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    completion_fence: vk::Fence,
    copy_count: usize,
}

pub struct VulkanResidentBufferReadback {
    bytes: Vec<u8>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl VulkanResidentBufferReadback {
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    pub fn range_bytes(&self, index: usize) -> Result<&[u8], VulkanError> {
        let range = self.ranges.get(index).ok_or_else(|| {
            VulkanError(format!(
                "resident buffer readback has {} ranges, not index {index}",
                self.ranges.len()
            ))
        })?;
        Ok(&self.bytes[range.clone()])
    }
}

#[derive(Clone, Copy)]
pub struct VulkanResidentBufferReadRange<'a> {
    source: &'a VulkanResidentBuffer,
    source_offset: usize,
    byte_len: usize,
}

impl<'a> VulkanResidentBufferReadRange<'a> {
    pub fn new(
        source: &'a VulkanResidentBuffer,
        source_offset: usize,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        validate_resident_transfer_range(source_offset, byte_len)?;
        source.byte_range(source_offset, byte_len)?;
        Ok(Self {
            source,
            source_offset,
            byte_len,
        })
    }
}

#[derive(Clone, Copy)]
pub struct VulkanResidentBufferWriteRange<'a> {
    destination: &'a VulkanResidentBuffer,
    destination_offset: usize,
    bytes: &'a [u8],
}

impl<'a> VulkanResidentBufferWriteRange<'a> {
    pub fn new(
        destination: &'a VulkanResidentBuffer,
        destination_offset: usize,
        bytes: &'a [u8],
    ) -> Result<Self, VulkanError> {
        validate_resident_transfer_range(destination_offset, bytes.len())?;
        destination.byte_range(destination_offset, bytes.len())?;
        Ok(Self {
            destination,
            destination_offset,
            bytes,
        })
    }
}

#[derive(Clone, Copy)]
pub struct VulkanResidentBufferRangeCopy<'a> {
    source: &'a VulkanResidentBuffer,
    destination: &'a VulkanResidentBuffer,
    source_offset: vk::DeviceSize,
    destination_offset: vk::DeviceSize,
    byte_len: vk::DeviceSize,
}

impl<'a> VulkanResidentBufferRangeCopy<'a> {
    pub fn new(
        source: &'a VulkanResidentBuffer,
        destination: &'a VulkanResidentBuffer,
        source_offset: usize,
        destination_offset: usize,
        byte_len: usize,
    ) -> Result<Self, VulkanError> {
        validate_resident_transfer_range(source_offset, byte_len)?;
        validate_resident_transfer_range(destination_offset, byte_len)?;
        source.byte_range(source_offset, byte_len)?;
        destination.byte_range(destination_offset, byte_len)?;
        Ok(Self {
            source,
            destination,
            source_offset: source_offset as vk::DeviceSize,
            destination_offset: destination_offset as vk::DeviceSize,
            byte_len: byte_len as vk::DeviceSize,
        })
    }
}

fn validate_resident_transfer_range(offset: usize, byte_len: usize) -> Result<(), VulkanError> {
    const VULKAN_BUFFER_COPY_ALIGNMENT: usize = 4;
    if byte_len == 0 {
        return Err(VulkanError(
            "resident buffer transfer length must not be zero".to_string(),
        ));
    }
    if !offset.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT)
        || !byte_len.is_multiple_of(VULKAN_BUFFER_COPY_ALIGNMENT)
    {
        return Err(VulkanError(format!(
            "resident buffer transfer offset and length must be multiples of {VULKAN_BUFFER_COPY_ALIGNMENT}, got offset {offset} and length {byte_len}"
        )));
    }
    Ok(())
}

pub struct VulkanResidentMappedBufferCopy {
    source_address: usize,
    destination_address: usize,
    byte_len: usize,
}

impl VulkanResidentMappedBufferCopy {
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn run(&self, len: usize) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "persistently mapped resident copy length must not be zero".to_string(),
            ));
        }
        if len != self.byte_len {
            return Err(VulkanError(format!(
                "persistently mapped resident copy binding byte length {} cannot run {} bytes",
                self.byte_len, len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.source_address as *const u8,
                self.destination_address as *mut u8,
                len,
            );
        }
        Ok(())
    }
}

impl VulkanResidentBufferCopy {
    pub fn byte_len(&self) -> usize {
        self.byte_len as usize
    }

    pub fn run(&self, len: usize) -> Result<(), VulkanError> {
        self.run_internal(len).map(|_| ())
    }

    pub fn run_with_device_duration(&self, len: usize) -> Result<u64, VulkanError> {
        self.run_internal(len)?.ok_or_else(|| {
            VulkanError(
                "resident byte copy was not created with timestamp measurement".to_string(),
            )
        })
    }

    fn run_internal(&self, len: usize) -> Result<Option<u64>, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte copy length must not be zero".to_string(),
            ));
        }
        let byte_len = len as vk::DeviceSize;
        if byte_len != self.byte_len {
            return Err(VulkanError(format!(
                "resident byte copy binding byte length {} cannot run {} bytes",
                self.byte_len, byte_len
            )));
        }

        unsafe {
            self.device
                .reset_fences(&[self.completion_fence])
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident byte copy completion fence: {error:?}"
                    ))
                })?;
            let command_buffers = [self.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, self.completion_fence)
                .map_err(|error| {
                    VulkanError(format!("failed to submit resident byte copy: {error:?}"))
                })?;
            RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            self.device
                .wait_for_fences(&[self.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!("failed waiting for resident byte copy: {error:?}"))
                })?;
            RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
            let device_duration_ns = if let Some(query_pool) = self.timestamp_query_pool {
                let mut timestamps = [0_u64; 2];
                self.device
                    .get_query_pool_results(
                        query_pool,
                        0,
                        &mut timestamps,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                    .map_err(|error| {
                        VulkanError(format!(
                            "failed to read resident byte copy timestamps: {error:?}"
                        ))
                    })?;
                let duration_ns = timestamps[1].wrapping_sub(timestamps[0]) as f64
                    * f64::from(self.timestamp_period_ns);
                if !duration_ns.is_finite()
                    || duration_ns <= 0.0
                    || duration_ns > u64::MAX as f64
                {
                    return Err(VulkanError(format!(
                        "resident byte copy produced invalid device duration {duration_ns}"
                    )));
                }
                Some((duration_ns.round() as u64).max(1))
            } else {
                None
            };
            Ok(device_duration_ns)
        }
    }
}

impl VulkanResidentBufferCopyBatch {
    pub fn copy_count(&self) -> usize {
        self.copy_count
    }

    pub fn run(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .reset_fences(&[self.completion_fence])
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to reset resident buffer copy batch fence: {error:?}"
                    ))
                })?;
            let command_buffers = [self.command_buffer];
            let submit_info = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            self.device
                .queue_submit(self.queue, &submit_info, self.completion_fence)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to submit resident buffer copy batch: {error:?}"
                    ))
                })?;
            RESIDENT_COPY_QUEUE_SUBMITS.fetch_add(1, Ordering::Relaxed);
            self.device
                .wait_for_fences(&[self.completion_fence], true, u64::MAX)
                .map_err(|error| {
                    VulkanError(format!(
                        "failed waiting for resident buffer copy batch: {error:?}"
                    ))
                })?;
            RESIDENT_COPY_WAITS.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl Drop for VulkanResidentBufferCopy {
    fn drop(&mut self) {
        unsafe {
            if let Some(query_pool) = self.timestamp_query_pool {
                self.device.destroy_query_pool(query_pool, None);
            }
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl Drop for VulkanResidentBufferCopyBatch {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl VulkanResidentKernelDispatch {
    pub fn semantic_label(&self) -> Option<&str> {
        self.semantic_label.as_deref()
    }

    pub fn descriptor_count(&self) -> usize {
        self.descriptor_count
    }

    pub fn workgroup_count_x(&self) -> u32 {
        self.workgroup_count_x
    }

    pub fn workgroup_count_y(&self) -> u32 {
        self.workgroup_count_y
    }

    pub fn local_size_x(&self) -> u32 {
        self.pipeline_key.local_size_x
    }

    pub fn estimated_work_units(&self) -> u64 {
        u64::from(self.workgroup_count_x)
            .saturating_mul(u64::from(self.workgroup_count_y))
            .saturating_mul(u64::from(self.pipeline_key.local_size_x))
    }

    pub fn estimated_memory_bytes(&self) -> u64 {
        self.estimated_memory_bytes
    }

    pub fn execution_family(&self) -> String {
        let operation = self
            .semantic_label
            .as_deref()
            .and_then(|label| semantic_label_field(label, "op"))
            .unwrap_or("unlabeled");
        format!(
            "{operation}@{}x{}x{}",
            self.workgroup_count_x,
            self.workgroup_count_y,
            self.pipeline_key.local_size_x
        )
    }

    pub fn push_constant_byte_count(&self) -> u32 {
        self.push_constant_byte_count
    }
}

impl Drop for VulkanResidentKernelDispatch {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl Drop for VulkanResidentKernelSequence {
    fn drop(&mut self) {
        unsafe {
            if let Some((query_pool, _)) = self.profiling_timestamp_query_pool {
                self.device.destroy_query_pool(query_pool, None);
            }
            if let Some(query_pool) = self.timestamp_query_pool {
                self.device.destroy_query_pool(query_pool, None);
            }
            self.device.destroy_fence(self.completion_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
