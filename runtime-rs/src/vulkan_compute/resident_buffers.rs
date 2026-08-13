impl VulkanResidentBuffer {
    pub(crate) fn command_binding_identity(&self) -> (u64, u64) {
        (self.device.handle().as_raw(), self.buffer.as_raw())
    }

    pub fn persistently_map(&mut self) -> Result<(), VulkanError> {
        if self.persistent_mapping.is_some() {
            return Ok(());
        }
        if !self.memory_access.is_directly_mappable() {
            return Err(VulkanError(
                "resident buffer memory is not host-visible and coherent".to_string(),
            ));
        }
        let pointer = unsafe {
            self.device
                .map_memory(
                    self.memory.ok_or_else(|| {
                        VulkanError(
                            "sparse resident buffers cannot be persistently mapped"
                                .to_string(),
                        )
                    })?,
                    0,
                    self.byte_capacity,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|error| {
                    VulkanError(format!(
                        "failed to persistently map resident buffer memory: {error:?}"
                    ))
                })?
        };
        self.persistent_mapping = Some(pointer as usize);
        self.persistent_mapping_requires_unmap = true;
        Ok(())
    }

    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity as usize
    }

    pub fn device_address(&self) -> Result<vk::DeviceAddress, VulkanError> {
        self.device_address.ok_or_else(|| {
            VulkanError(
                "resident buffer was not created with device-address support"
                    .to_string(),
            )
        })
    }

    pub fn is_device_addressable(&self) -> bool {
        self.device_address.is_some()
    }

    pub fn is_shared_host_backed(&self) -> bool {
        self._shared_host_allocation.is_some()
    }

    pub fn shares_host_allocation_with(&self, other: &Self) -> bool {
        self._shared_host_allocation
            .as_ref()
            .zip(other._shared_host_allocation.as_ref())
            .is_some_and(|(left, right)| Arc::ptr_eq(left, right))
    }

    pub fn is_shared_device_memory_backed(&self) -> bool {
        self._shared_device_memory_identity.is_some()
    }

    pub fn shares_device_memory_with(&self, other: &Self) -> bool {
        self._shared_device_memory_identity
            .as_ref()
            .zip(other._shared_device_memory_identity.as_ref())
            .is_some_and(|(left, right)| Arc::ptr_eq(left, right))
    }

    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.shares_host_allocation_with(other) || self.shares_device_memory_with(other)
    }

    pub fn is_persistently_mapped(&self) -> bool {
        self.persistent_mapping.is_some()
    }

    pub fn create_persistently_mapped_copy_to(
        &self,
        destination: &VulkanResidentBuffer,
        len: usize,
    ) -> Result<VulkanResidentMappedBufferCopy, VulkanError> {
        self.byte_len(len)?;
        destination.byte_len(len)?;
        let source_address = self.persistent_mapping.ok_or_else(|| {
            VulkanError("resident copy source is not persistently mapped".to_string())
        })?;
        let destination_address = destination.persistent_mapping.ok_or_else(|| {
            VulkanError("resident copy destination is not persistently mapped".to_string())
        })?;
        Ok(VulkanResidentMappedBufferCopy {
            source_address,
            destination_address,
            byte_len: len,
        })
    }

    pub fn write_bytes(&self, input: &[u8]) -> Result<(), VulkanError> {
        self.write_bytes_at(0, input)
    }

    pub fn write_bytes_at(&self, offset: usize, input: &[u8]) -> Result<(), VulkanError> {
        if input.is_empty() {
            return Err(VulkanError(
                "resident byte buffer write must not be empty".to_string(),
            ));
        }
        let end = offset
            .checked_add(input.len())
            .ok_or_else(|| VulkanError("resident byte buffer write overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot write {} bytes at offset {}",
                self.byte_capacity,
                input.len(),
                offset
            )));
        }
        let byte_len = input.len() as vk::DeviceSize;
        if let Some(address) = self.persistent_mapping {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    (address as *mut u8).add(offset),
                    input.len(),
                );
            }
            Ok(())
        } else if self.memory_access.is_directly_mappable() {
            unsafe {
                write_byte_memory_at(
                    &self.device,
                    self.memory.ok_or_else(|| {
                        VulkanError(
                            "directly mappable resident buffer has no bound memory"
                                .to_string(),
                        )
                    })?,
                    self.byte_capacity,
                    offset,
                    input,
                )
            }
        } else {
            unsafe {
                write_device_local_bytes(
                    &self.device,
                    self.buffer,
                    &self.memory_access,
                    offset as vk::DeviceSize,
                    byte_len,
                    input,
                )
            }
        }
    }

    pub fn read_bytes(&self, len: usize) -> Result<Vec<u8>, VulkanError> {
        self.read_bytes_at(0, len)
    }

    pub fn read_bytes_at(&self, offset: usize, len: usize) -> Result<Vec<u8>, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer length must not be zero".to_string(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| VulkanError("resident byte buffer read overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot read {} bytes at offset {}",
                self.byte_capacity, len, offset
            )));
        }
        let byte_len = len as vk::DeviceSize;
        if let Some(address) = self.persistent_mapping {
            Ok(
                unsafe { std::slice::from_raw_parts((address as *const u8).add(offset), len) }
                    .to_vec(),
            )
        } else if self.memory_access.is_directly_mappable() {
            unsafe {
                read_byte_memory_at(
                    &self.device,
                    self.memory.ok_or_else(|| {
                        VulkanError(
                            "directly mappable resident buffer has no bound memory"
                                .to_string(),
                        )
                    })?,
                    self.byte_capacity,
                    offset,
                    len,
                )
            }
        } else {
            unsafe {
                read_device_local_bytes(
                    &self.device,
                    self.buffer,
                    &self.memory_access,
                    offset as vk::DeviceSize,
                    byte_len,
                )
            }
        }
    }

    pub fn read_persistently_mapped_u32_le_at(&self, offset: usize) -> Result<u32, VulkanError> {
        let byte_count = std::mem::size_of::<u32>();
        let end = offset
            .checked_add(byte_count)
            .ok_or_else(|| VulkanError("resident u32 read overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot read a u32 at offset {}",
                self.byte_capacity, offset
            )));
        }
        let address = self.persistent_mapping.ok_or_else(|| {
            VulkanError("resident u32 read requires persistent mapping".to_string())
        })?;
        let bytes =
            unsafe { std::slice::from_raw_parts((address as *const u8).add(offset), byte_count) };
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn byte_len(&self, len: usize) -> Result<vk::DeviceSize, VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer length must not be zero".to_string(),
            ));
        }
        let byte_len = len as vk::DeviceSize;
        if byte_len > self.byte_capacity {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot hold {} bytes",
                self.byte_capacity, byte_len
            )));
        }
        Ok(byte_len)
    }

    fn byte_range(&self, offset: usize, len: usize) -> Result<(), VulkanError> {
        if len == 0 {
            return Err(VulkanError(
                "resident byte buffer range length must not be zero".to_string(),
            ));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| VulkanError("resident byte buffer range overflowed".to_string()))?;
        if end > self.byte_capacity as usize {
            return Err(VulkanError(format!(
                "resident byte buffer capacity {} cannot address {} bytes at offset {}",
                self.byte_capacity, len, offset
            )));
        }
        Ok(())
    }

    fn descriptor_buffer(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<vk::DescriptorBufferInfo, VulkanError> {
        self.byte_range(offset, len)?;
        Ok(vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: offset as vk::DeviceSize,
            range: len as vk::DeviceSize,
        })
    }
}

impl VulkanSharedHostAllocation {
    pub fn byte_capacity(&self) -> usize {
        self.byte_capacity
    }
}

impl Drop for VulkanSharedHostAllocation {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.address as *mut u8, self.layout);
        }
    }
}

impl Drop for VulkanTimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

impl Drop for VulkanResidentBuffer {
    fn drop(&mut self) {
        if let (Some(address), Some(registry)) =
            (self.device_address, &self.device_address_registry)
            && let Ok(mut registry) = registry.lock()
        {
            let _ = registry.unregister(self.buffer.as_raw(), address);
        }
        unsafe {
            if self.persistent_mapping_requires_unmap {
                if let Some(memory) = self.memory {
                    self.device.unmap_memory(memory);
                }
            }
            self.device.destroy_buffer(self.buffer, None);
            if let Some(memory) = self.memory {
                self.device.free_memory(memory, None);
            }
        }
    }
}
