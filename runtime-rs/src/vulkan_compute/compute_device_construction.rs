impl VulkanComputeDevice {
    pub fn available_compute_devices() -> Result<Vec<VulkanComputeDeviceInfo>, VulkanError> {
        Ok(VulkanComputeDeviceCatalog::discover()?
            .available_devices
            .clone())
    }

    pub fn new() -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(None, None)
    }

    pub fn new_for_physical_device_index(
        physical_device_index: usize,
    ) -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(Some(physical_device_index), None)
    }

    pub fn new_for_device_uuid(device_uuid: [u8; vk::UUID_SIZE]) -> Result<Self, VulkanError> {
        Self::new_with_physical_device_selector(None, Some(device_uuid))
    }

    fn new_with_physical_device_selector(
        requested_physical_device_index: Option<usize>,
        requested_device_uuid: Option<[u8; vk::UUID_SIZE]>,
    ) -> Result<Self, VulkanError> {
        VulkanComputeDeviceCatalog::discover()?
            .open_device(requested_physical_device_index, requested_device_uuid)
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn physical_device_id(&self) -> &str {
        &self.physical_device_id
    }

    pub fn pci_address(&self) -> Option<&str> {
        self.pci_address.as_deref()
    }

    pub fn has_enabled_device_extension(&self, extension_name: &str) -> bool {
        self.enabled_device_extensions.contains(extension_name)
            || vulkan_core_device_extension_version(extension_name)
                .is_some_and(|minimum| self.api_version >= minimum)
    }

    pub fn supports_conditional_compute_dispatch(&self) -> bool {
        self.conditional_rendering.is_some()
    }

    pub fn has_enabled_shader_feature(&self, feature: VulkanShaderFeature) -> bool {
        self.enabled_shader_features.contains(&feature)
    }

    pub fn supports_subgroup_operation(&self, operation: VulkanSubgroupOperation) -> bool {
        self.subgroup_supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            && self
                .subgroup_supported_operations
                .contains(operation.flag())
    }

    pub fn supports_cooperative_bfloat16_shape(&self, m: u32, n: u32, k: u32) -> bool {
        self.cooperative_bfloat16_shapes.contains(&(m, n, k))
    }

    pub fn supports_cooperative_float8_e4m3_shape(&self, m: u32, n: u32, k: u32) -> bool {
        self.cooperative_float8_e4m3_shapes.contains(&(m, n, k))
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    pub fn supports_compute_local_size_x(&self, local_size_x: u32) -> bool {
        local_size_x > 0
            && local_size_x <= self.max_compute_work_group_invocations
            && local_size_x <= self.max_compute_work_group_size_x
    }

    pub fn min_storage_buffer_offset_alignment(&self) -> usize {
        self.min_storage_buffer_offset_alignment
    }

    pub fn device_local_memory_bytes(&self) -> u64 {
        self.device_local_memory_bytes
    }

    pub fn available_device_local_memory_bytes(&self) -> u64 {
        if !self.memory_budget_supported {
            return self.device_local_memory_bytes;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            self.context
                .instance
                .get_physical_device_memory_properties2(self.physical_device, &mut properties);
        }
        (0..properties.memory_properties.memory_heap_count)
            .filter(|heap_index| {
                properties.memory_properties.memory_heaps[*heap_index as usize]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|heap_index| {
                let index = heap_index as usize;
                budget.heap_budget[index].saturating_sub(budget.heap_usage[index])
            })
            .max()
            .unwrap_or(0)
    }

    pub fn max_compute_work_group_count_x(&self) -> u32 {
        self.max_compute_work_group_count_x
    }

    pub fn supports_shared_host_memory(&self) -> bool {
        self.shared_host_memory_alignment.is_some()
    }

    pub fn supports_shared_device_memory(&self) -> bool {
        self.shared_device_memory_supported
    }

    pub fn supports_opaque_fd_timeline_semaphores(&self) -> bool {
        self.opaque_fd_timeline_semaphore_supported
    }

    pub fn owns_resident_buffer(&self, buffer: &VulkanResidentBuffer) -> bool {
        self.device.handle() == buffer.device.handle()
    }

    pub fn shares_logical_device_with(&self, other: &Self) -> bool {
        self.device.handle() == other.device.handle()
    }

    pub fn shares_physical_device_with(&self, other: &Self) -> bool {
        self.physical_device_id == other.physical_device_id
    }

    pub fn has_distinct_transfer_queue(&self) -> bool {
        self.transfer_queue_is_distinct
    }

    pub fn supports_buffer_device_address(&self) -> bool {
        self.buffer_device_address_supported
    }

}
