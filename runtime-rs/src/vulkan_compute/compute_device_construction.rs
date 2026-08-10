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
        let physical_device_id = format!("vulkan-uuid:{}", format_device_uuid(&device_uuid));
        let allowlist = BTreeSet::from([physical_device_id]);
        VulkanComputeDeviceCatalog::discover_allowed_physical_device_ids(&allowlist)?
            .open_device_uuid(device_uuid)
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

    pub fn supports_cooperative_sint8_shape(&self, m: u32, n: u32, k: u32) -> bool {
        self.cooperative_sint8_shapes.contains(&(m, n, k))
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
        query_available_device_local_memory_bytes(
            &self.context.instance,
            self.physical_device,
            self.memory_budget_supported,
            self.device_local_memory_bytes,
        )
    }

    pub fn device_local_memory_budget(&self) -> VulkanDeviceLocalMemoryBudget {
        self.device_local_memory_budget
    }

    pub fn remaining_reservable_device_local_memory_bytes(&self) -> u64 {
        self.device_local_memory_accounting()
            .map(|accounting| accounting.remaining_bytes)
            .unwrap_or(0)
    }

    pub fn admit_device_local_memory(
        &self,
        pending_fixed_bytes: u64,
    ) -> Result<VulkanDeviceLocalMemoryAdmission, VulkanError> {
        let accounting = self.device_local_memory_accounting()?;
        let allocatable_bytes = accounting
            .admissible_remaining_bytes
            .checked_sub(pending_fixed_bytes)
            .ok_or_else(|| {
                VulkanError(format!(
                    "pending fixed Vulkan residency needs {pending_fixed_bytes} bytes, but the stable device-local budget has only {} bytes remaining after {} tracked and {} untracked acquired bytes",
                    accounting.admissible_remaining_bytes,
                    accounting.tracked_allocation_bytes,
                    accounting.untracked_acquired_bytes,
                ))
            })?;
        Ok(VulkanDeviceLocalMemoryAdmission {
            baseline_available_bytes: accounting.baseline_available_bytes,
            currently_available_bytes: accounting.currently_available_bytes,
            reservable_bytes: accounting.reservable_bytes,
            acquired_bytes: accounting
                .tracked_allocation_bytes
                .saturating_add(accounting.untracked_acquired_bytes),
            pending_fixed_bytes,
            allocatable_bytes,
        })
    }

    pub fn device_local_memory_accounting(
        &self,
    ) -> Result<VulkanDeviceLocalMemoryAccounting, VulkanError> {
        let currently_available_bytes = self.available_device_local_memory_bytes();
        self.device_local_memory_budget_tracker
            .lock()
            .map(|tracker| tracker.accounting_at(currently_available_bytes))
            .map_err(|_| {
                VulkanError("device-local memory budget tracker was poisoned".to_string())
            })
    }

    /// Restores the device-local headroom protected when this physical device
    /// was opened. Residency reclaimers retire evictable resources through
    /// queue-ordered address invalidation before releasing their allocations.
    pub fn ensure_device_local_memory_headroom(
        &self,
    ) -> Result<VulkanDeviceLocalMemoryAccounting, VulkanError> {
        let recent = VulkanDeviceLocalMemoryBudgetTracker::recent_execution_accounting(
            &self.device_local_memory_budget_tracker,
            VULKAN_EXECUTION_HEADROOM_OBSERVATION_MAXIMUM_AGE,
        )?
        .map(Ok)
        .unwrap_or_else(|| {
            VulkanDeviceLocalMemoryBudgetTracker::execution_accounting(
                &self.device_local_memory_budget_tracker,
                Duration::ZERO,
                || self.available_device_local_memory_bytes(),
            )
        })?;
        if self
            .device_local_memory_budget
            .protected_headroom_deficit_at(recent.currently_available_bytes)
            == 0
        {
            return Ok(recent);
        }
        let reclaimers = VulkanDeviceLocalMemoryBudgetTracker::live_reclaimers(
            &self.device_local_memory_budget_tracker,
        )?;
        restore_protected_device_local_headroom(
            self.device_local_memory_budget,
            reclaimers,
            Duration::from_millis(250),
            || self.device_local_memory_accounting(),
        )
    }

    fn reserve_device_local_memory(
        &self,
        byte_count: u64,
    ) -> Result<Arc<VulkanDeviceLocalMemoryReservation>, VulkanError> {
        let initial = VulkanDeviceLocalMemoryReservation::acquire(
            &self.device_local_memory_budget_tracker,
            self.available_device_local_memory_bytes(),
            byte_count,
        );
        let Err(initial_error) = initial else {
            return initial;
        };
        let requested_bytes = usize::try_from(byte_count).map_err(|_| {
            VulkanError("device-local allocation request exceeds usize".to_string())
        })?;
        let reclaimers = VulkanDeviceLocalMemoryBudgetTracker::live_reclaimers(
            &self.device_local_memory_budget_tracker,
        )?;
        if reclaimers.is_empty() {
            return Err(initial_error);
        }
        let mut reclaimed_bytes = 0usize;
        let mut reclaimer_errors = Vec::new();
        for reclaimer in reclaimers {
            let available_bytes = self
                .device_local_memory_accounting()?
                .admissible_remaining_bytes;
            let requested_reclaim_bytes = usize::try_from(
                byte_count.saturating_sub(available_bytes),
            )
            .unwrap_or(requested_bytes);
            if requested_reclaim_bytes == 0 {
                return VulkanDeviceLocalMemoryReservation::acquire(
                    &self.device_local_memory_budget_tracker,
                    self.available_device_local_memory_bytes(),
                    byte_count,
                );
            }
            match reclaimer.reclaim_device_local_memory(requested_reclaim_bytes) {
                Ok(reclaimed) => reclaimed_bytes = reclaimed_bytes.saturating_add(reclaimed),
                Err(error) => reclaimer_errors.push(error.to_string()),
            }
            let started = Instant::now();
            loop {
                match VulkanDeviceLocalMemoryReservation::acquire(
                    &self.device_local_memory_budget_tracker,
                    self.available_device_local_memory_bytes(),
                    byte_count,
                ) {
                    Ok(reservation) => return Ok(reservation),
                    Err(_) if started.elapsed() < Duration::from_millis(250) => {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    Err(_) => break,
                }
            }
        }
        Err(VulkanError(format!(
            "{initial_error}; registered evictable stores released {reclaimed_bytes} bytes but the allocation still could not be admitted{}",
            if reclaimer_errors.is_empty() {
                String::new()
            } else {
                format!("; reclaimer errors: {}", reclaimer_errors.join(" | "))
            }
        )))
    }

    pub fn register_device_local_memory_reclaimer(
        &self,
        reclaimer: Arc<dyn VulkanDeviceLocalMemoryReclaimer>,
    ) -> Result<VulkanDeviceLocalMemoryReclaimerRegistration, VulkanError> {
        VulkanDeviceLocalMemoryBudgetTracker::register_reclaimer(
            &self.device_local_memory_budget_tracker,
            reclaimer,
        )
    }

    pub fn reserve_device_local_memory_capacity(
        &self,
        byte_count: usize,
    ) -> Result<VulkanDeviceLocalMemoryPermit, VulkanError> {
        VulkanDeviceLocalMemoryPermit::acquire(
            &self.device_local_memory_budget_tracker,
            self.available_device_local_memory_bytes(),
            u64::try_from(byte_count).map_err(|_| {
                VulkanError("device-local capacity permit exceeds u64".to_string())
            })?,
        )
    }

    fn commit_device_local_memory_capacity(
        &self,
        permit: VulkanDeviceLocalMemoryPermit,
        byte_count: u64,
    ) -> Result<Arc<VulkanDeviceLocalMemoryReservation>, VulkanError> {
        if !Arc::ptr_eq(
            &permit.tracker,
            &self.device_local_memory_budget_tracker,
        ) {
            return Err(VulkanError(
                "device-local capacity permit belongs to another physical device".to_string(),
            ));
        }
        permit.commit(byte_count)
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
