#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VulkanResidentBufferPoolKey {
    pub namespace: String,
    pub device_id: String,
    pub resource_id: String,
    pub content_identity: String,
    pub byte_offset: usize,
    pub byte_count: usize,
}

impl VulkanResidentBufferPoolKey {
    pub fn new(
        namespace: impl Into<String>,
        device_id: impl Into<String>,
        resource_id: impl Into<String>,
        content_identity: impl Into<String>,
        byte_offset: usize,
        byte_count: usize,
    ) -> Result<Self, VulkanError> {
        let key = Self {
            namespace: namespace.into(),
            device_id: device_id.into(),
            resource_id: resource_id.into(),
            content_identity: content_identity.into(),
            byte_offset,
            byte_count,
        };
        if key.namespace.is_empty()
            || key.device_id.is_empty()
            || key.resource_id.is_empty()
            || key.content_identity.is_empty()
            || key.byte_count == 0
            || key.byte_offset.checked_add(key.byte_count).is_none()
        {
            return Err(VulkanError(
                "resident buffer pool key is invalid".to_string(),
            ));
        }
        Ok(key)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentBufferPoolStats {
    pub resident_buffer_count: usize,
    pub resident_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub eviction_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanResidentBufferPoolDeviceRelease {
    pub resident_buffer_count: usize,
    pub resident_bytes: usize,
}

#[derive(Default)]
struct VulkanResidentBufferPoolState {
    buffers: BTreeMap<
        VulkanResidentBufferPoolKey,
        Arc<VulkanResidentBuffer>,
    >,
    hit_count: u64,
    miss_count: u64,
    eviction_count: u64,
    eviction_bytes: u64,
}

#[derive(Default)]
pub struct VulkanResidentBufferPool {
    state: RefCell<VulkanResidentBufferPoolState>,
    device_lifetime_guards:
        RefCell<BTreeMap<String, Rc<VulkanComputeDevice>>>,
}

impl VulkanResidentBufferPool {
    pub fn register_device(
        &self,
        device_id: impl Into<String>,
        device: Rc<VulkanComputeDevice>,
    ) -> Result<(), VulkanError> {
        let device_id = device_id.into();
        if device_id.is_empty() {
            return Err(VulkanError(
                "resident buffer pool device id is empty".to_string(),
            ));
        }
        let mut guards = self.device_lifetime_guards.borrow_mut();
        if let Some(existing) = guards.get(&device_id) {
            if !Rc::ptr_eq(existing, &device) {
                return Err(VulkanError(format!(
                    "resident buffer pool device id {device_id:?} was rebound to another Vulkan device"
                )));
            }
            return Ok(());
        }
        guards.insert(device_id, device);
        Ok(())
    }

    pub fn resident_buffer(
        &self,
        key: &VulkanResidentBufferPoolKey,
    ) -> Option<Arc<VulkanResidentBuffer>> {
        let mut state = self.state.borrow_mut();
        let buffer = state.buffers.get(key).cloned();
        if buffer.is_some() {
            state.hit_count = state.hit_count.saturating_add(1);
        } else {
            state.miss_count = state.miss_count.saturating_add(1);
        }
        buffer
    }

    pub fn allocate_unpublished(
        &self,
        key: &VulkanResidentBufferPoolKey,
    ) -> Result<Arc<VulkanResidentBuffer>, VulkanError> {
        let device = self
            .device_lifetime_guards
            .borrow()
            .get(&key.device_id)
            .cloned()
            .ok_or_else(|| {
                VulkanError(format!(
                    "resident buffer pool has no registered device {:?}",
                    key.device_id
                ))
            })?;
        match device.create_resident_buffer(key.byte_count) {
            Ok(buffer) => Ok(Arc::new(buffer)),
            Err(first_error) => {
                if self.evict_unreferenced() == 0 {
                    return Err(first_error);
                }
                device
                    .create_resident_buffer(key.byte_count)
                    .map(Arc::new)
                    .map_err(|retry_error| {
                        VulkanError(format!(
                            "resident parameter allocation failed before and after evicting idle pooled buffers: first={first_error}; retry={retry_error}"
                        ))
                    })
            }
        }
    }

    pub fn publish(
        &self,
        key: VulkanResidentBufferPoolKey,
        buffer: Arc<VulkanResidentBuffer>,
    ) -> Result<(), VulkanError> {
        if buffer.byte_capacity() != key.byte_count {
            return Err(VulkanError(format!(
                "resident buffer pool publication byte capacity {} does not match key byte count {}",
                buffer.byte_capacity(),
                key.byte_count,
            )));
        }
        let mut state = self.state.borrow_mut();
        if let Some(existing) = state.buffers.get(&key) {
            if !Arc::ptr_eq(existing, &buffer) {
                return Err(VulkanError(
                    "resident buffer pool key was published twice with different buffers"
                        .to_string(),
                ));
            }
            return Ok(());
        }
        state.buffers.insert(key, buffer);
        Ok(())
    }

    pub fn evict_unreferenced(&self) -> usize {
        let mut state = self.state.borrow_mut();
        let evicted = state
            .buffers
            .iter()
            .filter(|(_, buffer)| Arc::strong_count(buffer) == 1)
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let mut evicted_bytes = 0u64;
        for key in &evicted {
            if let Some(buffer) = state.buffers.remove(key) {
                evicted_bytes = evicted_bytes
                    .saturating_add(buffer.byte_capacity() as u64);
            }
        }
        state.eviction_count = state
            .eviction_count
            .saturating_add(evicted.len() as u64);
        state.eviction_bytes =
            state.eviction_bytes.saturating_add(evicted_bytes);
        evicted.len()
    }

    /// Evict idle pooled allocations for one still-registered device.
    ///
    /// Validation uses this before replacing a package whose placement moves
    /// parameters between devices. Keeping the device registered preserves
    /// its context while removing the previous placement before any
    /// replacement allocation can transiently overcommit VRAM.
    pub fn evict_unreferenced_device(
        &self,
        device_id: &str,
    ) -> Result<VulkanResidentBufferPoolDeviceRelease, VulkanError> {
        if device_id.is_empty() {
            return Err(VulkanError(
                "resident buffer pool eviction device id is empty".to_string(),
            ));
        }
        if !self
            .device_lifetime_guards
            .borrow()
            .contains_key(device_id)
        {
            return Err(VulkanError(format!(
                "resident buffer pool has no registered device {device_id:?}"
            )));
        }
        let mut state = self.state.borrow_mut();
        let pool_owners_by_buffer = state.buffers.values().fold(
            BTreeMap::<usize, usize>::new(),
            |mut owners, buffer| {
                *owners.entry(Arc::as_ptr(buffer) as usize).or_default() += 1;
                owners
            },
        );
        let keys = state
            .buffers
            .iter()
            .filter(|(key, buffer)| {
                key.device_id == device_id
                    && Arc::strong_count(buffer)
                        == pool_owners_by_buffer
                            .get(&(Arc::as_ptr(buffer) as usize))
                            .copied()
                            .unwrap_or_default()
            })
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let mut eviction = VulkanResidentBufferPoolDeviceRelease::default();
        for key in keys {
            let buffer = state
                .buffers
                .remove(&key)
                .expect("enumerated idle resident buffer remains present");
            eviction.resident_buffer_count =
                eviction.resident_buffer_count.saturating_add(1);
            eviction.resident_bytes = eviction
                .resident_bytes
                .saturating_add(buffer.byte_capacity());
            drop(buffer);
        }
        state.eviction_count = state
            .eviction_count
            .saturating_add(eviction.resident_buffer_count as u64);
        state.eviction_bytes = state
            .eviction_bytes
            .saturating_add(eviction.resident_bytes as u64);
        Ok(eviction)
    }

    /// Release one device's pooled allocations and its lifetime guard.
    ///
    /// This is intentionally stricter than ordinary cache eviction: shutdown
    /// is invalid while any mounted package still owns one of these buffers.
    /// Removing the guard only after all buffers are destroyed keeps the
    /// Vulkan device alive for every `vkFreeMemory` call.
    pub fn release_device(
        &self,
        device_id: &str,
    ) -> Result<VulkanResidentBufferPoolDeviceRelease, VulkanError> {
        if device_id.is_empty() {
            return Err(VulkanError(
                "resident buffer pool release device id is empty".to_string(),
            ));
        }
        if !self
            .device_lifetime_guards
            .borrow()
            .contains_key(device_id)
        {
            return Err(VulkanError(format!(
                "resident buffer pool has no registered device {device_id:?}"
            )));
        }
        let mut state = self.state.borrow_mut();
        let keys = state
            .buffers
            .iter()
            .filter(|(key, _)| key.device_id == device_id)
            .map(|(key, buffer)| (key.clone(), Arc::as_ptr(buffer) as usize))
            .collect::<Vec<_>>();
        let mut pool_owners_by_buffer = BTreeMap::<usize, usize>::new();
        for (_, buffer_address) in &keys {
            *pool_owners_by_buffer.entry(*buffer_address).or_default() += 1;
        }
        if let Some((key, buffer_address)) = keys.iter().find(|(key, address)| {
            let buffer = state
                .buffers
                .get(key)
                .expect("enumerated resident buffer remains present");
            Arc::strong_count(buffer)
                != pool_owners_by_buffer
                    .get(address)
                    .copied()
                    .unwrap_or_default()
        }) {
            let buffer = state
                .buffers
                .get(key)
                .expect("enumerated resident buffer remains present");
            let owner_count = Arc::strong_count(buffer);
            let pool_owner_count = pool_owners_by_buffer
                .get(buffer_address)
                .copied()
                .unwrap_or_default();
            return Err(VulkanError(format!(
                "resident buffer pool cannot release device {device_id:?}: \
                 buffer {:?} has {owner_count} owners but only \
                 {pool_owner_count} belong to this device's pool",
                key.resource_id
            )));
        }
        let mut release = VulkanResidentBufferPoolDeviceRelease::default();
        for (key, _) in keys {
            let buffer = state
                .buffers
                .remove(&key)
                .expect("enumerated resident buffer remains present");
            release.resident_buffer_count =
                release.resident_buffer_count.saturating_add(1);
            release.resident_bytes = release
                .resident_bytes
                .saturating_add(buffer.byte_capacity());
            drop(buffer);
        }
        state.eviction_count = state
            .eviction_count
            .saturating_add(release.resident_buffer_count as u64);
        state.eviction_bytes = state
            .eviction_bytes
            .saturating_add(release.resident_bytes as u64);
        drop(state);
        self.device_lifetime_guards
            .borrow_mut()
            .remove(device_id)
            .expect("validated resident buffer pool device guard exists");
        Ok(release)
    }

    pub fn registered_device_count(&self) -> usize {
        self.device_lifetime_guards.borrow().len()
    }

    pub fn stats(&self) -> VulkanResidentBufferPoolStats {
        let state = self.state.borrow();
        VulkanResidentBufferPoolStats {
            resident_buffer_count: state.buffers.len(),
            resident_bytes: state
                .buffers
                .values()
                .map(|buffer| buffer.byte_capacity())
                .sum(),
            hit_count: state.hit_count,
            miss_count: state.miss_count,
            eviction_count: state.eviction_count,
            eviction_bytes: state.eviction_bytes,
        }
    }
}
