#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct VulkanCompiledResourceSharedHostCacheSnapshot {
    capacity_bytes: usize,
    committed_bytes: usize,
    committed_bytes_by_store: BTreeMap<String, usize>,
}

struct VulkanCompiledResourceSharedHostCacheStoreState {
    committed_bytes: usize,
}

#[derive(Default)]
struct VulkanCompiledResourceSharedHostCacheState {
    stores: BTreeMap<String, VulkanCompiledResourceSharedHostCacheStoreState>,
    committed_bytes: usize,
}

struct VulkanCompiledResourceSharedHostCache {
    cache_id: String,
    capacity_bytes: usize,
    // This lock is always acquired before an individual store's residency
    // mutation lock. Reservations may reclaim a different store, so the
    // ordering prevents cross-store A -> B / B -> A deadlocks.
    mutation: std::sync::Mutex<()>,
    state: std::sync::Mutex<VulkanCompiledResourceSharedHostCacheState>,
}

struct VulkanCompiledResourceSharedHostCacheMutation<'a> {
    cache: &'a Arc<VulkanCompiledResourceSharedHostCache>,
    _guard: std::sync::MutexGuard<'a, ()>,
}

struct VulkanCompiledResourceSharedHostCacheReservation {
    cache: Arc<VulkanCompiledResourceSharedHostCache>,
    store_id: String,
    reserved_bytes: usize,
    settled: bool,
}

impl VulkanCompiledResourceSharedHostCache {
    fn new(
        cache_id: impl Into<String>,
        capacity_bytes: usize,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let cache_id = cache_id.into();
        if cache_id.trim().is_empty() || capacity_bytes == 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache has an invalid identity or capacity",
            ));
        }
        Ok(Self {
            cache_id,
            capacity_bytes,
            mutation: std::sync::Mutex::new(()),
            state: std::sync::Mutex::new(
                VulkanCompiledResourceSharedHostCacheState::default(),
            ),
        })
    }

    fn cache_id(&self) -> &str {
        &self.cache_id
    }

    fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    fn begin_mutation(
        self: &Arc<Self>,
    ) -> Result<
        VulkanCompiledResourceSharedHostCacheMutation<'_>,
        VulkanCompiledResourceDeviceStoreError,
    > {
        let guard = self.mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache mutation lock was poisoned",
            )
        })?;
        Ok(VulkanCompiledResourceSharedHostCacheMutation {
            cache: self,
            _guard: guard,
        })
    }

    fn register_store(
        &self,
        store_id: &str,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if store_id.trim().is_empty() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache cannot register an empty store identity",
            ));
        }
        let mut state = self.state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache state was poisoned",
            )
        })?;
        if state.stores.contains_key(store_id) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host cache store {store_id:?} was registered twice"
            )));
        }
        state.stores.insert(
            store_id.to_string(),
            VulkanCompiledResourceSharedHostCacheStoreState {
                committed_bytes: 0,
            },
        );
        Ok(())
    }

    fn reserve_capacity_uncoordinated(
        self: &Arc<Self>,
        store_id: &str,
        requested_bytes: usize,
    ) -> Result<VulkanCompiledResourceSharedHostCacheReservation, VulkanCompiledResourceDeviceStoreError>
    {
        if requested_bytes == 0 {
            return Ok(VulkanCompiledResourceSharedHostCacheReservation {
                cache: Arc::clone(self),
                store_id: store_id.to_string(),
                reserved_bytes: 0,
                settled: false,
            });
        }
        if requested_bytes > self.capacity_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host cache cannot reserve {requested_bytes} bytes from its {}-byte hard bound",
                self.capacity_bytes,
            )));
        }
        let mut state = self.state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache state was poisoned",
            )
        })?;
        if !state.stores.contains_key(store_id) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host cache has no registered store {store_id:?}"
            )));
        }
        let required_end = state
            .committed_bytes
            .checked_add(requested_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "shared compiled-resource host cache capacity overflowed",
                )
            })?;
        if required_end > self.capacity_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host cache cannot reserve {requested_bytes} bytes for {store_id:?}: {}/{} bytes are already committed; active admission cannot destroy another store's Vulkan backing",
                state.committed_bytes, self.capacity_bytes,
            )));
        }
        state.committed_bytes = required_end;
        let store = state
            .stores
            .get_mut(store_id)
            .expect("requesting shared host store was validated");
        store.committed_bytes = store
            .committed_bytes
            .checked_add(requested_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "shared compiled-resource host store capacity overflowed",
                )
            })?;
        drop(state);
        Ok(VulkanCompiledResourceSharedHostCacheReservation {
            cache: Arc::clone(self),
            store_id: store_id.to_string(),
            reserved_bytes: requested_bytes,
            settled: false,
        })
    }

    fn release_store_capacity_uncoordinated(
        &self,
        store_id: &str,
        released_bytes: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if released_bytes == 0 {
            return Ok(());
        }
        let mut state = self.state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache state was poisoned",
            )
        })?;
        let store = state.stores.get_mut(store_id).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host cache has no registered store {store_id:?}"
            ))
        })?;
        store.committed_bytes = store
            .committed_bytes
            .checked_sub(released_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "shared compiled-resource host store {store_id:?} released {released_bytes} bytes beyond its committed capacity"
                ))
            })?;
        state.committed_bytes = state
            .committed_bytes
            .checked_sub(released_bytes)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "shared compiled-resource host cache capacity underflowed",
                )
            })?;
        Ok(())
    }

    fn snapshot(
        &self,
    ) -> Result<VulkanCompiledResourceSharedHostCacheSnapshot, VulkanCompiledResourceDeviceStoreError>
    {
        let state = self.state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "shared compiled-resource host cache state was poisoned",
            )
        })?;
        Ok(VulkanCompiledResourceSharedHostCacheSnapshot {
            capacity_bytes: self.capacity_bytes,
            committed_bytes: state.committed_bytes,
            committed_bytes_by_store: state
                .stores
                .iter()
                .map(|(store_id, store)| (store_id.clone(), store.committed_bytes))
                .collect(),
        })
    }
}

impl VulkanCompiledResourceSharedHostCacheMutation<'_> {
    fn reserve_capacity(
        &self,
        store_id: &str,
        requested_bytes: usize,
    ) -> Result<
        VulkanCompiledResourceSharedHostCacheReservation,
        VulkanCompiledResourceDeviceStoreError,
    > {
        self.cache
            .reserve_capacity_uncoordinated(store_id, requested_bytes)
    }

    fn release_store_capacity(
        &self,
        store_id: &str,
        released_bytes: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        self.cache
            .release_store_capacity_uncoordinated(store_id, released_bytes)
    }
}

impl VulkanCompiledResourceSharedHostCacheReservation {
    fn settle(
        mut self,
        committed_bytes: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if committed_bytes > self.reserved_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "shared compiled-resource host allocation committed {committed_bytes} bytes after reserving only {}",
                self.reserved_bytes,
            )));
        }
        self.cache.release_store_capacity_uncoordinated(
            &self.store_id,
            self.reserved_bytes - committed_bytes,
        )?;
        self.settled = true;
        Ok(())
    }
}

impl Drop for VulkanCompiledResourceSharedHostCacheReservation {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self
                .cache
                .release_store_capacity_uncoordinated(&self.store_id, self.reserved_bytes);
        }
    }
}
