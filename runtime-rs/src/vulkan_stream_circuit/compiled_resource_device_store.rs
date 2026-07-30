#[derive(Debug)]
pub struct VulkanCompiledResourceDeviceStoreError(String);

impl VulkanCompiledResourceDeviceStoreError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for VulkanCompiledResourceDeviceStoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for VulkanCompiledResourceDeviceStoreError {}

struct VulkanCompiledResourceDeviceAddressState {
    transfer: VulkanResidentTransferStream,
    address_table: VulkanStableResourceAddressTable,
    publications:
        BTreeMap<String, Vec<VulkanStableResourceAddressPublication>>,
}

pub struct VulkanCompiledResourceDeviceStore {
    device_id: String,
    physical_device_id: String,
    logical_device_ids: Vec<String>,
    allowed_selector_ids: BTreeSet<String>,
    package_root: PathBuf,
    contract: Arc<CompiledResourceResidencyContract>,
    layout: Arc<VulkanCompiledResourceAddressLayout>,
    arena: VulkanStableResourceArena,
    address_state:
        std::sync::Mutex<VulkanCompiledResourceDeviceAddressState>,
    backing_store: CompiledResourceBackingStore,
    manager: DeviceResourceResidencyManager<VulkanResidentCompiledResource>,
    upload_alignment: usize,
    maximum_dynamic_payload_bytes: usize,
    maximum_allocation_byte_capacity: usize,
    always_resident_parameter_bytes: usize,
    runtime_working_set_device_bytes: usize,
    metadata_device_bytes: usize,
    transfer_staging_host_bytes: usize,
    coverage_index: Vec<VulkanCompiledResourceComponentCoverageIndex>,
    instrumentation: VulkanCompiledResourceStoreInstrumentation,
}

impl VulkanCompiledResourceDeviceStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &VulkanComputeDevice,
        device_id: impl Into<String>,
        physical_device_id: impl Into<String>,
        logical_device_ids: Vec<String>,
        package_root: impl Into<PathBuf>,
        contract: Arc<CompiledResourceResidencyContract>,
        layout: Arc<VulkanCompiledResourceAddressLayout>,
        allowed_selector_ids: BTreeSet<String>,
        maximum_dynamic_payload_bytes: usize,
        available_dynamic_device_bytes: usize,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
        always_resident_parameter_bytes: usize,
        runtime_working_set_device_bytes: usize,
        metadata_device_bytes: usize,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let device_id = device_id.into();
        let physical_device_id = physical_device_id.into();
        if device_id.trim().is_empty()
            || physical_device_id.trim().is_empty()
            || logical_device_ids.is_empty()
            || logical_device_ids
                .iter()
                .any(|logical_device_id| logical_device_id.trim().is_empty())
            || maximum_dynamic_payload_bytes == 0
            || available_dynamic_device_bytes == 0
            || maximum_group_byte_count == 0
            || maximum_ranges_per_group == 0
            || maximum_group_byte_count > maximum_dynamic_payload_bytes
            || layout.slot_count() == 0
            || allowed_selector_ids.is_empty()
            || metadata_device_bytes == 0
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled device-resource store has an invalid capacity or identity",
            ));
        }
        let upload_alignment =
            compiled_resource_upload_alignment(&contract, device)?;
        let maximum_group_resource_count =
            compiled_resource_maximum_resources_per_group_for_selectors(
                &contract,
                &allowed_selector_ids,
            )?;
        let maximum_addressable_resource_count = layout
            .selectors
            .iter()
            .filter(|selector| {
                allowed_selector_ids.contains(&selector.selector_id)
            })
            .flat_map(|selector| selector.resource_address_slots.iter())
            .flatten()
            .copied()
            .collect::<BTreeSet<_>>()
            .len();
        if maximum_addressable_resource_count == 0
            || allowed_selector_ids.iter().any(|selector_id| {
                !layout
                    .selectors
                    .iter()
                    .any(|selector| selector.selector_id == *selector_id)
            })
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled device-resource store selector ownership is invalid",
            ));
        }
        let per_resource_alignment_slack =
            upload_alignment.checked_sub(1).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource upload alignment underflowed",
                )
            })?;
        let maximum_allocation_byte_capacity = maximum_dynamic_payload_bytes
            .checked_add(
                maximum_addressable_resource_count
                    .checked_mul(per_resource_alignment_slack)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource allocation-padding capacity overflowed",
                        )
                    })?,
            )
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource allocation capacity overflowed",
                )
            })?;
        if maximum_allocation_byte_capacity > available_dynamic_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resources require up to {maximum_allocation_byte_capacity} physical allocation bytes for {maximum_dynamic_payload_bytes} payload bytes, but only {available_dynamic_device_bytes} device bytes are available"
            )));
        }
        let initial_chunk_byte_capacity = maximum_group_byte_count
            .checked_add(
                maximum_group_resource_count
                    .checked_mul(per_resource_alignment_slack)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource group alignment capacity overflowed",
                        )
                    })?,
            )
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource group allocation capacity overflowed",
                )
            })?;
        let address_table_byte_count = layout
            .slot_count()
            .checked_mul(32)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address-table capacity overflowed",
                )
            })?;
        let staging_byte_capacity =
            maximum_group_byte_count.max(address_table_byte_count);
        let mut transfer = device
            .create_resident_transfer_stream(2, staging_byte_capacity)
            .map_err(compiled_device_store_vulkan_error)?;
        let address_table = VulkanStableResourceAddressTable::new(
            device,
            &mut transfer,
            layout.slot_count(),
        )
        .map_err(compiled_device_store_vulkan_error)?;
        let arena = VulkanStableResourceArena::new(
            device,
            VulkanStableResourceArenaConfig::new(
                initial_chunk_byte_capacity,
                maximum_allocation_byte_capacity,
                upload_alignment,
            )
            .map_err(compiled_device_store_vulkan_error)?,
        )
        .map_err(compiled_device_store_vulkan_error)?;
        let retained_payload_bytes = maximum_group_byte_count
            .checked_mul(2)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource retained host-memory capacity overflowed",
                )
            })?;
        let package_root = package_root.into();
        let coverage_index = compiled_resource_component_coverage_index(
            &contract,
            &allowed_selector_ids,
        )?;
        let backing_store = CompiledResourceBackingStore::new(
            package_root.clone(),
            CompiledResourceBackingStoreLimits {
                worker_count: 2,
                queued_request_capacity: 2,
                maximum_ranges_per_group,
                maximum_logical_bytes_per_group: maximum_group_byte_count,
                maximum_retained_payload_bytes: retained_payload_bytes,
                maximum_coalesced_read_bytes: maximum_group_byte_count,
                maximum_coalescing_gap_bytes: 64 * 1024,
            },
        )
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to create compiled resource backing store: {error}"
            ))
        })?;
        let manager =
            DeviceResourceResidencyManager::new(
                device_id.clone(),
                maximum_dynamic_payload_bytes,
                0,
            )
            .map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "failed to create compiled resource residency manager: {error}"
                ))
            })?;
        Ok(Self {
            device_id,
            physical_device_id,
            logical_device_ids,
            allowed_selector_ids,
            package_root,
            contract,
            layout,
            arena,
            address_state: std::sync::Mutex::new(
                VulkanCompiledResourceDeviceAddressState {
                    transfer,
                    address_table,
                    publications: BTreeMap::new(),
                },
            ),
            backing_store,
            manager,
            upload_alignment,
            maximum_dynamic_payload_bytes,
            maximum_allocation_byte_capacity,
            always_resident_parameter_bytes,
            runtime_working_set_device_bytes,
            metadata_device_bytes,
            transfer_staging_host_bytes: staging_byte_capacity
                .checked_mul(2)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource transfer staging byte count overflowed",
                    )
                })?,
            coverage_index,
            instrumentation:
                VulkanCompiledResourceStoreInstrumentation::default(),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    pub fn allowed_selector_ids(&self) -> &BTreeSet<String> {
        &self.allowed_selector_ids
    }

    pub fn dynamic_buffers_for_components(
        &self,
        device: &VulkanComputeDevice,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
    ) -> Result<Arc<VulkanDynamicResourceBuffers>, VulkanCompiledResourceDeviceStoreError>
    {
        let state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        VulkanDynamicResourceBuffers::from_layout_for_components(
            device,
            &state.address_table,
            &self.layout,
            Some(execution_scope),
            component_ids,
        )
        .map(Arc::new)
        .map_err(compiled_device_store_vulkan_error)
    }

    pub fn load_selector_resource(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_index: usize,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let resolved = self.resolve_selector_resource(
            selector_id,
            resource_index,
        )?;
        let descriptor =
            DeviceResourceGroupDescriptor::from_resolved(&resolved).map_err(
                |error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource descriptor is invalid: {error}"
                    ))
                },
            )?;
        let resource_slots = self
            .layout
            .resource_slots_for_ids(&descriptor.resource_ids)
            .map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(error.to_string())
            })?;
        match self
            .manager
            .request(descriptor.clone(), owner)
            .map_err(compiled_device_store_residency_error)?
        {
            DeviceResourceResidencyRequest::Resident(lease) => {
                drop(lease);
                Ok(())
            }
            DeviceResourceResidencyRequest::Pending(waiter) => {
                let _blocking =
                    VulkanCompiledResourceBlockingTimer::new(
                        &self.instrumentation,
                    );
                waiter
                    .wait()
                    .map(drop)
                    .map_err(compiled_device_store_residency_error)
            }
            DeviceResourceResidencyRequest::LoadRequired(permit) => {
                let _blocking =
                    VulkanCompiledResourceBlockingTimer::new(
                        &self.instrumentation,
                    );
                let loaded = match self
                    .backing_store
                    .try_load(resolved)
                    .and_then(CompiledResourceLoadTicket::wait)
                {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        let residency_error = DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::Failed,
                            format!(
                                "compiled resource backing-store load failed: {error}"
                            ),
                        );
                        let _ = permit.fail(residency_error);
                        return Err(
                            VulkanCompiledResourceDeviceStoreError::new(
                                format!(
                                    "compiled resource backing-store load failed: {error}"
                                ),
                            ),
                        );
                    }
                };
                let mut state = self.address_state.lock().map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource address state was poisoned",
                    )
                })?;
                let VulkanCompiledResourceDeviceAddressState {
                    transfer,
                    address_table,
                    publications,
                } = &mut *state;
                if publications.contains_key(&descriptor.id) {
                    let residency_error = DeviceResourceResidencyError::new(
                        DeviceResourceResidencyErrorKind::IdentityConflict,
                        "compiled resource address publication exists without a resident directory entry",
                    );
                    let _ = permit.fail(residency_error);
                    return Err(VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource address publication exists without a resident directory entry",
                    ));
                }
                let upload_started = Instant::now();
                let upload = match upload_loaded_compiled_resource_group_to_stable_address_space(
                    device,
                    transfer,
                    &self.arena,
                    address_table,
                    &descriptor,
                    &loaded,
                    &resource_slots,
                    self.upload_alignment,
                ) {
                    Ok(upload) => upload,
                    Err(error) => {
                        let residency_error = DeviceResourceResidencyError::new(
                            DeviceResourceResidencyErrorKind::Failed,
                            format!("compiled resource upload failed: {error}"),
                        );
                        let _ = permit.fail(residency_error);
                        return Err(compiled_device_store_vulkan_error(error));
                    }
                };
                let (group, publications) = upload.into_parts();
                match permit.publish(group) {
                    Ok(lease) => {
                        state
                            .publications
                            .insert(descriptor.id.clone(), publications);
                        drop(lease);
                        let arena = self
                            .arena
                            .stats()
                            .map_err(compiled_device_store_vulkan_error)?;
                        self.instrumentation.record_upload(
                            descriptor.byte_count,
                            u64::try_from(
                                upload_started.elapsed().as_nanos(),
                            )
                            .unwrap_or(u64::MAX),
                            arena.committed_byte_capacity,
                        );
                        Ok(())
                    }
                    Err(error) => {
                        let VulkanCompiledResourceDeviceAddressState {
                            transfer,
                            address_table,
                            ..
                        } = &mut *state;
                        address_table
                            .clear_group(transfer, &publications)
                            .map_err(compiled_device_store_vulkan_error)?;
                        Err(compiled_device_store_residency_error(error))
                    }
                }
            }
        }
    }

    pub fn record_gpu_gate_misses(
        &self,
        selector_id: &str,
        miss_count: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let selector = self
            .contract
            .selectors
            .iter()
            .find(|selector| selector.id == selector_id)
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource GPU-miss report references unknown selector {selector_id:?}",
                ))
            })?;
        if !self.allowed_selector_ids.contains(selector_id) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource GPU-miss report references selector {selector_id:?} outside store {:?}",
                self.device_id,
            )));
        }
        self.instrumentation.record_gpu_gate_misses(
            &selector.execution_scope,
            &selector.component_id,
            miss_count,
        )
    }

    pub fn load_all_for_components(
        &self,
        device: &VulkanComputeDevice,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let selected = self
            .contract
            .selectors
            .iter()
            .filter(|selector| {
                selector.execution_scope == execution_scope
                    && component_ids.contains(&selector.component_id)
                    && self.allowed_selector_ids.contains(&selector.id)
            })
            .flat_map(|selector| {
                (0..selector.resource_count)
                    .map(move |resource_index| {
                        (selector.id.clone(), resource_index)
                    })
            })
            .collect::<Vec<_>>();
        let selected = self.unique_selector_resources(selected)?;
        for (selector_id, resource_index) in &selected {
            self.load_selector_resource(
                device,
                selector_id,
                *resource_index,
                owner.clone(),
            )?;
        }
        Ok(selected.len())
    }

    pub fn load_all_allowed(
        &self,
        device: &VulkanComputeDevice,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let selected = self
            .contract
            .selectors
            .iter()
            .filter(|selector| {
                self.allowed_selector_ids.contains(&selector.id)
            })
            .flat_map(|selector| {
                (0..selector.resource_count)
                    .map(move |resource_index| {
                        (selector.id.clone(), resource_index)
                    })
            })
            .collect::<Vec<_>>();
        let selected = self.unique_selector_resources(selected)?;
        for (selector_id, resource_index) in &selected {
            self.load_selector_resource(
                device,
                selector_id,
                *resource_index,
                owner.clone(),
            )?;
        }
        Ok(selected.len())
    }

    fn unique_selector_resources(
        &self,
        candidates: Vec<(String, usize)>,
    ) -> Result<Vec<(String, usize)>, VulkanCompiledResourceDeviceStoreError> {
        let mut selected_by_group = BTreeMap::new();
        for (selector_id, resource_index) in candidates {
            let group =
                self.resolve_selector_resource(&selector_id, resource_index)?;
            selected_by_group
                .entry(group.id().to_string())
                .or_insert((selector_id, resource_index));
        }
        Ok(selected_by_group.into_values().collect())
    }

    pub fn statistics(
        &self,
    ) -> Result<DeviceResourceResidencyStatistics, VulkanCompiledResourceDeviceStoreError>
    {
        self.manager
            .statistics()
            .map_err(compiled_device_store_residency_error)
    }

    pub fn mark_mount_complete(
        &self,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let residency = self.statistics()?;
        let arena = self
            .arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        self.instrumentation.mark_mount_complete(
            residency.dynamic_resident_bytes,
            residency.resident_group_count,
            arena.committed_byte_capacity,
        );
        Ok(())
    }

    pub fn residency_report(
        &self,
    ) -> Result<VulkanCompiledResourceStoreReport, VulkanCompiledResourceDeviceStoreError>
    {
        use std::sync::atomic::Ordering;

        let _address_state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let snapshot = self
            .manager
            .snapshot()
            .map_err(compiled_device_store_residency_error)?;
        let residency = snapshot.statistics;
        let directory = snapshot.directory;
        let resident_group_ids = directory
            .iter()
            .filter(|entry| entry.state == ResourceResidencyState::Resident)
            .map(|entry| entry.group_id.as_str())
            .collect::<BTreeSet<_>>();
        let arena = self
            .arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        self.instrumentation
            .high_water_committed_device_bytes
            .fetch_max(
                u64::try_from(arena.committed_byte_capacity)
                    .unwrap_or(u64::MAX),
                Ordering::AcqRel,
            );
        let backing = self.backing_store.statistics();
        let gpu_misses_by_component =
            self.instrumentation.gpu_misses_by_component()?;
        let mut components = self
            .coverage_index
            .iter()
            .map(|coverage| {
                VulkanCompiledResourceComponentCoverageReport {
                    execution_scope: coverage.execution_scope.clone(),
                    component_id: coverage.component_id.clone(),
                    addressable_unit_count: coverage.group_ids.len(),
                    resident_unit_count: coverage
                        .group_ids
                        .iter()
                        .filter(|group_id| {
                            resident_group_ids.contains(group_id.as_str())
                        })
                        .count(),
                    gpu_selection_count: 0,
                    gpu_resident_hit_count: 0,
                    gpu_miss_count: gpu_misses_by_component
                        .get(&(
                            coverage.execution_scope.clone(),
                            coverage.component_id.clone(),
                        ))
                        .copied()
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        components.sort_by(|left, right| {
            (
                left.execution_scope.as_str(),
                left.component_id.as_str(),
            )
                .cmp(&(
                    right.execution_scope.as_str(),
                    right.component_id.as_str(),
                ))
        });
        let mut scope_group_ids =
            BTreeMap::<String, BTreeSet<String>>::new();
        for coverage in &self.coverage_index {
            scope_group_ids
                .entry(coverage.execution_scope.clone())
                .or_default()
                .extend(coverage.group_ids.iter().cloned());
        }
        let scopes = scope_group_ids
            .into_iter()
            .map(|(execution_scope, group_ids)| {
                Ok(VulkanCompiledResourceScopeCoverageReport {
                    component_count: components
                        .iter()
                        .filter(|component| {
                            component.execution_scope == execution_scope
                        })
                        .count(),
                    addressable_unit_count: group_ids.len(),
                    resident_unit_count: group_ids
                        .iter()
                        .filter(|group_id| {
                            resident_group_ids.contains(group_id.as_str())
                        })
                        .count(),
                    gpu_selection_count: 0,
                    gpu_resident_hit_count: 0,
                    gpu_miss_count: components
                        .iter()
                        .filter(|component| {
                            component.execution_scope == execution_scope
                        })
                        .map(|component| component.gpu_miss_count)
                        .try_fold(0u64, |total, count| {
                            total.checked_add(count).ok_or_else(|| {
                                VulkanCompiledResourceDeviceStoreError::new(
                                    "compiled resource scope GPU-miss count overflowed",
                                )
                            })
                        })?,
                    execution_scope,
                })
            })
            .collect::<Result<Vec<_>, VulkanCompiledResourceDeviceStoreError>>()?;
        let addressable_unit_count = self
            .coverage_index
            .iter()
            .flat_map(|coverage| coverage.group_ids.iter())
            .collect::<BTreeSet<_>>()
            .len();
        let initial_committed_device_bytes = usize::try_from(
            self.instrumentation
                .initial_committed_device_bytes
                .load(Ordering::Acquire),
        )
        .unwrap_or(usize::MAX);
        let high_water_committed_device_bytes = usize::try_from(
            self.instrumentation
                .high_water_committed_device_bytes
                .load(Ordering::Acquire),
        )
        .unwrap_or(usize::MAX);
        let fixed_device_bytes = self
            .always_resident_parameter_bytes
            .checked_add(self.runtime_working_set_device_bytes)
            .and_then(|bytes| {
                bytes.checked_add(self.metadata_device_bytes)
            })
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource fixed device byte report overflowed",
                )
            })?;
        let total_device_bytes =
            |dynamic_bytes: usize, label: &str| {
                fixed_device_bytes
                    .checked_add(dynamic_bytes)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            format!(
                                "compiled resource {label} device byte report overflowed"
                            ),
                        )
                    })
            };
        Ok(VulkanCompiledResourceStoreReport {
            store_id: self.device_id.clone(),
            physical_device_id: self.physical_device_id.clone(),
            logical_device_ids: self.logical_device_ids.clone(),
            initial_device_bytes: total_device_bytes(
                initial_committed_device_bytes,
                "initial",
            )?,
            current_device_bytes: total_device_bytes(
                arena.committed_byte_capacity,
                "current",
            )?,
            maximum_device_bytes: total_device_bytes(
                self.maximum_allocation_byte_capacity,
                "maximum",
            )?,
            high_water_device_bytes: total_device_bytes(
                high_water_committed_device_bytes,
                "high-water",
            )?,
            always_resident_parameter_bytes:
                self.always_resident_parameter_bytes,
            runtime_working_set_device_bytes:
                self.runtime_working_set_device_bytes,
            metadata_device_bytes: self.metadata_device_bytes,
            transfer_staging_host_bytes:
                self.transfer_staging_host_bytes,
            initial_payload_bytes: usize::try_from(
                self.instrumentation
                    .initial_payload_bytes
                    .load(Ordering::Acquire),
            )
            .unwrap_or(usize::MAX),
            current_payload_bytes: residency.dynamic_resident_bytes,
            maximum_payload_bytes: self.maximum_dynamic_payload_bytes,
            high_water_payload_bytes:
                residency.high_water_dynamic_resident_bytes,
            addressable_unit_count,
            initial_resident_unit_count: usize::try_from(
                self.instrumentation
                    .initial_resident_unit_count
                    .load(Ordering::Acquire),
            )
            .unwrap_or(usize::MAX),
            resident_unit_count: residency.resident_group_count,
            high_water_resident_unit_count:
                residency.high_water_resident_group_count,
            loading_unit_count: residency.loading_group_count,
            failed_unit_count: residency.failed_group_count,
            gpu_selection_count: 0,
            gpu_resident_hit_count: 0,
            gpu_miss_count: gpu_misses_by_component
                .values()
                .try_fold(0u64, |total, count| {
                    total.checked_add(*count).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource store GPU-miss count overflowed",
                        )
                    })
                })?,
            residency_directory_hit_count: residency.hit_count,
            residency_load_required_count: residency.miss_count,
            deduplicated_load_count:
                residency.single_flight_join_count,
            successful_load_count: residency.successful_load_count,
            failed_load_count: residency.failed_load_count,
            cancelled_load_count: residency.cancelled_load_count,
            logical_read_count: backing.logical_ranges,
            physical_read_count: backing.physical_reads,
            logical_bytes_read: backing.logical_bytes,
            physical_bytes_read: backing.physical_bytes,
            uploaded_bytes: self
                .instrumentation
                .uploaded_bytes
                .load(Ordering::Relaxed),
            read_time_ns: backing.read_time_ns,
            upload_time_ns: self
                .instrumentation
                .upload_time_ns
                .load(Ordering::Relaxed),
            blocking_time_ns: self
                .instrumentation
                .blocking_time_ns
                .load(Ordering::Relaxed),
            scopes,
            components,
        })
    }

    pub fn unload(
        &self,
    ) -> Result<DeviceResourceResidencyRelease, VulkanCompiledResourceDeviceStoreError>
    {
        let release = self
            .manager
            .unload_device()
            .map_err(compiled_device_store_residency_error)?;
        let mut state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let publications = std::mem::take(&mut state.publications)
            .into_values()
            .flatten()
            .collect::<Vec<_>>();
        if !publications.is_empty() {
            let VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                ..
            } = &mut *state;
            address_table
                .clear_group(transfer, &publications)
                .map_err(compiled_device_store_vulkan_error)?;
        }
        Ok(release)
    }

    fn resolve_selector_resource(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Result<ResolvedCompiledResourceGroup, VulkanCompiledResourceDeviceStoreError>
    {
        let selector = self
            .contract
            .selectors
            .iter()
            .find(|selector| {
                selector.id == selector_id
                    && self.allowed_selector_ids.contains(&selector.id)
            })
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource selector {selector_id:?} is unknown"
                ))
            })?;
        if resource_index >= selector.resource_count {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource selector {selector_id:?} index {resource_index} exceeds {}",
                selector.resource_count
            )));
        }
        match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids,
            } => resolve_compiled_atomic_group(
                &self.contract,
                &atomic_group_ids[resource_index],
            )
            .map(ResolvedCompiledResourceGroup::Atomic),
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => resolve_compiled_partition_group(
                &self.package_root,
                &self.contract,
                partition_template_id,
                resource_index,
            )
            .map(ResolvedCompiledResourceGroup::Partition),
        }
        .map_err(|error| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "failed to resolve compiled resource selection: {error}"
            ))
        })
    }
}

fn compiled_resource_upload_alignment(
    contract: &CompiledResourceResidencyContract,
    device: &VulkanComputeDevice,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    let mut alignment = device
        .min_storage_buffer_offset_alignment()
        .max(std::mem::align_of::<u64>());
    for range_alignment in contract
        .resources
        .iter()
        .flat_map(|resource| {
            resource.ranges.iter().map(|range| range.alignment_bytes)
        })
        .chain(
            contract
                .partition_templates
                .iter()
                .flat_map(|template| &template.member_templates)
                .flat_map(|member| {
                    member
                        .range_templates
                        .iter()
                        .map(|range| range.alignment_bytes)
                }),
        )
    {
        alignment = alignment.max(range_alignment);
    }
    if !alignment.is_power_of_two() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource upload alignment {alignment} is not a power of two"
        )));
    }
    Ok(alignment)
}

fn compiled_resource_maximum_ranges_per_group(
    contract: &CompiledResourceResidencyContract,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    contract
        .atomic_groups
        .iter()
        .filter(|group| group.lifetime == CompiledResourceLifetime::Dynamic)
        .map(|group| {
            group.resource_ids.iter().try_fold(0usize, |total, resource_id| {
                let resource = contract
                    .resources
                    .iter()
                    .find(|resource| resource.id == *resource_id)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "dynamic group {:?} references missing resource {resource_id:?}",
                            group.id
                        ))
                    })?;
                total.checked_add(resource.ranges.len()).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource range count overflowed",
                    )
                })
            })
        })
        .chain(contract.partition_templates.iter().map(|template| {
            template
                .member_templates
                .iter()
                .try_fold(0usize, |total, member| {
                    total
                        .checked_add(member.range_templates.len())
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                "compiled partition range count overflowed",
                            )
                        })
                })
        }))
        .try_fold(0usize, |maximum, count| count.map(|count| maximum.max(count)))
}

fn compiled_resource_maximum_resources_per_group_for_selectors(
    contract: &CompiledResourceResidencyContract,
    selector_ids: &BTreeSet<String>,
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    contract
        .selectors
        .iter()
        .filter(|selector| selector_ids.contains(&selector.id))
        .map(|selector| match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids,
            } => atomic_group_ids
                .iter()
                .map(|group_id| {
                    contract
                        .atomic_groups
                        .iter()
                        .find(|group| group.id == *group_id)
                        .map(|group| group.resource_ids.len())
                        .ok_or_else(|| {
                            VulkanCompiledResourceDeviceStoreError::new(format!(
                                "compiled selector {:?} references missing group {group_id:?}",
                                selector.id
                            ))
                        })
                })
                .try_fold(0usize, |maximum, count| {
                    count.map(|count| maximum.max(count))
                }),
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => contract
                .partition_templates
                .iter()
                .find(|template| template.id == *partition_template_id)
                .map(|template| template.member_templates.len())
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled selector {:?} references missing partition template {partition_template_id:?}",
                        selector.id
                    ))
                }),
        })
        .try_fold(0usize, |maximum, count| {
            count.map(|count| maximum.max(count))
        })
        .and_then(|maximum| {
            if maximum == 0 {
                Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled device-resource store has no addressable group members",
                ))
            } else {
                Ok(maximum)
            }
        })
}

fn compiled_resource_component_coverage_index(
    contract: &CompiledResourceResidencyContract,
    selector_ids: &BTreeSet<String>,
) -> Result<
    Vec<VulkanCompiledResourceComponentCoverageIndex>,
    VulkanCompiledResourceDeviceStoreError,
> {
    let mut indexed =
        BTreeMap::<(String, String), BTreeSet<String>>::new();
    for selector in contract
        .selectors
        .iter()
        .filter(|selector| selector_ids.contains(&selector.id))
    {
        let group_ids = match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable {
                atomic_group_ids,
            } => atomic_group_ids.iter().cloned().collect::<Vec<_>>(),
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => {
                let template = contract
                    .partition_templates
                    .iter()
                    .find(|template| template.id == *partition_template_id)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled selector {:?} references missing partition template {partition_template_id:?}",
                            selector.id
                        ))
                    })?;
                (0..template.partition_count)
                    .map(|partition_index| {
                        derived_partition_resource_id(
                            &template.group_identity_seed,
                            partition_index,
                        )
                        .map_err(|error| {
                            VulkanCompiledResourceDeviceStoreError::new(
                                format!(
                                    "failed to derive addressable group identity: {error}"
                                ),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        indexed
            .entry((
                selector.execution_scope.clone(),
                selector.component_id.clone(),
            ))
            .or_default()
            .extend(group_ids);
    }
    Ok(indexed
        .into_iter()
        .map(
            |((execution_scope, component_id), group_ids)| {
                VulkanCompiledResourceComponentCoverageIndex {
                    execution_scope,
                    component_id,
                    group_ids,
                }
            },
        )
        .collect())
}

fn compiled_device_store_vulkan_error(
    error: VulkanError,
) -> VulkanCompiledResourceDeviceStoreError {
    VulkanCompiledResourceDeviceStoreError::new(error.to_string())
}

fn compiled_device_store_residency_error(
    error: DeviceResourceResidencyError,
) -> VulkanCompiledResourceDeviceStoreError {
    VulkanCompiledResourceDeviceStoreError::new(error.to_string())
}
