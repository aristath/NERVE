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
}

impl VulkanCompiledResourceDeviceStore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &VulkanComputeDevice,
        device_id: impl Into<String>,
        package_root: impl Into<PathBuf>,
        contract: Arc<CompiledResourceResidencyContract>,
        layout: Arc<VulkanCompiledResourceAddressLayout>,
        allowed_selector_ids: BTreeSet<String>,
        maximum_dynamic_payload_bytes: usize,
        available_dynamic_device_bytes: usize,
        maximum_group_byte_count: usize,
        maximum_ranges_per_group: usize,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        let device_id = device_id.into();
        if device_id.trim().is_empty()
            || maximum_dynamic_payload_bytes == 0
            || available_dynamic_device_bytes == 0
            || maximum_group_byte_count == 0
            || maximum_ranges_per_group == 0
            || maximum_group_byte_count > maximum_dynamic_payload_bytes
            || layout.slot_count() == 0
            || allowed_selector_ids.is_empty()
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
                waiter
                    .wait()
                    .map(drop)
                    .map_err(compiled_device_store_residency_error)
            }
            DeviceResourceResidencyRequest::LoadRequired(permit) => {
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
