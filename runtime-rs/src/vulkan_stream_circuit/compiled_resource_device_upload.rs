#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanResidentCompiledResourceRange {
    pub byte_offset: usize,
    pub byte_count: usize,
}

pub struct VulkanResidentCompiledResource {
    storage: VulkanResidentCompiledResourceStorage,
    ranges: Vec<VulkanResidentCompiledResourceRange>,
    byte_count: usize,
}

enum VulkanResidentCompiledResourceStorage {
    Direct(Arc<VulkanResidentBuffer>),
    Stable(Arc<VulkanStableResourceAllocation>),
}

impl VulkanResidentCompiledResource {
    pub fn buffer(&self) -> &VulkanResidentBuffer {
        match &self.storage {
            VulkanResidentCompiledResourceStorage::Direct(buffer) => buffer,
            VulkanResidentCompiledResourceStorage::Stable(allocation) => {
                allocation.buffer()
            }
        }
    }

    pub fn ranges(&self) -> &[VulkanResidentCompiledResourceRange] {
        &self.ranges
    }

    pub fn stable_device_address(&self) -> Option<u64> {
        match &self.storage {
            VulkanResidentCompiledResourceStorage::Direct(_) => None,
            VulkanResidentCompiledResourceStorage::Stable(allocation) => {
                Some(allocation.device_address())
            }
        }
    }

    fn stable_allocation(
        &self,
    ) -> Option<&Arc<VulkanStableResourceAllocation>> {
        match &self.storage {
            VulkanResidentCompiledResourceStorage::Direct(_) => None,
            VulkanResidentCompiledResourceStorage::Stable(allocation) => {
                Some(allocation)
            }
        }
    }

    fn stable_chunk_id(&self) -> Option<u64> {
        self.stable_allocation()
            .map(|allocation| allocation.chunk_id())
    }
}

impl DeviceResidentResourcePayload for VulkanResidentCompiledResource {
    fn byte_count(&self) -> usize {
        self.byte_count
    }
}

fn exchange_stable_compiled_resource_group_allocations(
    left: &DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    right: &DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
) -> Result<
    (
        DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
        DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    ),
    DeviceResourceResidencyError,
> {
    if left.resources().len() != right.resources().len() {
        return Err(DeviceResourceResidencyError::load_failed(
            "stable compiled resource groups have incompatible resource counts",
        ));
    }
    let rebuild = |
        logical: &DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
        storage: &DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    | {
        let resources = logical
            .resources()
            .iter()
            .zip(storage.resources())
            .map(|(logical_resource, storage_resource)| {
                let allocation = storage_resource
                    .payload()
                    .stable_allocation()
                    .ok_or_else(|| {
                        DeviceResourceResidencyError::load_failed(
                            "stable compiled resource retiering encountered direct storage",
                        )
                    })?;
                DeviceResidentResource::new(
                    logical_resource.descriptor().clone(),
                    VulkanResidentCompiledResource {
                        storage: VulkanResidentCompiledResourceStorage::Stable(
                            Arc::clone(allocation),
                        ),
                        ranges: logical_resource.payload().ranges.clone(),
                        byte_count: logical_resource.payload().byte_count,
                    },
                )
            })
            .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()?;
        DeviceResidentResourceGroup::new(logical.descriptor().clone(), resources)
    };
    Ok((rebuild(left, right)?, rebuild(right, left)?))
}

pub fn upload_loaded_compiled_resource_group(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    descriptor: &DeviceResourceGroupDescriptor,
    loaded: &LoadedCompiledResourceGroup,
) -> Result<
    DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    VulkanError,
> {
    let loaded_descriptor =
        device_resource_descriptor_from_loaded_group(loaded).map_err(|error| {
            VulkanError(format!(
                "loaded compiled resource group is invalid: {error}"
            ))
        })?;
    if loaded_descriptor != *descriptor {
        return Err(VulkanError(
            "loaded compiled resources do not match the reserved device group"
                .to_string(),
        ));
    }

    let mut allocations = Vec::with_capacity(loaded.resources.len());
    for (resource, resource_descriptor) in
        loaded.resources.iter().zip(&descriptor.resources)
    {
        let buffer = Arc::new(
            device.create_resident_buffer(resource_descriptor.byte_count)?,
        );
        let mut byte_offset = 0usize;
        let mut ranges = Vec::with_capacity(resource.ranges.len());
        for range in &resource.ranges {
            ranges.push(VulkanResidentCompiledResourceRange {
                byte_offset,
                byte_count: range.bytes.len(),
            });
            byte_offset =
                byte_offset.checked_add(range.bytes.len()).ok_or_else(|| {
                    VulkanError(
                        "compiled resource upload byte offset overflowed"
                            .to_string(),
                    )
                })?;
        }
        if byte_offset != resource_descriptor.byte_count {
            return Err(VulkanError(
                "compiled resource upload size does not match its descriptor"
                    .to_string(),
            ));
        }
        allocations.push((buffer, ranges));
    }

    let mut writes = Vec::new();
    for ((resource, _), (buffer, ranges)) in loaded
        .resources
        .iter()
        .zip(&descriptor.resources)
        .zip(&allocations)
    {
        for (range, placement) in resource.ranges.iter().zip(ranges) {
            writes.push(VulkanResidentBufferWriteRange::new(
                buffer,
                placement.byte_offset,
                &range.bytes,
            )?);
        }
    }
    let ticket = transfer.submit(&writes)?;
    transfer.wait(&ticket)?;

    let resources = descriptor
        .resources
        .iter()
        .cloned()
        .zip(allocations)
        .map(|(resource_descriptor, (buffer, ranges))| {
            let byte_count = resource_descriptor.byte_count;
            DeviceResidentResource::new(
                resource_descriptor,
                VulkanResidentCompiledResource {
                    storage: VulkanResidentCompiledResourceStorage::Direct(
                        buffer,
                    ),
                    ranges,
                    byte_count,
                },
            )
        })
        .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()
        .map_err(|error| {
            VulkanError(format!(
                "uploaded compiled resource payload is invalid: {error}"
            ))
        })?;
    DeviceResidentResourceGroup::new(descriptor.clone(), resources).map_err(
        |error| {
            VulkanError(format!(
                "uploaded compiled resource group cannot be published: {error}"
            ))
        },
    )
}

pub struct VulkanStableCompiledResourceUpload {
    resident_group:
        DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    publications: Vec<VulkanStableResourceAddressPublication>,
}

pub struct VulkanStableCompiledResourceUploadRequest<'a> {
    pub descriptor: &'a DeviceResourceGroupDescriptor,
    pub loaded: &'a LoadedCompiledResourceGroup,
    pub resource_slots: &'a [usize],
}

struct VulkanPreparedStableCompiledResourceUpload {
    resident_group: DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    address_resources: Vec<(usize, Arc<VulkanStableResourceAllocation>, u32)>,
}

impl VulkanStableCompiledResourceUpload {
    pub fn resident_group(
        &self,
    ) -> &DeviceResidentResourceGroup<VulkanResidentCompiledResource> {
        &self.resident_group
    }

    pub fn publications(
        &self,
    ) -> &[VulkanStableResourceAddressPublication] {
        &self.publications
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
        Vec<VulkanStableResourceAddressPublication>,
    ) {
        (self.resident_group, self.publications)
    }

    pub fn retire(
        self,
        transfer: &mut VulkanResidentTransferStream,
        address_table: &mut VulkanStableResourceAddressTable,
    ) -> Result<(), VulkanError> {
        address_table.clear_group(transfer, &self.publications)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn upload_loaded_compiled_resource_group_to_stable_address_space(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    arena: &VulkanStableResourceArena,
    address_table: &mut VulkanStableResourceAddressTable,
    descriptor: &DeviceResourceGroupDescriptor,
    loaded: &LoadedCompiledResourceGroup,
    resource_slots: &[usize],
    alignment: usize,
) -> Result<VulkanStableCompiledResourceUpload, VulkanError> {
    let mut uploads =
        upload_loaded_compiled_resource_groups_to_stable_address_space(
            device,
            transfer,
            arena,
            address_table,
            &[VulkanStableCompiledResourceUploadRequest {
                descriptor,
                loaded,
                resource_slots,
            }],
            alignment,
            None,
        )?;
    Ok(uploads
        .pop()
        .expect("one stable upload request returns one result"))
}

pub fn upload_loaded_compiled_resource_groups_to_stable_address_space(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    arena: &VulkanStableResourceArena,
    address_table: &mut VulkanStableResourceAddressTable,
    requests: &[VulkanStableCompiledResourceUploadRequest<'_>],
    alignment: usize,
    capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
) -> Result<Vec<VulkanStableCompiledResourceUpload>, VulkanError> {
    let prepared = prepare_loaded_compiled_resource_groups_for_stable_address_space(
        device,
        transfer,
        arena,
        requests,
        address_table.slot_count(),
        alignment,
        capacity_permit,
    )?;
    let expected_publication_count = prepared.iter().try_fold(
        0usize,
        |total, upload| {
            total.checked_add(upload.address_resources.len()).ok_or_else(|| {
                VulkanError("stable upload publication count overflowed".to_string())
            })
        },
    )?;
    let address_resources = prepared
        .iter()
        .flat_map(|upload| upload.address_resources.iter().cloned())
        .collect::<Vec<_>>();
    let publications =
        address_table.publish_tagged_group(transfer, &address_resources)?;
    if publications.len() != expected_publication_count {
        if !publications.is_empty() {
            address_table.clear_group(transfer, &publications)?;
        }
        return Err(VulkanError(
            "stable upload address publication result is inconsistent".to_string(),
        ));
    }
    let mut remaining_publications = publications.as_slice();
    let uploads = prepared
        .into_iter()
        .map(|prepared| {
            let (group_publications, following_publications) = remaining_publications
                .split_at(prepared.address_resources.len());
            remaining_publications = following_publications;
            VulkanStableCompiledResourceUpload {
                resident_group: prepared.resident_group,
                publications: group_publications.to_vec(),
            }
        })
        .collect::<Vec<_>>();
    debug_assert!(remaining_publications.is_empty());
    Ok(uploads)
}

#[allow(clippy::too_many_arguments)]
pub fn replace_loaded_compiled_resource_group_in_stable_address_space(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    arena: &VulkanStableResourceArena,
    address_table: &mut VulkanStableResourceAddressTable,
    current_publications: &[VulkanStableResourceAddressPublication],
    descriptor: &DeviceResourceGroupDescriptor,
    loaded: &LoadedCompiledResourceGroup,
    resource_slots: &[usize],
    alignment: usize,
    capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
) -> Result<VulkanStableCompiledResourceUpload, VulkanError> {
    let request = VulkanStableCompiledResourceUploadRequest {
        descriptor,
        loaded,
        resource_slots,
    };
    let mut prepared = prepare_loaded_compiled_resource_groups_for_stable_address_space(
        device,
        transfer,
        arena,
        &[request],
        address_table.slot_count(),
        alignment,
        capacity_permit,
    )?;
    let prepared = prepared
        .pop()
        .expect("one stable replacement request returns one prepared group");
    let publications = address_table.replace_group(
        transfer,
        current_publications,
        &prepared.address_resources,
    )?;
    Ok(VulkanStableCompiledResourceUpload {
        resident_group: prepared.resident_group,
        publications,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_loaded_compiled_resource_groups_for_stable_address_space(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    arena: &VulkanStableResourceArena,
    requests: &[VulkanStableCompiledResourceUploadRequest<'_>],
    address_slot_count: usize,
    alignment: usize,
    capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
) -> Result<Vec<VulkanPreparedStableCompiledResourceUpload>, VulkanError> {
    if requests.is_empty() {
        return Err(VulkanError(
            "stable resource upload batch must not be empty".to_string(),
        ));
    }
    let mut batch_slots = BTreeSet::new();
    for request in requests {
        let loaded_descriptor =
            device_resource_descriptor_from_loaded_group(request.loaded)
                .map_err(|error| {
                    VulkanError(format!(
                        "loaded compiled resource group is invalid: {error}"
                    ))
                })?;
        if loaded_descriptor != *request.descriptor {
            return Err(VulkanError(
                "loaded compiled resources do not match the reserved device group"
                    .to_string(),
            ));
        }
        if request.resource_slots.len()
            != request.descriptor.resources.len()
        {
            return Err(VulkanError(format!(
                "stable resource upload has {} address-table slots for {} resources",
                request.resource_slots.len(),
                request.descriptor.resources.len()
            )));
        }
        for slot in request.resource_slots {
            if !batch_slots.insert(*slot) {
                return Err(VulkanError(format!(
                    "stable resource upload batch repeats address-table slot {slot}"
                )));
            }
            if *slot >= address_slot_count {
                return Err(VulkanError(format!(
                    "stable resource upload slot {slot} exceeds address-table capacity {}",
                    address_slot_count
                )));
            }
        }

    }
    let allocation_byte_counts = requests
        .iter()
        .map(|request| {
            request
                .descriptor
                .resources
                .iter()
                .map(|resource| resource.byte_count)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let allocation_requests = requests
        .iter()
        .zip(&allocation_byte_counts)
        .map(|(request, byte_counts)| {
            (request.resource_slots, byte_counts.as_slice())
        })
        .collect::<Vec<_>>();
    let allocation_groups = match capacity_permit {
        Some(permit) => arena.allocate_groups_with_capacity_permit(
            device,
            &allocation_requests,
            alignment,
            permit,
        )?,
        None => arena.allocate_groups(device, &allocation_requests, alignment)?,
    };
    let mut prepared_groups = Vec::with_capacity(requests.len());
    for ((request, group_allocations), byte_counts) in requests
        .iter()
        .zip(allocation_groups)
        .zip(&allocation_byte_counts)
    {
        let mut allocations =
            Vec::with_capacity(request.loaded.resources.len());
        for ((resource, resource_descriptor), allocation) in request
            .loaded
            .resources
            .iter()
            .zip(&request.descriptor.resources)
            .zip(group_allocations)
        {
            if allocation.byte_count() != resource_descriptor.byte_count {
                return Err(VulkanError(
                    "stable resource allocation differs from its descriptor"
                        .to_string(),
                ));
            }
            let mut byte_offset = allocation.buffer_byte_offset();
            let mut ranges = Vec::with_capacity(resource.ranges.len());
            for range in &resource.ranges {
                ranges.push(VulkanResidentCompiledResourceRange {
                    byte_offset,
                    byte_count: range.bytes.len(),
                });
                byte_offset = byte_offset
                    .checked_add(range.bytes.len())
                    .ok_or_else(|| {
                        VulkanError(
                            "stable compiled resource upload byte offset overflowed"
                                .to_string(),
                        )
                    })?;
            }
            let expected_end = allocation
                .buffer_byte_offset()
                .checked_add(resource_descriptor.byte_count)
                .ok_or_else(|| {
                    VulkanError(
                        "stable compiled resource allocation end overflowed"
                            .to_string(),
                    )
                })?;
            if byte_offset != expected_end {
                return Err(VulkanError(
                    "stable compiled resource upload size does not match its descriptor"
                        .to_string(),
                ));
            }
            allocations.push((allocation, ranges));
        }
        debug_assert_eq!(allocations.len(), byte_counts.len());
        prepared_groups.push(allocations);
    }

    let mut writes = Vec::new();
    for (request, allocations) in requests.iter().zip(&prepared_groups) {
        for ((resource, _), (allocation, ranges)) in request
            .loaded
            .resources
            .iter()
            .zip(&request.descriptor.resources)
            .zip(allocations)
        {
            for (range, placement) in resource.ranges.iter().zip(ranges) {
                writes.push(VulkanResidentBufferWriteRange::new(
                    allocation.buffer(),
                    placement.byte_offset,
                    &range.bytes,
                )?);
            }
        }
    }
    let ticket = transfer.submit(&writes)?;
    transfer.wait(&ticket)?;

    let mut prepared_uploads = Vec::with_capacity(requests.len());
    for ((request, allocations), slots) in requests
        .iter()
        .zip(prepared_groups)
        .zip(requests.iter().map(|request| request.resource_slots))
    {
        let resources = request
            .descriptor
            .resources
            .iter()
            .cloned()
            .zip(allocations)
            .map(|(resource_descriptor, (allocation, ranges))| {
                let byte_count = resource_descriptor.byte_count;
                DeviceResidentResource::new(
                    resource_descriptor,
                    VulkanResidentCompiledResource {
                        storage:
                            VulkanResidentCompiledResourceStorage::Stable(
                                allocation,
                            ),
                        ranges,
                        byte_count,
                    },
                )
            })
            .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()
            .map_err(|error| {
                VulkanError(format!(
                    "stable uploaded compiled resource payload is invalid: {error}"
                ))
            })?;
        let resident_group = DeviceResidentResourceGroup::new(
            request.descriptor.clone(),
            resources,
        )
        .map_err(|error| {
            VulkanError(format!(
                "stable uploaded compiled resource group cannot be published: {error}"
            ))
        })?;
        let address_resources =
            slots
                .iter()
                .copied()
                .zip(resident_group.resources())
                .zip(&request.loaded.resources)
                .map(|((slot, resource), loaded_resource)| {
                    let allocation = resource
                        .payload()
                        .stable_allocation()
                        .expect("stable upload built a direct resource");
                    (
                        slot,
                        Arc::clone(allocation),
                        loaded_resource.representation.address_tag(),
                    )
                })
                .collect();
        prepared_uploads.push(VulkanPreparedStableCompiledResourceUpload {
            resident_group,
            address_resources,
        });
    }
    let prepared_publication_count = prepared_uploads.iter().try_fold(
        0usize,
        |total, upload| {
            total.checked_add(upload.address_resources.len()).ok_or_else(|| {
                VulkanError("stable upload publication count overflowed".to_string())
            })
        },
    )?;
    if prepared_publication_count != batch_slots.len() {
        return Err(VulkanError(
            "stable upload address publication plan is inconsistent"
                .to_string(),
        ));
    }
    Ok(prepared_uploads)
}

fn device_resource_descriptor_from_loaded_group(
    loaded: &LoadedCompiledResourceGroup,
) -> Result<DeviceResourceGroupDescriptor, DeviceResourceResidencyError> {
    let resources = loaded
        .resources
        .iter()
        .map(|resource| {
            let byte_count =
                resource.ranges.iter().try_fold(0usize, |total, range| {
                    total.checked_add(range.bytes.len()).ok_or_else(|| {
                        DeviceResourceResidencyError::invalid_descriptor(
                            "loaded compiled resource byte count overflowed",
                        )
                    })
                })?;
            Ok(DeviceResourceDescriptor {
                id: resource.id.clone(),
                byte_count,
                compatibility: resource.compatibility.clone(),
            })
        })
        .collect::<Result<Vec<_>, DeviceResourceResidencyError>>()?;
    DeviceResourceGroupDescriptor::new(
        loaded.id.clone(),
        loaded.resource_ids.clone(),
        loaded.dependencies.clone(),
        resources,
    )
}
