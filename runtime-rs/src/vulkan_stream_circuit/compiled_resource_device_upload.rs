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
}

impl DeviceResidentResourcePayload for VulkanResidentCompiledResource {
    fn byte_count(&self) -> usize {
        self.byte_count
    }
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
    let loaded_descriptor =
        device_resource_descriptor_from_loaded_group(loaded).map_err(
            |error| {
                VulkanError(format!(
                    "loaded compiled resource group is invalid: {error}"
                ))
            },
        )?;
    if loaded_descriptor != *descriptor {
        return Err(VulkanError(
            "loaded compiled resources do not match the reserved device group"
                .to_string(),
        ));
    }
    if resource_slots.len() != descriptor.resources.len() {
        return Err(VulkanError(format!(
            "stable resource upload has {} address-table slots for {} resources",
            resource_slots.len(),
            descriptor.resources.len()
        )));
    }
    let unique_slots = resource_slots.iter().copied().collect::<BTreeSet<_>>();
    if unique_slots.len() != resource_slots.len() {
        return Err(VulkanError(
            "stable resource upload repeats an address-table slot".to_string(),
        ));
    }
    if let Some(slot) = resource_slots
        .iter()
        .copied()
        .find(|slot| *slot >= address_table.slot_count())
    {
        return Err(VulkanError(format!(
            "stable resource upload slot {slot} exceeds address-table capacity {}",
            address_table.slot_count()
        )));
    }

    let mut allocations = Vec::with_capacity(loaded.resources.len());
    for (resource, resource_descriptor) in
        loaded.resources.iter().zip(&descriptor.resources)
    {
        let allocation = Arc::new(arena.allocate(
            device,
            resource_descriptor.byte_count,
            alignment,
        )?);
        let mut byte_offset = allocation.buffer_byte_offset();
        let mut ranges = Vec::with_capacity(resource.ranges.len());
        for range in &resource.ranges {
            ranges.push(VulkanResidentCompiledResourceRange {
                byte_offset,
                byte_count: range.bytes.len(),
            });
            byte_offset =
                byte_offset.checked_add(range.bytes.len()).ok_or_else(|| {
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

    let mut writes = Vec::new();
    for ((resource, _), (allocation, ranges)) in loaded
        .resources
        .iter()
        .zip(&descriptor.resources)
        .zip(&allocations)
    {
        for (range, placement) in resource.ranges.iter().zip(ranges) {
            writes.push(VulkanResidentBufferWriteRange::new(
                allocation.buffer(),
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
        .map(|(resource_descriptor, (allocation, ranges))| {
            let byte_count = resource_descriptor.byte_count;
            DeviceResidentResource::new(
                resource_descriptor,
                VulkanResidentCompiledResource {
                    storage: VulkanResidentCompiledResourceStorage::Stable(
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
    let resident_group =
        DeviceResidentResourceGroup::new(descriptor.clone(), resources)
            .map_err(|error| {
                VulkanError(format!(
                    "stable uploaded compiled resource group cannot be published: {error}"
                ))
            })?;
    let address_resources = resource_slots
        .iter()
        .copied()
        .zip(resident_group.resources())
        .map(|(slot, resource)| {
            let allocation = resource
                .payload()
                .stable_allocation()
                .expect("stable upload built a direct resource");
            (slot, Arc::clone(allocation))
        })
        .collect::<Vec<_>>();
    let publications =
        address_table.publish_group(transfer, &address_resources)?;
    Ok(VulkanStableCompiledResourceUpload {
        resident_group,
        publications,
    })
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
