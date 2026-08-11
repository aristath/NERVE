#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceReadbackValidation {
    group_ids: Vec<String>,
    resource_count: usize,
    byte_count: usize,
    output_digest: String,
}

impl VulkanCompiledResourceDeviceStore {
    fn validate_selector_resources_readback(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        resource_indices: &[usize],
    ) -> Result<VulkanCompiledResourceReadbackValidation, VulkanCompiledResourceDeviceStoreError>
    {
        if resource_indices.is_empty()
            || resource_indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource readback validation requires sorted unique resource indices",
            ));
        }
        let mut resolved_by_group = BTreeMap::new();
        for resource_index in resource_indices {
            let resolved = self.resolve_selector_resource(selector_id, *resource_index)?;
            if let Some(existing) =
                resolved_by_group.insert(resolved.id().to_string(), resolved.clone())
                && existing != resolved
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource readback validation resolved one group identity inconsistently",
                ));
            }
        }
        let expected_groups = resolved_by_group
            .values()
            .cloned()
            .map(|resolved| {
                self.backing_store
                    .try_load(resolved)
                    .map_err(|error| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource validation load failed: {error}",
                        ))
                    })?
                    .wait()
                    .map_err(|error| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource validation read failed: {error}",
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let allocation_groups = {
            let state = self.address_state.lock().map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address state was poisoned",
                )
            })?;
            expected_groups
                .iter()
                .map(|group| {
                    let publications = state.publications.get(&group.id).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource group {:?} has no resident address publication",
                            group.id,
                        ))
                    })?;
                    state
                        .address_table
                        .allocations_for_publications(publications)
                        .map_err(compiled_device_store_vulkan_error)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut read_ranges = Vec::new();
        let mut expected_payloads = Vec::new();
        let mut identities = Vec::new();
        for (group, allocations) in expected_groups.iter().zip(&allocation_groups) {
            if allocations.len() != group.resources.len() {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource group {:?} published {} allocations for {} resources",
                    group.id,
                    allocations.len(),
                    group.resources.len(),
                )));
            }
            for (resource, allocation) in group.resources.iter().zip(allocations) {
                let expected = resource
                    .ranges
                    .iter()
                    .flat_map(|range| range.bytes.iter().copied())
                    .collect::<Vec<_>>();
                if allocation.byte_count() != expected.len() {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled resource {:?} allocation has {} bytes but its verified payload has {}",
                        resource.id,
                        allocation.byte_count(),
                        expected.len(),
                    )));
                }
                read_ranges.push(
                    VulkanResidentBufferReadRange::new(
                        allocation.buffer(),
                        allocation.buffer_byte_offset(),
                        allocation.byte_count(),
                    )
                    .map_err(compiled_device_store_vulkan_error)?,
                );
                expected_payloads.push(expected);
                identities.push((group.id.clone(), resource.id.clone()));
            }
        }
        let readback = device
            .read_resident_buffer_ranges(&read_ranges)
            .map_err(compiled_device_store_vulkan_error)?;
        let mut digest = Sha256::new();
        digest.update(b"nerve.compiled_resource_readback.v1");
        let mut byte_count = 0usize;
        for (range_index, ((group_id, resource_id), expected)) in identities
            .iter()
            .zip(&expected_payloads)
            .enumerate()
        {
            let actual = readback
                .range_bytes(range_index)
                .map_err(compiled_device_store_vulkan_error)?;
            if actual != expected {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource readback differs from verified source for group {group_id:?} resource {resource_id:?}",
                )));
            }
            byte_count = byte_count.checked_add(actual.len()).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource readback byte count overflowed",
                )
            })?;
            for field in [group_id.as_bytes(), resource_id.as_bytes(), actual] {
                digest.update((field.len() as u64).to_le_bytes());
                digest.update(field);
            }
        }
        Ok(VulkanCompiledResourceReadbackValidation {
            group_ids: expected_groups
                .iter()
                .map(|group| group.id.clone())
                .collect(),
            resource_count: identities.len(),
            byte_count,
            output_digest: format!("sha256:{:x}", digest.finalize()),
        })
    }
}
