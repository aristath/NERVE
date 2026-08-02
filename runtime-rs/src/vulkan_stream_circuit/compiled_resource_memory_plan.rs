#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VulkanCompiledResourceMemoryTier {
    Device,
    HostVisible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VulkanCompiledResourceAllocationCohort {
    tier: VulkanCompiledResourceMemoryTier,
    chunk_id: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceMemoryPlan {
    group_tiers: BTreeMap<String, VulkanCompiledResourceMemoryTier>,
    device_payload_bytes: usize,
    host_visible_payload_bytes: usize,
}

impl VulkanCompiledResourceMemoryPlan {
    fn exact_tiered(
        group_payload_bytes: &BTreeMap<String, usize>,
        device_payload_capacity: usize,
        host_visible_payload_capacity: usize,
    ) -> Result<Self, VulkanCompiledResourceDeviceStoreError> {
        if group_payload_bytes.is_empty()
            || device_payload_capacity == 0
            || host_visible_payload_capacity == 0
            || group_payload_bytes
                .values()
                .any(|byte_count| *byte_count == 0)
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "tiered compiled-resource memory plan has an invalid capacity or group",
            ));
        }
        let mut group_tiers = BTreeMap::new();
        let mut device_payload_bytes = 0usize;
        let mut host_visible_payload_bytes = 0usize;
        for (group_id, byte_count) in group_payload_bytes {
            let device_end = device_payload_bytes
                .checked_add(*byte_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "tiered device payload byte count overflowed",
                    )
                })?;
            let tier = if device_end <= device_payload_capacity {
                device_payload_bytes = device_end;
                VulkanCompiledResourceMemoryTier::Device
            } else {
                host_visible_payload_bytes = host_visible_payload_bytes
                    .checked_add(*byte_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "tiered host-visible payload byte count overflowed",
                        )
                    })?;
                VulkanCompiledResourceMemoryTier::HostVisible
            };
            group_tiers.insert(group_id.clone(), tier);
        }
        if host_visible_payload_bytes > host_visible_payload_capacity {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "tiered compiled resources need {host_visible_payload_bytes} host-visible payload bytes, but only {host_visible_payload_capacity} are available"
            )));
        }
        Ok(Self {
            group_tiers,
            device_payload_bytes,
            host_visible_payload_bytes,
        })
    }

    fn tier_for_group(
        &self,
        group_id: &str,
    ) -> Result<VulkanCompiledResourceMemoryTier, VulkanCompiledResourceDeviceStoreError> {
        self.group_tiers.get(group_id).copied().ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "tiered compiled-resource memory plan omitted group {group_id:?}"
            ))
        })
    }

    fn validate_group_tier_exchange(
        &self,
        device_group_id: &str,
        host_visible_group_id: &str,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if device_group_id == host_visible_group_id
            || self.tier_for_group(device_group_id)? != VulkanCompiledResourceMemoryTier::Device
            || self.tier_for_group(host_visible_group_id)?
                != VulkanCompiledResourceMemoryTier::HostVisible
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "tiered compiled-resource exchange does not name one device and one host-visible group",
            ));
        }
        Ok(())
    }

    fn commit_group_tier_exchange(
        &mut self,
        device_group_id: &str,
        host_visible_group_id: &str,
    ) {
        debug_assert!(
            self.validate_group_tier_exchange(device_group_id, host_visible_group_id)
                .is_ok(),
            "compiled-resource tier exchange must be validated before commit",
        );
        self.group_tiers.insert(
            device_group_id.to_string(),
            VulkanCompiledResourceMemoryTier::HostVisible,
        );
        self.group_tiers.insert(
            host_visible_group_id.to_string(),
            VulkanCompiledResourceMemoryTier::Device,
        );
    }
}
