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
    group_payload_bytes: BTreeMap<String, usize>,
    group_tiers: BTreeMap<String, VulkanCompiledResourceMemoryTier>,
    dynamic_admission: bool,
    device_payload_capacity: usize,
    host_visible_payload_capacity: usize,
    device_payload_bytes: usize,
    host_visible_payload_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VulkanCompiledResourceTierAdmission {
    tiers: Vec<VulkanCompiledResourceMemoryTier>,
    newly_assigned_group_ids: Vec<String>,
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
            group_payload_bytes: group_payload_bytes.clone(),
            group_tiers,
            dynamic_admission: false,
            device_payload_capacity,
            host_visible_payload_capacity,
            device_payload_bytes,
            host_visible_payload_bytes,
        })
    }

    fn dynamic_tiered(
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
                "dynamic tiered compiled-resource memory plan has an invalid capacity or group",
            ));
        }
        Ok(Self {
            group_payload_bytes: group_payload_bytes.clone(),
            group_tiers: BTreeMap::new(),
            dynamic_admission: true,
            device_payload_capacity,
            host_visible_payload_capacity,
            device_payload_bytes: 0,
            host_visible_payload_bytes: 0,
        })
    }

    fn admit_groups(
        &mut self,
        groups: &[(String, usize)],
    ) -> Result<VulkanCompiledResourceTierAdmission, VulkanCompiledResourceDeviceStoreError> {
        let device_preferences = groups
            .iter()
            .map(|(group_id, _)| group_id.clone())
            .collect();
        self.admit_groups_with_device_preferences(groups, &device_preferences)
    }

    fn admit_groups_with_device_preferences(
        &mut self,
        groups: &[(String, usize)],
        device_preferences: &BTreeSet<String>,
    ) -> Result<VulkanCompiledResourceTierAdmission, VulkanCompiledResourceDeviceStoreError> {
        if groups.is_empty()
            || groups
                .iter()
                .map(|(group_id, _)| group_id)
                .collect::<BTreeSet<_>>()
                .len()
                != groups.len()
            || !device_preferences
                .iter()
                .all(|group_id| groups.iter().any(|(candidate, _)| candidate == group_id))
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "tiered compiled-resource admission is empty, repeats a group, or has an unknown device preference",
            ));
        }
        let mut device_payload_bytes = self.device_payload_bytes;
        let mut host_visible_payload_bytes = self.host_visible_payload_bytes;
        let mut assignments = Vec::with_capacity(groups.len());
        let mut newly_assigned_group_ids = Vec::new();
        for (group_id, byte_count) in groups {
            let expected = self.group_payload_bytes.get(group_id).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "tiered compiled-resource admission references unknown group {group_id:?}"
                ))
            })?;
            if expected != byte_count {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "tiered compiled-resource group {group_id:?} has {byte_count} bytes, expected {expected}"
                )));
            }
            if let Some(tier) = self.group_tiers.get(group_id).copied() {
                assignments.push((group_id.clone(), tier, false));
                continue;
            }
            if !self.dynamic_admission {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "complete tiered compiled-resource plan omitted group {group_id:?}"
                )));
            }
            let device_end = device_payload_bytes.checked_add(*byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "tiered device admission byte count overflowed",
                )
            })?;
            let host_end = host_visible_payload_bytes
                .checked_add(*byte_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "tiered host-visible admission byte count overflowed",
                    )
                })?;
            let tier = if device_preferences.contains(group_id)
                && device_end <= self.device_payload_capacity
            {
                device_payload_bytes = device_end;
                VulkanCompiledResourceMemoryTier::Device
            } else if host_end <= self.host_visible_payload_capacity {
                host_visible_payload_bytes = host_end;
                VulkanCompiledResourceMemoryTier::HostVisible
            } else if device_end <= self.device_payload_capacity {
                device_payload_bytes = device_end;
                VulkanCompiledResourceMemoryTier::Device
            } else {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "tiered compiled resources cannot admit group {group_id:?}: device tier would use {device_end}/{} bytes and host-visible tier would use {host_end}/{} bytes",
                    self.device_payload_capacity, self.host_visible_payload_capacity,
                )));
            };
            assignments.push((group_id.clone(), tier, true));
            newly_assigned_group_ids.push(group_id.clone());
        }
        for (group_id, tier, is_new) in &assignments {
            if *is_new {
                self.group_tiers.insert(group_id.clone(), *tier);
            }
        }
        self.device_payload_bytes = device_payload_bytes;
        self.host_visible_payload_bytes = host_visible_payload_bytes;
        Ok(VulkanCompiledResourceTierAdmission {
            tiers: assignments.into_iter().map(|(_, tier, _)| tier).collect(),
            newly_assigned_group_ids,
        })
    }

    fn release_dynamic_groups(
        &mut self,
        group_ids: &BTreeSet<String>,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if group_ids.is_empty() {
            return Ok(());
        }
        if !self.dynamic_admission {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "cannot release groups from a complete tiered compiled-resource plan",
            ));
        }
        for group_id in group_ids {
            let tier = self.group_tiers.remove(group_id).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "dynamic tiered compiled-resource plan has no admitted group {group_id:?}"
                ))
            })?;
            let byte_count = self.group_payload_bytes[group_id];
            match tier {
                VulkanCompiledResourceMemoryTier::Device => {
                    self.device_payload_bytes = self
                        .device_payload_bytes
                        .checked_sub(byte_count)
                        .expect("admitted device-tier bytes include the released group");
                }
                VulkanCompiledResourceMemoryTier::HostVisible => {
                    self.host_visible_payload_bytes = self
                        .host_visible_payload_bytes
                        .checked_sub(byte_count)
                        .expect("admitted host-tier bytes include the released group");
                }
            }
        }
        Ok(())
    }

    fn rollback_admission(
        &mut self,
        admission: &VulkanCompiledResourceTierAdmission,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        self.release_dynamic_groups(
            &admission
                .newly_assigned_group_ids
                .iter()
                .cloned()
                .collect(),
        )
    }

    fn clear_dynamic_admissions(
        &mut self,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if !self.dynamic_admission {
            return Ok(());
        }
        let group_ids = self.group_tiers.keys().cloned().collect();
        self.release_dynamic_groups(&group_ids)
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
