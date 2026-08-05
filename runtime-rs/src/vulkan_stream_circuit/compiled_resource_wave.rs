struct VulkanCompiledResourceWaveAdmission {
    group_tiers: BTreeMap<String, VulkanCompiledResourceMemoryTier>,
    new_group_ids: BTreeSet<String>,
    newly_assigned_group_ids: BTreeSet<String>,
    new_payload_bytes: usize,
}

fn compiled_resource_device_tier_group_ids<'a>(
    group_ids: impl IntoIterator<Item = &'a str>,
    admission: &VulkanCompiledResourceWaveAdmission,
) -> Result<BTreeSet<String>, VulkanCompiledResourceDeviceStoreError> {
    group_ids
        .into_iter()
        .filter(|group_id| admission.new_group_ids.contains(*group_id))
        .try_fold(BTreeSet::new(), |mut device_group_ids, group_id| {
            let tier = admission.group_tiers.get(group_id).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource wave omitted the reserved tier for new group {group_id:?}"
                ))
            })?;
            if *tier == VulkanCompiledResourceMemoryTier::Device {
                device_group_ids.insert(group_id.to_string());
            }
            Ok(device_group_ids)
        })
}

fn compiled_resource_preferred_device_group_ids<F>(
    groups: &[(String, usize)],
    available_device_payload_bytes: usize,
    available_device_allocation_bytes: usize,
    mut additional_allocation_bytes: F,
) -> Result<BTreeSet<String>, VulkanCompiledResourceDeviceStoreError>
where
    F: FnMut(&BTreeSet<String>) -> Result<usize, VulkanCompiledResourceDeviceStoreError>,
{
    let mut preferred = BTreeSet::new();
    let mut remaining_payload_bytes = available_device_payload_bytes;
    for (group_id, payload_bytes) in groups {
        if *payload_bytes > remaining_payload_bytes {
            continue;
        }
        preferred.insert(group_id.clone());
        if additional_allocation_bytes(&preferred)? <= available_device_allocation_bytes {
            remaining_payload_bytes -= *payload_bytes;
        } else {
            preferred.remove(group_id);
        }
    }
    Ok(preferred)
}

impl VulkanCompiledResourceDeviceStore {
    fn load_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
        owner: DeviceResourceResidencyOwnerId,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        self.reclaim_compiled_resource_wave_payload(
            selector_id,
            plans,
            protected_group_ids,
        )?;
        let mut admission = self.reserve_compiled_resource_wave_tiers(device, plans)?;
        let result = self.load_admitted_compiled_resource_wave(
            device,
            selector_id,
            plans,
            protected_group_ids,
            owner,
            &mut admission,
        );
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if !admission.newly_assigned_group_ids.is_empty()
                    && let Err(rollback_error) =
                        self.rollback_compiled_resource_wave_tiers(&admission)
                {
                    return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                        "{error}; additionally failed to roll back provisional tier admission: {rollback_error}",
                    )));
                }
                Err(error)
            }
        }
    }

    fn load_admitted_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
        owner: DeviceResourceResidencyOwnerId,
        admission: &mut VulkanCompiledResourceWaveAdmission,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let device_capacity_permit = if self.residency_policy.evicts_inactive_resources() {
            self.evict_for_compiled_resource_wave(
                device,
                selector_id,
                plans,
                protected_group_ids,
                admission,
            )?
        } else {
            None
        };
        let requests = self
            .manager
            .request_batch(plans.iter().map(|plan| plan.descriptor.clone()), owner)
            .map_err(compiled_device_store_residency_error)?;
        let mut pending = Vec::new();
        let mut required = Vec::new();
        for (plan, request) in plans.iter().zip(requests) {
            match request {
                DeviceResourceResidencyRequest::Resident(lease) => {
                    drop(lease);
                }
                DeviceResourceResidencyRequest::Pending(waiter) => {
                    pending.push(waiter);
                }
                DeviceResourceResidencyRequest::LoadRequired(permit) => {
                    required.push((plan, permit));
                }
            }
        }
        let _blocking = (!pending.is_empty() || !required.is_empty())
            .then(|| VulkanCompiledResourceBlockingTimer::new(&self.instrumentation));
        let mut submitted = Vec::with_capacity(required.len());
        for (plan, permit) in required {
            match self.backing_store.try_load(plan.resolved.clone()) {
                Ok(ticket) => submitted.push((plan, permit, ticket)),
                Err(error) => {
                    let message = format!("compiled resource backing-store load failed: {error}");
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                    return Err(VulkanCompiledResourceDeviceStoreError::new(message));
                }
            }
        }
        let mut loaded = Vec::with_capacity(submitted.len());
        for (plan, permit, ticket) in submitted {
            match ticket.wait() {
                Ok(group) => loaded.push((plan, permit, group)),
                Err(error) => {
                    let message = format!("compiled resource backing-store load failed: {error}");
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                    return Err(VulkanCompiledResourceDeviceStoreError::new(message));
                }
            }
        }
        if !loaded.is_empty() {
            self.publish_loaded_compiled_resource_wave(
                device,
                loaded,
                device_capacity_permit,
                admission,
            )?;
        }
        for waiter in pending {
            waiter
                .wait()
                .map(drop)
                .map_err(compiled_device_store_residency_error)?;
        }
        if !admission.newly_assigned_group_ids.is_empty() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource wave completed without publishing provisional tier assignments {:?}",
                admission.newly_assigned_group_ids
            )));
        }
        Ok(())
    }

    fn reclaim_compiled_resource_wave_payload(
        &self,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let residency_requirement = || {
            let snapshot = self
                .manager
                .snapshot()
                .map_err(compiled_device_store_residency_error)?;
            let known_group_ids = snapshot
                .directory
                .iter()
                .map(|entry| entry.group_id.as_str())
                .collect::<BTreeSet<_>>();
            let new_payload_bytes = plans
                .iter()
                .filter(|plan| !known_group_ids.contains(plan.descriptor.id.as_str()))
                .try_fold(0usize, |total, plan| {
                    total.checked_add(plan.descriptor.byte_count).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource admission byte count overflowed",
                        )
                    })
                })?;
            let used_payload_bytes = snapshot
                .statistics
                .always_resident_bytes
                .checked_add(snapshot.statistics.dynamic_resident_bytes)
                .and_then(|used| used.checked_add(snapshot.statistics.reserved_loading_bytes))
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource residency accounting overflowed",
                    )
                })?;
            let required_payload_bytes = used_payload_bytes
                .saturating_add(new_payload_bytes)
                .saturating_sub(snapshot.statistics.capacity_bytes);
            Ok::<_, VulkanCompiledResourceDeviceStoreError>((
                snapshot,
                new_payload_bytes,
                required_payload_bytes,
            ))
        };
        if residency_requirement()?.2 == 0 {
            return Ok(());
        }
        if !self.residency_policy.evicts_inactive_resources() {
            return Ok(());
        }
        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        let (snapshot, new_payload_bytes, required_payload_bytes) = residency_requirement()?;
        if required_payload_bytes == 0 {
            return Ok(());
        }
        let candidates = self
            .manager
            .eviction_candidates(protected_group_ids)
            .map_err(compiled_device_store_residency_error)?;
        let candidates = compiled_resource_selector_fair_eviction_candidates(
            &candidates,
            &snapshot.directory,
            &self.group_selector_ids,
            &self.selector_payload_budgets,
            selector_id,
            new_payload_bytes,
        )?;
        self.evict_inactive_capacity(
            &candidates,
            protected_group_ids,
            required_payload_bytes,
            0,
            true,
        )?;
        Ok(())
    }

    fn reserve_compiled_resource_wave_tiers(
        &self,
        device: &VulkanComputeDevice,
        plans: &[VulkanCompiledResourceLoadPlan],
    ) -> Result<VulkanCompiledResourceWaveAdmission, VulkanCompiledResourceDeviceStoreError> {
        let snapshot = self
            .manager
            .snapshot()
            .map_err(compiled_device_store_residency_error)?;
        let known_group_ids = snapshot
            .directory
            .iter()
            .map(|entry| entry.group_id.as_str())
            .collect::<BTreeSet<_>>();
        let new_plans = plans
            .iter()
            .filter(|plan| !known_group_ids.contains(plan.descriptor.id.as_str()))
            .collect::<Vec<_>>();
        let new_payload_bytes = new_plans.iter().try_fold(0usize, |total, plan| {
            total.checked_add(plan.descriptor.byte_count).ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource tier-reservation byte count overflowed",
                )
            })
        })?;
        let new_group_ids = new_plans
            .iter()
            .map(|plan| plan.descriptor.id.clone())
            .collect::<BTreeSet<_>>();
        let (group_tiers, newly_assigned_group_ids) = match &self.memory_plan {
            Some(memory_plan) if !new_plans.is_empty() => {
                let (dynamic_admission, available_device_payload_bytes) = {
                    let memory_plan = memory_plan.lock().map_err(|_| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource memory plan was poisoned",
                        )
                    })?;
                    (
                        memory_plan.dynamic_admission,
                        memory_plan
                            .device_payload_capacity
                            .saturating_sub(memory_plan.device_payload_bytes),
                    )
                };
                let groups = new_plans
                    .iter()
                    .map(|plan| (plan.descriptor.id.clone(), plan.descriptor.byte_count))
                    .collect::<Vec<_>>();
                let device_preferences = if dynamic_admission {
                    let globally_available_device_bytes = usize::try_from(
                        device
                            .device_local_memory_accounting()
                            .map_err(compiled_device_store_vulkan_error)?
                            .admissible_remaining_bytes,
                    )
                    .unwrap_or(usize::MAX);
                    let arena = self
                        .device_arena
                        .stats()
                        .map_err(compiled_device_store_vulkan_error)?;
                    let arena_available_device_bytes = self
                        .device_arena
                        .config()
                        .committed_byte_capacity
                        .saturating_sub(arena.committed_byte_capacity);
                    compiled_resource_preferred_device_group_ids(
                        &groups,
                        available_device_payload_bytes,
                        globally_available_device_bytes.min(arena_available_device_bytes),
                        |preferred_group_ids| {
                            let preferred_plans = new_plans
                                .iter()
                                .filter(|plan| {
                                    preferred_group_ids.contains(&plan.descriptor.id)
                                })
                                .collect::<Vec<_>>();
                            let allocation_byte_counts = preferred_plans
                                .iter()
                                .map(|plan| {
                                    plan.descriptor
                                        .resources
                                        .iter()
                                        .map(|resource| resource.byte_count)
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            let allocation_requests = preferred_plans
                                .iter()
                                .zip(&allocation_byte_counts)
                                .map(|(plan, byte_counts)| {
                                    (plan.resource_slots.as_slice(), byte_counts.as_slice())
                                })
                                .collect::<Vec<_>>();
                            if allocation_requests.is_empty() {
                                Ok(0)
                            } else {
                                self.device_arena
                                    .additional_committed_byte_capacity_for_groups(
                                        device,
                                        &allocation_requests,
                                        self.upload_alignment,
                                    )
                                    .map_err(compiled_device_store_vulkan_error)
                            }
                        },
                    )?
                } else {
                    new_group_ids.clone()
                };
                let mut memory_plan = memory_plan.lock().map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned",
                    )
                })?;
                let tier_admission = if dynamic_admission {
                    memory_plan
                        .admit_groups_with_device_preferences(&groups, &device_preferences)?
                } else {
                    memory_plan.admit_groups(&groups)?
                };
                (
                    new_plans
                        .iter()
                        .zip(tier_admission.tiers)
                        .map(|(plan, tier)| (plan.descriptor.id.clone(), tier))
                        .collect(),
                    tier_admission
                        .newly_assigned_group_ids
                        .into_iter()
                        .collect(),
                )
            }
            _ => (
                new_group_ids
                    .iter()
                    .cloned()
                    .map(|group_id| (group_id, VulkanCompiledResourceMemoryTier::Device))
                    .collect(),
                BTreeSet::new(),
            ),
        };
        Ok(VulkanCompiledResourceWaveAdmission {
            group_tiers,
            new_group_ids,
            newly_assigned_group_ids,
            new_payload_bytes,
        })
    }

    fn rollback_compiled_resource_wave_tiers(
        &self,
        admission: &VulkanCompiledResourceWaveAdmission,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        if admission.newly_assigned_group_ids.is_empty() {
            return Ok(());
        }
        let rollback = VulkanCompiledResourceTierAdmission {
            tiers: Vec::new(),
            newly_assigned_group_ids: admission
                .newly_assigned_group_ids
                .iter()
                .cloned()
                .collect(),
        };
        self.memory_plan
            .as_ref()
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource wave has provisional tiers without a memory plan",
                )
            })?
            .lock()
            .map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource memory plan was poisoned",
                )
            })?
            .rollback_admission(&rollback)
    }

    fn evict_inactive_capacity(
        &self,
        candidates: &[DeviceResourceResidencyEvictionCandidate],
        protected_group_ids: &BTreeSet<String>,
        required_payload_bytes: usize,
        required_device_bytes: usize,
        require_full_capacity: bool,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let mut address_state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let mut chunk_byte_capacities = BTreeMap::new();
        for cohort in address_state.chunk_groups.keys() {
            let arena = match cohort.tier {
                VulkanCompiledResourceMemoryTier::Device => &self.device_arena,
                VulkanCompiledResourceMemoryTier::HostVisible => {
                    self.host_visible_arena.as_ref().ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "host-visible allocation cohort has no stable arena",
                        )
                    })?
                }
            };
            let byte_capacity = arena
                .committed_byte_capacity_for_chunk(cohort.chunk_id)
                .map_err(compiled_device_store_vulkan_error)?;
            chunk_byte_capacities.insert(*cohort, byte_capacity);
        }
        let selection = compiled_resource_lru_eviction_selection(
            candidates,
            &address_state.group_chunks,
            &address_state.chunk_groups,
            &chunk_byte_capacities,
            protected_group_ids,
            required_payload_bytes,
            required_device_bytes,
        )?;
        if require_full_capacity
            && (selection.payload_bytes < required_payload_bytes
                || selection.device_bytes < required_device_bytes)
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource residency needs to reclaim {required_payload_bytes} payload and {required_device_bytes} device bytes, but inactive allocation cohorts provide only {} payload and {} device bytes",
                selection.payload_bytes, selection.device_bytes,
            )));
        }
        let reusable_device_bytes = selection.device_bytes;
        let selected_group_ids = selection.group_ids;
        if selected_group_ids.is_empty() {
            return Ok(0);
        }
        let eviction = self
            .manager
            .evict_inactive_groups(selected_group_ids.clone())
            .map_err(compiled_device_store_residency_error)?;
        let publications = selected_group_ids
            .iter()
            .flat_map(|group_id| {
                address_state
                    .publications
                    .get(group_id)
                    .into_iter()
                    .flatten()
                    .cloned()
            })
            .collect::<Vec<_>>();
        if publications.is_empty() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "resident eviction cohort has no address publications",
            ));
        }
        {
            let VulkanCompiledResourceDeviceAddressState {
                transfer,
                address_table,
                ..
            } = &mut *address_state;
            address_table
                .clear_group(transfer, &publications)
                .map_err(compiled_device_store_vulkan_error)?;
        }
        for group_id in &selected_group_ids {
            address_state.publications.remove(group_id);
            let chunks = address_state
                .group_chunks
                .remove(group_id)
                .unwrap_or_default();
            for chunk_id in chunks {
                let remove_chunk =
                    if let Some(groups) = address_state.chunk_groups.get_mut(&chunk_id) {
                        groups.remove(group_id);
                        groups.is_empty()
                    } else {
                        false
                    };
                if remove_chunk {
                    address_state.chunk_groups.remove(&chunk_id);
                }
            }
        }
        let release = eviction.release();
        drop(eviction);
        if let Some(memory_plan) = &self.memory_plan {
            memory_plan
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned",
                    )
                })?
                .release_dynamic_groups(&selected_group_ids)?;
        }
        if release.group_count == 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource eviction selected groups but released no logical residency",
            ));
        }
        Ok(reusable_device_bytes)
    }

    fn reclaim_inactive_device_memory(
        &self,
        requested_bytes: usize,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        if requested_bytes == 0 || !self.residency_policy.evicts_inactive_resources() {
            return Ok(0);
        }
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned",
            )
        })?;
        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        let mut released_device_bytes = self
            .device_arena
            .trim_inactive_backing(requested_bytes)
            .map_err(compiled_device_store_vulkan_error)?;
        if released_device_bytes >= requested_bytes {
            self.instrumentation
                .record_released_device_bytes(released_device_bytes);
            return Ok(released_device_bytes);
        }
        let protected_group_ids = BTreeSet::new();
        let candidates = self
            .manager
            .eviction_candidates(&protected_group_ids)
            .map_err(compiled_device_store_residency_error)?;
        self.evict_inactive_capacity(
            &candidates,
            &protected_group_ids,
            0,
            requested_bytes.saturating_sub(released_device_bytes),
            false,
        )?;
        released_device_bytes = released_device_bytes
            .checked_add(
                self.device_arena
                    .trim_inactive_backing(requested_bytes.saturating_sub(released_device_bytes))
                    .map_err(compiled_device_store_vulkan_error)?,
            )
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource released device capacity overflowed",
                )
            })?;
        if released_device_bytes > 0 {
            self.instrumentation
                .record_released_device_bytes(released_device_bytes);
        }
        Ok(released_device_bytes)
    }

    fn evict_for_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        selector_id: &str,
        plans: &[VulkanCompiledResourceLoadPlan],
        protected_group_ids: &BTreeSet<String>,
        admission: &VulkanCompiledResourceWaveAdmission,
    ) -> Result<Option<VulkanDeviceLocalMemoryPermit>, VulkanCompiledResourceDeviceStoreError> {
        let residency_requirement = || {
            let snapshot = self
                .manager
                .snapshot()
                .map_err(compiled_device_store_residency_error)?;
            let device_group_ids = compiled_resource_device_tier_group_ids(
                plans.iter().map(|plan| plan.descriptor.id.as_str()),
                admission,
            )?;
            let new_plans = plans
                .iter()
                .filter(|plan| device_group_ids.contains(&plan.descriptor.id))
                .collect::<Vec<_>>();
            let allocation_byte_counts = new_plans
                .iter()
                .map(|plan| {
                    plan.descriptor
                        .resources
                        .iter()
                        .map(|resource| resource.byte_count)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let allocation_requests = new_plans
                .iter()
                .zip(&allocation_byte_counts)
                .map(|(plan, byte_counts)| {
                    (plan.resource_slots.as_slice(), byte_counts.as_slice())
                })
                .collect::<Vec<_>>();
            let new_device_bytes = if allocation_requests.is_empty() {
                0
            } else {
                self.device_arena
                    .additional_committed_byte_capacity_for_groups(
                        device,
                        &allocation_requests,
                        self.upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?
            };
            let globally_available_device_bytes = usize::try_from(
                device
                    .device_local_memory_accounting()
                    .map_err(compiled_device_store_vulkan_error)?
                    .admissible_remaining_bytes,
            )
            .unwrap_or(usize::MAX);
            let arena = self
                .device_arena
                .stats()
                .map_err(compiled_device_store_vulkan_error)?;
            let arena_available_device_bytes = self
                .device_arena
                .config()
                .committed_byte_capacity
                .saturating_sub(arena.committed_byte_capacity);
            let required_device_bytes = new_device_bytes
                .saturating_sub(globally_available_device_bytes)
                .max(new_device_bytes.saturating_sub(arena_available_device_bytes));
            let used_payload_bytes = snapshot
                .statistics
                .always_resident_bytes
                .checked_add(snapshot.statistics.dynamic_resident_bytes)
                .and_then(|used| used.checked_add(snapshot.statistics.reserved_loading_bytes))
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource residency accounting overflowed",
                    )
                })?;
            let required_payload_bytes = used_payload_bytes
                .saturating_add(admission.new_payload_bytes)
                .saturating_sub(snapshot.statistics.capacity_bytes);
            Ok::<_, VulkanCompiledResourceDeviceStoreError>((
                snapshot,
                admission.new_payload_bytes,
                required_payload_bytes,
                new_device_bytes,
                required_device_bytes,
            ))
        };
        let initial_requirement = residency_requirement()?;
        if initial_requirement.2 == 0 && initial_requirement.4 == 0 {
            return if initial_requirement.3 == 0 {
                Ok(None)
            } else {
                device
                    .reserve_device_local_memory_capacity(initial_requirement.3)
                    .map(Some)
                    .map_err(compiled_device_store_vulkan_error)
            };
        }

        let _execution = self.execution_barrier.write().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        let (
            snapshot,
            new_payload_bytes,
            required_payload_bytes,
            new_device_bytes,
            required_device_bytes,
        ) = residency_requirement()?;
        if required_payload_bytes == 0 && required_device_bytes == 0 {
            return if new_device_bytes == 0 {
                Ok(None)
            } else {
                device
                    .reserve_device_local_memory_capacity(new_device_bytes)
                    .map(Some)
                    .map_err(compiled_device_store_vulkan_error)
            };
        }
        let candidates = self
            .manager
            .eviction_candidates(protected_group_ids)
            .map_err(compiled_device_store_residency_error)?;
        let candidates = compiled_resource_selector_fair_eviction_candidates(
            &candidates,
            &snapshot.directory,
            &self.group_selector_ids,
            &self.selector_payload_budgets,
            selector_id,
            new_payload_bytes,
        )?;
        self.evict_inactive_capacity(
            &candidates,
            protected_group_ids,
            required_payload_bytes,
            required_device_bytes,
            true,
        )?;
        let mut settled_requirement = residency_requirement()?;
        if settled_requirement.2 > 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource eviction left {} payload bytes above the residency budget",
                settled_requirement.2,
            )));
        }
        if settled_requirement.4 > 0 {
            // A differently shaped wave may not fit any retained inactive
            // chunk. Only this explicit arena-rebalancing path releases
            // physical backing; ordinary expert replacement reuses it.
            let released = self
                .device_arena
                .trim_inactive_backing(settled_requirement.4)
                .map_err(compiled_device_store_vulkan_error)?;
            if released > 0 {
                self.instrumentation.record_released_device_bytes(released);
            }
            settled_requirement = residency_requirement()?;
        }
        let new_device_bytes = settled_requirement.3;
        if settled_requirement.2 > 0 || settled_requirement.4 > 0 {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource eviction and arena rebalance left {} payload and {} device bytes unavailable",
                settled_requirement.2, settled_requirement.4,
            )));
        }
        if new_device_bytes == 0 {
            return Ok(None);
        }
        let committed_after = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?
            .committed_byte_capacity;
        let remaining_arena_bytes = self
            .device_arena
            .config()
            .committed_byte_capacity
            .saturating_sub(committed_after);
        if remaining_arena_bytes < new_device_bytes {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource eviction reclaimed its selected cohorts, but the next {new_device_bytes}-byte device allocation has only {remaining_arena_bytes} arena bytes available"
            )));
        }
        wait_for_compiled_resource_device_capacity(
            new_device_bytes,
            COMPILED_RESOURCE_DEVICE_CAPACITY_SETTLEMENT_TIMEOUT,
            || {
                usize::try_from(
                    device
                        .device_local_memory_accounting()
                        .map_err(compiled_device_store_vulkan_error)?
                        .admissible_remaining_bytes,
                )
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "available Vulkan device-local capacity exceeds the host address space",
                    )
                })
            },
        )?;
        device
            .reserve_device_local_memory_capacity(new_device_bytes)
            .map(Some)
            .map_err(compiled_device_store_vulkan_error)
    }

    fn begin_execution(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<VulkanCompiledResourceExecutionGuard<'_>, VulkanCompiledResourceDeviceStoreError>
    {
        device
            .ensure_device_local_memory_headroom()
            .map_err(compiled_device_store_vulkan_error)?;
        self.begin_execution_after_headroom_check()
    }

    fn begin_execution_after_headroom_check(
        &self,
    ) -> Result<VulkanCompiledResourceExecutionGuard<'_>, VulkanCompiledResourceDeviceStoreError>
    {
        let guard = self.execution_barrier.read().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource execution barrier was poisoned",
            )
        })?;
        Ok(VulkanCompiledResourceExecutionGuard { _guard: guard })
    }

    fn publish_loaded_compiled_resource_wave(
        &self,
        device: &VulkanComputeDevice,
        loaded: Vec<(
            &VulkanCompiledResourceLoadPlan,
            DeviceResourceLoadPermit<VulkanResidentCompiledResource>,
            LoadedCompiledResourceGroup,
        )>,
        device_capacity_permit: Option<VulkanDeviceLocalMemoryPermit>,
        admission: &mut VulkanCompiledResourceWaveAdmission,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        #[cfg(test)]
        if self
            .fail_next_upload_as_device_lost
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            let error = VulkanError(
                "injected compiled resource upload failure: ERROR_DEVICE_LOST".to_string(),
            );
            self.record_terminal_device_failure(&error)?;
            for (_, permit, _) in loaded {
                let _ = permit.fail(DeviceResourceResidencyError::load_failed(error.to_string()));
            }
            return Err(compiled_device_store_vulkan_error(error));
        }
        if let Err(error) = self.ensure_device_work_is_available() {
            for (_, permit, _) in loaded {
                let _ = permit.fail(DeviceResourceResidencyError::load_failed(error.to_string()));
            }
            return Err(error);
        }
        let mut state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        for (plan, _, _) in &loaded {
            if state.publications.contains_key(&plan.descriptor.id) {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address publication exists without a resident directory entry",
                ));
            }
        }
        let upload_started = Instant::now();
        let total_uploaded_bytes = loaded.iter().try_fold(0usize, |total, (plan, _, _)| {
            total
                .checked_add(plan.descriptor.byte_count)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource upload batch byte count overflowed",
                    )
                })
        })?;
        let VulkanCompiledResourceDeviceAddressState {
            transfer,
            address_table,
            publications: resident_publications,
            group_chunks,
            chunk_groups,
        } = &mut *state;
        let mut device_indices = Vec::new();
        let mut host_visible_indices = Vec::new();
        let tiers = loaded
            .iter()
            .map(|(plan, _, _)| {
                admission
                    .group_tiers
                    .get(&plan.descriptor.id)
                    .copied()
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource upload has no reserved tier for new group {:?}",
                            plan.descriptor.id
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (index, tier) in tiers.iter().copied().enumerate() {
            match tier {
                VulkanCompiledResourceMemoryTier::Device => device_indices.push(index),
                VulkanCompiledResourceMemoryTier::HostVisible => host_visible_indices.push(index),
            }
        }
        let device_capacity_permit = match device_capacity_permit {
            Some(permit) => Some(permit),
            None if !device_indices.is_empty() => {
                let allocation_byte_counts = device_indices
                    .iter()
                    .map(|index| {
                        loaded[*index]
                            .0
                            .descriptor
                            .resources
                            .iter()
                            .map(|resource| resource.byte_count)
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let allocation_requests = device_indices
                    .iter()
                    .zip(&allocation_byte_counts)
                    .map(|(index, byte_counts)| {
                        (loaded[*index].0.resource_slots.as_slice(), byte_counts.as_slice())
                    })
                    .collect::<Vec<_>>();
                let required_bytes = self
                    .device_arena
                    .additional_committed_byte_capacity_for_groups(
                        device,
                        &allocation_requests,
                        self.upload_alignment,
                    )
                    .map_err(compiled_device_store_vulkan_error)?;
                if required_bytes == 0 {
                    None
                } else {
                    Some(
                        device
                            .reserve_device_local_memory_capacity(required_bytes)
                            .map_err(compiled_device_store_vulkan_error)?,
                    )
                }
            }
            None => None,
        };
        let upload_result = (|| {
            let mut indexed_uploads = upload_compiled_resource_tier(
                device,
                transfer,
                &self.device_arena,
                address_table,
                &loaded,
                &device_indices,
                self.upload_alignment,
                device_capacity_permit,
            )?;
            if !host_visible_indices.is_empty() {
                let host_visible_arena = self.host_visible_arena.as_ref().ok_or_else(|| {
                    VulkanError(
                        "tiered compiled-resource plan has no host-visible arena".to_string(),
                    )
                })?;
                match upload_compiled_resource_tier(
                    device,
                    transfer,
                    host_visible_arena,
                    address_table,
                    &loaded,
                    &host_visible_indices,
                    self.upload_alignment,
                    None,
                ) {
                    Ok(host_uploads) => indexed_uploads.extend(host_uploads),
                    Err(error) => {
                        let publications = indexed_uploads
                            .iter()
                            .flat_map(|(_, upload)| upload.publications().iter().cloned())
                            .collect::<Vec<_>>();
                        if !publications.is_empty() {
                            address_table.clear_group(transfer, &publications)?;
                        }
                        return Err(error);
                    }
                }
            }
            indexed_uploads.sort_by_key(|(index, _)| *index);
            Ok(indexed_uploads
                .into_iter()
                .map(|(_, upload)| upload)
                .collect::<Vec<_>>())
        })();
        let uploads = match upload_result {
            Ok(uploads) => uploads,
            Err(error) => {
                if compiled_resource_vulkan_error_is_device_loss(&error) {
                    self.record_terminal_device_failure(&error)?;
                }
                let message = format!("compiled resource upload failed: {error}");
                for (_, permit, _) in loaded {
                    let _ = permit.fail(DeviceResourceResidencyError::load_failed(message.clone()));
                }
                return Err(compiled_device_store_vulkan_error(error));
            }
        };
        let mut staged = loaded
            .into_iter()
            .zip(uploads)
            .zip(tiers)
            .map(|(((plan, permit, _), upload), tier)| {
                let (group, publications) = upload.into_parts();
                let chunks = group
                    .resources()
                    .iter()
                    .filter_map(|resource| {
                        resource
                            .payload()
                            .stable_chunk_id()
                            .map(|chunk_id| VulkanCompiledResourceAllocationCohort {
                                tier,
                                chunk_id,
                            })
                    })
                    .collect::<BTreeSet<_>>();
                (
                    plan.descriptor.id.clone(),
                    permit,
                    group,
                    publications,
                    chunks,
                )
            })
            .collect::<Vec<_>>();
        while !staged.is_empty() {
            let (group_id, permit, group, publications, chunks) = staged.remove(0);
            if chunks.is_empty() {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "stable compiled resource publication has no allocation cohort",
                ));
            }
            match permit.publish(group) {
                Ok(lease) => {
                    resident_publications.insert(group_id.clone(), publications);
                    group_chunks.insert(group_id.clone(), chunks.clone());
                    for chunk_id in chunks {
                        chunk_groups
                            .entry(chunk_id)
                            .or_default()
                            .insert(group_id.clone());
                    }
                    admission.newly_assigned_group_ids.remove(&group_id);
                    drop(lease);
                }
                Err(error) => {
                    let mut unpublished = publications;
                    for (_, _, _, remaining_publications, _) in &staged {
                        unpublished.extend(remaining_publications.iter().cloned());
                    }
                    address_table
                        .clear_group(transfer, &unpublished)
                        .map_err(compiled_device_store_vulkan_error)?;
                    return Err(compiled_device_store_residency_error(error));
                }
            }
        }
        let arena = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?;
        self.instrumentation.record_upload(
            total_uploaded_bytes,
            u64::try_from(upload_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            arena.committed_byte_capacity,
        );
        Ok(())
    }
}
