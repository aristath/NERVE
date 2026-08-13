#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanCompiledResourceRepresentationReport {
    pub considered_group_count: usize,
    pub promoted_group_count: usize,
    pub promoted_source_bytes: usize,
    pub promoted_resident_bytes: usize,
    pub skipped_unstable_load_interval: bool,
    pub skipped_capacity_bytes: usize,
    pub elapsed_ns: u64,
}

#[derive(Default)]
struct VulkanCompiledResourceRepresentationHistory {
    observed_successful_load_count: Option<u64>,
    reclaimed_since_boundary: bool,
}

struct VulkanCompiledResourcePromotedRepresentation {
    source_resources: Vec<(usize, Arc<VulkanStableResourceAllocation>)>,
    _resident_group: DeviceResidentResourceGroup<VulkanResidentCompiledResource>,
    resident_payload_bytes: usize,
    selection_count: u64,
}

#[derive(Clone)]
struct VulkanCompiledResourceRepresentationCandidate {
    group_id: String,
    selector_id: String,
    resource_index: usize,
    selection_count: u64,
}

fn require_complete_targeted_representation_preparation(
    report: &VulkanCompiledResourceRepresentationReport,
    expected_group_count: usize,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    if report.considered_group_count == expected_group_count
        && report.promoted_group_count == expected_group_count
        && !report.skipped_unstable_load_interval
        && report.skipped_capacity_bytes == 0
    {
        return Ok(());
    }
    Err(VulkanCompiledResourceDeviceStoreError::new(format!(
        "targeted representation preparation promoted {} of {} warmed groups (considered {}, skipped_capacity_bytes={}, unstable={}); measured execution is forbidden",
        report.promoted_group_count,
        expected_group_count,
        report.considered_group_count,
        report.skipped_capacity_bytes,
        report.skipped_unstable_load_interval,
    )))
}

impl VulkanCompiledResourceDeviceStore {
    fn supports_adaptive_representations(&self) -> bool {
        self.representation_arena.is_some() && self.representation_backing_store.is_some()
    }

    fn available_representation_store_device_bytes(
        &self,
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let source_committed_bytes = self
            .device_arena
            .stats()
            .map_err(compiled_device_store_vulkan_error)?
            .committed_byte_capacity;
        let representation_committed_bytes = self
            .representation_arena
            .as_ref()
            .map(VulkanStableResourceArena::stats)
            .transpose()
            .map_err(compiled_device_store_vulkan_error)?
            .map(|stats| stats.committed_byte_capacity)
            .unwrap_or_default();
        Ok(self.maximum_allocation_byte_capacity.saturating_sub(
            source_committed_bytes.saturating_add(representation_committed_bytes),
        ))
    }

    /// Materialize the resident derivation for every currently loaded resource
    /// group that can use one.
    ///
    /// Normal stream execution deliberately waits for a stable prompt boundary
    /// before adapting representations. A targeted optimizer session has no
    /// prompt stream, so its discarded warmup establishes that boundary
    /// explicitly. This method converts only groups that the warmup actually
    /// loaded; it never guesses a future expert working set or expands every
    /// resource admitted by the selector.
    pub fn prepare_loaded_representations_for_measurement(
        &self,
        device: &VulkanComputeDevice,
    ) -> Result<
        VulkanCompiledResourceRepresentationReport,
        VulkanCompiledResourceDeviceStoreError,
    > {
        if !self.supports_adaptive_representations() {
            return Ok(VulkanCompiledResourceRepresentationReport::default());
        }
        if device.physical_device_id() != self.physical_device_id {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled representation device {:?} differs from store physical device {:?}",
                device.physical_device_id(),
                self.physical_device_id,
            )));
        }

        let inactive_group_ids = self
            .manager
            .eviction_candidates(&BTreeSet::new())
            .map_err(compiled_device_store_residency_error)?
            .into_iter()
            .map(|candidate| candidate.group_id)
            .collect::<BTreeSet<_>>();
        let device_tier_group_ids = match &self.memory_plan {
            Some(memory_plan) => memory_plan
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned while preparing representations",
                    )
                })?
                .group_tiers
                .iter()
                .filter(|(_, tier)| **tier == VulkanCompiledResourceMemoryTier::Device)
                .map(|(group_id, _)| group_id.clone())
                .collect::<BTreeSet<_>>(),
            None => inactive_group_ids.clone(),
        };
        let promoted_group_ids = self
            .address_state
            .lock()
            .map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address state was poisoned while preparing representations",
                )
            })?
            .promoted_representations
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();

        let mut expected_group_ids = BTreeSet::new();
        let mut domains = Vec::new();
        for selector in self.contract.selectors.iter().filter(|selector| {
            self.allowed_selector_ids.contains(&selector.id)
        }) {
            let mut selection_counts = vec![0u64; selector.resource_count];
            for (resource_index, selection_count) in
                selection_counts.iter_mut().enumerate()
            {
                let resolved = self.resolve_selector_resource(
                    &selector.id,
                    resource_index,
                )?;
                let group_id = resolved.id();
                if inactive_group_ids.contains(group_id)
                    && device_tier_group_ids.contains(group_id)
                    && !promoted_group_ids.contains(group_id)
                    && resolved.has_resident_derivation()
                {
                    *selection_count = 1;
                    expected_group_ids.insert(group_id.to_string());
                }
            }
            if selection_counts.iter().any(|count| *count > 0) {
                let co_selection_count = selector
                    .resource_count
                    .checked_mul(selector.resource_count.saturating_sub(1))
                    .and_then(|count| count.checked_div(2))
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled representation co-selection counter count overflowed",
                        )
                    })?;
                domains.push(VulkanSelectionTelemetryDomainSnapshot {
                    execution_scope: selector.execution_scope.clone(),
                    component_id: selector.component_id.clone(),
                    node_id: selector.node_id.clone(),
                    domain_id: selector.domain_id.clone(),
                    resource_count: selector.resource_count,
                    selection_counts,
                    co_selection_counts: vec![0; co_selection_count],
                });
            }
        }
        if expected_group_ids.is_empty() {
            return Ok(VulkanCompiledResourceRepresentationReport::default());
        }

        let started = Instant::now();
        let telemetry = VulkanSelectionTelemetrySnapshot { domains };
        let first = self.optimize_representations_from_selection_telemetry(
            device,
            &telemetry,
        )?;
        let mut report = if first.skipped_unstable_load_interval {
            self.optimize_representations_from_selection_telemetry(
                device,
                &telemetry,
            )?
        } else {
            first
        };
        report.elapsed_ns =
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        require_complete_targeted_representation_preparation(
            &report,
            expected_group_ids.len(),
        )?;
        Ok(report)
    }

    pub fn optimize_representations_from_selection_telemetry(
        &self,
        device: &VulkanComputeDevice,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<
        VulkanCompiledResourceRepresentationReport,
        VulkanCompiledResourceDeviceStoreError,
    > {
        let (Some(arena), Some(backing_store)) = (
            self.representation_arena.as_ref(),
            self.representation_backing_store.as_ref(),
        ) else {
            return Ok(VulkanCompiledResourceRepresentationReport::default());
        };
        if device.physical_device_id() != self.physical_device_id {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled representation device {:?} differs from store physical device {:?}",
                device.physical_device_id(),
                self.physical_device_id,
            )));
        }

        let successful_load_count = self.statistics()?.successful_load_count;
        {
            let mut history = self.representation_history.lock().map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled representation history was poisoned",
                )
            })?;
            if history.reclaimed_since_boundary
                || history.observed_successful_load_count != Some(successful_load_count)
            {
                history.reclaimed_since_boundary = false;
                history.observed_successful_load_count = Some(successful_load_count);
                return Ok(VulkanCompiledResourceRepresentationReport {
                    skipped_unstable_load_interval: true,
                    ..Default::default()
                });
            }
        }

        let started = Instant::now();
        let selection_counts = self.selection_counts_by_group(telemetry)?;
        let inactive_group_ids = self
            .manager
            .eviction_candidates(&BTreeSet::new())
            .map_err(compiled_device_store_residency_error)?
            .into_iter()
            .map(|candidate| candidate.group_id)
            .collect::<BTreeSet<_>>();
        let device_tier_group_ids = match &self.memory_plan {
            Some(memory_plan) => memory_plan
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource memory plan was poisoned while selecting representations",
                    )
                })?
                .group_tiers
                .iter()
                .filter(|(_, tier)| **tier == VulkanCompiledResourceMemoryTier::Device)
                .map(|(group_id, _)| group_id.clone())
                .collect::<BTreeSet<_>>(),
            None => inactive_group_ids.clone(),
        };
        let promoted_group_ids = self
            .address_state
            .lock()
            .map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address state was poisoned",
                )
            })?
            .promoted_representations
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut candidates = selection_counts
            .iter()
            .filter(|(_, selection_count)| **selection_count > 0)
            .filter(|(group_id, _)| {
                inactive_group_ids.contains(*group_id)
                    && device_tier_group_ids.contains(*group_id)
                    && !promoted_group_ids.contains(*group_id)
            })
            .filter_map(|(group_id, selection_count)| {
                self.group_selections.get(group_id).map(
                    |(selector_id, resource_index)| {
                        VulkanCompiledResourceRepresentationCandidate {
                            group_id: group_id.clone(),
                            selector_id: selector_id.clone(),
                            resource_index: *resource_index,
                            selection_count: *selection_count,
                        }
                    },
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .selection_count
                .cmp(&left.selection_count)
                .then_with(|| left.group_id.cmp(&right.group_id))
        });

        let mut report = VulkanCompiledResourceRepresentationReport {
            considered_group_count: candidates.len(),
            ..Default::default()
        };
        for candidate in candidates {
            let resolved = self.resolve_selector_resource(
                &candidate.selector_id,
                candidate.resource_index,
            )?;
            if resolved.id() != candidate.group_id || !resolved.has_resident_derivation() {
                continue;
            }
            let descriptor = DeviceResourceGroupDescriptor::from_resolved_representation(
                &resolved,
                CompiledResourceRepresentation::ResidentDerivation,
            )
            .map_err(compiled_device_store_residency_error)?;
            let resource_slots = self
                .layout
                .resource_slots_for_selection(
                    &candidate.selector_id,
                    candidate.resource_index,
                    &descriptor.resource_ids,
                )
                .map_err(|error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled representation address layout is invalid: {error}",
                    ))
                })?;
            let allocation_byte_counts = descriptor
                .resources
                .iter()
                .map(|resource| resource.byte_count)
                .collect::<Vec<_>>();
            let required_device_bytes = arena
                .additional_committed_byte_capacity_for_groups(
                    device,
                    &[(&resource_slots, &allocation_byte_counts)],
                    self.upload_alignment,
                )
                .map_err(compiled_device_store_vulkan_error)?;
            let globally_available_device_bytes = usize::try_from(
                device
                    .device_local_memory_accounting()
                    .map_err(compiled_device_store_vulkan_error)?
                    .admissible_remaining_bytes,
            )
            .unwrap_or(usize::MAX);
            let arena_stats = arena
                .stats()
                .map_err(compiled_device_store_vulkan_error)?;
            let arena_available_device_bytes = arena
                .config()
                .committed_byte_capacity
                .saturating_sub(arena_stats.committed_byte_capacity);
            let store_available_device_bytes =
                self.available_representation_store_device_bytes()?;
            if required_device_bytes
                > globally_available_device_bytes
                    .min(arena_available_device_bytes)
                    .min(store_available_device_bytes)
            {
                report.skipped_capacity_bytes = report
                    .skipped_capacity_bytes
                    .saturating_add(required_device_bytes);
                continue;
            }
            let capacity_permit = (required_device_bytes > 0)
                .then(|| device.reserve_device_local_memory_capacity(required_device_bytes))
                .transpose()
                .map_err(compiled_device_store_vulkan_error)?;
            let ticket = backing_store
                .try_load_representation(
                    resolved,
                    CompiledResourceRepresentation::ResidentDerivation,
                )
                .map_err(|error| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "compiled representation backing-store load failed: {error}",
                    ))
                })?;
            let loaded = ticket.wait().map_err(|error| {
                VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled representation backing-store load failed: {error}",
                ))
            })?;

            let _load = self.begin_load_operation()?;
            let _mutation = self.residency_mutation.lock().map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource residency mutation lock was poisoned",
                )
            })?;
            self.ensure_device_work_is_available()?;
            let still_inactive = self
                .manager
                .eviction_candidates(&BTreeSet::new())
                .map_err(compiled_device_store_residency_error)?
                .iter()
                .any(|entry| entry.group_id == candidate.group_id);
            if !still_inactive {
                drop(capacity_permit);
                continue;
            }
            let mut state = self.address_state.lock().map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource address state was poisoned",
                )
            })?;
            let still_device_tier = match &self.memory_plan {
                Some(memory_plan) => memory_plan
                    .lock()
                    .map_err(|_| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled resource memory plan was poisoned while committing a representation",
                        )
                    })?
                    .tier_for_group(&candidate.group_id)?
                    == VulkanCompiledResourceMemoryTier::Device,
                None => true,
            };
            if !still_device_tier {
                drop(capacity_permit);
                continue;
            }
            if required_device_bytes
                > self.available_representation_store_device_bytes()?
            {
                drop(capacity_permit);
                report.skipped_capacity_bytes = report
                    .skipped_capacity_bytes
                    .saturating_add(required_device_bytes);
                continue;
            }
            if state
                .promoted_representations
                .contains_key(&candidate.group_id)
            {
                drop(capacity_permit);
                continue;
            }
            let current_publications = state
                .publications
                .get(&candidate.group_id)
                .cloned()
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "resident compiled resource {:?} has no address publications",
                        candidate.group_id,
                    ))
                })?;
            if current_publications.iter().any(|publication| {
                publication.representation()
                    != CompiledResourceRepresentation::Source.address_tag()
            }) {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "compiled resource {:?} has an untracked non-source representation",
                    candidate.group_id,
                )));
            }
            let source_allocations = state
                .address_table
                .allocations_for_publications(&current_publications)
                .map_err(compiled_device_store_vulkan_error)?;
            let source_resources = current_publications
                .iter()
                .map(VulkanStableResourceAddressPublication::slot)
                .zip(source_allocations)
                .collect::<Vec<_>>();
            let source_payload_bytes = source_resources.iter().try_fold(
                0usize,
                |total, (_, allocation)| {
                    total.checked_add(allocation.byte_count()).ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(
                            "compiled representation source byte count overflowed",
                        )
                    })
                },
            )?;
            let resident_payload_bytes = descriptor.byte_count;
            let promoted_group_count = report
                .promoted_group_count
                .checked_add(1)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation promoted group count overflowed",
                    )
                })?;
            let promoted_source_bytes = report
                .promoted_source_bytes
                .checked_add(source_payload_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation promoted source byte count overflowed",
                    )
                })?;
            let promoted_resident_bytes = report
                .promoted_resident_bytes
                .checked_add(resident_payload_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation promoted resident byte count overflowed",
                    )
                })?;
            let upload = {
                let VulkanCompiledResourceDeviceAddressState {
                    transfer,
                    address_table,
                    ..
                } = &mut *state;
                replace_loaded_compiled_resource_group_in_stable_address_space(
                    device,
                    transfer,
                    arena,
                    address_table,
                    &current_publications,
                    &descriptor,
                    &loaded,
                    &resource_slots,
                    self.upload_alignment,
                    capacity_permit,
                )
                .map_err(compiled_device_store_vulkan_error)?
            };
            let (resident_group, publications) = upload.into_parts();
            state
                .publications
                .insert(candidate.group_id.clone(), publications);
            if state
                .promoted_representations
                .insert(
                    candidate.group_id.clone(),
                    VulkanCompiledResourcePromotedRepresentation {
                        source_resources,
                        _resident_group: resident_group,
                        resident_payload_bytes,
                        selection_count: candidate.selection_count,
                    },
                )
                .is_some()
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled representation promotion replaced an existing cache entry",
                ));
            }
            report.promoted_group_count = promoted_group_count;
            report.promoted_source_bytes = promoted_source_bytes;
            report.promoted_resident_bytes = promoted_resident_bytes;
        }
        report.elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(report)
    }

    fn restore_promoted_representations_locked(
        &self,
        state: &mut VulkanCompiledResourceDeviceAddressState,
        group_ids: &[String],
    ) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
        let mut restored_resident_bytes = 0usize;
        for group_id in group_ids {
            let Some(promoted) = state.promoted_representations.get(group_id) else {
                continue;
            };
            let current_publications = state
                .publications
                .get(group_id)
                .cloned()
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(format!(
                        "promoted compiled resource {group_id:?} has no address publications",
                    ))
                })?;
            let replacements = promoted
                .source_resources
                .iter()
                .map(|(slot, allocation)| {
                    (
                        *slot,
                        Arc::clone(allocation),
                        CompiledResourceRepresentation::Source.address_tag(),
                    )
                })
                .collect::<Vec<_>>();
            let publications = {
                let VulkanCompiledResourceDeviceAddressState {
                    transfer,
                    address_table,
                    ..
                } = &mut *state;
                address_table
                    .replace_group(transfer, &current_publications, &replacements)
                    .map_err(compiled_device_store_vulkan_error)?
            };
            state.publications.insert(group_id.clone(), publications);
            let promoted = state
                .promoted_representations
                .remove(group_id)
                .expect("promoted representation existed through atomic restoration");
            restored_resident_bytes = restored_resident_bytes
                .checked_add(promoted.resident_payload_bytes)
                .ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation restored byte count overflowed",
                    )
                })?;
            drop(promoted);
        }
        Ok(restored_resident_bytes)
    }

    fn prepare_promoted_representation_reclamation(
        &self,
        requested_bytes: usize,
    ) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
        let Some(arena) = &self.representation_arena else {
            return Ok(());
        };
        if arena
            .inactive_backing_byte_capacity()
            .map_err(compiled_device_store_vulkan_error)?
            >= requested_bytes
        {
            return Ok(());
        }
        let mut state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let mut candidates = state
            .promoted_representations
            .iter()
            .map(|(group_id, promoted)| {
                (
                    promoted.selection_count,
                    group_id.clone(),
                    promoted.resident_payload_bytes,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            (left.0, left.1.as_str()).cmp(&(right.0, right.1.as_str()))
        });
        // Representation allocations share retained slabs. Restoring only one
        // logical group cannot make a slab physically reclaimable when a
        // sibling remains, so a pressure boundary restores the full derived
        // cache as one physical cohort.
        let selected = candidates
            .into_iter()
            .map(|(_, group_id, _)| group_id)
            .collect::<Vec<_>>();
        self.restore_promoted_representations_locked(&mut state, &selected)?;
        drop(state);
        if !selected.is_empty() {
            self.representation_history
                .lock()
                .map_err(|_| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled representation history was poisoned during reclamation",
                    )
                })?
                .reclaimed_since_boundary = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod targeted_representation_preparation_tests {
    use super::*;

    #[test]
    fn targeted_representation_preparation_rejects_partial_promotion() {
        let report = VulkanCompiledResourceRepresentationReport {
            considered_group_count: 2,
            promoted_group_count: 1,
            promoted_source_bytes: 64,
            promoted_resident_bytes: 128,
            skipped_unstable_load_interval: false,
            skipped_capacity_bytes: 128,
            elapsed_ns: 7,
        };
        let error = require_complete_targeted_representation_preparation(
            &report,
            2,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("promoted 1 of 2 warmed groups"),
            "{error}",
        );
        assert!(
            error.to_string().contains("measured execution is forbidden"),
            "{error}",
        );
    }

    #[test]
    fn targeted_representation_preparation_accepts_only_the_complete_wave() {
        let report = VulkanCompiledResourceRepresentationReport {
            considered_group_count: 2,
            promoted_group_count: 2,
            promoted_source_bytes: 128,
            promoted_resident_bytes: 256,
            skipped_unstable_load_interval: false,
            skipped_capacity_bytes: 0,
            elapsed_ns: 7,
        };
        require_complete_targeted_representation_preparation(&report, 2)
            .unwrap();
    }
}
