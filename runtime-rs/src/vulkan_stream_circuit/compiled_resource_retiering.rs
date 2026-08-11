#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanCompiledResourceRetieringReport {
    pub considered_group_count: usize,
    pub promoted_group_count: usize,
    pub demoted_group_count: usize,
    pub promoted_payload_bytes: usize,
    pub copied_payload_bytes: usize,
    pub device_selection_count: u64,
    pub host_visible_selection_count: u64,
    pub elapsed_ns: u64,
}

#[derive(Clone)]
struct VulkanCompiledResourceRetieringCandidate {
    group_id: String,
    selection_count: u64,
    publications: Vec<VulkanStableResourceAddressPublication>,
    allocations: Vec<Arc<VulkanStableResourceAllocation>>,
}

struct VulkanCompiledResourcePayloadExchange {
    payload_bytes: usize,
    original_host_payloads: Vec<Vec<u8>>,
}

struct VulkanCompiledResourceCohortExchange {
    device_group_id: String,
    host_group_id: String,
    device_chunks: BTreeSet<VulkanCompiledResourceAllocationCohort>,
    host_chunks: BTreeSet<VulkanCompiledResourceAllocationCohort>,
    device_blocks: BTreeSet<VulkanCompiledResourceAllocationCohort>,
    host_blocks: BTreeSet<VulkanCompiledResourceAllocationCohort>,
}

struct VulkanPreparedCompiledResourceRetieringExchange {
    cold_device: VulkanCompiledResourceRetieringCandidate,
    hot_host: VulkanCompiledResourceRetieringCandidate,
    cohort_exchange: VulkanCompiledResourceCohortExchange,
    payload_bytes: usize,
}

impl VulkanCompiledResourceRetieringCandidate {
    fn layout_signature(&self) -> Vec<usize> {
        self.allocations
            .iter()
            .map(|allocation| allocation.byte_count())
            .collect()
    }
}

const RETIERING_PAYLOAD_COPIES_PER_EXCHANGE: u64 = 2;

fn observed_accesses_repay_exchange(
    hot_selection_count: u64,
    cold_selection_count: u64,
) -> bool {
    hot_selection_count
        .checked_sub(cold_selection_count)
        .is_some_and(|selection_advantage| {
            selection_advantage > RETIERING_PAYLOAD_COPIES_PER_EXCHANGE
        })
}

impl VulkanCompiledResourceDeviceStore {
    pub fn retier_from_selection_telemetry(
        &self,
        device: &VulkanComputeDevice,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<VulkanCompiledResourceRetieringReport, VulkanCompiledResourceDeviceStoreError> {
        let Some(memory_plan) = &self.memory_plan else {
            return Ok(VulkanCompiledResourceRetieringReport::default());
        };
        if device.physical_device_id() != self.physical_device_id {
            return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "compiled resource retiering device {:?} differs from store physical device {:?}",
                device.physical_device_id(),
                self.physical_device_id
            )));
        }
        let current_counts = self.selection_counts_by_group(telemetry)?;
        let mut selection_history = self
            .retiering_last_selection_counts
            .lock()
            .map_err(|_| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource retiering selection history was poisoned",
                )
            })?;
        let interval_counts = current_counts
            .iter()
            .map(|(group_id, current)| {
                let previous = selection_history.get(group_id).copied().unwrap_or(0);
                current
                    .checked_sub(previous)
                    .map(|count| (group_id.clone(), count))
                    .ok_or_else(|| {
                        VulkanCompiledResourceDeviceStoreError::new(format!(
                            "compiled resource selection counter for group {group_id:?} regressed from {previous} to {current}",
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;

        let started = Instant::now();
        let _load = self.begin_load_operation()?;
        let _mutation = self.residency_mutation.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource residency mutation lock was poisoned",
            )
        })?;
        self.ensure_device_work_is_available()?;
        let mut address_state = self.address_state.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address state was poisoned",
            )
        })?;
        let mut memory_plan = memory_plan.lock().map_err(|_| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource memory plan was poisoned",
            )
        })?;
        if memory_plan.group_tiers.len() != address_state.publications.len()
            || memory_plan
                .group_tiers
                .keys()
                .any(|group_id| !address_state.publications.contains_key(group_id))
        {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource tier assignments and resident address publications diverged",
            ));
        }

        let mut device_candidates = BTreeMap::<
            Vec<usize>,
            Vec<VulkanCompiledResourceRetieringCandidate>,
        >::new();
        let mut host_candidates = BTreeMap::<
            Vec<usize>,
            Vec<VulkanCompiledResourceRetieringCandidate>,
        >::new();
        for (group_id, tier) in &memory_plan.group_tiers {
            if address_state
                .promoted_representations
                .contains_key(group_id)
            {
                continue;
            }
            let publications = address_state
                .publications
                .get(group_id)
                .cloned()
                .expect("complete tiered publications were validated");
            let allocations = address_state
                .address_table
                .allocations_for_publications(&publications)
                .map_err(compiled_device_store_vulkan_error)?;
            let candidate = VulkanCompiledResourceRetieringCandidate {
                group_id: group_id.clone(),
                // Cumulative frequency is the stable working-set signal. Using only the
                // latest request makes equally shaped experts exchange tiers whenever
                // their rank changes by one access, even when the migration transfers
                // more bytes than the avoided host-visible reads.
                selection_count: current_counts.get(group_id).copied().unwrap_or(0),
                publications,
                allocations,
            };
            let signature = candidate.layout_signature();
            match tier {
                VulkanCompiledResourceMemoryTier::Device => {
                    device_candidates.entry(signature).or_default().push(candidate)
                }
                VulkanCompiledResourceMemoryTier::HostVisible => {
                    host_candidates.entry(signature).or_default().push(candidate)
                }
            }
        }
        let considered_group_count = memory_plan.group_tiers.len();
        let mut exchanges = Vec::new();
        for (signature, mut host) in host_candidates {
            let Some(mut resident) = device_candidates.remove(&signature) else {
                continue;
            };
            host.sort_by(|left, right| {
                right
                    .selection_count
                    .cmp(&left.selection_count)
                    .then_with(|| left.group_id.cmp(&right.group_id))
            });
            resident.sort_by(|left, right| {
                left.selection_count
                    .cmp(&right.selection_count)
                    .then_with(|| left.group_id.cmp(&right.group_id))
            });
            for (hot, cold) in host.into_iter().zip(resident) {
                // Exchanging equal layouts copies the cold payload to the host and the
                // hot payload to the device. Require the observed selection advantage
                // to repay both full-payload transfers before changing the stable
                // working set. This is a byte-traffic break-even rule, not a fixed
                // migration-count throttle.
                if !observed_accesses_repay_exchange(
                    hot.selection_count,
                    cold.selection_count,
                ) {
                    break;
                }
                exchanges.push((cold, hot));
            }
        }

        let prepared_exchanges = exchanges
            .into_iter()
            .map(|(cold_device, hot_host)| {
            memory_plan.validate_group_tier_exchange(
                &cold_device.group_id,
                &hot_host.group_id,
            )?;
            let cohort_exchange = prepare_compiled_resource_allocation_cohort_exchange(
                &address_state,
                &cold_device.group_id,
                &hot_host.group_id,
            )?;
            let payload_bytes = validate_compiled_resource_payload_exchange(
                &cold_device.allocations,
                &hot_host.allocations,
            )?;
            validate_compiled_resource_address_exchange(&cold_device, &hot_host)?;
            Ok(VulkanPreparedCompiledResourceRetieringExchange {
                cold_device,
                hot_host,
                cohort_exchange,
                payload_bytes,
            })
        })
        .collect::<Result<Vec<_>, VulkanCompiledResourceDeviceStoreError>>()?;
        let promoted_payload_bytes = prepared_exchanges.iter().try_fold(
            0usize,
            |total, exchange| {
                total.checked_add(exchange.payload_bytes).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource promoted payload byte count overflowed",
                    )
                })
            },
        )?;
        let copied_payload_bytes = promoted_payload_bytes.checked_mul(2).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource retiering copy byte count overflowed",
            )
        })?;
        // These counters describe where the completed interval actually executed.
        // Retiering happens after it, so attributing its accesses to the new tiers would
        // report a hypothetical future placement rather than measured traffic.
        let (device_selection_count, host_visible_selection_count) =
            compiled_resource_selection_counts_by_tier(&interval_counts, &memory_plan.group_tiers)?;
        let mut report = VulkanCompiledResourceRetieringReport {
            considered_group_count,
            promoted_group_count: prepared_exchanges.len(),
            demoted_group_count: prepared_exchanges.len(),
            promoted_payload_bytes,
            copied_payload_bytes,
            device_selection_count,
            host_visible_selection_count,
            ..VulkanCompiledResourceRetieringReport::default()
        };
        for exchange in prepared_exchanges {
            let VulkanPreparedCompiledResourceRetieringExchange {
                cold_device,
                hot_host,
                cohort_exchange,
                payload_bytes,
            } = exchange;
            let payload_exchange = exchange_compiled_resource_payloads(
                device,
                &mut address_state.transfer,
                &cold_device.allocations,
                &hot_host.allocations,
            )?;
            debug_assert_eq!(payload_exchange.payload_bytes, payload_bytes);

            #[cfg(test)]
            if self
                .fail_next_retiering_after_payload_exchange
                .swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                rollback_compiled_resource_payload_exchange(
                    &mut address_state.transfer,
                    &cold_device.allocations,
                    &hot_host.allocations,
                    &payload_exchange,
                )?;
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "injected compiled resource retiering failure after payload exchange",
                ));
            }

            let (hot_publications, cold_publications) = {
                let VulkanCompiledResourceDeviceAddressState {
                    transfer,
                    address_table,
                    ..
                } = &mut *address_state;
                match address_table.swap_groups(
                        transfer,
                        &hot_host.publications,
                        &cold_device.publications,
                    ) {
                    Ok(publications) => publications,
                    Err(exchange_error) => {
                        let rollback = rollback_compiled_resource_payload_exchange(
                            transfer,
                            &cold_device.allocations,
                            &hot_host.allocations,
                            &payload_exchange,
                        );
                        if let Err(rollback_error) = rollback {
                            let terminal = VulkanError(format!(
                                "compiled resource retiering address publication failed: {exchange_error}; payload rollback also failed: {rollback_error}"
                            ));
                            self.record_terminal_device_failure(&terminal)?;
                            return Err(VulkanCompiledResourceDeviceStoreError::new(
                                terminal.to_string(),
                            ));
                        }
                        return Err(compiled_device_store_vulkan_error(exchange_error));
                    }
                }
            };
            if let Err(exchange_error) = self.manager.transform_inactive_resident_groups(
                &cold_device.group_id,
                &hot_host.group_id,
                exchange_stable_compiled_resource_group_allocations,
            ) {
                let address_rollback = {
                    let VulkanCompiledResourceDeviceAddressState {
                        transfer,
                        address_table,
                        ..
                    } = &mut *address_state;
                    address_table.swap_groups(
                        transfer,
                        &hot_publications,
                        &cold_publications,
                    )
                };
                let payload_rollback = rollback_compiled_resource_payload_exchange(
                    &mut address_state.transfer,
                    &cold_device.allocations,
                    &hot_host.allocations,
                    &payload_exchange,
                );
                match (address_rollback, payload_rollback) {
                    (Ok((restored_hot, restored_cold)), Ok(())) => {
                        address_state.publications.insert(
                            hot_host.group_id.clone(),
                            restored_hot,
                        );
                        address_state.publications.insert(
                            cold_device.group_id.clone(),
                            restored_cold,
                        );
                        return Err(compiled_device_store_residency_error(
                            exchange_error,
                        ));
                    }
                    (address_result, payload_result) => {
                        let terminal = VulkanError(format!(
                            "compiled resource residency ownership exchange failed: {exchange_error}; address rollback: {}; payload rollback: {}",
                            address_result
                                .err()
                                .map_or_else(|| "ok".to_string(), |error| error.to_string()),
                            payload_result
                                .err()
                                .map_or_else(|| "ok".to_string(), |error| error.to_string()),
                        ));
                        self.record_terminal_device_failure(&terminal)?;
                        return Err(VulkanCompiledResourceDeviceStoreError::new(
                            terminal.to_string(),
                        ));
                    }
                }
            }
            address_state
                .publications
                .insert(hot_host.group_id.clone(), hot_publications);
            address_state
                .publications
                .insert(cold_device.group_id.clone(), cold_publications);
            commit_compiled_resource_allocation_cohort_exchange(
                &mut address_state,
                cohort_exchange,
            );
            memory_plan.commit_group_tier_exchange(
                &cold_device.group_id,
                &hot_host.group_id,
            );
        }
        report.elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.instrumentation.record_retiering(&report);
        *selection_history = current_counts;
        Ok(report)
    }

    fn selection_counts_by_group(
        &self,
        telemetry: &VulkanSelectionTelemetrySnapshot,
    ) -> Result<BTreeMap<String, u64>, VulkanCompiledResourceDeviceStoreError> {
        let mut counts = BTreeMap::<String, u64>::new();
        for domain in &telemetry.domains {
            let matching = self
                .contract
                .selectors
                .iter()
                .filter(|selector| {
                    self.allowed_selector_ids.contains(&selector.id)
                        && selector.execution_scope == domain.execution_scope
                        && selector.component_id == domain.component_id
                        && selector.node_id == domain.node_id
                        && selector.domain_id == domain.domain_id
                })
                .collect::<Vec<_>>();
            if matching.is_empty() {
                continue;
            }
            if matching.len() != 1 {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "selection telemetry domain {}.{}.{} ambiguously matches {} compiled selectors",
                    domain.component_id,
                    domain.node_id,
                    domain.domain_id,
                    matching.len()
                )));
            }
            let selector = matching[0];
            if domain.resource_count != selector.resource_count
                || domain.selection_counts.len() != selector.resource_count
            {
                return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                    "selection telemetry domain {}.{}.{} has {} counters for {} resources",
                    domain.component_id,
                    domain.node_id,
                    domain.domain_id,
                    domain.selection_counts.len(),
                    selector.resource_count
                )));
            }
            for (resource_index, selection_count) in
                domain.selection_counts.iter().copied().enumerate()
            {
                let group_id = self
                    .resolve_selector_resource(&selector.id, resource_index)?
                    .id()
                    .to_string();
                let total = counts.entry(group_id).or_default();
                *total = total.checked_add(selection_count).ok_or_else(|| {
                    VulkanCompiledResourceDeviceStoreError::new(
                        "compiled resource selection count overflowed",
                    )
                })?;
            }
        }
        Ok(counts)
    }
}

fn compiled_resource_selection_counts_by_tier(
    interval_counts: &BTreeMap<String, u64>,
    group_tiers: &BTreeMap<String, VulkanCompiledResourceMemoryTier>,
) -> Result<(u64, u64), VulkanCompiledResourceDeviceStoreError> {
    let mut device_selection_count = 0u64;
    let mut host_visible_selection_count = 0u64;
    for (group_id, tier) in group_tiers {
        let selection_count = interval_counts.get(group_id).copied().unwrap_or(0);
        let total = match tier {
            VulkanCompiledResourceMemoryTier::Device => &mut device_selection_count,
            VulkanCompiledResourceMemoryTier::HostVisible => {
                &mut host_visible_selection_count
            }
        };
        *total = total.checked_add(selection_count).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource tier selection count overflowed",
            )
        })?;
    }
    Ok((device_selection_count, host_visible_selection_count))
}

fn validate_compiled_resource_address_exchange(
    left: &VulkanCompiledResourceRetieringCandidate,
    right: &VulkanCompiledResourceRetieringCandidate,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    if left.publications.is_empty()
        || left.publications.len() != right.publications.len()
        || left.allocations.len() != left.publications.len()
        || right.allocations.len() != right.publications.len()
    {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource address exchange has incompatible group layouts",
        ));
    }
    let mut slots = BTreeSet::new();
    for publication in left.publications.iter().chain(&right.publications) {
        if !slots.insert(publication.slot()) {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address exchange repeats an address-table slot",
            ));
        }
        publication.generation().checked_add(1).ok_or_else(|| {
            VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address exchange exhausted a publication generation",
            )
        })?;
    }
    for (left_allocation, right_allocation) in left.allocations.iter().zip(&right.allocations) {
        if left_allocation.byte_count() != right_allocation.byte_count() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource address exchange has incompatible allocation byte counts",
            ));
        }
    }
    Ok(())
}

fn exchange_compiled_resource_payloads(
    device: &VulkanComputeDevice,
    transfer: &mut VulkanResidentTransferStream,
    device_allocations: &[Arc<VulkanStableResourceAllocation>],
    host_allocations: &[Arc<VulkanStableResourceAllocation>],
) -> Result<VulkanCompiledResourcePayloadExchange, VulkanCompiledResourceDeviceStoreError> {
    let payload_bytes =
        validate_compiled_resource_payload_exchange(device_allocations, host_allocations)?;
    let original_host_payloads = read_compiled_resource_payloads(host_allocations)?;
    let demotion_copies = device_allocations
        .iter()
        .zip(host_allocations)
        .map(|(device_allocation, host_allocation)| {
            VulkanResidentBufferRangeCopy::new(
                device_allocation.buffer(),
                host_allocation.buffer(),
                device_allocation.buffer_byte_offset(),
                host_allocation.buffer_byte_offset(),
                device_allocation.byte_count(),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(compiled_device_store_vulkan_error)?;
    if let Err(error) = device
        .create_resident_buffer_copy_batch(&demotion_copies)
        .and_then(|copy| copy.run())
    {
        restore_compiled_resource_host_payloads(host_allocations, &original_host_payloads)?;
        return Err(compiled_device_store_vulkan_error(error));
    }
    if let Err(promotion_error) = write_compiled_resource_device_payloads(
        transfer,
        device_allocations,
        &original_host_payloads,
    ) {
        let exchange = VulkanCompiledResourcePayloadExchange {
            payload_bytes,
            original_host_payloads,
        };
        let rollback = rollback_compiled_resource_payload_exchange(
            transfer,
            device_allocations,
            host_allocations,
            &exchange,
        );
        return match rollback {
            Ok(()) => Err(promotion_error),
            Err(rollback_error) => Err(VulkanCompiledResourceDeviceStoreError::new(format!(
                "{promotion_error}; payload rollback also failed: {rollback_error}"
            ))),
        };
    }
    Ok(VulkanCompiledResourcePayloadExchange {
        payload_bytes,
        original_host_payloads,
    })
}

fn validate_compiled_resource_payload_exchange(
    device_allocations: &[Arc<VulkanStableResourceAllocation>],
    host_allocations: &[Arc<VulkanStableResourceAllocation>],
) -> Result<usize, VulkanCompiledResourceDeviceStoreError> {
    if device_allocations.is_empty() || device_allocations.len() != host_allocations.len() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource retiering allocation layouts differ",
        ));
    }
    let mut payload_bytes = 0usize;
    for (device_allocation, host_allocation) in
        device_allocations.iter().zip(host_allocations)
    {
        if device_allocation.byte_count() != host_allocation.byte_count() {
            return Err(VulkanCompiledResourceDeviceStoreError::new(
                "compiled resource retiering allocation byte counts differ",
            ));
        }
        payload_bytes = payload_bytes
            .checked_add(device_allocation.byte_count())
            .ok_or_else(|| {
                VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource retiering payload byte count overflowed",
                )
            })?;
    }
    Ok(payload_bytes)
}

fn read_compiled_resource_payloads(
    allocations: &[Arc<VulkanStableResourceAllocation>],
) -> Result<Vec<Vec<u8>>, VulkanCompiledResourceDeviceStoreError> {
    allocations
        .iter()
        .map(|allocation| {
            allocation
                .buffer()
                .read_bytes_at(
                    allocation.buffer_byte_offset(),
                    allocation.byte_count(),
                )
                .map_err(compiled_device_store_vulkan_error)
        })
        .collect()
}

fn write_compiled_resource_device_payloads(
    transfer: &mut VulkanResidentTransferStream,
    allocations: &[Arc<VulkanStableResourceAllocation>],
    payloads: &[Vec<u8>],
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    if allocations.len() != payloads.len() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource payload write has mismatched allocation and payload counts",
        ));
    }
    let writes = allocations
        .iter()
        .zip(payloads)
        .map(|(allocation, payload)| {
            if allocation.byte_count() != payload.len() {
                return Err(VulkanError(
                    "compiled resource payload write has a mismatched byte count".to_string(),
                ));
            }
            VulkanResidentBufferWriteRange::new(
                allocation.buffer(),
                allocation.buffer_byte_offset(),
                payload,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(compiled_device_store_vulkan_error)?;
    let ticket = transfer
        .submit(&writes)
        .map_err(compiled_device_store_vulkan_error)?;
    transfer
        .wait(&ticket)
        .map_err(compiled_device_store_vulkan_error)?;
    Ok(())
}

fn restore_compiled_resource_host_payloads(
    allocations: &[Arc<VulkanStableResourceAllocation>],
    payloads: &[Vec<u8>],
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    if allocations.len() != payloads.len() {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource host restore has mismatched allocation and payload counts",
        ));
    }
    allocations
        .iter()
        .zip(payloads)
        .try_for_each(|(allocation, payload)| {
            if allocation.byte_count() != payload.len() {
                return Err(VulkanCompiledResourceDeviceStoreError::new(
                    "compiled resource host restore has a mismatched byte count",
                ));
            }
            allocation
                .buffer()
                .write_bytes_at(allocation.buffer_byte_offset(), payload)
                .map_err(compiled_device_store_vulkan_error)
        })
}

fn rollback_compiled_resource_payload_exchange(
    transfer: &mut VulkanResidentTransferStream,
    device_allocations: &[Arc<VulkanStableResourceAllocation>],
    host_allocations: &[Arc<VulkanStableResourceAllocation>],
    exchange: &VulkanCompiledResourcePayloadExchange,
) -> Result<(), VulkanCompiledResourceDeviceStoreError> {
    let original_device_payloads = read_compiled_resource_payloads(host_allocations)?;
    let device_result = write_compiled_resource_device_payloads(
        transfer,
        device_allocations,
        &original_device_payloads,
    );
    let host_result = restore_compiled_resource_host_payloads(
        host_allocations,
        &exchange.original_host_payloads,
    );
    match (device_result, host_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(device_error), Ok(())) => Err(device_error),
        (Ok(()), Err(host_error)) => Err(host_error),
        (Err(device_error), Err(host_error)) => Err(
            VulkanCompiledResourceDeviceStoreError::new(format!(
                "device payload rollback failed: {device_error}; host payload rollback failed: {host_error}"
            )),
        ),
    }
}

fn prepare_compiled_resource_allocation_cohort_exchange(
    state: &VulkanCompiledResourceDeviceAddressState,
    device_group_id: &str,
    host_group_id: &str,
) -> Result<VulkanCompiledResourceCohortExchange, VulkanCompiledResourceDeviceStoreError> {
    if device_group_id == host_group_id {
        return Err(VulkanCompiledResourceDeviceStoreError::new(
            "compiled resource cohort exchange names the same group twice",
        ));
    }
    let device_chunks = validated_compiled_resource_group_cohorts(
        device_group_id,
        &state.group_chunks,
        &state.chunk_groups,
        "device-tier slab",
    )?;
    let host_chunks = validated_compiled_resource_group_cohorts(
        host_group_id,
        &state.group_chunks,
        &state.chunk_groups,
        "host-tier slab",
    )?;
    let device_blocks = validated_compiled_resource_group_cohorts(
        device_group_id,
        &state.group_blocks,
        &state.block_groups,
        "device-tier allocation block",
    )?;
    let host_blocks = validated_compiled_resource_group_cohorts(
        host_group_id,
        &state.group_blocks,
        &state.block_groups,
        "host-tier allocation block",
    )?;
    Ok(VulkanCompiledResourceCohortExchange {
        device_group_id: device_group_id.to_string(),
        host_group_id: host_group_id.to_string(),
        device_chunks,
        host_chunks,
        device_blocks,
        host_blocks,
    })
}

fn validated_compiled_resource_group_cohorts(
    group_id: &str,
    group_cohorts: &BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    cohort_groups: &BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
    label: &str,
) -> Result<BTreeSet<VulkanCompiledResourceAllocationCohort>, VulkanCompiledResourceDeviceStoreError>
{
    let cohorts = group_cohorts.get(group_id).cloned().ok_or_else(|| {
        VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource {group_id:?} has no {label} cohort",
        ))
    })?;
    if cohorts.iter().any(|cohort| {
        !cohort_groups
            .get(cohort)
            .is_some_and(|groups| groups.contains(group_id))
    }) {
        return Err(VulkanCompiledResourceDeviceStoreError::new(format!(
            "compiled resource {group_id:?} has inconsistent {label} membership",
        )));
    }
    Ok(cohorts)
}

fn commit_compiled_resource_allocation_cohort_exchange(
    state: &mut VulkanCompiledResourceDeviceAddressState,
    exchange: VulkanCompiledResourceCohortExchange,
) {
    let VulkanCompiledResourceCohortExchange {
        device_group_id,
        host_group_id,
        device_chunks,
        host_chunks,
        device_blocks,
        host_blocks,
    } = exchange;
    swap_compiled_resource_group_cohorts(
        &device_group_id,
        &host_group_id,
        device_chunks,
        host_chunks,
        &mut state.group_chunks,
        &mut state.chunk_groups,
    );
    swap_compiled_resource_group_cohorts(
        &device_group_id,
        &host_group_id,
        device_blocks,
        host_blocks,
        &mut state.group_blocks,
        &mut state.block_groups,
    );
}

fn swap_compiled_resource_group_cohorts(
    device_group_id: &str,
    host_group_id: &str,
    device_cohorts: BTreeSet<VulkanCompiledResourceAllocationCohort>,
    host_cohorts: BTreeSet<VulkanCompiledResourceAllocationCohort>,
    group_cohorts: &mut BTreeMap<String, BTreeSet<VulkanCompiledResourceAllocationCohort>>,
    cohort_groups: &mut BTreeMap<VulkanCompiledResourceAllocationCohort, BTreeSet<String>>,
) {
    for cohort in &device_cohorts {
        if let Some(groups) = cohort_groups.get_mut(cohort) {
            groups.remove(device_group_id);
            groups.insert(host_group_id.to_string());
        }
    }
    for cohort in &host_cohorts {
        if let Some(groups) = cohort_groups.get_mut(cohort) {
            groups.remove(host_group_id);
            groups.insert(device_group_id.to_string());
        }
    }
    group_cohorts.insert(device_group_id.to_string(), host_cohorts);
    group_cohorts.insert(host_group_id.to_string(), device_cohorts);
}

#[cfg(test)]
mod retiering_policy_tests {
    use super::observed_accesses_repay_exchange;

    #[test]
    fn exchange_requires_more_avoided_payload_reads_than_payload_copies() {
        assert!(!observed_accesses_repay_exchange(10, 10));
        assert!(!observed_accesses_repay_exchange(11, 10));
        assert!(!observed_accesses_repay_exchange(12, 10));
        assert!(observed_accesses_repay_exchange(13, 10));
        assert!(!observed_accesses_repay_exchange(10, 13));
    }
}
