const VULKAN_COMPILED_RESOURCE_TRANSFER_STAGING_SLOT_COUNT: usize = 2;
const VULKAN_COMPILED_RESOURCE_SLAB_HEAP_FRACTION: u64 = 32;
const VULKAN_COMPILED_RESOURCE_MINIMUM_SLAB_BYTES: usize = 16 * 1024 * 1024;
const VULKAN_COMPILED_RESOURCE_MAXIMUM_SLAB_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct VulkanCompiledResourceStoreResidencyBytes {
    pub address_table_device_bytes: usize,
    pub parameter_slot_table_device_bytes: usize,
    pub metadata_device_bytes: usize,
    pub transfer_staging_slot_count: usize,
    pub transfer_staging_slot_byte_capacity: usize,
    pub transfer_staging_device_bytes: usize,
    pub maximum_load_wave_group_count: usize,
    pub maximum_load_wave_payload_bytes: usize,
    pub maximum_dynamic_allocation_padding_bytes: usize,
    pub maximum_representation_group_payload_bytes: usize,
    pub maximum_representation_load_wave_group_count: usize,
    pub maximum_representation_load_wave_payload_bytes: usize,
    pub maximum_representation_load_wave_allocation_padding_bytes: usize,
    pub retained_representation_cache_group_count: usize,
    pub retained_representation_cache_wave_count: usize,
    pub retained_representation_cache_payload_bytes: usize,
    pub retained_representation_cache_allocation_padding_bytes: usize,
    pub retained_representation_cache_identity: Option<String>,
}

impl VulkanCompiledResourceStoreResidencyBytes {
    pub fn fixed_device_bytes(&self) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.metadata_device_bytes,
            self.transfer_staging_device_bytes,
            "compiled resource store fixed device bytes",
        )
    }

    pub fn maximum_source_extra_device_bytes(
        &self,
    ) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.fixed_device_bytes()?,
            self.maximum_dynamic_allocation_padding_bytes,
            "compiled resource store maximum source extra device bytes",
        )
    }

    pub fn retained_representation_cache_device_bytes(
        &self,
    ) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.retained_representation_cache_payload_bytes,
            self.retained_representation_cache_allocation_padding_bytes,
            "compiled resource store retained representation cache device bytes",
        )
    }

    pub fn maximum_extra_device_bytes(
        &self,
    ) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.maximum_source_extra_device_bytes()?,
            self.retained_representation_cache_device_bytes()?,
            "compiled resource store maximum extra device bytes",
        )
    }
}

fn compiled_resource_source_payload_capacity(
    maximum_source_payload_bytes: usize,
    available_dynamic_device_bytes: usize,
    store: &VulkanCompiledResourceStoreResidencyBytes,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    let reserved_non_source_payload_bytes = checked_residency_add(
        store.maximum_dynamic_allocation_padding_bytes,
        store.retained_representation_cache_device_bytes()?,
        "compiled resource non-source-payload reservation",
    )?;
    Ok(maximum_source_payload_bytes.min(
        available_dynamic_device_bytes.saturating_sub(reserved_non_source_payload_bytes),
    ))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CompiledResourceRepresentationGroupResidencyBytes {
    payload_bytes: usize,
    resource_count: usize,
}

fn compiled_resource_representation_group_residency_bytes(
    contract: &CompiledResourceResidencyContract,
    index: &CompiledResourceContractIndex,
    selector: &CompiledResourceSelector,
    resource_index: usize,
) -> Result<CompiledResourceRepresentationGroupResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let (has_derivation, payload_bytes, resource_count) = match &selector.mapping {
        CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
            let group_id = atomic_group_ids.get(resource_index).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "compiled selector {:?} omits resource index {resource_index}",
                    selector.id,
                ))
            })?;
            let group = index.atomic_group(contract, group_id).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "compiled selector {:?} references missing atomic group {group_id:?}",
                    selector.id,
                ))
            })?;
            let resources = group
                .resource_ids
                .iter()
                .map(|resource_id| {
                    index.resource(contract, resource_id).ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "compiled atomic group {group_id:?} references missing resource {resource_id:?}",
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let has_derivation = resources
                .iter()
                .any(|resource| resource.resident_derivation.is_some());
            let payload_bytes = resources.iter().try_fold(0usize, |total, resource| {
                let bytes = resource
                    .resident_byte_count_for(
                        CompiledResourceRepresentation::ResidentDerivation,
                    )
                    .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
                checked_residency_add(
                    total,
                    bytes,
                    "compiled atomic-group representation bytes",
                )
            })?;
            (has_derivation, payload_bytes, resources.len())
        }
        CompiledResourceSelectorMapping::PartitionTemplate {
            partition_template_id,
        } => {
            if resource_index >= selector.resource_count {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "compiled selector {:?} resource index {resource_index} exceeds its {} resources",
                    selector.id, selector.resource_count,
                )));
            }
            let template = index
                .partition_template(contract, partition_template_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "compiled selector {:?} references missing partition template {partition_template_id:?}",
                        selector.id,
                    ))
                })?;
            if resource_index >= template.partition_count {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "compiled selector {:?} resource index {resource_index} exceeds partition template count {}",
                    selector.id, template.partition_count,
                )));
            }
            let has_derivation = template
                .member_templates
                .iter()
                .any(|member| member.resident_derivation.is_some());
            let payload_bytes = template.member_templates.iter().try_fold(
                0usize,
                |total, member| {
                    let bytes = member
                        .resident_byte_count_for(
                            CompiledResourceRepresentation::ResidentDerivation,
                        )
                        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
                    checked_residency_add(
                        total,
                        bytes,
                        "compiled partition representation bytes",
                    )
                },
            )?;
            (has_derivation, payload_bytes, template.member_templates.len())
        }
    };
    Ok(if has_derivation {
        CompiledResourceRepresentationGroupResidencyBytes {
            payload_bytes,
            resource_count,
        }
    } else {
        CompiledResourceRepresentationGroupResidencyBytes::default()
    })
}

fn compiled_resource_stable_slab_payload_bytes(
    device: &VulkanComputeDevice,
    arena_byte_capacity: usize,
    maximum_load_wave_payload_bytes: usize,
    alignment: usize,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    compiled_resource_stable_slab_payload_bytes_for_heap(
        device.device_local_memory_bytes(),
        arena_byte_capacity,
        maximum_load_wave_payload_bytes,
        alignment,
    )
}

fn compiled_resource_stable_slab_payload_bytes_for_heap(
    device_local_memory_bytes: u64,
    arena_byte_capacity: usize,
    maximum_load_wave_payload_bytes: usize,
    alignment: usize,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    if arena_byte_capacity == 0
        || maximum_load_wave_payload_bytes == 0
        || !alignment.is_power_of_two()
    {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource stable-slab inputs are invalid".to_string(),
        ));
    }
    let heap_scaled = usize::try_from(
        device_local_memory_bytes / VULKAN_COMPILED_RESOURCE_SLAB_HEAP_FRACTION,
    )
    .unwrap_or(usize::MAX);
    let desired = heap_scaled
        .max(VULKAN_COMPILED_RESOURCE_MINIMUM_SLAB_BYTES)
        .min(VULKAN_COMPILED_RESOURCE_MAXIMUM_SLAB_BYTES)
        .max(maximum_load_wave_payload_bytes)
        .min(arena_byte_capacity);
    let aligned = desired - (desired % alignment);
    if aligned < maximum_load_wave_payload_bytes {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled resource arena admits {arena_byte_capacity} bytes but one aligned load wave needs {maximum_load_wave_payload_bytes} bytes",
        )));
    }
    Ok(aligned)
}

fn compiled_resource_contract_minimum_upload_alignment(
    contract: &CompiledResourceResidencyContract,
) -> Result<usize, VulkanRuntimeResidencyPlanError> {
    let alignment = contract
        .resources
        .iter()
        .flat_map(|resource| resource.ranges.iter().map(|range| range.alignment_bytes))
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
        .fold(std::mem::align_of::<u64>(), usize::max);
    if !alignment.is_power_of_two() {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled resource upload alignment {alignment} is not a power of two"
        )));
    }
    Ok(alignment)
}

fn plan_compiled_resource_store_residency(
    contract: &CompiledResourceResidencyContract,
    layout: &VulkanCompiledResourceAddressLayout,
    allowed_selector_ids: &BTreeSet<String>,
    maximum_group_byte_count: usize,
    upload_alignment: usize,
) -> Result<VulkanCompiledResourceStoreResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let ownership = VulkanCompiledResourceSelectorOwnership::all(contract, allowed_selector_ids)?;
    plan_compiled_resource_store_residency_for_ownership(
        contract,
        layout,
        &ownership,
        maximum_group_byte_count,
        upload_alignment,
    )
}

fn plan_compiled_resource_store_residency_for_ownership(
    contract: &CompiledResourceResidencyContract,
    layout: &VulkanCompiledResourceAddressLayout,
    ownership: &VulkanCompiledResourceSelectorOwnership,
    maximum_group_byte_count: usize,
    upload_alignment: usize,
) -> Result<VulkanCompiledResourceStoreResidencyBytes, VulkanRuntimeResidencyPlanError> {
    let allowed_selector_ids = ownership.selector_ids();
    if maximum_group_byte_count == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource store residency requires selectors and a nonempty atomic group"
                .to_string(),
        ));
    }
    if !upload_alignment.is_power_of_two() {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled resource store upload alignment {upload_alignment} is not a power of two"
        )));
    }
    let selected = contract
        .selectors
        .iter()
        .filter(|selector| allowed_selector_ids.contains(&selector.id))
        .collect::<Vec<_>>();
    if selected.len() != allowed_selector_ids.len() {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource store selector ownership references an unknown selector"
                .to_string(),
        ));
    }
    let source_payload_bytes_by_slot = layout
        .source_payload_bytes_by_address_slot_for_ownership(contract, ownership)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let contract_index = CompiledResourceContractIndex::new(contract)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let mut maximum_load_wave_group_count = 0usize;
    let mut maximum_load_wave_payload_bytes = 0usize;
    let mut observed_maximum_group_bytes = 0usize;
    let mut maximum_representation_group_payload_bytes = 0usize;
    let mut maximum_representation_load_wave_group_count = 0usize;
    let mut maximum_representation_load_wave_payload_bytes = 0usize;
    let mut maximum_representation_load_wave_resource_count = 0usize;
    let mut retained_representation_cache_group_count = 0usize;
    let mut retained_representation_cache_wave_count = 0usize;
    let mut retained_representation_cache_payload_bytes = 0usize;
    let mut retained_representation_cache_resource_count = 0usize;
    let mut retained_representation_cache_selector_ids = BTreeSet::new();
    for selector in &selected {
        let selector_layout = layout
            .selectors
            .iter()
            .find(|layout| layout.selector_id == selector.id)
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "compiled resource store selector {:?} has no address layout",
                    selector.id,
                ))
            })?;
        let mut owned_group_payload_bytes = ownership
            .resources(&selector.id)
            .into_iter()
            .flatten()
            .map(|resource_index| {
                selector_layout
                    .mapping
                    .resource_slots(*resource_index)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "compiled resource store selector {:?} resource {resource_index} has no address slots",
                            selector.id,
                        ))
                    })?
                    .into_iter()
                    .try_fold(0usize, |total, slot| {
                        let bytes = source_payload_bytes_by_slot.get(&slot).ok_or_else(|| {
                            VulkanRuntimeResidencyPlanError(format!(
                                "compiled resource store selector {:?} address slot {slot} has no source payload",
                                selector.id,
                            ))
                        })?;
                        checked_residency_add(
                            total,
                            *bytes,
                            "compiled resource selector group bytes",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        owned_group_payload_bytes.sort_unstable_by(|left, right| right.cmp(left));
        let selection_count = selector
            .encoding
            .selection_count_per_activation
            .min(owned_group_payload_bytes.len());
        maximum_load_wave_group_count = maximum_load_wave_group_count.max(selection_count);
        let selector_load_wave_bytes = owned_group_payload_bytes
            .iter()
            .take(selection_count)
            .try_fold(0usize, |total, bytes| {
                checked_residency_add(
                    total,
                    *bytes,
                    "compiled resource maximum load-wave bytes",
                )
            })?;
        maximum_load_wave_payload_bytes =
            maximum_load_wave_payload_bytes.max(selector_load_wave_bytes);
        observed_maximum_group_bytes = observed_maximum_group_bytes.max(
            owned_group_payload_bytes.first().copied().unwrap_or_default(),
        );

        let mut representation_groups = ownership
            .resources(&selector.id)
            .into_iter()
            .flatten()
            .map(|resource_index| {
                compiled_resource_representation_group_residency_bytes(
                    contract,
                    &contract_index,
                    selector,
                    *resource_index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        representation_groups.sort_unstable_by(|left, right| {
            right
                .payload_bytes
                .cmp(&left.payload_bytes)
                .then_with(|| right.resource_count.cmp(&left.resource_count))
        });
        maximum_representation_group_payload_bytes =
            maximum_representation_group_payload_bytes.max(
                representation_groups
                    .first()
                    .map(|group| group.payload_bytes)
                    .unwrap_or_default(),
            );
        let selected_representation_groups = representation_groups
            .iter()
            .filter(|group| group.payload_bytes > 0)
            .take(selection_count)
            .collect::<Vec<_>>();
        let selector_representation_payload_bytes = selected_representation_groups
            .iter()
            .try_fold(0usize, |total, group| {
                checked_residency_add(
                    total,
                    group.payload_bytes,
                    "compiled resource representation load-wave bytes",
                )
            })?;
        let selector_representation_resource_count = selected_representation_groups
            .iter()
            .try_fold(0usize, |total, group| {
                checked_residency_add(
                    total,
                    group.resource_count,
                    "compiled resource representation load-wave resource count",
                )
            })?;
        maximum_representation_load_wave_group_count =
            maximum_representation_load_wave_group_count
                .max(selected_representation_groups.len());
        maximum_representation_load_wave_payload_bytes =
            maximum_representation_load_wave_payload_bytes
                .max(selector_representation_payload_bytes);
        maximum_representation_load_wave_resource_count =
            maximum_representation_load_wave_resource_count
                .max(selector_representation_resource_count);
        retained_representation_cache_group_count = checked_residency_add(
            retained_representation_cache_group_count,
            selected_representation_groups.len(),
            "compiled resource retained representation group count",
        )?;
        retained_representation_cache_payload_bytes = checked_residency_add(
            retained_representation_cache_payload_bytes,
            selector_representation_payload_bytes,
            "compiled resource retained representation payload bytes",
        )?;
        retained_representation_cache_resource_count = checked_residency_add(
            retained_representation_cache_resource_count,
            selector_representation_resource_count,
            "compiled resource retained representation resource count",
        )?;
        if !selected_representation_groups.is_empty() {
            retained_representation_cache_wave_count = checked_residency_add(
                retained_representation_cache_wave_count,
                1,
                "compiled resource retained representation wave count",
            )?;
            retained_representation_cache_selector_ids.insert(selector.id.clone());
        }
    }
    if maximum_load_wave_group_count == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource store selector load wave is empty".to_string(),
        ));
    }
    if observed_maximum_group_bytes > maximum_group_byte_count {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled resource store observes a {observed_maximum_group_bytes}-byte group but residency admits only {maximum_group_byte_count} bytes",
        )));
    }
    let components_by_scope =
        compiled_resource_store_components_by_scope(contract, &allowed_selector_ids)?;
    let parameter_slot_table_device_bytes = components_by_scope.iter().try_fold(
        0usize,
        |total, (scope, component_ids)| {
            let bytes = layout
                .parameter_slot_table_byte_count_for_components(scope, component_ids)
                .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
            checked_residency_add(
                total,
                bytes,
                "compiled resource parameter-slot metadata bytes",
            )
        },
    )?;
    let address_table_device_bytes = layout
        .address_table_byte_count()
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let metadata_device_bytes = checked_residency_add(
        address_table_device_bytes,
        parameter_slot_table_device_bytes,
        "compiled resource metadata bytes",
    )?;
    let transfer_staging_slot_byte_capacity = maximum_load_wave_payload_bytes
        .max(maximum_representation_group_payload_bytes)
        .max(address_table_device_bytes);
    let transfer_staging_device_bytes = checked_residency_mul(
        transfer_staging_slot_byte_capacity,
        VULKAN_COMPILED_RESOURCE_TRANSFER_STAGING_SLOT_COUNT,
        "compiled resource transfer staging bytes",
    )?;
    let addressable_resource_count = layout
        .addressable_slot_count_for_ownership(ownership)
        .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
    let maximum_dynamic_allocation_padding_bytes = checked_residency_mul(
        addressable_resource_count,
        upload_alignment.saturating_sub(1),
        "compiled resource maximum allocation padding",
    )?;
    let maximum_representation_load_wave_allocation_padding_bytes = checked_residency_mul(
        checked_residency_add(
            maximum_representation_load_wave_resource_count,
            maximum_representation_load_wave_group_count,
            "compiled resource representation wave allocation count",
        )?,
        upload_alignment.saturating_sub(1),
        "compiled resource maximum representation load-wave allocation padding",
    )?;
    let retained_representation_cache_allocation_padding_bytes = checked_residency_mul(
        checked_residency_add(
            retained_representation_cache_resource_count,
            retained_representation_cache_group_count,
            "compiled resource retained representation allocation count",
        )?,
        upload_alignment.saturating_sub(1),
        "compiled resource retained representation cache allocation padding",
    )?;
    let retained_representation_cache_identity =
        (!retained_representation_cache_selector_ids.is_empty()).then(|| {
            let mut digest = Sha256::new();
            digest.update(b"nerve.compiled_resource_representation_cache.v1");
            for selector_id in retained_representation_cache_selector_ids {
                digest.update((selector_id.len() as u128).to_le_bytes());
                digest.update(selector_id.as_bytes());
            }
            for value in [
                retained_representation_cache_group_count,
                retained_representation_cache_wave_count,
                retained_representation_cache_payload_bytes,
                retained_representation_cache_allocation_padding_bytes,
            ] {
                digest.update((value as u128).to_le_bytes());
            }
            format!("sha256:{:x}", digest.finalize())
        });
    Ok(VulkanCompiledResourceStoreResidencyBytes {
        address_table_device_bytes,
        parameter_slot_table_device_bytes,
        metadata_device_bytes,
        transfer_staging_slot_count: VULKAN_COMPILED_RESOURCE_TRANSFER_STAGING_SLOT_COUNT,
        transfer_staging_slot_byte_capacity,
        transfer_staging_device_bytes,
        maximum_load_wave_group_count,
        maximum_load_wave_payload_bytes,
        maximum_dynamic_allocation_padding_bytes,
        maximum_representation_group_payload_bytes,
        maximum_representation_load_wave_group_count,
        maximum_representation_load_wave_payload_bytes,
        maximum_representation_load_wave_allocation_padding_bytes,
        retained_representation_cache_group_count,
        retained_representation_cache_wave_count,
        retained_representation_cache_payload_bytes,
        retained_representation_cache_allocation_padding_bytes,
        retained_representation_cache_identity,
    })
}

fn compiled_resource_store_components_by_scope(
    contract: &CompiledResourceResidencyContract,
    allowed_selector_ids: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeSet<String>>, VulkanRuntimeResidencyPlanError> {
    let mut components_by_scope = BTreeMap::<String, BTreeSet<String>>::new();
    let mut found = 0usize;
    for selector in contract
        .selectors
        .iter()
        .filter(|selector| allowed_selector_ids.contains(&selector.id))
    {
        found += 1;
        components_by_scope
            .entry(selector.execution_scope.clone())
            .or_default()
            .insert(selector.component_id.clone());
    }
    if found != allowed_selector_ids.len() {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource store selector ownership references an unknown selector"
                .to_string(),
        ));
    }
    Ok(components_by_scope)
}
