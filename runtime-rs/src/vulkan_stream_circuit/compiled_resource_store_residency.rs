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
}

impl VulkanCompiledResourceStoreResidencyBytes {
    pub fn fixed_device_bytes(&self) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.metadata_device_bytes,
            self.transfer_staging_device_bytes,
            "compiled resource store fixed device bytes",
        )
    }

    pub fn maximum_extra_device_bytes(
        &self,
    ) -> Result<usize, VulkanRuntimeResidencyPlanError> {
        checked_residency_add(
            self.fixed_device_bytes()?,
            self.maximum_dynamic_allocation_padding_bytes,
            "compiled resource store maximum extra device bytes",
        )
    }
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
    let mut maximum_load_wave_group_count = 0usize;
    let mut maximum_load_wave_payload_bytes = 0usize;
    let mut observed_maximum_group_bytes = 0usize;
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
    let transfer_staging_slot_byte_capacity =
        maximum_load_wave_payload_bytes.max(address_table_device_bytes);
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
