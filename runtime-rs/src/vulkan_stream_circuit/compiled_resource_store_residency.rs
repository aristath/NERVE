const VULKAN_COMPILED_RESOURCE_TRANSFER_STAGING_SLOT_COUNT: usize = 2;

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
    if allowed_selector_ids.is_empty() || maximum_group_byte_count == 0 {
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
    let maximum_load_wave_group_count = selected
        .iter()
        .map(|selector| selector.encoding.selection_count_per_activation)
        .max()
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "compiled resource store has no selected load wave".to_string(),
            )
        })?;
    if maximum_load_wave_group_count == 0 {
        return Err(VulkanRuntimeResidencyPlanError(
            "compiled resource store selector load wave is empty".to_string(),
        ));
    }
    let components_by_scope =
        compiled_resource_store_components_by_scope(contract, allowed_selector_ids)?;
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
    let maximum_load_wave_payload_bytes = checked_residency_mul(
        maximum_group_byte_count,
        maximum_load_wave_group_count,
        "compiled resource maximum load-wave bytes",
    )?;
    let transfer_staging_slot_byte_capacity =
        maximum_load_wave_payload_bytes.max(address_table_device_bytes);
    let transfer_staging_device_bytes = checked_residency_mul(
        transfer_staging_slot_byte_capacity,
        VULKAN_COMPILED_RESOURCE_TRANSFER_STAGING_SLOT_COUNT,
        "compiled resource transfer staging bytes",
    )?;
    let addressable_resource_count = layout
        .addressable_slot_count_for_selectors(allowed_selector_ids)
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
