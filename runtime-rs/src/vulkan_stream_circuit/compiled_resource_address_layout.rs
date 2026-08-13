#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledSelectorAddressLayout {
    pub selector_id: String,
    pub execution_scope: String,
    pub component_id: String,
    pub node_id: String,
    pub selection_signal: String,
    pub mapping: VulkanCompiledSelectorAddressMapping,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanCompiledSelectorAddressMapping {
    GroupTable {
        resource_address_slots: Vec<usize>,
        resource_address_slot_offsets: Vec<usize>,
    },
    PartitionTemplate {
        member_slot_bases: Vec<usize>,
        member_resource_identity_seeds: Vec<String>,
        resource_count: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledParameterSlotTable {
    pub key: VulkanDynamicResourceBindingKey,
    pub selector_id: String,
    pub execution_scope: String,
    pub parameter_ids: Vec<String>,
    pub resource_count: usize,
    pub mapping: VulkanCompiledParameterSlotMapping,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanCompiledParameterSlotMapping {
    Partitioned { parameter_slot_bases: Vec<usize> },
    Explicit { parameter_slots: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceAddressLayout {
    slot_count: usize,
    concrete_resource_slots: BTreeMap<String, usize>,
    pub selectors: Vec<VulkanCompiledSelectorAddressLayout>,
    pub parameter_slot_tables: Vec<VulkanCompiledParameterSlotTable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceAddressLayoutError(pub String);

impl Display for VulkanCompiledResourceAddressLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanCompiledResourceAddressLayoutError {}

impl VulkanCompiledSelectorAddressMapping {
    pub fn resource_count(&self) -> usize {
        match self {
            Self::GroupTable {
                resource_address_slot_offsets,
                ..
            } => resource_address_slot_offsets.len().saturating_sub(1),
            Self::PartitionTemplate { resource_count, .. } => *resource_count,
        }
    }

    pub fn maximum_resource_member_count(&self) -> usize {
        match self {
            Self::GroupTable {
                resource_address_slot_offsets,
                ..
            } => resource_address_slot_offsets
                .windows(2)
                .map(|bounds| bounds[1].saturating_sub(bounds[0]))
                .max()
                .unwrap_or(0),
            Self::PartitionTemplate {
                member_slot_bases,
                ..
            } => member_slot_bases.len(),
        }
    }

    pub fn resource_slots(
        &self,
        resource_index: usize,
    ) -> Option<Vec<usize>> {
        match self {
            Self::GroupTable {
                resource_address_slots,
                resource_address_slot_offsets,
            } => {
                let next_index = resource_index.checked_add(1)?;
                let bounds = resource_address_slot_offsets
                    .get(resource_index..=next_index)?;
                Some(
                    resource_address_slots
                        .get(bounds[0]..bounds[1])?
                        .to_vec(),
                )
            }
            Self::PartitionTemplate {
                member_slot_bases,
                resource_count,
                ..
            } if resource_index < *resource_count => Some(
                member_slot_bases
                    .iter()
                    .map(|base| base.checked_add(resource_index))
                    .collect::<Option<Vec<_>>>()?,
            ),
            Self::PartitionTemplate { .. } => None,
        }
    }

    pub fn slot_ranges(
        &self,
    ) -> Result<Vec<(usize, usize)>, VulkanCompiledResourceAddressLayoutError>
    {
        match self {
            Self::GroupTable {
                resource_address_slots,
                ..
            } => resource_address_slots
                .iter()
                .map(|slot| {
                    slot.checked_add(1)
                        .map(|end| (*slot, end))
                        .ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(
                                "compiled concrete address slot range overflowed"
                                    .to_string(),
                            )
                        })
                })
                .collect(),
            Self::PartitionTemplate {
                member_slot_bases,
                resource_count,
                ..
            } => member_slot_bases
                .iter()
                .map(|base| {
                    base.checked_add(*resource_count)
                        .map(|end| (*base, end))
                        .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(
                            "compiled partition address slot range overflowed"
                                .to_string(),
                        )
                    })
                })
                .collect(),
        }
    }

    pub fn overlaps(
        &self,
        other: &Self,
    ) -> Result<bool, VulkanCompiledResourceAddressLayoutError> {
        let left = self.slot_ranges()?;
        let right = other.slot_ranges()?;
        for left_range in &left {
            if right.iter().any(|right_range| {
                left_range.0 < right_range.1
                    && right_range.0 < left_range.1
            }) {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl VulkanCompiledParameterSlotTable {
    pub fn slot_count(&self) -> Option<usize> {
        match &self.mapping {
            VulkanCompiledParameterSlotMapping::Partitioned {
                parameter_slot_bases,
            } => self.resource_count.checked_mul(parameter_slot_bases.len()),
            VulkanCompiledParameterSlotMapping::Explicit { parameter_slots } => {
                Some(parameter_slots.len())
            }
        }
    }

    pub fn slots(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        match &self.mapping {
            VulkanCompiledParameterSlotMapping::Partitioned {
                parameter_slot_bases,
            } => Box::new((0..self.resource_count).flat_map(
                move |resource_index| {
                    parameter_slot_bases
                        .iter()
                        .map(move |base| base + resource_index)
                },
            )),
            VulkanCompiledParameterSlotMapping::Explicit { parameter_slots } => {
                Box::new(parameter_slots.iter().copied())
            }
        }
    }
}

impl VulkanCompiledResourceAddressLayout {
    pub fn from_contract(
        contract: &CompiledResourceResidencyContract,
    ) -> Result<Self, VulkanCompiledResourceAddressLayoutError> {
        let contract_index = CompiledResourceContractIndex::new(contract).map_err(|error| {
            VulkanCompiledResourceAddressLayoutError(format!(
                "compiled resource contract index is invalid: {error}"
            ))
        })?;
        let mut concrete_slots = BTreeMap::new();
        let mut slot_count = 0usize;
        for resource in contract.resources.iter().filter(|resource| {
            resource.lifetime == CompiledResourceLifetime::Dynamic
        }) {
            let slot = slot_count;
            slot_count = slot_count.checked_add(1).ok_or_else(|| {
                VulkanCompiledResourceAddressLayoutError(
                    "compiled concrete address slot count overflowed".to_string(),
                )
            })?;
            concrete_slots.insert(resource.id.clone(), slot);
        }

        let mut partition_slots = BTreeMap::new();
        for template in &contract.partition_templates {
            for member in &template.member_templates {
                let member_slot_base = slot_count;
                slot_count = slot_count
                    .checked_add(template.partition_count)
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(
                            "compiled partition address slot count overflowed"
                                .to_string(),
                        )
                    })?;
                partition_slots.insert(
                    (
                        template.id.clone(),
                        member.resource_identity_seed.clone(),
                    ),
                    member_slot_base,
                );
            }
        }
        if slot_count == 0 && !contract.selectors.is_empty() {
            return Err(VulkanCompiledResourceAddressLayoutError(
                "compiled selectors have no dynamic address slots".to_string(),
            ));
        }

        let mut selectors = Vec::with_capacity(contract.selectors.len());
        for selector in &contract.selectors {
            let mapping = match &selector.mapping {
                CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids,
                } => {
                    let mut resource_address_slots = Vec::new();
                    let mut resource_address_slot_offsets =
                        Vec::with_capacity(atomic_group_ids.len() + 1);
                    resource_address_slot_offsets.push(0);
                    for group_id in atomic_group_ids {
                        let group = contract_index
                            .atomic_group(contract, group_id)
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(
                                    format!(
                                        "selector {:?} maps missing atomic group {group_id:?}",
                                        selector.id
                                    ),
                                )
                            })?;
                        let slots = group
                            .resource_ids
                            .iter()
                            .map(|resource_id| {
                                concrete_slots
                                    .get(resource_id)
                                    .copied()
                                    .ok_or_else(|| {
                                        VulkanCompiledResourceAddressLayoutError(
                                            format!(
                                                "dynamic group {group_id:?} resource {resource_id:?} has no address slot"
                                            ),
                                        )
                                    })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        resource_address_slots.extend(slots);
                        resource_address_slot_offsets
                            .push(resource_address_slots.len());
                    }
                    VulkanCompiledSelectorAddressMapping::GroupTable {
                        resource_address_slots,
                        resource_address_slot_offsets,
                    }
                }
                CompiledResourceSelectorMapping::PartitionTemplate {
                    partition_template_id,
                } => {
                    let template = contract_index
                        .partition_template(contract, partition_template_id)
                        .ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(format!(
                                "selector {:?} maps missing partition template {partition_template_id:?}",
                                selector.id
                            ))
                        })?;
                    let member_slot_bases = template
                        .member_templates
                        .iter()
                        .map(|member| {
                            partition_slots
                                .get(&(
                                    template.id.clone(),
                                    member.resource_identity_seed.clone(),
                                ))
                                .copied()
                                .ok_or_else(|| {
                                    VulkanCompiledResourceAddressLayoutError(
                                        "compiled partition address slot is missing"
                                            .to_string(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    VulkanCompiledSelectorAddressMapping::PartitionTemplate {
                        member_slot_bases,
                        member_resource_identity_seeds: template
                            .member_templates
                            .iter()
                            .map(|member| {
                                member.resource_identity_seed.clone()
                            })
                            .collect(),
                        resource_count: template.partition_count,
                    }
                }
            };
            if mapping.resource_count() != selector.resource_count
                || mapping.maximum_resource_member_count() == 0
            {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "selector {:?} has an incomplete resource address layout",
                    selector.id
                )));
            }
            selectors.push(VulkanCompiledSelectorAddressLayout {
                selector_id: selector.id.clone(),
                execution_scope: selector.execution_scope.clone(),
                component_id: selector.component_id.clone(),
                node_id: selector.node_id.clone(),
                selection_signal: selector.selection_signal.clone(),
                mapping,
            });
        }

        let mut partition_bindings: BTreeMap<
            (String, String, String, String, String),
            Vec<(&str, &str, usize)>,
        > = BTreeMap::new();
        for binding in &contract.bindings {
            if let CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id,
                resource_identity_seed,
                selection_signal,
                parameter_slot,
            } = &binding.mapping
            {
                partition_bindings
                    .entry((
                        binding.execution_scope.clone(),
                        binding.component_id.clone(),
                        binding.node_id.clone(),
                        partition_template_id.clone(),
                        selection_signal.clone(),
                    ))
                    .or_default()
                    .push((
                        binding.parameter_id.as_str(),
                        resource_identity_seed.as_str(),
                        *parameter_slot,
                    ));
            }
        }

        let mut parameter_slot_tables = Vec::new();
        for ((scope, component_id, node_id, template_id, selection_signal), bindings) in
            partition_bindings
        {
            let matching_selectors = contract
                .selectors
                .iter()
                .filter(|selector| {
                    selector.execution_scope == scope
                        && selector.component_id == component_id
                        && selector.selection_signal == selection_signal
                        && matches!(
                            &selector.mapping,
                            CompiledResourceSelectorMapping::PartitionTemplate {
                                partition_template_id
                            } if *partition_template_id == template_id
                        )
                })
                .collect::<Vec<_>>();
            if matching_selectors.len() != 1 {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "{scope} {component_id}.{node_id} dynamic parameters do not map exactly one selector"
                )));
            }
            let selector = matching_selectors[0];
            let mut parameter_ids = vec![None; bindings.len()];
            let mut parameter_slot_bases = vec![None; bindings.len()];
            for (parameter_id, seed, parameter_slot) in bindings {
                if parameter_slot >= parameter_ids.len()
                    || parameter_ids[parameter_slot]
                        .replace(parameter_id.to_string())
                        .is_some()
                {
                    return Err(VulkanCompiledResourceAddressLayoutError(
                        format!(
                            "{scope} {component_id}.{node_id} repeats or skips a selected parameter slot"
                        ),
                    ));
                }
                parameter_slot_bases[parameter_slot] = Some(
                    partition_slots
                        .get(&(template_id.clone(), seed.to_string()))
                        .copied()
                        .ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(format!(
                                "{scope} {component_id}.{node_id} selected parameter has no address slot"
                            ))
                        })?,
                );
            }
            if parameter_ids.iter().any(Option::is_none)
                || parameter_slot_bases.iter().any(Option::is_none)
            {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "{scope} {component_id}.{node_id} selected parameter slots are incomplete"
                )));
            }
            parameter_slot_tables.push(VulkanCompiledParameterSlotTable {
                key: VulkanDynamicResourceBindingKey::new(
                    component_id,
                    node_id,
                    &selection_signal,
                ),
                selector_id: selector.id.clone(),
                execution_scope: scope,
                parameter_ids: parameter_ids.into_iter().flatten().collect(),
                resource_count: selector.resource_count,
                mapping: VulkanCompiledParameterSlotMapping::Partitioned {
                    parameter_slot_bases: parameter_slot_bases
                        .into_iter()
                        .flatten()
                        .collect(),
                },
            });
        }

        let mut independent_bindings: BTreeMap<
            (String, String, String, String),
            Vec<(&str, &str, &str, usize, usize)>,
        > = BTreeMap::new();
        for binding in &contract.bindings {
            if let CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id,
                resource_id,
                selection_signal,
                selector_index,
                parameter_slot,
            } = &binding.mapping
            {
                independent_bindings
                    .entry((
                        binding.execution_scope.clone(),
                        binding.component_id.clone(),
                        binding.node_id.clone(),
                        selection_signal.clone(),
                    ))
                    .or_default()
                    .push((
                        binding.parameter_id.as_str(),
                        atomic_group_id.as_str(),
                        resource_id.as_str(),
                        *selector_index,
                        *parameter_slot,
                    ));
            }
        }
        for ((scope, component_id, node_id, selection_signal), bindings) in
            independent_bindings
        {
            let matching_selectors = contract
                .selectors
                .iter()
                .filter(|selector| {
                    selector.execution_scope == scope
                        && selector.component_id == component_id
                        && selector.selection_signal == selection_signal
                        && matches!(
                            &selector.mapping,
                            CompiledResourceSelectorMapping::GroupTable { .. }
                        )
                })
                .collect::<Vec<_>>();
            if matching_selectors.len() != 1 {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "{scope} {component_id}.{node_id} independent parameters do not map exactly one selector"
                )));
            }
            let selector = matching_selectors[0];
            let CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } =
                &selector.mapping
            else {
                unreachable!("matching selector was restricted to group tables")
            };
            let parameters_per_resource = bindings
                .iter()
                .map(|binding| binding.4)
                .max()
                .and_then(|maximum| maximum.checked_add(1))
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} has no independent parameter slots"
                    ))
                })?;
            let table_len = selector
                .resource_count
                .checked_mul(parameters_per_resource)
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} independent parameter table overflowed"
                    ))
                })?;
            let mut parameter_slots = vec![None; table_len];
            let mut parameter_ids = vec![None; parameters_per_resource];
            for (parameter_id, atomic_group_id, resource_id, selector_index, parameter_slot) in
                bindings
            {
                if atomic_group_ids.get(selector_index).map(String::as_str)
                    != Some(atomic_group_id)
                    || parameter_slot >= parameters_per_resource
                {
                    return Err(VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} independent parameter binding disagrees with its selector"
                    )));
                }
                let address_slot = concrete_slots.get(resource_id).copied().ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} independent parameter has no address slot"
                    ))
                })?;
                let table_index = selector_index * parameters_per_resource + parameter_slot;
                if parameter_slots[table_index].replace(address_slot).is_some() {
                    return Err(VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} repeats an independent parameter slot"
                    )));
                }
                if selector_index == 0 {
                    parameter_ids[parameter_slot] = Some(parameter_id.to_string());
                }
            }
            if parameter_slots.iter().any(Option::is_none)
                || parameter_ids.iter().any(Option::is_none)
            {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "{scope} {component_id}.{node_id} independent parameter slots are incomplete"
                )));
            }
            parameter_slot_tables.push(VulkanCompiledParameterSlotTable {
                key: VulkanDynamicResourceBindingKey::new(
                    component_id,
                    node_id,
                    selection_signal,
                ),
                selector_id: selector.id.clone(),
                execution_scope: scope,
                parameter_ids: parameter_ids.into_iter().flatten().collect(),
                resource_count: selector.resource_count,
                mapping: VulkanCompiledParameterSlotMapping::Explicit {
                    parameter_slots: parameter_slots.into_iter().flatten().collect(),
                },
            });
        }
        parameter_slot_tables.sort_by(|left, right| left.key.cmp(&right.key));
        if parameter_slot_tables
            .windows(2)
            .any(|pair| pair[0].key == pair[1].key)
        {
            return Err(VulkanCompiledResourceAddressLayoutError(
                "compiled dynamic parameter-slot keys are not unique".to_string(),
            ));
        }

        Ok(Self {
            slot_count,
            concrete_resource_slots: concrete_slots,
            selectors,
            parameter_slot_tables,
        })
    }

    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    pub fn metadata_byte_count_for_components(
        &self,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
    ) -> Result<usize, VulkanCompiledResourceAddressLayoutError> {
        let address_table_bytes = self.address_table_byte_count()?;
        self.parameter_slot_table_byte_count_for_components(
            execution_scope,
            component_ids,
        )?
        .checked_add(address_table_bytes)
        .ok_or_else(|| {
            VulkanCompiledResourceAddressLayoutError(
                "compiled resource metadata byte count overflowed".to_string(),
            )
        })
    }

    pub fn address_table_byte_count(
        &self,
    ) -> Result<usize, VulkanCompiledResourceAddressLayoutError> {
        self.slot_count().checked_mul(32).ok_or_else(|| {
            VulkanCompiledResourceAddressLayoutError(
                "compiled resource address-table byte count overflowed".to_string(),
            )
        })
    }

    pub fn parameter_slot_table_byte_count_for_components(
        &self,
        execution_scope: &str,
        component_ids: &BTreeSet<String>,
    ) -> Result<usize, VulkanCompiledResourceAddressLayoutError> {
        self.parameter_slot_tables
            .iter()
            .filter(|table| {
                table.execution_scope == execution_scope
                    && component_ids.contains(&table.key.component_id)
            })
            .try_fold(0usize, |total, table| {
                table
                    .slot_count()
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(
                            "compiled resource parameter-slot count overflowed"
                                .to_string(),
                        )
                    })?
                    .checked_mul(size_of::<u32>())
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(
                            "compiled resource metadata byte count overflowed"
                                .to_string(),
                        )
                    })
            })
    }

    pub fn addressable_slot_count_for_selectors(
        &self,
        selector_ids: &BTreeSet<String>,
    ) -> Result<usize, VulkanCompiledResourceAddressLayoutError> {
        let mut ranges = self
            .selectors
            .iter()
            .filter(|selector| selector_ids.contains(&selector.selector_id))
            .map(|selector| selector.mapping.slot_ranges())
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        let mut total = 0usize;
        let mut current: Option<(usize, usize)> = None;
        for range in ranges {
            if range.0 >= range.1 {
                return Err(VulkanCompiledResourceAddressLayoutError(
                    "compiled selector address range is empty".to_string(),
                ));
            }
            match &mut current {
                Some(current) if range.0 <= current.1 => {
                    current.1 = current.1.max(range.1);
                }
                Some(current) => {
                    total = total
                        .checked_add(current.1 - current.0)
                        .ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(
                                "compiled selector address count overflowed"
                                    .to_string(),
                            )
                        })?;
                    *current = range;
                }
                None => current = Some(range),
            }
        }
        if let Some(current) = current {
            total = total
                .checked_add(current.1 - current.0)
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(
                        "compiled selector address count overflowed".to_string(),
                    )
                })?;
        }
        Ok(total)
    }

    pub fn addressable_slot_count_for_ownership(
        &self,
        ownership: &VulkanCompiledResourceSelectorOwnership,
    ) -> Result<usize, VulkanCompiledResourceAddressLayoutError> {
        Ok(self.addressable_slots_for_ownership(ownership)?.len())
    }

    pub fn addressable_slots_for_ownership(
        &self,
        ownership: &VulkanCompiledResourceSelectorOwnership,
    ) -> Result<BTreeSet<usize>, VulkanCompiledResourceAddressLayoutError> {
        let mut slots = BTreeSet::new();
        for (selector_id, resource_indices) in ownership.iter() {
            let selector = self
                .selectors
                .iter()
                .find(|selector| selector.selector_id == selector_id)
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "compiled ownership selector {selector_id:?} has no address layout"
                    ))
                })?;
            for resource_index in resource_indices {
                let resource_slots = selector
                    .mapping
                    .resource_slots(*resource_index)
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(format!(
                            "compiled ownership selector {selector_id:?} resource {resource_index} has no address slots"
                        ))
                    })?;
                slots.extend(resource_slots);
            }
        }
        Ok(slots)
    }

    /// Returns the mandatory source-representation payload behind every
    /// addressable slot owned by one physical store. Resident derivations are
    /// adaptive secondary allocations: they can replace a published address,
    /// but the source bytes remain the authoritative cache payload needed to
    /// construct or restore that representation.
    pub fn source_payload_bytes_by_address_slot_for_ownership(
        &self,
        contract: &CompiledResourceResidencyContract,
        ownership: &VulkanCompiledResourceSelectorOwnership,
    ) -> Result<BTreeMap<usize, usize>, VulkanCompiledResourceAddressLayoutError> {
        let contract_index = CompiledResourceContractIndex::new(contract).map_err(|error| {
            VulkanCompiledResourceAddressLayoutError(error.to_string())
        })?;
        let mut payload_by_slot = BTreeMap::new();
        for (selector_id, resource_indices) in ownership.iter() {
            let selector = contract
                .selectors
                .iter()
                .find(|selector| selector.id == selector_id)
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "compiled ownership selector {selector_id:?} has no contract",
                    ))
                })?;
            let layout = self
                .selectors
                .iter()
                .find(|layout| layout.selector_id == selector_id)
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "compiled ownership selector {selector_id:?} has no address layout",
                    ))
                })?;
            for resource_index in resource_indices {
                let slots = layout.mapping.resource_slots(*resource_index).ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "compiled ownership selector {selector_id:?} resource {resource_index} has no address slots",
                    ))
                })?;
                let payloads = match &selector.mapping {
                    CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                        let group_id = atomic_group_ids.get(*resource_index).ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(format!(
                                "compiled ownership selector {selector_id:?} resource {resource_index} has no atomic group",
                            ))
                        })?;
                        let group = contract_index.atomic_group(contract, group_id).ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(format!(
                                "compiled ownership selector {selector_id:?} references missing group {group_id:?}",
                            ))
                        })?;
                        let projection = ownership.source_projection(selector_id, *resource_index);
                        group
                            .resource_ids
                            .iter()
                            .map(|resource_id| {
                                if let Some(projection) = projection {
                                    projection
                                        .resources
                                        .get(resource_id)
                                        .map(|resource| resource.byte_count)
                                        .ok_or_else(|| {
                                            VulkanCompiledResourceAddressLayoutError(format!(
                                                "compiled ownership projection omits resource {resource_id:?}",
                                            ))
                                        })
                                } else {
                                    let resource = contract_index
                                        .resource(contract, resource_id)
                                        .ok_or_else(|| {
                                            VulkanCompiledResourceAddressLayoutError(format!(
                                                "compiled ownership group references missing resource {resource_id:?}",
                                            ))
                                        })?;
                                    compiled_resource_bytes(resource).map_err(|error| {
                                        VulkanCompiledResourceAddressLayoutError(error.to_string())
                                    })
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    CompiledResourceSelectorMapping::PartitionTemplate {
                        partition_template_id,
                    } => {
                        if ownership.source_projection(selector_id, *resource_index).is_some() {
                            return Err(VulkanCompiledResourceAddressLayoutError(format!(
                                "compiled partition selector {selector_id:?} cannot use a source projection",
                            )));
                        }
                        let template = contract_index
                            .partition_template(contract, partition_template_id)
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(format!(
                                    "compiled ownership selector {selector_id:?} references missing template {partition_template_id:?}",
                                ))
                            })?;
                        template
                            .member_templates
                            .iter()
                            .map(|member| {
                                member.range_templates.iter().try_fold(
                                    0usize,
                                    |total, range| {
                                        total.checked_add(range.byte_count).ok_or_else(|| {
                                            VulkanCompiledResourceAddressLayoutError(
                                                "compiled partition payload bytes overflowed"
                                                    .to_string(),
                                            )
                                        })
                                    },
                                )
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                };
                if slots.len() != payloads.len() {
                    return Err(VulkanCompiledResourceAddressLayoutError(format!(
                        "compiled ownership selector {selector_id:?} resource {resource_index} has {} slots but {} payloads",
                        slots.len(), payloads.len(),
                    )));
                }
                for (slot, bytes) in slots.into_iter().zip(payloads) {
                    if bytes == 0 {
                        return Err(VulkanCompiledResourceAddressLayoutError(
                            "compiled ownership payload slot is empty".to_string(),
                        ));
                    }
                    if let Some(previous) = payload_by_slot.insert(slot, bytes)
                        && previous != bytes
                    {
                        return Err(VulkanCompiledResourceAddressLayoutError(format!(
                            "compiled ownership address slot {slot} changes from {previous} to {bytes} bytes",
                        )));
                    }
                }
            }
        }
        Ok(payload_by_slot)
    }

    pub fn selector(
        &self,
        execution_scope: &str,
        component_id: &str,
        node_id: &str,
    ) -> Option<&VulkanCompiledSelectorAddressLayout> {
        self.selectors.iter().find(|selector| {
            selector.execution_scope == execution_scope
                && selector.component_id == component_id
                && selector.node_id == node_id
        })
    }

    pub fn parameter_slot_table(
        &self,
        key: &VulkanDynamicResourceBindingKey,
    ) -> Option<&VulkanCompiledParameterSlotTable> {
        self.parameter_slot_tables
            .binary_search_by(|table| table.key.cmp(key))
            .ok()
            .map(|index| &self.parameter_slot_tables[index])
    }

    pub fn resource_slots_for_ids(
        &self,
        resource_ids: &[String],
    ) -> Result<Vec<usize>, VulkanCompiledResourceAddressLayoutError> {
        resource_ids
            .iter()
            .map(|resource_id| {
                self.concrete_resource_slots
                    .get(resource_id)
                    .copied()
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(format!(
                            "compiled resource {resource_id:?} has no stable address slot"
                        ))
                    })
            })
            .collect()
    }

    pub fn resource_slots_for_selector_index(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Result<Vec<usize>, VulkanCompiledResourceAddressLayoutError> {
        self.selectors
            .iter()
            .find(|selector| selector.selector_id == selector_id)
            .and_then(|selector| {
                selector.mapping.resource_slots(resource_index)
            })
            .ok_or_else(|| {
                VulkanCompiledResourceAddressLayoutError(format!(
                    "compiled selector {selector_id:?} resource {resource_index} has no stable address slots"
                ))
            })
    }

    pub fn resource_slots_for_selection(
        &self,
        selector_id: &str,
        resource_index: usize,
        resource_ids: &[String],
    ) -> Result<Vec<usize>, VulkanCompiledResourceAddressLayoutError> {
        let selector = self
            .selectors
            .iter()
            .find(|selector| selector.selector_id == selector_id)
            .ok_or_else(|| {
                VulkanCompiledResourceAddressLayoutError(format!(
                    "compiled selector {selector_id:?} has no stable address layout"
                ))
            })?;
        match &selector.mapping {
            VulkanCompiledSelectorAddressMapping::GroupTable { .. } => {
                let slots = self.resource_slots_for_ids(resource_ids)?;
                let declared = selector
                    .mapping
                    .resource_slots(resource_index)
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(format!(
                            "compiled selector {selector_id:?} resource {resource_index} is absent"
                        ))
                    })?;
                let mut sorted_slots = slots.clone();
                sorted_slots.sort_unstable();
                let mut sorted_declared = declared;
                sorted_declared.sort_unstable();
                if sorted_slots != sorted_declared {
                    return Err(VulkanCompiledResourceAddressLayoutError(
                        "compiled concrete resource identities differ from their selector mapping"
                            .to_string(),
                    ));
                }
                Ok(slots)
            }
            VulkanCompiledSelectorAddressMapping::PartitionTemplate {
                member_slot_bases,
                member_resource_identity_seeds,
                resource_count,
            } => {
                if resource_index >= *resource_count
                    || resource_ids.len() != member_slot_bases.len()
                    || member_slot_bases.len()
                        != member_resource_identity_seeds.len()
                {
                    return Err(VulkanCompiledResourceAddressLayoutError(
                        "compiled partition selection differs from its address layout"
                            .to_string(),
                    ));
                }
                resource_ids
                    .iter()
                    .map(|resource_id| {
                        let member_index = member_resource_identity_seeds
                            .iter()
                            .position(|seed| {
                                derived_partition_resource_id(
                                    seed,
                                    resource_index,
                                )
                                .is_ok_and(|derived| {
                                    derived == *resource_id
                                })
                            })
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(
                                    "compiled partition resource identity has no address slot"
                                        .to_string(),
                                )
                            })?;
                        member_slot_bases[member_index]
                            .checked_add(resource_index)
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(
                                    "compiled partition address slot overflowed"
                                        .to_string(),
                                )
                            })
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod compiled_resource_address_layout_tests {
    use super::*;

    fn content_id(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn independent_selected_contract(
    ) -> CompiledResourceResidencyContract {
        let compatibility = CompiledResourceCompatibility {
            device_api: "vulkan".to_string(),
            storage_class: "storage_buffer".to_string(),
            read_only: true,
            required_features: Vec::new(),
        };
        let resources = ['1', '2', '3', '4']
            .into_iter()
            .map(|byte| CompiledImmutableResource {
                id: content_id(byte),
                lifetime: CompiledResourceLifetime::Dynamic,
                ranges: Vec::new(),
                dependencies: Vec::new(),
                compatibility: compatibility.clone(),
                resident_derivation: None,
            })
            .collect::<Vec<_>>();
        let groups = vec![
            CompiledAtomicResidencyGroup {
                id: content_id('5'),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: vec![resources[0].id.clone(), resources[1].id.clone()],
                dependencies: Vec::new(),
            },
            CompiledAtomicResidencyGroup {
                id: content_id('6'),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: vec![resources[2].id.clone(), resources[3].id.clone()],
                dependencies: Vec::new(),
            },
        ];
        let parameter_ids = [
            "expert_0_scale",
            "expert_0_weight",
            "expert_1_scale",
            "expert_1_weight",
        ];
        let bindings = parameter_ids
            .into_iter()
            .enumerate()
            .map(|(index, parameter_id)| {
                let selector_index = index / 2;
                let parameter_slot = index % 2;
                CompiledResourceBinding {
                    execution_scope: "target".to_string(),
                    component_id: "layer_00".to_string(),
                    node_id: "expert_compute".to_string(),
                    parameter_id: parameter_id.to_string(),
                    mapping:
                        CompiledResourceBindingMapping::SelectedAtomicGroup {
                            atomic_group_id: groups[selector_index].id.clone(),
                            resource_id: resources[index].id.clone(),
                            selection_signal: "selected_experts".to_string(),
                            selector_index,
                            parameter_slot,
                        },
                }
            })
            .collect();
        CompiledResourceResidencyContract {
            schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
            identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
            state_machine_schema:
                RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
            supported_policies: vec![
                ResourceResidencyPolicy::DemandRetained,
                ResourceResidencyPolicy::Eager,
            ],
            resources,
            atomic_groups: groups.clone(),
            partition_templates: Vec::new(),
            bindings,
            selectors: vec![CompiledResourceSelector {
                id: content_id('7'),
                execution_scope: "target".to_string(),
                component_id: "layer_00".to_string(),
                node_id: "router".to_string(),
                domain_id: "experts".to_string(),
                resource_count: 2,
                selection_signal: "selected_experts".to_string(),
                execution_signal: "selected_experts".to_string(),
                execution_calibration_word_base: 0,
                encoding: CompiledResourceSelectionEncoding {
                    element_type:
                        CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: 0xffff,
                    calibration_word_base: 0,
                },
                mapping: CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids: groups
                        .iter()
                        .map(|group| group.id.clone())
                        .collect(),
                },
            }],
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn independent_parameters_lower_to_explicit_selector_major_slot_tables() {
        let contract = independent_selected_contract();

        let layout =
            VulkanCompiledResourceAddressLayout::from_contract(&contract)
                .unwrap();
        let table = layout
            .parameter_slot_table(&VulkanDynamicResourceBindingKey::new(
                "layer_00",
                "expert_compute",
                "selected_experts",
            ))
            .unwrap();

        assert_eq!(table.resource_count, 2);
        assert_eq!(table.slot_count(), Some(4));
        assert_eq!(table.slots().collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(
            table.mapping,
            VulkanCompiledParameterSlotMapping::Explicit {
                parameter_slots: vec![0, 1, 2, 3],
            }
        );
    }

    #[test]
    fn independent_parameter_layout_rejects_incomplete_slots() {
        let mut contract = independent_selected_contract();
        contract.bindings.pop();

        let error =
            VulkanCompiledResourceAddressLayout::from_contract(&contract)
                .unwrap_err();

        assert!(error.to_string().contains("slots are incomplete"));
    }

    #[test]
    fn partitioned_parameters_lower_to_dense_stable_slot_tables() {
        let template_id = content_id('a');
        let group_seed = content_id('b');
        let weight_seed = content_id('c');
        let scale_seed = content_id('d');
        let selector_id = content_id('e');
        let template = CompiledPartitionTemplate {
            id: template_id.clone(),
            partition_count: 3,
            lifetime: CompiledResourceLifetime::Dynamic,
            group_identity_seed: group_seed,
            member_templates: vec![
                CompiledPartitionMemberTemplate {
                    resource_identity_seed: weight_seed.clone(),
                    range_templates: Vec::new(),
                    compatibility: CompiledResourceCompatibility {
                        device_api: "vulkan".to_string(),
                        storage_class: "storage_buffer".to_string(),
                        read_only: true,
                        required_features: Vec::new(),
                    },
                    resident_derivation: None,
                },
                CompiledPartitionMemberTemplate {
                    resource_identity_seed: scale_seed.clone(),
                    range_templates: Vec::new(),
                    compatibility: CompiledResourceCompatibility {
                        device_api: "vulkan".to_string(),
                        storage_class: "storage_buffer".to_string(),
                        read_only: true,
                        required_features: Vec::new(),
                    },
                    resident_derivation: None,
                },
            ],
            dependencies: Vec::new(),
        };
        let mut contract = CompiledResourceResidencyContract {
            schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
            identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
            state_machine_schema:
                RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
            supported_policies: vec![
                ResourceResidencyPolicy::DemandRetained,
                ResourceResidencyPolicy::Eager,
            ],
            resources: Vec::new(),
            atomic_groups: Vec::new(),
            partition_templates: vec![template],
            bindings: vec![
                CompiledResourceBinding {
                    execution_scope: "target".to_string(),
                    component_id: "component".to_string(),
                    node_id: "selected_compute".to_string(),
                    parameter_id: "bank".to_string(),
                    mapping:
                        CompiledResourceBindingMapping::PartitionTemplateMember {
                            partition_template_id: template_id.clone(),
                            resource_identity_seed: weight_seed.clone(),
                            selection_signal: "selected".to_string(),
                            parameter_slot: 0,
                        },
                },
                CompiledResourceBinding {
                    execution_scope: "target".to_string(),
                    component_id: "component".to_string(),
                    node_id: "selected_compute".to_string(),
                    parameter_id: "scale".to_string(),
                    mapping:
                        CompiledResourceBindingMapping::PartitionTemplateMember {
                            partition_template_id: template_id.clone(),
                            resource_identity_seed: scale_seed.clone(),
                            selection_signal: "selected".to_string(),
                            parameter_slot: 1,
                        },
                },
            ],
            selectors: vec![CompiledResourceSelector {
                id: selector_id.clone(),
                execution_scope: "target".to_string(),
                component_id: "component".to_string(),
                node_id: "choose".to_string(),
                domain_id: "resources".to_string(),
                resource_count: 3,
                selection_signal: "selected".to_string(),
                execution_signal: "selected".to_string(),
                execution_calibration_word_base: 0,
                encoding: CompiledResourceSelectionEncoding {
                    element_type:
                        CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: 0xffff,
                    calibration_word_base: 0,
                },
                mapping:
                    CompiledResourceSelectorMapping::PartitionTemplate {
                        partition_template_id: template_id,
                    },
            }],
            checkpoints: Vec::new(),
        };
        let mut structurally_reused_selector = contract.selectors[0].clone();
        structurally_reused_selector.id = content_id('f');
        structurally_reused_selector.node_id = "other_choose".to_string();
        structurally_reused_selector.selection_signal = "other_selected".to_string();
        contract.selectors.push(structurally_reused_selector);

        let layout =
            VulkanCompiledResourceAddressLayout::from_contract(&contract)
                .unwrap();

        assert_eq!(layout.slot_count(), 6);
        let selector = layout
            .selector("target", "component", "choose")
            .unwrap();
        assert_eq!(
            (0..selector.mapping.resource_count())
                .map(|index| selector.mapping.resource_slots(index).unwrap())
                .collect::<Vec<_>>(),
            [vec![0, 3], vec![1, 4], vec![2, 5]]
        );
        let table = layout
            .parameter_slot_table(&VulkanDynamicResourceBindingKey::new(
                "component",
                "selected_compute",
                "selected",
            ))
            .unwrap();
        assert_eq!(table.selector_id, selector_id);
        assert_eq!(table.parameter_ids, ["bank", "scale"]);
        assert_eq!(
            table.slots().collect::<Vec<_>>(),
            [0, 3, 1, 4, 2, 5]
        );

        let mut resource_ids = vec![
            derived_partition_resource_id(&weight_seed, 1).unwrap(),
            derived_partition_resource_id(&scale_seed, 1).unwrap(),
        ];
        resource_ids.sort();
        let selected_slots = layout
            .resource_slots_for_selection(&selector_id, 1, &resource_ids)
            .unwrap();
        for (resource_id, slot) in resource_ids.iter().zip(selected_slots) {
            let expected = if resource_id
                == &derived_partition_resource_id(&weight_seed, 1).unwrap()
            {
                1
            } else {
                4
            };
            assert_eq!(slot, expected);
        }
    }

    #[test]
    fn million_partition_layout_keeps_host_metadata_compact() {
        const PARTITION_COUNT: usize = 1_000_000;
        let template_id = content_id('1');
        let weight_seed = content_id('2');
        let scale_seed = content_id('3');
        let selector_id = content_id('4');
        let compatibility = CompiledResourceCompatibility {
            device_api: "vulkan".to_string(),
            storage_class: "storage_buffer".to_string(),
            read_only: true,
            required_features: Vec::new(),
        };
        let contract = CompiledResourceResidencyContract {
            schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
            identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
            state_machine_schema:
                RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
            supported_policies: vec![
                ResourceResidencyPolicy::DemandRetained,
                ResourceResidencyPolicy::Eager,
            ],
            resources: Vec::new(),
            atomic_groups: Vec::new(),
            partition_templates: vec![CompiledPartitionTemplate {
                id: template_id.clone(),
                partition_count: PARTITION_COUNT,
                lifetime: CompiledResourceLifetime::Dynamic,
                group_identity_seed: content_id('5'),
                member_templates: vec![
                    CompiledPartitionMemberTemplate {
                        resource_identity_seed: weight_seed.clone(),
                        range_templates: Vec::new(),
                        compatibility: compatibility.clone(),
                        resident_derivation: None,
                    },
                    CompiledPartitionMemberTemplate {
                        resource_identity_seed: scale_seed.clone(),
                        range_templates: Vec::new(),
                        compatibility: compatibility.clone(),
                        resident_derivation: None,
                    },
                ],
                dependencies: Vec::new(),
            }],
            bindings: vec![
                CompiledResourceBinding {
                    execution_scope: "target".to_string(),
                    component_id: "optional_projection".to_string(),
                    node_id: "selected_projection".to_string(),
                    parameter_id: "weight".to_string(),
                    mapping:
                        CompiledResourceBindingMapping::PartitionTemplateMember {
                            partition_template_id: template_id.clone(),
                            resource_identity_seed: weight_seed,
                            selection_signal: "selected_feature".to_string(),
                            parameter_slot: 0,
                        },
                },
                CompiledResourceBinding {
                    execution_scope: "target".to_string(),
                    component_id: "optional_projection".to_string(),
                    node_id: "selected_projection".to_string(),
                    parameter_id: "scale".to_string(),
                    mapping:
                        CompiledResourceBindingMapping::PartitionTemplateMember {
                            partition_template_id: template_id.clone(),
                            resource_identity_seed: scale_seed,
                            selection_signal: "selected_feature".to_string(),
                            parameter_slot: 1,
                        },
                },
            ],
            selectors: vec![CompiledResourceSelector {
                id: selector_id,
                execution_scope: "target".to_string(),
                component_id: "optional_projection".to_string(),
                node_id: "feature_switch".to_string(),
                domain_id: "optional_features".to_string(),
                resource_count: PARTITION_COUNT,
                selection_signal: "selected_feature".to_string(),
                execution_signal: "selected_feature".to_string(),
                execution_calibration_word_base: 0,
                encoding: CompiledResourceSelectionEncoding {
                    element_type:
                        CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: u32::MAX,
                    calibration_word_base: 0,
                },
                mapping:
                    CompiledResourceSelectorMapping::PartitionTemplate {
                        partition_template_id: template_id,
                    },
            }],
            checkpoints: Vec::new(),
        };

        let layout =
            VulkanCompiledResourceAddressLayout::from_contract(&contract)
                .unwrap();
        assert_eq!(layout.slot_count(), 2 * PARTITION_COUNT);
        let VulkanCompiledSelectorAddressMapping::PartitionTemplate {
            member_slot_bases,
            member_resource_identity_seeds,
            resource_count,
        } = &layout.selectors[0].mapping
        else {
            panic!("partition template expanded into an explicit group table");
        };
        assert_eq!(*resource_count, PARTITION_COUNT);
        assert_eq!(member_slot_bases, &[0, PARTITION_COUNT]);
        assert_eq!(member_resource_identity_seeds.len(), 2);
        assert_eq!(layout.selectors[0].mapping.slot_ranges().unwrap().len(), 2);
        assert_eq!(
            layout
                .addressable_slot_count_for_selectors(&BTreeSet::from([
                    layout.selectors[0].selector_id.clone(),
                ]))
                .unwrap(),
            2 * PARTITION_COUNT
        );
        assert_eq!(
            layout.parameter_slot_tables[0].mapping,
            VulkanCompiledParameterSlotMapping::Partitioned {
                parameter_slot_bases: vec![0, PARTITION_COUNT],
            }
        );
        assert_eq!(
            layout.parameter_slot_tables[0].slot_count(),
            Some(2 * PARTITION_COUNT)
        );
        assert_eq!(
            layout.selectors[0]
                .mapping
                .resource_slots(PARTITION_COUNT - 1)
                .unwrap(),
            [PARTITION_COUNT - 1, 2 * PARTITION_COUNT - 1]
        );
    }
}
