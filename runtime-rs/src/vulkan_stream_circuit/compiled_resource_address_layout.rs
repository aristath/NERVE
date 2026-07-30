#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceAddressSlot {
    pub slot: usize,
    pub resource_id: String,
    pub partition_template_id: Option<String>,
    pub resource_identity_seed: Option<String>,
    pub partition_index: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledSelectorAddressLayout {
    pub selector_id: String,
    pub execution_scope: String,
    pub component_id: String,
    pub node_id: String,
    pub selection_signal: String,
    pub resource_address_slots: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledParameterSlotTable {
    pub key: VulkanDynamicResourceBindingKey,
    pub selector_id: String,
    pub execution_scope: String,
    pub parameter_ids: Vec<String>,
    pub resource_count: usize,
    pub slots: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceAddressLayout {
    pub slots: Vec<VulkanCompiledResourceAddressSlot>,
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

impl VulkanCompiledResourceAddressLayout {
    pub fn from_contract(
        contract: &CompiledResourceResidencyContract,
    ) -> Result<Self, VulkanCompiledResourceAddressLayoutError> {
        let mut slots = Vec::new();
        let mut concrete_slots = BTreeMap::new();
        for resource in contract.resources.iter().filter(|resource| {
            resource.lifetime == CompiledResourceLifetime::Dynamic
        }) {
            let slot = slots.len();
            concrete_slots.insert(resource.id.clone(), slot);
            slots.push(VulkanCompiledResourceAddressSlot {
                slot,
                resource_id: resource.id.clone(),
                partition_template_id: None,
                resource_identity_seed: None,
                partition_index: None,
            });
        }

        let mut partition_slots = BTreeMap::new();
        for template in &contract.partition_templates {
            for member in &template.member_templates {
                for partition_index in 0..template.partition_count {
                    let resource_id = derived_partition_resource_id(
                        &member.resource_identity_seed,
                        partition_index,
                    )
                    .map_err(|error| {
                        VulkanCompiledResourceAddressLayoutError(format!(
                            "could not derive compiled partition resource identity: {error}"
                        ))
                    })?;
                    let slot = slots.len();
                    partition_slots.insert(
                        (
                            template.id.clone(),
                            member.resource_identity_seed.clone(),
                            partition_index,
                        ),
                        slot,
                    );
                    slots.push(VulkanCompiledResourceAddressSlot {
                        slot,
                        resource_id,
                        partition_template_id: Some(template.id.clone()),
                        resource_identity_seed: Some(
                            member.resource_identity_seed.clone(),
                        ),
                        partition_index: Some(partition_index),
                    });
                }
            }
        }
        if slots.is_empty() && !contract.selectors.is_empty() {
            return Err(VulkanCompiledResourceAddressLayoutError(
                "compiled selectors have no dynamic address slots".to_string(),
            ));
        }

        let mut selectors = Vec::with_capacity(contract.selectors.len());
        for selector in &contract.selectors {
            let resource_address_slots = match &selector.mapping {
                CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids,
                } => atomic_group_ids
                    .iter()
                    .map(|group_id| {
                        let group = contract
                            .atomic_groups
                            .iter()
                            .find(|group| group.id == *group_id)
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(
                                    format!(
                                        "selector {:?} maps missing atomic group {group_id:?}",
                                        selector.id
                                    ),
                                )
                            })?;
                        group
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
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                CompiledResourceSelectorMapping::PartitionTemplate {
                    partition_template_id,
                } => {
                    let template = contract
                        .partition_templates
                        .iter()
                        .find(|template| template.id == *partition_template_id)
                        .ok_or_else(|| {
                            VulkanCompiledResourceAddressLayoutError(format!(
                                "selector {:?} maps missing partition template {partition_template_id:?}",
                                selector.id
                            ))
                        })?;
                    (0..template.partition_count)
                        .map(|partition_index| {
                            template
                                .member_templates
                                .iter()
                                .map(|member| {
                                    partition_slots
                                        .get(&(
                                            template.id.clone(),
                                            member
                                                .resource_identity_seed
                                                .clone(),
                                            partition_index,
                                        ))
                                        .copied()
                                        .ok_or_else(|| {
                                            VulkanCompiledResourceAddressLayoutError(
                                                "compiled partition address slot is missing"
                                                    .to_string(),
                                            )
                                        })
                                })
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
            };
            if resource_address_slots.len() != selector.resource_count
                || resource_address_slots.iter().any(Vec::is_empty)
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
                resource_address_slots,
            });
        }

        let mut partition_bindings: BTreeMap<
            (String, String, String, String),
            Vec<(&str, &str)>,
        > = BTreeMap::new();
        for binding in &contract.bindings {
            if let CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id,
                resource_identity_seed,
            } = &binding.mapping
            {
                partition_bindings
                    .entry((
                        binding.execution_scope.clone(),
                        binding.component_id.clone(),
                        binding.node_id.clone(),
                        partition_template_id.clone(),
                    ))
                    .or_default()
                    .push((
                        binding.parameter_id.as_str(),
                        resource_identity_seed.as_str(),
                    ));
            }
        }

        let mut parameter_slot_tables =
            Vec::with_capacity(partition_bindings.len());
        for ((scope, component_id, node_id, template_id), mut bindings) in
            partition_bindings
        {
            bindings.sort_unstable();
            if bindings.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(VulkanCompiledResourceAddressLayoutError(format!(
                    "{scope} {component_id}.{node_id} repeats a selected parameter binding"
                )));
            }
            let matching_selectors = contract
                .selectors
                .iter()
                .filter(|selector| {
                    selector.execution_scope == scope
                        && selector.component_id == component_id
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
            let parameter_ids = bindings
                .iter()
                .map(|(parameter_id, _)| (*parameter_id).to_string())
                .collect::<Vec<_>>();
            let table_slot_count = selector
                .resource_count
                .checked_mul(parameter_ids.len())
                .ok_or_else(|| {
                    VulkanCompiledResourceAddressLayoutError(format!(
                        "{scope} {component_id}.{node_id} parameter-slot table capacity overflowed"
                    ))
                })?;
            let mut table_slots = Vec::with_capacity(table_slot_count);
            for resource_index in 0..selector.resource_count {
                for (_, seed) in &bindings {
                    table_slots.push(
                        partition_slots
                            .get(&(template_id.clone(), (*seed).to_string(), resource_index))
                            .copied()
                            .ok_or_else(|| {
                                VulkanCompiledResourceAddressLayoutError(format!(
                                    "{scope} {component_id}.{node_id} selected parameter has no address slot"
                                ))
                            })?,
                    );
                }
            }
            parameter_slot_tables.push(VulkanCompiledParameterSlotTable {
                key: VulkanDynamicResourceBindingKey::new(
                    component_id,
                    node_id,
                    &selector.selection_signal,
                ),
                selector_id: selector.id.clone(),
                execution_scope: scope,
                parameter_ids,
                resource_count: selector.resource_count,
                slots: table_slots,
            });
        }
        parameter_slot_tables.sort_by(|left, right| left.key.cmp(&right.key));

        Ok(Self {
            slots,
            selectors,
            parameter_slot_tables,
        })
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
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
                    .slots
                    .len()
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
        let slots_by_resource_id = self
            .slots
            .iter()
            .map(|slot| (slot.resource_id.as_str(), slot.slot))
            .collect::<BTreeMap<_, _>>();
        resource_ids
            .iter()
            .map(|resource_id| {
                slots_by_resource_id
                    .get(resource_id.as_str())
                    .copied()
                    .ok_or_else(|| {
                        VulkanCompiledResourceAddressLayoutError(format!(
                            "compiled resource {resource_id:?} has no stable address slot"
                        ))
                    })
            })
            .collect()
    }
}

#[cfg(test)]
mod compiled_resource_address_layout_tests {
    use super::*;

    fn content_id(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
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
                },
            ],
            dependencies: Vec::new(),
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
                encoding: CompiledResourceSelectionEncoding {
                    element_type:
                        CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: 0xffff,
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

        assert_eq!(layout.slot_count(), 6);
        let selector = layout
            .selector("target", "component", "choose")
            .unwrap();
        assert_eq!(
            selector.resource_address_slots,
            vec![vec![0, 3], vec![1, 4], vec![2, 5]]
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
        assert_eq!(table.slots, [0, 3, 1, 4, 2, 5]);

        let mut resource_ids = vec![
            derived_partition_resource_id(&weight_seed, 1).unwrap(),
            derived_partition_resource_id(&scale_seed, 1).unwrap(),
        ];
        resource_ids.sort();
        let expected_slots = resource_ids
            .iter()
            .map(|resource_id| {
                layout
                    .slots
                    .iter()
                    .find(|slot| slot.resource_id == *resource_id)
                    .unwrap()
                    .slot
            })
            .collect::<Vec<_>>();
        assert_eq!(
            layout.resource_slots_for_ids(&resource_ids).unwrap(),
            expected_slots
        );
    }
}
