#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceSelectorOwnership {
    resources_by_selector: BTreeMap<String, BTreeSet<usize>>,
}

impl VulkanCompiledResourceSelectorOwnership {
    pub fn all(
        contract: &CompiledResourceResidencyContract,
        selector_ids: &BTreeSet<String>,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        let resources_by_selector = selector_ids
            .iter()
            .map(|selector_id| {
                let selector = contract
                    .selectors
                    .iter()
                    .find(|selector| selector.id == *selector_id)
                    .ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "compiled resource ownership references unknown selector {selector_id:?}"
                        ))
                    })?;
                Ok((
                    selector_id.clone(),
                    (0..selector.resource_count).collect::<BTreeSet<_>>(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Self::from_resource_indices(contract, resources_by_selector)
    }

    pub fn from_resource_indices(
        contract: &CompiledResourceResidencyContract,
        resources_by_selector: BTreeMap<String, BTreeSet<usize>>,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        if resources_by_selector.is_empty() {
            return Err(VulkanRuntimeResidencyPlanError(
                "compiled resource ownership must contain at least one selector".to_string(),
            ));
        }
        for (selector_id, indices) in &resources_by_selector {
            let selector = contract
                .selectors
                .iter()
                .find(|selector| selector.id == *selector_id)
                .ok_or_else(|| {
                    VulkanRuntimeResidencyPlanError(format!(
                        "compiled resource ownership references unknown selector {selector_id:?}"
                    ))
                })?;
            if indices.is_empty() || indices.iter().any(|index| *index >= selector.resource_count) {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "compiled resource ownership for selector {selector_id:?} is empty or outside its {} resources",
                    selector.resource_count
                )));
            }
        }
        Ok(Self {
            resources_by_selector,
        })
    }

    pub fn selector_ids(&self) -> BTreeSet<String> {
        self.resources_by_selector.keys().cloned().collect()
    }

    pub fn resources(&self, selector_id: &str) -> Option<&BTreeSet<usize>> {
        self.resources_by_selector.get(selector_id)
    }

    pub fn owns(&self, selector_id: &str, resource_index: usize) -> bool {
        self.resources(selector_id)
            .is_some_and(|indices| indices.contains(&resource_index))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &BTreeSet<usize>)> {
        self.resources_by_selector
            .iter()
            .map(|(selector_id, indices)| (selector_id.as_str(), indices))
    }
}

#[cfg(test)]
mod compiled_resource_selector_ownership_tests {
    use super::*;

    fn contract() -> CompiledResourceResidencyContract {
        CompiledResourceResidencyContract {
            schema: "fixture".to_string(),
            identity_algorithm: "fixture".to_string(),
            state_machine_schema: "fixture".to_string(),
            supported_policies: Vec::new(),
            resources: Vec::new(),
            atomic_groups: Vec::new(),
            partition_templates: Vec::new(),
            bindings: Vec::new(),
            selectors: vec![CompiledResourceSelector {
                id: "experts".to_string(),
                execution_scope: "target".to_string(),
                component_id: "layer".to_string(),
                node_id: "router".to_string(),
                domain_id: "experts".to_string(),
                resource_count: 8,
                selection_signal: "routes".to_string(),
                encoding: CompiledResourceSelectionEncoding {
                    element_type: CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 2,
                    index_shift: 0,
                    index_mask: 7,
                },
                mapping: CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids: (0..8).map(|index| format!("group-{index}")).collect(),
                },
            }],
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn selector_ownership_is_explicit_and_range_checked() {
        let contract = contract();
        let ownership = VulkanCompiledResourceSelectorOwnership::from_resource_indices(
            &contract,
            BTreeMap::from([("experts".to_string(), BTreeSet::from([4, 5, 6, 7]))]),
        )
        .unwrap();
        assert!(!ownership.owns("experts", 3));
        assert!(ownership.owns("experts", 4));
        assert_eq!(ownership.selector_ids(), BTreeSet::from(["experts".to_string()]));
        let layout = VulkanCompiledResourceAddressLayout {
            slot_count: 8,
            concrete_resource_slots: BTreeMap::new(),
            selectors: vec![VulkanCompiledSelectorAddressLayout {
                selector_id: "experts".to_string(),
                execution_scope: "target".to_string(),
                component_id: "layer".to_string(),
                node_id: "router".to_string(),
                selection_signal: "routes".to_string(),
                mapping: VulkanCompiledSelectorAddressMapping::GroupTable {
                    resource_address_slots: (0..8).collect(),
                    resource_address_slot_offsets: (0..=8).collect(),
                },
            }],
            parameter_slot_tables: Vec::new(),
        };
        assert_eq!(
            layout
                .addressable_slot_count_for_ownership(&ownership)
                .unwrap(),
            4
        );
        let residency = plan_compiled_resource_store_residency_for_ownership(
            &contract,
            &layout,
            &ownership,
            32,
            4,
        )
        .unwrap();
        assert_eq!(residency.maximum_load_wave_group_count, 2);
        assert_eq!(residency.maximum_load_wave_payload_bytes, 64);
        assert_eq!(residency.maximum_dynamic_allocation_padding_bytes, 12);

        let error = VulkanCompiledResourceSelectorOwnership::from_resource_indices(
            &contract,
            BTreeMap::from([("experts".to_string(), BTreeSet::from([8]))]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("outside its 8 resources"));
    }
}
