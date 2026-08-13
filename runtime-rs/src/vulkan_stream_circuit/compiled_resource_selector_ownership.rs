#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceSelectorOwnership {
    resources_by_selector: BTreeMap<String, BTreeSet<usize>>,
    source_projections_by_selector:
        BTreeMap<String, BTreeMap<usize, VulkanCompiledResourceSourceProjection>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceSourceProjection {
    pub resources: BTreeMap<String, VulkanCompiledResourceSourceRangeProjection>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCompiledResourceSourceRangeProjection {
    pub source_byte_count: usize,
    pub byte_offset: usize,
    pub byte_count: usize,
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
        Self::from_resources_and_source_projections(
            contract,
            resources_by_selector,
            BTreeMap::new(),
        )
    }

    pub fn from_resources_and_source_projections(
        contract: &CompiledResourceResidencyContract,
        mut resources_by_selector: BTreeMap<String, BTreeSet<usize>>,
        source_projections_by_selector: BTreeMap<
            String,
            BTreeMap<usize, VulkanCompiledResourceSourceProjection>,
        >,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        if resources_by_selector.is_empty() {
            if source_projections_by_selector.is_empty() {
                return Err(VulkanRuntimeResidencyPlanError(
                    "compiled resource ownership must contain at least one selector".to_string(),
                ));
            }
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
        let projected = Self::from_source_projections_validated(
            contract,
            &source_projections_by_selector,
        )?;
        for (selector_id, indices) in projected {
            if resources_by_selector
                .get(&selector_id)
                .is_some_and(|whole| !whole.is_disjoint(&indices))
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "compiled resource ownership mixes whole and projected resource indices for selector {selector_id:?}",
                )));
            }
            resources_by_selector
                .entry(selector_id)
                .or_default()
                .extend(indices);
        }
        Ok(Self {
            resources_by_selector,
            source_projections_by_selector,
        })
    }

    pub fn from_source_projections(
        contract: &CompiledResourceResidencyContract,
        source_projections_by_selector: BTreeMap<
            String,
            BTreeMap<usize, VulkanCompiledResourceSourceProjection>,
        >,
    ) -> Result<Self, VulkanRuntimeResidencyPlanError> {
        Self::from_resources_and_source_projections(
            contract,
            BTreeMap::new(),
            source_projections_by_selector,
        )
    }

    fn from_source_projections_validated(
        contract: &CompiledResourceResidencyContract,
        source_projections_by_selector: &BTreeMap<
            String,
            BTreeMap<usize, VulkanCompiledResourceSourceProjection>,
        >,
    ) -> Result<BTreeMap<String, BTreeSet<usize>>, VulkanRuntimeResidencyPlanError> {
        let index = CompiledResourceContractIndex::new(contract)
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        let mut resources_by_selector = BTreeMap::new();
        for (selector_id, projections) in source_projections_by_selector {
            let selector = index.selector(contract, selector_id).ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "compiled resource projection references unknown selector {selector_id:?}",
                ))
            })?;
            if projections.is_empty()
                || projections
                    .keys()
                    .any(|resource_index| *resource_index >= selector.resource_count)
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "compiled resource projections for selector {selector_id:?} are empty or outside its {} resources",
                    selector.resource_count,
                )));
            }
            for (resource_index, projection) in projections {
                validate_compiled_resource_source_projection(
                    contract,
                    &index,
                    selector,
                    *resource_index,
                    projection,
                )?;
            }
            resources_by_selector.insert(
                selector_id.clone(),
                projections.keys().copied().collect(),
            );
        }
        Ok(resources_by_selector)
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

    pub fn source_projection(
        &self,
        selector_id: &str,
        resource_index: usize,
    ) -> Option<&VulkanCompiledResourceSourceProjection> {
        self.source_projections_by_selector
            .get(selector_id)?
            .get(&resource_index)
    }

    pub fn has_source_projections(&self) -> bool {
        !self.source_projections_by_selector.is_empty()
    }

    /// Returns whether every executable resource in `self` is addressable by
    /// `other` with the same physical source projection.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.resources_by_selector.iter().all(|(selector_id, indices)| {
            indices.iter().all(|resource_index| {
                other.owns(selector_id, *resource_index)
                    && self.source_projection(selector_id, *resource_index)
                        == other.source_projection(selector_id, *resource_index)
            })
        })
    }

    pub fn project_resolved_group(
        &self,
        selector_id: &str,
        resource_index: usize,
        group: ResolvedCompiledResourceGroup,
    ) -> Result<ResolvedCompiledResourceGroup, VulkanRuntimeResidencyPlanError> {
        let Some(projection) = self.source_projection(selector_id, resource_index) else {
            return Ok(group);
        };
        let project_resources = |resources: Vec<ResolvedCompiledResource>| {
            resources
                .into_iter()
                .map(|resource| {
                    let projected = projection.resources.get(&resource.id).ok_or_else(|| {
                        VulkanRuntimeResidencyPlanError(format!(
                            "compiled source projection omits resolved resource {:?}",
                            resource.id,
                        ))
                    })?;
                    project_resolved_compiled_resource(resource, projected)
                })
                .collect::<Result<Vec<_>, _>>()
        };
        match group {
            ResolvedCompiledResourceGroup::Atomic(mut group) => {
                group.resources = project_resources(group.resources)?;
                Ok(ResolvedCompiledResourceGroup::Atomic(group))
            }
            ResolvedCompiledResourceGroup::Partition(_) => Err(VulkanRuntimeResidencyPlanError(
                "compiled source projections cannot resolve a partition-template group"
                    .to_string(),
            )),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &BTreeSet<usize>)> {
        self.resources_by_selector
            .iter()
            .map(|(selector_id, indices)| (selector_id.as_str(), indices))
    }
}

fn compiled_resource_selector_ownership_from_distributed_device_plan(
    contract: &CompiledResourceResidencyContract,
    device: &VulkanDistributedSelectedResourceDevicePlan,
) -> Result<VulkanCompiledResourceSelectorOwnership, VulkanRuntimeResidencyPlanError> {
    let mut resources_by_selector = BTreeMap::new();
    let mut projections_by_selector = BTreeMap::new();
    for selector in &device.selectors {
        let whole = !selector.owned_resource_indices.is_empty();
        let fragmented = !selector.fragmented_resources.is_empty();
        if whole == fragmented {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "distributed selector {:?} on {:?} must own either whole resources or fragments",
                selector.selector_id, device.device_id,
            )));
        }
        if whole
            && resources_by_selector
                .insert(
                    selector.selector_id.clone(),
                    selector
                        .owned_resource_indices
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>(),
                )
                .is_some()
        {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "distributed selector {:?} is repeated on {:?}",
                selector.selector_id, device.device_id,
            )));
        }
        for fragment in &selector.fragmented_resources {
            let projection = VulkanCompiledResourceSourceProjection {
                resources: fragment
                    .resources
                    .iter()
                    .map(|resource| {
                        (
                            resource.resource_id.clone(),
                            VulkanCompiledResourceSourceRangeProjection {
                                source_byte_count: resource.source_byte_count,
                                byte_offset: resource.byte_offset,
                                byte_count: resource.byte_count,
                            },
                        )
                    })
                    .collect(),
            };
            if projections_by_selector
                .entry(selector.selector_id.clone())
                .or_insert_with(BTreeMap::new)
                .insert(fragment.resource_index, projection)
                .is_some()
            {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "distributed selector {:?} repeats fragment resource {} on {:?}",
                    selector.selector_id, fragment.resource_index, device.device_id,
                )));
            }
        }
    }
    VulkanCompiledResourceSelectorOwnership::from_resources_and_source_projections(
        contract,
        resources_by_selector,
        projections_by_selector,
    )
}

fn project_resolved_compiled_resource(
    mut resource: ResolvedCompiledResource,
    projection: &VulkanCompiledResourceSourceRangeProjection,
) -> Result<ResolvedCompiledResource, VulkanRuntimeResidencyPlanError> {
    if resource.resident_derivation.is_some() {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "resolved source projection for {:?} cannot use a resident derivation",
            resource.id,
        )));
    }
    let end = projection
        .byte_offset
        .checked_add(projection.byte_count)
        .ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "resolved source projection range overflowed".to_string(),
            )
        })?;
    let mut logical_offset = 0usize;
    let mut selected = Vec::new();
    for range in resource.ranges {
        let range_end = logical_offset.checked_add(range.byte_count).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "resolved resource logical range overflowed".to_string(),
            )
        })?;
        if logical_offset < end && projection.byte_offset < range_end {
            if logical_offset < projection.byte_offset || range_end > end {
                return Err(VulkanRuntimeResidencyPlanError(format!(
                    "resolved source projection for {:?} cuts through a compiler-owned integrity range",
                    resource.id,
                )));
            }
            selected.push(range);
        }
        logical_offset = range_end;
    }
    let selected_bytes = selected.iter().try_fold(0usize, |total, range| {
        total.checked_add(range.byte_count).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(
                "resolved projected resource byte count overflowed".to_string(),
            )
        })
    })?;
    if logical_offset != projection.source_byte_count || selected_bytes != projection.byte_count {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "resolved source projection for {:?} differs from its compiled extent",
            resource.id,
        )));
    }
    resource.ranges = selected;
    Ok(resource)
}

fn validate_compiled_resource_source_projection(
    contract: &CompiledResourceResidencyContract,
    index: &CompiledResourceContractIndex,
    selector: &CompiledResourceSelector,
    resource_index: usize,
    projection: &VulkanCompiledResourceSourceProjection,
) -> Result<(), VulkanRuntimeResidencyPlanError> {
    let group = match &selector.mapping {
        CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => index
            .atomic_group(contract, &atomic_group_ids[resource_index])
            .ok_or_else(|| {
                VulkanRuntimeResidencyPlanError(format!(
                    "compiled selector {:?} references a missing atomic group",
                    selector.id,
                ))
            })?,
        CompiledResourceSelectorMapping::PartitionTemplate { .. } => {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "compiled selector {:?} cannot project partition-template resources",
                selector.id,
            )));
        }
    };
    let expected_ids = group
        .resource_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let projected_ids = projection
        .resources
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if expected_ids != projected_ids {
        return Err(VulkanRuntimeResidencyPlanError(format!(
            "compiled selector {:?} resource {resource_index} projection must cover every atomic-group member",
            selector.id,
        )));
    }
    for (resource_id, projected) in &projection.resources {
        let resource = index.resource(contract, resource_id).ok_or_else(|| {
            VulkanRuntimeResidencyPlanError(format!(
                "compiled source projection references missing resource {resource_id:?}",
            ))
        })?;
        let source_byte_count = resource
            .source_byte_count()
            .map_err(|error| VulkanRuntimeResidencyPlanError(error.to_string()))?;
        if resource.resident_derivation.is_some()
            || projected.source_byte_count != source_byte_count
            || projected.byte_count == 0
            || projected
                .byte_offset
                .checked_add(projected.byte_count)
                .is_none_or(|end| end > source_byte_count)
            || !compiled_resource_projection_matches_complete_ranges(resource, projected)
        {
            return Err(VulkanRuntimeResidencyPlanError(format!(
                "compiled source projection for resource {resource_id:?} is not an exact union of compiler-owned source ranges",
            )));
        }
    }
    Ok(())
}

fn compiled_resource_projection_matches_complete_ranges(
    resource: &CompiledImmutableResource,
    projection: &VulkanCompiledResourceSourceRangeProjection,
) -> bool {
    let Some(end) = projection.byte_offset.checked_add(projection.byte_count) else {
        return false;
    };
    let mut logical_offset = 0usize;
    let mut selected_bytes = 0usize;
    for range in &resource.ranges {
        let Some(range_end) = logical_offset.checked_add(range.byte_count) else {
            return false;
        };
        let overlaps = logical_offset < end && projection.byte_offset < range_end;
        if overlaps {
            if logical_offset < projection.byte_offset || range_end > end {
                return false;
            }
            let Some(total) = selected_bytes.checked_add(range.byte_count) else {
                return false;
            };
            selected_bytes = total;
        }
        logical_offset = range_end;
    }
    selected_bytes == projection.byte_count && logical_offset == projection.source_byte_count
}

#[cfg(test)]
mod compiled_resource_selector_ownership_tests {
    use super::*;

    fn contract() -> CompiledResourceResidencyContract {
        let resources = (0..8)
            .map(|index| CompiledImmutableResource {
                id: format!("resource-{index}"),
                lifetime: CompiledResourceLifetime::Dynamic,
                ranges: vec![CompiledResourceByteRange {
                    artifact_path: "weights.bin".to_string(),
                    byte_offset: index * 64,
                    byte_count: (index + 1) * 8,
                    alignment_bytes: 8,
                    integrity: CompiledResourceRangeIntegrity {
                        algorithm: "sha256".to_string(),
                        digest: format!("{index:x}").repeat(64),
                    },
                }],
                dependencies: Vec::new(),
                compatibility: CompiledResourceCompatibility {
                    device_api: "vulkan".to_string(),
                    storage_class: "storage_buffer".to_string(),
                    read_only: true,
                    required_features: Vec::new(),
                },
                resident_derivation: None,
            })
            .collect::<Vec<_>>();
        let atomic_groups = resources
            .iter()
            .enumerate()
            .map(|(index, resource)| CompiledAtomicResidencyGroup {
                id: format!("group-{index}"),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: vec![resource.id.clone()],
                dependencies: Vec::new(),
            })
            .collect();
        CompiledResourceResidencyContract {
            schema: "fixture".to_string(),
            identity_algorithm: "fixture".to_string(),
            state_machine_schema: "fixture".to_string(),
            supported_policies: Vec::new(),
            resources,
            atomic_groups,
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
                execution_signal: "routes".to_string(),
                execution_calibration_word_base: 0,
                encoding: CompiledResourceSelectionEncoding {
                    element_type: CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 2,
                    index_shift: 0,
                    index_mask: 7,
                    calibration_word_base: 0,
                },
                mapping: CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids: (0..8).map(|index| format!("group-{index}")).collect(),
                },
            }],
            checkpoints: Vec::new(),
        }
    }

    fn projection_contract() -> CompiledResourceResidencyContract {
        let gate_id = format!("sha256:{}", "1".repeat(64));
        let up_id = format!("sha256:{}", "2".repeat(64));
        let group_id = format!("sha256:{}", "3".repeat(64));
        let resource = |id: &str, base: usize| CompiledImmutableResource {
            id: id.to_string(),
            lifetime: CompiledResourceLifetime::Dynamic,
            ranges: [0usize, 8]
                .into_iter()
                .map(|relative| CompiledResourceByteRange {
                    artifact_path: "weights.bin".to_string(),
                    byte_offset: base + relative,
                    byte_count: 8,
                    alignment_bytes: 8,
                    integrity: CompiledResourceRangeIntegrity {
                        algorithm: "sha256".to_string(),
                        digest: "a".repeat(64),
                    },
                })
                .collect(),
            dependencies: Vec::new(),
            compatibility: CompiledResourceCompatibility {
                device_api: "vulkan".to_string(),
                storage_class: "storage_buffer".to_string(),
                read_only: true,
                required_features: Vec::new(),
            },
            resident_derivation: None,
        };
        CompiledResourceResidencyContract {
            schema: "fixture".to_string(),
            identity_algorithm: "fixture".to_string(),
            state_machine_schema: "fixture".to_string(),
            supported_policies: Vec::new(),
            resources: vec![resource(&gate_id, 0), resource(&up_id, 16)],
            atomic_groups: vec![CompiledAtomicResidencyGroup {
                id: group_id.clone(),
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: vec![gate_id, up_id],
                dependencies: Vec::new(),
            }],
            partition_templates: Vec::new(),
            bindings: Vec::new(),
            selectors: vec![CompiledResourceSelector {
                id: "experts".to_string(),
                execution_scope: "target".to_string(),
                component_id: "layer".to_string(),
                node_id: "router".to_string(),
                domain_id: "experts".to_string(),
                resource_count: 1,
                selection_signal: "routes".to_string(),
                execution_signal: "routes".to_string(),
                execution_calibration_word_base: 0,
                encoding: CompiledResourceSelectionEncoding {
                    element_type: CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: 0,
                    calibration_word_base: 0,
                },
                mapping: CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids: vec![group_id],
                },
            }],
            checkpoints: Vec::new(),
        }
    }

    fn resident_derivation(source_byte_count: usize) -> CompiledResourceResidentDerivation {
        CompiledResourceResidentDerivation {
            schema: RESIDENT_DERIVATION_SCHEMA.to_string(),
            kind: CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3,
            source_byte_count,
            resident_byte_count: source_byte_count * 2,
            required_features: vec![
                "shader_float8".to_string(),
                "shader_int8".to_string(),
                "shader_mixed_float_dot_product_float8_acc_float32".to_string(),
            ],
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
        let addressability = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &BTreeSet::from(["experts".to_string()]),
        )
        .unwrap();
        assert!(ownership.is_subset_of(&addressability));
        assert!(!addressability.is_subset_of(&ownership));
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
            64,
            4,
        )
        .unwrap();
        assert_eq!(residency.maximum_load_wave_group_count, 2);
        assert_eq!(residency.maximum_load_wave_payload_bytes, 120);
        assert_eq!(residency.maximum_dynamic_allocation_padding_bytes, 12);
        assert_eq!(residency.maximum_representation_group_payload_bytes, 0);
        assert_eq!(residency.maximum_representation_load_wave_group_count, 0);
        assert_eq!(residency.maximum_representation_load_wave_payload_bytes, 0);
        assert_eq!(residency.retained_representation_cache_group_count, 0);
        assert_eq!(residency.retained_representation_cache_wave_count, 0);
        assert_eq!(residency.retained_representation_cache_payload_bytes, 0);
        assert_eq!(residency.retained_representation_cache_identity, None);
        assert_eq!(
            residency.retained_representation_cache_device_bytes().unwrap(),
            0,
        );

        let error = VulkanCompiledResourceSelectorOwnership::from_resource_indices(
            &contract,
            BTreeMap::from([("experts".to_string(), BTreeSet::from([8]))]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("outside its 8 resources"));
    }

    #[test]
    fn representation_cache_retains_one_complete_selected_wave_per_selector() {
        let mut contract = contract();
        for resource in &mut contract.resources {
            let source_bytes = resource.source_byte_count().unwrap();
            resource.resident_derivation = Some(resident_derivation(source_bytes));
        }
        let mut first = contract.selectors.pop().unwrap();
        first.id = "experts-a".to_string();
        first.resource_count = 4;
        first.encoding.index_mask = 3;
        first.mapping = CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: (0..4).map(|index| format!("group-{index}")).collect(),
        };
        let mut second = first.clone();
        second.id = "experts-b".to_string();
        second.component_id = "layer-b".to_string();
        second.mapping = CompiledResourceSelectorMapping::GroupTable {
            atomic_group_ids: (4..8).map(|index| format!("group-{index}")).collect(),
        };
        contract.selectors = vec![first, second];
        let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
        let ownership = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &BTreeSet::from(["experts-a".to_string(), "experts-b".to_string()]),
        )
        .unwrap();

        let residency = plan_compiled_resource_store_residency_for_ownership(
            &contract,
            &layout,
            &ownership,
            64,
            4,
        )
        .unwrap();

        assert_eq!(residency.maximum_load_wave_payload_bytes, 120);
        assert_eq!(residency.maximum_representation_group_payload_bytes, 128);
        assert_eq!(residency.maximum_representation_load_wave_group_count, 2);
        assert_eq!(residency.maximum_representation_load_wave_payload_bytes, 240);
        assert_eq!(
            residency.maximum_representation_load_wave_allocation_padding_bytes,
            12,
        );
        assert_eq!(residency.retained_representation_cache_group_count, 4);
        assert_eq!(residency.retained_representation_cache_wave_count, 2);
        assert_eq!(residency.retained_representation_cache_payload_bytes, 352);
        assert!(
            residency
                .retained_representation_cache_identity
                .as_deref()
                .is_some_and(|identity| identity.starts_with("sha256:")),
        );
        assert_eq!(
            residency.retained_representation_cache_allocation_padding_bytes,
            24,
        );
        assert_eq!(residency.transfer_staging_slot_byte_capacity, 256);
    }

    #[test]
    fn representation_group_duplicates_unchanged_atomic_members_exactly() {
        let mut contract = projection_contract();
        for resource in &mut contract.resources {
            for range in &mut resource.ranges {
                range.byte_count = 40;
            }
        }
        contract.resources[0].resident_derivation = Some(resident_derivation(80));
        let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
        let ownership = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &BTreeSet::from(["experts".to_string()]),
        )
        .unwrap();

        let residency = plan_compiled_resource_store_residency_for_ownership(
            &contract,
            &layout,
            &ownership,
            160,
            8,
        )
        .unwrap();

        assert_eq!(residency.maximum_load_wave_payload_bytes, 160);
        assert_eq!(residency.maximum_representation_group_payload_bytes, 240);
        assert_eq!(residency.maximum_representation_load_wave_payload_bytes, 240);
        assert_eq!(
            residency.maximum_representation_load_wave_allocation_padding_bytes,
            21,
        );
        assert_eq!(residency.retained_representation_cache_wave_count, 1);
        assert_eq!(residency.retained_representation_cache_payload_bytes, 240);
        assert!(residency.retained_representation_cache_identity.is_some());
        assert_eq!(
            residency.retained_representation_cache_allocation_padding_bytes,
            21,
        );
        assert_eq!(residency.transfer_staging_slot_byte_capacity, 240);
    }

    #[test]
    fn partition_template_representation_cache_uses_member_derivations() {
        let mut contract = contract();
        contract.resources.clear();
        contract.atomic_groups.clear();
        contract.partition_templates = vec![CompiledPartitionTemplate {
            id: "partition-template".to_string(),
            partition_count: 4,
            lifetime: CompiledResourceLifetime::Dynamic,
            group_identity_seed: "partition-group".to_string(),
            member_templates: vec![
                CompiledPartitionMemberTemplate {
                    resource_identity_seed: "weight".to_string(),
                    range_templates: vec![CompiledResourceRangeTemplate {
                        artifact_path: "weights.bin".to_string(),
                        base_byte_offset: 0,
                        stride_bytes: 64,
                        byte_count: 64,
                        alignment_bytes: 8,
                        integrity: CompiledResourceRangeIntegrityTemplate {
                            algorithm: "sha256_table".to_string(),
                            digest_table_path: "digests.bin".to_string(),
                            digest_table_byte_offset: 0,
                            digest_stride_bytes: 32,
                            table_sha256: "a".repeat(64),
                        },
                    }],
                    compatibility: CompiledResourceCompatibility {
                        device_api: "vulkan".to_string(),
                        storage_class: "storage_buffer".to_string(),
                        read_only: true,
                        required_features: Vec::new(),
                    },
                    resident_derivation: Some(resident_derivation(64)),
                },
                CompiledPartitionMemberTemplate {
                    resource_identity_seed: "scale".to_string(),
                    range_templates: vec![CompiledResourceRangeTemplate {
                        artifact_path: "scales.bin".to_string(),
                        base_byte_offset: 0,
                        stride_bytes: 8,
                        byte_count: 8,
                        alignment_bytes: 8,
                        integrity: CompiledResourceRangeIntegrityTemplate {
                            algorithm: "sha256_table".to_string(),
                            digest_table_path: "scale-digests.bin".to_string(),
                            digest_table_byte_offset: 0,
                            digest_stride_bytes: 32,
                            table_sha256: "b".repeat(64),
                        },
                    }],
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
        }];
        contract.selectors[0].resource_count = 4;
        contract.selectors[0].encoding.index_mask = 3;
        contract.selectors[0].mapping = CompiledResourceSelectorMapping::PartitionTemplate {
            partition_template_id: "partition-template".to_string(),
        };
        let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
        let ownership = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &BTreeSet::from(["experts".to_string()]),
        )
        .unwrap();

        let residency = plan_compiled_resource_store_residency_for_ownership(
            &contract,
            &layout,
            &ownership,
            72,
            8,
        )
        .unwrap();

        assert_eq!(residency.maximum_load_wave_payload_bytes, 144);
        assert_eq!(residency.maximum_representation_group_payload_bytes, 136);
        assert_eq!(residency.maximum_representation_load_wave_payload_bytes, 272);
        assert_eq!(
            residency.maximum_representation_load_wave_allocation_padding_bytes,
            42,
        );
        assert_eq!(residency.retained_representation_cache_group_count, 2);
        assert_eq!(residency.retained_representation_cache_wave_count, 1);
        assert_eq!(residency.retained_representation_cache_payload_bytes, 272);
        assert!(residency.retained_representation_cache_identity.is_some());
        assert_eq!(
            residency.retained_representation_cache_allocation_padding_bytes,
            42,
        );
    }

    #[test]
    fn source_projection_owns_only_complete_compiler_integrity_ranges() {
        let contract = projection_contract();
        let projection = |offset, count| VulkanCompiledResourceSourceProjection {
            resources: [
                format!("sha256:{}", "1".repeat(64)),
                format!("sha256:{}", "2".repeat(64)),
            ]
                .into_iter()
                .map(|resource_id| {
                    (
                        resource_id,
                        VulkanCompiledResourceSourceRangeProjection {
                            source_byte_count: 16,
                            byte_offset: offset,
                            byte_count: count,
                        },
                    )
                })
                .collect(),
        };
        let ownership = VulkanCompiledResourceSelectorOwnership::from_source_projections(
            &contract,
            BTreeMap::from([(
                "experts".to_string(),
                BTreeMap::from([(0, projection(8, 8))]),
            )]),
        )
        .unwrap();
        assert!(ownership.owns("experts", 0));
        assert!(ownership.has_source_projections());
        assert!(ownership.is_subset_of(&ownership));
        let whole = VulkanCompiledResourceSelectorOwnership::all(
            &contract,
            &BTreeSet::from(["experts".to_string()]),
        )
        .unwrap();
        assert!(
            !ownership.is_subset_of(&whole),
            "a tensor fragment cannot silently execute against a whole-resource store mapping",
        );

        let resolved = CompiledResourceContractIndex::new(&contract)
            .unwrap()
            .resolve_atomic_group(&contract, &format!("sha256:{}", "3".repeat(64)))
            .unwrap();
        let projected = ownership
            .project_resolved_group(
                "experts",
                0,
                ResolvedCompiledResourceGroup::Atomic(resolved),
            )
            .unwrap();
        assert!(projected
            .resources()
            .iter()
            .all(|resource| resource.ranges.len() == 1 && resource.source_byte_count().unwrap() == 8));
        let index = CompiledResourceContractIndex::new(&contract).unwrap();
        let layout = VulkanCompiledResourceAddressLayout::from_contract(&contract).unwrap();
        let projected_payloads = layout
            .source_payload_bytes_by_address_slot_for_ownership(&contract, &ownership)
            .unwrap();
        assert_eq!(projected_payloads.len(), 2);
        assert!(projected_payloads.values().all(|bytes| *bytes == 8));
        assert_eq!(projected_payloads.values().sum::<usize>(), 16);
        let whole_payloads = layout
            .source_payload_bytes_by_address_slot_for_ownership(&contract, &whole)
            .unwrap();
        assert_eq!(whole_payloads.len(), 2);
        assert!(whole_payloads.values().all(|bytes| *bytes == 16));
        assert_eq!(whole_payloads.values().sum::<usize>(), 32);
        let sparse = compiled_resource_sparse_group_layouts(
            &contract,
            &index,
            &layout,
            &ownership,
            CompiledResourceRepresentation::Source,
        )
        .unwrap();
        assert_eq!(sparse.len(), 1);
        assert_eq!(
            compiled_resource_group_layout_payload_bytes(&sparse[0]).unwrap(),
            16,
        );
        let cache = compiled_resource_selector_cache_policy(&contract, &ownership, 16).unwrap();
        assert_eq!(
            cache.group_payload_bytes.values().copied().sum::<usize>(),
            16,
        );
        let resource_ids = [
            format!("sha256:{}", "1".repeat(64)),
            format!("sha256:{}", "2".repeat(64)),
        ];
        let selection = DeviceResourceSelection {
            concrete_dynamic: resource_ids.iter().cloned().collect(),
            projected_dynamic_bytes: resource_ids
                .iter()
                .cloned()
                .map(|resource_id| (resource_id, 8))
                .collect(),
            projected_group_bytes: BTreeMap::from([(
                format!("sha256:{}", "3".repeat(64)),
                16,
            )]),
            ..DeviceResourceSelection::default()
        };
        let residency = compiled_parameter_residency_bytes(
            &contract,
            &index,
            &selection,
            ResourceResidencyPolicy::DemandPaged,
        )
        .unwrap();
        assert_eq!(residency.maximum_addressable_bytes, 16);
        assert_eq!(residency.staging_headroom_bytes, 16);

        let error = VulkanCompiledResourceSelectorOwnership::from_source_projections(
            &contract,
            BTreeMap::from([(
                "experts".to_string(),
                BTreeMap::from([(0, projection(4, 8))]),
            )]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("exact union"));
    }
}
