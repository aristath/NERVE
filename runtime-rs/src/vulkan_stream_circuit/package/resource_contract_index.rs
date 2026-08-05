use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct CompiledResourceContractIndex {
    resources: HashMap<String, usize>,
    atomic_groups: HashMap<String, usize>,
    resource_atomic_groups: HashMap<String, Vec<usize>>,
    partition_templates: HashMap<String, usize>,
    selectors: HashMap<String, usize>,
}

impl CompiledResourceContractIndex {
    pub(crate) fn new(
        contract: &CompiledResourceResidencyContract,
    ) -> io::Result<Self> {
        let resources = indexed_resource_contract_ids(
                "resource",
                contract.resources.iter().map(|resource| resource.id.as_str()),
            )?;
        let atomic_groups = indexed_resource_contract_ids(
                "atomic group",
                contract.atomic_groups.iter().map(|group| group.id.as_str()),
            )?;
        let mut resource_atomic_groups = HashMap::<String, Vec<usize>>::new();
        for (group_index, group) in contract.atomic_groups.iter().enumerate() {
            for resource_id in &group.resource_ids {
                resource_atomic_groups
                    .entry(resource_id.clone())
                    .or_default()
                    .push(group_index);
            }
        }
        Ok(Self {
            resources,
            atomic_groups,
            resource_atomic_groups,
            partition_templates: indexed_resource_contract_ids(
                "partition template",
                contract
                    .partition_templates
                    .iter()
                    .map(|template| template.id.as_str()),
            )?,
            selectors: indexed_resource_contract_ids(
                "selector",
                contract.selectors.iter().map(|selector| selector.id.as_str()),
            )?,
        })
    }

    pub(crate) fn resource<'a>(
        &self,
        contract: &'a CompiledResourceResidencyContract,
        id: &str,
    ) -> Option<&'a CompiledImmutableResource> {
        self.resources
            .get(id)
            .and_then(|index| contract.resources.get(*index))
    }

    pub(crate) fn atomic_group<'a>(
        &self,
        contract: &'a CompiledResourceResidencyContract,
        id: &str,
    ) -> Option<&'a CompiledAtomicResidencyGroup> {
        self.atomic_groups
            .get(id)
            .and_then(|index| contract.atomic_groups.get(*index))
    }

    pub(crate) fn atomic_group_indices_for_resource(
        &self,
        resource_id: &str,
    ) -> &[usize] {
        self.resource_atomic_groups
            .get(resource_id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub(crate) fn partition_template<'a>(
        &self,
        contract: &'a CompiledResourceResidencyContract,
        id: &str,
    ) -> Option<&'a CompiledPartitionTemplate> {
        self.partition_templates
            .get(id)
            .and_then(|index| contract.partition_templates.get(*index))
    }

    pub(crate) fn selector<'a>(
        &self,
        contract: &'a CompiledResourceResidencyContract,
        id: &str,
    ) -> Option<&'a CompiledResourceSelector> {
        self.selectors
            .get(id)
            .and_then(|index| contract.selectors.get(*index))
    }

    pub(crate) fn resolve_atomic_group(
        &self,
        contract: &CompiledResourceResidencyContract,
        atomic_group_id: &str,
    ) -> io::Result<ResolvedCompiledAtomicGroup> {
        validate_content_id("atomic group id", atomic_group_id)?;
        let group = self
            .atomic_group(contract, atomic_group_id)
            .ok_or_else(|| invalid_residency_error("unknown atomic group"))?;
        let resources = group
            .resource_ids
            .iter()
            .map(|resource_id| {
                let resource = self.resource(contract, resource_id).ok_or_else(|| {
                    invalid_residency_error(format!(
                        "atomic group references unknown resource {resource_id:?}"
                    ))
                })?;
                Ok(ResolvedCompiledResource {
                    id: resource.id.clone(),
                    ranges: resource
                        .ranges
                        .iter()
                        .map(|range| ResolvedCompiledResourceRange {
                            artifact_path: range.artifact_path.clone(),
                            byte_offset: range.byte_offset,
                            byte_count: range.byte_count,
                            alignment_bytes: range.alignment_bytes,
                            sha256: range.integrity.digest.clone(),
                        })
                        .collect(),
                    compatibility: resource.compatibility.clone(),
                    resident_derivation: resource.resident_derivation.clone(),
                })
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(ResolvedCompiledAtomicGroup {
            schema: RESOLVED_ATOMIC_GROUP_SCHEMA.to_string(),
            id: group.id.clone(),
            resource_ids: group.resource_ids.clone(),
            dependencies: group.dependencies.clone(),
            resources,
        })
    }
}

fn indexed_resource_contract_ids<'a>(
    label: &str,
    ids: impl Iterator<Item = &'a str>,
) -> io::Result<HashMap<String, usize>> {
    let mut indexed = HashMap::new();
    for (index, id) in ids.enumerate() {
        if indexed.insert(id.to_string(), index).is_some() {
            return invalid_residency(format!(
                "compiled resource contract repeats {label} id {id:?}"
            ));
        }
    }
    Ok(indexed)
}

#[cfg(test)]
mod resource_contract_index_tests {
    use super::*;

    fn content_id(value: usize) -> String {
        format!("sha256:{value:064x}")
    }

    fn compatibility() -> CompiledResourceCompatibility {
        CompiledResourceCompatibility {
            device_api: "vulkan".to_string(),
            storage_class: "storage_buffer".to_string(),
            read_only: true,
            required_features: Vec::new(),
        }
    }

    fn contract_with_resource_count(resource_count: usize) -> CompiledResourceResidencyContract {
        let resources = (0..resource_count)
            .map(|index| CompiledImmutableResource {
                id: content_id(index + 1),
                lifetime: CompiledResourceLifetime::Dynamic,
                ranges: vec![CompiledResourceByteRange {
                    artifact_path: "weights/bank.bin".to_string(),
                    byte_offset: index * 4,
                    byte_count: 4,
                    alignment_bytes: 4,
                    integrity: CompiledResourceRangeIntegrity {
                        algorithm: "sha256".to_string(),
                        digest: format!("{:064x}", index + 1),
                    },
                }],
                dependencies: Vec::new(),
                compatibility: compatibility(),
                resident_derivation: None,
            })
            .collect::<Vec<_>>();
        let group_id = content_id(resource_count + 1);
        CompiledResourceResidencyContract {
            schema: COMPILED_RESOURCE_RESIDENCY_SCHEMA.to_string(),
            identity_algorithm: RESOURCE_IDENTITY_ALGORITHM.to_string(),
            state_machine_schema: RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA.to_string(),
            supported_policies: vec![ResourceResidencyPolicy::DemandRetained],
            atomic_groups: vec![CompiledAtomicResidencyGroup {
                id: group_id,
                lifetime: CompiledResourceLifetime::Dynamic,
                resource_ids: resources
                    .last()
                    .map(|resource| vec![resource.id.clone()])
                    .unwrap_or_default(),
                dependencies: Vec::new(),
            }],
            resources,
            partition_templates: Vec::new(),
            bindings: Vec::new(),
            selectors: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn compiled_resource_contract_index_rejects_duplicate_namespace_ids() {
        let mut contract = contract_with_resource_count(2);
        contract.resources[1].id = contract.resources[0].id.clone();

        let error = CompiledResourceContractIndex::new(&contract).unwrap_err();

        assert!(error.to_string().contains("repeats resource id"));
    }

    #[test]
    fn compiled_resource_contract_index_resolves_large_contract_like_public_contract_api() {
        let contract = contract_with_resource_count(16_384);
        let group_id = contract.atomic_groups[0].id.clone();
        let index = CompiledResourceContractIndex::new(&contract).unwrap();

        let indexed = index.resolve_atomic_group(&contract, &group_id).unwrap();
        let canonical = resolve_compiled_atomic_group(&contract, &group_id).unwrap();

        assert_eq!(indexed, canonical);
        assert_eq!(
            index
                .resource(&contract, &contract.resources[12_345].id)
                .unwrap(),
            &contract.resources[12_345]
        );
        assert_eq!(
            index.atomic_group_indices_for_resource(&contract.resources[16_383].id),
            &[0]
        );
        assert!(
            index
                .atomic_group_indices_for_resource(&contract.resources[12_345].id)
                .is_empty()
        );
    }
}
