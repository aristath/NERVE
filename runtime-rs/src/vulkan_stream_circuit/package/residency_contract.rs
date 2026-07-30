pub const COMPILED_RESOURCE_RESIDENCY_SCHEMA: &str =
    "nerve.compiled_resource_residency.v2";
pub const RESOURCE_IDENTITY_ALGORITHM: &str =
    "nerve.resource_identity_sha256.v1";
pub const RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA: &str =
    "nerve.resource_residency_state_machine.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceResidencyContract {
    pub schema: String,
    pub identity_algorithm: String,
    pub state_machine_schema: String,
    pub supported_policies: Vec<ResourceResidencyPolicy>,
    pub resources: Vec<CompiledImmutableResource>,
    pub atomic_groups: Vec<CompiledAtomicResidencyGroup>,
    pub partition_templates: Vec<CompiledPartitionTemplate>,
    pub bindings: Vec<CompiledResourceBinding>,
    pub selectors: Vec<CompiledResourceSelector>,
    pub checkpoints: Vec<CompiledResidencyCheckpoint>,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ResourceResidencyPolicy {
    DemandRetained,
    Eager,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CompiledResourceLifetime {
    AlwaysResident,
    Dynamic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledImmutableResource {
    pub id: String,
    pub lifetime: CompiledResourceLifetime,
    pub ranges: Vec<CompiledResourceByteRange>,
    pub dependencies: Vec<String>,
    pub compatibility: CompiledResourceCompatibility,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceByteRange {
    pub artifact_path: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub alignment_bytes: usize,
    pub integrity: CompiledResourceRangeIntegrity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceRangeIntegrity {
    pub algorithm: String,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceCompatibility {
    pub device_api: String,
    pub storage_class: String,
    pub read_only: bool,
    pub required_features: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledAtomicResidencyGroup {
    pub id: String,
    pub lifetime: CompiledResourceLifetime,
    pub resource_ids: Vec<String>,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceBinding {
    pub execution_scope: String,
    pub component_id: String,
    pub node_id: String,
    pub parameter_id: String,
    pub mapping: CompiledResourceBindingMapping,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CompiledResourceBindingMapping {
    AtomicGroup {
        atomic_group_id: String,
        resource_id: String,
    },
    PartitionTemplateMember {
        partition_template_id: String,
        resource_identity_seed: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPartitionTemplate {
    pub id: String,
    pub partition_count: usize,
    pub lifetime: CompiledResourceLifetime,
    pub group_identity_seed: String,
    pub member_templates: Vec<CompiledPartitionMemberTemplate>,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledPartitionMemberTemplate {
    pub resource_identity_seed: String,
    pub range_templates: Vec<CompiledResourceRangeTemplate>,
    pub compatibility: CompiledResourceCompatibility,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceRangeTemplate {
    pub artifact_path: String,
    pub base_byte_offset: usize,
    pub stride_bytes: usize,
    pub byte_count: usize,
    pub alignment_bytes: usize,
    pub integrity: CompiledResourceRangeIntegrityTemplate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceRangeIntegrityTemplate {
    pub algorithm: String,
    pub digest_table_path: String,
    pub digest_table_byte_offset: usize,
    pub digest_stride_bytes: usize,
    pub table_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceSelector {
    pub id: String,
    pub execution_scope: String,
    pub component_id: String,
    pub node_id: String,
    pub domain_id: String,
    pub resource_count: usize,
    pub selection_signal: String,
    pub encoding: CompiledResourceSelectionEncoding,
    pub mapping: CompiledResourceSelectorMapping,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceSelectionEncoding {
    pub element_type: CompiledResourceSelectionElementType,
    pub selection_count_per_activation: usize,
    pub index_shift: u32,
    pub index_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompiledResourceSelectionElementType {
    U32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CompiledResourceSelectorMapping {
    GroupTable { atomic_group_ids: Vec<String> },
    PartitionTemplate { partition_template_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResidencyCheckpoint {
    pub id: String,
    pub execution_scope: String,
    pub component_id: String,
    pub after_node_id: String,
    pub resume_node_id: String,
    pub selector_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceResidencyState {
    Absent,
    Requested,
    Loading,
    Resident,
    Failed,
}

impl ResourceResidencyState {
    pub fn can_transition_to(
        self,
        following: Self,
        explicit_lifecycle: bool,
    ) -> bool {
        match self {
            Self::Absent => following == Self::Requested,
            Self::Requested => {
                matches!(following, Self::Loading | Self::Failed | Self::Absent)
            }
            Self::Loading => {
                matches!(following, Self::Resident | Self::Failed | Self::Absent)
            }
            Self::Resident | Self::Failed => {
                explicit_lifecycle && following == Self::Absent
            }
        }
    }
}

pub fn derived_partition_resource_id(
    identity_seed: &str,
    partition_index: usize,
) -> io::Result<String> {
    validate_content_id("partition identity seed", identity_seed)?;
    resource_content_id(
        "partition",
        serde_json::json!({
            "identity_seed": identity_seed,
            "partition_index": partition_index,
        }),
    )
}

fn resource_content_id(kind: &str, payload: Value) -> io::Result<String> {
    let canonical = serde_json::to_vec(&serde_json::json!({
        "algorithm": RESOURCE_IDENTITY_ALGORITHM,
        "kind": kind,
        "payload": payload,
    }))
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    Ok(format!(
        "sha256:{}",
        Sha256::digest(canonical)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

pub(crate) fn compiled_resource_identity(
    resource: &CompiledImmutableResource,
) -> io::Result<String> {
    resource_content_id(
        "resource",
        serde_json::json!({
            "lifetime": resource.lifetime,
            "ranges": resource.ranges.iter().map(|range| serde_json::json!({
                "byte_count": range.byte_count,
                "alignment_bytes": range.alignment_bytes,
                "integrity": range.integrity,
            })).collect::<Vec<_>>(),
            "dependencies": resource.dependencies,
            "compatibility": resource.compatibility,
        }),
    )
}

fn compiled_atomic_group_identity(
    group: &CompiledAtomicResidencyGroup,
) -> io::Result<String> {
    resource_content_id(
        "atomic_group",
        serde_json::json!({
            "lifetime": group.lifetime,
            "resource_ids": group.resource_ids,
            "dependencies": group.dependencies,
        }),
    )
}

fn compiled_partition_template_identity(
    template: &CompiledPartitionTemplate,
) -> io::Result<String> {
    resource_content_id(
        "partition_template",
        serde_json::json!({
            "partition_count": template.partition_count,
            "lifetime": template.lifetime,
            "group_identity_seed": template.group_identity_seed,
            "member_templates": template.member_templates.iter().map(|member| serde_json::json!({
                "resource_identity_seed": member.resource_identity_seed,
                "range_templates": member.range_templates.iter().map(|range| serde_json::json!({
                    "base_byte_offset": range.base_byte_offset,
                    "stride_bytes": range.stride_bytes,
                    "byte_count": range.byte_count,
                    "alignment_bytes": range.alignment_bytes,
                    "integrity": {
                        "algorithm": range.integrity.algorithm,
                        "digest_stride_bytes": range.integrity.digest_stride_bytes,
                        "table_sha256": range.integrity.table_sha256,
                    },
                })).collect::<Vec<_>>(),
                "compatibility": member.compatibility,
            })).collect::<Vec<_>>(),
            "dependencies": template.dependencies,
        }),
    )
}

fn compiled_partition_group_identity_seed(
    partition_count: usize,
    members: &[CompiledPartitionMemberTemplate],
) -> io::Result<String> {
    if partition_count == 0 {
        return invalid_residency(
            "partition group identity requires a positive partition count",
        );
    }
    let seeds = members
        .iter()
        .map(|member| member.resource_identity_seed.as_str())
        .collect::<Vec<_>>();
    if !is_strictly_sorted(&seeds) {
        return invalid_residency(
            "partition group identity requires sorted unique member seeds",
        );
    }
    resource_content_id(
        "partition_group_seed",
        serde_json::json!({
            "partition_count": partition_count,
            "resource_identity_seeds": seeds,
        }),
    )
}

fn compiled_selector_identity(
    selector: &CompiledResourceSelector,
) -> io::Result<String> {
    resource_content_id(
        "selector",
        serde_json::json!({
            "execution_scope": selector.execution_scope,
            "component_id": selector.component_id,
            "node_id": selector.node_id,
            "domain_id": selector.domain_id,
            "resource_count": selector.resource_count,
            "selection_signal": selector.selection_signal,
            "encoding": selector.encoding,
            "mapping": selector.mapping,
        }),
    )
}

fn compiled_checkpoint_identity(
    checkpoint: &CompiledResidencyCheckpoint,
) -> io::Result<String> {
    resource_content_id(
        "checkpoint",
        serde_json::json!({
            "execution_scope": checkpoint.execution_scope,
            "component_id": checkpoint.component_id,
            "after_node_id": checkpoint.after_node_id,
            "resume_node_id": checkpoint.resume_node_id,
            "selector_ids": checkpoint.selector_ids,
        }),
    )
}

fn validate_compiled_resource_residency(
    package_root: &Path,
    manifest: &VulkanResidentModelPackageManifest,
) -> io::Result<()> {
    let contract = &manifest.resource_residency;
    if contract.schema != COMPILED_RESOURCE_RESIDENCY_SCHEMA
        || contract.identity_algorithm != RESOURCE_IDENTITY_ALGORITHM
        || contract.state_machine_schema
            != RESOURCE_RESIDENCY_STATE_MACHINE_SCHEMA
        || contract.supported_policies
            != [
                ResourceResidencyPolicy::DemandRetained,
                ResourceResidencyPolicy::Eager,
            ]
    {
        return invalid_residency("compiled resource residency header is invalid");
    }

    let mut resource_ids = BTreeSet::new();
    let mut artifact_ranges: BTreeMap<&str, Vec<(usize, usize, &str)>> =
        BTreeMap::new();
    for resource in &contract.resources {
        validate_content_id("resource id", &resource.id)?;
        if !resource_ids.insert(resource.id.as_str()) {
            return invalid_residency("compiled resource ids are not unique");
        }
        if resource.ranges.is_empty() {
            return invalid_residency(format!(
                "compiled resource {:?} has no ranges",
                resource.id
            ));
        }
        validate_sorted_content_ids(
            "resource dependencies",
            &resource.dependencies,
        )?;
        validate_resource_compatibility(&resource.compatibility)?;
        let mut previous_range = None;
        for range in &resource.ranges {
            validate_resident_package_relative_path(
                "compiled resource artifact",
                &range.artifact_path,
            )?;
            if range.byte_count == 0
                || !range.alignment_bytes.is_power_of_two()
                || range.byte_offset % range.alignment_bytes != 0
                || range.integrity.algorithm != "sha256"
                || !is_lower_hex_sha256(&range.integrity.digest)
            {
                return invalid_residency(format!(
                    "compiled resource {:?} has an invalid byte range",
                    resource.id
                ));
            }
            let range_key = (
                range.artifact_path.as_str(),
                range.byte_offset,
                range.byte_count,
            );
            if previous_range.is_some_and(|previous| previous >= range_key) {
                return invalid_residency(format!(
                    "compiled resource {:?} ranges are not unique and sorted",
                    resource.id
                ));
            }
            previous_range = Some(range_key);
            let end = range
                .byte_offset
                .checked_add(range.byte_count)
                .ok_or_else(|| invalid_residency_error("resource range overflowed"))?;
            let artifact_bytes = fs::metadata(package_root.join(&range.artifact_path))
                .map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to inspect compiled resource artifact {:?}: {error}",
                            range.artifact_path
                        ),
                    )
                })?
                .len();
            if u64::try_from(end).ok().is_none_or(|end| end > artifact_bytes) {
                return invalid_residency(format!(
                    "compiled resource {:?} exceeds artifact {:?}",
                    resource.id, range.artifact_path
                ));
            }
            artifact_ranges
                .entry(&range.artifact_path)
                .or_default()
                .push((range.byte_offset, end, &resource.id));
        }
        if compiled_resource_identity(resource)? != resource.id {
            return invalid_residency(format!(
                "compiled resource {:?} identity does not match its contract",
                resource.id
            ));
        }
    }
    if contract
        .resources
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
    {
        return invalid_residency("compiled resources are not sorted by id");
    }
    for ranges in artifact_ranges.values_mut() {
        ranges.sort_unstable();
        if ranges
            .windows(2)
            .any(|pair| pair[1].0 < pair[0].1)
        {
            return invalid_residency("compiled resource byte ranges overlap");
        }
    }
    validate_dependency_graph(
        "resource",
        contract
            .resources
            .iter()
            .map(|resource| (resource.id.as_str(), resource.dependencies.as_slice())),
        &resource_ids,
    )?;

    let mut group_ids = BTreeSet::new();
    let mut resource_membership: BTreeMap<&str, usize> = BTreeMap::new();
    for group in &contract.atomic_groups {
        validate_content_id("atomic group id", &group.id)?;
        if !group_ids.insert(group.id.as_str())
            || group.resource_ids.is_empty()
        {
            return invalid_residency(
                "compiled atomic groups require unique ids and members",
            );
        }
        validate_sorted_content_ids("atomic group resources", &group.resource_ids)?;
        validate_sorted_content_ids(
            "atomic group dependencies",
            &group.dependencies,
        )?;
        for resource_id in &group.resource_ids {
            let resource = contract
                .resources
                .iter()
                .find(|resource| resource.id == *resource_id)
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled atomic group references an unknown resource",
                    )
                })?;
            if resource.lifetime != group.lifetime {
                return invalid_residency(
                    "compiled atomic group lifetime disagrees with a resource",
                );
            }
            *resource_membership.entry(resource_id).or_default() += 1;
        }
        if compiled_atomic_group_identity(group)? != group.id {
            return invalid_residency(format!(
                "compiled atomic group {:?} identity does not match its contract",
                group.id
            ));
        }
    }
    if contract
        .atomic_groups
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
        || resource_ids.iter().any(|resource_id| {
            resource_membership.get(resource_id).copied() != Some(1)
        })
    {
        return invalid_residency(
            "compiled resources must belong to one sorted atomic group",
        );
    }
    validate_dependency_graph(
        "atomic group",
        contract
            .atomic_groups
            .iter()
            .map(|group| (group.id.as_str(), group.dependencies.as_slice())),
        &group_ids,
    )?;

    let mut template_ids = BTreeSet::new();
    for template in &contract.partition_templates {
        validate_partition_template(package_root, template, &group_ids)?;
        if !template_ids.insert(template.id.as_str()) {
            return invalid_residency("compiled partition template ids are not unique");
        }
    }
    if contract
        .partition_templates
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
    {
        return invalid_residency("compiled partition templates are not sorted");
    }
    validate_compiled_partition_storage(package_root, contract)?;

    validate_bindings_against_package(manifest, &group_ids, &template_ids)?;
    validate_selectors_and_checkpoints(manifest, &template_ids)
}

fn validate_resource_compatibility(
    compatibility: &CompiledResourceCompatibility,
) -> io::Result<()> {
    if compatibility.device_api != "vulkan"
        || compatibility.storage_class != "storage_buffer"
        || !compatibility.read_only
        || compatibility
            .required_features
            .iter()
            .any(|feature| feature.is_empty())
        || !is_strictly_sorted(&compatibility.required_features)
    {
        return invalid_residency("compiled resource compatibility is invalid");
    }
    Ok(())
}

fn validate_partition_template(
    package_root: &Path,
    template: &CompiledPartitionTemplate,
    group_ids: &BTreeSet<&str>,
) -> io::Result<()> {
    validate_content_id("partition template id", &template.id)?;
    validate_content_id(
        "partition group identity seed",
        &template.group_identity_seed,
    )?;
    if template.partition_count == 0
        || template.lifetime != CompiledResourceLifetime::Dynamic
        || template.member_templates.is_empty()
    {
        return invalid_residency("compiled partition template is invalid");
    }
    validate_sorted_content_ids(
        "partition template dependencies",
        &template.dependencies,
    )?;
    if template
        .dependencies
        .iter()
        .any(|dependency| !group_ids.contains(dependency.as_str()))
    {
        return invalid_residency(
            "compiled partition template has an unknown dependency",
        );
    }
    let mut member_seeds = Vec::new();
    for member in &template.member_templates {
        validate_content_id(
            "partition resource identity seed",
            &member.resource_identity_seed,
        )?;
        member_seeds.push(member.resource_identity_seed.as_str());
        validate_resource_compatibility(&member.compatibility)?;
        if member.range_templates.is_empty() {
            return invalid_residency(
                "compiled partition member has no range templates",
            );
        }
        for range in &member.range_templates {
            validate_resident_package_relative_path(
                "partition range artifact",
                &range.artifact_path,
            )?;
            validate_resident_package_relative_path(
                "partition digest table",
                &range.integrity.digest_table_path,
            )?;
            if range.stride_bytes == 0
                || range.byte_count == 0
                || !range.alignment_bytes.is_power_of_two()
                || range.base_byte_offset % range.alignment_bytes != 0
                || range.stride_bytes % range.alignment_bytes != 0
                || range.integrity.algorithm != "sha256_table"
                || !is_lower_hex_sha256(&range.integrity.table_sha256)
            {
                return invalid_residency(
                    "compiled partition range template is invalid",
                );
            }
            if range.stride_bytes < range.byte_count {
                return invalid_residency(
                    "partition range stride overlaps adjacent resources",
                );
            }
            if range.integrity.digest_stride_bytes != 32
                || range.integrity.digest_table_byte_offset % 32 != 0
            {
                return invalid_residency(
                    "partition SHA-256 table must use aligned 32-byte entries",
                );
            }
            let partition_offset = (template.partition_count - 1)
                .checked_mul(range.stride_bytes)
                .and_then(|offset| range.base_byte_offset.checked_add(offset))
                .and_then(|offset| offset.checked_add(range.byte_count))
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled partition range template overflowed",
                    )
                })?;
            if u64::try_from(partition_offset).ok().is_none_or(|end| {
                fs::metadata(package_root.join(&range.artifact_path))
                    .map(|metadata| end > metadata.len())
                    .unwrap_or(true)
            }) {
                return invalid_residency(
                    "compiled partition range template exceeds its artifact",
                );
            }
            let digest_end = (template.partition_count - 1)
                .checked_mul(range.integrity.digest_stride_bytes)
                .and_then(|offset| {
                    range
                        .integrity
                        .digest_table_byte_offset
                        .checked_add(offset)
                })
                .and_then(|offset| offset.checked_add(32))
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled partition digest table range overflowed",
                    )
                })?;
            if u64::try_from(digest_end).ok().is_none_or(|end| {
                fs::metadata(
                    package_root.join(&range.integrity.digest_table_path),
                )
                .map(|metadata| end > metadata.len())
                .unwrap_or(true)
            }) {
                return invalid_residency(
                    "compiled partition digest table is too small",
                );
            }
        }
    }
    if !is_strictly_sorted(&member_seeds)
        || compiled_partition_group_identity_seed(
            template.partition_count,
            &template.member_templates,
        )? != template.group_identity_seed
        || compiled_partition_template_identity(template)? != template.id
    {
        return invalid_residency(
            "compiled partition template identity or member order is invalid",
        );
    }
    Ok(())
}

fn validate_bindings_against_package(
    manifest: &VulkanResidentModelPackageManifest,
    group_ids: &BTreeSet<&str>,
    template_ids: &BTreeSet<&str>,
) -> io::Result<()> {
    let semantics = compiled_parameter_semantics(manifest)?;
    let mut keys = Vec::new();
    let mut bound_semantics = BTreeSet::new();
    let mut bound_concrete_resources = BTreeSet::new();
    let mut bound_partition_members = BTreeSet::new();
    for binding in &manifest.resource_residency.bindings {
        let mapping_key = match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup {
                atomic_group_id,
                resource_id,
            } => {
                let group = manifest
                    .resource_residency
                    .atomic_groups
                    .iter()
                    .find(|group| group.id == *atomic_group_id);
                if !group_ids.contains(atomic_group_id.as_str())
                    || group.is_none_or(|group| {
                        !group.resource_ids.contains(resource_id)
                    })
                {
                    return invalid_residency(
                        "compiled resource binding maps a resource outside its atomic group",
                    );
                }
                bound_concrete_resources.insert(resource_id.as_str());
                format!("atomic_group|{atomic_group_id}|{resource_id}|")
            }
            CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id,
                resource_identity_seed,
            } => {
                validate_content_id(
                    "resource binding partition template id",
                    partition_template_id,
                )?;
                validate_content_id(
                    "resource binding partition resource seed",
                    resource_identity_seed,
                )?;
                let template = manifest
                    .resource_residency
                    .partition_templates
                    .iter()
                    .find(|template| template.id == *partition_template_id);
                if !template_ids.contains(partition_template_id.as_str())
                    || template.is_none_or(|template| {
                        !template.member_templates.iter().any(|member| {
                            member.resource_identity_seed
                                == *resource_identity_seed
                        })
                    })
                {
                    return invalid_residency(
                        "compiled resource binding maps an unknown partition member",
                    );
                }
                bound_partition_members.insert((
                    partition_template_id.as_str(),
                    resource_identity_seed.as_str(),
                ));
                format!(
                    "partition_template_member||{partition_template_id}|{resource_identity_seed}"
                )
            }
        };
        let key = (
            binding.execution_scope.clone(),
            binding.component_id.clone(),
            binding.node_id.clone(),
            binding.parameter_id.clone(),
            mapping_key,
        );
        let semantic_key = (
            binding.execution_scope.clone(),
            binding.component_id.clone(),
            binding.node_id.clone(),
            binding.parameter_id.clone(),
        );
        if [
            binding.execution_scope.as_str(),
            binding.component_id.as_str(),
            binding.node_id.as_str(),
            binding.parameter_id.as_str(),
        ]
            .iter()
            .any(|value| value.trim().is_empty())
            || !semantics.contains(&semantic_key)
            || !bound_semantics.insert(semantic_key)
        {
            return invalid_residency(
                "compiled resource binding does not match package semantics",
            );
        }
        keys.push(key);
    }
    if !is_strictly_sorted(&keys) || bound_semantics != semantics {
        return invalid_residency(
            "compiled resource bindings must exactly cover package parameters",
        );
    }
    let expected_concrete_resources = manifest
        .resource_residency
        .atomic_groups
        .iter()
        .flat_map(|group| group.resource_ids.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let expected_partition_members = manifest
        .resource_residency
        .partition_templates
        .iter()
        .flat_map(|template| {
            template.member_templates.iter().map(|member| {
                (
                    template.id.as_str(),
                    member.resource_identity_seed.as_str(),
                )
            })
        })
        .collect::<BTreeSet<_>>();
    if bound_concrete_resources != expected_concrete_resources
        || bound_partition_members != expected_partition_members
    {
        return invalid_residency(
            "compiled resource bindings do not cover atomic membership",
        );
    }
    Ok(())
}

fn compiled_parameter_semantics(
    manifest: &VulkanResidentModelPackageManifest,
) -> io::Result<BTreeSet<(String, String, String, String)>> {
    let mut semantics = BTreeSet::new();
    collect_graph_parameter_semantics(
        "target",
        &manifest.circuit_graph,
        &mut semantics,
    )?;
    for decoder in &manifest.speculative_decoders {
        let scope = format!("draft:{}", decoder.id);
        collect_graph_parameter_semantics(
            &scope,
            &decoder.circuit_graph,
            &mut semantics,
        )?;
    }
    Ok(semantics)
}

fn collect_graph_parameter_semantics(
    scope: &str,
    graph: &VulkanResidentPackageCircuitGraph,
    semantics: &mut BTreeSet<(String, String, String, String)>,
) -> io::Result<()> {
    for component in &graph.components {
        for node in &component.circuit.nodes {
            for parameter_id in &node.params {
                if !component.params.refs.contains_key(parameter_id) {
                    return invalid_residency(
                        "compiled circuit parameter semantics are invalid",
                    );
                }
                semantics.insert((
                    scope.to_string(),
                    component.component_id.clone(),
                    node.id.clone(),
                    parameter_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_selectors_and_checkpoints(
    manifest: &VulkanResidentModelPackageManifest,
    template_ids: &BTreeSet<&str>,
) -> io::Result<()> {
    let contract = &manifest.resource_residency;
    let mut selector_ids = BTreeSet::new();
    let mut selected_groups = BTreeSet::new();
    let mut selected_templates = BTreeSet::new();
    for selector in &contract.selectors {
        validate_content_id("selector id", &selector.id)?;
        if !selector_ids.insert(selector.id.as_str())
            || selector.resource_count == 0
            || [
                selector.execution_scope.as_str(),
                selector.component_id.as_str(),
                selector.node_id.as_str(),
                selector.domain_id.as_str(),
                selector.selection_signal.as_str(),
            ]
            .iter()
            .any(|value| value.trim().is_empty())
        {
            return invalid_residency("compiled resource selector is invalid");
        }
        let (_, selector_node) = scoped_component_node(
            manifest,
            &selector.execution_scope,
            &selector.component_id,
            &selector.node_id,
        )
        .ok_or_else(|| {
            invalid_residency_error(
                "compiled selector does not name a packaged node",
            )
        })?;
        let selection_domain = selector_node
            .attrs
            .as_object()
            .and_then(|attrs| attrs.get("selection_domain"))
            .and_then(Value::as_object);
        let encoded_selection = serde_json::to_value(&selector.encoding)
            .map_err(|error| invalid_residency_error(error.to_string()))?;
        if selection_domain.is_none_or(|domain| {
            domain.len() != 4
                || ![
                    "id",
                    "resource_count",
                    "selection_signal",
                    "encoding",
                ]
                .iter()
                .all(|field| domain.contains_key(*field))
        }) || selection_domain
            .and_then(|domain| domain.get("id"))
            .and_then(Value::as_str)
            != Some(selector.domain_id.as_str())
            || selection_domain
                .and_then(|domain| domain.get("resource_count"))
                .and_then(Value::as_u64)
                .and_then(|count| usize::try_from(count).ok())
                != Some(selector.resource_count)
            || selection_domain
                .and_then(|domain| domain.get("selection_signal"))
                .and_then(Value::as_str)
                != Some(selector.selection_signal.as_str())
            || selection_domain
                .and_then(|domain| domain.get("encoding"))
                != Some(&encoded_selection)
        {
            return invalid_residency(
                "compiled selector disagrees with its node selection domain",
            );
        }
        if !selector_node
            .outputs
            .iter()
            .any(|output| output == &selector.selection_signal)
            || selector.encoding.selection_count_per_activation == 0
            || selector.encoding.index_shift >= u32::BITS
            || selector.encoding.index_mask == 0
            || selector.encoding.index_mask
                > u32::MAX >> selector.encoding.index_shift
            || (selector.encoding.index_mask != u32::MAX
                && selector.encoding.index_mask
                    & (selector.encoding.index_mask + 1)
                    != 0)
            || u32::try_from(selector.resource_count - 1)
                .ok()
                .is_none_or(|maximum_index| {
                    maximum_index & selector.encoding.index_mask
                        != maximum_index
                })
        {
            return invalid_residency(
                "compiled selector has an invalid physical selection encoding",
            );
        }
        match &selector.mapping {
            CompiledResourceSelectorMapping::GroupTable { atomic_group_ids } => {
                if atomic_group_ids.len() != selector.resource_count {
                    return invalid_residency(
                        "selector group table has the wrong length",
                    );
                }
                for group_id in atomic_group_ids {
                    let group = contract
                        .atomic_groups
                        .iter()
                        .find(|group| group.id == *group_id)
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "selector maps an unknown atomic group",
                            )
                        })?;
                    if group.lifetime != CompiledResourceLifetime::Dynamic {
                        return invalid_residency(
                            "selector maps an always-resident atomic group",
                        );
                    }
                    selected_groups.insert(group_id.as_str());
                }
            }
            CompiledResourceSelectorMapping::PartitionTemplate {
                partition_template_id,
            } => {
                let template = contract
                    .partition_templates
                    .iter()
                    .find(|template| template.id == *partition_template_id)
                    .ok_or_else(|| {
                        invalid_residency_error(
                            "selector maps an unknown partition template",
                        )
                    })?;
                if template.partition_count != selector.resource_count {
                    return invalid_residency(
                        "selector partition mapping is inconsistent",
                    );
                }
                selected_templates.insert(partition_template_id.as_str());
            }
        }
        if compiled_selector_identity(selector)? != selector.id {
            return invalid_residency(
                "compiled selector identity does not match its semantics",
            );
        }
    }
    if contract
        .selectors
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
        || selected_groups
            != contract
                .atomic_groups
                .iter()
                .filter(|group| {
                    group.lifetime == CompiledResourceLifetime::Dynamic
                })
                .map(|group| group.id.as_str())
                .collect()
        || selected_templates != *template_ids
    {
        return invalid_residency(
            "compiled dynamic resources are not mapped by a selector",
        );
    }

    let mut checkpoint_ids = BTreeSet::new();
    let mut selector_owners: BTreeMap<&str, usize> = BTreeMap::new();
    for checkpoint in &contract.checkpoints {
        validate_content_id("residency checkpoint id", &checkpoint.id)?;
        if !checkpoint_ids.insert(checkpoint.id.as_str())
            || checkpoint.selector_ids.is_empty()
            || [
                checkpoint.execution_scope.as_str(),
                checkpoint.component_id.as_str(),
                checkpoint.after_node_id.as_str(),
                checkpoint.resume_node_id.as_str(),
            ]
            .iter()
            .any(|value| value.trim().is_empty())
        {
            return invalid_residency("compiled residency checkpoint is invalid");
        }
        let (after_index, _) = scoped_component_node(
            manifest,
            &checkpoint.execution_scope,
            &checkpoint.component_id,
            &checkpoint.after_node_id,
        )
        .ok_or_else(|| {
            invalid_residency_error(
                "compiled checkpoint has an unknown after-node",
            )
        })?;
        let (resume_index, _) = scoped_component_node(
            manifest,
            &checkpoint.execution_scope,
            &checkpoint.component_id,
            &checkpoint.resume_node_id,
        )
        .ok_or_else(|| {
            invalid_residency_error(
                "compiled checkpoint has an unknown resume-node",
            )
        })?;
        if after_index >= resume_index {
            return invalid_residency(
                "compiled checkpoint does not resume after its selector",
            );
        }
        validate_sorted_content_ids(
            "checkpoint selectors",
            &checkpoint.selector_ids,
        )?;
        for selector_id in &checkpoint.selector_ids {
            let selector = contract
                .selectors
                .iter()
                .find(|selector| selector.id == *selector_id)
                .ok_or_else(|| {
                    invalid_residency_error(
                        "checkpoint references an unknown selector",
                    )
                })?;
            if selector.execution_scope != checkpoint.execution_scope
                || selector.component_id != checkpoint.component_id
                || selector.node_id != checkpoint.after_node_id
            {
                return invalid_residency(
                    "checkpoint crosses a selector execution boundary",
                );
            }
            *selector_owners.entry(selector_id).or_default() += 1;
        }
        if compiled_checkpoint_identity(checkpoint)? != checkpoint.id {
            return invalid_residency(
                "compiled checkpoint identity does not match its semantics",
            );
        }
    }
    if contract
        .checkpoints
        .windows(2)
        .any(|pair| pair[0].id >= pair[1].id)
        || selector_ids
            .iter()
            .any(|selector_id| selector_owners.get(selector_id).copied() != Some(1))
    {
        return invalid_residency(
            "compiled selectors must belong to one sorted checkpoint",
        );
    }
    Ok(())
}

fn scoped_component_node<'a>(
    manifest: &'a VulkanResidentModelPackageManifest,
    execution_scope: &str,
    component_id: &str,
    node_id: &str,
) -> Option<(usize, &'a CircuitNode)> {
    let graph = if execution_scope == "target" {
        &manifest.circuit_graph
    } else {
        let decoder_id = execution_scope.strip_prefix("draft:")?;
        &manifest
            .speculative_decoders
            .iter()
            .find(|decoder| decoder.id == decoder_id)?
            .circuit_graph
    };
    let component = graph
        .components
        .iter()
        .find(|component| component.component_id == component_id)?;
    component
        .circuit
        .nodes
        .iter()
        .enumerate()
        .find(|(_, node)| node.id == node_id)
}

pub(super) fn validate_content_id(label: &str, value: &str) -> io::Result<()> {
    if !value
        .strip_prefix("sha256:")
        .is_some_and(is_lower_hex_sha256)
    {
        return invalid_residency(format!(
            "{label} must be a content-addressed SHA-256 id"
        ));
    }
    Ok(())
}

fn validate_sorted_content_ids(
    label: &str,
    values: &[String],
) -> io::Result<()> {
    for value in values {
        validate_content_id(label, value)?;
    }
    if !is_strictly_sorted(values) {
        return invalid_residency(format!("{label} must be unique and sorted"));
    }
    Ok(())
}

fn validate_dependency_graph<'a>(
    label: &str,
    nodes: impl IntoIterator<Item = (&'a str, &'a [String])>,
    known: &BTreeSet<&str>,
) -> io::Result<()> {
    let graph = nodes.into_iter().collect::<BTreeMap<_, _>>();
    if graph.values().flat_map(|dependencies| dependencies.iter()).any(|dependency| {
        !known.contains(dependency.as_str())
    }) {
        return invalid_residency(format!("{label} has an unknown dependency"));
    }
    fn visit<'a>(
        node: &'a str,
        graph: &BTreeMap<&'a str, &'a [String]>,
        visiting: &mut BTreeSet<&'a str>,
        visited: &mut BTreeSet<&'a str>,
    ) -> bool {
        if visiting.contains(node) {
            return false;
        }
        if visited.contains(node) {
            return true;
        }
        visiting.insert(node);
        let valid = graph[node]
            .iter()
            .all(|dependency| visit(dependency, graph, visiting, visited));
        visiting.remove(node);
        visited.insert(node);
        valid
    }
    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    if graph
        .keys()
        .any(|node| !visit(node, &graph, &mut visiting, &mut visited))
    {
        return invalid_residency(format!("{label} dependencies contain a cycle"));
    }
    Ok(())
}

fn is_strictly_sorted<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn invalid_residency<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_residency_error(message))
}

fn invalid_residency_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod shared_template_tests {
    use super::*;
    use crate::test_support::tiny_model_package_manifest_path;

    #[test]
    fn compatible_partition_template_is_shareable_across_selectors() {
        let mut manifest = VulkanResidentModelPackageManifest::from_json_file(
            &tiny_model_package_manifest_path(),
        )
        .unwrap();
        let template_id =
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_string();
        manifest.resource_residency.partition_templates =
            vec![CompiledPartitionTemplate {
                id: template_id.clone(),
                partition_count: 3,
                lifetime: CompiledResourceLifetime::Dynamic,
                group_identity_seed:
                    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                        .to_string(),
                member_templates: Vec::new(),
                dependencies: Vec::new(),
            }];

        let prototype = manifest
            .circuit_graph
            .components
            .iter()
            .find(|component| component.component_id == "layer_00")
            .unwrap()
            .clone();
        let mut selectors = Vec::new();
        let mut checkpoints = Vec::new();
        for component_id in ["selector_owner_a", "selector_owner_b"] {
            let mut component = prototype.clone();
            component.component_id = component_id.to_string();
            let selector_node_id = component.circuit.nodes[0].id.clone();
            let resume_node_id = component.circuit.nodes[1].id.clone();
            let selection_signal =
                component.circuit.nodes[0].outputs[0].clone();
            component.circuit.nodes[0].attrs = serde_json::json!({
                "selection_domain": {
                    "id": "shared_partitions",
                    "resource_count": 3,
                    "selection_signal": selection_signal.clone(),
                    "encoding": {
                        "element_type": "u32",
                        "selection_count_per_activation": 1,
                        "index_shift": 0,
                        "index_mask": 0xffff
                    }
                }
            });
            manifest.circuit_graph.components.push(component);

            let mut selector = CompiledResourceSelector {
                id: String::new(),
                execution_scope: "target".to_string(),
                component_id: component_id.to_string(),
                node_id: selector_node_id.clone(),
                domain_id: "shared_partitions".to_string(),
                resource_count: 3,
                selection_signal: selection_signal.clone(),
                encoding: CompiledResourceSelectionEncoding {
                    element_type: CompiledResourceSelectionElementType::U32,
                    selection_count_per_activation: 1,
                    index_shift: 0,
                    index_mask: 0xffff,
                },
                mapping: CompiledResourceSelectorMapping::PartitionTemplate {
                    partition_template_id: template_id.clone(),
                },
            };
            selector.id = compiled_selector_identity(&selector).unwrap();
            let mut checkpoint = CompiledResidencyCheckpoint {
                id: String::new(),
                execution_scope: "target".to_string(),
                component_id: component_id.to_string(),
                after_node_id: selector_node_id,
                resume_node_id,
                selector_ids: vec![selector.id.clone()],
            };
            checkpoint.id = compiled_checkpoint_identity(&checkpoint).unwrap();
            selectors.push(selector);
            checkpoints.push(checkpoint);
        }
        selectors.sort_by(|left, right| left.id.cmp(&right.id));
        checkpoints.sort_by(|left, right| left.id.cmp(&right.id));
        manifest.resource_residency.selectors = selectors;
        manifest.resource_residency.checkpoints = checkpoints;

        let template_ids = BTreeSet::from([template_id.as_str()]);
        validate_selectors_and_checkpoints(&manifest, &template_ids).unwrap();
    }
}
