pub const COMPILED_RESOURCE_RESIDENCY_SCHEMA: &str =
    "nerve.compiled_resource_residency.v4";
pub const RESIDENT_DERIVATION_SCHEMA: &str =
    "nerve.resident_derivation.v1";
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
    DemandPaged,
    DemandRetained,
    Eager,
}

impl ResourceResidencyPolicy {
    pub fn as_runtime_name(self) -> &'static str {
        match self {
            Self::DemandPaged => "demand-paged",
            Self::DemandRetained => "demand-retained",
            Self::Eager => "eager",
        }
    }

    pub fn is_demand_loaded(self) -> bool {
        matches!(self, Self::DemandPaged | Self::DemandRetained)
    }

    pub fn evicts_inactive_resources(self) -> bool {
        self == Self::DemandPaged
    }

    pub(crate) fn required_compiled_loading_policy(self) -> Self {
        match self {
            Self::DemandPaged => Self::DemandRetained,
            policy => policy,
        }
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident_derivation: Option<CompiledResourceResidentDerivation>,
}

impl CompiledImmutableResource {
    pub fn source_byte_count(&self) -> io::Result<usize> {
        self.ranges.iter().try_fold(0usize, |total, range| {
            total.checked_add(range.byte_count).ok_or_else(|| {
                invalid_residency_error("compiled resource source size overflowed")
            })
        })
    }

    pub fn resident_byte_count_for(
        &self,
        representation: CompiledResourceRepresentation,
    ) -> io::Result<usize> {
        match (representation, &self.resident_derivation) {
            (CompiledResourceRepresentation::ResidentDerivation, Some(derivation)) => {
                Ok(derivation.resident_byte_count)
            }
            _ => self.source_byte_count(),
        }
    }

    pub fn supports_representation(
        &self,
        representation: CompiledResourceRepresentation,
    ) -> bool {
        representation == CompiledResourceRepresentation::Source
            || self.resident_derivation.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledResourceResidentDerivation {
    pub schema: String,
    pub kind: CompiledResourceResidentDerivationKind,
    pub source_byte_count: usize,
    pub resident_byte_count: usize,
    pub required_features: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompiledResourceResidentDerivationKind {
    Mxfp4E2m1ToFp8E4m3,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum CompiledResourceRepresentation {
    #[default]
    Source = 0,
    ResidentDerivation = 1,
}

impl CompiledResourceRepresentation {
    pub fn address_tag(self) -> u32 {
        self as u32
    }
}

impl CompiledResourceResidentDerivation {
    pub fn validate_for_source_byte_count(
        &self,
        source_byte_count: usize,
    ) -> std::io::Result<()> {
        let expected_resident_byte_count = source_byte_count
            .checked_mul(2)
            .ok_or_else(|| invalid_residency_error("resident derivation size overflowed"))?;
        let expected_features = [
            "shader_float8",
            "shader_int8",
            "shader_mixed_float_dot_product_float8_acc_float32",
        ];
        if self.schema != RESIDENT_DERIVATION_SCHEMA
            || self.kind
                != CompiledResourceResidentDerivationKind::Mxfp4E2m1ToFp8E4m3
            || self.source_byte_count != source_byte_count
            || self.resident_byte_count != expected_resident_byte_count
            || self.required_features
                != expected_features.map(str::to_string)
        {
            return invalid_residency("compiled resident derivation is inconsistent");
        }
        Ok(())
    }
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
    SelectedAtomicGroup {
        atomic_group_id: String,
        resource_id: String,
        selection_signal: String,
        selector_index: usize,
        parameter_slot: usize,
    },
    PartitionTemplateMember {
        partition_template_id: String,
        resource_identity_seed: String,
        selection_signal: String,
        parameter_slot: usize,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resident_derivation: Option<CompiledResourceResidentDerivation>,
}

impl CompiledPartitionMemberTemplate {
    pub fn source_byte_count(&self) -> io::Result<usize> {
        self.range_templates
            .iter()
            .try_fold(0usize, |total, range| {
                total.checked_add(range.byte_count).ok_or_else(|| {
                    invalid_residency_error(
                        "compiled partition source size overflowed",
                    )
                })
            })
    }

    pub fn resident_byte_count_for(
        &self,
        representation: CompiledResourceRepresentation,
    ) -> io::Result<usize> {
        match (representation, &self.resident_derivation) {
            (CompiledResourceRepresentation::ResidentDerivation, Some(derivation)) => {
                Ok(derivation.resident_byte_count)
            }
            _ => self.source_byte_count(),
        }
    }

    pub fn supports_representation(
        &self,
        representation: CompiledResourceRepresentation,
    ) -> bool {
        representation == CompiledResourceRepresentation::Source
            || self.resident_derivation.is_some()
    }
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

impl CompiledResourceResidencyContract {
    pub fn inspection_report(
        &self,
    ) -> io::Result<crate::RuntimeResourceResidencyInspectionReport> {
        let resource_bytes = self
            .resources
            .iter()
            .map(|resource| {
                resource
                    .source_byte_count()
                .map(|bytes| (resource.id.as_str(), bytes))
            })
            .collect::<io::Result<BTreeMap<_, _>>>()?;
        let group_bytes = self
            .atomic_groups
            .iter()
            .map(|group| {
                group
                    .resource_ids
                    .iter()
                    .try_fold(0usize, |total, resource_id| {
                        total
                            .checked_add(
                                resource_bytes
                                    .get(resource_id.as_str())
                                    .copied()
                                    .ok_or_else(|| {
                                        invalid_residency_error(
                                            "compiled residency group inspection references a missing resource",
                                        )
                                    })?,
                            )
                            .ok_or_else(|| {
                                invalid_residency_error(
                                    "compiled residency group inspection byte count overflowed",
                                )
                            })
                    })
                    .map(|bytes| (group.id.as_str(), bytes))
            })
            .collect::<io::Result<BTreeMap<_, _>>>()?;
        let template_bytes = self
            .partition_templates
            .iter()
            .map(|template| {
                template
                    .member_templates
                    .iter()
                    .try_fold(0usize, |total, member| {
                        let member_bytes = member.source_byte_count()?;
                        total.checked_add(member_bytes).ok_or_else(|| {
                            invalid_residency_error(
                                "compiled partition inspection byte count overflowed",
                            )
                        })
                    })
                    .map(|bytes| (template.id.as_str(), bytes))
            })
            .collect::<io::Result<BTreeMap<_, _>>>()?;

        let always_group_ids = self
            .atomic_groups
            .iter()
            .filter(|group| {
                group.lifetime
                    == CompiledResourceLifetime::AlwaysResident
            })
            .map(|group| group.id.as_str())
            .collect::<BTreeSet<_>>();
        let dynamic_group_ids = self
            .atomic_groups
            .iter()
            .filter(|group| {
                group.lifetime == CompiledResourceLifetime::Dynamic
            })
            .map(|group| group.id.as_str())
            .collect::<BTreeSet<_>>();
        let always_resident = runtime_resource_residency_class_report(
            "always_resident",
            "declared always_resident by the compiled contract; required before execution and never selected at a residency checkpoint",
            &always_group_ids,
            &group_bytes,
            self.resources.iter().filter(|resource| {
                resource.lifetime
                    == CompiledResourceLifetime::AlwaysResident
            }).count(),
            &[],
        )?;
        let dynamic_templates = self
            .partition_templates
            .iter()
            .collect::<Vec<_>>();
        let dynamically_addressable =
            runtime_resource_residency_class_report(
                "dynamic",
                "addressable through compiled selectors at physical residency checkpoints; demand-retained loads selected accesses while eager loads the same declared set at mount",
                &dynamic_group_ids,
                &group_bytes,
                self.resources
                    .iter()
                    .filter(|resource| {
                        resource.lifetime
                            == CompiledResourceLifetime::Dynamic
                    })
                    .count(),
                &dynamic_templates,
            )?;

        let mut scope_units =
            BTreeMap::<String, BTreeMap<String, usize>>::new();
        let mut scope_components =
            BTreeMap::<String, BTreeSet<String>>::new();
        let mut scope_selectors = BTreeMap::<String, usize>::new();
        for selector in &self.selectors {
            let units = scope_units
                .entry(selector.execution_scope.clone())
                .or_default();
            scope_components
                .entry(selector.execution_scope.clone())
                .or_default()
                .insert(selector.component_id.clone());
            *scope_selectors
                .entry(selector.execution_scope.clone())
                .or_default() += 1;
            match &selector.mapping {
                CompiledResourceSelectorMapping::GroupTable {
                    atomic_group_ids,
                } => {
                    for group_id in atomic_group_ids {
                        units.insert(
                            group_id.clone(),
                            group_bytes
                                .get(group_id.as_str())
                                .copied()
                                .ok_or_else(|| {
                                    invalid_residency_error(
                                        "compiled selector inspection references a missing group",
                                    )
                                })?,
                        );
                    }
                }
                CompiledResourceSelectorMapping::PartitionTemplate {
                    partition_template_id,
                } => {
                    let template = self
                        .partition_templates
                        .iter()
                        .find(|template| {
                            template.id == *partition_template_id
                        })
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "compiled selector inspection references a missing partition template",
                            )
                        })?;
                    let bytes = template_bytes
                        .get(template.id.as_str())
                        .copied()
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "compiled selector inspection is missing partition bytes",
                            )
                        })?;
                    for partition_index in 0..template.partition_count {
                        units.insert(
                            derived_partition_resource_id(
                                &template.group_identity_seed,
                                partition_index,
                            )?,
                            bytes,
                        );
                    }
                }
            }
        }
        let checkpoint_counts = self.checkpoints.iter().fold(
            BTreeMap::<String, usize>::new(),
            |mut counts, checkpoint| {
                *counts
                    .entry(checkpoint.execution_scope.clone())
                    .or_default() += 1;
                counts
            },
        );
        let scopes = scope_units
            .into_iter()
            .map(|(execution_scope, units)| {
                let maximum_payload_bytes = units
                    .values()
                    .try_fold(0usize, |total, bytes| {
                        total.checked_add(*bytes).ok_or_else(|| {
                            invalid_residency_error(
                                "compiled residency scope inspection byte count overflowed",
                            )
                        })
                    })?;
                Ok(crate::RuntimeResourceResidencyScopeInspectionReport {
                    component_count: scope_components
                        .get(&execution_scope)
                        .map(BTreeSet::len)
                        .unwrap_or_default(),
                    selector_count: scope_selectors
                        .get(&execution_scope)
                        .copied()
                        .unwrap_or_default(),
                    checkpoint_count: checkpoint_counts
                        .get(&execution_scope)
                        .copied()
                        .unwrap_or_default(),
                    addressable_unit_count: units.len(),
                    maximum_payload_bytes,
                    execution_scope,
                })
            })
            .collect::<io::Result<Vec<_>>>()?;

        Ok(crate::RuntimeResourceResidencyInspectionReport {
            schema:
                "nerve.runtime_resource_residency_inspection.v1"
                    .to_string(),
            supported_policies: self
                .supported_policies
                .iter()
                .map(|policy| policy.as_runtime_name().to_string())
                .collect(),
            always_resident,
            dynamically_addressable,
            scopes,
        })
    }
}

fn runtime_resource_residency_class_report(
    lifetime: &str,
    reason: &str,
    group_ids: &BTreeSet<&str>,
    group_bytes: &BTreeMap<&str, usize>,
    concrete_resource_count: usize,
    templates: &[&CompiledPartitionTemplate],
) -> io::Result<crate::RuntimeResourceResidencyClassInspectionReport> {
    let template_unit_count = templates
        .iter()
        .try_fold(0usize, |total, template| {
            total.checked_add(template.partition_count).ok_or_else(|| {
                invalid_residency_error(
                    "compiled residency inspection unit count overflowed",
                )
            })
        })?;
    let template_resource_count =
        templates.iter().try_fold(0usize, |total, template| {
            let member_count = template
                .partition_count
                .checked_mul(template.member_templates.len())
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled residency inspection resource count overflowed",
                    )
                })?;
            total.checked_add(member_count).ok_or_else(|| {
                invalid_residency_error(
                    "compiled residency inspection resource count overflowed",
                )
            })
        })?;
    let group_payload_bytes =
        group_ids.iter().try_fold(0usize, |total, group_id| {
            total
                .checked_add(
                    group_bytes.get(group_id).copied().ok_or_else(
                        || {
                            invalid_residency_error(
                                "compiled residency inspection is missing group bytes",
                            )
                        },
                    )?,
                )
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled residency inspection byte count overflowed",
                    )
                })
        })?;
    let template_payload_bytes =
        templates.iter().try_fold(0usize, |total, template| {
            let unit_bytes = template
                .member_templates
                .iter()
                .try_fold(0usize, |unit_total, member| {
                    let member_bytes = member.source_byte_count()?;
                    unit_total.checked_add(member_bytes).ok_or_else(|| {
                        invalid_residency_error(
                            "compiled residency inspection byte count overflowed",
                        )
                    })
                })?;
            total
                .checked_add(
                    unit_bytes
                        .checked_mul(template.partition_count)
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "compiled residency inspection byte count overflowed",
                            )
                        })?,
                )
                .ok_or_else(|| {
                    invalid_residency_error(
                        "compiled residency inspection byte count overflowed",
                    )
                })
        })?;
    Ok(crate::RuntimeResourceResidencyClassInspectionReport {
        lifetime: lifetime.to_string(),
        reason: reason.to_string(),
        unit_count: group_ids
            .len()
            .checked_add(template_unit_count)
            .ok_or_else(|| {
                invalid_residency_error(
                    "compiled residency inspection unit count overflowed",
                )
            })?,
        resource_count: concrete_resource_count
            .checked_add(template_resource_count)
            .ok_or_else(|| {
                invalid_residency_error(
                    "compiled residency inspection resource count overflowed",
                )
            })?,
        maximum_payload_bytes: group_payload_bytes
            .checked_add(template_payload_bytes)
            .ok_or_else(|| {
                invalid_residency_error(
                    "compiled residency inspection byte count overflowed",
                )
            })?,
    })
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
    let mut payload = serde_json::json!({
            "lifetime": resource.lifetime,
            "ranges": resource.ranges.iter().map(|range| serde_json::json!({
                "byte_count": range.byte_count,
                "alignment_bytes": range.alignment_bytes,
                "integrity": range.integrity,
            })).collect::<Vec<_>>(),
            "dependencies": resource.dependencies,
            "compatibility": resource.compatibility,
        });
    if let Some(derivation) = &resource.resident_derivation {
        payload
            .as_object_mut()
            .expect("resource identity payload is an object")
            .insert(
                "resident_derivation".to_string(),
                serde_json::to_value(derivation).map_err(io::Error::other)?,
            );
    }
    resource_content_id("resource", payload)
}

pub(super) fn compiled_atomic_group_identity(
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

pub(super) fn compiled_partition_template_identity(
    template: &CompiledPartitionTemplate,
) -> io::Result<String> {
    let member_payloads = template
        .member_templates
        .iter()
        .map(|member| {
            let mut payload = serde_json::json!({
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
            });
            if let Some(derivation) = &member.resident_derivation {
                payload
                    .as_object_mut()
                    .expect("partition member identity payload is an object")
                    .insert(
                        "resident_derivation".to_string(),
                        serde_json::to_value(derivation).map_err(io::Error::other)?,
                    );
            }
            Ok(payload)
        })
        .collect::<io::Result<Vec<_>>>()?;
    resource_content_id(
        "partition_template",
        serde_json::json!({
            "partition_count": template.partition_count,
            "lifetime": template.lifetime,
            "group_identity_seed": template.group_identity_seed,
            "member_templates": member_payloads,
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

pub(super) fn compiled_selector_identity(
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

pub(super) fn compiled_checkpoint_identity(
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

pub(super) fn validate_compiled_resource_residency(
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
    let contract_index = CompiledResourceContractIndex::new(contract)?;

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
        if let Some(derivation) = &resource.resident_derivation {
            let source_byte_count = resource.ranges.iter().try_fold(
                0usize,
                |total, range| {
                    total.checked_add(range.byte_count).ok_or_else(|| {
                        invalid_residency_error(
                            "compiled resource source size overflowed",
                        )
                    })
                },
            )?;
            derivation.validate_for_source_byte_count(source_byte_count)?;
            if derivation.required_features.iter().any(|feature| {
                !resource.compatibility.required_features.contains(feature)
            }) {
                return invalid_residency(
                    "compiled resource compatibility omits resident derivation features",
                );
            }
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
            let resource = contract_index
                .resource(contract, resource_id)
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

    validate_bindings_against_package(
        manifest,
        &contract_index,
        &group_ids,
        &template_ids,
    )?;
    validate_selectors_and_checkpoints(manifest, &contract_index, &template_ids)
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
        if let Some(derivation) = &member.resident_derivation {
            let source_byte_count = member.range_templates.iter().try_fold(
                0usize,
                |total, range| {
                    total.checked_add(range.byte_count).ok_or_else(|| {
                        invalid_residency_error(
                            "partition member source size overflowed",
                        )
                    })
                },
            )?;
            derivation.validate_for_source_byte_count(source_byte_count)?;
            if derivation.required_features.iter().any(|feature| {
                !member.compatibility.required_features.contains(feature)
            }) {
                return invalid_residency(
                    "partition member compatibility omits resident derivation features",
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
    contract_index: &CompiledResourceContractIndex,
    group_ids: &BTreeSet<&str>,
    template_ids: &BTreeSet<&str>,
) -> io::Result<()> {
    let semantics = compiled_parameter_semantics(manifest)?;
    let mut keys = Vec::new();
    let mut bound_semantics = BTreeSet::new();
    let mut bound_concrete_resources = BTreeSet::new();
    let mut bound_partition_members = BTreeSet::new();
    let mut selected_slots: BTreeMap<
        (String, String, String, String),
        Vec<(String, usize, usize)>,
    > = BTreeMap::new();
    let mut partition_slots: BTreeMap<
        (String, String, String, String, String),
        Vec<usize>,
    > = BTreeMap::new();
    for binding in &manifest.resource_residency.bindings {
        let mapping_key = match &binding.mapping {
            CompiledResourceBindingMapping::AtomicGroup {
                atomic_group_id,
                resource_id,
            } => {
                let group = contract_index
                    .atomic_group(&manifest.resource_residency, atomic_group_id);
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
            CompiledResourceBindingMapping::SelectedAtomicGroup {
                atomic_group_id,
                resource_id,
                selection_signal,
                selector_index,
                parameter_slot,
            } => {
                let group = contract_index
                    .atomic_group(&manifest.resource_residency, atomic_group_id);
                if !group_ids.contains(atomic_group_id.as_str())
                    || group.is_none_or(|group| {
                        group.lifetime != CompiledResourceLifetime::Dynamic
                            || !group.resource_ids.contains(resource_id)
                    })
                    || selection_signal.trim().is_empty()
                {
                    return invalid_residency(
                        "compiled selected resource binding maps outside its dynamic atomic group",
                    );
                }
                bound_concrete_resources.insert(resource_id.as_str());
                selected_slots
                    .entry((
                        binding.execution_scope.clone(),
                        binding.component_id.clone(),
                        binding.node_id.clone(),
                        selection_signal.clone(),
                    ))
                    .or_default()
                    .push((
                        atomic_group_id.clone(),
                        *selector_index,
                        *parameter_slot,
                    ));
                format!(
                    "selected_atomic_group|{atomic_group_id}|{resource_id}|{selection_signal}|{selector_index}|{parameter_slot}"
                )
            }
            CompiledResourceBindingMapping::PartitionTemplateMember {
                partition_template_id,
                resource_identity_seed,
                selection_signal,
                parameter_slot,
            } => {
                validate_content_id(
                    "resource binding partition template id",
                    partition_template_id,
                )?;
                validate_content_id(
                    "resource binding partition resource seed",
                    resource_identity_seed,
                )?;
                if selection_signal.trim().is_empty() {
                    return invalid_residency(
                        "compiled partition resource binding has no selection signal",
                    );
                }
                let template = contract_index
                    .partition_template(&manifest.resource_residency, partition_template_id);
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
                partition_slots
                    .entry((
                        binding.execution_scope.clone(),
                        binding.component_id.clone(),
                        binding.node_id.clone(),
                        partition_template_id.clone(),
                        selection_signal.clone(),
                    ))
                    .or_default()
                    .push(*parameter_slot);
                format!(
                    "partition_template_member||{partition_template_id}|{resource_identity_seed}|{selection_signal}|{parameter_slot}"
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
    for ((scope, component_id, node_id, selection_signal), slots) in
        selected_slots
    {
        let matching_selectors = manifest
            .resource_residency
            .selectors
            .iter()
            .filter(|selector| {
                selector.execution_scope == scope
                    && selector.component_id == component_id
                    && selector.selection_signal == selection_signal
                    && matches!(
                        &selector.mapping,
                        CompiledResourceSelectorMapping::GroupTable {
                            atomic_group_ids
                        } if slots.iter().all(|(group_id, selector_index, _)| {
                            atomic_group_ids.get(*selector_index) == Some(group_id)
                        })
                    )
            })
            .collect::<Vec<_>>();
        if matching_selectors.len() != 1 {
            return invalid_residency(format!(
                "compiled selected resource bindings for {scope} {component_id}.{node_id} do not map exactly one group-table selector"
            ));
        }
        let resource_count = matching_selectors[0].resource_count;
        let mut slots_by_selector: BTreeMap<usize, BTreeSet<usize>> =
            BTreeMap::new();
        for (_, selector_index, parameter_slot) in slots {
            if !slots_by_selector
                .entry(selector_index)
                .or_default()
                .insert(parameter_slot)
            {
                return invalid_residency(format!(
                    "compiled selected resource bindings for {scope} {component_id}.{node_id} repeat a selector parameter slot"
                ));
            }
        }
        if slots_by_selector.len() != resource_count
            || slots_by_selector
                .keys()
                .copied()
                .ne(0..resource_count)
        {
            return invalid_residency(format!(
                "compiled selected resource bindings for {scope} {component_id}.{node_id} do not cover every selector index"
            ));
        }
        let parameter_count = slots_by_selector
            .first_key_value()
            .map(|(_, slots)| slots.len())
            .unwrap_or(0);
        if parameter_count == 0
            || slots_by_selector.values().any(|slots| {
                slots.len() != parameter_count
                    || slots.iter().copied().ne(0..parameter_count)
            })
        {
            return invalid_residency(format!(
                "compiled selected resource bindings for {scope} {component_id}.{node_id} do not define one contiguous parameter-slot layout"
            ));
        }
    }
    for ((scope, component_id, node_id, template_id, selection_signal), slots) in
        partition_slots
    {
        let matching_selector_count = manifest
            .resource_residency
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
            .count();
        let unique_slots = slots.iter().copied().collect::<BTreeSet<_>>();
        if matching_selector_count != 1
            || unique_slots.len() != slots.len()
            || unique_slots.iter().copied().ne(0..slots.len())
        {
            return invalid_residency(format!(
                "compiled partition resource bindings for {scope} {component_id}.{node_id} do not define one selector and contiguous parameter-slot layout"
            ));
        }
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
    contract_index: &CompiledResourceContractIndex,
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
                    let group = contract_index
                        .atomic_group(contract, group_id)
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
                let template = contract_index
                    .partition_template(contract, partition_template_id)
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
            let selector = contract_index
                .selector(contract, selector_id)
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
        let contract_index =
            CompiledResourceContractIndex::new(&manifest.resource_residency).unwrap();
        validate_selectors_and_checkpoints(&manifest, &contract_index, &template_ids).unwrap();
    }
}
