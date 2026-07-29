use std::io::{Read as _, Seek as _, SeekFrom};

pub const RESOLVED_PARTITION_GROUP_SCHEMA: &str =
    "nerve.resolved_partition_group.v1";
pub const RESOLVED_ATOMIC_GROUP_SCHEMA: &str =
    "nerve.resolved_atomic_group.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedCompiledResourceRange {
    pub artifact_path: String,
    pub byte_offset: usize,
    pub byte_count: usize,
    pub alignment_bytes: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedCompiledResource {
    pub id: String,
    pub ranges: Vec<ResolvedCompiledResourceRange>,
    pub compatibility: CompiledResourceCompatibility,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedCompiledPartitionGroup {
    pub schema: String,
    pub partition_template_id: String,
    pub partition_index: usize,
    pub id: String,
    pub resource_ids: Vec<String>,
    pub dependencies: Vec<String>,
    pub resources: Vec<ResolvedCompiledResource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ResolvedCompiledAtomicGroup {
    pub schema: String,
    pub id: String,
    pub resource_ids: Vec<String>,
    pub dependencies: Vec<String>,
    pub resources: Vec<ResolvedCompiledResource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedCompiledResourceGroup {
    Atomic(ResolvedCompiledAtomicGroup),
    Partition(ResolvedCompiledPartitionGroup),
}

impl From<ResolvedCompiledAtomicGroup> for ResolvedCompiledResourceGroup {
    fn from(group: ResolvedCompiledAtomicGroup) -> Self {
        Self::Atomic(group)
    }
}

impl From<ResolvedCompiledPartitionGroup> for ResolvedCompiledResourceGroup {
    fn from(group: ResolvedCompiledPartitionGroup) -> Self {
        Self::Partition(group)
    }
}

impl ResolvedCompiledResourceGroup {
    pub fn id(&self) -> &str {
        match self {
            Self::Atomic(group) => &group.id,
            Self::Partition(group) => &group.id,
        }
    }

    pub fn resource_ids(&self) -> &[String] {
        match self {
            Self::Atomic(group) => &group.resource_ids,
            Self::Partition(group) => &group.resource_ids,
        }
    }

    pub fn dependencies(&self) -> &[String] {
        match self {
            Self::Atomic(group) => &group.dependencies,
            Self::Partition(group) => &group.dependencies,
        }
    }

    pub fn resources(&self) -> &[ResolvedCompiledResource] {
        match self {
            Self::Atomic(group) => &group.resources,
            Self::Partition(group) => &group.resources,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PhysicalInterval {
    start: usize,
    end: usize,
    identity: String,
}

pub fn resolve_compiled_partition_group(
    package_root: &Path,
    contract: &CompiledResourceResidencyContract,
    partition_template_id: &str,
    partition_index: usize,
) -> io::Result<ResolvedCompiledPartitionGroup> {
    validate_content_id("partition template id", partition_template_id)?;
    let template = contract
        .partition_templates
        .iter()
        .find(|template| template.id == partition_template_id)
        .ok_or_else(|| invalid_residency_error("unknown partition template"))?;
    if partition_index >= template.partition_count {
        return invalid_residency(format!(
            "partition index {partition_index} exceeds template count {}",
            template.partition_count
        ));
    }

    let mut resources = Vec::with_capacity(template.member_templates.len());
    for member in &template.member_templates {
        let mut ranges = Vec::with_capacity(member.range_templates.len());
        for range in &member.range_templates {
            validate_resident_package_relative_path(
                "partition range artifact",
                &range.artifact_path,
            )?;
            validate_resident_package_relative_path(
                "partition digest table",
                &range.integrity.digest_table_path,
            )?;
            if range.byte_count == 0
                || range.stride_bytes < range.byte_count
                || !range.alignment_bytes.is_power_of_two()
                || range.base_byte_offset % range.alignment_bytes != 0
                || range.stride_bytes % range.alignment_bytes != 0
                || range.integrity.algorithm != "sha256_table"
                || range.integrity.digest_stride_bytes != 32
                || range.integrity.digest_table_byte_offset % 32 != 0
                || !is_lower_hex_sha256(&range.integrity.table_sha256)
            {
                return invalid_residency(
                    "partition range template is invalid",
                );
            }
            let byte_offset = partition_index
                .checked_mul(range.stride_bytes)
                .and_then(|offset| range.base_byte_offset.checked_add(offset))
                .ok_or_else(|| {
                    invalid_residency_error(
                        "resolved partition byte offset overflowed",
                    )
                })?;
            if byte_offset % range.alignment_bytes != 0 {
                return invalid_residency(
                    "resolved partition byte offset violates alignment",
                );
            }
            let digest_offset = partition_index
                .checked_mul(range.integrity.digest_stride_bytes)
                .and_then(|offset| {
                    range
                        .integrity
                        .digest_table_byte_offset
                        .checked_add(offset)
                })
                .ok_or_else(|| {
                    invalid_residency_error(
                        "resolved partition digest offset overflowed",
                    )
                })?;
            let mut digest = [0u8; 32];
            let digest_path =
                package_root.join(&range.integrity.digest_table_path);
            let mut digest_file = fs::File::open(&digest_path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to open partition digest table {}: {error}",
                        digest_path.display()
                    ),
                )
            })?;
            digest_file
                .seek(SeekFrom::Start(u64::try_from(digest_offset).map_err(
                    |_| {
                        invalid_residency_error(
                            "partition digest offset does not fit u64",
                        )
                    },
                )?))
                .and_then(|_| digest_file.read_exact(&mut digest))
                .map_err(|error| {
                    io::Error::new(
                        error.kind(),
                        format!(
                            "failed to read partition digest at {}:{digest_offset}: {error}",
                            digest_path.display()
                        ),
                    )
                })?;
            ranges.push(ResolvedCompiledResourceRange {
                artifact_path: range.artifact_path.clone(),
                byte_offset,
                byte_count: range.byte_count,
                alignment_bytes: range.alignment_bytes,
                sha256: lower_hex(&digest),
            });
        }
        resources.push(ResolvedCompiledResource {
            id: derived_partition_resource_id(
                &member.resource_identity_seed,
                partition_index,
            )?,
            ranges,
            compatibility: member.compatibility.clone(),
        });
    }
    resources.sort_by(|left, right| left.id.cmp(&right.id));
    let resource_ids = resources
        .iter()
        .map(|resource| resource.id.clone())
        .collect();
    Ok(ResolvedCompiledPartitionGroup {
        schema: RESOLVED_PARTITION_GROUP_SCHEMA.to_string(),
        partition_template_id: template.id.clone(),
        partition_index,
        id: derived_partition_resource_id(
            &template.group_identity_seed,
            partition_index,
        )?,
        resource_ids,
        dependencies: template.dependencies.clone(),
        resources,
    })
}

pub fn resolve_compiled_atomic_group(
    contract: &CompiledResourceResidencyContract,
    atomic_group_id: &str,
) -> io::Result<ResolvedCompiledAtomicGroup> {
    validate_content_id("atomic group id", atomic_group_id)?;
    let group = contract
        .atomic_groups
        .iter()
        .find(|group| group.id == atomic_group_id)
        .ok_or_else(|| invalid_residency_error("unknown atomic group"))?;
    let resources = group
        .resource_ids
        .iter()
        .map(|resource_id| {
            let resource = contract
                .resources
                .iter()
                .find(|resource| resource.id == *resource_id)
                .ok_or_else(|| {
                    invalid_residency_error(format!(
                        "atomic group references unknown resource {resource_id:?}"
                    ))
                })?;
            let ranges = resource
                .ranges
                .iter()
                .map(|range| ResolvedCompiledResourceRange {
                    artifact_path: range.artifact_path.clone(),
                    byte_offset: range.byte_offset,
                    byte_count: range.byte_count,
                    alignment_bytes: range.alignment_bytes,
                    sha256: range.integrity.digest.clone(),
                })
                .collect();
            Ok(ResolvedCompiledResource {
                id: resource.id.clone(),
                ranges,
                compatibility: resource.compatibility.clone(),
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

pub fn read_verified_compiled_resource_range(
    package_root: &Path,
    range: &ResolvedCompiledResourceRange,
) -> io::Result<Vec<u8>> {
    validate_resident_package_relative_path(
        "resolved resource artifact",
        &range.artifact_path,
    )?;
    if range.byte_count == 0
        || !range.alignment_bytes.is_power_of_two()
        || range.byte_offset % range.alignment_bytes != 0
        || !is_lower_hex_sha256(&range.sha256)
    {
        return invalid_residency("resolved resource range is invalid");
    }
    let path = package_root.join(&range.artifact_path);
    let mut source = fs::File::open(&path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to open compiled resource artifact {}: {error}",
                path.display()
            ),
        )
    })?;
    source.seek(SeekFrom::Start(
        u64::try_from(range.byte_offset).map_err(|_| {
            invalid_residency_error(
                "resolved resource offset does not fit u64",
            )
        })?,
    ))?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(range.byte_count).map_err(|error| {
        invalid_residency_error(format!(
            "failed to reserve {} resource bytes: {error}",
            range.byte_count
        ))
    })?;
    payload.resize(range.byte_count, 0);
    source.read_exact(&mut payload).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to read compiled resource range {}:{}+{}: {error}",
                range.artifact_path, range.byte_offset, range.byte_count
            ),
        )
    })?;
    if lower_hex(&Sha256::digest(&payload)) != range.sha256 {
        return invalid_residency(format!(
            "compiled resource range {}:{}+{} failed SHA-256",
            range.artifact_path, range.byte_offset, range.byte_count
        ));
    }
    Ok(payload)
}

pub fn read_verified_compiled_partition_group(
    package_root: &Path,
    group: &ResolvedCompiledPartitionGroup,
) -> io::Result<BTreeMap<String, Vec<Vec<u8>>>> {
    let actual_resource_ids = group
        .resources
        .iter()
        .map(|resource| resource.id.clone())
        .collect::<Vec<_>>();
    let unique_resource_ids = actual_resource_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if group.schema != RESOLVED_PARTITION_GROUP_SCHEMA
        || group.resource_ids != actual_resource_ids
        || unique_resource_ids.len() != group.resource_ids.len()
        || validate_content_id("resolved partition group id", &group.id)
            .is_err()
        || group
            .resource_ids
            .iter()
            .any(|id| validate_content_id("resolved resource id", id).is_err())
    {
        return invalid_residency("resolved partition group is inconsistent");
    }
    let mut loaded = BTreeMap::new();
    for resource in &group.resources {
        let mut ranges = Vec::with_capacity(resource.ranges.len());
        for range in &resource.ranges {
            ranges.push(read_verified_compiled_resource_range(
                package_root,
                range,
            )?);
        }
        loaded.insert(resource.id.clone(), ranges);
    }
    Ok(loaded)
}

fn validate_compiled_partition_storage(
    package_root: &Path,
    contract: &CompiledResourceResidencyContract,
) -> io::Result<()> {
    let mut artifact_intervals: BTreeMap<String, Vec<PhysicalInterval>> =
        BTreeMap::new();
    for resource in &contract.resources {
        for range in &resource.ranges {
            let end = range
                .byte_offset
                .checked_add(range.byte_count)
                .ok_or_else(|| {
                    invalid_residency_error(
                        "concrete resource interval overflowed",
                    )
                })?;
            artifact_intervals
                .entry(range.artifact_path.clone())
                .or_default()
                .push(PhysicalInterval {
                    start: range.byte_offset,
                    end,
                    identity: format!("concrete|{}", resource.id),
                });
        }
    }

    let mut table_contracts = BTreeMap::<String, String>::new();
    let mut digest_intervals: BTreeMap<String, Vec<PhysicalInterval>> =
        BTreeMap::new();
    for template in &contract.partition_templates {
        for member in &template.member_templates {
            for range in &member.range_templates {
                if range.stride_bytes < range.byte_count {
                    return invalid_residency(
                        "partition range stride overlaps adjacent resources",
                    );
                }
                match table_contracts
                    .entry(range.integrity.digest_table_path.clone())
                {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(range.integrity.table_sha256.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if entry.get()
                            != &range.integrity.table_sha256 =>
                    {
                        return invalid_residency(
                            "partition digest table has conflicting integrity contracts",
                        );
                    }
                    std::collections::btree_map::Entry::Occupied(_) => {}
                }
                for partition_index in 0..template.partition_count {
                    let byte_offset = partition_index
                        .checked_mul(range.stride_bytes)
                        .and_then(|offset| {
                            range.base_byte_offset.checked_add(offset)
                        })
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "partition data interval overflowed",
                            )
                        })?;
                    let byte_end = byte_offset
                        .checked_add(range.byte_count)
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "partition data interval overflowed",
                            )
                        })?;
                    let digest_offset = partition_index
                        .checked_mul(range.integrity.digest_stride_bytes)
                        .and_then(|offset| {
                            range
                                .integrity
                                .digest_table_byte_offset
                                .checked_add(offset)
                        })
                        .ok_or_else(|| {
                            invalid_residency_error(
                                "partition digest interval overflowed",
                            )
                        })?;
                    let digest_end =
                        digest_offset.checked_add(32).ok_or_else(|| {
                            invalid_residency_error(
                                "partition digest interval overflowed",
                            )
                        })?;
                    let identity = format!(
                        "dynamic|{}|{}|{}|{}|{}|{}",
                        member.resource_identity_seed,
                        range.artifact_path,
                        byte_offset,
                        range.byte_count,
                        range.integrity.digest_table_path,
                        digest_offset,
                    );
                    artifact_intervals
                        .entry(range.artifact_path.clone())
                        .or_default()
                        .push(PhysicalInterval {
                            start: byte_offset,
                            end: byte_end,
                            identity: identity.clone(),
                        });
                    digest_intervals
                        .entry(range.integrity.digest_table_path.clone())
                        .or_default()
                        .push(PhysicalInterval {
                            start: digest_offset,
                            end: digest_end,
                            identity,
                        });
                }
            }
        }
    }

    for intervals in artifact_intervals.values_mut() {
        sort_deduplicate_and_reject_overlaps(
            intervals,
            "compiled concrete and partition resource ranges overlap",
        )?;
    }
    for (table_path, expected_sha256) in table_contracts {
        let path = package_root.join(&table_path);
        let (actual_sha256, table_bytes) =
            file_sha256_and_size(&path, "partition digest table")?;
        if actual_sha256 != expected_sha256 {
            return invalid_residency(format!(
                "partition digest table {table_path:?} failed SHA-256"
            ));
        }
        let intervals = digest_intervals
            .get_mut(&table_path)
            .expect("validated table has digest intervals");
        sort_deduplicate_and_reject_overlaps(
            intervals,
            "partition digest table entries overlap",
        )?;
        let mut cursor = 0usize;
        for interval in intervals {
            if interval.start != cursor {
                return invalid_residency(format!(
                    "partition digest table {table_path:?} has a coverage gap at {cursor}"
                ));
            }
            cursor = interval.end;
        }
        if cursor != table_bytes {
            return invalid_residency(format!(
                "partition digest table {table_path:?} covers {cursor} of {} bytes",
                table_bytes
            ));
        }
    }
    Ok(())
}

fn file_sha256_and_size(
    path: &Path,
    label: &str,
) -> io::Result<(String, usize)> {
    let mut source = fs::File::open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to open {label} {}: {error}", path.display()),
        )
    })?;
    let mut digest = Sha256::new();
    let mut size = 0usize;
    let mut chunk = [0u8; 1024 * 1024];
    loop {
        let count = source.read(&mut chunk).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("failed to read {label} {}: {error}", path.display()),
            )
        })?;
        if count == 0 {
            break;
        }
        size = size.checked_add(count).ok_or_else(|| {
            invalid_residency_error(format!(
                "{label} {} size overflowed",
                path.display()
            ))
        })?;
        digest.update(&chunk[..count]);
    }
    Ok((lower_hex(&digest.finalize()), size))
}

fn sort_deduplicate_and_reject_overlaps(
    intervals: &mut Vec<PhysicalInterval>,
    message: &str,
) -> io::Result<()> {
    intervals.sort();
    intervals.dedup();
    if intervals
        .windows(2)
        .any(|pair| pair[1].start < pair[0].end)
    {
        return invalid_residency(message);
    }
    Ok(())
}

fn lower_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod resource_range_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn resolves_and_verifies_only_the_requested_partition() {
        let fixture = RangeFixture::new();
        fixture.validate();
        let resolved = resolve_compiled_partition_group(
            &fixture.root,
            &fixture.contract,
            &fixture.contract.partition_templates[0].id,
            1,
        )
        .unwrap();

        assert_eq!(resolved.resources.len(), 1);
        assert_eq!(resolved.resources[0].ranges[0].byte_offset, 4);
        assert_eq!(
            read_verified_compiled_partition_group(&fixture.root, &resolved)
                .unwrap()
                .into_values()
                .next()
                .unwrap(),
            vec![b"efgh".to_vec()],
        );

        let artifact = fixture.root.join("weights/bank.bin");
        let mut payload = fs::read(&artifact).unwrap();
        payload[0] ^= 0xff;
        fs::write(&artifact, payload).unwrap();

        // Partition one remains independently readable and verifiable.
        assert_eq!(
            read_verified_compiled_partition_group(&fixture.root, &resolved)
                .unwrap()
                .into_values()
                .next()
                .unwrap(),
            vec![b"efgh".to_vec()],
        );
        let corrupt = resolve_compiled_partition_group(
            &fixture.root,
            &fixture.contract,
            &fixture.contract.partition_templates[0].id,
            0,
        )
        .unwrap();
        assert!(
            read_verified_compiled_partition_group(&fixture.root, &corrupt)
                .unwrap_err()
                .to_string()
                .contains("failed SHA-256")
        );
    }

    #[test]
    fn compact_partition_contract_is_relocatable() {
        let first = RangeFixture::new();
        let second = RangeFixture::new();
        first.validate();
        second.validate();
        let first_group = resolve_compiled_partition_group(
            &first.root,
            &first.contract,
            &first.contract.partition_templates[0].id,
            0,
        )
        .unwrap();
        let second_group = resolve_compiled_partition_group(
            &second.root,
            &second.contract,
            &second.contract.partition_templates[0].id,
            0,
        )
        .unwrap();

        assert_eq!(first_group, second_group);
    }

    #[test]
    fn rejects_truncated_or_corrupt_partition_storage() {
        let truncated = RangeFixture::new();
        let artifact = truncated.root.join("weights/bank.bin");
        fs::OpenOptions::new()
            .write(true)
            .open(&artifact)
            .unwrap()
            .set_len(7)
            .unwrap();
        assert!(
            truncated
                .validate_result()
                .unwrap_err()
                .to_string()
                .contains("exceeds its artifact")
        );

        let corrupt = RangeFixture::new();
        let table = corrupt.root.join("integrity/partitions.sha256");
        let mut payload = fs::read(&table).unwrap();
        payload[0] ^= 0xff;
        fs::write(&table, payload).unwrap();
        assert!(
            corrupt
                .validate_result()
                .unwrap_err()
                .to_string()
                .contains("failed SHA-256")
        );
    }

    #[test]
    fn rejects_overlapping_ranges_and_uncovered_digest_bytes() {
        let mut overlapping = RangeFixture::new();
        let range = &mut overlapping.contract.partition_templates[0]
            .member_templates[0]
            .range_templates[0];
        range.stride_bytes = 2;
        range.alignment_bytes = 2;
        assert!(
            overlapping
                .validate_result()
                .unwrap_err()
                .to_string()
                .contains("overlaps adjacent")
        );

        let suffix = RangeFixture::new();
        let table = suffix.root.join("integrity/partitions.sha256");
        let mut payload = fs::read(&table).unwrap();
        payload.extend_from_slice(&[0u8; 32]);
        fs::write(&table, &payload).unwrap();
        let mut contract = suffix.contract.clone();
        contract.partition_templates[0].member_templates[0].range_templates[0]
            .integrity
            .table_sha256 = lower_hex(&Sha256::digest(&payload));
        assert!(
            validate_compiled_partition_storage(&suffix.root, &contract)
                .unwrap_err()
                .to_string()
                .contains("covers 64 of 96")
        );
    }

    #[test]
    fn rejects_unsafe_templates_and_inconsistent_resolved_groups() {
        let mut unsafe_fixture = RangeFixture::new();
        unsafe_fixture.contract.partition_templates[0].member_templates[0]
            .range_templates[0]
            .artifact_path = "../outside.bin".to_string();
        assert!(
            resolve_compiled_partition_group(
                &unsafe_fixture.root,
                &unsafe_fixture.contract,
                &unsafe_fixture.contract.partition_templates[0].id,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("must stay inside")
        );

        let fixture = RangeFixture::new();
        let mut resolved = resolve_compiled_partition_group(
            &fixture.root,
            &fixture.contract,
            &fixture.contract.partition_templates[0].id,
            0,
        )
        .unwrap();
        resolved.resources.push(resolved.resources[0].clone());
        resolved.resource_ids.push(resolved.resource_ids[0].clone());
        assert!(
            read_verified_compiled_partition_group(&fixture.root, &resolved)
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
    }

    struct RangeFixture {
        root: PathBuf,
        contract: CompiledResourceResidencyContract,
    }

    impl RangeFixture {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root = std::env::temp_dir().join(format!(
                "nerve-resource-range-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("weights")).unwrap();
            fs::create_dir_all(root.join("integrity")).unwrap();
            fs::write(root.join("weights/bank.bin"), b"abcdefgh").unwrap();
            let table = [
                Sha256::digest(b"abcd").as_slice(),
                Sha256::digest(b"efgh").as_slice(),
            ]
            .concat();
            fs::write(root.join("integrity/partitions.sha256"), &table).unwrap();

            let compatibility = CompiledResourceCompatibility {
                device_api: "vulkan".to_string(),
                storage_class: "storage_buffer".to_string(),
                read_only: true,
                required_features: Vec::new(),
            };
            let member = CompiledPartitionMemberTemplate {
                resource_identity_seed: resource_content_id(
                    "partition_resource_seed",
                    serde_json::json!({"fixture": "bank"}),
                )
                .unwrap(),
                range_templates: vec![CompiledResourceRangeTemplate {
                    artifact_path: "weights/bank.bin".to_string(),
                    base_byte_offset: 0,
                    stride_bytes: 4,
                    byte_count: 4,
                    alignment_bytes: 4,
                    integrity: CompiledResourceRangeIntegrityTemplate {
                        algorithm: "sha256_table".to_string(),
                        digest_table_path:
                            "integrity/partitions.sha256".to_string(),
                        digest_table_byte_offset: 0,
                        digest_stride_bytes: 32,
                        table_sha256: lower_hex(&Sha256::digest(&table)),
                    },
                }],
                compatibility,
            };
            let mut template = CompiledPartitionTemplate {
                id: String::new(),
                partition_count: 2,
                lifetime: CompiledResourceLifetime::Dynamic,
                group_identity_seed: String::new(),
                member_templates: vec![member],
                dependencies: Vec::new(),
            };
            template.group_identity_seed =
                compiled_partition_group_identity_seed(
                    template.partition_count,
                    &template.member_templates,
                )
                .unwrap();
            template.id =
                compiled_partition_template_identity(&template).unwrap();
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
                bindings: Vec::new(),
                selectors: Vec::new(),
                checkpoints: Vec::new(),
            };
            Self { root, contract }
        }

        fn validate(&self) {
            self.validate_result().unwrap();
        }

        fn validate_result(&self) -> io::Result<()> {
            validate_partition_template(
                &self.root,
                &self.contract.partition_templates[0],
                &BTreeSet::new(),
            )?;
            validate_compiled_partition_storage(&self.root, &self.contract)
        }
    }

    impl Drop for RangeFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}
