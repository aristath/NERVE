use super::artifacts::{
    array, confined_path, invalid, invalid_error, read_json, read_object, require_schema, text,
};
use super::catalog::validate_mount_plan;
use super::{
    RUNTIME_MOUNT_PLAN_SCHEMA, RuntimeMountPlan, RuntimeStagedCandidate,
    STAGED_CANDIDATE_INTEGRITY_FILE, STAGED_CANDIDATE_INTEGRITY_SCHEMA,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::Path;

const CANDIDATE_BUILD_PLAN_SCHEMA: &str = "nerve.optimizer.candidate_build_plan.v1";
const REPRESENTATION_CANDIDATE_SCHEMA: &str = "nerve.optimizer.representation_candidate.v1";
const SOURCE_PACKAGE_SEAL_SCHEMA: &str = "nerve.optimizer.source_package_seal.v2";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StagedIntegrity {
    schema: String,
    candidate_id: String,
    construction_id: String,
    files: Vec<StagedIntegrityFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StagedIntegrityFile {
    path: String,
    byte_count: u64,
    sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcePackageSeal {
    schema: String,
    package_id: String,
    manifest_digest: String,
    optimizer_stage_digest: String,
    exact_baseline_digest: String,
    scope_catalog_digest: String,
    package_integrity_contract_digest: String,
    source_inputs: BTreeMap<String, SourcePackageInputSeal>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcePackageInputSeal {
    digest: String,
    signature: SourceFileSignature,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SourceFileSignature {
    device: u64,
    inode: u64,
    byte_count: u64,
    modified_ns: u64,
    changed_ns: u64,
}

impl RuntimeStagedCandidate {
    pub fn load(
        package_root: impl AsRef<Path>,
        candidate_root: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let package_root = package_root.as_ref().canonicalize()?;
        let candidate_input = candidate_root.as_ref();
        if fs::symlink_metadata(candidate_input)?
            .file_type()
            .is_symlink()
        {
            return invalid("staged candidate root must not be a symbolic link");
        }
        let candidate_root = candidate_input.canonicalize()?;
        if !candidate_root.is_dir() {
            return invalid("staged candidate root must be a directory");
        }
        if candidate_root.starts_with(&package_root) || package_root.starts_with(&candidate_root) {
            return invalid("staged candidate and source package must be isolated trees");
        }

        let integrity_path = confined_path(
            &candidate_root,
            STAGED_CANDIDATE_INTEGRITY_FILE,
            "staged candidate integrity",
        )?;
        let integrity: StagedIntegrity = read_json(&integrity_path, "staged candidate integrity")?;
        validate_candidate_id(&integrity.candidate_id)?;
        if integrity.schema != STAGED_CANDIDATE_INTEGRITY_SCHEMA
            || integrity.construction_id.is_empty()
        {
            return invalid("staged candidate integrity identity is invalid");
        }
        validate_integrity(&candidate_root, &integrity)?;

        let candidate = read_object(
            &confined_path(
                &candidate_root,
                "contracts/candidate.json",
                "staged candidate contract",
            )?,
            "staged candidate contract",
        )?;
        require_schema(
            &candidate,
            REPRESENTATION_CANDIDATE_SCHEMA,
            "staged candidate contract",
        )?;
        if text(&candidate, "candidate_id", "staged candidate contract")? != integrity.candidate_id
        {
            return invalid("staged candidate contract and integrity identity disagree");
        }

        let build_plan = read_object(
            &confined_path(
                &candidate_root,
                "contracts/build_plan.json",
                "staged candidate build plan",
            )?,
            "staged candidate build plan",
        )?;
        require_schema(
            &build_plan,
            CANDIDATE_BUILD_PLAN_SCHEMA,
            "staged candidate build plan",
        )?;
        let source_seal: SourcePackageSeal = read_json(
            &confined_path(
                &candidate_root,
                "contracts/source_seal.json",
                "staged source package seal",
            )?,
            "staged source package seal",
        )?;
        verify_source_inputs(&package_root, &build_plan, &source_seal)?;

        let mount_plan: RuntimeMountPlan = read_json(
            &confined_path(
                &candidate_root,
                "contracts/mount_plan.json",
                "staged runtime mount plan",
            )?,
            "staged runtime mount plan",
        )?;
        if mount_plan.schema != RUNTIME_MOUNT_PLAN_SCHEMA {
            return invalid("staged runtime mount plan schema is unsupported");
        }
        let source_component_ids = mount_plan
            .regions
            .iter()
            .flat_map(|region| region.component_replacements.iter())
            .map(|replacement| replacement.source_component_id.clone())
            .collect::<BTreeSet<_>>();
        validate_mount_plan(
            &candidate_root,
            &mount_plan,
            &integrity.candidate_id,
            &source_component_ids,
        )?;

        Ok(Self {
            candidate_id: integrity.candidate_id,
            candidate_root,
            source_component_ids: source_component_ids.into_iter().collect(),
            mount_plan,
        })
    }
}

fn validate_integrity(candidate_root: &Path, integrity: &StagedIntegrity) -> io::Result<()> {
    let expected_paths = collect_regular_files(candidate_root)?
        .into_iter()
        .filter(|path| path != STAGED_CANDIDATE_INTEGRITY_FILE)
        .collect::<BTreeSet<_>>();
    let mut previous: Option<&str> = None;
    let mut declared_paths = BTreeSet::new();
    for record in &integrity.files {
        if previous.is_some_and(|value| value >= record.path.as_str()) {
            return invalid("staged candidate integrity paths must be sorted and unique");
        }
        previous = Some(&record.path);
        let path = confined_path(candidate_root, &record.path, "staged candidate artifact")?;
        if !path.is_file()
            || fs::metadata(&path)?.len() != record.byte_count
            || sha256_file(&path)? != record.sha256
        {
            return invalid(format!(
                "staged candidate artifact failed integrity validation: {:?}",
                record.path
            ));
        }
        declared_paths.insert(record.path.clone());
    }
    if declared_paths != expected_paths {
        return invalid("staged candidate integrity does not cover its exact artifact tree");
    }
    Ok(())
}

fn collect_regular_files(root: &Path) -> io::Result<Vec<String>> {
    fn walk(root: &Path, directory: &Path, output: &mut Vec<String>) -> io::Result<()> {
        let mut entries = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return invalid("staged candidate tree contains a symbolic link");
            }
            if file_type.is_dir() {
                walk(root, &path, output)?;
            } else if file_type.is_file() {
                output.push(
                    path.strip_prefix(root)
                        .map_err(|_| invalid_error("staged candidate artifact escaped its root"))?
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            } else {
                return invalid("staged candidate tree contains a non-regular artifact");
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    walk(root, root, &mut files)?;
    files.sort();
    Ok(files)
}

fn verify_source_inputs(
    package_root: &Path,
    build_plan: &serde_json::Map<String, serde_json::Value>,
    source_seal: &SourcePackageSeal,
) -> io::Result<()> {
    if source_seal.schema != SOURCE_PACKAGE_SEAL_SCHEMA
        || [
            &source_seal.package_id,
            &source_seal.manifest_digest,
            &source_seal.optimizer_stage_digest,
            &source_seal.exact_baseline_digest,
            &source_seal.scope_catalog_digest,
            &source_seal.package_integrity_contract_digest,
        ]
        .iter()
        .any(|value| value.is_empty())
    {
        return invalid("staged source package seal identity is invalid");
    }
    let sources = array(build_plan, "source_inputs", "staged candidate build plan")?;
    if sources.is_empty() {
        return invalid("staged candidate build plan has no sealed source inputs");
    }
    if source_seal.source_inputs.len() != sources.len() {
        return invalid("source package seal does not cover its exact source inputs");
    }
    let mut previous: Option<&str> = None;
    for source in sources {
        let source = source
            .as_object()
            .ok_or_else(|| invalid_error("staged source input must be an object"))?;
        if source.len() != 2 {
            return invalid("staged source input fields are invalid");
        }
        let path = text(source, "path", "staged source input")?;
        let digest = text(source, "digest", "staged source input")?;
        if previous.is_some_and(|value| value >= path) {
            return invalid("staged source input paths must be sorted and unique");
        }
        previous = Some(path);
        let sealed = source_seal.source_inputs.get(path).ok_or_else(|| {
            invalid_error("source package seal does not cover its exact source inputs")
        })?;
        if sealed.digest != digest {
            return invalid(format!(
                "source package seal digest disagrees with build plan: {path:?}",
            ));
        }
        let source_path = confined_path(package_root, path, "staged source input")?;
        if source_file_signature(&source_path)? != sealed.signature {
            return invalid(format!(
                "staged source input changed after candidate construction: {path:?}",
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn source_file_signature(path: &Path) -> io::Result<SourceFileSignature> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return invalid("staged source input must be a regular file");
    }
    Ok(SourceFileSignature {
        device: metadata.dev(),
        inode: metadata.ino(),
        byte_count: metadata.len(),
        modified_ns: unix_timestamp_ns(metadata.mtime(), metadata.mtime_nsec())?,
        changed_ns: unix_timestamp_ns(metadata.ctime(), metadata.ctime_nsec())?,
    })
}

#[cfg(not(unix))]
fn source_file_signature(_path: &Path) -> io::Result<SourceFileSignature> {
    invalid("source package seals require Unix file identity metadata")
}

fn unix_timestamp_ns(seconds: i64, nanoseconds: i64) -> io::Result<u64> {
    if seconds < 0 || !(0..1_000_000_000).contains(&nanoseconds) {
        return invalid("staged source input has invalid file timestamps");
    }
    u64::try_from(seconds)
        .ok()
        .and_then(|seconds| seconds.checked_mul(1_000_000_000))
        .and_then(|value| value.checked_add(nanoseconds as u64))
        .ok_or_else(|| invalid_error("staged source input timestamp overflowed"))
}

fn validate_candidate_id(candidate_id: &str) -> io::Result<()> {
    let suffix = candidate_id
        .strip_prefix("candidate_")
        .ok_or_else(|| invalid_error("staged candidate identity is invalid"))?;
    if suffix.len() != 32
        || !suffix
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid("staged candidate identity is invalid");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
