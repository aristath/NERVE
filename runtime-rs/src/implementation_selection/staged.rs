use super::artifacts::{
    array, confined_path, invalid, invalid_error, read_json, read_object, require_schema, text,
};
use super::catalog::validate_mount_plan;
use super::{
    RUNTIME_MOUNT_PLAN_SCHEMA, RuntimeMountPlan, RuntimeStagedCandidate,
    STAGED_ARTIFACT_DIGEST_SCHEMA, STAGED_CANDIDATE_INTEGRITY_FILE,
    STAGED_CANDIDATE_INTEGRITY_SCHEMA,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

const CANDIDATE_BUILD_PLAN_SCHEMA: &str = "nerve.optimizer.candidate_build_plan.v1";
const REPRESENTATION_CANDIDATE_SCHEMA: &str = "nerve.optimizer.representation_candidate.v1";

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
        verify_source_inputs(&package_root, &build_plan)?;

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
            .component_replacements
            .iter()
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
) -> io::Result<()> {
    let sources = array(build_plan, "source_inputs", "staged candidate build plan")?;
    if sources.is_empty() {
        return invalid("staged candidate build plan has no sealed source inputs");
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
        let source_path = confined_path(package_root, path, "staged source input")?;
        let observed = format!(
            "{STAGED_ARTIFACT_DIGEST_SCHEMA}:{}",
            sha256_file(&source_path)?
        );
        if observed != digest {
            return invalid(format!(
                "staged source input changed after candidate construction: {path:?}",
            ));
        }
    }
    Ok(())
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
