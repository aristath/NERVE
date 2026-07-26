use super::schema::{CalibrationArtifactRecord, HardwareCalibrationWorkload};
use crate::vulkan::read_spirv_words;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;

pub(super) fn compile_calibration_shader(
    workload: &HardwareCalibrationWorkload,
    artifact_index: usize,
    source: &str,
    stage: &str,
    artifact_directory: &Path,
) -> Result<(Vec<u32>, CalibrationArtifactRecord), String> {
    let artifact = workload.artifacts.get(artifact_index).ok_or_else(|| {
        format!(
            "Vulkan workload {} has no shader artifact at index {artifact_index}",
            workload.workload_id
        )
    })?;
    let stem = sanitize_artifact_name(&format!("{}_{}", artifact.name, workload.workload_id));
    let source_path = artifact_directory.join(format!("{stem}.{stage}"));
    let spirv_path = artifact_directory.join(format!("{stem}.spv"));
    fs::create_dir_all(artifact_directory).map_err(|error| {
        format!("could not create shader artifact directory {artifact_directory:?}: {error}")
    })?;
    fs::write(&source_path, source)
        .map_err(|error| format!("could not write shader source {source_path:?}: {error}"))?;
    let output = Command::new("glslangValidator")
        .arg("-V")
        .arg("--target-env")
        .arg("vulkan1.4")
        .arg("-S")
        .arg(stage)
        .arg("-o")
        .arg(&spirv_path)
        .arg(&source_path)
        .output()
        .map_err(|error| format!("could not execute glslangValidator: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "glslangValidator rejected calibration shader {}:\n{}{}",
            workload.workload_id,
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout)
        ));
    }
    let bytes = fs::read(&spirv_path)
        .map_err(|error| format!("could not read compiled shader {spirv_path:?}: {error}"))?;
    let words = read_spirv_words(&spirv_path)
        .map_err(|error| format!("could not parse compiled shader {spirv_path:?}: {error}"))?;
    let record = calibration_artifact_record(
        workload,
        artifact_index,
        &bytes,
        spirv_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("compiled shader path {spirv_path:?} has no file name"))?,
    )?;
    Ok((words, record))
}

pub(super) fn persist_calibration_artifact(
    workload: &HardwareCalibrationWorkload,
    artifact_index: usize,
    extension: &str,
    bytes: &[u8],
    artifact_directory: &Path,
) -> Result<CalibrationArtifactRecord, String> {
    let artifact = workload.artifacts.get(artifact_index).ok_or_else(|| {
        format!(
            "Vulkan workload {} has no artifact at index {artifact_index}",
            workload.workload_id
        )
    })?;
    fs::create_dir_all(artifact_directory).map_err(|error| {
        format!("could not create artifact directory {artifact_directory:?}: {error}")
    })?;
    let file_name = format!(
        "{}_{}.{}",
        sanitize_artifact_name(&artifact.name),
        workload.workload_id,
        sanitize_artifact_name(extension)
    );
    let path = artifact_directory.join(&file_name);
    fs::write(&path, bytes)
        .map_err(|error| format!("could not write calibration artifact {path:?}: {error}"))?;
    calibration_artifact_record(workload, artifact_index, bytes, &file_name)
}

fn calibration_artifact_record(
    workload: &HardwareCalibrationWorkload,
    artifact_index: usize,
    bytes: &[u8],
    relative_path: &str,
) -> Result<CalibrationArtifactRecord, String> {
    let artifact = workload.artifacts.get(artifact_index).ok_or_else(|| {
        format!(
            "Vulkan workload {} has no artifact at index {artifact_index}",
            workload.workload_id
        )
    })?;
    Ok(CalibrationArtifactRecord {
        name: artifact.name.clone(),
        kind: artifact.kind.clone(),
        digest: format!(
            "nerve.calibration_artifact_sha256.v1:{:x}",
            Sha256::digest(&bytes)
        ),
        byte_length: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        relative_path: relative_path.to_string(),
    })
}

fn sanitize_artifact_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}
