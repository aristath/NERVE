use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use nerve_runtime::{VulkanResidentModelPackageManifest, VulkanResidentRuntimeModel};

pub struct CalibrationPackage {
    source_path: PathBuf,
    manifest_dir: PathBuf,
    runtime_model: VulkanResidentRuntimeModel,
}

impl CalibrationPackage {
    pub fn load(source_path: &Path) -> Result<Self, io::Error> {
        let manifest =
            VulkanResidentModelPackageManifest::from_json_file(source_path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "failed to load compiled package {}: {error}",
                        source_path.display()
                    ),
                )
            })?;
        let runtime_model = manifest
            .mount_runtime_graph_controls(None, &BTreeMap::new(), &[], None)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
        Ok(Self {
            source_path: source_path.to_path_buf(),
            manifest_dir: source_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            runtime_model,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn manifest_dir(&self) -> &Path {
        &self.manifest_dir
    }

    pub fn runtime_model(&self) -> &VulkanResidentRuntimeModel {
        &self.runtime_model
    }

    pub fn reject_output_collision(&self, output: &Path) -> Result<(), io::Error> {
        reject_output_collision(&self.source_path, output)
    }
}

fn reject_output_collision(source: &Path, output: &Path) -> Result<(), io::Error> {
    if output.exists() && fs::canonicalize(source)? == fs::canonicalize(output)? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "calibration output must not replace the compiled package manifest",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_test_directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-gpu-bench-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn package_catalog_cannot_overwrite_its_source_manifest() {
        let directory = temporary_test_directory("source-collision");
        fs::create_dir_all(&directory).unwrap();
        let package = directory.join("package.json");
        fs::write(&package, b"source").unwrap();
        let alias = directory.join(".").join("package.json");

        let error = reject_output_collision(&package, &alias).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("must not replace"));
        fs::remove_dir_all(directory).unwrap();
    }
}
