use std::fs;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

const COMPILER_FINGERPRINT_SCHEMA: &str = "nerve.package_compiler_sha256.v2";
const COMPILER_SOURCE_MANIFEST: &str = "compiler_sources.txt";
const HARDWARE_DISCOVERY_FINGERPRINT_SCHEMA: &str = "nerve.hardware_discovery_sha256.v1";
const HARDWARE_CALIBRATOR_FINGERPRINT_SCHEMA: &str = "nerve.hardware_calibrator_sha256.v1";

fn directory_files(path: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to read compiler input directory {path:?}: {error}"))
        .map(|entry| entry.expect("failed to read compiler input entry").path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            Some((format!("{prefix}/{name}"), path))
        })
        .collect()
}

fn directory_files_with_extension(
    path: &Path,
    prefix: &str,
    extension: &str,
) -> Vec<(String, PathBuf)> {
    directory_files(path, prefix)
        .into_iter()
        .filter(|(_relative, path)| {
            path.extension().and_then(|value| value.to_str()) == Some(extension)
        })
        .collect()
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository_root = manifest_dir
        .parent()
        .expect("runtime crate must live inside the repository");
    let compiler_dir = repository_root.join("nerve");
    let compiler_source_manifest = compiler_dir.join(COMPILER_SOURCE_MANIFEST);
    println!(
        "cargo:rerun-if-changed={}",
        compiler_source_manifest.display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("shaders").display()
    );
    let descriptor_dir = compiler_dir.join("representation_optimizer/descriptors");
    println!("cargo:rerun-if-changed={}", descriptor_dir.display());
    let source_manifest = fs::read_to_string(&compiler_source_manifest).unwrap_or_else(|error| {
        panic!(
            "failed to read compiler source manifest {:?}: {error}",
            compiler_source_manifest
        )
    });
    let relative_sources = source_manifest
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let mut sorted_sources = relative_sources.clone();
    sorted_sources.sort_unstable();
    sorted_sources.dedup();
    assert!(
        !relative_sources.is_empty() && relative_sources == sorted_sources,
        "compiler source manifest {:?} must contain unique sorted paths",
        compiler_source_manifest
    );
    let mut inputs = relative_sources
        .into_iter()
        .map(|relative| {
            let path = Path::new(relative);
            assert!(
                path.components().count() >= 2
                    && path.components().next() == Some(Component::Normal("nerve".as_ref()))
                    && path
                        .components()
                        .all(|component| matches!(component, Component::Normal(_)))
                    && path.extension().and_then(|value| value.to_str()) == Some("py"),
                "invalid compiler source path {relative:?} in {:?}",
                compiler_source_manifest
            );
            let source = repository_root.join(path);
            assert!(
                source.is_file(),
                "compiler source {relative:?} declared by {:?} is missing",
                compiler_source_manifest
            );
            (relative.to_string(), source)
        })
        .chain(directory_files_with_extension(
            &descriptor_dir,
            "nerve/representation_optimizer/descriptors",
            "json",
        ))
        .chain(directory_files(
            &manifest_dir.join("shaders"),
            "runtime-rs/shaders",
        ))
        .collect::<Vec<_>>();
    inputs.sort_by(|left, right| left.0.cmp(&right.0));

    let mut digest = Sha256::new();
    for (relative, path) in inputs {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative_bytes = relative.as_bytes();
        let source_bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read compiler input {path:?}: {error}"));
        digest.update((relative_bytes.len() as u64).to_le_bytes());
        digest.update(relative_bytes);
        digest.update((source_bytes.len() as u64).to_le_bytes());
        digest.update(source_bytes);
    }
    println!(
        "cargo:rustc-env=NERVE_PACKAGE_COMPILER_FINGERPRINT={COMPILER_FINGERPRINT_SCHEMA}:{:x}",
        digest.finalize()
    );

    let hardware_discovery_sources = [
        "Cargo.lock",
        "Cargo.toml",
        "build.rs",
        "src/hardware_profile.rs",
        "src/hardware_profile/cpu.rs",
        "src/hardware_profile/schema.rs",
        "src/lib.rs",
        "src/vulkan_compute.rs",
        "src/vulkan_compute/device_catalog.rs",
        "src/vulkan_compute/device_types.rs",
        "src/vulkan_compute/features.rs",
        "src/vulkan_compute/hardware_profile.rs",
        "src/vulkan_compute/physical_device_capabilities.rs",
    ];
    let mut hardware_digest = Sha256::new();
    for relative in hardware_discovery_sources {
        let path = manifest_dir.join(relative);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read(&path).unwrap_or_else(|error| {
            panic!("failed to read hardware-discovery input {path:?}: {error}")
        });
        hardware_digest.update((relative.len() as u64).to_le_bytes());
        hardware_digest.update(relative.as_bytes());
        hardware_digest.update((source.len() as u64).to_le_bytes());
        hardware_digest.update(source);
    }
    println!(
        "cargo:rustc-env=NERVE_HARDWARE_DISCOVERY_FINGERPRINT={HARDWARE_DISCOVERY_FINGERPRINT_SCHEMA}:{:x}",
        hardware_digest.finalize()
    );

    let hardware_calibrator_sources = [
        "Cargo.lock",
        "Cargo.toml",
        "build.rs",
        "src/bin/nerve_calibrate.rs",
        "src/hardware_calibration.rs",
        "src/hardware_calibration/cpu.rs",
        "src/hardware_calibration/runner.rs",
        "src/hardware_calibration/schema.rs",
        "src/hardware_profile/schema.rs",
        "../nerve/hardware_calibration/__init__.py",
        "../nerve/hardware_calibration/contracts.py",
        "../nerve/hardware_calibration/planning.py",
        "../nerve/hardware_calibration/publication.py",
        "../nerve/hardware_calibration/statistics.py",
    ];
    let mut calibrator_digest = Sha256::new();
    for relative in hardware_calibrator_sources {
        let path = manifest_dir.join(relative);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read(&path).unwrap_or_else(|error| {
            panic!("failed to read hardware-calibrator input {path:?}: {error}")
        });
        calibrator_digest.update((relative.len() as u64).to_le_bytes());
        calibrator_digest.update(relative.as_bytes());
        calibrator_digest.update((source.len() as u64).to_le_bytes());
        calibrator_digest.update(source);
    }
    println!(
        "cargo:rustc-env=NERVE_HARDWARE_CALIBRATOR_FINGERPRINT={HARDWARE_CALIBRATOR_FINGERPRINT_SCHEMA}:{:x}",
        calibrator_digest.finalize()
    );
}
