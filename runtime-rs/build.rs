use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const HARDWARE_DISCOVERY_FINGERPRINT_SCHEMA: &str = "nerve.hardware_discovery_sha256.v1";
const HARDWARE_CALIBRATOR_FINGERPRINT_SCHEMA: &str = "nerve.hardware_calibrator_sha256.v1";
const RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA: &str = "nerve.runtime_implementation_sha256.v1";

fn recursive_directory_files_with_extension(
    path: &Path,
    prefix: &str,
    extension: &str,
) -> Vec<(String, PathBuf)> {
    let mut pending = vec![path.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).unwrap_or_else(|error| {
            panic!("failed to read runtime input directory {directory:?}: {error}")
        }) {
            let path = entry.expect("failed to read runtime input entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
                let relative = path
                    .strip_prefix(
                        path.ancestors()
                            .find(|ancestor| {
                                ancestor.file_name().and_then(|value| value.to_str()) == Some("src")
                            })
                            .expect("runtime source path has src ancestor"),
                    )
                    .expect("runtime source remains below src")
                    .to_string_lossy()
                    .replace('\\', "/");
                files.push((format!("{prefix}/{relative}"), path));
            }
        }
    }
    files
}

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("Cargo provides CARGO_MANIFEST_DIR to the build script"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/gpu_residency_gate.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("gpu_residency_gate.spv"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/distributed_sum_f32.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("distributed_sum_f32.spv"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/distributed_sum_f32_to_bf16.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("distributed_sum_f32_to_bf16.spv"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/distributed_commit_residency_fault.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("distributed_commit_residency_fault.spv"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/distributed_sum_f32_add_bf16_residual.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("distributed_sum_f32_add_bf16_residual.spv"),
    );
    compile_runtime_shader(
        &manifest_dir.join("shaders/distributed_sum_f32_scale_packed_bf16_to_bf16.comp"),
        &PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"))
            .join("distributed_sum_f32_scale_packed_bf16_to_bf16.spv"),
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

    let mut runtime_inputs = [
        "Cargo.lock",
        "Cargo.toml",
        "build.rs",
        "shaders/distributed_commit_residency_fault.comp",
        "shaders/distributed_sum_f32.comp",
        "shaders/distributed_sum_f32_to_bf16.comp",
        "shaders/distributed_sum_f32_add_bf16_residual.comp",
        "shaders/distributed_sum_f32_scale_packed_bf16_to_bf16.comp",
        "shaders/gpu_residency_gate.comp",
    ]
    .into_iter()
    .map(|relative| (relative.to_string(), manifest_dir.join(relative)))
    .chain(recursive_directory_files_with_extension(
        &manifest_dir.join("src"),
        "src",
        "rs",
    ))
    .collect::<Vec<_>>();
    runtime_inputs.sort_by(|left, right| left.0.cmp(&right.0));
    let mut runtime_digest = Sha256::new();
    for (relative, path) in runtime_inputs {
        println!("cargo:rerun-if-changed={}", path.display());
        let source = fs::read(&path).unwrap_or_else(|error| {
            panic!("failed to read runtime-implementation input {path:?}: {error}")
        });
        runtime_digest.update((relative.len() as u64).to_le_bytes());
        runtime_digest.update(relative.as_bytes());
        runtime_digest.update((source.len() as u64).to_le_bytes());
        runtime_digest.update(source);
    }
    println!(
        "cargo:rustc-env=NERVE_RUNTIME_IMPLEMENTATION_FINGERPRINT={RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA}:{:x}",
        runtime_digest.finalize()
    );
}

fn compile_runtime_shader(source: &Path, output: &Path) {
    println!("cargo:rerun-if-changed={}", source.display());
    let compiler = std::env::var_os("GLSLANG_VALIDATOR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("glslangValidator"));
    let result = Command::new(&compiler)
        .arg("--target-env")
        .arg("vulkan1.4")
        .arg("-V")
        .arg("-S")
        .arg("comp")
        .arg("-o")
        .arg(output)
        .arg(source)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to launch {:?} while compiling runtime shader {:?}: {error}",
                compiler, source
            )
        });
    if !result.status.success() {
        panic!(
            "failed to compile runtime shader {:?} for Vulkan 1.4: {}{}",
            source,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr),
        );
    }
}
