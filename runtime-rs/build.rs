use std::fs;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

const COMPILER_FINGERPRINT_SCHEMA: &str = "nerve.package_compiler_sha256.v2";
const COMPILER_SOURCE_MANIFEST: &str = "compiler_sources.txt";

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
}
