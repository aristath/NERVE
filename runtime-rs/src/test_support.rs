use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMPORARY_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn new(label: &str) -> Self {
        let sequence = TEMPORARY_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("nerve-{label}-{}-{sequence}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

pub(crate) fn tiny_model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-fixtures")
        .join("tiny_model")
}

pub(crate) fn tiny_model_lowered_graph_path() -> PathBuf {
    tiny_model_dir()
        .join("lowered")
        .join("execution_graph.circuits.json")
}

pub(crate) fn tiny_model_tensor_index_path() -> PathBuf {
    tiny_model_dir().join("tensors.json")
}

#[cfg(feature = "vulkan")]
pub(crate) fn tiny_model_package_manifest_path() -> PathBuf {
    tiny_model_dir().join("vulkan_resident_package.json")
}
