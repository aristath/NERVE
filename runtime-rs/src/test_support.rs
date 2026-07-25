use std::path::{Path, PathBuf};

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
