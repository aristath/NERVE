fn read_json<T: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<T, CircuitArtifactError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn resolve_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

fn product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::tiny_model_lowered_graph_path;

    fn fixture_model_index_path() -> PathBuf {
        tiny_model_lowered_graph_path()
    }

    include!("tests/circuit_contracts.rs");
    include!("tests/placement_routes.rs");
    include!("tests/capacity_packed_placement.rs");
    include!("tests/runtime_reports.rs");
    include!("tests/runtime_graph.rs");
}
