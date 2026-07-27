use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::stream_circuit::{
    CircuitNode, CircuitPort, ParameterRef, ResolvedCircuitArtifact, ResolvedLoweredExecutionGraph,
    StatePort, StreamCircuit,
};

pub const TENSOR_INDEX_SCHEMA: &str = "nerve.tensor_index.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitPlanError(pub String);

impl Display for CircuitPlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CircuitPlanError {}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorIndex {
    pub schema: String,
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorMetadata>,
}

impl TensorIndex {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitPlanError> {
        Self::from_json_file_with_source_root(path, None)
    }

    pub fn from_package_json_file(path: impl AsRef<Path>) -> Result<Self, CircuitPlanError> {
        let path = path.as_ref();
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        Self::from_json_file_with_source_root(path, Some(root))
    }

    pub fn from_package_fragment_json_file(
        path: impl AsRef<Path>,
        package_root: impl AsRef<Path>,
    ) -> Result<Self, CircuitPlanError> {
        Self::from_json_file_with_source_root(
            path,
            Some(package_root.as_ref()),
        )
    }

    fn from_json_file_with_source_root(
        path: impl AsRef<Path>,
        source_root: Option<&Path>,
    ) -> Result<Self, CircuitPlanError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|error| CircuitPlanError(error.to_string()))?;
        let mut index: Self =
            serde_json::from_slice(&bytes).map_err(|error| CircuitPlanError(error.to_string()))?;
        if index.schema != TENSOR_INDEX_SCHEMA {
            return Err(CircuitPlanError(format!(
                "unsupported tensor index schema {:?}",
                index.schema
            )));
        }
        let root = source_root
            .map(|root| {
                root.canonicalize().map_err(|error| {
                    CircuitPlanError(format!(
                        "tensor-index source root is unreadable: {error}"
                    ))
                })
            })
            .transpose()?;
        let root = root.as_deref().unwrap_or_else(|| {
            path.parent().unwrap_or_else(|| Path::new("."))
        });
        for (tensor, metadata) in &mut index.tensors {
            if let Some(source_file) = &metadata.source_file {
                let source_path = Path::new(source_file);
                if source_root.is_some()
                    && (source_file.is_empty()
                        || source_path.is_absolute()
                        || source_path.components().any(|component| {
                            matches!(
                                component,
                                std::path::Component::ParentDir
                                    | std::path::Component::RootDir
                                    | std::path::Component::Prefix(_)
                            )
                        }))
                {
                    return Err(CircuitPlanError(format!(
                        "package tensor {tensor:?} source path {source_file:?} must stay inside the package"
                    )));
                }
                if !source_path.is_absolute() {
                    let resolved = root.join(source_path);
                    let resolved = if source_root.is_some() {
                        resolved.canonicalize().map_err(|error| {
                            CircuitPlanError(format!(
                                "package tensor {tensor:?} source file is unreadable: {error}"
                            ))
                        })?
                    } else {
                        resolved
                    };
                    if source_root.is_some() && !resolved.starts_with(root) {
                        return Err(CircuitPlanError(format!(
                            "package tensor {tensor:?} source path escapes the package"
                        )));
                    }
                    metadata.source_file =
                        Some(resolved.to_string_lossy().into_owned());
                }
            } else if source_root.is_some() {
                return Err(CircuitPlanError(format!(
                    "package tensor {tensor:?} has no source file"
                )));
            }
            if source_root.is_some()
                && metadata.data_sha256.as_deref().is_none_or(|digest| {
                    digest.len() != 64
                        || !digest
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
            {
                return Err(CircuitPlanError(format!(
                    "package tensor {tensor:?} has no valid data SHA-256"
                )));
            }
        }
        Ok(index)
    }

    pub fn merge_fragment(
        &mut self,
        fragment: TensorIndex,
    ) -> Result<(), CircuitPlanError> {
        if fragment.schema != TENSOR_INDEX_SCHEMA {
            return Err(CircuitPlanError(format!(
                "unsupported tensor-index fragment schema {:?}",
                fragment.schema
            )));
        }
        for (tensor, metadata) in fragment.tensors {
            if self.tensors.contains_key(&tensor) {
                return Err(CircuitPlanError(format!(
                    "tensor-index fragment collides with existing tensor {tensor:?}"
                )));
            }
            self.tensors.insert(tensor, metadata);
        }
        Ok(())
    }

    pub fn tensor_shape(&self, tensor: &str) -> Option<&[usize]> {
        self.tensors.get(tensor).map(|metadata| {
            metadata
                .logical_shape
                .as_deref()
                .unwrap_or(metadata.shape.as_slice())
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    #[serde(default)]
    pub logical_shape: Option<Vec<usize>>,
    #[serde(default)]
    pub parameter_count: Option<usize>,
    #[serde(default)]
    pub byte_count: Option<usize>,
    #[serde(default)]
    pub data_offsets: Option<Vec<usize>>,
    #[serde(default)]
    pub source_file: Option<String>,
    #[serde(default)]
    pub data_sha256: Option<String>,
    #[serde(default)]
    pub layout: Option<String>,
}

#[cfg(test)]
mod tensor_index_fragment_tests {
    use super::*;

    fn metadata() -> TensorMetadata {
        TensorMetadata {
            dtype: "BF16".to_string(),
            shape: vec![2, 2],
            logical_shape: None,
            parameter_count: Some(4),
            byte_count: Some(8),
            data_offsets: Some(vec![0, 8]),
            source_file: Some("fixture.bin".to_string()),
            data_sha256: Some("0".repeat(64)),
            layout: Some("row_major".to_string()),
        }
    }

    #[test]
    fn tensor_index_fragment_adds_new_parameters_without_overriding_source() {
        let mut source = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([(
                "source.weight".to_string(),
                metadata(),
            )]),
        };
        let fragment = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([(
                "candidate.weight".to_string(),
                metadata(),
            )]),
        };

        source.merge_fragment(fragment).unwrap();

        assert_eq!(source.tensors.len(), 2);
        assert!(source.tensors.contains_key("candidate.weight"));
    }

    #[test]
    fn tensor_index_fragment_cannot_shadow_exact_parameters() {
        let mut source = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([(
                "source.weight".to_string(),
                metadata(),
            )]),
        };
        let fragment = TensorIndex {
            schema: TENSOR_INDEX_SCHEMA.to_string(),
            tensors: BTreeMap::from([(
                "source.weight".to_string(),
                metadata(),
            )]),
        };

        let error = source.merge_fragment(fragment).unwrap_err();

        assert!(error.to_string().contains("collides"));
        assert_eq!(source.tensors.len(), 1);
    }
}
