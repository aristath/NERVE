#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeModelPathKind {
    CompiledPackage { manifest: PathBuf },
    SafetensorsSource { model_dir: PathBuf },
}

pub fn classify_runtime_model_path(
    path: impl AsRef<Path>,
) -> Result<RuntimeModelPathKind, RuntimeEditorError> {
    let path = path.as_ref();
    if path.is_file()
        && path.file_name().and_then(|name| name.to_str()) == Some(RUNTIME_PACKAGE_MANIFEST_FILE)
    {
        validate_runtime_package_manifest_header(path)?;
        return Ok(RuntimeModelPathKind::CompiledPackage {
            manifest: path.to_path_buf(),
        });
    }
    if !path.is_dir() {
        return Err(RuntimeEditorError(format!(
            "model path does not exist or is not a directory: {}",
            path.display()
        )));
    }
    let manifest = path.join(RUNTIME_PACKAGE_MANIFEST_FILE);
    if manifest.is_file() {
        validate_runtime_package_manifest_header(&manifest)?;
        return Ok(RuntimeModelPathKind::CompiledPackage { manifest });
    }
    let has_safetensors = path.read_dir()?.filter_map(Result::ok).any(|entry| {
            entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors")
        });
    if path.join("config.json").is_file()
        || path.join("tokenizer.json").is_file()
        || has_safetensors
    {
        return Ok(RuntimeModelPathKind::SafetensorsSource {
            model_dir: path.to_path_buf(),
        });
    }
    Err(RuntimeEditorError(format!(
        "{} is neither a NERVE package nor a discoverable Safetensors model",
        path.display()
    )))
}

fn validate_runtime_package_manifest_header(path: &Path) -> Result<(), RuntimeEditorError> {
    let file = std::fs::File::open(path)?;
    let manifest: Value = serde_json::from_reader(file).map_err(|error| {
        RuntimeEditorError(format!(
            "compiled package manifest is not valid JSON: {error}; path: {}",
            path.display()
        ))
    })?;
    let schema = manifest
        .get("schema")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    if schema != VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA {
        return Err(RuntimeEditorError(format!(
            "unsupported resident model package manifest schema {schema:?}; recompile the model"
        )));
    }
    if manifest
        .get("package_id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(RuntimeEditorError(format!(
            "compiled package manifest has no package_id: {}",
            path.display()
        )));
    }
    Ok(())
}
