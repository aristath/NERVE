use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

pub(super) fn confined_path(root: &Path, reference: &str, label: &str) -> io::Result<PathBuf> {
    if reference.is_empty() {
        return invalid(format!("{label} path must not be empty"));
    }
    let relative = Path::new(reference);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || relative.to_string_lossy() != reference
    {
        return invalid(format!("{label} must be a canonical package-relative path"));
    }
    let mut lexical = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            unreachable!("relative components were validated above");
        };
        lexical.push(component);
        let metadata = fs::symlink_metadata(&lexical)
            .map_err(|error| invalid_error(format!("{label} is missing or unreadable: {error}")))?;
        if metadata.file_type().is_symlink() {
            return invalid(format!("{label} crosses a symbolic link"));
        }
    }
    let canonical = lexical
        .canonicalize()
        .map_err(|error| invalid_error(format!("{label} is missing or unreadable: {error}")))?;
    if !canonical.starts_with(root) {
        return invalid(format!("{label} escapes its artifact root"));
    }
    Ok(canonical)
}

pub(super) fn read_json<T: DeserializeOwned>(path: &Path, label: &str) -> io::Result<T> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| invalid_error(format!("invalid {label} JSON: {error}")))
}

pub(super) fn read_object(path: &Path, label: &str) -> io::Result<Map<String, Value>> {
    let value: Value = read_json(path, label)?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| invalid_error(format!("{label} must be a JSON object")))
}

pub(super) fn from_value<T: DeserializeOwned>(value: Value, label: &str) -> io::Result<T> {
    serde_json::from_value(value)
        .map_err(|error| invalid_error(format!("invalid {label}: {error}")))
}

pub(super) fn require_schema(
    document: &Map<String, Value>,
    expected: &str,
    label: &str,
) -> io::Result<()> {
    let schema = text(document, "schema", label)?;
    if schema != expected {
        return invalid(format!("unsupported {label} schema {schema:?}"));
    }
    Ok(())
}

pub(super) fn required<'a>(
    document: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<&'a Value> {
    document
        .get(field)
        .ok_or_else(|| invalid_error(format!("{label} is missing {field:?}")))
}

pub(super) fn text<'a>(
    document: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<&'a str> {
    required(document, field, label)?
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid_error(format!("{label} {field:?} must be non-empty text")))
}

pub(super) fn object<'a>(
    document: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<&'a Map<String, Value>> {
    required(document, field, label)?
        .as_object()
        .ok_or_else(|| invalid_error(format!("{label} {field:?} must be an object")))
}

pub(super) fn array<'a>(
    document: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<&'a Vec<Value>> {
    required(document, field, label)?
        .as_array()
        .ok_or_else(|| invalid_error(format!("{label} {field:?} must be an array")))
}

pub(super) fn string_array(
    document: &Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<Vec<String>> {
    let values = array(document, field, label)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    invalid_error(format!(
                        "{label} {field:?} must contain only non-empty text"
                    ))
                })
        })
        .collect::<io::Result<Vec<_>>>()?;
    if !strictly_sorted_unique(&values.iter().map(String::as_str).collect::<Vec<_>>()) {
        return invalid(format!("{label} {field:?} must be sorted and unique"));
    }
    Ok(values)
}

pub(super) fn unsigned(
    document: &Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<usize> {
    let value = unsigned_u64(document, field, label)?;
    usize::try_from(value).map_err(|_| invalid_error(format!("{label} {field:?} exceeds usize")))
}

pub(super) fn unsigned_u64(
    document: &Map<String, Value>,
    field: &str,
    label: &str,
) -> io::Result<u64> {
    required(document, field, label)?
        .as_u64()
        .ok_or_else(|| invalid_error(format!("{label} {field:?} must be an unsigned integer")))
}

pub(super) fn strictly_sorted_unique(values: &[&str]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub(super) fn invalid<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_error(message))
}

pub(super) fn invalid_error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
