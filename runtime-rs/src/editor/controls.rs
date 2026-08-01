pub fn runtime_editor_control_schema(index: usize, raw: &Value) -> RuntimeEditorControlSchema {
    let object = raw.as_object();
    let string_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_str))
                .map(ToOwned::to_owned)
        })
    };
    let bool_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_bool))
        })
    };
    let value_field = |names: &[&str]| {
        object.and_then(|object| names.iter().find_map(|name| object.get(*name).cloned()))
    };
    let number_field = |names: &[&str]| {
        object.and_then(|object| {
            names
                .iter()
                .find_map(|name| object.get(*name).and_then(Value::as_f64))
        })
    };
    let id = string_field(&["id", "property_id"]).unwrap_or_else(|| format!("control_{index}"));
    let name = string_field(&["name", "label"]).unwrap_or_else(|| id.clone());
    let declared_type = string_field(&["value_type", "type"])
        .unwrap_or_else(|| "unspecified".to_string())
        .to_lowercase();
    let choices = object
        .and_then(|object| {
            ["choices", "values", "enum"]
                .into_iter()
                .find_map(|name| object.get(name).and_then(Value::as_array))
        })
        .map(|choices| {
            choices
                .iter()
                .map(|choice| {
                    if let Some(object) = choice.as_object() {
                        let value = object.get("value").cloned().unwrap_or(Value::Null);
                        let label = object
                            .get("label")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| display_json_value(&value));
                        RuntimeEditorControlChoice { value, label }
                    } else {
                        RuntimeEditorControlChoice {
                            value: choice.clone(),
                            label: display_json_value(choice),
                        }
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let kind = match declared_type.as_str() {
        "bool" | "boolean" | "toggle" => RuntimeEditorControlKind::Boolean,
        "int" | "integer" | "u32" | "u64" | "i32" | "i64" => RuntimeEditorControlKind::Integer,
        "float" | "number" | "f32" | "f64" => RuntimeEditorControlKind::Number,
        "string" | "text" => RuntimeEditorControlKind::Text,
        "enum" | "enumeration" | "select" if !choices.is_empty() => {
            RuntimeEditorControlKind::Enumeration { choices }
        }
        "readonly" | "read_only" => RuntimeEditorControlKind::ReadOnly,
        _ => RuntimeEditorControlKind::Unsupported {
            declared_type: declared_type.clone(),
        },
    };
    RuntimeEditorControlSchema {
        id,
        name,
        description: string_field(&["description", "help"]),
        kind,
        current_value: value_field(&["current_value", "current", "value"]),
        default_value: value_field(&["default_value", "default"]),
        minimum: number_field(&["minimum", "min"]),
        maximum: number_field(&["maximum", "max"]),
        step: number_field(&["step"]),
        units: string_field(&["units", "unit"]),
        editable_at_runtime: bool_field(&["editable_at_runtime", "runtime_editable"])
            .unwrap_or(false),
        requires_state_reset: bool_field(&["requires_state_reset"]).unwrap_or(false),
        requires_remount: bool_field(&["requires_remount"]).unwrap_or(false),
        requires_recompile: bool_field(&["requires_recompile"]).unwrap_or(false),
        scope: string_field(&["scope"])
            .unwrap_or_else(|| "instance".to_string())
            .to_lowercase(),
        raw: raw.clone(),
    }
}

pub fn validate_runtime_editor_control_value(
    schema: &RuntimeEditorControlSchema,
    value: &Value,
) -> Result<(), RuntimeEditorError> {
    if !schema.editable_at_runtime {
        return Err(RuntimeEditorError(format!(
            "control {:?} is not editable at runtime",
            schema.id
        )));
    }
    if schema.scope != "instance" {
        return Err(RuntimeEditorError(format!(
            "control {:?} has {:?} scope; this editor changes node instances",
            schema.id, schema.scope
        )));
    }
    let numeric = match &schema.kind {
        RuntimeEditorControlKind::Boolean if value.is_boolean() => None,
        RuntimeEditorControlKind::Integer
            if value.as_i64().is_some() || value.as_u64().is_some() =>
        {
            value.as_f64()
        }
        RuntimeEditorControlKind::Number if value.as_f64().is_some() => value.as_f64(),
        RuntimeEditorControlKind::Text if value.is_string() => None,
        RuntimeEditorControlKind::Enumeration { choices }
            if choices.iter().any(|choice| choice.value == *value) =>
        {
            None
        }
        RuntimeEditorControlKind::ReadOnly => {
            return Err(RuntimeEditorError(format!(
                "control {:?} is read-only",
                schema.id
            )));
        }
        RuntimeEditorControlKind::Unsupported { declared_type } => {
            return Err(RuntimeEditorError(format!(
                "control {:?} uses unsupported value type {:?}",
                schema.id, declared_type
            )));
        }
        _ => {
            return Err(RuntimeEditorError(format!(
                "control {:?} received a value incompatible with its declared type",
                schema.id
            )));
        }
    };
    if let Some(numeric) = numeric {
        if schema.minimum.is_some_and(|minimum| numeric < minimum) {
            return Err(RuntimeEditorError(format!(
                "value {numeric} is below minimum {}",
                schema.minimum.unwrap_or_default()
            )));
        }
        if schema.maximum.is_some_and(|maximum| numeric > maximum) {
            return Err(RuntimeEditorError(format!(
                "value {numeric} is above maximum {}",
                schema.maximum.unwrap_or_default()
            )));
        }
        if let Some(step) = schema.step {
            if !step.is_finite() || step <= 0.0 {
                return Err(RuntimeEditorError(format!(
                    "control {:?} declares a non-positive or non-finite step",
                    schema.id
                )));
            }
            if matches!(schema.kind, RuntimeEditorControlKind::Integer)
                && (step - step.round()).abs() > f64::EPSILON
            {
                return Err(RuntimeEditorError(format!(
                    "integer control {:?} declares non-whole step {step}",
                    schema.id
                )));
            }
            let anchor = schema.minimum.unwrap_or(0.0);
            let step_position = (numeric - anchor) / step;
            let tolerance = 1e-9 * step_position.abs().max(1.0);
            if (step_position - step_position.round()).abs() > tolerance {
                return Err(RuntimeEditorError(format!(
                    "value {numeric} does not align to step {step} from {anchor}",
                )));
            }
        }
    }
    Ok(())
}

fn display_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn allocate_instance_id(
    source_id: &str,
    occurrence: usize,
    used_instance_ids: &BTreeSet<String>,
) -> String {
    let preferred = if occurrence == 1 {
        source_id.to_string()
    } else {
        format!("{source_id}@{occurrence}")
    };
    if !used_instance_ids.contains(&preferred) {
        return preferred;
    }
    let mut suffix = occurrence.max(2);
    loop {
        let candidate = format!("{source_id}@{suffix}");
        if !used_instance_ids.contains(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn available_layer_range(source_by_layer: &BTreeMap<usize, Vec<String>>) -> String {
    let mut layers = source_by_layer.keys().copied();
    let Some(mut start) = layers.next() else {
        return "none".to_string();
    };
    let mut end = start;
    let mut ranges = Vec::new();
    for layer in layers {
        if end.checked_add(1) == Some(layer) {
            end = layer;
        } else {
            ranges.push(if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            });
            start = layer;
            end = layer;
        }
    }
    ranges.push(if start == end {
        start.to_string()
    } else {
        format!("{start}-{end}")
    });
    ranges.join(", ")
}

#[cfg(test)]
mod control_tests {
    use super::*;

    #[test]
    fn available_layer_description_does_not_invent_layers_in_sparse_catalogs() {
        let layers = BTreeMap::from([
            (0, vec!["zero".to_string()]),
            (2, vec!["two".to_string()]),
            (3, vec!["three".to_string()]),
            (7, vec!["seven".to_string()]),
        ]);
        assert_eq!(available_layer_range(&layers), "0, 2-3, 7");
    }

    #[test]
    fn control_schema_normalizes_aliases_choices_and_lifecycle_metadata() {
        let schema = runtime_editor_control_schema(
            3,
            &serde_json::json!({
                "property_id": "precision",
                "label": "Precision",
                "help": "Execution representation",
                "value_type": "SELECT",
                "values": [
                    {"value": "fp8", "label": "FP8 native"},
                    "bf16"
                ],
                "value": "fp8",
                "default": "bf16",
                "runtime_editable": true,
                "requires_state_reset": true,
                "requires_remount": true,
                "requires_recompile": false,
                "scope": "INSTANCE"
            }),
        );
        assert_eq!(schema.id, "precision");
        assert_eq!(schema.name, "Precision");
        assert_eq!(schema.description.as_deref(), Some("Execution representation"));
        assert_eq!(schema.scope, "instance");
        assert_eq!(schema.current_value, Some(serde_json::json!("fp8")));
        assert_eq!(schema.default_value, Some(serde_json::json!("bf16")));
        assert!(schema.editable_at_runtime);
        assert!(schema.requires_state_reset);
        assert!(schema.requires_remount);
        assert!(!schema.requires_recompile);
        let RuntimeEditorControlKind::Enumeration { choices } = schema.kind else {
            panic!("select control was not represented as an enumeration");
        };
        assert_eq!(choices[0].label, "FP8 native");
        assert_eq!(choices[1].label, "bf16");
    }

    #[test]
    fn control_validation_rejects_noneditable_scope_type_and_enum_violations() {
        let editable_boolean = runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id":"enabled", "type":"boolean", "editable_at_runtime":true,
                "scope":"instance"
            }),
        );
        assert!(
            validate_runtime_editor_control_value(&editable_boolean, &Value::Bool(true)).is_ok()
        );
        assert!(
            validate_runtime_editor_control_value(&editable_boolean, &serde_json::json!(1))
                .unwrap_err()
                .to_string()
                .contains("declared type")
        );

        let mut noneditable = editable_boolean.clone();
        noneditable.editable_at_runtime = false;
        assert!(
            validate_runtime_editor_control_value(&noneditable, &Value::Bool(true))
                .unwrap_err()
                .to_string()
                .contains("not editable")
        );
        let mut shared = editable_boolean.clone();
        shared.scope = "source".to_string();
        assert!(
            validate_runtime_editor_control_value(&shared, &Value::Bool(true))
                .unwrap_err()
                .to_string()
                .contains("source")
        );

        for (declared_type, expected) in [
            ("readonly", "read-only"),
            ("unrecognized", "unsupported value type"),
        ] {
            let schema = runtime_editor_control_schema(
                0,
                &serde_json::json!({
                    "id":"contract", "type":declared_type,
                    "editable_at_runtime":true, "scope":"instance"
                }),
            );
            assert!(
                validate_runtime_editor_control_value(&schema, &Value::Null)
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }

        let enumeration = runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id":"mode", "type":"enum", "choices":["a", "b"],
                "editable_at_runtime":true, "scope":"instance"
            }),
        );
        assert!(
            validate_runtime_editor_control_value(&enumeration, &serde_json::json!("c"))
                .unwrap_err()
                .to_string()
                .contains("declared type")
        );
    }

    #[test]
    fn numeric_control_validation_enforces_bounds_step_and_step_contract() {
        let integer = runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id":"window", "type":"integer", "min":2, "max":10, "step":2,
                "editable_at_runtime":true, "scope":"instance"
            }),
        );
        assert!(validate_runtime_editor_control_value(&integer, &serde_json::json!(4)).is_ok());
        for (value, expected) in [(1, "below minimum"), (11, "above maximum"), (5, "align")]
        {
            assert!(
                validate_runtime_editor_control_value(&integer, &serde_json::json!(value))
                    .unwrap_err()
                    .to_string()
                    .contains(expected)
            );
        }

        let mut bad_step = integer.clone();
        bad_step.step = Some(0.0);
        assert!(
            validate_runtime_editor_control_value(&bad_step, &serde_json::json!(4))
                .unwrap_err()
                .to_string()
                .contains("non-positive")
        );
        bad_step.step = Some(0.5);
        assert!(
            validate_runtime_editor_control_value(&bad_step, &serde_json::json!(4))
                .unwrap_err()
                .to_string()
                .contains("non-whole")
        );

        let number = runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id":"ratio", "type":"number", "min":0.0, "max":1.0, "step":0.1,
                "editable_at_runtime":true, "scope":"instance"
            }),
        );
        assert!(validate_runtime_editor_control_value(&number, &serde_json::json!(0.3)).is_ok());
    }

    #[test]
    fn instance_id_allocator_never_reuses_an_existing_identity() {
        let used = BTreeSet::from([
            "layer_00".to_string(),
            "layer_00@2".to_string(),
            "layer_00@3".to_string(),
        ]);
        assert_eq!(allocate_instance_id("layer_00", 1, &used), "layer_00@4");
        assert_eq!(allocate_instance_id("layer_01", 1, &used), "layer_01");
    }
}
