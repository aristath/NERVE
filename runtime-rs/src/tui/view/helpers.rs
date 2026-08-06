fn buffer_line(buffer: &TextBuffer, width: usize) -> (usize, Line<'static>, usize) {
    let chars = buffer.text().chars().collect::<Vec<_>>();
    let cursor = buffer.cursor().min(chars.len());
    let mut start = cursor;
    let mut cursor_cells = 0usize;
    let cursor_capacity = width.saturating_sub(1);
    while start > 0 {
        let character_cells = chars[start - 1].width().unwrap_or(0);
        if cursor_cells + character_cells > cursor_capacity {
            break;
        }
        cursor_cells += character_cells;
        start -= 1;
    }
    let mut end = start;
    let mut line_cells = 0usize;
    while end < chars.len() {
        let character_cells = chars[end].width().unwrap_or(0);
        if line_cells + character_cells > width {
            break;
        }
        line_cells += character_cells;
        end += 1;
    }
    let selection = buffer.selection();
    let mut spans = Vec::new();
    let mut run_start = start;
    let mut run_selected = selection.is_some_and(|(left, right)| start >= left && start < right);
    for index in start..end {
        let selected = selection.is_some_and(|(left, right)| index >= left && index < right);
        if selected != run_selected {
            spans.push(buffer_span(&chars[run_start..index], run_selected));
            run_start = index;
            run_selected = selected;
        }
    }
    spans.push(buffer_span(&chars[run_start..end], run_selected));
    (start, Line::from(spans), cursor_cells)
}

fn buffer_span(chars: &[char], selected: bool) -> Span<'static> {
    let text = chars.iter().collect::<String>();
    if selected {
        Span::styled(text, Style::default().fg(Color::Black).bg(SIGNAL))
    } else {
        Span::styled(text, Style::default().fg(TEXT))
    }
}

fn device_color_map(instances: &[RuntimeEditorInstance]) -> BTreeMap<String, Color> {
    let colors = [
        Color::Rgb(103, 194, 255),
        Color::Rgb(150, 220, 150),
        Color::Rgb(199, 151, 255),
        Color::Rgb(92, 214, 196),
        Color::Rgb(255, 150, 190),
    ];
    let mut map = BTreeMap::new();
    for instance in instances {
        let next = colors[map.len() % colors.len()];
        map.entry(instance.device_id.clone()).or_insert(next);
    }
    map
}

fn centered_rect(
    area: Rect,
    percent_x: u16,
    percent_y: u16,
    min_width: u16,
    min_height: u16,
) -> Rect {
    let width = ((area.width as u32 * percent_x as u32 / 100) as u16)
        .max(min_width.min(area.width))
        .min(area.width);
    let height = ((area.height as u32 * percent_y as u32 / 100) as u16)
        .max(min_height.min(area.height))
        .min(area.height);
    Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    )
}

fn centered_line_area(area: Rect, height: u16) -> Rect {
    Rect::new(
        area.x,
        area.y + area.height.saturating_sub(height) / 2,
        area.width,
        height.min(area.height),
    )
}

fn truncate(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if value.width() <= width {
        return value.to_string();
    }
    if width == 1 {
        return "…".to_string();
    }
    let mut result = String::new();
    for character in value.chars() {
        let next = character.width().unwrap_or(0);
        if result.width() + next + 1 > width {
            break;
        }
        result.push(character);
    }
    result.push('…');
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::AppAction;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn rendered_text(app: &mut App, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .flat_map(|y| (0..width).map(move |x| buffer[(x, y)].symbol()))
            .collect::<String>()
    }

    #[test]
    fn truncation_respects_terminal_cell_width() {
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("λx", 4), "λx");
        assert_eq!(truncate("界x", 2), "…");
        assert_eq!(truncate("anything", 1), "…");
        assert_eq!(truncate("anything", 0), "");
    }

    #[test]
    fn centered_rect_never_exceeds_even_a_zero_sized_terminal() {
        for area in [Rect::new(0, 0, 0, 0), Rect::new(2, 3, 10, 4)] {
            let centered = centered_rect(area, 90, 90, 50, 20);
            assert!(centered.x >= area.x);
            assert!(centered.y >= area.y);
            assert!(centered.right() <= area.right());
            assert!(centered.bottom() <= area.bottom());
        }
    }

    #[test]
    fn text_buffer_view_keeps_cursor_visible() {
        let buffer = TextBuffer::new("[0,1,2,3]");
        let (start, line, cursor) = buffer_line(&buffer, 5);
        assert_eq!(start, 5);
        assert_eq!(cursor, 4);
        assert_eq!(line.to_string(), "2,3]");
    }

    #[test]
    fn text_buffer_view_uses_terminal_cell_width_for_wide_characters() {
        let buffer = TextBuffer::new("界界x");
        let (_, line, cursor) = buffer_line(&buffer, 3);
        assert!(line.width() <= 3);
        assert!(cursor <= 3);
        assert_eq!(line.to_string(), "x");
        assert_eq!(cursor, 1);
    }

    #[test]
    fn loaded_workspace_renders_signal_chain_at_normal_and_small_sizes() {
        let package = crate::test_support::tiny_model_dir();
        let editor = crate::editor::load_runtime_model_editor_without_hardware(package).unwrap();
        let mut app = App::new();
        app.install_editor(editor);
        for (width, height) in [(80, 24), (40, 12)] {
            let rendered = rendered_text(&mut app, width, height);
            assert!(rendered.contains("STREAM GRAPH"));
            assert!(rendered.contains("EXECUTION GRAPH"));
            assert!(rendered.contains("ZERO-BASED"));
            if width == 80 {
                assert!(rendered.contains("residency: eager"));
                assert_eq!(
                    app.action_at(width - 1, 1),
                    Some(AppAction::ToggleResourceResidencyPolicy)
                );
            }
        }
    }

    #[test]
    fn model_selector_and_node_modal_keep_actions_visible_in_small_terminals() {
        let mut app = App::new();
        let rendered = rendered_text(&mut app, 40, 12);
        assert!(rendered.contains("OPEN MODEL"));
        assert!(rendered.contains("Cancel"));

        let schema = crate::runtime_editor_control_schema(
            0,
            &serde_json::json!({
                "id": "window",
                "name": "Window",
                "description": "Local temporal span",
                "type": "integer",
                "current": 4,
                "min": 2,
                "max": 10,
                "step": 2,
                "editable_at_runtime": true,
                "scope": "instance"
            }),
        );
        let property = super::super::app::NodePropertyDraft::new(schema, serde_json::json!(4));
        let mut modal = NodeModalState {
            instance_id: "layer_00".to_string(),
            source: crate::RuntimeEditorSourceComponent {
                source_id: "layer_00".to_string(),
                layer_index: Some(0),
                operator_type: "transformer".to_string(),
                runtime_role: crate::CircuitRuntimeRole::SignalProcessor,
                implementation: "compiled_circuit".to_string(),
                behavioral_role: "stream_transform".to_string(),
                input_shape: vec![64],
                output_shape: vec![64],
                state_ports: Vec::new(),
                controls: Vec::new(),
                control_schemas: Vec::new(),
                parameter_ref_count: 4,
                node_count: 8,
                kernel_count: 3,
                semantic_modules: vec![crate::RuntimeEditorSemanticModule {
                    id: "layer".to_string(),
                    role: "layer".to_string(),
                    responsibility: "Editable source layer".to_string(),
                    parent_id: None,
                    child_ids: Vec::new(),
                    source_node_ids: vec!["project".to_string()],
                    parameter_ref_ids: vec!["weight".to_string()],
                    owned_state_port_ids: Vec::new(),
                    input_signals: vec!["input_frame".to_string()],
                    output_signals: vec!["output_frame".to_string()],
                    optimized_node_ids: vec!["fused_project".to_string()],
                    kernel_node_ids: vec!["fused_project".to_string()],
                    measured_cost: None,
                }],
                semantic_module_root_id: Some("layer".to_string()),
                implementation_options: Vec::new(),
            },
            occurrence: 1,
            device_ids: vec!["gpu0".to_string()],
            device_labels: vec!["gpu0 · fixture".to_string()],
            device_selectable: vec![true],
            device_diagnostics: Vec::new(),
            device_index: 0,
            original_device_id: "gpu0".to_string(),
            selected_implementation_id: None,
            implementation_selection_error: None,
            enabled: true,
            policy: NodePolicyKind::Independent,
            policy_targets: Vec::new(),
            policy_target_index: 0,
            properties: vec![property],
            anatomy_expanded: false,
            anatomy_scroll: 0,
            focus_row: 5,
            error: None,
        };
        modal.source.implementation_options.push(
            crate::RuntimeEditorImplementationOption {
                implementation_id:
                    "implementation_verified_fixture".to_string(),
                candidate_id: "candidate_verified_fixture".to_string(),
                scope_ids: vec!["scope_layer_00".to_string()],
                runtime_predicate: serde_json::from_value(
                    serde_json::json!({
                        "schema": crate::RUNTIME_IMPLEMENTATION_PREDICATE_SCHEMA,
                        "predicate_id": "runtime_predicate_fixture",
                        "hardware": {
                            "measured_profile_ids": [format!("hardware_profile_{}", "1".repeat(32))],
                            "capability_classes": ["gpu_fixture"],
                            "device_kinds": ["gpu"],
                            "apis": ["vulkan"],
                            "required_processes": [],
                            "required_features": []
                        },
                        "execution": {
                            "phases": ["decode", "prefill"],
                            "alternative_phases": ["decode", "prefill"],
                            "source_retained_phases": [],
                            "activation_batch": {
                                "minimum": 1,
                                "maximum": 65536
                            },
                            "context_activations": {
                                "minimum": 0,
                                "maximum": 65536
                            },
                            "state_activations": {
                                "minimum": 0,
                                "maximum": 65536
                            },
                            "speculative_draft_token_counts": [0],
                            "residency_policies": ["eager"]
                        },
                        "placement": {
                            "mode": "local",
                            "minimum_device_count": 1,
                            "maximum_device_count": 1,
                            "required_interconnects": []
                        }
                    }),
                )
                .unwrap(),
                representation: serde_json::json!({
                    "kind": "finite_state_recurrence"
                }),
                provenance: serde_json::json!({
                    "provider": {
                        "id": "fixture_provider",
                        "version": "1"
                    }
                }),
                benchmark_id: "benchmark_fixture".to_string(),
                validation_id: "validation_fixture".to_string(),
                validation_status: "passed".to_string(),
                decision_reason: "verified measured win".to_string(),
            },
        );
        modal.selected_implementation_id =
            Some("implementation_verified_fixture".to_string());
        modal.focus_row = modal.apply_row();
        app.overlay = Some(Overlay::Node(modal));
        let rendered = rendered_text(&mut app, 40, 12);
        assert!(rendered.contains("NODE"));
        assert!(rendered.contains("Apply"));
        assert!(rendered.contains("Cancel"));
        app.dispatch(crate::tui::AppAction::ToggleModuleAnatomy);
        let rendered = rendered_text(&mut app, 100, 40);
        assert!(rendered.contains("Editable source layer"));
        assert!(rendered.contains("fused_project"));
        assert!(rendered.contains("implementation_verified_fixture"));
        assert!(rendered.contains("finite_state_recurrence"));
        assert!(rendered.contains("fixture_provider"));
        assert!(rendered.contains("passed"));
    }

    #[test]
    fn modal_render_map_does_not_expose_workspace_controls_behind_the_overlay() {
        let mut app = App::new();
        let _ = rendered_text(&mut app, 80, 24);
        assert_eq!(app.action_at(79, 0), None);
    }
}
