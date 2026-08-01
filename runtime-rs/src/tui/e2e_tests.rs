use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use serde_json::Value;

use super::app::{ModelSelectorFocus, NodePropertyDraft, Overlay, SourceDiscovery};
use super::terminal::action_from_event;
use super::{App, AppAction, CompilerLaunch, FocusRegion};
use crate::test_support::TempDir;
use crate::{
    RuntimeEditorError, RuntimeModelEditor, StreamCircuitNodeInstanceStatePolicy,
    VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA,
};

struct TuiHarness {
    app: App,
    terminal: Terminal<TestBackend>,
    width: u16,
    height: u16,
}

impl TuiHarness {
    fn new() -> Self {
        let width = 120;
        let height = 40;
        Self {
            app: App::new(),
            terminal: Terminal::new(TestBackend::new(width, height)).unwrap(),
            width,
            height,
        }
    }

    fn with_editor_loader(mut self, editor: RuntimeModelEditor) -> Self {
        self.app.editor_loader = Arc::new(move |_| Ok(editor.clone()));
        self
    }

    fn render(&mut self) -> String {
        self.terminal
            .draw(|frame| super::view::render(frame, &mut self.app))
            .unwrap();
        let buffer = self.terminal.backend().buffer();
        (0..self.height)
            .map(|row| {
                (0..self.width)
                    .map(|column| buffer[(column, row)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn resize(&mut self, width: u16, height: u16) {
        self.terminal.backend_mut().resize(width, height);
        self.terminal
            .resize(Rect::new(0, 0, width, height))
            .unwrap();
        self.width = width;
        self.height = height;
    }

    fn event(&mut self, event: Event) -> Option<AppAction> {
        let action = action_from_event(&self.app, event);
        if let Some(action) = action.clone() {
            self.app.dispatch(action);
        }
        action
    }

    fn key(&mut self, code: KeyCode, modifiers: KeyModifiers) -> Option<AppAction> {
        self.event(Event::Key(KeyEvent::new(code, modifiers)))
    }

    fn paste(&mut self, value: impl Into<String>) -> Option<AppAction> {
        self.event(Event::Paste(value.into()))
    }

    fn set_path(&mut self, path: &Path) {
        self.key(KeyCode::Char('a'), KeyModifiers::CONTROL);
        self.paste(path.display().to_string());
    }

    fn text_position(&mut self, needle: &str) -> Option<(u16, u16)> {
        self.render();
        let buffer = self.terminal.backend().buffer();
        for row in 0..self.height {
            for start in 0..self.width {
                let mut candidate = String::new();
                for column in start..self.width {
                    candidate.push_str(buffer[(column, row)].symbol());
                    if candidate == needle {
                        return Some((start, row));
                    }
                    if !needle.starts_with(&candidate) {
                        break;
                    }
                }
            }
        }
        None
    }

    fn mouse_action_on_text(&mut self, needle: &str) -> Option<AppAction> {
        let (column, row) = self
            .text_position(needle)
            .unwrap_or_else(|| panic!("rendered TUI did not contain {needle:?}"));
        action_from_event(
            &self.app,
            Event::Mouse(MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column,
                row,
                modifiers: KeyModifiers::NONE,
            }),
        )
    }

    fn click_text(&mut self, needle: &str) -> Option<AppAction> {
        let action = self.mouse_action_on_text(needle);
        if let Some(action) = action.clone() {
            self.app.dispatch(action);
        }
        action
    }

    fn wait_for(&mut self, condition: impl Fn(&App) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition(&self.app) {
            self.app.poll_compiler();
            assert!(Instant::now() < deadline, "TUI workflow timed out");
            thread::sleep(Duration::from_millis(5));
        }
        while self.app.compiler_job.is_some() {
            self.app.poll_compiler();
            assert!(Instant::now() < deadline, "compiler process did not exit");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn wait_until(&mut self, condition: impl Fn(&App) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while !condition(&self.app) {
            self.app.poll_compiler();
            assert!(Instant::now() < deadline, "TUI workflow timed out");
            thread::sleep(Duration::from_millis(5));
        }
    }
}

fn deterministic_editor_with_two_devices() -> RuntimeModelEditor {
    let package = crate::test_support::tiny_model_dir();
    let initial = crate::editor::load_runtime_model_editor_without_hardware(&package).unwrap();
    let first = initial.available_devices()[0].clone();
    let mut second = first.clone();
    second.device_id = "gpu1".to_string();
    second.runtime_device_id = Some("gpu1".to_string());
    second.physical_device_id = Some("test:1".to_string());
    second.physical_device_index = Some(1);
    second.device_name = Some("Second deterministic device".to_string());
    RuntimeModelEditor::load_with_available_devices(&package, vec![first, second]).unwrap()
}

fn deterministic_editor_with_incompatible_second_device() -> RuntimeModelEditor {
    let package = crate::test_support::tiny_model_dir();
    let initial = crate::editor::load_runtime_model_editor_without_hardware(&package).unwrap();
    let first = initial.available_devices()[0].clone();
    let mut second = first.clone();
    second.device_id = "limited-gpu".to_string();
    second.runtime_device_id = Some("limited-gpu".to_string());
    second.physical_device_id = Some("test:limited".to_string());
    second.physical_device_index = Some(1);
    second.device_name = Some("Limited deterministic GPU".to_string());
    let target: Value =
        serde_json::from_slice(&fs::read(package.join("compiler_target.json")).unwrap()).unwrap();
    second.hardware_profile =
        Some(serde_json::from_value(target["hardware_profiles"][0].clone()).unwrap());
    RuntimeModelEditor::load_with_available_devices(&package, vec![first, second]).unwrap()
}

fn write_package_header(path: &Path, schema: &str, package_id: &str) {
    fs::create_dir_all(path).unwrap();
    fs::write(
        path.join(crate::RUNTIME_PACKAGE_MANIFEST_FILE),
        serde_json::to_vec(&serde_json::json!({
            "schema": schema,
            "package_id": package_id,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_safetensors_source(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::write(
        path.join("config.json"),
        r#"{"model_type":"e2e_fixture","architectures":["FixtureCircuit"]}"#,
    )
    .unwrap();
    fs::write(path.join("model.safetensors"), b"fixture").unwrap();
    fs::write(path.join("tokenizer.json"), b"{}").unwrap();
    fs::write(
        path.join("tokenizer_config.json"),
        r#"{"chat_template":"{{ messages }}"}"#,
    )
    .unwrap();
}

fn fake_compiler_launch(package_manifest: &Path, working_directory: &Path) -> CompilerLaunch {
    let package = serde_json::to_string(&package_manifest.display().to_string()).unwrap();
    let script = r#"
import json, sys
sequence = 0
def emit(kind, **payload):
    global sequence
    print(json.dumps({"schema":"nerve.compiler_event.v1","sequence":sequence,"type":kind,**payload}), flush=True)
    sequence += 1
source = {
    "model_type":"e2e_fixture",
    "architectures":["FixtureCircuit"],
    "weight_files":["model.safetensors"],
    "tokenizer_files":["tokenizer.json", "tokenizer_config.json"],
    "has_chat_template":True,
}
if "--discover-model" in sys.argv:
    emit("DiscoveryStarted")
    emit("SourceDiscovered", source=source)
    emit("Completed", discovery=source)
elif "--compile-model" in sys.argv:
    emit("ValidationStarted")
    emit("ComponentLoweringStarted", current=1, total=1, component_id="layer_00")
    emit("PackageValidationStarted")
    emit("Completed", package={"package_manifest":__PACKAGE__})
else:
    emit("Failed", diagnostics=[{"message":"unexpected compiler mode"}])
    raise SystemExit(2)
"#
    .replace("__PACKAGE__", &package);
    CompilerLaunch::new(
        "python3",
        [
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ],
        working_directory,
    )
}

fn cancelling_compiler_launch(working_directory: &Path) -> CompilerLaunch {
    let script = r#"
import json, signal, sys, time
sequence = 0
def emit(kind, **payload):
    global sequence
    print(json.dumps({"schema":"nerve.compiler_event.v1","sequence":sequence,"type":kind,**payload}), flush=True)
    sequence += 1
def cancel(_signal, _frame):
    emit("Cancelled")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, cancel)
emit("DiscoveryStarted")
while True:
    time.sleep(0.05)
"#;
    CompilerLaunch::new(
        "python3",
        [
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ],
        working_directory,
    )
}

fn uncooperative_compiler_launch(working_directory: &Path) -> CompilerLaunch {
    let script = r#"
import json, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(json.dumps({"schema":"nerve.compiler_event.v1","sequence":0,"type":"DiscoveryStarted"}), flush=True)
while True:
    time.sleep(0.05)
"#;
    CompilerLaunch::new(
        "python3",
        [
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ],
        working_directory,
    )
    .with_cancel_grace_period(Duration::from_millis(50))
}

fn scripted_compiler_launch(script: &str, working_directory: &Path) -> CompilerLaunch {
    CompilerLaunch::new(
        "python3",
        [
            OsString::from("-u"),
            OsString::from("-c"),
            OsString::from(script),
        ],
        working_directory,
    )
}

fn mark_source_as_discovered(harness: &mut TuiHarness) {
    let Some(Overlay::ModelSelector(selector)) = &mut harness.app.overlay else {
        panic!("model selector is not open");
    };
    selector.discovery = Some(SourceDiscovery {
        model_type: "e2e_fixture".to_string(),
        architecture: vec!["FixtureCircuit".to_string()],
        weight_files: vec!["model.safetensors".to_string()],
        tokenizer_files: vec!["tokenizer.json".to_string()],
        has_chat_template: true,
        raw: serde_json::json!({"model_type":"e2e_fixture"}),
    });
    selector.focus = ModelSelectorFocus::Action;
}

#[test]
fn e2e_stale_and_failed_packages_are_not_actionable() {
    let root = TempDir::new("tui-e2e-invalid-packages");
    let stale = root.path().join("stale");
    let missing_identity = root.path().join("missing-identity");
    let malformed = root.path().join("malformed");
    let broken = root.path().join("broken");
    write_package_header(&stale, "nerve.vulkan_resident_model_package.v4", "stale");
    fs::create_dir_all(&missing_identity).unwrap();
    fs::write(
        missing_identity.join(crate::RUNTIME_PACKAGE_MANIFEST_FILE),
        serde_json::to_vec(&serde_json::json!({
            "schema": VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA,
        }))
        .unwrap(),
    )
    .unwrap();
    fs::create_dir_all(&malformed).unwrap();
    fs::write(
        malformed.join(crate::RUNTIME_PACKAGE_MANIFEST_FILE),
        b"{this is not JSON",
    )
    .unwrap();
    write_package_header(
        &broken,
        VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA,
        "broken",
    );
    let load_count = Arc::new(AtomicUsize::new(0));
    let observed = load_count.clone();
    let mut harness = TuiHarness::new();
    harness.app.editor_loader = Arc::new(move |_| {
        observed.fetch_add(1, Ordering::SeqCst);
        Err(RuntimeEditorError(
            "synthetic package validation failure".to_string(),
        ))
    });

    harness.set_path(&stale);
    let rendered = harness.render();
    assert!(rendered.contains("recompile the model"));
    assert!(rendered.contains("Unavailable"));
    assert_eq!(harness.mouse_action_on_text("Unavailable"), None);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(load_count.load(Ordering::SeqCst), 0);

    harness.set_path(&missing_identity);
    assert!(harness.render().contains("has no package_id"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(load_count.load(Ordering::SeqCst), 0);

    harness.set_path(&malformed);
    assert!(harness.render().contains("is not valid JSON"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(load_count.load(Ordering::SeqCst), 0);

    harness.set_path(&broken);
    assert!(harness.render().contains("Load model"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(load_count.load(Ordering::SeqCst), 1);
    let rendered = harness.render();
    assert!(rendered.contains("synthetic package validation failure"));
    assert!(rendered.contains("Unavailable"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert_eq!(load_count.load(Ordering::SeqCst), 1);
}

#[test]
fn e2e_failed_model_replacement_preserves_the_loaded_graph() {
    let root = TempDir::new("tui-e2e-failed-replacement");
    let broken = root.path().join("broken");
    write_package_header(
        &broken,
        VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA,
        "broken",
    );
    let mut harness = TuiHarness::new();
    harness
        .app
        .install_editor(deterministic_editor_with_two_devices());
    harness.key(KeyCode::Char('d'), KeyModifiers::CONTROL);
    let original_ids = harness
        .app
        .instances()
        .iter()
        .map(|instance| instance.instance_id.clone())
        .collect::<Vec<_>>();
    let original_package = harness
        .app
        .editor
        .as_ref()
        .unwrap()
        .package_id()
        .to_string();
    harness.app.editor_loader = Arc::new(|_| {
        Err(RuntimeEditorError(
            "replacement package failed integrity validation".to_string(),
        ))
    });

    harness.key(KeyCode::Char('o'), KeyModifiers::CONTROL);
    harness.set_path(&broken);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::ModelSelector(_))
    ));
    assert!(harness.render().contains("failed integrity validation"));
    harness.key(KeyCode::Esc, KeyModifiers::NONE);

    assert!(harness.app.overlay.is_none());
    assert_eq!(
        harness.app.editor.as_ref().unwrap().package_id(),
        original_package
    );
    assert_eq!(
        harness
            .app
            .instances()
            .iter()
            .map(|instance| instance.instance_id.clone())
            .collect::<Vec<_>>(),
        original_ids
    );
}

#[test]
fn e2e_package_graph_sequence_mouse_and_node_modal_workflow() {
    let editor = deterministic_editor_with_two_devices();
    let package = crate::test_support::tiny_model_dir();
    let mut harness = TuiHarness::new().with_editor_loader(editor);

    harness.set_path(&package);
    assert_eq!(
        harness.click_text("Load model"),
        Some(AppAction::ModelAction)
    );
    assert!(harness.app.overlay.is_none());
    assert_eq!(harness.app.focus(), FocusRegion::Graph);
    assert_eq!(harness.app.selected_instance(), Some("layer_00"));
    assert_eq!(harness.app.instances().len(), 1);
    assert!(harness.render().contains("layer_00"));

    harness.key(KeyCode::Char('d'), KeyModifiers::CONTROL);
    let duplicated = harness.app.instances();
    assert_eq!(duplicated.len(), 2);
    assert_eq!(duplicated[0].instance_id, "layer_00");
    assert_eq!(duplicated[1].instance_id, "layer_00@2");
    assert_eq!(harness.app.selected_instance(), Some("layer_00@2"));

    harness.key(KeyCode::Left, KeyModifiers::ALT);
    let reordered = harness.app.instances();
    assert_eq!(reordered[0].instance_id, "layer_00@2");
    assert_eq!(reordered[1].instance_id, "layer_00");
    assert_eq!(harness.app.sequence.text(), "[0,0]");

    harness.key(KeyCode::Delete, KeyModifiers::NONE);
    assert_eq!(
        harness
            .app
            .instances()
            .iter()
            .map(|instance| instance.instance_id.as_str())
            .collect::<Vec<_>>(),
        ["layer_00"]
    );
    harness.key(KeyCode::Char('d'), KeyModifiers::CONTROL);

    harness.key(KeyCode::Tab, KeyModifiers::NONE);
    assert_eq!(harness.app.focus(), FocusRegion::Sequence);
    harness.key(KeyCode::Char('a'), KeyModifiers::CONTROL);
    harness.paste("[0,");
    assert!(harness.app.sequence_error.is_some());
    assert_eq!(
        harness.app.editor.as_ref().unwrap().layer_instances().len(),
        2
    );
    assert!(harness.render().contains("! Expected"));
    harness.paste("0]");
    assert!(harness.app.sequence_error.is_none());
    assert_eq!(harness.app.sequence.text(), "[0,0]");
    harness.key(KeyCode::Tab, KeyModifiers::NONE);

    harness.key(KeyCode::Home, KeyModifiers::NONE);
    assert_eq!(
        harness.click_text("layer_00"),
        Some(AppAction::OpenNode("layer_00".to_string()))
    );
    assert!(matches!(harness.app.overlay, Some(Overlay::Node(_))));
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    for _ in 0..4 {
        harness.key(KeyCode::Down, KeyModifiers::NONE);
    }
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(harness.app.overlay.is_none());
    assert_eq!(
        harness
            .app
            .editor
            .as_ref()
            .unwrap()
            .layer_instances()
            .iter()
            .find(|instance| instance.instance_id == "layer_00")
            .unwrap()
            .device_id,
        "gpu1"
    );

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    assert_eq!(
        harness
            .app
            .editor
            .as_ref()
            .unwrap()
            .layer_instances()
            .iter()
            .find(|instance| instance.instance_id == "layer_00")
            .unwrap()
            .device_id,
        "gpu1"
    );

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    for _ in 0..3 {
        harness.key(KeyCode::Down, KeyModifiers::NONE);
    }
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(
        !harness
            .app
            .editor
            .as_ref()
            .unwrap()
            .layer_instances()
            .iter()
            .find(|instance| instance.instance_id == "layer_00")
            .unwrap()
            .enabled
    );
    assert!(harness.render().contains("BYPASS"));
}

#[test]
fn e2e_browser_discovery_transpilation_and_fast_exit_delivery() {
    let root = TempDir::new("tui-e2e-compiler");
    let source = root.path().join("source");
    write_safetensors_source(&source);
    let package = crate::test_support::tiny_model_dir();
    let editor = deterministic_editor_with_two_devices();
    let mut harness = TuiHarness::new().with_editor_loader(editor);
    harness.app.compiler_launch = Ok(fake_compiler_launch(
        &package.join(crate::RUNTIME_PACKAGE_MANIFEST_FILE),
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    ));

    harness.set_path(root.path());
    harness.key(KeyCode::Tab, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::ModelSelector(ref selector))
            if selector.focus == ModelSelectorFocus::Browser
    ));
    assert_eq!(
        harness.click_text("source/"),
        Some(AppAction::ModelBrowserOpen(1))
    );
    assert!(harness.render().contains("Inspect source"));
    harness.key(KeyCode::Tab, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Compiler(_))));
    assert!(harness.render().contains("DISCOVERING MODEL"));
    harness.wait_for(|app| {
        matches!(
            app.overlay,
            Some(Overlay::ModelSelector(ref selector)) if selector.discovery.is_some()
        )
    });
    let rendered = harness.render();
    assert!(rendered.contains("e2e_fixture"));
    assert!(rendered.contains("FixtureCircuit"));
    assert!(rendered.contains("Transpile and load"));

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Compiler(_))));
    assert!(harness.render().contains("TRANSPILING MODEL"));
    harness.wait_for(|app| app.editor.is_some() && app.overlay.is_none());
    assert_eq!(
        harness.app.editor.as_ref().unwrap().package_id(),
        "model_d119caf1_vulkan_resident"
    );
    assert_eq!(harness.app.selected_instance(), Some("layer_00"));
}

#[test]
fn e2e_help_does_not_hide_or_lose_background_compiler_completion() {
    let root = TempDir::new("tui-e2e-compiler-help");
    let source = root.path().join("source");
    write_safetensors_source(&source);
    let package = crate::test_support::tiny_model_dir();
    let editor = deterministic_editor_with_two_devices();
    let mut harness = TuiHarness::new().with_editor_loader(editor);
    harness.app.compiler_launch = Ok(fake_compiler_launch(
        &package.join(crate::RUNTIME_PACKAGE_MANIFEST_FILE),
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    ));

    harness.set_path(&source);
    mark_source_as_discovered(&mut harness);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Compiler(_))));
    harness.key(KeyCode::F(1), KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Help)));
    assert!(harness.render().contains("MODEL COMPILER"));

    harness.wait_for(|app| app.editor.is_some());
    assert!(matches!(harness.app.overlay, Some(Overlay::Help)));
    assert!(harness.render().contains("EXECUTION GRAPH"));
    harness.key(KeyCode::F(1), KeyModifiers::NONE);
    assert!(harness.app.overlay.is_none());
    assert_eq!(harness.app.selected_instance(), Some("layer_00"));
    assert!(harness.app.status.contains("Loaded"));
}

#[test]
fn e2e_compiler_cancellation_returns_to_the_preserved_source() {
    let root = TempDir::new("tui-e2e-cancel");
    let source = root.path().join("source");
    write_safetensors_source(&source);
    let mut harness = TuiHarness::new();
    harness.app.compiler_launch = Ok(cancelling_compiler_launch(
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    ));

    harness.set_path(&source);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.wait_until(|app| {
        matches!(app.overlay, Some(Overlay::Compiler(ref progress)) if !progress.events.is_empty())
    });
    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Compiler(ref progress)) if progress.cancelling
    ));
    harness.wait_for(|app| matches!(app.overlay, Some(Overlay::ModelSelector(_))));
    assert!(harness.app.status.contains("cancelled"));
    assert!(harness.app.editor.is_none());
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::ModelSelector(ref selector)) if selector.selected_path() == source
    ));
}

#[test]
fn e2e_uncooperative_compiler_is_force_stopped_without_publishing() {
    let root = TempDir::new("tui-e2e-forced-cancel");
    let source = root.path().join("source");
    write_safetensors_source(&source);
    let mut harness = TuiHarness::new();
    harness.app.compiler_launch = Ok(uncooperative_compiler_launch(
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    ));

    harness.set_path(&source);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.wait_until(|app| {
        matches!(app.overlay, Some(Overlay::Compiler(ref progress)) if !progress.events.is_empty())
    });
    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    harness.wait_for(|app| matches!(app.overlay, Some(Overlay::ModelSelector(_))));
    let Some(Overlay::ModelSelector(selector)) = &harness.app.overlay else {
        panic!("forced cancellation did not return to model selection");
    };
    assert_eq!(selector.selected_path(), source);
    assert!(
        selector
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.contains("force-stopped"))
    );
    assert!(harness.app.status.contains("cancelled"));
    assert!(harness.app.editor.is_none());
}

#[test]
fn e2e_incomplete_source_is_diagnosed_by_the_compiler_and_remains_retryable() {
    let root = TempDir::new("tui-e2e-incomplete-source");
    let source = root.path().join("source");
    fs::create_dir_all(&source).unwrap();
    fs::write(
        source.join("config.json"),
        r#"{"model_type":"incomplete_fixture"}"#,
    )
    .unwrap();
    fs::write(source.join("model.safetensors"), b"fixture").unwrap();
    let script = r#"import json
print(json.dumps({"schema":"nerve.compiler_event.v1","sequence":0,"type":"Failed","diagnostics":[{"message":"tokenizer.json is missing"}]}), flush=True)
raise SystemExit(1)"#;
    let mut harness = TuiHarness::new();
    harness.app.compiler_launch = Ok(scripted_compiler_launch(
        script,
        Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap(),
    ));

    harness.set_path(&source);
    assert!(harness.render().contains("Inspect source"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.wait_for(|app| matches!(app.overlay, Some(Overlay::ModelSelector(_))));
    let Some(Overlay::ModelSelector(selector)) = &harness.app.overlay else {
        panic!("compiler failure did not return to source selection");
    };
    assert_eq!(selector.selected_path(), source);
    assert!(
        selector
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.contains("tokenizer.json is missing"))
    );
    assert_eq!(selector.current_action_label(), "Inspect source");
    assert!(harness.app.editor.is_none());
}

#[test]
fn e2e_compiler_protocol_violations_and_crashes_cannot_publish() {
    let root = TempDir::new("tui-e2e-bad-compiler");
    let source = root.path().join("source");
    write_safetensors_source(&source);
    let package = crate::test_support::tiny_model_dir().join(crate::RUNTIME_PACKAGE_MANIFEST_FILE);
    let package_json = serde_json::to_string(&package.display().to_string()).unwrap();
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let cases = [
        (
            format!(
                r#"import json,sys
print(json.dumps({{"schema":"nerve.compiler_event.v1","sequence":0,"type":"Completed","package":{{"package_manifest":{package_json}}}}}), flush=True)
raise SystemExit(7)"#
            ),
            "reported completion but exited",
        ),
        (
            format!(
                r#"import json
print("not-json", flush=True)
print(json.dumps({{"schema":"nerve.compiler_event.v1","sequence":0,"type":"Completed","package":{{"package_manifest":{package_json}}}}}), flush=True)"#
            ),
            "Compiler protocol failed",
        ),
        (
            "raise SystemExit(0)".to_string(),
            "without a terminal structured event",
        ),
        (
            r#"import json
print(json.dumps({"schema":"nerve.compiler_event.v1","sequence":0,"type":"Completed","package":{}}), flush=True)"#
                .to_string(),
            "without a package_manifest",
        ),
        (
            format!(
                r#"import json
print(json.dumps({{"schema":"nerve.compiler_event.v1","sequence":4,"type":"Completed","package":{{"package_manifest":{package_json}}}}}), flush=True)"#
            ),
            "stream started at sequence 4",
        ),
    ];

    for (index, (script, expected)) in cases.into_iter().enumerate() {
        let load_count = Arc::new(AtomicUsize::new(0));
        let observed = load_count.clone();
        let mut harness = TuiHarness::new();
        harness.app.editor_loader = Arc::new(move |_| {
            observed.fetch_add(1, Ordering::SeqCst);
            Err(RuntimeEditorError("loader must not run".to_string()))
        });
        harness.app.compiler_launch = Ok(scripted_compiler_launch(&script, workspace));
        harness.set_path(&source);
        mark_source_as_discovered(&mut harness);
        harness.key(KeyCode::Enter, KeyModifiers::NONE);
        harness.wait_for(|app| matches!(app.overlay, Some(Overlay::ModelSelector(_))));
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            0,
            "bad compiler case {index} invoked the model loader"
        );
        let Some(Overlay::ModelSelector(selector)) = &harness.app.overlay else {
            panic!("bad compiler case {index} did not return to model selection");
        };
        assert!(
            selector
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.contains(expected)),
            "bad compiler case {index} did not report {expected:?}: {:?}",
            selector.diagnostics
        );
        assert!(harness.app.editor.is_none());
    }
}

#[test]
fn e2e_unavailable_device_and_invalid_property_cannot_mutate_the_draft() {
    let package = crate::test_support::tiny_model_dir();
    let initial = crate::editor::load_runtime_model_editor_without_hardware(&package).unwrap();
    let mut unavailable = initial.available_devices()[0].clone();
    unavailable.available = false;
    unavailable.can_host_runtime_components_on_physical_device = Some(false);
    unavailable.error = Some("device was removed".to_string());
    let editor =
        RuntimeModelEditor::load_with_available_devices(&package, vec![unavailable]).unwrap();
    let mut harness = TuiHarness::new();
    harness.app.install_editor(editor);
    assert!(harness.render().contains("! UNAVAIL"));

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    for _ in 0..4 {
        harness.key(KeyCode::Down, KeyModifiers::NONE);
    }
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal))
            if modal.error.as_deref().is_some_and(|error| error.contains("unavailable"))
    ));
    harness.key(KeyCode::Esc, KeyModifiers::NONE);

    let editor = deterministic_editor_with_two_devices();
    harness.app.install_editor(editor);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    let schema = crate::runtime_editor_control_schema(
        0,
        &serde_json::json!({
            "id":"window",
            "name":"Window",
            "type":"integer",
            "current":4,
            "min":2,
            "max":10,
            "step":2,
            "editable_at_runtime":true,
            "scope":"instance"
        }),
    );
    let Some(Overlay::Node(modal)) = &mut harness.app.overlay else {
        panic!("node modal did not open");
    };
    modal
        .properties
        .push(NodePropertyDraft::new(schema, serde_json::json!(4)));
    modal.focus_row = 4;
    harness.key(KeyCode::Char('a'), KeyModifiers::CONTROL);
    harness.paste("5");
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal))
            if modal.error.as_deref().is_some_and(|error| error.contains("step"))
    ));
    assert!(
        harness.app.editor.as_ref().unwrap().layer_instances()[0]
            .control_values
            .is_empty()
    );
}

#[test]
fn e2e_component_incompatible_device_is_explained_and_cannot_be_selected() {
    let mut editor = deterministic_editor_with_incompatible_second_device();
    let error = editor
        .validate_instance_device_compatibility("layer_00", "limited-gpu")
        .unwrap_err();
    assert!(
        error.to_string().contains("unsupported")
            || error.to_string().contains("no compatible prefill")
    );
    editor
        .set_instance_device("layer_00", "limited-gpu")
        .unwrap_err();
    assert_ne!(editor.layer_instances()[0].device_id, "limited-gpu");

    let mut harness = TuiHarness::new();
    harness.app.install_editor(editor);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    let rendered = harness.render();
    assert!(rendered.contains("INCOMPATIBLE"));
    let Some(Overlay::Node(modal)) = &harness.app.overlay else {
        panic!("node modal did not open");
    };
    assert_eq!(modal.device_selectable, [true, false]);
    assert_eq!(modal.device_index, 0);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal)) if modal.device_index == 0
    ));
}

#[test]
fn e2e_state_policy_sources_are_compatible_and_failed_edits_are_transactional() {
    let editor = deterministic_editor_with_two_devices();
    let mut harness = TuiHarness::new();
    harness.app.install_editor(editor);

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    let Some(Overlay::Node(modal)) = &harness.app.overlay else {
        panic!("node modal did not open");
    };
    assert!(
        modal.policy_targets.is_empty(),
        "stateless system components must not be offered as state sources"
    );
    harness.key(KeyCode::Esc, KeyModifiers::NONE);

    harness.key(KeyCode::Char('d'), KeyModifiers::CONTROL);
    assert_eq!(harness.app.selected_instance(), Some("layer_00@2"));
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    let Some(Overlay::Node(modal)) = &harness.app.overlay else {
        panic!("duplicated node modal did not open");
    };
    assert_eq!(modal.policy_targets, ["layer_00"]);

    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(harness.app.overlay.is_none());
    assert!(matches!(
        harness
            .app
            .editor
            .as_ref()
            .unwrap()
            .layer_instances()[1]
            .state_policy,
        StreamCircuitNodeInstanceStatePolicy::CloneFrom { ref instance_id }
            if instance_id == "layer_00"
    ));

    // A share cannot silently cross device boundaries. The failed compound
    // edit must preserve both the old device and the old clone policy.
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal))
            if modal.error.as_deref().is_some_and(|error| error.contains("cannot share state across devices"))
    ));
    let duplicate = &harness.app.editor.as_ref().unwrap().layer_instances()[1];
    assert_ne!(duplicate.device_id, "gpu1");
    assert!(matches!(
        duplicate.state_policy,
        StreamCircuitNodeInstanceStatePolicy::CloneFrom { ref instance_id }
            if instance_id == "layer_00"
    ));

    // A node serving as another instance's state source cannot be disabled.
    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    harness.key(KeyCode::Home, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    for _ in 0..3 {
        harness.key(KeyCode::Down, KeyModifiers::NONE);
    }
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal))
            if modal.error.as_deref().is_some_and(|error| error.contains("must both be enabled"))
    ));
    assert!(harness.app.editor.as_ref().unwrap().layer_instances()[0].enabled);

    // The reverse dependency would form a cycle and must also roll back.
    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Right, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Down, KeyModifiers::NONE);
    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::Node(ref modal))
            if modal.error.as_deref().is_some_and(|error| error.contains("dependency cycle"))
    ));
    assert!(matches!(
        harness.app.editor.as_ref().unwrap().layer_instances()[0].state_policy,
        StreamCircuitNodeInstanceStatePolicy::Fresh
    ));

    harness.key(KeyCode::Esc, KeyModifiers::NONE);
    harness.key(KeyCode::Delete, KeyModifiers::NONE);
    assert_eq!(harness.app.instances().len(), 2);
    assert!(harness.app.status.contains("unknown instance"));
}

#[test]
fn e2e_resize_rebuilds_hit_targets_without_losing_graph_selection() {
    let editor = deterministic_editor_with_two_devices();
    let mut harness = TuiHarness::new();
    harness.app.install_editor(editor);
    assert!(harness.render().contains("layer_00"));
    let selection = harness.app.selected_instance().map(str::to_string);

    harness.resize(40, 12);
    assert_eq!(harness.event(Event::Resize(40, 12)), None);
    let rendered = harness.render();
    assert!(rendered.contains("STREAM GRAPH"));
    assert!(rendered.contains("EXECUTION GRAPH"));
    assert_eq!(harness.app.selected_instance(), selection.as_deref());

    harness.key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Node(_))));
    let rendered = harness.render();
    assert!(rendered.contains("Apply"));
    assert!(rendered.contains("Cancel"));
}

#[test]
fn e2e_global_help_and_mouse_controls_preserve_the_active_workflow() {
    let mut harness = TuiHarness::new();
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::ModelSelector(_))
    ));
    let mouse_capture = harness.app.mouse_capture();

    harness.key(KeyCode::F(1), KeyModifiers::NONE);
    assert!(matches!(harness.app.overlay, Some(Overlay::Help)));
    let rendered = harness.render();
    assert!(rendered.contains("MODEL SELECTION"));
    assert!(rendered.contains("edit source path"));
    harness.key(KeyCode::Char('m'), KeyModifiers::CONTROL);
    assert_eq!(harness.app.mouse_capture(), !mouse_capture);
    harness.key(KeyCode::F(1), KeyModifiers::NONE);
    assert!(matches!(
        harness.app.overlay,
        Some(Overlay::ModelSelector(_))
    ));

    harness
        .app
        .install_editor(deterministic_editor_with_two_devices());
    harness.key(KeyCode::F(1), KeyModifiers::NONE);
    assert!(harness.render().contains("EXECUTION GRAPH"));
    harness.key(KeyCode::F(1), KeyModifiers::NONE);
}
