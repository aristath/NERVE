from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.automation.command import (
    OptimizePackageOutcome,
    optimize_compiled_package,
    resolve_package_manifest,
)


def test_optimizer_command_wires_fresh_default_paths_and_unbounded_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "compiled"
    package.mkdir()
    manifest = package / "vulkan_resident_package.json"
    manifest.write_text("{}")
    captured: dict[str, object] = {}
    prepared = SimpleNamespace(targets=("target",))
    optimization = SimpleNamespace(report={"status": "completed"})

    def prepare(**kwargs):
        captured["prepare"] = kwargs
        return prepared

    def run(**kwargs):
        captured["run"] = kwargs
        return optimization

    monkeypatch.setattr(
        "nerve.representation_optimizer.automation.command."
        "prepare_runtime_optimization_targets",
        prepare,
    )
    monkeypatch.setattr(
        "nerve.representation_optimizer.automation.command.run_automated_optimizer",
        run,
    )

    outcome = optimize_compiled_package(
        package,
        selected_device_ids=("vulkan-uuid:idle",),
    )

    assert outcome.optimization is optimization
    assert captured["prepare"]["package_manifest"] == manifest
    assert captured["prepare"]["run_root"] == (tmp_path / ".compiled-optimizer-run")
    assert captured["run"]["package_dir"] == package
    assert captured["run"]["output_package_dir"] == (tmp_path / "compiled-optimized")
    assert captured["run"]["targets"] == ("target",)
    assert all(value is None for value in captured["run"]["budget"].to_json().values())


def test_optimizer_command_requires_package_manifest_identity(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "manifest.json"
    wrong.write_text("{}")

    with pytest.raises(ModelCompileError, match="package path is invalid"):
        resolve_package_manifest(wrong)


def test_optimizer_json_response_indexes_persisted_report_without_replaying_it(
    tmp_path: Path,
) -> None:
    report = {
        "status": "completed_no_changes",
        "report_id": "report-id",
        "summary": {"scope_count": 2_904},
        "publication": {"status": "not_required"},
        "event_journal": {
            "artifact_ref": "events.jsonl",
            "event_count": 5_866,
        },
        "scopes": [{"claims": ["large duplicated evidence"]}],
        "candidates": [{"details": "large candidate evidence"}],
    }
    optimization = SimpleNamespace(
        report=report,
        report_path=tmp_path / "run" / "report.json",
        output_package_dir=tmp_path / "compiled",
    )
    targets = SimpleNamespace(
        to_json=lambda: {"target_ids": ["amd-target"]},
    )

    document = OptimizePackageOutcome(
        optimization=optimization,
        targets=targets,
    ).to_json()

    assert document["optimization"]["report_path"] == str(
        tmp_path / "run" / "report.json"
    )
    assert document["optimization"]["summary"] == report["summary"]
    assert "scopes" not in document["optimization"]
    assert "candidates" not in document["optimization"]
    assert len(__import__("json").dumps(document)) < 1_000


def test_builtin_provider_imports_in_a_clean_interpreter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from nerve.representation_optimizer.providers.builtin "
                "import load_builtin_provider_registry; "
                "load_builtin_provider_registry()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
