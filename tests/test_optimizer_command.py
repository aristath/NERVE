from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.automation.command import (
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
        "nerve.representation_optimizer.automation.command."
        "run_automated_optimizer",
        run,
    )

    outcome = optimize_compiled_package(
        package,
        selected_device_ids=("vulkan-uuid:idle",),
    )

    assert outcome.optimization is optimization
    assert captured["prepare"]["package_manifest"] == manifest
    assert captured["prepare"]["run_root"] == (
        tmp_path / ".compiled-optimizer-run"
    )
    assert captured["run"]["package_dir"] == package
    assert captured["run"]["output_package_dir"] == (
        tmp_path / "compiled-optimized"
    )
    assert captured["run"]["targets"] == ("target",)
    assert all(
        value is None
        for value in captured["run"]["budget"].to_json().values()
    )


def test_optimizer_command_requires_package_manifest_identity(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "manifest.json"
    wrong.write_text("{}")

    with pytest.raises(ModelCompileError, match="package path is invalid"):
        resolve_package_manifest(wrong)
