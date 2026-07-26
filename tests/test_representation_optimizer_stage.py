from __future__ import annotations

from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError, read_json
from nerve.representation_optimizer.stage import (
    OPTIMIZER_CONTRACT_SCHEMAS,
    initialize_optimizer_stage,
    load_optimizer_stage,
)


def write_exact_baseline(package_dir: Path) -> tuple[dict[str, object], Path]:
    baseline = {
        "schema": "nerve.lowered_execution_graph.v1",
        "graph": {"topology": "explicit_graph", "circuits": [], "edges": []},
    }
    path = package_dir / "lowered" / "execution_graph.circuits.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"schema":"nerve.lowered_execution_graph.v1",'
        '"graph":{"topology":"explicit_graph","circuits":[],"edges":[]}}\n'
    )
    return baseline, path


def test_optimizer_stage_is_between_semantic_lowering_and_publication(
    tmp_path: Path,
) -> None:
    baseline, baseline_path = write_exact_baseline(tmp_path)

    artifact = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=tmp_path,
        lowered_index=baseline,
        lowered_index_path=baseline_path,
    )

    document = load_optimizer_stage(artifact.path, package_dir=tmp_path)
    assert document["compiler_position"] == {
        "after": "exact_semantic_lowering",
        "before": "physical_package_publication",
    }
    assert document["status"] == "exact_baseline_retained"
    assert document["exact_baseline"]["mutable"] is False
    assert document["session"]["candidates"] == []
    assert document["contract_schemas"] == list(OPTIMIZER_CONTRACT_SCHEMAS)
    assert artifact.package_reference(tmp_path) == "optimization/stage.json"


def test_optimizer_stage_serialization_is_deterministic(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_baseline, first_path = write_exact_baseline(first_root)
    second_baseline, second_path = write_exact_baseline(second_root)

    first = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=first_root,
        lowered_index=first_baseline,
        lowered_index_path=first_path,
    )
    second = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=second_root,
        lowered_index=second_baseline,
        lowered_index_path=second_path,
    )

    assert first.path.read_bytes() == second.path.read_bytes()


def test_optimizer_stage_detects_exact_baseline_mutation(tmp_path: Path) -> None:
    baseline, baseline_path = write_exact_baseline(tmp_path)
    artifact = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=tmp_path,
        lowered_index=baseline,
        lowered_index_path=baseline_path,
    )
    baseline_path.write_text('{"mutated":true}\n')

    with pytest.raises(ModelCompileError, match="digest does not match"):
        load_optimizer_stage(artifact.path, package_dir=tmp_path)


def test_optimizer_stage_rejects_path_escape(tmp_path: Path) -> None:
    baseline, baseline_path = write_exact_baseline(tmp_path)
    artifact = initialize_optimizer_stage(
        package_id="fixture_package",
        package_dir=tmp_path,
        lowered_index=baseline,
        lowered_index_path=baseline_path,
    )
    document = read_json(artifact.path)
    document["exact_baseline"]["artifact_ref"] = "../outside.json"
    artifact.path.write_text(__import__("json").dumps(document))

    with pytest.raises(ModelCompileError, match="stay inside"):
        load_optimizer_stage(artifact.path, package_dir=tmp_path)
