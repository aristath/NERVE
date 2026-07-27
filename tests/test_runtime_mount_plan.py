from __future__ import annotations

import json
from pathlib import Path

import pytest

from nerve.representation_optimizer.contracts import (
    ContractValidationError,
)
from nerve.representation_optimizer.mounting import (
    RuntimeMountPlan,
    validate_runtime_mount_artifacts,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
)


def build_plan() -> CandidateBuildPlan:
    return CandidateBuildPlan.from_json(
        {
            "schema": "nerve.optimizer.candidate_build_plan.v1",
            "phases": [
                "semantic_construction",
                "ordinary_lowering",
                "physical_optimization",
            ],
            "source_inputs": [],
            "outputs": [
                {
                    "path": "overlays/component.json",
                    "kind": "runtime_component_overlay",
                    "lifetime": "mount",
                    "producer_phase": "ordinary_lowering",
                    "resident_bytes": 0,
                    "validator_id": "json_contract",
                    "validation_contract": {
                        "schema": (
                            "nerve.optimizer."
                            "vulkan_component_overlay.v1"
                        ),
                        "object_required": True,
                    },
                },
                {
                    "path": "tensor_fragment.json",
                    "kind": "tensor_index_fragment",
                    "lifetime": "mount",
                    "producer_phase": "semantic_construction",
                    "resident_bytes": 0,
                    "validator_id": "json_contract",
                    "validation_contract": {
                        "schema": "nerve.tensor_index.v1",
                        "object_required": True,
                    },
                },
            ],
            "resource_limits": {
                "maximum_construction_time_ns": None,
                "maximum_temporary_bytes": None,
                "maximum_staging_bytes": None,
            },
        }
    )


def mount_document() -> dict[str, object]:
    return {
        "schema": "nerve.optimizer.runtime_mount_plan.v1",
        "candidate_id": "candidate_fixture",
        "adapter_id": (
            "vulkan_stream_circuit_component_overlay.v1"
        ),
        "component_replacements": [
            {
                "source_component_id": "component",
                "overlay_ref": "overlays/component.json",
            }
        ],
        "tensor_index_refs": ["tensor_fragment.json"],
    }


def test_mount_plan_binds_executable_artifacts_to_candidate_outputs():
    mount = RuntimeMountPlan.from_json(
        mount_document(),
        candidate_id="candidate_fixture",
        build_plan=build_plan(),
    )

    assert (
        mount.to_json()["component_replacements"][0][
            "source_component_id"
        ]
        == "component"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "candidate_id",
            "candidate_other",
            "different candidate",
        ),
        (
            "adapter_id",
            "unknown_adapter.v1",
            "unsupported runtime mount adapter",
        ),
    ],
)
def test_mount_plan_rejects_unbound_identity(
    field: str,
    value: str,
    message: str,
):
    document = mount_document()
    document[field] = value

    with pytest.raises(ContractValidationError, match=message):
        RuntimeMountPlan.from_json(
            document,
            candidate_id="candidate_fixture",
            build_plan=build_plan(),
        )


def test_mount_plan_rejects_undeclared_or_escaping_artifacts():
    for reference in (
        "overlays/undeclared.json",
        "../component.json",
    ):
        document = mount_document()
        document["component_replacements"][0][
            "overlay_ref"
        ] = reference

        with pytest.raises(ContractValidationError):
            RuntimeMountPlan.from_json(
                document,
                candidate_id="candidate_fixture",
                build_plan=build_plan(),
            )


def test_mount_artifacts_reject_source_identity_drift(
    tmp_path: Path,
):
    overlay = tmp_path / "overlays" / "component.json"
    overlay.parent.mkdir()
    overlay.write_text(
        json.dumps(
            {
                "schema": (
                    "nerve.optimizer.vulkan_component_overlay.v1"
                ),
                "source_component_id": "other_component",
                "component": {},
                "execution": {},
            }
        )
    )
    (tmp_path / "tensor_fragment.json").write_text(
        json.dumps(
            {
                "schema": "nerve.tensor_index.v1",
                "tensors": {},
            }
        )
    )
    mount = RuntimeMountPlan.from_json(
        mount_document(),
        candidate_id="candidate_fixture",
        build_plan=build_plan(),
    )

    with pytest.raises(
        ContractValidationError,
        match="source component disagrees",
    ):
        validate_runtime_mount_artifacts(tmp_path, mount)
