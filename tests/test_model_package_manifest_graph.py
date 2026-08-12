import json
from pathlib import Path

from nerve.model_package_manifest import (
    package_auxiliary_circuit_graph,
    package_circuit_graph,
)


def _compiled_circuit(tensor: str) -> dict:
    return {
        "id": "circuit_v1",
        "nodes": [],
        "parameters": {
            "layout": "logical_parameter_refs",
            "storage": "package_tensor_index",
            "refs": {"weight": {"tensor": tensor}},
        },
    }


def _write_stale_artifacts(root: Path) -> None:
    component = root / "component"
    component.mkdir()
    (component / "params.json").write_text(
        json.dumps(
            {
                "schema": "nerve.circuit_params.v1",
                "circuit": "circuit_v1",
                "layout": "logical_parameter_refs",
                "storage": "package_tensor_index",
                "refs": {"weight": {"tensor": "canonical.weight"}},
            }
        )
    )
    (component / "state.json").write_text(
        json.dumps(
            {
                "schema": "nerve.circuit_state.v1",
                "circuit": "circuit_v1",
                "state_ports": [],
            }
        )
    )


def _ref() -> dict:
    return {
        "id": "component",
        "operator_type": "sparse_attention",
        "runtime_role": "signal_processor",
        "implementation": "compiled",
        "behavioral_role": "exact",
        "params": "component/params.json",
        "state": "component/state.json",
    }


def test_target_manifest_params_are_derived_from_compiled_circuit(
    tmp_path: Path,
) -> None:
    _write_stale_artifacts(tmp_path)
    circuit = _compiled_circuit("physical.weight")
    lowered_index = {
        "graph": {
            "circuits": [_ref()],
            "topology": "explicit_graph",
            "edges": [],
            "boundary": {},
        }
    }

    graph = package_circuit_graph(
        lowered_index,
        tmp_path,
        {"component": circuit},
    )

    component = graph["components"][0]
    assert component["circuit"]["parameters"]["refs"] == {
        "weight": {"tensor": "physical.weight"}
    }
    assert component["params"]["refs"] == component["circuit"]["parameters"]["refs"]


def test_draft_manifest_params_are_derived_from_compiled_circuit(
    tmp_path: Path,
) -> None:
    _write_stale_artifacts(tmp_path)
    circuit = _compiled_circuit("physical.weight")
    draft = {
        "id": "draft",
        "type": "parallel_backbone_markov",
        "circuits": [_ref()],
        "topology": "explicit_graph",
        "edges": [],
        "boundary": {},
    }

    graph = package_auxiliary_circuit_graph(
        draft,
        tmp_path,
        {"component": circuit},
    )

    component = graph["components"][0]
    assert component["params"]["refs"] == {"weight": {"tensor": "physical.weight"}}
    assert component["params"]["refs"] == component["circuit"]["parameters"]["refs"]
