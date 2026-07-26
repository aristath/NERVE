from __future__ import annotations

from copy import deepcopy

import pytest

from nerve.representation_optimizer.contracts import ContractValidationError
from nerve.representation_optimizer.representation_ir import (
    RepresentationGraphDocument,
    finalize_representation_graph,
    plan_representation_graph,
    representation_graph_id,
    validate_representation_graph,
)
from tests.representation_graph_fixtures import exact_representation_graph


def test_heterogeneous_graph_preserves_logical_and_physical_contracts():
    graph = exact_representation_graph()
    validate_representation_graph(graph)

    logical_hidden = next(
        item for item in graph["logical_contracts"] if item["id"] == "logical.hidden"
    )
    spectral = next(
        item
        for item in graph["physical_representations"]
        if item["id"] == "repr.signal.spectral"
    )
    state = next(
        item for item in graph["resources"] if item["id"] == "resource.state"
    )
    topology = next(
        item for item in graph["resources"] if item["id"] == "resource.topology"
    )

    assert logical_hidden == {
        "id": "logical.hidden",
        "signal": "hidden",
        "shape": [8],
        "dtype": "BF16",
    }
    assert spectral["physical_shape"] == [5]
    assert spectral["kind"] == "spectral_coefficients"
    assert state["physical_representation_id"] == "repr.state.compact"
    assert topology["physical_representation_id"] == "repr.topology.events"


def test_planner_keeps_native_representation_across_scopes_and_prices_transducers():
    plan = plan_representation_graph(exact_representation_graph())

    assert plan.execution_order == (
        "node.decode_input",
        "node.scope_a",
        "node.scope_b",
        "node.decode_output",
    )
    assert plan.transducer_node_ids == (
        "node.decode_input",
        "node.decode_output",
    )
    assert plan.native_cross_scope_connection_ids == ("connection.cross_scope",)
    assert plan.source_materialization_connection_ids == ()
    assert plan.absorbed_transform_ids == ("transform.parameter_basis",)
    assert dict(plan.aggregate_cost_metrics) == {
        "latency_ns": 124.0,
        "transferred_bytes": 32.0,
    }


def test_incompatible_connection_is_rejected_instead_of_implicitly_converting():
    graph = exact_representation_graph()
    scope_b = next(node for node in graph["nodes"] if node["id"] == "node.scope_b")
    scope_b["inputs"][0]["physical_representation_id"] = "repr.signal.dense"
    graph["graph_id"] = representation_graph_id(graph)

    with pytest.raises(
        ContractValidationError,
        match="incompatible physical representation",
    ):
        validate_representation_graph(graph)


def test_multi_scope_island_requires_a_non_materialized_native_connection():
    graph = exact_representation_graph()
    graph["connections"][0]["materializes_source"] = True
    graph["graph_id"] = ""
    with pytest.raises(
        ContractValidationError,
        match="does not retain a native representation",
    ):
        finalize_representation_graph(graph)


def test_absorbed_basis_change_requires_parameter_resource_and_proof():
    graph = exact_representation_graph()
    graph["absorbed_transforms"][0]["parameter_resource_ids"] = ["resource.state"]
    graph["graph_id"] = ""
    with pytest.raises(
        ContractValidationError,
        match="must target parameters",
    ):
        finalize_representation_graph(graph)

    graph = exact_representation_graph()
    graph["absorbed_transforms"][0]["proof_ref"] = ""
    graph["graph_id"] = ""
    with pytest.raises(ContractValidationError, match="proof_ref"):
        finalize_representation_graph(graph)


def test_every_physical_artifact_and_executable_has_scope_provenance():
    graph = exact_representation_graph()
    for collection in ("signals", "resources", "nodes", "physical_kernels"):
        for record in graph[collection]:
            assert record["provenance"]["scope_ids"]
            assert record["provenance"]["evidence_refs"]

    graph["resources"][0]["provenance"]["scope_ids"] = []
    graph["graph_id"] = ""
    with pytest.raises(ContractValidationError, match="must not be empty"):
        finalize_representation_graph(graph)


def test_verified_approximation_exposes_confidence_and_correction_request():
    graph = exact_representation_graph()
    correction_node = {
        "id": "node.correction",
        "kind": "correction",
        "operation": "exact_residual_correction",
        "inputs": [
            {
                "id": "input",
                "signal_id": "signal.native_output",
                "physical_representation_id": "repr.signal.spectral",
            }
        ],
        "outputs": [],
        "resource_ids": [],
        "state_read_ids": [],
        "state_write_ids": [],
        "cost": None,
        "provenance": {
            "scope_ids": ["scope_b"],
            "source_node_ids": ["source.output"],
            "evidence_refs": ["evidence_fixture"],
            "transform_refs": [],
        },
    }
    graph["nodes"].insert(0, correction_node)
    graph["confidence"] = {
        "mode": "verified_approximation",
        "score": 0.999,
        "basis": "bounded residual with exact correction",
        "evidence_refs": ["evidence_fixture"],
    }
    graph["correction_requests"] = [
        {
            "id": "correction.residual",
            "trigger": {"kind": "residual_bound", "maximum": 0.001},
            "correction_node_id": "node.correction",
            "fallback_scope_ids": ["scope_b"],
            "output_port_ids": ["port.output"],
            "error_contract": {"maximum_absolute_error": 0.001},
            "provenance": {
                "scope_ids": ["scope_b"],
                "source_node_ids": ["source.output"],
                "evidence_refs": ["evidence_fixture"],
                "transform_refs": [],
            },
        }
    ]
    graph["graph_id"] = ""
    approximate = finalize_representation_graph(graph)

    assert RepresentationGraphDocument.from_json(approximate).to_json() == approximate

    missing_correction = deepcopy(approximate)
    missing_correction["correction_requests"] = []
    missing_correction["graph_id"] = ""
    with pytest.raises(
        ContractValidationError,
        match="requires at least one correction request",
    ):
        finalize_representation_graph(missing_correction)


def test_graph_identity_covers_physical_layout_and_is_copy_safe():
    graph = exact_representation_graph()
    document = RepresentationGraphDocument.from_json(graph)
    mutated = document.to_json()
    mutated["physical_representations"][0]["storage"]["layout"] = "column_major"

    assert document.to_json() == graph
    with pytest.raises(ContractValidationError, match="graph_id must be canonical"):
        validate_representation_graph(mutated)


def test_planner_rejects_executable_cycle():
    graph = exact_representation_graph()
    scope_a = next(node for node in graph["nodes"] if node["id"] == "node.scope_a")
    scope_a["inputs"].append(
        {
            "id": "recurrent",
            "signal_id": "signal.native_output",
            "physical_representation_id": "repr.signal.spectral",
        }
    )
    graph["connections"].append(
        {
            "id": "connection.recurrent",
            "producer": {"node_id": "node.scope_b", "port_id": "output"},
            "consumer": {"node_id": "node.scope_a", "port_id": "recurrent"},
            "signal_id": "signal.native_output",
            "materializes_source": False,
        },
    )
    graph["graph_id"] = ""
    graph = finalize_representation_graph(graph)

    with pytest.raises(ContractValidationError, match="executable cycle"):
        plan_representation_graph(graph)
