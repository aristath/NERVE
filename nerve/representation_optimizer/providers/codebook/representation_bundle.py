from __future__ import annotations

from copy import deepcopy

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.providers.codebook.member_paths import (
    member_path,
)
from nerve.representation_optimizer.representation_ir import (
    REPRESENTATION_GRAPH_SCHEMA,
    finalize_representation_graph,
)


def bundle_representation_graphs(
    *,
    candidate: Json,
    members: tuple[tuple[str, Json], ...],
) -> Json:
    """Compose independently editable exact regions into one physical bundle."""

    if not members:
        raise ModelCompileError("representation bundle requires at least one member")
    graphs = []
    for scope_id, source in members:
        if source["scope_ids"] != [scope_id]:
            raise ModelCompileError(
                "representation bundle member does not belong to its scope"
            )
        graphs.append(_namespace_graph(scope_id, source))

    document = {
        "schema": REPRESENTATION_GRAPH_SCHEMA,
        "graph_id": "",
        "candidate_id": candidate["candidate_id"],
        "scope_ids": list(candidate["scope_ids"]),
        "source_contract_digests": dict(
            zip(
                candidate["scope_ids"],
                candidate["source_contract_digests"],
                strict=True,
            )
        ),
        "logical_contracts": _records(graphs, "logical_contracts"),
        "physical_representations": _records(
            graphs, "physical_representations"
        ),
        "signals": _records(graphs, "signals"),
        "resources": _records(graphs, "resources"),
        "nodes": _records(graphs, "nodes"),
        "connections": _records(graphs, "connections"),
        "public_ports": _records(graphs, "public_ports"),
        "islands": _records(graphs, "islands"),
        "absorbed_transforms": _records(graphs, "absorbed_transforms"),
        "physical_kernels": _records(graphs, "physical_kernels"),
        "confidence": {
            "mode": "exact",
            "score": 1.0,
            "basis": (
                "every non-overlapping member has an exhaustive exact proof "
                "and preserves its public semantic boundary"
            ),
            "evidence_refs": sorted(
                {
                    evidence
                    for graph in graphs
                    for evidence in graph["confidence"]["evidence_refs"]
                }
            ),
        },
        "unresolved": _records(graphs, "unresolved"),
        "correction_requests": _records(graphs, "correction_requests"),
    }
    return finalize_representation_graph(document)


def _records(graphs: list[Json], field: str) -> list[Json]:
    return sorted(
        (
            deepcopy(record)
            for graph in graphs
            for record in graph[field]
        ),
        key=lambda record: record["id"],
    )


def _namespace_graph(scope_id: str, source: Json) -> Json:
    graph = deepcopy(source)
    prefix = f"{scope_id}."

    logical = {item["id"]: prefix + item["id"] for item in graph["logical_contracts"]}
    physical = {
        item["id"]: prefix + item["id"]
        for item in graph["physical_representations"]
    }
    signals = {item["id"]: prefix + item["id"] for item in graph["signals"]}
    resources = {item["id"]: prefix + item["id"] for item in graph["resources"]}
    nodes = {item["id"]: prefix + item["id"] for item in graph["nodes"]}
    transforms = {
        item["id"]: prefix + item["id"] for item in graph["absorbed_transforms"]
    }

    for record in graph["logical_contracts"]:
        record["id"] = logical[record["id"]]
    for record in graph["physical_representations"]:
        record["id"] = physical[record["id"]]
    for record in graph["signals"]:
        record["id"] = signals[record["id"]]
        record["logical_contract_id"] = logical[record["logical_contract_id"]]
        record["physical_representation_id"] = physical[
            record["physical_representation_id"]
        ]
        _provenance(record, transforms)
    for record in graph["resources"]:
        record["id"] = resources[record["id"]]
        record["logical_contract_id"] = logical[record["logical_contract_id"]]
        record["physical_representation_id"] = physical[
            record["physical_representation_id"]
        ]
        record["artifact"]["path"] = member_path(
            scope_id, record["artifact"]["path"]
        )
        _provenance(record, transforms)
    for record in graph["nodes"]:
        record["id"] = nodes[record["id"]]
        for port in (*record["inputs"], *record["outputs"]):
            port["signal_id"] = signals[port["signal_id"]]
            port["physical_representation_id"] = physical[
                port["physical_representation_id"]
            ]
        record["resource_ids"] = [resources[value] for value in record["resource_ids"]]
        _provenance(record, transforms)
    for record in graph["public_ports"]:
        record["id"] = prefix + record["id"]
        record["logical_contract_id"] = logical[record["logical_contract_id"]]
        record["signal_id"] = signals[record["signal_id"]]
        record["node_id"] = nodes[record["node_id"]]
    for record in graph["absorbed_transforms"]:
        old_id = record["id"]
        record["id"] = transforms[old_id]
        record["source_representation_id"] = physical[
            record["source_representation_id"]
        ]
        record["target_representation_id"] = physical[
            record["target_representation_id"]
        ]
        record["adjacent_node_ids"] = [
            nodes[value] for value in record["adjacent_node_ids"]
        ]
        record["parameter_resource_ids"] = [
            resources[value] for value in record["parameter_resource_ids"]
        ]
        record["proof_ref"] = member_path(scope_id, record["proof_ref"])
        _provenance(record, transforms)
    for record in graph["physical_kernels"]:
        record["id"] = prefix + record["id"]
        record["node_ids"] = [nodes[value] for value in record["node_ids"]]
        record["artifact"]["path"] = member_path(
            scope_id, record["artifact"]["path"]
        )
        _provenance(record, transforms)
    return graph


def _provenance(record: Json, transforms: dict[str, str]) -> None:
    provenance = record["provenance"]
    provenance["transform_refs"] = [
        transforms.get(value, value) for value in provenance["transform_refs"]
    ]
