from __future__ import annotations

from dataclasses import dataclass

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import ContractValidationError
from nerve.representation_optimizer.representation_ir.contracts import (
    RepresentationGraphDocument,
)


@dataclass(frozen=True)
class RepresentationGraphPlan:
    graph_id: str
    execution_order: tuple[str, ...]
    transducer_node_ids: tuple[str, ...]
    correction_node_ids: tuple[str, ...]
    island_ids: tuple[str, ...]
    native_cross_scope_connection_ids: tuple[str, ...]
    source_materialization_connection_ids: tuple[str, ...]
    absorbed_transform_ids: tuple[str, ...]
    aggregate_cost_metrics: tuple[tuple[str, float], ...]


def plan_representation_graph(
    value: Json | RepresentationGraphDocument,
) -> RepresentationGraphPlan:
    document = (
        value
        if isinstance(value, RepresentationGraphDocument)
        else RepresentationGraphDocument.from_json(value)
    )
    graph = document.to_json()
    nodes = {node["id"]: node for node in graph["nodes"]}
    incoming = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for connection in graph["connections"]:
        producer = connection["producer"]["node_id"]
        consumer = connection["consumer"]["node_id"]
        incoming[consumer] += 1
        outgoing[producer].append(consumer)
    ready = sorted(node_id for node_id, count in incoming.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for consumer in sorted(outgoing[node_id]):
            incoming[consumer] -= 1
            if incoming[consumer] == 0:
                ready.append(consumer)
                ready.sort()
    if len(order) != len(nodes):
        cyclic = sorted(node_id for node_id, count in incoming.items() if count)
        raise ContractValidationError(
            f"representation graph contains an executable cycle: {cyclic}"
        )

    aggregate: dict[str, float] = {}
    for record in [*graph["nodes"], *graph["physical_kernels"]]:
        cost = record.get("cost")
        if cost is None:
            continue
        for name, value in cost["metrics"].items():
            aggregate[name] = aggregate.get(name, 0.0) + float(value)

    node_scopes = {
        node["id"]: set(node["provenance"]["scope_ids"]) for node in graph["nodes"]
    }
    native_cross_scope = []
    materialized = []
    for connection in graph["connections"]:
        if connection["materializes_source"]:
            materialized.append(connection["id"])
        elif (
            node_scopes[connection["producer"]["node_id"]]
            != node_scopes[connection["consumer"]["node_id"]]
        ):
            native_cross_scope.append(connection["id"])

    return RepresentationGraphPlan(
        graph_id=document.graph_id,
        execution_order=tuple(order),
        transducer_node_ids=tuple(
            node["id"] for node in graph["nodes"] if node["kind"] == "transducer"
        ),
        correction_node_ids=tuple(
            node["id"] for node in graph["nodes"] if node["kind"] == "correction"
        ),
        island_ids=tuple(island["id"] for island in graph["islands"]),
        native_cross_scope_connection_ids=tuple(native_cross_scope),
        source_materialization_connection_ids=tuple(materialized),
        absorbed_transform_ids=tuple(
            transform["id"] for transform in graph["absorbed_transforms"]
        ),
        aggregate_cost_metrics=tuple(sorted(aggregate.items())),
    )
