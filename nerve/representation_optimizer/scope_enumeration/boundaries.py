from __future__ import annotations

from collections import defaultdict

from nerve.compilation import Json
from nerve.representation_optimizer.scope_enumeration.graph import (
    NodeKey,
    SemanticDependencyGraph,
    SignalKey,
)


def derive_scope_boundary(
    graph: SemanticDependencyGraph,
    node_keys: set[NodeKey],
) -> Json:
    inputs: dict[str, Json] = {}
    outputs: dict[str, Json] = {}
    controls: dict[str, Json] = {}
    randomness: dict[str, Json] = {}
    parameters: dict[str, Json] = {}
    states: dict[str, Json] = {}
    randomness_consumers: dict[str, list[NodeKey]] = defaultdict(list)

    for node_key in _ordered_nodes(graph, node_keys):
        component_id, node_id = node_key
        component = graph.component_by_id[component_id]
        node = graph.node(node_key)
        for raw_signal in node.get("inputs", []):
            signal: SignalKey = (component_id, str(raw_signal))
            if graph.is_state_signal(signal):
                continue
            producer = graph.signal_producer_for_input(signal)
            if producer in node_keys:
                continue
            kind = graph.boundary_input_kind(signal)
            identifier = f"{kind}:{component_id}/{signal[1]}"
            upstream_edge = graph.inter_component_edge(signal)
            record = {
                "id": identifier,
                "component_id": component_id,
                "signal_id": signal[1],
                "definition": graph.signal_definition(signal),
                "upstream": (
                    _signal_endpoint(graph.inter_component_source(signal))
                    if upstream_edge is not None
                    else None
                ),
                "upstream_connection": (
                    _connection_record(upstream_edge)
                    if upstream_edge is not None
                    else None
                ),
            }
            if kind == "control":
                controls[identifier] = record
            elif kind == "randomness":
                randomness[identifier] = record
                randomness_consumers[identifier].append(node_key)
            else:
                inputs[identifier] = record

        for raw_signal in node.get("outputs", []):
            signal = (component_id, str(raw_signal))
            consumers = graph.signal_consumers(signal)
            if (
                consumers
                and all(consumer in node_keys for consumer in consumers)
                and not graph.is_public_output(signal)
            ):
                continue
            if (
                not consumers
                and not graph.is_public_output(signal)
                and not graph.is_declared_boundary_output(signal)
            ):
                continue
            identifier = f"output:{component_id}/{signal[1]}"
            outputs[identifier] = {
                "id": identifier,
                "component_id": component_id,
                "signal_id": signal[1],
                "definition": graph.signal_definition(signal),
                "downstream_consumers": [
                    f"{consumer[0]}/{consumer[1]}"
                    for consumer in consumers
                    if consumer not in node_keys
                ],
                "downstream_connections": [
                    _connection_record(edge)
                    for edge in graph.outgoing_edges(signal)
                ],
                "public": graph.is_public_output(signal),
            }

        for parameter_id in node.get("params", []):
            parameter_id = str(parameter_id)
            identifier = f"parameter:{component_id}/{parameter_id}"
            parameters[identifier] = {
                "id": identifier,
                "component_id": component_id,
                "parameter_ref_id": parameter_id,
                "definition": dict(
                    component.circuit["parameters"]["refs"][parameter_id]
                ),
            }

        state_reads = {str(value) for value in node.get("state_reads", [])}
        state_writes = {str(value) for value in node.get("state_writes", [])}
        for state_id in sorted(state_reads | state_writes):
            identifier = f"state:{component_id}/{state_id}"
            record = states.setdefault(
                identifier,
                {
                    "id": identifier,
                    "component_id": component_id,
                    "state_port_id": state_id,
                    "access": [],
                    "definition": _state_definition(
                        component.circuit,
                        state_id,
                    ),
                },
            )
            if state_id in state_reads and "read" not in record["access"]:
                record["access"].append("read")
            if state_id in state_writes and "write" not in record["access"]:
                record["access"].append("write")

    for state in states.values():
        state["access"].sort()

    for identifier, consumer_keys in randomness_consumers.items():
        randomness[identifier]["consumers"] = [
            f"{component_id}/{node_id}"
            for component_id, node_id in consumer_keys
        ]
        semantics = {
            str(
                graph.node(consumer)
                .get("attrs", {})
                .get("randomness", "")
            )
            for consumer in consumer_keys
        } - {""}
        randomness[identifier]["semantics"] = sorted(semantics)

    for node_key in _ordered_nodes(graph, node_keys):
        node = graph.node(node_key)
        semantic = node.get("attrs", {}).get("randomness")
        if not isinstance(semantic, str) or not semantic:
            continue
        if any(
            f"{node_key[0]}/{node_key[1]}" in record.get("consumers", [])
            for record in randomness.values()
        ):
            continue
        identifier = f"randomness:implicit/{node_key[0]}/{node_key[1]}"
        randomness[identifier] = {
            "id": identifier,
            "component_id": node_key[0],
            "source_node_id": node_key[1],
            "semantics": [semantic],
            "definition": {"kind": "implicit_runtime_randomness"},
        }

    return {
        "inputs": _sorted_records(inputs),
        "outputs": _sorted_records(outputs),
        "parameters": _sorted_records(parameters),
        "states": _sorted_records(states),
        "controls": _sorted_records(controls),
        "randomness": _sorted_records(randomness),
        "dependencies": _dependency_edges(graph, node_keys),
    }


def _ordered_nodes(
    graph: SemanticDependencyGraph,
    node_keys: set[NodeKey],
) -> tuple[NodeKey, ...]:
    component_order = graph.component_order
    node_order = {
        (component.component_id, str(node["id"])): node_index
        for component in graph.components
        for node_index, node in enumerate(component.nodes)
    }
    return tuple(
        sorted(
            node_keys,
            key=lambda key: (component_order[key[0]], node_order[key]),
        )
    )


def _sorted_records(records: dict[str, Json]) -> list[Json]:
    return [records[key] for key in sorted(records)]


def _signal_endpoint(signal: SignalKey | None) -> Json | None:
    if signal is None:
        return None
    return {"component_id": signal[0], "signal_id": signal[1]}


def _connection_record(edge: Json) -> Json:
    return {
        "edge_id": str(edge["id"]),
        "connection": dict(edge["connection"]),
        "source": dict(edge["source"]),
        "destination": dict(edge["destination"]),
    }


def _dependency_edges(
    graph: SemanticDependencyGraph,
    node_keys: set[NodeKey],
) -> list[Json]:
    dependencies = []
    for edge in graph.edges:
        source = edge["source"]
        destination = edge["destination"]
        source_signal = (
            str(source["component_id"]),
            str(source["port_id"]),
        )
        destination_signal = (
            str(destination["component_id"]),
            str(destination["port_id"]),
        )
        producer = graph.local_signal_producer(source_signal)
        covered_consumers = [
            consumer
            for consumer in graph.local_signal_consumers(destination_signal)
            if consumer in node_keys
        ]
        if producer not in node_keys or not covered_consumers:
            continue
        dependencies.append(
            {
                "edge_id": str(edge["id"]),
                "connection": dict(edge["connection"]),
                "source": dict(source),
                "destination": dict(destination),
                "covered_consumer_node_ids": [
                    f"{component_id}/{node_id}"
                    for component_id, node_id in covered_consumers
                ],
            }
        )
    return dependencies


def _state_definition(circuit: Json, state_id: str) -> Json:
    for state in circuit.get("state_ports", []):
        if str(state.get("id")) == state_id:
            return dict(state)
    raise KeyError(state_id)
