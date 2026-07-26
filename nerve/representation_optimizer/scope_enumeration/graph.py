from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nerve.compilation import Json, ModelCompileError, read_json


NodeKey = tuple[str, str]
SignalKey = tuple[str, str]


@dataclass(frozen=True)
class ComponentView:
    component_id: str
    operator_type: str
    runtime_role: str
    implementation: str
    artifact_ref: str
    circuit: Json
    nodes: tuple[Json, ...]
    modules: tuple[Json, ...]

    @property
    def node_by_id(self) -> dict[str, Json]:
        return {str(node["id"]): node for node in self.nodes}

    @property
    def module_by_id(self) -> dict[str, Json]:
        return {str(module["id"]): module for module in self.modules}

    def subtree_node_ids(self, module_id: str) -> tuple[str, ...]:
        by_id = self.module_by_id
        if module_id not in by_id:
            raise ModelCompileError(
                f"component {self.component_id!r} has no semantic module {module_id!r}"
            )
        ordered = {str(node["id"]): index for index, node in enumerate(self.nodes)}
        found: set[str] = set()
        visiting: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visiting:
                raise ModelCompileError(
                    f"component {self.component_id!r} semantic module tree "
                    f"contains a cycle at {current_id!r}"
                )
            module = by_id.get(current_id)
            if module is None:
                raise ModelCompileError(
                    f"component {self.component_id!r} semantic module "
                    f"references missing child {current_id!r}"
                )
            visiting.add(current_id)
            for node_id in module.get("source_node_ids", []):
                if node_id not in ordered:
                    raise ModelCompileError(
                        f"semantic module {self.component_id}/{current_id} "
                        f"references missing source node {node_id!r}"
                    )
                found.add(str(node_id))
            for child_id in module.get("child_ids", []):
                visit(str(child_id))
            visiting.remove(current_id)

        visit(module_id)
        return tuple(sorted(found, key=ordered.__getitem__))

    def qualified_nodes(self, node_ids: Iterable[str]) -> tuple[NodeKey, ...]:
        return tuple((self.component_id, node_id) for node_id in node_ids)


@dataclass(frozen=True)
class SemanticDependencyGraph:
    components: tuple[ComponentView, ...]
    edges: tuple[Json, ...]
    external_inputs: tuple[Json, ...]
    public_outputs: tuple[Json, ...]
    graph_artifact_ref: str | None

    @classmethod
    def from_lowered_package(
        cls,
        *,
        package_dir: Path,
        lowered_index: Json,
        lowered_index_ref: str,
    ) -> SemanticDependencyGraph:
        graph = lowered_index.get("graph")
        if not isinstance(graph, dict):
            raise ModelCompileError("lowered execution graph has no graph object")
        raw_circuits = graph.get("circuits")
        if not isinstance(raw_circuits, list):
            raise ModelCompileError("lowered execution graph has no circuit list")
        components = []
        for index, raw_entry in enumerate(raw_circuits):
            if not isinstance(raw_entry, dict):
                raise ModelCompileError(
                    f"lowered circuit entry {index} must be an object"
                )
            artifact_ref = raw_entry.get("circuit")
            if not isinstance(artifact_ref, str) or not artifact_ref:
                raise ModelCompileError(
                    f"lowered circuit entry {index} has no circuit artifact"
                )
            relative = Path(artifact_ref)
            if relative.is_absolute() or ".." in relative.parts:
                raise ModelCompileError(
                    f"lowered circuit artifact must stay inside package: {artifact_ref}"
                )
            circuit_path = package_dir / "lowered" / relative
            if not circuit_path.is_file():
                raise ModelCompileError(
                    f"lowered circuit artifact is missing: {circuit_path}"
                )
            components.append(
                _component_view(
                    raw_entry,
                    read_json(circuit_path),
                    f"lowered/{relative.as_posix()}",
                )
            )
        boundary = graph.get("boundary", {})
        if not isinstance(boundary, dict):
            raise ModelCompileError("lowered execution graph boundary must be an object")
        return cls.from_documents(
            components=components,
            edges=graph.get("edges", []),
            external_inputs=boundary.get("external_inputs", []),
            public_outputs=boundary.get("public_outputs", []),
            graph_artifact_ref=lowered_index_ref,
        )

    @classmethod
    def from_documents(
        cls,
        *,
        components: Iterable[ComponentView],
        edges: object,
        external_inputs: object = (),
        public_outputs: object = (),
        graph_artifact_ref: str | None = None,
    ) -> SemanticDependencyGraph:
        component_tuple = tuple(components)
        if graph_artifact_ref is not None:
            graph_relative = Path(graph_artifact_ref)
            if (
                not graph_artifact_ref
                or graph_relative.is_absolute()
                or ".." in graph_relative.parts
            ):
                raise ModelCompileError(
                    "semantic dependency graph artifact must stay inside the package"
                )
        component_ids = [component.component_id for component in component_tuple]
        if len(component_ids) != len(set(component_ids)):
            raise ModelCompileError("semantic dependency graph component ids must be unique")
        raw_edges = _object_list(edges, "semantic dependency graph edges")
        raw_external = _object_list(
            external_inputs,
            "semantic dependency graph external inputs",
        )
        raw_public = _object_list(
            public_outputs,
            "semantic dependency graph public outputs",
        )
        result = cls(
            components=component_tuple,
            edges=tuple(raw_edges),
            external_inputs=tuple(raw_external),
            public_outputs=tuple(raw_public),
            graph_artifact_ref=graph_artifact_ref,
        )
        result._validate()
        return result

    @property
    def component_by_id(self) -> dict[str, ComponentView]:
        return {component.component_id: component for component in self.components}

    @property
    def component_order(self) -> dict[str, int]:
        return {
            component.component_id: index
            for index, component in enumerate(self.components)
        }

    @property
    def all_node_keys(self) -> tuple[NodeKey, ...]:
        return tuple(
            (component.component_id, str(node["id"]))
            for component in self.components
            for node in component.nodes
        )

    def component_nodes(self, component_id: str) -> tuple[NodeKey, ...]:
        component = self.component_by_id[component_id]
        return tuple(
            (component_id, str(node["id"]))
            for node in component.nodes
        )

    def node(self, key: NodeKey) -> Json:
        component_id, node_id = key
        try:
            return self.component_by_id[component_id].node_by_id[node_id]
        except KeyError as error:
            raise ModelCompileError(
                f"semantic dependency graph has no source node {component_id}/{node_id}"
            ) from error

    def local_signal_producer(self, signal: SignalKey) -> NodeKey | None:
        component_id, signal_id = signal
        matches = [
            (component_id, str(node["id"]))
            for node in self.component_by_id[component_id].nodes
            if signal_id in node.get("outputs", [])
        ]
        if len(matches) > 1:
            raise ModelCompileError(
                f"signal {component_id}/{signal_id} has multiple source producers"
            )
        return matches[0] if matches else None

    def signal_producer_for_input(self, signal: SignalKey) -> NodeKey | None:
        local = self.local_signal_producer(signal)
        if local is not None:
            return local
        source = self.inter_component_source(signal)
        if source is None:
            return None
        return self.local_signal_producer(source)

    def inter_component_source(self, destination: SignalKey) -> SignalKey | None:
        edge = self.inter_component_edge(destination)
        return _endpoint(edge["source"]) if edge is not None else None

    def inter_component_edge(self, destination: SignalKey) -> Json | None:
        matches = [
            edge
            for edge in self.edges
            if _endpoint(edge.get("destination")) == destination
        ]
        if len(matches) > 1:
            raise ModelCompileError(
                f"signal {destination[0]}/{destination[1]} has multiple graph producers"
            )
        return matches[0] if matches else None

    def outgoing_edges(self, source: SignalKey) -> tuple[Json, ...]:
        return tuple(
            edge
            for edge in self.edges
            if _endpoint(edge.get("source")) == source
        )

    def local_signal_consumers(self, signal: SignalKey) -> tuple[NodeKey, ...]:
        component_id, signal_id = signal
        return tuple(
            (component_id, str(node["id"]))
            for node in self.component_by_id[component_id].nodes
            if signal_id in node.get("inputs", [])
        )

    def signal_consumers(self, signal: SignalKey) -> tuple[NodeKey, ...]:
        consumers = list(self.local_signal_consumers(signal))
        for edge in self.outgoing_edges(signal):
            destination = _endpoint(edge["destination"])
            assert destination is not None
            consumers.extend(self.local_signal_consumers(destination))
        return tuple(dict.fromkeys(consumers))

    def is_public_output(self, signal: SignalKey) -> bool:
        return any(
            _endpoint(record.get("endpoint")) == signal
            for record in self.public_outputs
        )

    def boundary_input_kind(self, signal: SignalKey) -> str:
        component = self.component_by_id[signal[0]]
        for record in component.circuit.get("boundary", {}).get("controls", []):
            if str(record.get("id")) == signal[1]:
                return "control"
        for record in component.circuit.get("boundary", {}).get("inputs", []):
            if str(record.get("id")) != signal[1]:
                continue
            if record.get("component_port") == "randomness" or record.get(
                "signal"
            ) == "random_seed":
                return "randomness"
            return "input"
        return "input"

    def is_declared_boundary_input(self, signal: SignalKey) -> bool:
        component = self.component_by_id[signal[0]]
        boundary = component.circuit.get("boundary", {})
        return any(
            str(record.get("id")) == signal[1]
            for field in ("inputs", "controls")
            for record in boundary.get(field, [])
        )

    def is_declared_boundary_output(self, signal: SignalKey) -> bool:
        component = self.component_by_id[signal[0]]
        return any(
            str(record.get("id")) == signal[1]
            for record in component.circuit.get("boundary", {}).get("outputs", [])
        )

    def is_state_signal(self, signal: SignalKey) -> bool:
        component = self.component_by_id[signal[0]]
        return signal[1] in {
            str(state.get("id"))
            for state in component.circuit.get("state_ports", [])
        }

    def signal_definition(self, signal: SignalKey) -> Json:
        component = self.component_by_id[signal[0]]
        boundary = component.circuit.get("boundary", {})
        for field in ("inputs", "outputs", "controls"):
            for record in boundary.get(field, []):
                if str(record.get("id")) == signal[1]:
                    return dict(record)
        return {"id": signal[1]}

    def _validate(self) -> None:
        components = self.component_by_id
        edge_ids = [edge.get("id") for edge in self.edges]
        if (
            any(not isinstance(edge_id, str) or not edge_id for edge_id in edge_ids)
            or len(edge_ids) != len(set(edge_ids))
        ):
            raise ModelCompileError(
                "semantic dependency graph edge ids must be non-empty and unique"
            )
        for component in self.components:
            _validate_component(component)
            for node in component.nodes:
                for parameter in node.get("params", []):
                    if parameter not in component.circuit["parameters"]["refs"]:
                        raise ModelCompileError(
                            f"node {component.component_id}/{node['id']} references "
                            f"unknown parameter {parameter!r}"
                        )
                for state in (
                    *node.get("state_reads", []),
                    *node.get("state_writes", []),
                ):
                    if state not in {
                        str(port["id"])
                        for port in component.circuit.get("state_ports", [])
                    }:
                        raise ModelCompileError(
                            f"node {component.component_id}/{node['id']} references "
                            f"unknown state {state!r}"
                        )
        for index, edge in enumerate(self.edges):
            source = _endpoint(edge.get("source"))
            destination = _endpoint(edge.get("destination"))
            connection = edge.get("connection")
            if source is None or destination is None:
                raise ModelCompileError(
                    f"semantic dependency graph edge {index} has invalid endpoints"
                )
            if (
                not isinstance(connection, dict)
                or not isinstance(connection.get("kind"), str)
                or not connection["kind"]
            ):
                raise ModelCompileError(
                    f"semantic dependency graph edge {index} has no connection semantics"
                )
            if source[0] not in components or destination[0] not in components:
                raise ModelCompileError(
                    f"semantic dependency graph edge {index} references unknown component"
                )
            if self.local_signal_producer(source) is None:
                raise ModelCompileError(
                    f"semantic dependency graph edge {index} source "
                    f"{source[0]}/{source[1]} has no producer"
                )
            destination_consumers = self.local_signal_consumers(destination)
            if not destination_consumers:
                raise ModelCompileError(
                    f"semantic dependency graph edge {index} destination "
                    f"{destination[0]}/{destination[1]} has no consumer"
                )
        for signal in self.all_output_signals():
            self.local_signal_producer(signal)
        for signal in self.all_input_signals():
            if self.is_state_signal(signal):
                continue
            self.inter_component_source(signal)
            if (
                self.signal_producer_for_input(signal) is None
                and not self.is_declared_boundary_input(signal)
            ):
                raise ModelCompileError(
                    f"signal {signal[0]}/{signal[1]} has no producer or declared boundary"
                )

    def all_input_signals(self) -> tuple[SignalKey, ...]:
        return tuple(
            (component.component_id, str(signal))
            for component in self.components
            for node in component.nodes
            for signal in node.get("inputs", [])
        )

    def all_output_signals(self) -> tuple[SignalKey, ...]:
        return tuple(
            (component.component_id, str(signal))
            for component in self.components
            for node in component.nodes
            for signal in node.get("outputs", [])
        )


def component_view(
    *,
    component_id: str,
    operator_type: str,
    runtime_role: str,
    implementation: str,
    artifact_ref: str,
    circuit: Json,
) -> ComponentView:
    return _component_view(
        {
            "id": component_id,
            "operator_type": operator_type,
            "runtime_role": runtime_role,
            "implementation": implementation,
        },
        circuit,
        artifact_ref,
    )


def _component_view(entry: Json, circuit: Json, artifact_ref: str) -> ComponentView:
    component_id = str(entry.get("id", ""))
    if not component_id:
        raise ModelCompileError("lowered circuit entry has no component id")
    source_component_id = circuit.get("source", {}).get("component_id")
    if source_component_id != component_id:
        raise ModelCompileError(
            f"lowered circuit {artifact_ref} source component does not match "
            f"index component {component_id!r}"
        )
    raw_nodes = circuit.get("semantic_execution_nodes", circuit.get("nodes"))
    if not isinstance(raw_nodes, list):
        raise ModelCompileError(
            f"lowered circuit {artifact_ref} has no semantic source node list"
        )
    nodes = tuple(_object_list(raw_nodes, f"{artifact_ref} nodes"))
    raw_tree = circuit.get("semantic_module_tree")
    if raw_tree is None:
        modules: tuple[Json, ...] = ()
    elif not isinstance(raw_tree, dict):
        raise ModelCompileError(
            f"lowered circuit {artifact_ref} semantic module tree must be an object"
        )
    else:
        modules = tuple(
            _object_list(
                raw_tree.get("modules"),
                f"{artifact_ref} semantic modules",
            )
        )
    return ComponentView(
        component_id=component_id,
        operator_type=str(entry.get("operator_type", "")),
        runtime_role=str(entry.get("runtime_role", "")),
        implementation=str(entry.get("implementation", circuit.get("implementation", ""))),
        artifact_ref=artifact_ref,
        circuit=circuit,
        nodes=nodes,
        modules=modules,
    )


def _validate_component(component: ComponentView) -> None:
    node_ids = [str(node.get("id", "")) for node in component.nodes]
    if any(not node_id for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ModelCompileError(
            f"component {component.component_id!r} source node ids must be non-empty and unique"
        )
    module_ids = [str(module.get("id", "")) for module in component.modules]
    if any(not module_id for module_id in module_ids) or len(module_ids) != len(
        set(module_ids)
    ):
        raise ModelCompileError(
            f"component {component.component_id!r} semantic module ids must be non-empty and unique"
        )
    if not isinstance(component.circuit.get("parameters", {}).get("refs"), dict):
        raise ModelCompileError(
            f"component {component.component_id!r} has no parameter reference map"
        )


def _object_list(value: object, label: str) -> list[Json]:
    if not isinstance(value, (list, tuple)):
        raise ModelCompileError(f"{label} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ModelCompileError(f"{label} must contain only objects")
    return [dict(item) for item in value]


def _endpoint(value: Any) -> SignalKey | None:
    if not isinstance(value, dict):
        return None
    component_id = value.get("component_id")
    port_id = value.get("port_id")
    if not isinstance(component_id, str) or not isinstance(port_id, str):
        return None
    return component_id, port_id
