from __future__ import annotations

from dataclasses import dataclass, field

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import stable_contract_id
from nerve.representation_optimizer.scope_enumeration.graph import (
    ComponentView,
    NodeKey,
    SemanticDependencyGraph,
)


_PRIMARY_KIND_PRIORITY = (
    "feedback_transducer",
    "input_transducer",
    "output_transducer",
    "sampler",
    "stateful_system",
    "layer",
    "cross_layer_group",
    "representation_island",
    "coupled_region",
    "semantic_module",
    "operator",
)


@dataclass
class ScopeDraft:
    node_keys: set[NodeKey]
    classifications: set[str] = field(default_factory=set)
    semantic_module_ids: set[str] = field(default_factory=set)
    semantic_roles: set[str] = field(default_factory=set)

    @property
    def region_key(self) -> tuple[NodeKey, ...]:
        return tuple(sorted(self.node_keys))

    @property
    def primary_kind(self) -> str:
        for kind in _PRIMARY_KIND_PRIORITY:
            if kind in self.classifications:
                return kind
        raise ValueError(
            f"scope draft has no optimizer scope kind: {sorted(self.classifications)}"
        )

    def merge(self, other: ScopeDraft) -> None:
        self.classifications.update(other.classifications)
        self.semantic_module_ids.update(other.semantic_module_ids)
        self.semantic_roles.update(other.semantic_roles)


@dataclass
class ScopeDraftCollection:
    drafts: dict[tuple[NodeKey, ...], ScopeDraft] = field(default_factory=dict)
    diagnostics: list[Json] = field(default_factory=list)

    def add(self, draft: ScopeDraft) -> None:
        if not draft.node_keys:
            self.reject(
                classification=next(iter(draft.classifications), "unknown"),
                component_ids=[],
                semantic_module_ids=sorted(draft.semantic_module_ids),
                reason="semantic scope has no executable source nodes",
            )
            return
        draft.classifications.add(draft.primary_kind)
        existing = self.drafts.get(draft.region_key)
        if existing is None:
            self.drafts[draft.region_key] = draft
        else:
            existing.merge(draft)
            existing.classifications.add(existing.primary_kind)

    def reject(
        self,
        *,
        classification: str,
        component_ids: list[str],
        semantic_module_ids: list[str],
        reason: str,
    ) -> None:
        diagnostic = {
            "diagnostic_id": stable_contract_id(
                "scope_diagnostic",
                classification,
                component_ids,
                semantic_module_ids,
                reason,
            ),
            "classification": classification,
            "component_ids": component_ids,
            "semantic_module_ids": semantic_module_ids,
            "reason": reason,
        }
        if diagnostic["diagnostic_id"] not in {
            existing["diagnostic_id"] for existing in self.diagnostics
        }:
            self.diagnostics.append(diagnostic)


def enumerate_scope_drafts(graph: SemanticDependencyGraph) -> ScopeDraftCollection:
    collection = ScopeDraftCollection()
    leaf_owners = _leaf_module_owners(graph, collection)

    for component in graph.components:
        _add_operator_scopes(collection, component)
        _add_semantic_module_scopes(collection, component, leaf_owners)
        _add_layer_and_region_scopes(
            collection,
            component,
            leaf_owners,
        )
        _add_stateful_scopes(collection, component)
        _add_transducer_scopes(collection, component)
        _add_internal_islands(collection, component, leaf_owners)

    _add_cross_layer_groups(collection, graph, leaf_owners)
    _add_component_islands_and_feedback(collection, graph)
    collection.diagnostics.sort(key=lambda item: str(item["diagnostic_id"]))
    return collection


def _add_operator_scopes(
    collection: ScopeDraftCollection,
    component: ComponentView,
) -> None:
    for node in component.nodes:
        node_id = str(node["id"])
        collection.add(
            ScopeDraft(
                node_keys={(component.component_id, node_id)},
                classifications={"operator"},
                semantic_roles={str(node.get("op", "operator"))},
            )
        )


def _leaf_module_owners(
    graph: SemanticDependencyGraph,
    collection: ScopeDraftCollection,
) -> dict[NodeKey, str | None]:
    result: dict[NodeKey, str | None] = {}
    for component in graph.components:
        for module in component.modules:
            if module.get("child_ids"):
                continue
            qualified_module = _qualified_module(component, module)
            if module.get("virtual"):
                collection.reject(
                    classification="semantic_module",
                    component_ids=[component.component_id],
                    semantic_module_ids=[qualified_module],
                    reason=(
                        "virtual semantic module shares a physical implementation "
                        "and has no independently owned executable dependency boundary"
                    ),
                )
                continue
            for node_id in module.get("source_node_ids", []):
                key = (component.component_id, str(node_id))
                if key in result:
                    result[key] = None
                else:
                    result[key] = qualified_module
    return result


def _add_semantic_module_scopes(
    collection: ScopeDraftCollection,
    component: ComponentView,
    leaf_owners: dict[NodeKey, str | None],
) -> None:
    for module in component.modules:
        if module.get("child_ids") or module.get("virtual"):
            continue
        module_id = str(module["id"])
        qualified_module = _qualified_module(component, module)
        node_ids = tuple(str(value) for value in module.get("source_node_ids", []))
        node_keys = set(component.qualified_nodes(node_ids))
        if not node_keys:
            collection.reject(
                classification="semantic_module",
                component_ids=[component.component_id],
                semantic_module_ids=[qualified_module],
                reason="semantic leaf module has no independently executable source nodes",
            )
            continue
        if any(leaf_owners.get(key) != qualified_module for key in node_keys):
            collection.reject(
                classification="semantic_module",
                component_ids=[component.component_id],
                semantic_module_ids=[qualified_module],
                reason="semantic source-node ownership is ambiguous across leaf modules",
            )
            continue
        collection.add(
            ScopeDraft(
                node_keys=node_keys,
                classifications={"semantic_leaf_module", "semantic_module"},
                semantic_module_ids={qualified_module},
                semantic_roles={
                    str(module.get("role", "semantic_module")),
                    str(module.get("responsibility", "")),
                }
                - {""},
            )
        )


def _add_layer_and_region_scopes(
    collection: ScopeDraftCollection,
    component: ComponentView,
    leaf_owners: dict[NodeKey, str | None],
) -> None:
    by_id = component.module_by_id
    if not by_id:
        return
    root_id = component.circuit["semantic_module_tree"].get("root_module_id")
    if isinstance(root_id, str) and root_id in by_id:
        root = by_id[root_id]
        _add_owned_semantic_region(
            collection,
            component=component,
            module=root,
            node_keys=set(
                component.qualified_nodes(component.subtree_node_ids(root_id))
            ),
            classifications={"layer"},
            semantic_roles={
                str(root.get("role", "layer")),
                str(root.get("responsibility", "")),
            }
            - {""},
            leaf_owners=leaf_owners,
        )

    for module in component.modules:
        role = str(module.get("role", ""))
        module_id = str(module["id"])
        child_ids = [
            str(value)
            for value in module.get("child_ids", [])
            if not component.module_by_id[str(value)].get("virtual")
        ]
        is_named_region = role in {"token_mixer", "feature_transform"}
        is_coupled_sibling_region = len(child_ids) >= 2
        if not is_named_region and not is_coupled_sibling_region:
            continue
        nodes = set(
            component.qualified_nodes(component.subtree_node_ids(module_id))
        )
        classifications = {"coupled_region"}
        if is_named_region:
            classifications.add(f"{role}_region")
        if is_coupled_sibling_region:
            classifications.add("coupled_sibling_operations")
        if not nodes:
            collection.reject(
                classification="coupled_region",
                component_ids=[component.component_id],
                semantic_module_ids=[_qualified_module(component, module)],
                reason=f"{role} region has no executable source nodes",
            )
            continue
        _add_owned_semantic_region(
            collection,
            component=component,
            module=module,
            node_keys=nodes,
            classifications=classifications,
            semantic_roles={
                role,
                str(module.get("responsibility", "")),
            }
            - {""},
            leaf_owners=leaf_owners,
        )


def _add_owned_semantic_region(
    collection: ScopeDraftCollection,
    *,
    component: ComponentView,
    module: Json,
    node_keys: set[NodeKey],
    classifications: set[str],
    semantic_roles: set[str],
    leaf_owners: dict[NodeKey, str | None],
) -> None:
    if any(leaf_owners.get(node_key) is None for node_key in node_keys):
        collection.reject(
            classification=next(iter(sorted(classifications))),
            component_ids=[component.component_id],
            semantic_module_ids=[_qualified_module(component, module)],
            reason=(
                "semantic region contains a source node without one unambiguous "
                "leaf-module owner"
            ),
        )
        return
    collection.add(
        ScopeDraft(
            node_keys=node_keys,
            classifications=classifications,
            semantic_module_ids={_qualified_module(component, module)},
            semantic_roles=semantic_roles,
        )
    )


def _add_stateful_scopes(
    collection: ScopeDraftCollection,
    component: ComponentView,
) -> None:
    for state in component.circuit.get("state_ports", []):
        state_id = str(state["id"])
        node_keys = {
            (component.component_id, str(node["id"]))
            for node in component.nodes
            if state_id
            in {
                *map(str, node.get("state_reads", [])),
                *map(str, node.get("state_writes", [])),
            }
        }
        if not node_keys:
            collection.reject(
                classification="stateful_system",
                component_ids=[component.component_id],
                semantic_module_ids=[],
                reason=f"state {state_id!r} has no source writer or reader",
            )
            continue
        collection.add(
            ScopeDraft(
                node_keys=node_keys,
                classifications={"state_writer_reader_system", "stateful_system"},
                semantic_roles={
                    f"state:{state_id}",
                    str(state.get("type", "state")),
                },
            )
        )


def _add_transducer_scopes(
    collection: ScopeDraftCollection,
    component: ComponentView,
) -> None:
    kind = {
        "input_transducer": "input_transducer",
        "output_transducer": "output_transducer",
        "sampler": "sampler",
    }.get(component.runtime_role)
    if kind is None:
        return
    collection.add(
        ScopeDraft(
            node_keys=set(
                (component.component_id, str(node["id"]))
                for node in component.nodes
            ),
            classifications={kind, "whole_transducer"},
            semantic_roles={component.runtime_role, component.operator_type},
        )
    )


def _add_internal_islands(
    collection: ScopeDraftCollection,
    component: ComponentView,
    leaf_owners: dict[NodeKey, str | None],
) -> None:
    producer_by_signal = {
        str(output): (component.component_id, str(node["id"]))
        for node in component.nodes
        for output in node.get("outputs", [])
    }
    for consumer in component.nodes:
        consumer_key = (component.component_id, str(consumer["id"]))
        consumer_module = leaf_owners.get(consumer_key)
        if consumer_module is None:
            continue
        for signal in consumer.get("inputs", []):
            producer_key = producer_by_signal.get(str(signal))
            if producer_key is None:
                continue
            producer_module = leaf_owners.get(producer_key)
            if producer_module is None or producer_module == consumer_module:
                continue
            collection.add(
                ScopeDraft(
                    node_keys={producer_key, consumer_key},
                    classifications={
                        "adjacent_producer_consumer",
                        "representation_island",
                    },
                    semantic_module_ids={producer_module, consumer_module},
                    semantic_roles={"adjacent semantic representation boundary"},
                )
            )


def _add_cross_layer_groups(
    collection: ScopeDraftCollection,
    graph: SemanticDependencyGraph,
    leaf_owners: dict[NodeKey, str | None],
) -> None:
    groups: dict[tuple[str, str], list[tuple[ComponentView, Json]]] = {}
    for component in graph.components:
        if component.runtime_role != "signal_processor":
            continue
        root_id = component.circuit.get("semantic_module_tree", {}).get(
            "root_module_id"
        )
        for module in component.modules:
            module_id = str(module["id"])
            if module_id == root_id or module.get("virtual"):
                continue
            groups.setdefault(
                (module_id, str(module.get("role", ""))),
                [],
            ).append((component, module))
    for (_module_id, role), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        node_keys = set()
        semantic_ids = set()
        semantic_roles = {role} - {""}
        for component, module in members:
            module_nodes = component.subtree_node_ids(str(module["id"]))
            node_keys.update(component.qualified_nodes(module_nodes))
            semantic_ids.add(_qualified_module(component, module))
            responsibility = str(module.get("responsibility", ""))
            if responsibility:
                semantic_roles.add(responsibility)
        if not node_keys:
            continue
        if any(leaf_owners.get(node_key) is None for node_key in node_keys):
            collection.reject(
                classification="cross_layer_group",
                component_ids=[component.component_id for component, _ in members],
                semantic_module_ids=sorted(semantic_ids),
                reason=(
                    "cross-layer semantic group contains a source node without "
                    "one unambiguous leaf-module owner"
                ),
            )
            continue
        collection.add(
            ScopeDraft(
                node_keys=node_keys,
                classifications={"cross_layer_group", "repeated_corresponding_modules"},
                semantic_module_ids=semantic_ids,
                semantic_roles=semantic_roles,
            )
        )


def _add_component_islands_and_feedback(
    collection: ScopeDraftCollection,
    graph: SemanticDependencyGraph,
) -> None:
    for edge in graph.edges:
        source = edge["source"]
        destination = edge["destination"]
        source_id = str(source["component_id"])
        destination_id = str(destination["component_id"])
        source_signal = (source_id, str(source["port_id"]))
        destination_signal = (destination_id, str(destination["port_id"]))
        producer = graph.local_signal_producer(source_signal)
        consumers = graph.signal_consumers(destination_signal)
        node_keys = ({producer} if producer is not None else set()) | set(consumers)
        connection = edge.get("connection", {})
        classification = (
            "feedback_transducer"
            if connection.get("kind") == "temporal_feedback"
            else "representation_island"
        )
        if producer is None or not consumers:
            collection.reject(
                classification=classification,
                component_ids=[source_id, destination_id],
                semantic_module_ids=[],
                reason=(
                    "component connection has no unambiguous executable "
                    "producer/consumer boundary"
                ),
            )
            continue
        if connection.get("kind") == "temporal_feedback":
            collection.add(
                ScopeDraft(
                    node_keys=node_keys,
                    classifications={"feedback_transducer"},
                    semantic_roles={"generation feedback with temporal delay"},
                )
            )
        else:
            collection.add(
                ScopeDraft(
                    node_keys=node_keys,
                    classifications={
                        "adjacent_producer_consumer",
                        "representation_island",
                    },
                    semantic_roles={"adjacent component representation boundary"},
                )
            )


def _qualified_module(component: ComponentView, module: Json) -> str:
    return f"{component.component_id}/{module['id']}"
