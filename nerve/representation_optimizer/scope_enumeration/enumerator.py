from __future__ import annotations

from collections import Counter
from copy import deepcopy

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    OPTIMIZATION_SCOPE_SCHEMA,
    SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    optimization_scope_catalog_id,
    source_behavior_contract_digest,
    stable_contract_id,
    validate_contract,
)
from nerve.representation_optimizer.scope_enumeration.boundaries import (
    derive_scope_boundary,
)
from nerve.representation_optimizer.scope_enumeration.drafts import (
    ScopeDraft,
    enumerate_scope_drafts,
)
from nerve.representation_optimizer.scope_enumeration.graph import (
    NodeKey,
    SemanticDependencyGraph,
)


def enumerate_optimization_scope_catalog(
    *,
    package_id: str,
    graph: SemanticDependencyGraph,
) -> Json:
    collection = enumerate_scope_drafts(graph)
    scopes = []
    source_contracts = []
    for draft in sorted(
        collection.drafts.values(),
        key=lambda item: _ordered_node_keys(graph, item),
    ):
        scope, source_contract = _materialize_scope(
            package_id=package_id,
            graph=graph,
            draft=draft,
        )
        scopes.append(scope)
        source_contracts.append(source_contract)
    order = sorted(
        range(len(scopes)),
        key=lambda index: str(scopes[index]["scope_id"]),
    )
    scopes = [scopes[index] for index in order]
    source_contracts = [source_contracts[index] for index in order]

    classification_counts = Counter(
        classification
        for scope in scopes
        for classification in scope["extensions"]["classifications"]
    )
    document: Json = {
        "schema": OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
        "catalog_id": "",
        "package_id": package_id,
        "scopes": scopes,
        "source_contracts": source_contracts,
        "diagnostics": collection.diagnostics,
        "summary": {
            "scope_count": len(scopes),
            "source_contract_count": len(source_contracts),
            "rejected_scope_count": len(collection.diagnostics),
            "classification_counts": dict(sorted(classification_counts.items())),
        },
    }
    document["catalog_id"] = optimization_scope_catalog_id(document)
    validate_contract(
        document,
        expected_schema=OPTIMIZATION_SCOPE_CATALOG_SCHEMA,
    )
    return document


def _materialize_scope(
    *,
    package_id: str,
    graph: SemanticDependencyGraph,
    draft: ScopeDraft,
) -> tuple[Json, Json]:
    ordered_nodes = _ordered_node_keys(graph, draft)
    component_ids = _ordered_component_ids(graph, ordered_nodes)
    source_node_ids = [
        f"{component_id}/{node_id}"
        for component_id, node_id in ordered_nodes
    ]
    semantic_module_ids = _ordered_semantic_module_ids(
        graph,
        draft.semantic_module_ids,
    )
    kind = draft.primary_kind
    classifications = sorted({*draft.classifications, kind})
    boundary = derive_scope_boundary(graph, set(ordered_nodes))
    region_id = stable_contract_id(
        "semantic_region",
        package_id,
        source_node_ids,
    )
    scope_id = stable_contract_id(
        "scope",
        package_id,
        kind,
        component_ids,
        semantic_module_ids,
        source_node_ids,
    )
    artifact_refs = (
        [graph.graph_artifact_ref]
        if graph.graph_artifact_ref is not None
        else []
    ) + [
        graph.component_by_id[component_id].artifact_ref
        for component_id in component_ids
    ]
    artifact_refs = list(dict.fromkeys(artifact_refs))
    semantic_roles = sorted(draft.semantic_roles) or [kind]
    source_contract: Json = {
        "schema": SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
        "scope_id": scope_id,
        "semantic_role": " | ".join(semantic_roles),
        "interface": deepcopy(boundary),
        "exact_reference": {
            "implementation_id": stable_contract_id(
                "exact_implementation",
                package_id,
                region_id,
                artifact_refs,
            ),
            "artifact_refs": artifact_refs,
        },
        "contract_digest": "",
    }
    source_contract["contract_digest"] = source_behavior_contract_digest(
        source_contract
    )
    scope: Json = {
        "schema": OPTIMIZATION_SCOPE_SCHEMA,
        "scope_id": scope_id,
        "package_id": package_id,
        "kind": kind,
        "members": {
            "component_ids": component_ids,
            "semantic_module_ids": semantic_module_ids,
            "source_node_ids": source_node_ids,
        },
        "boundary": deepcopy(boundary),
        "source_contract_digest": source_contract["contract_digest"],
        "extensions": {
            "classifications": classifications,
            "region_id": region_id,
            "semantic_roles": semantic_roles,
        },
    }
    validate_contract(scope, expected_schema=OPTIMIZATION_SCOPE_SCHEMA)
    validate_contract(
        source_contract,
        expected_schema=SOURCE_BEHAVIOR_CONTRACT_SCHEMA,
    )
    return scope, source_contract


def _ordered_node_keys(
    graph: SemanticDependencyGraph,
    draft: ScopeDraft,
) -> tuple[NodeKey, ...]:
    component_order = graph.component_order
    node_order = {
        (component.component_id, str(node["id"])): node_index
        for component in graph.components
        for node_index, node in enumerate(component.nodes)
    }
    return tuple(
        sorted(
            draft.node_keys,
            key=lambda key: (component_order[key[0]], node_order[key]),
        )
    )


def _ordered_component_ids(
    graph: SemanticDependencyGraph,
    node_keys: tuple[NodeKey, ...],
) -> list[str]:
    present = {component_id for component_id, _node_id in node_keys}
    return [
        component.component_id
        for component in graph.components
        if component.component_id in present
    ]


def _ordered_semantic_module_ids(
    graph: SemanticDependencyGraph,
    qualified_ids: set[str],
) -> list[str]:
    order = {
        f"{component.component_id}/{module['id']}": (
            component_index,
            module_index,
        )
        for component_index, component in enumerate(graph.components)
        for module_index, module in enumerate(component.modules)
    }
    return sorted(
        qualified_ids,
        key=lambda value: order.get(value, (len(graph.components), value)),
    )
