from __future__ import annotations

from nerve.compilation import Json, ModelCompileError


ADJACENT_REPRESENTATION_ROLE = "adjacent semantic representation boundary"


def is_adjacent_representation_scope(
    scope: Json,
    source_contract: Json,
) -> bool:
    return (
        scope.get("kind") == "representation_island"
        and len(scope.get("members", {}).get("component_ids", [])) == 1
        and source_contract.get("semantic_role")
        == ADJACENT_REPRESENTATION_ROLE
    )


def exact_adjacent_boundary_records(
    *,
    component_id: str,
    producer_node_id: str,
    consumer_node_ids: tuple[str, ...],
    scoped_contracts: tuple[tuple[Json, Json], ...],
    evidence_by_scope: dict[str, tuple[str, ...]],
    require_all: bool = True,
) -> tuple[tuple[Json, Json, tuple[str, ...]], ...] | None:
    """Resolve unambiguous catalog boundaries for exact producer/consumer edges.

    ``require_all=False`` is for a rewrite whose edges may remain inside one
    semantic module: it owns every catalogued inter-module boundary without
    inventing ceremonial scopes for internal edges.
    """
    if (
        not component_id
        or not producer_node_id
        or not consumer_node_ids
        or len(consumer_node_ids) != len(set(consumer_node_ids))
    ):
        raise ModelCompileError("adjacent representation boundary request is invalid")
    prefix = f"{component_id}/"
    indexed: dict[frozenset[str], list[tuple[Json, Json]]] = {}
    for scope, contract in scoped_contracts:
        source_ids = scope.get("members", {}).get("source_node_ids")
        if (
            not isinstance(source_ids, list)
            or len(source_ids) != 2
            or any(
                not isinstance(node_id, str) or not node_id.startswith(prefix)
                for node_id in source_ids
            )
        ):
            continue
        local_ids = frozenset(node_id.removeprefix(prefix) for node_id in source_ids)
        if len(local_ids) != 2:
            continue
        indexed.setdefault(local_ids, []).append((scope, contract))
    records = []
    for consumer_node_id in consumer_node_ids:
        matches = indexed.get(
            frozenset((producer_node_id, consumer_node_id)),
            [],
        )
        if len(matches) > 1:
            return None
        if not matches:
            if require_all:
                return None
            continue
        scope, contract = matches[0]
        scope_id = str(scope["scope_id"])
        evidence_ids = evidence_by_scope.get(scope_id, ())
        if not evidence_ids:
            return None
        records.append((scope, contract, evidence_ids))
    scope_ids = [str(scope["scope_id"]) for scope, _contract, _evidence in records]
    if len(scope_ids) != len(set(scope_ids)):
        raise ModelCompileError("adjacent representation boundary scopes overlap")
    return tuple(records)


__all__ = [
    "ADJACENT_REPRESENTATION_ROLE",
    "exact_adjacent_boundary_records",
    "is_adjacent_representation_scope",
]
