from __future__ import annotations

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.providers.adjacent_boundaries import (
    exact_adjacent_boundary_records,
)


def _record(consumer: str, index: int = 0):
    scope_id = f"scope_{consumer}_{index}"
    scope = {
        "scope_id": scope_id,
        "members": {
            "source_node_ids": [
                "component/producer",
                f"component/{consumer}",
            ]
        },
    }
    contract = {"scope_id": scope_id, "contract_digest": f"digest_{consumer}_{index}"}
    return scope, contract


def _resolve(records, *, require_all: bool = True):
    return exact_adjacent_boundary_records(
        component_id="component",
        producer_node_id="producer",
        consumer_node_ids=("left", "right"),
        scoped_contracts=tuple(records),
        evidence_by_scope={
            scope["scope_id"]: (f"evidence_{scope['scope_id']}",)
            for scope, _contract in records
        },
        require_all=require_all,
    )


def test_resolver_preserves_requested_edge_order() -> None:
    records = (_record("right"), _record("left"))

    resolved = _resolve(records)

    assert resolved is not None
    assert [scope["scope_id"] for scope, _contract, _evidence in resolved] == [
        "scope_left_0",
        "scope_right_0",
    ]


def test_resolver_owns_only_catalogued_inter_module_edges_when_requested() -> None:
    resolved = _resolve((_record("left"),), require_all=False)

    assert resolved is not None
    assert [scope["scope_id"] for scope, _contract, _evidence in resolved] == [
        "scope_left_0"
    ]


def test_resolver_fails_closed_when_a_required_edge_is_missing() -> None:
    assert _resolve((_record("left"),)) is None


def test_resolver_fails_closed_on_ambiguous_edge_or_missing_evidence() -> None:
    assert _resolve((_record("left", 0), _record("left", 1)), require_all=False) is None

    left = _record("left")
    right = _record("right")
    assert (
        exact_adjacent_boundary_records(
            component_id="component",
            producer_node_id="producer",
            consumer_node_ids=("left", "right"),
            scoped_contracts=(left, right),
            evidence_by_scope={left[0]["scope_id"]: ("evidence",)},
        )
        is None
    )


def test_resolver_rejects_duplicate_consumer_requests() -> None:
    with pytest.raises(ModelCompileError, match="request is invalid"):
        exact_adjacent_boundary_records(
            component_id="component",
            producer_node_id="producer",
            consumer_node_ids=("left", "left"),
            scoped_contracts=(_record("left"),),
            evidence_by_scope={},
        )
