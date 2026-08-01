from __future__ import annotations

import re

from nerve.model_transpiler_types import Json, ModelTranspileError


EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>.+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.weight$"
)


def has_independent_sparse_experts(tensors: dict[str, Json], prefix: str) -> bool:
    expected_prefix = f"{prefix}.ffn.experts."
    return any(
        name.startswith(expected_prefix) and name.endswith(".w1.weight")
        for name in tensors
    )


def discover_independent_sparse_experts(
    tensors: dict[str, Json],
    config: Json,
    *,
    prefix: str,
    hidden_size: int,
    vocab_size: int,
) -> tuple[dict[str, str], Json, int]:
    projections: dict[int, dict[str, str]] = {}
    for name in tensors:
        match = EXPERT_WEIGHT_PATTERN.fullmatch(name)
        if match is None or match.group("prefix") != prefix:
            continue
        projections.setdefault(int(match.group("expert")), {})[
            match.group("projection")
        ] = name
    expert_ids = sorted(projections)
    if not expert_ids:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has no independently stored experts"
        )
    if expert_ids != list(range(len(expert_ids))):
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has non-contiguous independent expert ids"
        )
    configured_count = int(
        config.get("n_routed_experts")
        or config.get("num_local_experts")
        or config.get("num_experts")
        or 0
    )
    if configured_count and configured_count != len(expert_ids):
        raise ModelTranspileError(
            f"layer prefix {prefix!r} exposes {len(expert_ids)} experts but config declares {configured_count}"
        )

    parameters: dict[str, str] = {}
    intermediate_size: int | None = None
    source_expert_format = str(config.get("expert_dtype") or "native").lower()
    for expert in expert_ids:
        found = projections[expert]
        if set(found) != {"w1", "w2", "w3"}:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} expert {expert} is incomplete: "
                f"found {sorted(found)}, expected ['w1', 'w2', 'w3']"
            )
        w1_shape = _logical_weight_shape(
            tensors, found["w1"], source_expert_format=source_expert_format
        )
        w2_shape = _logical_weight_shape(
            tensors, found["w2"], source_expert_format=source_expert_format
        )
        w3_shape = _logical_weight_shape(
            tensors, found["w3"], source_expert_format=source_expert_format
        )
        expert_intermediate = w1_shape[0]
        if (
            w1_shape != [expert_intermediate, hidden_size]
            or w3_shape != w1_shape
            or w2_shape != [hidden_size, expert_intermediate]
        ):
            raise ModelTranspileError(
                f"layer prefix {prefix!r} expert {expert} has incompatible projection shapes "
                f"w1={w1_shape}, w2={w2_shape}, w3={w3_shape}"
            )
        if intermediate_size is None:
            intermediate_size = expert_intermediate
        elif intermediate_size != expert_intermediate:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} mixes expert intermediate widths"
            )
        for projection, name in found.items():
            parameter_id = f"routed_expert_{expert:03d}_{projection}"
            parameters[parameter_id] = name
            _attach_source_scale(tensors, parameters, parameter_id, name)

    assert intermediate_size is not None
    router = f"{prefix}.ffn.gate.weight"
    if router not in tensors:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has no sparse expert router weight"
        )
    _require_shape(tensors, router, [len(expert_ids), hidden_size])
    parameters["moe_router"] = router
    route_table = f"{prefix}.ffn.gate.tid2eid"
    selection_bias = f"{prefix}.ffn.gate.bias"
    has_route_table = route_table in tensors
    has_selection_bias = selection_bias in tensors
    if has_route_table == has_selection_bias:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has ambiguous expert selection: exactly one of "
            "a token-id route table or score-selection bias is required"
        )
    experts_per_token = int(
        config.get("num_experts_per_tok")
        or config.get("n_activated_experts")
        or config.get("experts_per_token")
        or 0
    )
    if experts_per_token <= 0 or experts_per_token > len(expert_ids):
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has invalid experts-per-token {experts_per_token}"
        )
    if has_route_table:
        _require_shape(tensors, route_table, [vocab_size, experts_per_token])
        if tensors[route_table].get("dtype") not in {"I32", "I64"}:
            raise ModelTranspileError("expert token-id route table must be integer")
        parameters["moe_route_table"] = route_table
        selection = "token_id_table"
    else:
        _require_shape(tensors, selection_bias, [len(expert_ids)])
        parameters["moe_router_selection_bias"] = selection_bias
        selection = "score_topk"

    shared_names = {
        projection: f"{prefix}.ffn.shared_experts.{projection}.weight"
        for projection in ("w1", "w2", "w3")
    }
    present_shared = {
        projection: name for projection, name in shared_names.items() if name in tensors
    }
    if present_shared != shared_names:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} has an incomplete shared expert"
        )
    _require_logical_shape(
        tensors, shared_names["w1"], [intermediate_size, hidden_size]
    )
    _require_logical_shape(
        tensors, shared_names["w2"], [hidden_size, intermediate_size]
    )
    _require_logical_shape(
        tensors, shared_names["w3"], [intermediate_size, hidden_size]
    )
    for projection, name in shared_names.items():
        parameter_id = f"shared_expert_{projection}"
        parameters[parameter_id] = name
        _attach_source_scale(tensors, parameters, parameter_id, name)

    activation = str(
        config.get("moe_router_activation") or config.get("scoring_func") or "softmax"
    ).lower()
    if activation not in {"softmax", "sigmoid", "sqrtsoftplus"}:
        raise ModelTranspileError(
            f"unsupported independent expert router activation {activation!r}"
        )
    routing: Json = {
        "selection": selection,
        "activation": activation,
        "normalize_selected": bool(config.get("norm_topk_prob", True)),
        "routed_scaling_factor": float(
            config.get("routed_scaling_factor") or config.get("route_scale") or 1.0
        ),
    }
    if has_route_table:
        routing["route_table"] = route_table
    else:
        routing["selection_bias"] = selection_bias
    return (
        parameters,
        {
            "expert_storage": "independent_resources",
            "source_expert_format": source_expert_format,
            "expert_ids": expert_ids,
            "experts_per_token": experts_per_token,
            "routing": routing,
            "shared_expert_count": 1,
            "swiglu_limit": float(config.get("swiglu_limit") or 0.0),
        },
        intermediate_size,
    )


def _logical_weight_shape(
    tensors: dict[str, Json], name: str, *, source_expert_format: str
) -> list[int]:
    shape = _shape(tensors, name)
    if len(shape) != 2:
        raise ModelTranspileError(f"expert tensor {name!r} must be a matrix")
    if source_expert_format == "fp4" and tensors[name].get("dtype") == "I8":
        return [shape[0], shape[1] * 2]
    return shape


def _attach_source_scale(
    tensors: dict[str, Json], parameters: dict[str, str], parameter_id: str, weight: str
) -> None:
    scale = weight.removesuffix(".weight") + ".scale"
    if scale in tensors:
        parameters[f"{parameter_id}_scale"] = scale
    elif tensors[weight].get("dtype") in {"I8", "F8_E4M3"}:
        raise ModelTranspileError(
            f"quantized expert tensor {weight!r} is missing its source scale"
        )


def _shape(tensors: dict[str, Json], name: str) -> list[int]:
    return [int(value) for value in tensors[name].get("shape", [])]


def _require_shape(tensors: dict[str, Json], name: str, expected: list[int]) -> None:
    actual = _shape(tensors, name)
    if actual != expected:
        raise ModelTranspileError(
            f"sparse expert tensor {name!r} has shape {actual}, expected {expected}"
        )


def _require_logical_shape(
    tensors: dict[str, Json], name: str, expected: list[int]
) -> None:
    actual = [
        int(value)
        for value in tensors[name].get("logical_shape", tensors[name].get("shape", []))
    ]
    if actual != expected:
        raise ModelTranspileError(
            f"shared expert tensor {name!r} has shape {actual}, expected {expected}"
        )
