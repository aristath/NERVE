from __future__ import annotations

from nerve.model_transpiler_types import Json, ModelTranspileError


EXPERT_PROJECTION_ROLE_ALIASES = {
    "w1": "w1",
    "gate_proj": "w1",
    "w2": "w2",
    "down_proj": "w2",
    "w3": "w3",
    "up_proj": "w3",
}


def parse_independent_expert_projection_weight(
    tensor_name: str,
) -> tuple[str, int, str, str] | None:
    """Return the structural expert root, index, canonical role, and storage.

    Source checkpoints are allowed to spell an equivalent SwiGLU projection as
    either ``w1/w2/w3`` or ``gate_proj/down_proj/up_proj``.  Everything after
    this compiler boundary uses the canonical roles, so runtime execution does
    not depend on the source model family or tensor dialect.
    """

    parts = tensor_name.split(".")
    if (
        len(parts) < 5
        or parts[-4] != "experts"
        or parts[-1] not in {"weight", "weight_packed"}
    ):
        return None
    try:
        expert = int(parts[-3])
    except ValueError:
        return None
    if expert < 0:
        return None
    projection = EXPERT_PROJECTION_ROLE_ALIASES.get(parts[-2])
    if projection is None:
        return None
    root = ".".join(parts[:-4])
    if not root:
        return None
    return root, expert, projection, parts[-1]


def has_independent_sparse_experts(tensors: dict[str, Json], prefix: str) -> bool:
    expected_root = f"{prefix}."
    candidate_roots = {
        parsed[0]
        for name in tensors
        if (parsed := parse_independent_expert_projection_weight(name)) is not None
        and parsed[0].startswith(expected_root)
        and parsed[2] == "w1"
    }
    return any(
        not _has_aggregate_expert_representation(tensors, root)
        for root in candidate_roots
    )


def _has_aggregate_expert_representation(
    tensors: dict[str, Json], expert_root: str
) -> bool:
    """Whether the source/compiler already exposes one executable expert bank.

    An aggregate bank and independently addressable experts are two physical
    representations of the same semantic MoE.  Representation availability,
    not a model name, decides which discovery path owns the layer.
    """

    return all(
        name in tensors
        for name in (
            f"{expert_root}.experts.gate_up_proj",
            f"{expert_root}.experts.down_proj",
        )
    )


def discover_independent_sparse_experts(
    tensors: dict[str, Json],
    config: Json,
    *,
    prefix: str,
    hidden_size: int,
    vocab_size: int,
) -> tuple[dict[str, str], Json, int]:
    candidates: dict[str, dict[int, dict[str, str]]] = {}
    for name in tensors:
        parsed = parse_independent_expert_projection_weight(name)
        if parsed is None:
            continue
        root, expert, projection, _storage = parsed
        if not root.startswith(f"{prefix}.") or _has_aggregate_expert_representation(
            tensors, root
        ):
            continue
        found = candidates.setdefault(root, {}).setdefault(expert, {})
        if projection in found:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} has ambiguous {projection} projection "
                f"for independent expert {expert}"
            )
        found[projection] = name
    if len(candidates) != 1:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} must expose exactly one independently stored "
            f"expert block; found {sorted(candidates)}"
        )
    expert_root, projections = next(iter(candidates.items()))
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
    source_expert_formats: set[str] = set()
    for expert in expert_ids:
        found = projections[expert]
        if set(found) != {"w1", "w2", "w3"}:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} expert {expert} is incomplete: "
                f"found {sorted(found)}, expected ['w1', 'w2', 'w3']"
            )
        source_expert_formats.update(
            _source_weight_format(tensors[name]) for name in found.values()
        )
        w1_shape = _logical_weight_shape(tensors, found["w1"])
        w2_shape = _logical_weight_shape(tensors, found["w2"])
        w3_shape = _logical_weight_shape(tensors, found["w3"])
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
    if len(source_expert_formats) != 1:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} mixes routed expert representations: "
            + ", ".join(sorted(source_expert_formats))
        )
    source_expert_format = next(iter(source_expert_formats))
    router_candidates = [
        name
        for name in (
            f"{expert_root}.gate.weight",
            f"{expert_root}.router.weight",
        )
        if name in tensors
    ]
    if len(router_candidates) != 1:
        raise ModelTranspileError(
            f"layer prefix {prefix!r} must expose exactly one sparse expert router "
            f"weight; found {router_candidates}"
        )
    router = router_candidates[0]
    _require_shape(tensors, router, [len(expert_ids), hidden_size])
    parameters["moe_router"] = router
    router_root = router.removesuffix(".weight")
    route_table = f"{router_root}.tid2eid"
    selection_bias = f"{router_root}.bias"
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

    source_role_aliases = {
        canonical: tuple(
            source
            for source, normalized in EXPERT_PROJECTION_ROLE_ALIASES.items()
            if normalized == canonical
        )
        for canonical in ("w1", "w2", "w3")
    }
    shared_names: dict[str, str] = {}
    for projection, aliases in source_role_aliases.items():
        matches = [
            name
            for shared_root in (
                f"{expert_root}.shared_experts",
                f"{expert_root}.shared_expert",
            )
            for alias in aliases
            for storage in ("weight", "weight_packed")
            if (name := f"{shared_root}.{alias}.{storage}") in tensors
        ]
        if len(matches) != 1:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} must expose exactly one shared expert "
                f"{projection} projection; found {matches}"
            )
        shared_names[projection] = matches[0]
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


def _logical_weight_shape(tensors: dict[str, Json], name: str) -> list[int]:
    shape = [
        int(value)
        for value in tensors[name].get("logical_shape", tensors[name].get("shape", []))
    ]
    if len(shape) != 2:
        raise ModelTranspileError(f"expert tensor {name!r} must be a matrix")
    return shape


def _source_weight_format(info: Json) -> str:
    quantization = info.get("quantization")
    if isinstance(quantization, dict) and quantization.get("format"):
        return str(quantization["format"])
    return str(info.get("dtype") or "unknown")


def _attach_source_scale(
    tensors: dict[str, Json], parameters: dict[str, str], parameter_id: str, weight: str
) -> None:
    quantization = tensors[weight].get("quantization")
    declared_scale = (
        quantization.get("scales") if isinstance(quantization, dict) else None
    )
    scale = (
        str(declared_scale)
        if isinstance(declared_scale, str) and declared_scale
        else weight.removesuffix(".weight") + ".scale"
    )
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
