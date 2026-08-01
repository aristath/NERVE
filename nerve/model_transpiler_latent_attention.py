from __future__ import annotations

from nerve.model_transpiler_types import Json, ModelTranspileError


LATENT_ATTENTION_SUFFIXES = {
    "q_input_projection": "attn.wq_a.weight",
    "q_input_norm": "attn.q_norm.weight",
    "q_head_projection": "attn.wq_b.weight",
    "kv_projection": "attn.wkv.weight",
    "kv_norm": "attn.kv_norm.weight",
    "attention_sinks": "attn.attn_sink",
    "out_group_projection": "attn.wo_a.weight",
    "attention_out_projection": "attn.wo_b.weight",
}

COMPRESSOR_SUFFIXES = {
    "position_bias": "attn.compressor.ape",
    "kv_projection": "attn.compressor.wkv.weight",
    "gate_projection": "attn.compressor.wgate.weight",
    "norm": "attn.compressor.norm.weight",
}

INDEXER_SUFFIXES = {
    "q_projection": "attn.indexer.wq_b.weight",
    "head_weight_projection": "attn.indexer.weights_proj.weight",
    "compressor_position_bias": "attn.indexer.compressor.ape",
    "compressor_kv_projection": "attn.indexer.compressor.wkv.weight",
    "compressor_gate_projection": "attn.indexer.compressor.wgate.weight",
    "compressor_norm": "attn.indexer.compressor.norm.weight",
}


def has_latent_sparse_attention(tensors: dict[str, Json], prefix: str) -> bool:
    return f"{prefix}.{LATENT_ATTENTION_SUFFIXES['q_input_projection']}" in tensors


def discover_latent_sparse_attention(
    tensors: dict[str, Json],
    config: Json,
    *,
    prefix: str,
    layer_index: int,
    hidden_size: int,
    num_attention_heads: int,
    head_width: int,
    window_size: int | None,
) -> tuple[dict[str, str], Json]:
    parameters = _required_parameters(
        tensors,
        prefix,
        LATENT_ATTENTION_SUFFIXES,
        contract="latent sparse attention",
    )
    q_rank = int(
        config.get("q_lora_rank")
        or _shape(tensors, parameters["q_input_projection"])[0]
    )
    output_rank = int(config.get("o_lora_rank") or 0)
    output_groups = int(config.get("o_groups") or 1)
    rotary_width = int(config.get("qk_rope_head_dim") or head_width)
    if q_rank <= 0 or output_rank <= 0 or output_groups <= 0:
        raise ModelTranspileError(
            f"latent attention layer {prefix!r} has invalid low-rank dimensions"
        )
    if num_attention_heads % output_groups:
        raise ModelTranspileError(
            f"latent attention layer {prefix!r} output groups do not divide its heads"
        )
    if rotary_width <= 0 or rotary_width > head_width or rotary_width % 2:
        raise ModelTranspileError(
            f"latent attention layer {prefix!r} has invalid rotary width {rotary_width}"
        )
    _require_shape(tensors, parameters["q_input_projection"], [q_rank, hidden_size])
    _require_shape(tensors, parameters["q_input_norm"], [q_rank])
    _require_shape(
        tensors,
        parameters["q_head_projection"],
        [num_attention_heads * head_width, q_rank],
    )
    _require_shape(tensors, parameters["kv_projection"], [head_width, hidden_size])
    _require_shape(tensors, parameters["kv_norm"], [head_width])
    _require_shape(tensors, parameters["attention_sinks"], [num_attention_heads])
    grouped_input_width = num_attention_heads * head_width // output_groups
    _require_shape(
        tensors,
        parameters["out_group_projection"],
        [output_groups * output_rank, grouped_input_width],
    )
    _require_shape(
        tensors,
        parameters["attention_out_projection"],
        [hidden_size, output_groups * output_rank],
    )

    compression_ratio = _compression_ratio(config, layer_index)
    compression, compressor_parameters = _discover_compressor(
        tensors,
        prefix=prefix,
        ratio=compression_ratio,
        hidden_size=hidden_size,
        head_width=head_width,
    )
    parameters.update(compressor_parameters)
    indexer, indexer_parameters = _discover_indexer(
        tensors,
        config,
        prefix=prefix,
        compression_ratio=compression_ratio,
        hidden_size=hidden_size,
        q_rank=q_rank,
        rotary_width=rotary_width,
    )
    parameters.update(indexer_parameters)

    if window_size is None or window_size <= 0:
        raise ModelTranspileError(
            f"latent sparse attention layer {prefix!r} requires a positive local window"
        )
    base_theta = float(config.get("rope_theta") or 0.0)
    rope_theta = (
        float(config.get("compress_rope_theta") or 0.0)
        if compression is not None
        else base_theta
    )
    if rope_theta <= 0.0:
        raise ModelTranspileError(
            f"latent sparse attention layer {prefix!r} requires a positive RoPE theta"
        )
    rope_parameters: Json = {
        "rope_theta": rope_theta,
        "partial_rotary_factor": rotary_width / head_width,
        "rope_type": "default",
    }
    if compression is not None:
        scaling = config.get("rope_scaling")
        if isinstance(scaling, dict):
            rope_parameters.update(scaling)
            rope_parameters["rope_type"] = str(
                scaling.get("rope_type") or scaling.get("type") or "default"
            )
            rope_parameters["rope_theta"] = rope_theta

    return parameters, {
        "type": "latent_sparse_attention",
        "query_rank": q_rank,
        "output_rank": output_rank,
        "output_groups": output_groups,
        "window_size": window_size,
        "compression": compression,
        "indexer": indexer,
        "rotary_width": rotary_width,
        "rope_parameters": rope_parameters,
    }


def _compression_ratio(config: Json, layer_index: int) -> int:
    ratios = config.get("compress_ratios")
    if ratios is None:
        return 0
    if not isinstance(ratios, list) or layer_index >= len(ratios):
        raise ModelTranspileError(
            "compress_ratios must contain one entry for every decoder layer"
        )
    ratio = int(ratios[layer_index])
    if ratio < 0:
        raise ModelTranspileError("attention compression ratio cannot be negative")
    return ratio


def _discover_compressor(
    tensors: dict[str, Json],
    *,
    prefix: str,
    ratio: int,
    hidden_size: int,
    head_width: int,
) -> tuple[Json | None, dict[str, str]]:
    names = {role: f"{prefix}.{suffix}" for role, suffix in COMPRESSOR_SUFFIXES.items()}
    present = {role: name for role, name in names.items() if name in tensors}
    if ratio == 0:
        if present:
            raise ModelTranspileError(
                f"attention compressor tensors exist for uncompressed layer {prefix!r}"
            )
        return None, {}
    if present != names:
        raise ModelTranspileError(
            f"incomplete attention compressor tensor contract for {prefix!r}: "
            f"expected {sorted(names.values())}, found {sorted(present.values())}"
        )

    position_shape = _shape(tensors, names["position_bias"])
    if len(position_shape) != 2 or position_shape[0] != ratio:
        raise ModelTranspileError(
            f"attention compressor position bias for {prefix!r} has invalid shape {position_shape}"
        )
    if position_shape[1] % head_width:
        raise ModelTranspileError(
            f"attention compressor width for {prefix!r} is not a multiple of head width"
        )
    coefficient = position_shape[1] // head_width
    if coefficient not in {1, 2}:
        raise ModelTranspileError(
            f"attention compressor for {prefix!r} has unsupported lane coefficient {coefficient}"
        )
    projected_width = coefficient * head_width
    _require_shape(tensors, names["kv_projection"], [projected_width, hidden_size])
    _require_shape(tensors, names["gate_projection"], [projected_width, hidden_size])
    _require_shape(tensors, names["norm"], [head_width])
    return {
        "ratio": ratio,
        "overlap": coefficient == 2,
        "lane_coefficient": coefficient,
        "pooling": "learned_position_biased_softmax",
    }, {f"compressor_{role}": name for role, name in names.items()}


def _discover_indexer(
    tensors: dict[str, Json],
    config: Json,
    *,
    prefix: str,
    compression_ratio: int,
    hidden_size: int,
    q_rank: int,
    rotary_width: int,
) -> tuple[Json, dict[str, str]]:
    names = {role: f"{prefix}.{suffix}" for role, suffix in INDEXER_SUFFIXES.items()}
    present = {role: name for role, name in names.items() if name in tensors}
    if not present:
        return {
            "selection": (
                "chronological_compressed_positions" if compression_ratio else "none"
            )
        }, {}
    if compression_ratio == 0:
        raise ModelTranspileError(
            f"attention indexer tensors exist for uncompressed layer {prefix!r}"
        )
    if present != names:
        raise ModelTranspileError(
            f"incomplete attention indexer tensor contract for {prefix!r}: "
            f"expected {sorted(names.values())}, found {sorted(present.values())}"
        )

    index_heads = int(config.get("index_n_heads") or 0)
    index_head_width = int(config.get("index_head_dim") or 0)
    top_k = int(config.get("index_topk") or 0)
    if index_heads <= 0 or index_head_width <= 0 or top_k <= 0:
        raise ModelTranspileError(
            f"attention indexer for {prefix!r} has invalid dimensions"
        )
    if rotary_width > index_head_width:
        raise ModelTranspileError(
            f"attention indexer for {prefix!r} is narrower than the rotary width"
        )
    _require_shape(
        tensors,
        names["q_projection"],
        [index_heads * index_head_width, q_rank],
    )
    _require_shape(
        tensors,
        names["head_weight_projection"],
        [index_heads, hidden_size],
    )
    position_shape = _shape(tensors, names["compressor_position_bias"])
    if len(position_shape) != 2 or position_shape[0] != compression_ratio:
        raise ModelTranspileError(
            f"attention indexer compressor for {prefix!r} has invalid position shape"
        )
    if position_shape[1] % index_head_width:
        raise ModelTranspileError(
            f"attention indexer compressor for {prefix!r} has invalid width"
        )
    coefficient = position_shape[1] // index_head_width
    projected_width = coefficient * index_head_width
    _require_shape(
        tensors,
        names["compressor_kv_projection"],
        [projected_width, hidden_size],
    )
    _require_shape(
        tensors,
        names["compressor_gate_projection"],
        [projected_width, hidden_size],
    )
    _require_shape(tensors, names["compressor_norm"], [index_head_width])
    return {
        "selection": "learned_topk",
        "heads": index_heads,
        "head_width": index_head_width,
        "top_k": top_k,
        "compressor_lane_coefficient": coefficient,
        "activation_format": "packed_fp4_candidate",
        "rotation": "hadamard",
    }, {f"indexer_{role}": name for role, name in names.items()}


def _required_parameters(
    tensors: dict[str, Json],
    prefix: str,
    suffixes: dict[str, str],
    *,
    contract: str,
) -> dict[str, str]:
    names = {role: f"{prefix}.{suffix}" for role, suffix in suffixes.items()}
    present = {role: name for role, name in names.items() if name in tensors}
    if present != names:
        raise ModelTranspileError(
            f"incomplete {contract} tensor contract for {prefix!r}: "
            f"expected {sorted(names.values())}, found {sorted(present.values())}"
        )
    return names


def _shape(tensors: dict[str, Json], name: str) -> list[int]:
    return [
        int(value)
        for value in tensors[name].get("logical_shape", tensors[name].get("shape", []))
    ]


def _require_shape(tensors: dict[str, Json], name: str, expected: list[int]) -> None:
    actual = _shape(tensors, name)
    if actual != expected:
        raise ModelTranspileError(
            f"latent attention tensor {name!r} has shape {actual}, expected {expected}"
        )
