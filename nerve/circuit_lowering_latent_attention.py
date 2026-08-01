from __future__ import annotations

from nerve.circuit_lowering_nodes import _ffn_tail
from nerve.circuit_lowering_helpers import _linear_params, _norm_attrs
from nerve.circuit_lowering_sparse_experts import independent_sparse_moe_body
from nerve.model_transpiler_types import Json


def latent_sparse_attention_nodes(
    component: Json,
    *,
    parameters: Json,
) -> list[Json]:
    numerics = component["numerics"]
    operator = component["operator"]
    attributes = operator["attributes"]
    heads = operator["heads"]
    residual_mixer = component.get("residual_mixer")
    operator_input = "input_frame"
    nodes: list[Json] = []
    if residual_mixer is not None:
        nodes.extend(
            _hyper_connection_pre_nodes(
                stage="attention",
                input_signal="input_frame",
                output_signal="operator_input",
                mixer=residual_mixer,
                normalization_epsilon=float(numerics["rms_norm_eps"]),
            )
        )
        operator_input = "operator_input"
    nodes.extend(
        [
            {
                "id": "operator_norm",
                "op": "rms_norm",
                "inputs": [operator_input],
                "outputs": ["operator_norm_out"],
                "params": ["operator_norm"],
                "attrs": _norm_attrs(numerics),
            },
            {
                "id": "query_input_projection",
                "op": "linear",
                "inputs": ["operator_norm_out"],
                "outputs": ["query_rank"],
                "params": _linear_params("q_input_projection", parameters),
            },
            {
                "id": "query_input_norm",
                "op": "rms_norm",
                "inputs": ["query_rank"],
                "outputs": ["query_rank_normed"],
                "params": ["q_input_norm"],
                "attrs": _norm_attrs(numerics),
            },
            {
                "id": "query_head_projection",
                "op": "linear",
                "inputs": ["query_rank_normed"],
                "outputs": ["query_heads"],
                "params": _linear_params("q_head_projection", parameters),
            },
            {
                "id": "query_head_norm",
                "op": "rms_norm_per_head_unscaled",
                "inputs": ["query_heads"],
                "outputs": ["query_heads_normed"],
                "attrs": {
                    **_norm_attrs(numerics),
                    **heads,
                    "head_count": int(heads["query_heads"]),
                },
            },
            {
                "id": "query_rope",
                "op": "rotary_position_embedding",
                "inputs": ["query_heads_normed"],
                "outputs": ["query_positioned"],
                "attrs": _rope_attrs(
                    numerics, heads, head_count=int(heads["query_heads"])
                ),
            },
            {
                "id": "key_value_projection",
                "op": "linear",
                "inputs": ["operator_norm_out"],
                "outputs": ["key_value_latent"],
                "params": _linear_params("kv_projection", parameters),
            },
            {
                "id": "key_value_norm",
                "op": "rms_norm",
                "inputs": ["key_value_latent"],
                "outputs": ["key_value_normed"],
                "params": ["kv_norm"],
                "attrs": _norm_attrs(numerics),
            },
            {
                "id": "key_value_rope",
                "op": "rotary_position_embedding",
                "inputs": ["key_value_normed"],
                "outputs": ["key_value_positioned"],
                "attrs": _rope_attrs(
                    numerics, heads, head_count=int(heads["key_value_heads"])
                ),
            },
            {
                "id": "local_memory_update",
                "op": "rolling_state_update",
                "inputs": ["key_value_positioned", "local_kv_memory"],
                "outputs": ["local_kv_values"],
                "state_reads": ["local_kv_memory"],
                "state_writes": ["local_kv_memory"],
                "attrs": {
                    "update": "ring_append",
                    "capacity": int(attributes["window_size"]),
                },
            },
        ]
    )

    execution_contract = component.get("execution_contract") or {}
    parallel_context = (
        execution_contract.get("type") == "parallel_query_with_external_kv_context"
    )
    if parallel_context:
        by_id = {node["id"]: node for node in nodes}
        by_id["query_rope"]["attrs"].update(
            {
                "position_offset": int(execution_contract["query_position_offset"]),
                "position_mode": "parallel_block",
            }
        )
        by_id["key_value_projection"].update(
            {
                "inputs": ["main_context"],
                "outputs": ["context_key_value_latent"],
            }
        )
        by_id["key_value_norm"].update(
            {
                "inputs": ["context_key_value_latent"],
                "outputs": ["context_key_value_normed"],
            }
        )
        by_id["key_value_rope"].update(
            {
                "inputs": ["context_key_value_normed"],
                "outputs": ["context_key_value_positioned"],
            }
        )
        by_id["key_value_rope"]["attrs"].update(
            {"position_offset": 0, "position_mode": "committed_context"}
        )
        by_id["local_memory_update"].update(
            {
                "inputs": ["context_key_value_positioned", "local_kv_memory"],
                "attrs": {
                    **by_id["local_memory_update"]["attrs"],
                    "source": "committed_target_context",
                },
            }
        )
        nodes.extend(
            [
                {
                    "id": "query_key_value_projection",
                    "op": "linear",
                    "inputs": ["operator_norm_out"],
                    "outputs": ["query_key_value_latent"],
                    "params": _linear_params("kv_projection", parameters),
                },
                {
                    "id": "query_key_value_norm",
                    "op": "rms_norm",
                    "inputs": ["query_key_value_latent"],
                    "outputs": ["query_key_value_normed"],
                    "params": ["kv_norm"],
                    "attrs": _norm_attrs(numerics),
                },
                {
                    "id": "query_key_value_rope",
                    "op": "rotary_position_embedding",
                    "inputs": ["query_key_value_normed"],
                    "outputs": ["query_key_value_positioned"],
                    "attrs": {
                        **_rope_attrs(
                            numerics,
                            heads,
                            head_count=int(heads["key_value_heads"]),
                        ),
                        "position_offset": int(
                            execution_contract["query_position_offset"]
                        ),
                        "position_mode": "parallel_block",
                    },
                },
            ]
        )

    compression = attributes["compression"]
    if compression is not None:
        nodes.extend(
            [
                {
                    "id": "memory_compressor",
                    "op": "learned_gated_kv_compression",
                    "inputs": ["operator_norm_out", "compressor_accumulator"],
                    "outputs": ["compressed_candidate"],
                    "params": [
                        "compressor_position_bias",
                        "compressor_kv_projection",
                        "compressor_gate_projection",
                        "compressor_norm",
                    ],
                    "state_reads": ["compressor_accumulator"],
                    "state_writes": ["compressor_accumulator"],
                    "attrs": compression,
                },
                {
                    "id": "compressed_memory_update",
                    "op": "conditional_append_state_update",
                    "inputs": ["compressed_candidate", "compressed_kv_memory"],
                    "outputs": ["compressed_kv_values"],
                    "state_reads": ["compressed_kv_memory"],
                    "state_writes": ["compressed_kv_memory"],
                    "attrs": {"period": int(compression["ratio"])},
                },
            ]
        )

    indexer = attributes["indexer"]
    if indexer["selection"] == "learned_topk":
        nodes.append(
            {
                "id": "compressed_memory_indexer",
                "op": "learned_topk_index",
                "inputs": [
                    "operator_norm_out",
                    "query_rank_normed",
                    "compressed_kv_values",
                    "indexer_compressor_accumulator",
                    "indexer_kv_memory",
                ],
                "outputs": ["compressed_indices"],
                "params": [
                    "indexer_q_projection",
                    "indexer_head_weight_projection",
                    "indexer_compressor_position_bias",
                    "indexer_compressor_kv_projection",
                    "indexer_compressor_gate_projection",
                    "indexer_compressor_norm",
                ],
                "state_reads": [
                    "indexer_compressor_accumulator",
                    "indexer_kv_memory",
                ],
                "state_writes": [
                    "indexer_compressor_accumulator",
                    "indexer_kv_memory",
                ],
                "attrs": indexer,
            }
        )
    elif compression is not None:
        nodes.append(
            {
                "id": "compressed_memory_indexer",
                "op": "chronological_compressed_index",
                "inputs": ["compressed_kv_values"],
                "outputs": ["compressed_indices"],
                "attrs": {
                    "ratio": int(compression["ratio"]),
                    "causal": True,
                },
            }
        )

    attention_inputs = ["query_positioned", "local_kv_values"]
    if parallel_context:
        attention_inputs.append("query_key_value_positioned")
    if compression is not None:
        attention_inputs.extend(["compressed_kv_values", "compressed_indices"])
    nodes.extend(
        [
            {
                "id": "sparse_attention_read",
                "op": "indexed_sparse_attention",
                "inputs": attention_inputs,
                "outputs": ["attention_heads"],
                "params": ["attention_sinks"],
                "attrs": {
                    "causal": not parallel_context,
                    "scale": float(numerics["attention_scale"]),
                    "window_size": int(attributes["window_size"]),
                    **(
                        {
                            "intra_block_visibility": execution_contract[
                                "intra_block_visibility"
                            ],
                            "query_state": execution_contract["query_state"],
                        }
                        if parallel_context
                        else {}
                    ),
                    **heads,
                },
            },
            {
                "id": "attention_inverse_rope",
                "op": "inverse_rotary_position_embedding",
                "inputs": ["attention_heads"],
                "outputs": ["attention_unpositioned"],
                "attrs": {
                    **_rope_attrs(
                        numerics, heads, head_count=int(heads["query_heads"])
                    ),
                    **(
                        {
                            "position_offset": int(
                                execution_contract["query_position_offset"]
                            ),
                            "position_mode": "parallel_block",
                        }
                        if parallel_context
                        else {}
                    ),
                },
            },
            {
                "id": "grouped_output_projection",
                "op": "grouped_linear",
                "inputs": ["attention_unpositioned"],
                "outputs": ["attention_ranked"],
                "params": _linear_params("out_group_projection", parameters),
                "attrs": {
                    "groups": int(attributes["output_groups"]),
                    "rank_per_group": int(attributes["output_rank"]),
                },
            },
            {
                "id": "attention_out_projection",
                "op": "linear",
                "inputs": ["attention_ranked"],
                "outputs": ["operator_out"],
                "params": _linear_params("attention_out_projection", parameters),
            },
        ]
    )
    if residual_mixer is None:
        nodes.extend(
            _ffn_tail(
                operator_output="operator_out",
                numerics=numerics,
                feed_forward=component["feed_forward"],
                parameters=parameters,
            )
        )
    else:
        if component["feed_forward"].get("expert_storage") != "independent_resources":
            raise ValueError(
                "hyper-connected latent attention currently requires independently addressed sparse experts"
            )
        nodes.extend(
            [
                {
                    "id": "operator_residual",
                    "op": "hyper_connection_post",
                    "inputs": [
                        "operator_out",
                        "input_frame",
                        "hyper_attention_post",
                        "hyper_attention_combination",
                    ],
                    "outputs": ["operator_residual_out"],
                    "attrs": {
                        **_hyper_connection_attrs(residual_mixer),
                        "output_element_bytes": [2],
                    },
                },
                *_hyper_connection_pre_nodes(
                    stage="feed_forward",
                    input_signal="operator_residual_out",
                    output_signal="ffn_input",
                    mixer=residual_mixer,
                    normalization_epsilon=float(numerics["rms_norm_eps"]),
                ),
                {
                    "id": "ffn_norm",
                    "op": "rms_norm",
                    "inputs": ["ffn_input"],
                    "outputs": ["ffn_norm_out"],
                    "params": ["ffn_norm"],
                    "attrs": _norm_attrs(numerics),
                },
                *independent_sparse_moe_body(
                    feed_forward=component["feed_forward"],
                    parameters=parameters,
                ),
                {
                    "id": "ffn_residual",
                    "op": "hyper_connection_post",
                    "inputs": [
                        "ffn_out",
                        "operator_residual_out",
                        "hyper_feed_forward_post",
                        "hyper_feed_forward_combination",
                    ],
                    "outputs": ["output_frame"],
                    "attrs": {
                        **_hyper_connection_attrs(residual_mixer),
                        "output_element_bytes": [2],
                    },
                },
            ]
        )
    return nodes


def _hyper_connection_pre_nodes(
    *,
    stage: str,
    input_signal: str,
    output_signal: str,
    mixer: Json,
    normalization_epsilon: float,
) -> list[Json]:
    parameter_prefix = f"hyper_{stage}"
    signal_prefix = f"hyper_{stage}"
    return [
        {
            "id": f"{signal_prefix}_function",
            "op": "normalized_linear",
            "inputs": [input_signal],
            "outputs": [f"{signal_prefix}_mixes"],
            "params": [f"{parameter_prefix}_function"],
            "attrs": {
                "normalization": "root_mean_square",
                "normalization_epsilon": normalization_epsilon,
                "multiplicity": int(mixer["multiplicity"]),
                "output_element_bytes": [4],
            },
        },
        {
            "id": f"{signal_prefix}_sinkhorn",
            "op": "hyper_connection_sinkhorn",
            "inputs": [f"{signal_prefix}_mixes"],
            "outputs": [
                f"{signal_prefix}_pre",
                f"{signal_prefix}_post",
                f"{signal_prefix}_combination",
            ],
            "params": [
                f"{parameter_prefix}_scale",
                f"{parameter_prefix}_base",
            ],
            "attrs": {
                **_hyper_connection_attrs(mixer),
                "output_element_bytes": [4, 4, 4],
            },
        },
        {
            "id": f"{signal_prefix}_reduce",
            "op": "hyper_connection_reduce",
            "inputs": [input_signal, f"{signal_prefix}_pre"],
            "outputs": [output_signal],
            "attrs": {
                "multiplicity": int(mixer["multiplicity"]),
                "output_element_bytes": [2],
            },
        },
    ]


def _hyper_connection_attrs(mixer: Json) -> Json:
    return {
        "multiplicity": int(mixer["multiplicity"]),
        "sinkhorn_iterations": int(mixer["sinkhorn_iterations"]),
        "epsilon": float(mixer["epsilon"]),
    }


def _rope_attrs(numerics: Json, heads: Json, *, head_count: int) -> Json:
    return {
        "position_source": "stream_tick",
        "theta": float(numerics["rope_theta"]),
        "rope_type": str(numerics.get("rope_type", "default")),
        "scaling": numerics.get("rope_scaling"),
        "interleaved": bool(numerics["rope_interleaved"]),
        "rotary_width": int(numerics["rotary_width"]),
        **heads,
        "head_count": head_count,
    }
