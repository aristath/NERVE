from __future__ import annotations

from pathlib import Path

import pytest

from nerve.circuit_lowering import build_component_circuit
from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_layer
from nerve.model_transpiler_types import ModelTranspileError


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


def _add_dense_ffn(tensors: dict[str, dict[str, object]], prefix: str) -> None:
    tensors.update(
        {
            f"{prefix}.ffn_norm.weight": _tensor([8]),
            f"{prefix}.feed_forward.w1.weight": _tensor([16, 8]),
            f"{prefix}.feed_forward.w2.weight": _tensor([8, 16]),
            f"{prefix}.feed_forward.w3.weight": _tensor([16, 8]),
        }
    )


def _add_latent_attention(
    tensors: dict[str, dict[str, object]],
    prefix: str,
    *,
    compression_ratio: int,
    learned_indexer: bool,
) -> None:
    tensors.update(
        {
            f"{prefix}.attn_norm.weight": _tensor([8]),
            f"{prefix}.attn.wq_a.weight": _tensor([4, 8]),
            f"{prefix}.attn.q_norm.weight": _tensor([4]),
            f"{prefix}.attn.wq_b.weight": _tensor([16, 4]),
            f"{prefix}.attn.wkv.weight": _tensor([4, 8]),
            f"{prefix}.attn.kv_norm.weight": _tensor([4]),
            f"{prefix}.attn.attn_sink": _tensor([4], "F32"),
            f"{prefix}.attn.wo_a.weight": _tensor([4, 8]),
            f"{prefix}.attn.wo_b.weight": _tensor([8, 4]),
        }
    )
    if compression_ratio:
        coefficient = 2 if compression_ratio == 4 else 1
        tensors.update(
            {
                f"{prefix}.attn.compressor.ape": _tensor(
                    [compression_ratio, coefficient * 4], "F32"
                ),
                f"{prefix}.attn.compressor.wkv.weight": _tensor([coefficient * 4, 8]),
                f"{prefix}.attn.compressor.wgate.weight": _tensor([coefficient * 4, 8]),
                f"{prefix}.attn.compressor.norm.weight": _tensor([4]),
            }
        )
    if learned_indexer:
        tensors.update(
            {
                f"{prefix}.attn.indexer.wq_b.weight": _tensor([8, 4]),
                f"{prefix}.attn.indexer.weights_proj.weight": _tensor([4, 8]),
                f"{prefix}.attn.indexer.compressor.ape": _tensor([4, 4], "F32"),
                f"{prefix}.attn.indexer.compressor.wkv.weight": _tensor([4, 8]),
                f"{prefix}.attn.indexer.compressor.wgate.weight": _tensor([4, 8]),
                f"{prefix}.attn.indexer.compressor.norm.weight": _tensor([2]),
            }
        )


def _source() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config: dict[str, object] = {
        "model_type": "synthetic_latent_sparse_decoder",
        "hidden_size": 8,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "o_lora_rank": 2,
        "o_groups": 2,
        "qk_rope_head_dim": 2,
        "sliding_window": 8,
        "compress_ratios": [0, 4, 8],
        "compress_rope_theta": 160_000.0,
        "index_n_heads": 4,
        "index_head_dim": 2,
        "index_topk": 16,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 1024,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    tensors = {
        "embed.weight": _tensor([32, 8]),
        "norm.weight": _tensor([8]),
        "head.weight": _tensor([32, 8]),
    }
    for index, ratio in enumerate((0, 4, 8)):
        prefix = f"layers.{index}"
        _add_dense_ffn(tensors, prefix)
        _add_latent_attention(
            tensors,
            prefix,
            compression_ratio=ratio,
            learned_indexer=ratio == 4,
        )
    return config, tensors


def test_discovers_sliding_and_compressed_latent_attention_topologies() -> None:
    config, tensors = _source()

    structure = discover_model_structure(Path("synthetic"), config, tensors)

    sliding, learned, deterministic = structure.layers
    assert {layer.operator_type for layer in structure.layers} == {
        "latent_sparse_attention"
    }
    assert sliding.operator_attributes["compression"] is None
    assert sliding.operator_attributes["window_size"] == 8
    assert learned.operator_attributes["compression"]["ratio"] == 4
    assert learned.operator_attributes["compression"]["overlap"] is True
    assert learned.operator_attributes["indexer"]["selection"] == "learned_topk"
    assert learned.operator_attributes["indexer"]["top_k"] == 16
    assert deterministic.operator_attributes["compression"]["ratio"] == 8
    assert deterministic.operator_attributes["compression"]["overlap"] is False
    assert deterministic.operator_attributes["indexer"] == {
        "selection": "chronological_compressed_positions"
    }
    assert learned.rotary_width == 2
    assert learned.rope_theta == 160_000.0
    assert learned.rope_type == "yarn"
    assert sliding.rope_theta == 10_000.0
    assert sliding.rope_type == "default"
    component = make_layer(structure, learned)
    circuit = build_component_circuit(component, Path("layer_01.json"))
    assert component["operator_type"] == "latent_sparse_attention"
    assert component["reference_decomposition"]["topology"][1]["type"] == (
        "latent_sparse_attention_operator"
    )
    assert [port["id"] for port in component["state_ports"]] == [
        "local_kv_memory",
        "compressed_kv_memory",
        "compressor_accumulator",
        "indexer_compressor_accumulator",
        "indexer_kv_memory",
    ]
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["query_input_projection"]["op"] == "linear"
    assert nodes["memory_compressor"]["op"] == "learned_gated_kv_compression"
    assert nodes["compressed_memory_indexer"]["op"] == "learned_topk_index"
    assert nodes["sparse_attention_read"]["op"] == "indexed_sparse_attention"
    assert nodes["grouped_output_projection"]["op"] == "grouped_linear"
    deterministic_component = make_layer(structure, deterministic)
    deterministic_circuit = build_component_circuit(
        deterministic_component, Path("layer_02.json")
    )
    deterministic_modules = {
        module["id"]: module
        for module in deterministic_circuit["semantic_module_tree"]["modules"]
    }
    assert (
        deterministic_modules["layer.token_mixer.memory_index"]["owned_state_port_ids"]
        == []
    )


def test_composes_latent_attention_independent_experts_and_hyper_connections() -> None:
    config, tensors = _source()
    config.update(
        {
            "num_hidden_layers": 1,
            "compress_ratios": [0],
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "n_routed_experts": 3,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "moe_intermediate_size": 6,
            "scoring_func": "sqrtsoftplus",
            "routed_scaling_factor": 1.5,
            "norm_topk_prob": True,
            "swiglu_limit": 10.0,
        }
    )
    tensors = {
        name: tensor
        for name, tensor in tensors.items()
        if not name.startswith(("layers.1.", "layers.2."))
    }
    for name in (
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
    ):
        del tensors[name]
    tensors.update(
        {
            "hc_head_fn": _tensor([4, 32], "F32"),
            "hc_head_base": _tensor([4], "F32"),
            "hc_head_scale": _tensor([1], "F32"),
            "layers.0.hc_attn_fn": _tensor([24, 32], "F32"),
            "layers.0.hc_attn_base": _tensor([24], "F32"),
            "layers.0.hc_attn_scale": _tensor([3], "F32"),
            "layers.0.hc_ffn_fn": _tensor([24, 32], "F32"),
            "layers.0.hc_ffn_base": _tensor([24], "F32"),
            "layers.0.hc_ffn_scale": _tensor([3], "F32"),
            "layers.0.ffn.gate.weight": _tensor([3, 8]),
            "layers.0.ffn.gate.tid2eid": _tensor([32, 2], "I64"),
            "layers.0.ffn.shared_experts.w1.weight": _tensor([6, 8]),
            "layers.0.ffn.shared_experts.w2.weight": _tensor([8, 6]),
            "layers.0.ffn.shared_experts.w3.weight": _tensor([6, 8]),
        }
    )
    for expert in range(3):
        tensors[f"layers.0.ffn.experts.{expert}.w1.weight"] = _tensor([6, 8])
        tensors[f"layers.0.ffn.experts.{expert}.w2.weight"] = _tensor([8, 6])
        tensors[f"layers.0.ffn.experts.{expert}.w3.weight"] = _tensor([6, 8])

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    component = make_layer(structure, structure.layers[0])
    circuit = build_component_circuit(component, Path("layer_00.json"))

    reference_topology = component["reference_decomposition"]["topology"]
    assert [node["id"] for node in reference_topology] == [
        "hyper_attention_function",
        "hyper_attention_sinkhorn",
        "hyper_attention_reduce",
        "operator_norm",
        "operator",
        "operator_residual",
        "hyper_feed_forward_function",
        "hyper_feed_forward_sinkhorn",
        "hyper_feed_forward_reduce",
        "ffn_norm",
        "feed_forward",
        "ffn_residual",
    ]
    assert reference_topology[5]["type"] == "sinkhorn_hyper_connection_post"
    assert reference_topology[-1]["type"] == "sinkhorn_hyper_connection_post"
    assert not any(node["type"] == "residual_add" for node in reference_topology)
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert nodes["hyper_attention_function"]["attrs"] == {
        "normalization": "root_mean_square",
        "normalization_epsilon": 1e-6,
        "multiplicity": 4,
        "output_element_bytes": [4],
    }
    assert nodes["hyper_attention_sinkhorn"]["attrs"][
        "output_element_bytes"
    ] == [4, 4, 4]
    assert nodes["hyper_attention_reduce"]["attrs"]["output_element_bytes"] == [2]
    assert nodes["operator_residual"]["attrs"]["output_element_bytes"] == [2]
    node_ids = [node["id"] for node in circuit["nodes"]]
    assert node_ids.index("hyper_attention_reduce") < node_ids.index("operator_norm")
    assert node_ids.index("operator_residual") < node_ids.index(
        "hyper_feed_forward_function"
    )
    assert node_ids.index("hyper_feed_forward_reduce") < node_ids.index("ffn_norm")
    assert node_ids[-1] == "ffn_residual"
    modules = {
        module["id"]: module for module in circuit["semantic_module_tree"]["modules"]
    }
    assert modules["layer.token_mixer.hyper_connection"]["source_node_ids"] == [
        "hyper_attention_function",
        "hyper_attention_sinkhorn",
        "hyper_attention_reduce",
    ]
    assert modules["layer.feature_transform.hyper_connection"]["source_node_ids"] == [
        "hyper_feed_forward_function",
        "hyper_feed_forward_sinkhorn",
        "hyper_feed_forward_reduce",
    ]


def test_rejects_incomplete_compressor_tensor_contract() -> None:
    config, tensors = _source()
    del tensors["layers.1.attn.compressor.wgate.weight"]

    with pytest.raises(ModelTranspileError, match="incomplete attention compressor"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_rejects_partial_learned_indexer_tensor_contract() -> None:
    config, tensors = _source()
    del tensors["layers.1.attn.indexer.weights_proj.weight"]

    with pytest.raises(ModelTranspileError, match="incomplete attention indexer"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_lowers_parallel_query_context_as_committed_state_and_transient_block() -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    component = make_layer(
        structure,
        structure.layers[0],
        component_id="draft_00_layer_00",
        runtime_role="draft_processor",
        execution_contract={
            "type": "parallel_query_with_external_kv_context",
            "context_input": "main_context",
            "context_state_update": "committed_target_only",
            "query_state": "transient",
            "intra_block_visibility": "all",
            "query_position_offset": 1,
        },
    )

    circuit = build_component_circuit(component, Path("draft_00_layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}

    assert [port["id"] for port in circuit["boundary"]["inputs"]] == [
        "input_frame",
        "main_context",
    ]
    assert nodes["key_value_projection"]["inputs"] == ["main_context"]
    assert nodes["local_memory_update"]["attrs"]["source"] == (
        "committed_target_context"
    )
    assert nodes["query_key_value_projection"]["inputs"] == ["operator_norm_out"]
    assert nodes["query_key_value_rope"]["attrs"]["position_mode"] == ("parallel_block")
    assert nodes["sparse_attention_read"]["attrs"]["causal"] is False
    assert nodes["sparse_attention_read"]["attrs"]["intra_block_visibility"] == "all"
    assert nodes["sparse_attention_read"]["attrs"]["query_state"] == ("transient")
