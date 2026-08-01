from __future__ import annotations

from pathlib import Path

import pytest

from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_model_graph
from nerve.model_transpiler_types import ModelTranspileError


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


def _add_layer(
    tensors: dict[str, dict[str, object]], prefix: str, *, hidden_size: int
) -> None:
    tensors.update(
        {
            f"{prefix}.attn_norm.weight": _tensor([hidden_size]),
            f"{prefix}.attention.wq.weight": _tensor([hidden_size, hidden_size]),
            f"{prefix}.attention.wk.weight": _tensor([hidden_size // 2, hidden_size]),
            f"{prefix}.attention.wv.weight": _tensor([hidden_size // 2, hidden_size]),
            f"{prefix}.attention.wo.weight": _tensor([hidden_size, hidden_size]),
            f"{prefix}.ffn_norm.weight": _tensor([hidden_size]),
            f"{prefix}.feed_forward.w1.weight": _tensor([hidden_size * 2, hidden_size]),
            f"{prefix}.feed_forward.w2.weight": _tensor([hidden_size, hidden_size * 2]),
            f"{prefix}.feed_forward.w3.weight": _tensor([hidden_size * 2, hidden_size]),
            f"{prefix}.hc_attn_fn": _tensor([24, hidden_size * 4], "F32"),
            f"{prefix}.hc_attn_base": _tensor([24], "F32"),
            f"{prefix}.hc_attn_scale": _tensor([3], "F32"),
            f"{prefix}.hc_ffn_fn": _tensor([24, hidden_size * 4], "F32"),
            f"{prefix}.hc_ffn_base": _tensor([24], "F32"),
            f"{prefix}.hc_ffn_scale": _tensor([3], "F32"),
        }
    )


def _source() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    hidden_size = 8
    vocab_size = 32
    markov_rank = 2
    config: dict[str, object] = {
        "model_type": "synthetic_parallel_draft_decoder",
        "hidden_size": hidden_size,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": hidden_size * 2,
        "vocab_size": vocab_size,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 31,
        "dspark_target_layer_ids": [0, 1, 2],
        "dspark_markov_rank": markov_rank,
    }
    tensors = {
        "embed.weight": _tensor([vocab_size, hidden_size]),
        "norm.weight": _tensor([hidden_size]),
        "head.weight": _tensor([vocab_size, hidden_size]),
        "hc_head_fn": _tensor([4, hidden_size * 4], "F32"),
        "hc_head_base": _tensor([4], "F32"),
        "hc_head_scale": _tensor([1], "F32"),
    }
    for index in range(3):
        _add_layer(tensors, f"layers.{index}", hidden_size=hidden_size)
        _add_layer(tensors, f"mtp.{index}", hidden_size=hidden_size)
    tensors.update(
        {
            "mtp.0.main_proj.weight": _tensor([hidden_size, hidden_size * 3]),
            "mtp.0.main_norm.weight": _tensor([hidden_size]),
            "mtp.2.norm.weight": _tensor([hidden_size]),
            "mtp.2.hc_head_fn": _tensor([4, hidden_size * 4], "F32"),
            "mtp.2.hc_head_base": _tensor([4], "F32"),
            "mtp.2.hc_head_scale": _tensor([1], "F32"),
            "mtp.2.markov_head.markov_w1.weight": _tensor([vocab_size, markov_rank]),
            "mtp.2.markov_head.markov_w2.weight": _tensor([vocab_size, markov_rank]),
            "mtp.2.confidence_head.proj.weight": _tensor(
                [1, hidden_size + markov_rank]
            ),
        }
    )
    return config, tensors


def test_discovers_parallel_backbone_markov_draft_from_tensor_contract(
    tmp_path: Path,
) -> None:
    config, tensors = _source()

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    [draft] = structure.draft_execution_graphs
    graph = make_model_graph(structure, tmp_path, {"source": {}})["graph"][
        "draft_execution_graphs"
    ][0]

    assert draft.prefix == "mtp"
    assert len(draft.layers) == 3
    assert graph["type"] == "parallel_backbone_markov"
    assert graph["target_features"] == {
        "layer_indices": [0, 1, 2],
        "lane_reduction": "mean",
        "concatenation_order": "declared_layer_order",
    }
    assert graph["proposal_contract"] == {
        "schedule": "parallel_backbone_then_sequential_markov",
        "configured_block_size": 5,
        "noise_token_id": 31,
        "sampling": "greedy",
        "confidence_prefix": "first_sigmoid_below_runtime_threshold",
        "verification": "lossless_target_longest_matching_prefix",
    }
    assert graph["input_adapter"]["params"] == {
        "target_projection": {"tensor": "mtp.0.main_proj.weight"},
        "target_norm": {"tensor": "mtp.0.main_norm.weight"},
    }
    assert graph["output_transducer"]["params"] == {
        "head_function": {"tensor": "mtp.2.hc_head_fn"},
        "head_base": {"tensor": "mtp.2.hc_head_base"},
        "head_scale": {"tensor": "mtp.2.hc_head_scale"},
        "norm": {"tensor": "mtp.2.norm.weight"},
        "projection": {"tensor": "head.weight"},
        "markov_embedding": {"tensor": "mtp.2.markov_head.markov_w1.weight"},
        "markov_projection": {"tensor": "mtp.2.markov_head.markov_w2.weight"},
        "confidence_projection": {"tensor": "mtp.2.confidence_head.proj.weight"},
    }


def test_rejects_parallel_markov_draft_when_confidence_shape_is_incompatible() -> None:
    config, tensors = _source()
    tensors["mtp.2.confidence_head.proj.weight"] = _tensor([1, 8])

    with pytest.raises(ModelTranspileError, match="confidence"):
        discover_model_structure(Path("synthetic"), config, tensors)
