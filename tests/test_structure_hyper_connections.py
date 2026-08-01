from __future__ import annotations

from pathlib import Path

import pytest

from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_layer, make_model_graph
from nerve.model_transpiler_types import ModelTranspileError


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


def _source() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config: dict[str, object] = {
        "model_type": "synthetic_hyper_connected_decoder",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
    }
    prefix = "layers.0"
    tensors = {
        "embed.weight": _tensor([32, 8]),
        "norm.weight": _tensor([8]),
        "head.weight": _tensor([32, 8]),
        "hc_head_fn": _tensor([4, 32], "F32"),
        "hc_head_base": _tensor([4], "F32"),
        "hc_head_scale": _tensor([1], "F32"),
        f"{prefix}.input_layernorm.weight": _tensor([8]),
        f"{prefix}.post_attention_layernorm.weight": _tensor([8]),
        f"{prefix}.attention.wq.weight": _tensor([8, 8]),
        f"{prefix}.attention.wk.weight": _tensor([8, 8]),
        f"{prefix}.attention.wv.weight": _tensor([8, 8]),
        f"{prefix}.attention.wo.weight": _tensor([8, 8]),
        f"{prefix}.feed_forward.w1.weight": _tensor([16, 8]),
        f"{prefix}.feed_forward.w2.weight": _tensor([8, 16]),
        f"{prefix}.feed_forward.w3.weight": _tensor([16, 8]),
        f"{prefix}.hc_attn_fn": _tensor([24, 32], "F32"),
        f"{prefix}.hc_attn_base": _tensor([24], "F32"),
        f"{prefix}.hc_attn_scale": _tensor([3], "F32"),
        f"{prefix}.hc_ffn_fn": _tensor([24, 32], "F32"),
        f"{prefix}.hc_ffn_base": _tensor([24], "F32"),
        f"{prefix}.hc_ffn_scale": _tensor([3], "F32"),
    }
    return config, tensors


def test_discovers_sinkhorn_hyper_connection_from_tensor_contract() -> None:
    config, tensors = _source()

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    layer = structure.layers[0]
    component = make_layer(structure, layer)
    graph = make_model_graph(structure, Path("transpiled"), {"source": {}})

    assert structure.stream_shape == (4, 8)
    assert structure.stream_mixer == {
        "type": "sinkhorn_hyper_connection",
        "multiplicity": 4,
        "sinkhorn_iterations": 20,
        "epsilon": 1e-6,
        "head": {
            "function": "hc_head_fn",
            "base": "hc_head_base",
            "scale": "hc_head_scale",
        },
    }
    assert layer.boundary_shape == (4, 8)
    assert layer.residual_mixer == {
        "type": "sinkhorn_hyper_connection",
        "attention": {
            "function": "layers.0.hc_attn_fn",
            "base": "layers.0.hc_attn_base",
            "scale": "layers.0.hc_attn_scale",
        },
        "feed_forward": {
            "function": "layers.0.hc_ffn_fn",
            "base": "layers.0.hc_ffn_base",
            "scale": "layers.0.hc_ffn_scale",
        },
    }
    assert component["ports"]["inputs"][0]["shape"] == [4, 8]
    assert component["ports"]["outputs"][0]["shape"] == [4, 8]
    assert component["residual_mixer"] == layer.residual_mixer
    assert graph["graph"]["input_transducer"]["attrs"]["stream_expansion"] == {
        "type": "repeat",
        "multiplicity": 4,
    }
    collapse = graph["graph"]["output_transducer"]["components"][0]
    assert collapse["type"] == "sinkhorn_hyper_connection_head"
    assert collapse["attrs"]["input_shape"] == [4, 8]
    assert collapse["attrs"]["output_shape"] == [8]
    assert collapse["params"] == {
        "function": {"tensor": "hc_head_fn"},
        "base": {"tensor": "hc_head_base"},
        "scale": {"tensor": "hc_head_scale"},
    }


def test_rejects_incomplete_hyper_connection_tensor_set() -> None:
    config, tensors = _source()
    del tensors["layers.0.hc_ffn_scale"]

    with pytest.raises(ModelTranspileError, match="incomplete hyper-connection"):
        discover_model_structure(Path("synthetic"), config, tensors)
