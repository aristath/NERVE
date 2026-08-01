from __future__ import annotations

from pathlib import Path

from nerve.model_transpiler_discovery import (
    discover_layer_root,
    discover_model_structure,
)


def _tensor() -> dict[str, object]:
    return {"dtype": "BF16", "shape": [8, 8]}


def test_discovers_root_level_decoder_layers() -> None:
    tensors = {
        "layers.0.attention.wq.weight": _tensor(),
        "layers.0.feed_forward.w1.weight": _tensor(),
        "layers.1.attention.wq.weight": _tensor(),
        "layers.1.feed_forward.w1.weight": _tensor(),
    }

    assert discover_layer_root(tensors) == ("layers", (0, 1))


def test_root_level_decoder_wins_over_equal_auxiliary_stack() -> None:
    tensors = {
        "embed_tokens.weight": _tensor(),
        "norm.weight": _tensor(),
        "layers.0.attention.wq.weight": _tensor(),
        "layers.0.feed_forward.w1.weight": _tensor(),
        "layers.1.attention.wq.weight": _tensor(),
        "layers.1.feed_forward.w1.weight": _tensor(),
    }
    for layer_index in range(2):
        for tensor_index in range(16):
            tensors[
                f"draft.layers.{layer_index}.auxiliary_{tensor_index}.weight"
            ] = _tensor()

    assert discover_layer_root(
        tensors,
        config={"num_hidden_layers": 2},
    ) == ("layers", (0, 1))


def test_discovers_complete_root_level_decoder_structure() -> None:
    tensors = {
        "embed_tokens.weight": {"dtype": "BF16", "shape": [32, 8]},
        "norm.weight": {"dtype": "BF16", "shape": [8]},
        "layers.0.input_layernorm.weight": {"dtype": "BF16", "shape": [8]},
        "layers.0.post_attention_layernorm.weight": {
            "dtype": "BF16",
            "shape": [8],
        },
        "layers.0.attention.wq.weight": _tensor(),
        "layers.0.attention.wk.weight": _tensor(),
        "layers.0.attention.wv.weight": _tensor(),
        "layers.0.attention.wo.weight": _tensor(),
        "layers.0.feed_forward.w1.weight": {"dtype": "BF16", "shape": [16, 8]},
        "layers.0.feed_forward.w2.weight": {"dtype": "BF16", "shape": [8, 16]},
        "layers.0.feed_forward.w3.weight": {"dtype": "BF16", "shape": [16, 8]},
    }
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }

    structure = discover_model_structure(Path("synthetic"), config, tensors)

    assert structure.tensors["token_embedding"] == "embed_tokens.weight"
    assert structure.tensors["output_norm"] == "norm.weight"
    assert structure.layers[0].prefix == "layers.0"
