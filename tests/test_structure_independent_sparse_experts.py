from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nerve.circuit_lowering import build_component_circuit
from nerve.circuit_ir import validate_circuit
from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_layer
from nerve.model_transpiler_quantization import annotate_mxfp4_expert_tensors
from nerve.model_transpiler_types import ModelTranspileError


def _tensor(shape: list[int], dtype: str = "BF16") -> dict[str, object]:
    return {"dtype": dtype, "shape": shape}


def _add_attention(tensors: dict[str, dict[str, object]], prefix: str) -> None:
    tensors.update(
        {
            f"{prefix}.input_layernorm.weight": _tensor([8]),
            f"{prefix}.post_attention_layernorm.weight": _tensor([8]),
            f"{prefix}.attention.wq.weight": _tensor([8, 8]),
            f"{prefix}.attention.wk.weight": _tensor([8, 8]),
            f"{prefix}.attention.wv.weight": _tensor([8, 8]),
            f"{prefix}.attention.wo.weight": _tensor([8, 8]),
        }
    )


def _add_sparse_experts(
    tensors: dict[str, dict[str, object]], prefix: str, *, hash_routing: bool
) -> None:
    tensors[f"{prefix}.ffn.gate.weight"] = _tensor([3, 8])
    if hash_routing:
        tensors[f"{prefix}.ffn.gate.tid2eid"] = _tensor([32, 2], "I64")
    else:
        tensors[f"{prefix}.ffn.gate.bias"] = _tensor([3], "F32")
    for expert in range(3):
        tensors[f"{prefix}.ffn.experts.{expert}.w1.weight"] = _tensor([6, 8])
        tensors[f"{prefix}.ffn.experts.{expert}.w2.weight"] = _tensor([8, 6])
        tensors[f"{prefix}.ffn.experts.{expert}.w3.weight"] = _tensor([6, 8])
    tensors[f"{prefix}.ffn.shared_experts.w1.weight"] = _tensor([6, 8])
    tensors[f"{prefix}.ffn.shared_experts.w2.weight"] = _tensor([8, 6])
    tensors[f"{prefix}.ffn.shared_experts.w3.weight"] = _tensor([6, 8])


def _source() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    config: dict[str, object] = {
        "model_type": "synthetic_independent_expert_decoder",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "n_routed_experts": 3,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "moe_intermediate_size": 6,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "norm_topk_prob": True,
        "swiglu_limit": 10.0,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
    }
    tensors = {
        "model.embed_tokens.weight": _tensor([32, 8]),
        "model.norm.weight": _tensor([8]),
    }
    for index in range(2):
        prefix = f"model.layers.{index}"
        _add_attention(tensors, prefix)
        _add_sparse_experts(tensors, prefix, hash_routing=index == 0)
    return config, tensors


def test_discovers_independently_addressable_experts_and_per_layer_routing() -> None:
    config, tensors = _source()

    structure = discover_model_structure(Path("synthetic"), config, tensors)

    hashed, scored = structure.layers
    assert structure.num_experts == 3
    assert structure.experts_per_token == 2
    assert hashed.feed_forward_type == "sparse_moe"
    assert hashed.feed_forward_attributes["expert_storage"] == "independent_resources"
    assert hashed.feed_forward_attributes["expert_ids"] == [0, 1, 2]
    assert hashed.feed_forward_attributes["routing"]["selection"] == "token_id_table"
    assert scored.feed_forward_attributes["routing"]["selection"] == "score_topk"
    assert scored.feed_forward_attributes["routing"]["selection_bias"] == (
        "model.layers.1.ffn.gate.bias"
    )
    assert scored.feed_forward_attributes["routing"]["activation"] == "sqrtsoftplus"
    assert scored.feed_forward_attributes["swiglu_limit"] == 10.0
    for expert in range(3):
        for projection in ("w1", "w2", "w3"):
            parameter = f"routed_expert_{expert:03d}_{projection}"
            assert hashed.tensors[parameter] == (
                f"model.layers.0.ffn.experts.{expert}.{projection}.weight"
            )
    component = make_layer(structure, hashed)
    circuit = build_component_circuit(component, Path("layer_00.json"))
    assert validate_circuit(circuit).ok
    nodes = {node["id"]: node for node in circuit["nodes"]}
    assert component["ports"]["controls"] == [
        {
            "id": "token_id",
            "signal": "token_id",
            "shape": [],
            "dtype": "U32",
            "runtime_source": "input_token_id",
        }
    ]
    assert nodes["moe_topk"]["inputs"] == ["moe_router_logits", "token_id"]
    selected = nodes["sparse_moe_gate_up"]["attrs"]["selected_parameter_accesses"][0]
    assert selected["selection_signal"] == "moe_routes"
    assert selected["mapping"][1] == {
        "selector": 1,
        "parameter_ids": [
            "routed_expert_001_w1",
            "routed_expert_001_w3",
        ],
    }
    assert "routed_expert_002_w2" in nodes["sparse_moe_down"]["params"]


def test_sparse_expert_discovery_depends_on_structure_not_model_identity() -> None:
    config, tensors = _source()
    config["architectures"] = ["KnownSparseDecoder"]
    known = discover_model_structure(Path("synthetic"), config, tensors)

    future_config = dict(config)
    future_config["model_type"] = "previously_unseen_decoder"
    future_config["architectures"] = ["PreviouslyUnseenSparseDecoder"]
    future = discover_model_structure(Path("synthetic"), future_config, tensors)

    assert future.model_type == "previously_unseen_decoder"
    assert future.architectures == ("PreviouslyUnseenSparseDecoder",)
    assert replace(
        future,
        model_type=known.model_type,
        architectures=known.architectures,
    ) == known
    assert make_layer(future, future.layers[0]) == make_layer(known, known.layers[0])


def test_sparse_expert_discovery_normalizes_unseen_equivalent_tensor_roles() -> None:
    config, tensors = _source()
    future_config = dict(config)
    future_config["model_type"] = "previously_unseen_sparse_decoder"
    future_tensors: dict[str, dict[str, object]] = {}
    for name, info in tensors.items():
        renamed = name.replace(".ffn.", ".future_sparse_block.")
        renamed = renamed.replace(".gate.", ".router.")
        renamed = renamed.replace(".shared_experts.", ".shared_expert.")
        renamed = renamed.replace(".w1.weight", ".gate_proj.weight")
        renamed = renamed.replace(".w2.weight", ".down_proj.weight")
        renamed = renamed.replace(".w3.weight", ".up_proj.weight")
        future_tensors[renamed] = info

    future = discover_model_structure(
        Path("synthetic"), future_config, future_tensors
    )

    hashed, scored = future.layers
    assert hashed.feed_forward_attributes["routing"]["selection"] == "token_id_table"
    assert scored.feed_forward_attributes["routing"]["selection"] == "score_topk"
    assert hashed.tensors["routed_expert_001_w1"] == (
        "model.layers.0.future_sparse_block.experts.1.gate_proj.weight"
    )
    assert hashed.tensors["routed_expert_001_w2"] == (
        "model.layers.0.future_sparse_block.experts.1.down_proj.weight"
    )
    assert hashed.tensors["routed_expert_001_w3"] == (
        "model.layers.0.future_sparse_block.experts.1.up_proj.weight"
    )
    assert hashed.tensors["shared_expert_w1"] == (
        "model.layers.0.future_sparse_block.shared_expert.gate_proj.weight"
    )
    assert hashed.feed_forward_type == "sparse_moe"


def test_mxfp4_annotation_normalizes_unseen_equivalent_expert_tensor_roles() -> None:
    config = {"future_quantization": {"expert_dtype": "fp4"}}
    tensors: dict[str, dict[str, object]] = {}
    projection_shapes = {
        "gate_proj": [6, 16],
        "down_proj": [8, 16],
        "up_proj": [6, 16],
    }
    for role, storage_shape in projection_shapes.items():
        name = f"future.blocks.0.sparse.experts.0.{role}.weight"
        tensors[name] = {
            "dtype": "I8",
            "shape": storage_shape,
            "byte_count": storage_shape[0] * storage_shape[1],
        }
        tensors[name.removesuffix(".weight") + ".scale"] = _tensor(
            [storage_shape[0], 1], "F8_E8M0"
        )

    annotate_mxfp4_expert_tensors(config, tensors)

    for role, storage_shape in projection_shapes.items():
        info = tensors[f"future.blocks.0.sparse.experts.0.{role}.weight"]
        assert info["logical_shape"] == [storage_shape[0], storage_shape[1] * 2]
        assert info["quantization"]["format"] == "mxfp4_e2m1"


def test_aggregate_expert_representation_takes_precedence_by_structure() -> None:
    config, tensors = _source()
    del tensors["model.layers.0.ffn.gate.tid2eid"]
    tensors["model.layers.0.ffn.gate.bias"] = _tensor([3], "F32")
    aggregate_source: dict[str, dict[str, object]] = {}
    for name, info in tensors.items():
        renamed = name.replace(".ffn.", ".mlp.")
        renamed = renamed.replace(".w1.weight", ".gate_proj.weight")
        renamed = renamed.replace(".w2.weight", ".down_proj.weight")
        renamed = renamed.replace(".w3.weight", ".up_proj.weight")
        copied = dict(info)
        if ".experts." in renamed and renamed.endswith(".weight"):
            byte_count = 2
            for dimension in copied["shape"]:
                byte_count *= int(dimension)
            copied.update(
                {
                    "byte_count": byte_count,
                    "source_file": "synthetic.safetensors",
                    "source_header_bytes": 0,
                    "data_offsets": [0, byte_count],
                }
            )
        aggregate_source[renamed] = copied

    structure = discover_model_structure(Path("synthetic"), config, aggregate_source)

    for layer in structure.layers:
        assert layer.feed_forward_type == "sparse_moe"
        assert layer.tensors["moe_input"].endswith(".mlp.experts.gate_up_proj")
        assert "routed_expert_000_w1" not in layer.tensors


def test_rejects_ambiguous_independent_expert_projection_aliases() -> None:
    config, tensors = _source()
    tensors["model.layers.0.ffn.experts.0.gate_proj.weight"] = _tensor([6, 8])

    with pytest.raises(ModelTranspileError, match="ambiguous w1 projection"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_rejects_incomplete_independent_expert() -> None:
    config, tensors = _source()
    del tensors["model.layers.1.ffn.experts.2.w3.weight"]

    with pytest.raises(ModelTranspileError, match="expert 2 is incomplete"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_rejects_ambiguous_hash_and_score_routing() -> None:
    config, tensors = _source()
    tensors["model.layers.0.ffn.gate.bias"] = _tensor([3], "F32")

    with pytest.raises(ModelTranspileError, match="ambiguous expert selection"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_keeps_mxfp4_weight_and_scale_independently_selectable_per_expert() -> None:
    config, tensors = _source()
    for name, info in tuple(tensors.items()):
        if ".ffn.experts." not in name or not name.endswith(".weight"):
            continue
        logical_shape = list(info["shape"])
        info["dtype"] = "I8"
        info["shape"] = [logical_shape[0], logical_shape[1] // 2]
        info["logical_shape"] = logical_shape
        scale = name.removesuffix(".weight") + ".scale"
        info["quantization"] = {
            "format": "mxfp4_e2m1",
            "scales": scale,
        }
        tensors[scale] = _tensor([logical_shape[0], 1], "F8_E8M0")

    structure = discover_model_structure(Path("synthetic"), config, tensors)
    component = make_layer(structure, structure.layers[0])
    circuit = build_component_circuit(component, Path("layer_00.json"))
    nodes = {node["id"]: node for node in circuit["nodes"]}

    assert structure.layers[0].feed_forward_attributes["source_expert_format"] == (
        "mxfp4_e2m1"
    )
    assert nodes["sparse_moe_gate_up"]["attrs"]["selected_parameter_accesses"][0][
        "mapping"
    ][1] == {
        "selector": 1,
        "parameter_ids": [
            "routed_expert_001_w1",
            "routed_expert_001_w1_scale",
            "routed_expert_001_w3",
            "routed_expert_001_w3_scale",
        ],
    }
