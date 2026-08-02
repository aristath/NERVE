from __future__ import annotations

from pathlib import Path

import pytest

from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_layer, make_model_graph
from nerve.model_transpiler_types import ModelTranspileError
from nerve.circuit_lowering_system import build_system_circuits
from nerve.circuit_ir import validate_circuit
from nerve.model_package import (
    ROW_MAJOR_LAYOUT,
    compile_shader_artifacts,
    copy_shader_templates,
    shader_file_for_node,
    workgroup_count_x_for_node,
)


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
        "multiplicity": 4,
        "sinkhorn_iterations": 20,
        "epsilon": 1e-6,
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
    system = build_system_circuits(graph)
    input_circuit = system["input_transducer"]
    [input_adapter] = system["pre_processors"]
    [output_adapter] = system["post_processors"]
    output_circuit = system["output_transducer"]
    assert input_circuit["boundary"]["outputs"][0]["shape"] == [8]
    assert [node["op"] for node in input_circuit["nodes"]] == [
        "embedding_lookup",
    ]
    assert input_adapter["boundary"]["inputs"][0]["shape"] == [8]
    assert input_adapter["boundary"]["outputs"][0]["shape"] == [4, 8]
    assert input_adapter["nodes"][0]["op"] == "repeat_stream_lanes"
    assert output_adapter["boundary"]["inputs"][0]["shape"] == [4, 8]
    assert output_adapter["boundary"]["outputs"][0]["shape"] == [8]
    assert output_adapter["nodes"][0]["op"] == "sinkhorn_hyper_connection_head"
    assert output_circuit["boundary"]["inputs"][0]["shape"] == [8]
    assert [node["op"] for node in output_circuit["nodes"]] == [
        "rms_norm",
        "linear_projection",
    ]


def test_rejects_incomplete_hyper_connection_tensor_set() -> None:
    config, tensors = _source()
    del tensors["layers.0.hc_ffn_scale"]

    with pytest.raises(ModelTranspileError, match="incomplete hyper-connection"):
        discover_model_structure(Path("synthetic"), config, tensors)


def test_system_circuit_canonicalizes_disabled_top_k_to_runtime_zero() -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    graph = make_model_graph(structure, Path("transpiled"), {"source": {}})
    graph["sampling"] = {
        "method": "temperature_top_p",
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }

    sampler = build_system_circuits(graph)["sampler"]

    assert sampler["nodes"][0]["attrs"]["top_k"] == 0


def test_compiles_stream_boundary_adapters_as_normal_processors(
    tmp_path: Path,
) -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    graph = make_model_graph(structure, Path("transpiled"), {"source": {}})
    system = build_system_circuits(graph)
    [input_adapter] = system["pre_processors"]
    [output_adapter] = system["post_processors"]
    tensor_index = {
        "tensors": {
            name: {**tensor, "layout": ROW_MAJOR_LAYOUT}
            for name, tensor in tensors.items()
        }
    }

    assert validate_circuit(input_adapter).ok
    assert validate_circuit(output_adapter).ok
    input_node = input_adapter["nodes"][0]
    output_node = output_adapter["nodes"][0]
    input_shader = shader_file_for_node(
        input_adapter, input_node, tensor_index, graph["dimensions"]
    )
    output_shader = shader_file_for_node(
        output_adapter, output_node, tensor_index, graph["dimensions"]
    )

    assert input_shader == "repeat_stream_lanes_bf16_m4_h8.comp"
    assert output_shader == "hyper_head_block_b1_m4_h8_eps1e-06.comp"
    assert workgroup_count_x_for_node(input_adapter, input_node, tensor_index) == 1
    assert workgroup_count_x_for_node(output_adapter, output_node, tensor_index) == 1
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {input_shader, output_shader})
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / input_shader.replace(".comp", ".spv")).is_file()
    assert (tmp_path / output_shader.replace(".comp", ".spv")).is_file()
