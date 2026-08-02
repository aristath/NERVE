from __future__ import annotations

from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.circuit_ir import validate_circuit
from nerve.circuit_lowering import lower_parallel_markov_draft_graph
from nerve.circuit_lowering_system import build_draft_system_circuits
from nerve.model_transpiler_discovery import discover_model_structure
from nerve.model_transpiler_graph import make_model_graph
from nerve.model_transpiler_types import ModelTranspileError
from nerve.model_package import (
    ROW_MAJOR_LAYOUT,
    compile_shader_artifacts,
    copy_shader_templates,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_validation import _validate_speculative_source_taps


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
        "minimum_draft_tokens": 5,
        "default_draft_tokens": 5,
        "noise_token_id": 31,
        "sampling": "greedy",
        "confidence_prefix": "first_sigmoid_below_runtime_threshold",
        "verification": "lossless_target_longest_matching_prefix",
    }
    assert graph["input_adapter"]["params"] == {
        "token_embedding": {"tensor": "embed.weight"},
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


def test_lowers_parallel_markov_boundaries_and_sequential_dependency() -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    model = make_model_graph(structure, Path("transpiled"), {"source": {}})
    [draft] = model["graph"]["draft_execution_graphs"]

    input_circuit, output_circuit = build_draft_system_circuits(model, draft)

    assert validate_circuit(input_circuit).ok
    assert validate_circuit(output_circuit).ok
    assert [port["id"] for port in input_circuit["boundary"]["inputs"]] == [
        "anchor_token_id",
        "target_hidden_0",
        "target_hidden_1",
        "target_hidden_2",
    ]
    assert [port["shape"] for port in input_circuit["boundary"]["inputs"][1:]] == [
        [4, 8],
        [4, 8],
        [4, 8],
    ]
    assert [port["signal"] for port in input_circuit["boundary"]["inputs"][1:]] == [
        "frame",
        "frame",
        "frame",
    ]
    assert [port["id"] for port in input_circuit["boundary"]["outputs"]] == [
        "query_frames",
        "main_context",
        "anchor_token_passthrough",
    ]
    query_output = input_circuit["boundary"]["outputs"][0]
    assert query_output["shape"] == [5, 4, 8]
    assert input_circuit["boundary"]["outputs"][2]["source"] == "anchor_token_id"
    concatenations = [
        node for node in input_circuit["nodes"] if node["op"] == "concatenate"
    ]
    assert [node["attrs"]["part_widths"] for node in concatenations] == [
        [8, 8],
        [16, 8],
    ]
    query_block = next(
        node for node in input_circuit["nodes"] if node["id"] == "query_embedding_block"
    )
    assert query_block["op"] == "anchor_noise_embedding_block"
    assert query_block["params"] == ["token_embedding"]
    assert query_block["attrs"] == {
        "minimum_block_size": 5,
        "block_size": 5,
        "noise_token_id": 31,
        "anchor_position": 0,
        "runtime_extensible": False,
        "runtime_selectable_prefix": True,
        "hidden_size": 8,
        "stream_multiplicity": 4,
        "output_layout": "block_stream_hidden",
    }
    assert all(node["op"] != "identity" for node in input_circuit["nodes"])
    markov = next(
        node
        for node in output_circuit["nodes"]
        if node["id"] == "markov_argmax_partials_00"
    )
    assert markov["inputs"] == ["base_logits", "anchor_token_id"]
    assert markov["attrs"] == {
        "rank": 2,
        "sampling": "greedy",
        "dependency": "previous_sampled_token",
        "position": 0,
        "block_width": 5,
        "vocabulary_size": 32,
        "vocabulary_tile_width": 256,
        "output_element_bytes": [4, 2],
    }
    reductions = [
        node
        for node in output_circuit["nodes"]
        if node["op"] == "argmax_candidate_reduce"
    ]
    assert len(reductions) == 5
    assert reductions[-1]["outputs"] == ["draft_token_04"]
    assert next(
        node for node in output_circuit["nodes"] if node["id"] == "draft_token_pack"
    )["inputs"] == [f"draft_token_{index:02d}" for index in range(5)]
    assert all(
        node["op"] != "sequential_markov_greedy" for node in output_circuit["nodes"]
    )
    assert [port["id"] for port in output_circuit["boundary"]["outputs"]] == [
        "draft_token_ids",
        "confidence_logits",
    ]


def test_lowers_proposal_execution_mode_from_graph_semantics() -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    model = make_model_graph(structure, Path("transpiled"), {"source": {}})
    [draft] = model["graph"]["draft_execution_graphs"]
    input_ref = {"id": "input", "runtime_role": "draft_input_adapter"}
    output_ref = {"id": "output", "runtime_role": "draft_output_transducer"}
    layer_refs = [
        {"id": f"processor_{index}", "runtime_role": "draft_processor"}
        for index in range(3)
    ]

    lowered = lower_parallel_markov_draft_graph(
        draft,
        layer_refs=layer_refs,
        input_ref=input_ref,
        output_ref=output_ref,
    )

    assert lowered["execution_contract"] == {
        "mode": "parallel_block",
        "block_width": 5,
        "processor_schedule": "parallel_lanes",
        "output_schedule": "compiled_component_graph",
    }


def test_compiles_anchor_noise_embedding_block_as_one_copy_kernel(
    tmp_path: Path,
) -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    model = make_model_graph(structure, Path("transpiled"), {"source": {}})
    [draft] = model["graph"]["draft_execution_graphs"]
    input_circuit, _ = build_draft_system_circuits(model, draft)
    node = next(
        item
        for item in input_circuit["nodes"]
        if item["op"] == "anchor_noise_embedding_block"
    )
    mean_node = next(
        item for item in input_circuit["nodes"] if item["op"] == "mean_stream_lanes"
    )
    tensor_index = {
        "tensors": {
            name: {**tensor, "layout": ROW_MAJOR_LAYOUT}
            for name, tensor in tensors.items()
        }
    }

    shader_file = shader_file_for_node(
        input_circuit,
        node,
        tensor_index,
        model["dimensions"],
    )
    mean_shader_file = shader_file_for_node(
        input_circuit,
        mean_node,
        tensor_index,
        model["dimensions"],
    )

    assert shader_file == "anchor_noise_embedding_block_b5_m4_h8_noise31.comp"
    assert mean_shader_file == "mean_stream_lanes_bf16_m4_h8.comp"
    assert workgroup_count_x_for_node(input_circuit, node, tensor_index) == 2
    assert workgroup_count_x_for_node(input_circuit, mean_node, tensor_index) == 1
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader_file, mean_shader_file})
    shader = (tmp_path / shader_file).read_text()
    assert "token_id * HIDDEN_WORDS" in shader
    assert "frame_index == 0u" in shader
    assert "stream_index" in shader
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / shader_file.replace(".comp", ".spv")).is_file()
    assert (tmp_path / mean_shader_file.replace(".comp", ".spv")).is_file()

    node["attrs"]["noise_token_id"] = 32
    with pytest.raises(ModelCompileError, match="invalid contract"):
        shader_file_for_node(
            input_circuit,
            node,
            tensor_index,
            model["dimensions"],
        )


def test_compiles_complete_parallel_markov_output_schedule(tmp_path: Path) -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    model = make_model_graph(structure, Path("transpiled"), {"source": {}})
    [draft] = model["graph"]["draft_execution_graphs"]
    _, output_circuit = build_draft_system_circuits(model, draft)
    tensor_index = {
        "tensors": {
            name: {**tensor, "layout": ROW_MAJOR_LAYOUT}
            for name, tensor in tensors.items()
        }
    }

    shader_files = {
        node["id"]: shader_file_for_node(
            output_circuit,
            node,
            tensor_index,
            model["dimensions"],
        )
        for node in output_circuit["nodes"]
    }

    assert shader_files == {
        "stream_head": "hyper_head_block_b5_m4_h8_eps1e-06.comp",
        "output_norm": "rms_norm_block_b5_bf16_h8_eps1e-06_offset0.comp",
        "base_projection": "linear_projection_block_b5_bf16_8x32_f32.comp",
        **{
            f"markov_argmax_partials_{position:02d}": (
                f"markov_argmax_partials_b5_p{position}_v32_r2_t256.comp"
            )
            for position in range(5)
        },
        **{
            f"markov_argmax_reduce_{position:02d}": ("argmax_candidate_reduce_c1.comp")
            for position in range(5)
        },
        "confidence_projection": "confidence_projection_block_b5_bf16_h8_r2.comp",
        "draft_token_pack": "pack_token_block_b5.comp",
    }
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, set(shader_files.values()))
    compile_shader_artifacts(tmp_path)
    assert all(
        (tmp_path / shader_file.replace(".comp", ".spv")).is_file()
        for shader_file in shader_files.values()
    )


def test_compiles_source_owned_recommended_parallel_draft_width(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text(
        "Launch with {'num_speculative_tokens': 7} for the recommended path."
    )
    config, tensors = _source()

    structure = discover_model_structure(tmp_path, config, tensors)
    [draft] = structure.draft_execution_graphs
    graph = make_model_graph(structure, Path("transpiled"), {"source": {}})["graph"][
        "draft_execution_graphs"
    ][0]

    assert draft.attributes["proposal_contract"]["minimum_draft_tokens"] == 5
    assert draft.attributes["proposal_contract"]["default_draft_tokens"] == 7
    assert graph["query_block"]["block_size"] == 7


def test_lowers_explicit_query_context_and_target_feature_wiring() -> None:
    config, tensors = _source()
    structure = discover_model_structure(Path("synthetic"), config, tensors)
    [draft] = make_model_graph(structure, Path("transpiled"), {"source": {}})["graph"][
        "draft_execution_graphs"
    ]
    input_ref = {
        "id": "draft_00_input_adapter",
        "runtime_role": "draft_input_adapter",
    }
    layer_refs = [
        {
            "id": f"draft_00_layer_{index:02d}",
            "runtime_role": "draft_processor",
        }
        for index in range(3)
    ]
    output_ref = {
        "id": "draft_00_output_transducer",
        "runtime_role": "draft_output_transducer",
    }

    lowered = lower_parallel_markov_draft_graph(
        draft,
        layer_refs=layer_refs,
        input_ref=input_ref,
        output_ref=output_ref,
    )

    assert lowered["topology"] == "explicit_graph"
    assert len(lowered["edges"]) == 8
    assert [edge["connection"] for edge in lowered["edges"][:4]] == [
        {"kind": "parallel_block_scatter", "width": 5},
        {"kind": "forward"},
        {"kind": "forward"},
        {"kind": "parallel_block_gather", "width": 5},
    ]
    context_edges = [
        edge
        for edge in lowered["edges"]
        if edge["connection"]["kind"] == "shared_context"
    ]
    assert [edge["destination"]["component_id"] for edge in context_edges] == [
        "draft_00_layer_00",
        "draft_00_layer_01",
        "draft_00_layer_02",
    ]
    assert all(
        edge["connection"]["state_update"] == "committed_target_only"
        for edge in context_edges
    )
    assert [
        item["source_tap"]
        for item in lowered["boundary"]["external_inputs"]
        if "source_tap" in item
    ] == [
        {
            "component_id": f"layer_{index:02d}",
            "port_id": "output_frame",
            "instance_selection": "last_in_execution_order",
        }
        for index in range(3)
    ]


def _source_tap_validation_fixture() -> tuple[dict, dict, dict]:
    graph = {
        "boundary": {
            "external_inputs": [
                {
                    "id": "target_hidden_0",
                    "source_tap": {
                        "component_id": "target_processor",
                        "port_id": "output_frame",
                        "instance_selection": "last_in_execution_order",
                    },
                    "endpoint": {
                        "component_id": "draft_input",
                        "port_id": "target_hidden_0",
                    },
                }
            ]
        }
    }
    decoder_circuits = {
        "draft_input": {
            "boundary": {
                "inputs": [
                    {
                        "id": "target_hidden_0",
                        "signal": "frame",
                        "shape": [4, 16],
                    }
                ]
            }
        }
    }
    target_circuits = {
        "target_processor": {
            "boundary": {
                "outputs": [
                    {
                        "id": "output_frame",
                        "signal": "frame",
                        "shape": [4, 16],
                    }
                ]
            }
        }
    }
    return graph, decoder_circuits, target_circuits


def test_validates_typed_speculative_source_tap_geometry() -> None:
    graph, decoder_circuits, target_circuits = _source_tap_validation_fixture()

    _validate_speculative_source_taps("draft", graph, decoder_circuits, target_circuits)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unknown_component", "unknown target output"),
        ("wrong_shape", "incompatible geometry"),
        ("wrong_selection", "unsupported instance selection"),
    ],
)
def test_rejects_invalid_speculative_source_taps(mutation: str, expected: str) -> None:
    graph, decoder_circuits, target_circuits = _source_tap_validation_fixture()
    source_tap = graph["boundary"]["external_inputs"][0]["source_tap"]
    if mutation == "unknown_component":
        source_tap["component_id"] = "missing"
    elif mutation == "wrong_shape":
        target_circuits["target_processor"]["boundary"]["outputs"][0]["shape"] = [16]
    else:
        source_tap["instance_selection"] = "first"

    with pytest.raises(ModelCompileError, match=expected):
        _validate_speculative_source_taps(
            "draft", graph, decoder_circuits, target_circuits
        )
