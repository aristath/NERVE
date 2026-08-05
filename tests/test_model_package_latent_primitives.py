from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nerve.model_package import (
    ModelCompileError,
    ROW_MAJOR_LAYOUT,
    compile_shader_artifacts,
    copy_shader_templates,
    local_size_x_for_shader_file,
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_batching import (
    causal_scan_batch_shader_file,
    causal_scan_workgroup_count_x,
)


def _fixture() -> tuple[dict[str, object], dict[str, object]]:
    nodes = [
        {
            "id": "remember",
            "op": "rolling_state_update",
            "inputs": ["current_kv", "local_kv_memory"],
            "outputs": ["local_kv_values"],
            "state_reads": ["local_kv_memory"],
            "state_writes": ["local_kv_memory"],
            "attrs": {"update": "ring_append", "capacity": 4},
        },
        {
            "id": "derotate",
            "op": "inverse_rotary_position_embedding",
            "inputs": ["attention_heads"],
            "outputs": ["attention_unpositioned"],
            "attrs": {
                "position_source": "stream_tick",
                "position_offset": 1,
                "theta": 10_000.0,
                "rope_type": "default",
                "scaling": None,
                "interleaved": False,
                "rotary_width": 32,
                "head_count": 2,
                "head_width": 64,
            },
        },
        {
            "id": "group_project",
            "op": "grouped_linear",
            "inputs": ["attention_unpositioned"],
            "outputs": ["attention_ranked"],
            "params": ["group_weight", "group_scale"],
            "attrs": {"groups": 2, "rank_per_group": 32},
        },
        {
            "id": "bounded_activation",
            "op": "bounded_silu_multiply",
            "inputs": ["gate", "up"],
            "outputs": ["hidden"],
            "attrs": {"element_count": 128, "limit": 10.0},
        },
    ]
    circuit = {
        "id": "latent_primitives",
        "boundary": {"controls": []},
        "state_ports": [
            {
                "id": "local_kv_memory",
                "type": "rolling_attention_memory",
                "shape_per_token": [128],
                "capacity": 4,
                "dtype": "BF16",
            }
        ],
        "parameters": {
            "refs": {
                "group_weight": {"tensor": "group.weight"},
                "group_scale": {"tensor": "group.scale"},
            }
        },
        "nodes": nodes,
    }
    tensor_index = {
        "tensors": {
            "group.weight": {
                "dtype": "F8_E4M3",
                "shape": [64, 128],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "group.scale": {
                "dtype": "F8_E8M0",
                "shape": [1, 1],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }
    return circuit, tensor_index


def test_compiles_latent_attention_primitives(tmp_path: Path) -> None:
    circuit, tensor_index = _fixture()
    shaders = {
        node["id"]: shader_file_for_node(
            circuit, node, tensor_index, {"hidden_size": 128}
        )
        for node in circuit["nodes"]
    }

    assert shaders == {
        "remember": "rolling_state_ring_append_bf16_4x128__sc6.comp",
        "derotate": (
            "inverse_rotary_bf16_2x64_r32_theta10000_half_po1__sc2.comp"
        ),
        "group_project": (
            "grouped_linear_fp8_e4m3_se8m0_b64x128_g2_256x64.comp"
        ),
        "bounded_activation": "bounded_silu_multiply_bf16_128_limit10.comp",
    }
    group_node = circuit["nodes"][2]
    assert workgroup_count_x_for_node(circuit, group_node, tensor_index) == 4
    assert local_size_x_for_shader_file(shaders["group_project"], group_node) == 1024

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, set(shaders.values()))
    rendered = {
        name: (tmp_path / shader).read_text() for name, shader in shaders.items()
    }
    assert "uint slot = stream_control.words[1] % FRAME_COUNT;" in rendered[
        "remember"
    ]
    assert "for (uint frame" not in rendered["remember"]
    assert "const float ROPE_DIRECTION = -1.0;" in rendered["derotate"]
    assert "const int POSITION_OFFSET = 1;" in rendered["derotate"]
    assert "uint offset = (group * INPUT_SIZE + column) >> 1u;" in rendered[
        "group_project"
    ]
    assert "gate = min(gate, LIMIT);" in rendered["bounded_activation"]
    assert all("{{" not in source for source in rendered.values())
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == 4


@pytest.mark.parametrize(
    ("node_id", "mutation", "message"),
    [
        (
            "remember",
            lambda circuit, _tensors: circuit["state_ports"][0].update(capacity=3),
            "incompatible state geometry",
        ),
        (
            "derotate",
            lambda circuit, _tensors: circuit["nodes"][1]["attrs"].update(
                rotary_width=65
            ),
            "invalid contract",
        ),
        (
            "group_project",
            lambda circuit, _tensors: circuit["nodes"][2]["attrs"].update(
                rank_per_group=31
            ),
            "invalid contract",
        ),
        (
            "group_project",
            lambda _circuit, tensors: tensors["tensors"]["group.scale"].update(
                shape=[1, 2]
            ),
            "requires 128-column blocks",
        ),
        (
            "bounded_activation",
            lambda circuit, _tensors: circuit["nodes"][3]["attrs"].update(limit=0.0),
            "invalid contract",
        ),
    ],
)
def test_rejects_malformed_latent_attention_primitives(
    node_id: str, mutation, message: str
) -> None:
    circuit, tensor_index = _fixture()
    mutation(circuit, tensor_index)
    node = next(node for node in circuit["nodes"] if node["id"] == node_id)

    with pytest.raises(ModelCompileError, match=message):
        shader_file_for_node(circuit, node, tensor_index, {"hidden_size": 128})


def test_forward_rope_keeps_zero_offset_filename_stable() -> None:
    circuit, tensor_index = _fixture()
    node = deepcopy(circuit["nodes"][1])
    node["id"] = "rotate"
    node["op"] = "rotary_position_embedding"
    node["attrs"].pop("position_offset")

    assert shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 128}
    ) == "rotary_bf16_2x64_r32_theta10000_half__sc2.comp"


def test_partial_rope_can_target_the_tail_of_each_head(tmp_path: Path) -> None:
    circuit, tensor_index = _fixture()
    node = deepcopy(circuit["nodes"][1])
    node["attrs"]["rotary_scope"] = "tail"

    shader = shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 128}
    )
    assert shader == (
        "inverse_rotary_bf16_2x64_r32_theta10000_half_tail_po1__sc2.comp"
    )
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader})
    rendered = (tmp_path / shader).read_text()
    assert "const uint ROTARY_OFFSET = HEAD_WIDTH - ROTARY_WIDTH;" in rendered
    assert "dim < ROTARY_OFFSET || dim >= ROTARY_OFFSET + ROTARY_WIDTH" in rendered
    assert "{{" not in rendered
    compile_shader_artifacts(tmp_path)


def test_compiles_local_kv_rope_with_exact_fp8_qat_contract(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _fixture()
    node = deepcopy(circuit["nodes"][1])
    node["id"] = "position_and_quantize_local_kv"
    node["op"] = "rotary_position_embedding"
    node["attrs"].update(
        {
            "head_count": 1,
            "head_width": 512,
            "rotary_width": 64,
            "rotary_scope": "tail",
            "position_offset": 0,
            "activation_quantization": {
                "format": "fp8_e4m3",
                "scale_format": "e8m0_power_of_two",
                "block_columns": 64,
                "scope": "non_rotary_dimensions",
                "mode": "quantize_dequantize",
            },
        }
    )

    shader = shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 128}
    )
    assert shader == (
        "rotary_qdq_fp8_e4m3_spow2_b64_bf16_1x512_r64_"
        "theta10000_half_tail__sc2.comp"
    )
    temporal_shader = causal_scan_batch_shader_file(shader)
    assert temporal_shader == (
        "rotary_qdq_fp8_e4m3_spow2_b64_temporal_bf16_1x512_r64_"
        "theta10000_half_tail.comp"
    )
    assert causal_scan_workgroup_count_x(shader) == 1

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader, temporal_shader})
    rendered = (tmp_path / shader).read_text()
    assert "const uint NON_ROTARY_WIDTH = HEAD_WIDTH - ROTARY_WIDTH;" in rendered
    assert "const uint ROTARY_OFFSET = NON_ROTARY_WIDTH;" in rendered
    assert "exp2(ceil(log2(max(maximum, 1e-4) / 448.0)))" in rendered
    assert "fe4m3vec4" in rendered
    assert "dim < NON_ROTARY_WIDTH" in rendered
    assert "{{" not in rendered
    temporal_rendered = (tmp_path / temporal_shader).read_text()
    assert "batch_control.start_stream_tick_low + position" in temporal_rendered
    assert "{{" not in temporal_rendered
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / shader.replace(".comp", ".spv")).is_file()
    assert (tmp_path / temporal_shader.replace(".comp", ".spv")).is_file()


def test_rms_norm_uses_its_parameter_width_instead_of_model_hidden_size() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "query_norm": {"tensor": "query_norm.weight"},
                "kv_norm": {"tensor": "kv_norm.weight"},
            }
        }
    }
    tensor_index = {
        "tensors": {
            "query_norm.weight": {"dtype": "BF16", "shape": [1024]},
            "kv_norm.weight": {"dtype": "BF16", "shape": [512]},
        }
    }
    node = lambda node_id, parameter: {
        "id": node_id,
        "op": "rms_norm",
        "inputs": [f"{node_id}_input"],
        "outputs": [f"{node_id}_output"],
        "params": [parameter],
        "attrs": {"eps": 1e-6, "weight_offset": 0.0},
    }

    assert shader_file_for_node(
        circuit,
        node("query_input_norm", "query_norm"),
        tensor_index,
        {"hidden_size": 4096},
    ) == "rms_norm_bf16_h1024_eps1e-06_offset0.comp"
    assert shader_file_for_node(
        circuit,
        node("key_value_norm", "kv_norm"),
        tensor_index,
        {"hidden_size": 4096},
    ) == "rms_norm_bf16_h512_eps1e-06_offset0.comp"


def test_rms_norm_keeps_parameter_width_with_fused_quantized_outputs() -> None:
    circuit = {
        "parameters": {"refs": {"norm": {"tensor": "norm.weight"}}}
    }
    tensor_index = {
        "tensors": {"norm.weight": {"dtype": "BF16", "shape": [1024]}}
    }
    node = {
        "id": "norm",
        "op": "rms_norm",
        "inputs": ["input"],
        "outputs": ["normalized", "normalized_fp8", "normalized_scale"],
        "params": ["norm"],
        "attrs": {
            "eps": 1e-6,
            "weight_offset": 0.0,
            "physical_output_representations": [
                {
                    "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
                    "logical_signal": "normalized",
                    "element_count": 1024,
                    "block_columns": 128,
                }
            ],
        },
    }

    assert shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 4096}
    ) == "rms_norm_quantize_fp8_e4m3_b128_h1024_eps1e-06_offset0.comp"

    node["attrs"]["physical_output_representations"][0]["contract"] = (
        "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
    )
    assert shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 4096}
    ) == (
        "rms_norm_quantize_fp8_e4m3_spow2_b128_h1024_"
        "eps1e-06_offset0.comp"
    )


def test_compiles_parallel_block_latent_attention(tmp_path: Path) -> None:
    circuit, tensor_index = _fixture()
    circuit["parameters"]["refs"]["attention_sinks"] = {
        "tensor": "attention.sinks"
    }
    tensor_index["tensors"]["attention.sinks"] = {
        "dtype": "F32",
        "shape": [2],
        "layout": ROW_MAJOR_LAYOUT,
    }
    node = {
        "id": "attend",
        "op": "indexed_sparse_attention",
        "inputs": ["query", "local_kv_values", "query_kv"],
        "outputs": ["attention_heads"],
        "params": ["attention_sinks"],
        "attrs": {
            "causal": False,
            "scale": 0.125,
            "window_size": 4,
            "intra_block_visibility": "all",
            "query_state": "transient",
            "query_heads": 2,
            "key_value_heads": 1,
            "head_width": 64,
        },
    }
    circuit["nodes"].append(node)

    shader_file = shader_file_for_node(
        circuit, node, tensor_index, {"hidden_size": 128}
    )

    assert shader_file == (
        "indexed_sparse_attention_bf16_q2_kv1_d64_w4_"
        "scale0.125__sc6.comp"
    )
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 2
    assert local_size_x_for_shader_file(shader_file, node) == 64
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    parallel_shader_file = shader_file.replace(
        "indexed_sparse_attention_", "indexed_sparse_attention_parallel_", 1
    ).replace("__sc6", "")
    copy_shader_templates(
        shader_source_dir, tmp_path, {shader_file, parallel_shader_file}
    )
    source = (tmp_path / shader_file).read_text()
    assert "const uint LOCAL_WINDOW = 4u;" in source
    assert "const uint MAX_PARALLEL_BLOCK = 64u;" in source
    assert "uint slot = absolute_tick % capacity;" in source
    assert "uintBitsToFloat(attention_sinks.words[query_head])" in source
    assert "shared float score_partials[MAX_SUBGROUP_COUNT];" in source
    assert "subgroup < gl_NumSubgroups" in source
    assert "token * SUBGROUP_COUNT" not in source
    assert all("{{" not in line for line in source.splitlines())
    parallel_source = (tmp_path / parallel_shader_file).read_text()
    assert "uint batch_width;" in parallel_source
    assert "return gl_WorkGroupID.y;" in parallel_source
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / shader_file.replace(".comp", ".spv")).is_file()
    assert (tmp_path / parallel_shader_file.replace(".comp", ".spv")).is_file()


def test_rejects_indexed_attention_without_f32_per_head_sinks() -> None:
    circuit, tensor_index = _fixture()
    circuit["parameters"]["refs"]["attention_sinks"] = {
        "tensor": "attention.sinks"
    }
    tensor_index["tensors"]["attention.sinks"] = {
        "dtype": "BF16",
        "shape": [1],
        "layout": ROW_MAJOR_LAYOUT,
    }
    node = {
        "id": "attend",
        "op": "indexed_sparse_attention",
        "inputs": ["query", "local_kv_values", "query_kv"],
        "outputs": ["attention_heads"],
        "params": ["attention_sinks"],
        "attrs": {
            "causal": False,
            "scale": 0.125,
            "window_size": 4,
            "intra_block_visibility": "all",
            "query_state": "transient",
            "query_heads": 2,
            "key_value_heads": 1,
            "head_width": 64,
        },
    }
    circuit["nodes"].append(node)

    with pytest.raises(ModelCompileError, match="invalid contract"):
        shader_file_for_node(circuit, node, tensor_index, {"hidden_size": 128})


def test_compiles_stateful_learned_compressor_as_typed_pool_and_finalize_stages(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _fixture()
    circuit["state_ports"].append(
        {
            "id": "compressor_accumulator",
            "type": "gated_pooling_memory",
            "shape": [2, 8, 256],
            "dtype": "F32",
            "update": "position_biased_softmax_pool",
        }
    )
    circuit["state_ports"].append(
        {
            "id": "compressed_kv_memory",
            "type": "append_only_attention_memory",
            "shape_per_token": [128],
            "dtype": "BF16",
            "growth": "one_per_4_activations",
        }
    )
    circuit["parameters"]["refs"].update(
        {
            "compressor_position_bias": {"tensor": "compressor.ape"},
            "compressor_kv_projection": {"tensor": "compressor.wkv"},
            "compressor_gate_projection": {"tensor": "compressor.wgate"},
            "compressor_norm": {"tensor": "compressor.norm"},
        }
    )
    tensor_index["tensors"].update(
        {
            "compressor.ape": {
                "dtype": "F32",
                "shape": [4, 256],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "compressor.wkv": {
                "dtype": "BF16",
                "shape": [256, 128],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "compressor.wgate": {
                "dtype": "BF16",
                "shape": [256, 128],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "compressor.norm": {
                "dtype": "BF16",
                "shape": [128],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    )
    pool = {
        "id": "memory_compressor_pool",
        "op": "learned_gated_kv_pool",
        "inputs": ["current_kv", "compressor_accumulator"],
        "outputs": ["compressed_pooled_f32"],
        "params": [
            "compressor_position_bias",
            "compressor_kv_projection",
            "compressor_gate_projection",
        ],
        "state_reads": ["compressor_accumulator"],
        "state_writes": ["compressor_accumulator"],
        "attrs": {
            "ratio": 4,
            "overlap": True,
            "lane_coefficient": 2,
            "pooling": "learned_position_biased_softmax",
            "hidden_size": 128,
            "head_width": 128,
            "output_element_bytes": [4],
        },
    }
    finalize = {
        "id": "memory_compressor_finalize",
        "op": "compressed_kv_finalize",
        "inputs": ["compressed_pooled_f32"],
        "outputs": ["compressed_candidate"],
        "params": ["compressor_norm"],
        "attrs": {
            "position_source": "stream_tick",
            "theta": 10_000.0,
            "rope_type": "default",
            "scaling": None,
            "interleaved": False,
            "rotary_width": 64,
            "rotary_scope": "tail",
            "head_count": 1,
            "head_width": 128,
            "query_heads": 1,
            "key_value_heads": 1,
            "normalization_epsilon": 1e-6,
            "position_offset": -3,
            "activation_quantization": {
                "format": "fp8_e4m3",
                "scale_format": "e8m0_power_of_two",
                "block_columns": 64,
                "scope": "non_rotary_dimensions",
                "mode": "quantize_dequantize",
            },
            "output_element_bytes": [2],
        },
    }
    append = {
        "id": "compressed_memory_update",
        "op": "conditional_append_state_update",
        "inputs": ["compressed_candidate", "compressed_kv_memory"],
        "outputs": ["compressed_kv_values"],
        "state_reads": ["compressed_kv_memory"],
        "state_writes": ["compressed_kv_memory"],
        "attrs": {"period": 4},
    }
    chronological = {
        "id": "compressed_memory_indexer",
        "op": "chronological_compressed_index",
        "inputs": ["compressed_kv_values"],
        "outputs": ["compressed_indices"],
        "attrs": {
            "ratio": 4,
            "causal": True,
            "index_offset": 128,
            "max_indices": 1024,
            "output_element_bytes": [4],
        },
    }
    circuit["nodes"].extend([pool, finalize, append, chronological])

    pool_file = shader_file_for_node(
        circuit, pool, tensor_index, {"hidden_size": 128}
    )
    finalize_file = shader_file_for_node(
        circuit, finalize, tensor_index, {"hidden_size": 128}
    )
    append_file = shader_file_for_node(
        circuit, append, tensor_index, {"hidden_size": 128}
    )
    chronological_file = shader_file_for_node(
        circuit, chronological, tensor_index, {"hidden_size": 128}
    )

    assert pool_file == (
        "learned_gated_kv_pool_bf16_f32_h128_d128_r4_c2__sc8.comp"
    )
    assert finalize_file == (
        "compressed_kv_finalize_f32_bf16_d128_r64_eps1e-06_"
        "theta10000_half_po-3_qfp8e4m3b64__sc3.comp"
    )
    assert append_file == "conditional_append_state_bf16_d128_p4__sc6.comp"
    assert chronological_file == (
        "chronological_compressed_index_u32_m1024_r4_o128__sc3.comp"
    )
    assert workgroup_count_x_for_node(circuit, pool, tensor_index) == 128
    assert workgroup_count_x_for_node(circuit, finalize, tensor_index) == 1
    assert workgroup_count_x_for_node(circuit, append, tensor_index) == 1
    assert workgroup_count_x_for_node(circuit, chronological, tensor_index) == 1
    assert local_size_x_for_shader_file(pool_file, pool) == 64
    assert local_size_x_for_shader_file(finalize_file, finalize) == 128
    assert local_size_x_for_shader_file(append_file, append) == 64
    assert local_size_x_for_shader_file(chronological_file, chronological) == 1024

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {pool_file, finalize_file, append_file, chronological_file},
    )
    pool_source = (tmp_path / pool_file).read_text()
    finalize_source = (tmp_path / finalize_file).read_text()
    append_source = (tmp_path / append_file).read_text()
    chronological_source = (tmp_path / chronological_file).read_text()
    assert "const uint LANE_COEFFICIENT = 2u;" in pool_source
    assert "state_score_index" in pool_source
    assert "if ((position + 1u) % COMPRESSION_RATIO != 0u)" in pool_source
    assert "round_power_of_two_scale" in finalize_source
    assert "const int POSITION_OFFSET = -3;" in finalize_source
    assert "dim < NON_ROTARY_WIDTH" in finalize_source
    assert "uint compressed_slot = position / PERIOD;" in append_source
    assert "indices.values[position] = INDEX_OFFSET + position;" in (
        chronological_source
    )
    assert "{{" not in pool_source
    assert "{{" not in finalize_source
    assert "{{" not in append_source
    assert "{{" not in chronological_source
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / pool_file.replace(".comp", ".spv")).is_file()
    assert (tmp_path / finalize_file.replace(".comp", ".spv")).is_file()
    assert (tmp_path / append_file.replace(".comp", ".spv")).is_file()
    assert (tmp_path / chronological_file.replace(".comp", ".spv")).is_file()


def test_compiles_learned_index_transform_scores_and_exact_radix_topk(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _fixture()
    circuit["state_ports"].append(
        {
            "id": "indexer_kv_memory",
            "type": "append_only_index_memory",
            "shape_per_token": [128],
            "dtype": "BF16",
            "growth": "with_compressed_kv_memory",
        }
    )
    circuit["parameters"]["refs"]["indexer_norm"] = {
        "tensor": "indexer.norm"
    }
    tensor_index["tensors"]["indexer.norm"] = {
        "dtype": "BF16",
        "shape": [128],
        "layout": ROW_MAJOR_LAYOUT,
    }
    transform_attrs = {
        "position_source": "stream_tick",
        "theta": 160_000.0,
        "rope_type": "default",
        "scaling": None,
        "interleaved": False,
        "rotary_width": 64,
        "rotary_scope": "tail",
        "head_count": 64,
        "head_width": 128,
        "query_heads": 64,
        "key_value_heads": 1,
        "rotation": "hadamard",
        "activation_quantization": {
            "format": "fp4_e2m1",
            "scale_format": "e8m0_power_of_two",
            "block_columns": 32,
            "mode": "quantize_dequantize",
        },
        "output_element_bytes": [2],
    }
    query_transform = {
        "id": "indexer_query_transform",
        "op": "index_vector_transform",
        "inputs": ["indexer_query_heads"],
        "outputs": ["indexer_query_transformed"],
        "attrs": transform_attrs,
    }
    finalizer = {
        "id": "indexer_compressor_finalize",
        "op": "compressed_index_kv_finalize",
        "inputs": ["indexer_pooled_f32"],
        "outputs": ["indexer_candidate"],
        "params": ["indexer_norm"],
        "attrs": {
            **transform_attrs,
            "head_count": 1,
            "normalization_epsilon": 1e-6,
            "position_offset": -3,
        },
    }
    append = {
        "id": "indexer_memory_update",
        "op": "conditional_append_state_update",
        "inputs": ["indexer_candidate", "indexer_kv_memory"],
        "outputs": ["indexer_kv_values"],
        "state_reads": ["indexer_kv_memory"],
        "state_writes": ["indexer_kv_memory"],
        "attrs": {"period": 4},
    }
    scores = {
        "id": "indexer_scores",
        "op": "learned_index_scores",
        "inputs": [
            "indexer_query_transformed",
            "indexer_head_weights",
            "indexer_kv_values",
        ],
        "outputs": ["indexer_scores_f32"],
        "attrs": {
            "heads": 64,
            "head_width": 128,
            "ratio": 4,
            "max_compressed_positions": 1024,
            "score_scale": (64 * 128) ** -0.5,
            "score_activation": "relu_then_head_weighted_sum",
            "output_element_bytes": [4],
        },
    }
    topk = {
        "id": "compressed_memory_indexer",
        "op": "radix_topk_index",
        "inputs": ["indexer_scores_f32"],
        "outputs": ["compressed_indices"],
        "attrs": {
            "top_k": 512,
            "ratio": 4,
            "index_offset": 128,
            "max_scores": 1024,
            "ordering": "descending_float_score",
            "output_element_bytes": [4],
        },
    }
    circuit["nodes"].extend([query_transform, finalizer, append, scores, topk])
    dimensions = {"hidden_size": 128}
    files = {
        node["id"]: shader_file_for_node(circuit, node, tensor_index, dimensions)
        for node in (query_transform, finalizer, append, scores, topk)
    }

    assert files == {
        "indexer_query_transform": (
            "index_vector_transform_bf16_h64_d128_r64_theta160000_half_"
            "qfp4e2m1b32__sc2.comp"
        ),
        "indexer_compressor_finalize": (
            "compressed_index_kv_finalize_f32_bf16_d128_r64_eps1e-06_"
            "theta160000_half_po-3_qfp4e2m1b32__sc3.comp"
        ),
        "indexer_memory_update": "conditional_append_state_bf16_d128_p4__sc6.comp",
        "indexer_scores": (
            "learned_index_scores_bf16_f32_h64_d128_r4_m1024_c256_"
            "scale0.0110485435__sc5.comp"
        ),
        "compressed_memory_indexer": (
            "radix_topk_index_f32_u32_m1024_k512_r4_o128__sc2.comp"
        ),
    }
    assert workgroup_count_x_for_node(circuit, query_transform, tensor_index) == 64
    assert workgroup_count_x_for_node(circuit, scores, tensor_index) == 4
    assert workgroup_count_x_for_node(circuit, topk, tensor_index) == 1
    assert local_size_x_for_shader_file(files["indexer_scores"], scores) == 1024
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, set(files.values()))
    score_source = (tmp_path / files["indexer_scores"]).read_text()
    topk_source = (tmp_path / files["compressed_memory_indexer"]).read_text()
    assert "shared uint query_cache[QUERY_WORDS];" in score_source
    assert "uint selected_prefix;" in topk_source
    assert "for (int bit_index = 31; bit_index >= 0; --bit_index)" in topk_source
    assert "shared uint selected_keys[512];" in topk_source
    assert "selected_candidate_precedes" in topk_source
    assert "atomicAdd" not in topk_source
    assert "left_index < right_index" in topk_source
    assert all("{{" not in (tmp_path / name).read_text() for name in files.values())
    compile_shader_artifacts(tmp_path)
    assert len(list(tmp_path.glob("*.spv"))) == len(files)


def test_rejects_radix_topk_that_cannot_be_sorted_by_one_workgroup(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_file = "radix_topk_index_f32_u32_m2048_k1025_r1_o0__sc2.comp"

    with pytest.raises(ModelCompileError, match="invalid radix top-k shape"):
        copy_shader_templates(shader_source_dir, tmp_path, {shader_file})


def test_compiles_causal_indexed_attention_with_compressed_memory(
    tmp_path: Path,
) -> None:
    circuit, tensor_index = _fixture()
    circuit["state_ports"].append(
        {
            "id": "compressed_kv_memory",
            "type": "append_only_attention_memory",
            "shape_per_token": [128],
            "dtype": "BF16",
            "growth": "one_per_4_activations",
        }
    )
    circuit["parameters"]["refs"]["attention_sinks"] = {
        "tensor": "attention.sinks"
    }
    tensor_index["tensors"]["attention.sinks"] = {
        "dtype": "F32",
        "shape": [2],
        "layout": ROW_MAJOR_LAYOUT,
    }
    circuit["nodes"].extend(
        [
            {
                "id": "compress",
                "op": "conditional_append_state_update",
                "inputs": ["candidate", "compressed_kv_memory"],
                "outputs": ["compressed_kv_values"],
                "state_reads": ["compressed_kv_memory"],
                "state_writes": ["compressed_kv_memory"],
                "attrs": {"period": 4},
            },
            {
                "id": "index",
                "op": "chronological_compressed_index",
                "inputs": ["compressed_kv_values"],
                "outputs": ["compressed_indices"],
                "attrs": {"ratio": 4, "causal": True},
            },
        ]
    )
    node = {
        "id": "attend",
        "op": "indexed_sparse_attention",
        "inputs": [
            "query",
            "local_kv_values",
            "compressed_kv_values",
            "compressed_indices",
        ],
        "outputs": ["attention_heads"],
        "params": ["attention_sinks"],
        "attrs": {
            "causal": True,
            "scale": 0.125,
            "window_size": 4,
            "query_heads": 2,
            "key_value_heads": 1,
            "head_width": 64,
        },
    }
    circuit["nodes"].append(node)

    shader_file = shader_file_for_node(
        circuit,
        node,
        tensor_index,
        {"hidden_size": 128, "max_position_embeddings": 4096},
    )

    assert shader_file == (
        "indexed_sparse_attention_main_bf16_q2_kv1_d64_w4_"
        "r4_k1024_scale0.125__sc8.comp"
    )
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(shader_source_dir, tmp_path, {shader_file})
    source = (tmp_path / shader_file).read_text()
    assert "const uint COMPRESSION_RATIO = 4u;" in source
    assert "const uint MAX_COMPRESSED_INDICES = 1024u;" in source
    assert "index - LOCAL_WINDOW < compressed_count" in source
    assert "binding = 8) readonly buffer StreamControl" in source
    assert "shared float score_partials[MAX_SUBGROUP_COUNT];" in source
    assert "subgroup < gl_NumSubgroups" in source
    assert "tile_token * SUBGROUP_COUNT" not in source
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / shader_file.replace(".comp", ".spv")).is_file()
