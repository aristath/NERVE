from copy import deepcopy

from model_package_layout_common import *
from nerve.model_package_batching import sparse_moe_route_scheduling_shader_file
from nerve.model_package_shader_selection import local_size_x_for_shader_file
from nerve.model_package_shader_compiler import compile_shader_artifacts
from nerve.model_package_tensors import physical_input_prequantization_spec

def test_compiler_renders_per_head_softplus_attention_gate(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    primary = "softplus_multiply_bf16_q72_d128_per_head.comp"
    batch = "softplus_multiply_batch16_bf16_q72_d128_per_head.comp"
    node = {
        "id": "attention_output_gate",
        "op": "softplus_multiply",
        "attrs": {"query_heads": 72, "head_width": 128, "per_head": True},
    }

    assert (
        shader_file_for_node(
            {}, node, {}, {"hidden_size": 3072, "intermediate_size": 1024}
        )
        == primary
    )
    copy_shader_templates(shader_source_dir, tmp_path, {primary, batch})

    primary_source = (tmp_path / primary).read_text()
    batch_source = (tmp_path / batch).read_text()
    assert "const uint QUERY_HEADS = 72u;" in primary_source
    assert "const uint HEAD_WIDTH = 128u;" in primary_source
    assert "const bool PER_HEAD = 1 != 0;" in primary_source
    assert "element / HEAD_WIDTH" in primary_source
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch_source
    assert "batch_index * GATE_WORDS" in batch_source
    assert "{{" not in primary_source
    assert "{{" not in batch_source


def test_compiler_renders_sparse_moe_and_scaled_residual_components(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "moe_route_compact_batch1_i512_k8_t256.comp",
        "moe_route_count_batch1_i512_k8_t512.comp",
        "scaled_add_bf16_1024_scale0.22.comp",
        "moe_topk_bf16_e32_k8.comp",
        "sparse_moe_gate_up_bf16_h1024_i512_e32_k8.comp",
        "sparse_moe_gate_up_batch1_bf16_h1024_i512_e32_k8.comp",
        "sparse_moe_down_bf16_h1024_i512_e32_k8.comp",
        "sparse_moe_down_batch1_bf16_h1024_i512_e32_k8.comp",
        "moe_reduce_bf16_h1024_k8_scale1.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    scaled_add = (tmp_path / "scaled_add_bf16_1024_scale0.22.comp").read_text()
    router = (tmp_path / "moe_topk_bf16_e32_k8.comp").read_text()
    gate_up = (tmp_path / "sparse_moe_gate_up_bf16_h1024_i512_e32_k8.comp").read_text()
    down = (tmp_path / "sparse_moe_down_bf16_h1024_i512_e32_k8.comp").read_text()
    reduce = (tmp_path / "moe_reduce_bf16_h1024_k8_scale1.comp").read_text()
    assert "const float RESIDUAL_SCALE = 0.22;" in scaled_add
    assert "const uint NUM_EXPERTS = 32u;" in router
    assert "const uint EXPERTS_PER_TOKEN = 8u;" in router
    assert "buffer SelectionTelemetry" in router
    assert "atomicAdd(selection_telemetry.counts[expert], 1u);" in router
    assert "left * (2u * NUM_EXPERTS - left - 1u) / 2u" in router
    assert "selection_telemetry.counts[NUM_EXPERTS + pair_index]" in router
    assert "shared float router_scores[NUM_EXPERTS];" in router
    assert "shared float top_scores[EXPERTS_PER_TOKEN];" in router
    assert "router_scores[expert] = read_router(expert);" in router
    assert "float score = router_scores[expert];" in router
    assert "router_scores[subgroup_expert] =" in router
    assert "already_selected" not in router
    assert "top_scores[insertion] = top_scores[insertion - 1u];" in router
    assert "if (gl_NumSubgroups == 1u)" in router
    assert "subgroup_best_scores[gl_SubgroupID]" in router
    assert "const uint INTERMEDIATE_SIZE = 512u;" in gate_up
    assert "const uint INTERMEDIATE_SIZE = 512u;" in down
    assert "const uint HIDDEN_SIZE = 1024u;" in reduce
    assert "route >= EXPERTS_PER_TOKEN" in gate_up
    assert "route >= EXPERTS_PER_TOKEN" in down
    assert "uint expert_count;" in gate_up
    assert "uint expert_count;" in down
    assert "dispatch_control.expert_count != 0u" in gate_up
    assert "dispatch_control.expert_count != 0u" in down
    assert "for (uint route = 0u; route < EXPERTS_PER_TOKEN; route++)" in reduce
    assert "route < NUM_EXPERTS" not in gate_up
    assert "route < NUM_EXPERTS" not in down
    assert "buffer DynamicResourceAddresses" in gate_up
    assert "buffer DynamicParameterSlots" in down
    assert "GL_EXT_buffer_reference2" in gate_up
    assert all(
        "{{" not in (tmp_path / shader_file).read_text() for shader_file in shader_files
    )
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / "moe_route_compact_batch1_i512_k8_t256.spv").is_file()
    assert (tmp_path / "moe_route_count_batch1_i512_k8_t512.spv").is_file()


def test_sparse_gate_reuses_blockwise_fp8_representation() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "moe_input": {"tensor": "experts.gate_up"},
                "moe_input_scale_inv": {"tensor": "experts.gate_up_scale"},
            }
        }
    }
    node = {
        "id": "sparse_moe_gate_up",
        "op": "sparse_moe_gate_up",
        "inputs": ["normalized_fp8", "normalized_scale", "routes"],
        "outputs": ["expert_intermediates"],
        "params": ["moe_input", "moe_input_scale_inv"],
        "attrs": {
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "experts_per_token": 8,
            "physical_input_contract": (
                "bf16_blockwise_fp8_e4m3_f32_scale.v1"
            ),
        },
    }
    tensor_index = {
        "tensors": {
            "experts.gate_up": {
                "dtype": "F8_E4M3",
                "shape": [256, 1024, 2048],
                "layout": "row_major",
            },
            "experts.gate_up_scale": {
                "dtype": "BF16",
                "shape": [256, 8, 16],
                "layout": "row_major",
            },
        }
    }
    expected = (
        "sparse_moe_gate_up_prequant_fp8_e4m3_"
        "b128x128_h2048_i512_e256_k8.comp"
    )

    assert physical_input_prequantization_spec(circuit, node, tensor_index) == {
        "contract": "bf16_blockwise_fp8_e4m3_f32_scale.v1",
        "input_size": 2048,
        "block_columns": 128,
    }
    assert shader_file_for_node(
        circuit,
        node,
        tensor_index,
        {"hidden_size": 2048, "intermediate_size": 512},
    ) == expected
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 128
    assert local_size_x_for_shader_file(expected, node) == 512
    assert frame_parallel_batch_shader_file(expected) == expected.replace(
        "_prequant_", "_batch1_prequant_", 1
    )
    assert sparse_moe_route_scheduling_shader_file(expected) == (
        "moe_route_compact_batch1_i512_k8_t16.comp"
    )
    kernel = component_kernel_spec(
        execution_index=0,
        node=node,
        circuit=circuit,
        shader_file=expected,
        local_size_x=512,
        workgroup_count_x=128,
    )
    assert kernel["batch_implementations"][0]["stages"][0][
        "descriptor_bindings"
    ] == [
        {"binding": 1, "source_binding": 2},
        {"binding": 2, "source_binding": 3},
    ]


def test_sparse_down_reuses_one_route_aware_fp8_intermediate() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "moe_output": {"tensor": "experts.down"},
                "moe_output_scale_inv": {"tensor": "experts.down_scale"},
            }
        }
    }
    node = {
        "id": "sparse_moe_down",
        "op": "sparse_moe_down",
        "inputs": [
            "intermediate_fp8",
            "intermediate_scale",
            "route_map",
            "routes",
        ],
        "outputs": ["expert_outputs"],
        "params": ["moe_output", "moe_output_scale_inv"],
        "attrs": {
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "experts_per_token": 8,
            "physical_input_contract": (
                "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_"
                "u32_route_map.v1"
            ),
        },
    }
    tensor_index = {
        "tensors": {
            "experts.down": {
                "dtype": "F8_E4M3",
                "shape": [256, 2048, 512],
                "layout": "row_major",
            },
            "experts.down_scale": {
                "dtype": "BF16",
                "shape": [256, 16, 4],
                "layout": "row_major",
            },
        }
    }
    expected = (
        "sparse_moe_down_prequant_fp8_e4m3_"
        "b128x128_h2048_i512_e256_k8.comp"
    )

    assert physical_input_prequantization_spec(circuit, node, tensor_index) == {
        "contract": (
            "bf16_sparse_moe_intermediate_blockwise_fp8_e4m3_f32_scale_"
            "u32_route_map.v1"
        ),
        "input_size": 4096,
        "block_columns": 128,
        "experts_per_token": 8,
    }
    assert shader_file_for_node(
        circuit,
        node,
        tensor_index,
        {"hidden_size": 2048, "intermediate_size": 512},
    ) == expected
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 256
    assert local_size_x_for_shader_file(expected, node) == 512
    assert frame_parallel_batch_shader_file(expected) == expected.replace(
        "_down_", "_down_batch1_", 1
    )


def test_compiler_renders_route_aware_sparse_moe_fp8_intermediate(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        (
            "sparse_moe_down_prequant_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        ),
        (
            "sparse_moe_down_batch1_prequant_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        ),
        (
            "sparse_moe_gate_up_prequant_emit_intermediate_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        ),
        (
            "sparse_moe_gate_up_batch1_prequant_emit_intermediate_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        ),
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    down = (
        tmp_path
        / (
            "sparse_moe_down_prequant_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        )
    ).read_text()
    batch_down = (
        tmp_path
        / (
            "sparse_moe_down_batch1_prequant_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        )
    ).read_text()
    emitting_gate_up = (
        tmp_path
        / (
            "sparse_moe_gate_up_prequant_emit_intermediate_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        )
    ).read_text()
    emitting_batch_gate_up = (
        tmp_path
        / (
            "sparse_moe_gate_up_batch1_prequant_emit_intermediate_fp8_e4m3_"
            "b128x128_h2048_i512_e256_k8.comp"
        )
    ).read_text()
    assert "readonly buffer RouteMap" in down
    assert "uint expert_count;" in down
    assert "dispatch_control.expert_count != 0u" in down
    assert "shared fe4m3vec4 cached_intermediate" in down
    assert "subgroupClusteredMax" not in down
    assert "readonly buffer RouteMaps" in batch_down
    assert "const uint TILE_ROWS = 128u;" in emitting_gate_up
    assert "binding = 8) readonly buffer DynamicParameterSlots" in emitting_gate_up
    assert "expert_route_map.values[route] = route;" in emitting_gate_up
    assert "compact_batch * EXPERTS_PER_TOKEN" in emitting_batch_gate_up
    assert "buffer ExpertIntermediates" in emitting_gate_up
    assert "uint expert_count;" in emitting_gate_up
    assert "dispatch_control.expert_count != 0u" in emitting_gate_up
    assert "buffer ExpertIntermediates" in emitting_batch_gate_up
    assert all(
        "{{" not in (tmp_path / shader_file).read_text()
        for shader_file in shader_files
    )
    compile_shader_artifacts(tmp_path)


def test_compiler_renders_sigmoid_router_with_selection_bias(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    circuit = {
        "parameters": {
            "refs": {
                "moe_router_correction_bias": {"tensor": "router.bias"},
            }
        }
    }
    node = {
        "id": "moe_topk",
        "op": "moe_topk",
        "params": ["moe_router_correction_bias"],
        "attrs": {
            "num_experts": 256,
            "experts_per_token": 10,
            "activation": "sigmoid",
            "normalize_selected": True,
            "logit_softcap": 0.0,
            "selection_bias": True,
        },
    }
    tensor_index = {
        "tensors": {
            "router.bias": {
                "dtype": "F32",
                "shape": [256],
                "layout": "row_major",
            }
        }
    }
    primary = "moe_topk_sigmoid_bf16_e256_k10_norm1_cap0_biasf32.comp"
    batch = "moe_topk_batch1_sigmoid_bf16_e256_k10_norm1_cap0_biasf32.comp"
    reduce_node = {
        "id": "moe_reduce",
        "op": "moe_reduce",
        "attrs": {
            "hidden_size": 3072,
            "experts_per_token": 10,
            "routed_scaling_factor": 2.5,
        },
    }
    reduce_file = "moe_reduce_bf16_h3072_k10_scale2.5.comp"

    assert (
        shader_file_for_node(
            circuit,
            node,
            tensor_index,
            {"hidden_size": 3072, "intermediate_size": 1024},
        )
        == primary
    )
    assert frame_parallel_batch_shader_file(primary) == batch
    assert (
        shader_file_for_node(
            {}, reduce_node, {}, {"hidden_size": 3072, "intermediate_size": 1024}
        )
        == reduce_file
    )
    assert workgroup_count_x_for_node({}, reduce_node, {}) == 6
    assert local_size_x_for_shader_file(reduce_file, reduce_node) == 256
    copy_shader_templates(shader_source_dir, tmp_path, {primary, batch, reduce_file})

    primary_source = (tmp_path / primary).read_text()
    batch_source = (tmp_path / batch).read_text()
    assert "const bool ROUTER_SIGMOID = 1 != 0;" in primary_source
    assert "const bool NORMALIZE_SELECTED = 1 != 0;" in primary_source
    assert "ROUTED_SCALE" not in primary_source
    assert "RouterSelectionBias" in primary_source
    assert "uintBitsToFloat(router_selection_bias.words[expert])" in primary_source
    assert "binding = 1) buffer ExpertRoutes" in primary_source
    assert "binding = 2) readonly buffer RouterSelectionBias" in primary_source
    assert "binding = 3) buffer SelectionTelemetry" in primary_source
    assert "atomicAdd(selection_telemetry.counts[expert], 1u);" in primary_source
    assert "selection_telemetry.counts[NUM_EXPERTS + pair_index]" in primary_source
    assert "shared float router_values[NUM_EXPERTS];" in primary_source
    assert "shared uint selected_experts[NUM_EXPERTS];" in primary_source
    assert "router_values[expert] = router_logit(expert);" in primary_source
    assert "selected_experts[expert] = 0u;" in primary_source
    assert "selected_experts[expert] == 0u" in primary_source
    assert "already_selected" not in primary_source
    assert "float logit = router_values[expert];" in primary_source
    assert "if (gl_NumSubgroups == 1u)" in primary_source
    assert "subgroup_best_scores[gl_SubgroupID]" in primary_source
    assert "binding = 2) buffer ExpertRoutes" not in primary_source
    assert "gl_WorkGroupID.y" in batch_source
    assert "binding = 3) buffer SelectionTelemetry" in batch_source
    assert "selection_telemetry.counts[NUM_EXPERTS + pair_index]" in batch_source
    assert (
        "router_values[expert] = router_logit(batch_index, expert);"
        in batch_source
    )
    assert "selected_experts[expert] = 0u;" in batch_source
    assert "already_selected" not in batch_source
    assert "if (gl_NumSubgroups == 1u)" in batch_source
    reduce_source = (tmp_path / reduce_file).read_text()
    assert "layout(local_size_x = 256" in reduce_source
    assert "uint word = gl_GlobalInvocationID.x;" in reduce_source
    assert "word += invocation_count" in reduce_source
    assert "const float ROUTED_SCALE = 2.5;" in reduce_source
    assert "f32_to_bf16(lo * ROUTED_SCALE)" in reduce_source
    assert "{{" not in primary_source
    assert "{{" not in batch_source


def test_compiler_renders_score_selected_sqrtsoftplus_router(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    circuit = {
        "parameters": {
            "refs": {
                "selection_bias": {"tensor": "router.selection_bias"},
            }
        }
    }
    node = {
        "id": "route",
        "op": "moe_route",
        "inputs": ["router_logits"],
        "outputs": ["routes"],
        "params": ["selection_bias"],
        "attrs": {
            "experts_per_token": 7,
            "routed_resource_count": 256,
            "routed_selection_count": 6,
            "always_selected_resources": [
                {"resource_index": 256, "weight": 1.0}
            ],
            "selection": "score_topk",
            "activation": "sqrtsoftplus",
            "normalize_selected": True,
            "routed_scaling_factor": 1.5,
            "selection_bias": "router.selection_bias",
            "selection_domain": {
                "id": "experts",
                "resource_count": 257,
                "selection_signal": "routes",
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": 7,
                    "index_shift": 0,
                    "index_mask": 511,
                },
            },
        },
    }
    tensor_index = {
        "tensors": {
            "router.selection_bias": {
                "dtype": "F32",
                "shape": [256],
                "layout": "row_major",
            }
        }
    }
    primary = (
        "moe_router_score_topk_sqrtsoftplus_bf16_r256_k6_a1w1_"
        "norm1_scale1.5_biasf32.comp"
    )
    batch = (
        "moe_router_batch1_score_topk_sqrtsoftplus_bf16_r256_k6_a1w1_"
        "norm1_scale1.5_biasf32.comp"
    )

    assert (
        shader_file_for_node(
            circuit,
            node,
            tensor_index,
            {"hidden_size": 4096, "intermediate_size": 2176},
        )
        == primary
    )
    assert frame_parallel_batch_shader_file(primary) == batch
    copy_shader_templates(shader_source_dir, tmp_path, {primary, batch})

    primary_source = (tmp_path / primary).read_text()
    batch_source = (tmp_path / batch).read_text()
    assert "sqrt(softplus(value))" in primary_source
    assert "return activated + uintBitsToFloat" in primary_source
    assert "float weight = router_values[expert];" in primary_source
    assert "weight = weight / denominator * ROUTED_SCALE;" in primary_source
    assert "const float ROUTED_SCALE = 1.5;" in primary_source
    assert "binding = 1) buffer ExpertRoutes" in primary_source
    assert "binding = 2) readonly buffer RouterSelectionBias" in primary_source
    assert "binding = 3) buffer SelectionTelemetry" in primary_source
    assert "const uint TOTAL_RESOURCE_COUNT = ROUTED_RESOURCE_COUNT +" in primary_source
    assert "const uint TOTAL_SELECTION_COUNT = ROUTED_SELECTION_COUNT +" in primary_source
    assert "top_indices[route] = ALWAYS_SELECTED_RESOURCE_START + offset;" in primary_source
    assert "top_weights[route] = ALWAYS_SELECTED_WEIGHT;" in primary_source
    assert (
        "selection_telemetry.counts[TOTAL_RESOURCE_COUNT + pair_index]"
        in primary_source
    )
    assert "gl_WorkGroupID.y" in batch_source
    assert "batch_index * ROUTER_WORDS" in batch_source
    assert "batch_index * TOTAL_SELECTION_COUNT" in batch_source
    assert "{{" not in primary_source
    assert "{{" not in batch_source
    compile_shader_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("always_selected", "message"),
    [
        ([], "invalid routing geometry"),
        ([{"resource_index": 258, "weight": 1.0}], "malformed"),
        ([{"resource_index": 256, "weight": 0.0}], "malformed"),
        (
            [
                {"resource_index": 256, "weight": 1.0},
                {"resource_index": 257, "weight": 0.5},
            ],
            "one common always-selected resource weight",
        ),
    ],
)
def test_independent_router_rejects_ambiguous_always_selected_contracts(
    always_selected: list[dict[str, object]],
    message: str,
) -> None:
    node = {
        "id": "route",
        "op": "moe_route",
        "inputs": ["router_logits"],
        "outputs": ["routes"],
        "params": ["selection_bias"],
        "attrs": {
            "experts_per_token": 7,
            "routed_resource_count": 256,
            "routed_selection_count": 6,
            "always_selected_resources": always_selected,
            "selection": "score_topk",
            "activation": "sqrtsoftplus",
            "normalize_selected": True,
            "routed_scaling_factor": 1.5,
            "selection_domain": {
                "id": "experts",
                "resource_count": 257,
                "selection_signal": "routes",
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": 7,
                    "index_shift": 0,
                    "index_mask": 511,
                },
            },
        },
    }
    circuit = {
        "parameters": {
            "refs": {"selection_bias": {"tensor": "router.selection_bias"}}
        }
    }
    tensor_index = {
        "tensors": {
            "router.selection_bias": {
                "dtype": "F32",
                "shape": [256],
                "layout": "row_major",
            }
        }
    }

    with pytest.raises(ModelCompileError, match=message):
        shader_file_for_node(
            circuit,
            node,
            tensor_index,
            {"hidden_size": 4096, "intermediate_size": 2176},
        )


def test_independent_router_rejects_stale_total_selection_domain() -> None:
    node = {
        "id": "route",
        "op": "moe_route",
        "inputs": ["router_logits"],
        "outputs": ["routes"],
        "params": ["selection_bias"],
        "attrs": {
            "experts_per_token": 7,
            "routed_resource_count": 256,
            "routed_selection_count": 6,
            "always_selected_resources": [
                {"resource_index": 256, "weight": 1.0}
            ],
            "selection": "score_topk",
            "activation": "sqrtsoftplus",
            "normalize_selected": True,
            "routed_scaling_factor": 1.5,
            "selection_domain": {
                "id": "experts",
                "resource_count": 256,
                "selection_signal": "routes",
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": 6,
                    "index_shift": 0,
                    "index_mask": 255,
                },
            },
        },
    }
    circuit = {
        "parameters": {
            "refs": {"selection_bias": {"tensor": "router.selection_bias"}}
        }
    }
    tensor_index = {
        "tensors": {
            "router.selection_bias": {
                "dtype": "F32",
                "shape": [256],
                "layout": "row_major",
            }
        }
    }

    with pytest.raises(ModelCompileError, match="total selection-domain contract"):
        shader_file_for_node(
            circuit,
            node,
            tensor_index,
            {"hidden_size": 4096, "intermediate_size": 2176},
        )


def test_compiler_renders_exact_table_preselection_and_router_weighting(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    circuit = {
        "parameters": {
            "refs": {
                "route_table": {"tensor": "router.token_routes"},
            }
        }
    }
    preselection = {
        "id": "preselect",
        "op": "parameter_table_resource_preselection",
        "inputs": ["token_id"],
        "outputs": ["resource_routes"],
        "params": ["route_table"],
        "attrs": {
            "experts_per_token": 7,
            "routed_resource_count": 256,
            "routed_selection_count": 6,
            "always_selected_resources": [
                {"resource_index": 256, "weight": 1.0}
            ],
            "predictable_dependency": {
                "schema": "nerve.predictable_resource_selection.v1",
                "kind": "parameter_table_lookup",
                "key_signal": "token_id",
                "table_parameter": "route_table",
                "selection_semantics": "exact",
            },
            "selection_domain": {
                "id": "experts",
                "resource_count": 257,
                "selection_signal": "resource_routes",
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": 7,
                    "index_shift": 0,
                    "index_mask": 511,
                },
            },
        },
    }
    weighting = {
        "id": "route",
        "op": "moe_route",
        "inputs": ["router_logits", "resource_routes"],
        "outputs": ["routes"],
        "params": [],
        "attrs": {
            "experts_per_token": 7,
            "routed_resource_count": 256,
            "routed_selection_count": 6,
            "always_selected_resources": [
                {"resource_index": 256, "weight": 1.0}
            ],
            "selection": "preselected_resource_indices",
            "activation": "sqrtsoftplus",
            "normalize_selected": True,
            "routed_scaling_factor": 1.5,
        },
    }
    tensor_index = {
        "tensors": {
            "router.token_routes": {
                "dtype": "I64",
                "shape": [129280, 6],
                "layout": "row_major",
            }
        }
    }
    preselection_primary = "resource_preselect_table_r256_k6_a1_v129280_tablei64.comp"
    preselection_batch = (
        "resource_preselect_batch1_table_r256_k6_a1_v129280_tablei64.comp"
    )
    weighting_primary = (
        "moe_router_preselected_sqrtsoftplus_bf16_r256_k6_a1w1_"
        "norm1_scale1.5.comp"
    )
    weighting_batch = (
        "moe_router_batch1_preselected_sqrtsoftplus_bf16_r256_k6_a1w1_"
        "norm1_scale1.5.comp"
    )

    assert (
        shader_file_for_node(
            circuit,
            preselection,
            tensor_index,
            {"hidden_size": 4096, "intermediate_size": 2176},
        )
        == preselection_primary
    )
    assert (
        shader_file_for_node(
            circuit,
            weighting,
            tensor_index,
            {"hidden_size": 4096, "intermediate_size": 2176},
        )
        == weighting_primary
    )
    assert (
        frame_parallel_batch_shader_file(preselection_primary) == preselection_batch
    )
    assert frame_parallel_batch_shader_file(weighting_primary) == weighting_batch
    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {
            preselection_primary,
            preselection_batch,
            weighting_primary,
            weighting_batch,
        },
    )

    preselection_source = (tmp_path / preselection_primary).read_text()
    preselection_batch_source = (tmp_path / preselection_batch).read_text()
    weighting_source = (tmp_path / weighting_primary).read_text()
    assert "const uint VOCAB_SIZE = 129280u;" in preselection_source
    assert "const uint TABLE_WORD_STRIDE = 2u;" in preselection_source
    assert "selection_table.words[table_element * TABLE_WORD_STRIDE]" in preselection_source
    assert "high_word != 0u" in preselection_source
    assert "for (uint previous = 0u; previous < lane; previous++)" in preselection_source
    assert "resource_selections.words[selection] = 0xffffffffu;" in preselection_source
    assert "binding = 0) readonly buffer TokenId" in preselection_source
    assert "binding = 1) buffer ResourceSelections" in preselection_source
    assert "binding = 2) readonly buffer SelectionTable" in preselection_source
    assert "binding = 3) buffer SelectionTelemetry" in preselection_source
    assert "high_word != 0u" in preselection_batch_source
    assert "sqrt(softplus(value))" in weighting_source
    assert "weight = weight / denominator * ROUTED_SCALE;" in weighting_source
    assert "binding = 0) readonly buffer RouterLogits" in weighting_source
    assert "binding = 1) readonly buffer ResourceSelections" in weighting_source
    assert "binding = 2) buffer ExpertRoutes" in weighting_source
    assert (
        "selection_telemetry.counts[TOTAL_RESOURCE_COUNT + pair_index]"
        in preselection_source
    )
    weighting_batch_source = (tmp_path / weighting_batch).read_text()
    assert "gl_WorkGroupID.y" in preselection_batch_source
    assert "token_ids.values[batch_index]" in preselection_batch_source
    assert "batch_index * ROUTER_WORDS" in weighting_batch_source
    assert "batch_index * TOTAL_SELECTION_COUNT" in weighting_batch_source
    assert "{{" not in preselection_source
    assert "{{" not in preselection_batch_source
    assert "{{" not in weighting_source
    assert "{{" not in weighting_batch_source
    compile_shader_artifacts(tmp_path)


def test_resource_preselection_rejects_advisory_or_implicit_dependencies() -> None:
    circuit = {
        "parameters": {"refs": {"route_table": {"tensor": "router.routes"}}}
    }
    tensor_index = {
        "tensors": {
            "router.routes": {
                "dtype": "I32",
                "shape": [1024, 2],
                "layout": "row_major",
            }
        }
    }
    base = {
        "id": "preselect",
        "op": "parameter_table_resource_preselection",
        "inputs": ["token_id"],
        "outputs": ["resource_routes"],
        "params": ["route_table"],
        "attrs": {
            "experts_per_token": 3,
            "routed_resource_count": 4,
            "routed_selection_count": 2,
            "always_selected_resources": [
                {"resource_index": 4, "weight": 1.0}
            ],
            "predictable_dependency": {
                "schema": "nerve.predictable_resource_selection.v1",
                "kind": "parameter_table_lookup",
                "key_signal": "token_id",
                "table_parameter": "route_table",
                "selection_semantics": "exact",
            },
            "selection_domain": {
                "id": "experts",
                "resource_count": 5,
                "selection_signal": "resource_routes",
                "encoding": {
                    "element_type": "u32",
                    "selection_count_per_activation": 3,
                    "index_shift": 0,
                    "index_mask": 7,
                },
            },
        },
    }

    for mutation in (
        lambda node: node["attrs"]["predictable_dependency"].update(
            {"selection_semantics": "advisory"}
        ),
        lambda node: node["attrs"].pop("predictable_dependency"),
        lambda node: node["attrs"]["predictable_dependency"].update(
            {"table_parameter": "different_table"}
        ),
    ):
        node = deepcopy(base)
        mutation(node)
        with pytest.raises(ModelCompileError, match="predictable dependency contract"):
            shader_file_for_node(
                circuit,
                node,
                tensor_index,
                {"hidden_size": 128, "intermediate_size": 128},
            )


def test_compiler_renders_native_compressed_tensors_int4_sparse_experts(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    circuit = {
        "parameters": {
            "refs": {
                "moe_input": {"tensor": "experts.gate_up"},
                "moe_input_scales": {"tensor": "experts.gate_up_scales"},
                "moe_output": {"tensor": "experts.down"},
                "moe_output_scales": {"tensor": "experts.down_scales"},
            }
        }
    }
    attrs = {
        "hidden_size": 3072,
        "intermediate_size": 1024,
        "num_experts": 256,
        "experts_per_token": 10,
    }
    gate_up = {
        "id": "sparse_moe_gate_up",
        "op": "sparse_moe_gate_up",
        "params": ["moe_input", "moe_input_scales"],
        "attrs": attrs,
    }
    down = {
        "id": "sparse_moe_down",
        "op": "sparse_moe_down",
        "params": ["moe_output", "moe_output_scales"],
        "attrs": attrs,
    }
    tensor_index = {
        "tensors": {
            "experts.gate_up": {
                "dtype": "I32",
                "shape": [256, 2048, 384],
                "logical_shape": [256, 2048, 3072],
                "layout": "row_major",
                "quantization": {
                    "format": "compressed_tensors_pack_quantized",
                    "bits": 4,
                    "group_size": 32,
                    "symmetric": True,
                    "signed_offset": 8,
                },
            },
            "experts.gate_up_scales": {
                "dtype": "BF16",
                "shape": [256, 2048, 96],
                "layout": "row_major",
            },
            "experts.down": {
                "dtype": "I32",
                "shape": [256, 3072, 128],
                "logical_shape": [256, 3072, 1024],
                "layout": "row_major",
                "quantization": {
                    "format": "compressed_tensors_pack_quantized",
                    "bits": 4,
                    "group_size": 32,
                    "symmetric": True,
                    "signed_offset": 8,
                },
            },
            "experts.down_scales": {
                "dtype": "BF16",
                "shape": [256, 3072, 32],
                "layout": "row_major",
            },
        }
    }
    dimensions = {"hidden_size": 3072, "intermediate_size": 1024}
    gate_file = "sparse_moe_gate_up_int4_ct_sbf16_g32_h3072_i1024_e256_k10.comp"
    down_file = "sparse_moe_down_int4_ct_sbf16_g32_h3072_i1024_e256_k10.comp"
    batch_gate = gate_file.replace("_int4_ct_", "_batch1_int4_ct_")
    batch_down = down_file.replace("_int4_ct_", "_batch1_int4_ct_")
    route_compaction = "moe_route_compact_batch1_i1024_k10_t64.comp"
    route_count = "moe_route_count_batch1_i1024_k10_t192.comp"

    assert shader_file_for_node(circuit, gate_up, tensor_index, dimensions) == gate_file
    assert shader_file_for_node(circuit, down, tensor_index, dimensions) == down_file
    assert frame_parallel_batch_shader_file(gate_file) == batch_gate
    assert frame_parallel_batch_shader_file(down_file) == batch_down
    assert workgroup_count_x_for_node(circuit, gate_up, tensor_index) == 640
    assert workgroup_count_x_for_node(circuit, down, tensor_index) == 1920
    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {
            gate_file,
            down_file,
            batch_gate,
            batch_down,
            route_compaction,
            route_count,
        },
    )

    gate_source = (tmp_path / gate_file).read_text()
    down_source = (tmp_path / down_file).read_text()
    batch_source = (tmp_path / batch_gate).read_text()
    assert "SPV_KHR_integer_dot_product" not in gate_source
    assert "int8_dot4" not in gate_source
    assert "quantized_hidden" not in gate_source
    assert "read_hiddenx4(batch_index, packed_column * 8u)" in gate_source
    assert "quantized_intermediate" not in down_source
    assert (
        "read_intermediatex4(batch_index, route, packed_column * 8u)"
        in down_source
    )
    assert "const uint GROUP_SIZE = 32u;" in gate_source
    assert "expert_scales.words[index >> 1u]" in gate_source
    assert "buffer DynamicResourceAddresses" in gate_source
    assert "buffer DynamicParameterSlots" in down_source
    assert "uint expert_count;" in gate_source
    assert "owned_expert_count != 0u" in gate_source
    assert "owned_expert_count != 0u" in down_source
    assert "route_weight" in down_source
    assert "gl_WorkGroupID.y" in batch_source
    assert "layout(push_constant) uniform BatchControl" in batch_source
    assert all(
        "{{" not in (tmp_path / shader_file).read_text()
        for shader_file in {
            gate_file,
            down_file,
            batch_gate,
            batch_down,
            route_compaction,
        }
    )
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / route_compaction.replace(".comp", ".spv")).is_file()


def test_compiler_rejects_mismatched_int4_sparse_expert_scales() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "moe_input": {"tensor": "experts.gate_up"},
                "moe_input_scales": {"tensor": "experts.gate_up_scales"},
            }
        }
    }
    node = {
        "id": "sparse_moe_gate_up",
        "op": "sparse_moe_gate_up",
        "params": ["moe_input", "moe_input_scales"],
        "attrs": {
            "hidden_size": 32,
            "intermediate_size": 16,
            "num_experts": 2,
            "experts_per_token": 1,
        },
    }
    tensor_index = {
        "tensors": {
            "experts.gate_up": {
                "dtype": "I32",
                "shape": [2, 32, 4],
                "logical_shape": [2, 32, 32],
                "layout": "row_major",
                "quantization": {
                    "format": "compressed_tensors_pack_quantized",
                    "bits": 4,
                    "group_size": 32,
                    "symmetric": True,
                    "signed_offset": 8,
                },
            },
            "experts.gate_up_scales": {
                "dtype": "BF16",
                "shape": [2, 31, 1],
                "layout": "row_major",
            },
        }
    }

    with pytest.raises(ModelCompileError, match="scale shape or dtype"):
        shader_file_for_node(
            circuit,
            node,
            tensor_index,
            {"hidden_size": 32, "intermediate_size": 16},
        )


def test_sparse_moe_workgroups_scale_with_selected_routes_not_total_experts() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "moe_input": {"tensor": "experts.gate_up"},
                "moe_output": {"tensor": "experts.down"},
            }
        }
    }
    tensor_index = {
        "tensors": {
            "experts.gate_up": {
                "dtype": "BF16",
                "shape": [256, 2048, 2048],
                "layout": "row_major",
            },
            "experts.down": {
                "dtype": "BF16",
                "shape": [256, 2048, 1024],
                "layout": "row_major",
            },
        }
    }
    attrs = {
        "hidden_size": 2048,
        "intermediate_size": 1024,
        "experts_per_token": 8,
    }
    small_expert_pool = {
        "id": "sparse_moe_gate_up",
        "op": "sparse_moe_gate_up",
        "params": ["moe_input"],
        "attrs": {**attrs, "num_experts": 32},
    }
    large_expert_pool = {
        **small_expert_pool,
        "attrs": {**attrs, "num_experts": 256},
    }
    down = {
        "id": "sparse_moe_down",
        "op": "sparse_moe_down",
        "params": ["moe_output"],
        "attrs": {**attrs, "num_experts": 256},
    }

    assert workgroup_count_x_for_node(circuit, small_expert_pool, tensor_index) == 4096
    assert workgroup_count_x_for_node(circuit, large_expert_pool, tensor_index) == 4096
    assert workgroup_count_x_for_node(circuit, down, tensor_index) == 8192
