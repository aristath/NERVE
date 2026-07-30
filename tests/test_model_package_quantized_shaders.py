from model_package_layout_common import *
from nerve.model_package_shader_compiler import compile_shader_artifacts
from nerve.model_package_tensors import (
    can_fuse_native_parallel_linears,
    physical_input_prequantization_spec,
)


def test_parallel_linear_shader_selector_rejects_invalid_metadata_and_layout() -> None:
    node = {
        "id": "qkv",
        "op": "parallel_linear_3way",
        "inputs": ["hidden"],
        "outputs": ["q", "k", "v"],
        "params": ["q_weight", "k_weight", "v_weight"],
        "attrs": {"branch_count": 2},
    }
    circuit = {
        "parameters": {
            "refs": {
                parameter_id: {"tensor": parameter_id}
                for parameter_id in node["params"]
            }
        }
    }
    tensor_index = {
        "tensors": {
            parameter_id: {
                "dtype": "BF16",
                "shape": [512, 1024],
                "layout": ROW_MAJOR_LAYOUT,
            }
            for parameter_id in node["params"]
        }
    }
    dimensions = {"hidden_size": 1024, "intermediate_size": 2560}

    with pytest.raises(ModelCompileError, match="inconsistent branch metadata"):
        shader_file_for_node(circuit, node, tensor_index, dimensions)

    node["attrs"]["branch_count"] = 3
    tensor_index["tensors"]["v_weight"]["layout"] = "unknown_layout"
    with pytest.raises(ModelCompileError, match="unsupported layouts"):
        shader_file_for_node(circuit, node, tensor_index, dimensions)


def test_parallel_linear_shader_selector_supports_fp8_weight_scale_pairs() -> None:
    node = {
        "id": "qk",
        "op": "parallel_linear_2way",
        "inputs": ["hidden"],
        "outputs": ["q", "k"],
        "params": [
            "q_weight",
            "q_weight_scale_inv",
            "k_weight",
            "k_weight_scale_inv",
        ],
        "attrs": {"branch_count": 2, "branch_parameter_counts": [2, 2]},
    }
    circuit = {
        "parameters": {
            "refs": {
                parameter_id: {"tensor": parameter_id}
                for parameter_id in node["params"]
            }
        }
    }
    tensor_index = {
        "tensors": {
            "q_weight": {
                "dtype": "F8_E4M3",
                "shape": [5120, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "k_weight": {
                "dtype": "F8_E4M3",
                "shape": [1024, 5120],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "q_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [40, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
            "k_weight_scale_inv": {
                "dtype": "BF16",
                "shape": [8, 40],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }

    dimensions = {"hidden_size": 5120, "intermediate_size": 5120}

    source_nodes = [
        {
            "id": "q",
            "op": "linear",
            "inputs": ["hidden"],
            "outputs": ["q"],
            "params": ["q_weight", "q_weight_scale_inv"],
        },
        {
            "id": "k",
            "op": "linear",
            "inputs": ["hidden"],
            "outputs": ["k"],
            "params": ["k_weight", "k_weight_scale_inv"],
        },
    ]
    assert can_fuse_native_parallel_linears(circuit, source_nodes, tensor_index)
    assert shader_file_for_node(circuit, node, tensor_index, dimensions) == (
        "parallel_linear_2way_fp8_e4m3_b128x128_5120x5120_1024.comp"
    )
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 320


def test_linear_shader_selector_supports_internal_q8_0_weights() -> None:
    node = {
        "id": "project",
        "op": "linear",
        "inputs": ["hidden"],
        "outputs": ["projected"],
        "params": ["weight"],
    }
    circuit = {"parameters": {"refs": {"weight": {"tensor": "weight"}}}}
    tensor_index = {
        "tensors": {
            "weight": {
                "dtype": "Q8_0",
                "shape": [768, 16, 9],
                "logical_shape": [768, 512],
                "byte_count": 768 * 16 * 36,
                "layout": ROW_MAJOR_LAYOUT,
            }
        }
    }
    dimensions = {"hidden_size": 512, "intermediate_size": 2048}

    assert shader_file_for_node(circuit, node, tensor_index, dimensions) == (
        "linear_q8_0_512x768.comp"
    )
    assert workgroup_count_x_for_node(circuit, node, tensor_index) == 24

    node["op"] = "linear_residual"
    assert shader_file_for_node(circuit, node, tensor_index, dimensions) == (
        "linear_residual_q8_0_512x768.comp"
    )


def test_parallel_and_fused_shader_selectors_support_internal_q8_0_weights() -> None:
    circuit = {
        "parameters": {
            "refs": {
                parameter_id: {"tensor": parameter_id}
                for parameter_id in ("gate", "up", "q", "k")
            }
        }
    }
    tensor_index = {
        "tensors": {
            parameter_id: {
                "dtype": "Q8_0",
                "shape": [768, 16, 9],
                "logical_shape": [768, 512],
                "byte_count": 768 * 16 * 36,
                "layout": ROW_MAJOR_LAYOUT,
            }
            for parameter_id in ("gate", "up", "q", "k")
        }
    }
    dimensions = {"hidden_size": 512, "intermediate_size": 768}
    parallel = {
        "id": "qk",
        "op": "parallel_linear_2way",
        "inputs": ["hidden"],
        "outputs": ["q", "k"],
        "params": ["q", "k"],
        "attrs": {"branch_count": 2, "branch_parameter_counts": [1, 1]},
    }
    fused = {
        "id": "ffn_gate_up",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["hidden"],
        "outputs": ["ffn"],
        "params": ["gate", "up"],
        "attrs": {
            "branch_count": 2,
            "element_count": 768,
            "intermediate_rounding": "BF16",
        },
    }

    assert shader_file_for_node(circuit, parallel, tensor_index, dimensions) == (
        "parallel_linear_2way_q8_0_512x768.comp"
    )
    assert workgroup_count_x_for_node(circuit, parallel, tensor_index) == 24
    assert shader_file_for_node(circuit, fused, tensor_index, dimensions) == (
        "parallel_linear_silu_multiply_q8_0_512x768.comp"
    )
    assert workgroup_count_x_for_node(circuit, fused, tensor_index) == 24


def test_compiler_batches_internal_q8_0_dense_kernels() -> None:
    for shader_file in (
        "linear_q8_0_512x768.comp",
        "parallel_linear_2way_q8_0_512x768.comp",
        "parallel_linear_silu_multiply_q8_0_512x768.comp",
    ):
        spec = component_kernel_spec(
            execution_index=0,
            node={"id": "project", "op": "linear"},
            circuit={},
            shader_file=shader_file,
            local_size_x=64,
            workgroup_count_x=24,
        )
        assert spec["batch_mode"] == "weight_shared"
        assert [
            implementation["lane_tile_width"]
            for implementation in spec["batch_implementations"]
        ] == [2, 4, 8, 16]


def test_compiler_renders_internal_q8_0_linear_shaders(tmp_path: Path) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_q8_0_512x768.comp",
        "linear_batch16_q8_0_512x768.comp",
        "linear_bias_q8_0_512x768.comp",
        "linear_bias_batch16_q8_0_512x768.comp",
        "linear_residual_q8_0_512x768.comp",
        "linear_residual_batch16_q8_0_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_q8_0_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_q8_0_512x768.comp").read_text()
    residual = (tmp_path / "linear_residual_q8_0_512x768.comp").read_text()
    batch = (tmp_path / "linear_batch16_q8_0_512x768.comp").read_text()
    assert "const uint INPUT_SIZE = 512u;" in linear
    assert "const uint OUTPUT_SIZE = 768u;" in linear
    assert "const uint OUTPUT_TILE_ROWS = 32u;" in linear
    assert "const uint Q8_BLOCK_WORDS = 9u;" in linear
    assert "#extension GL_EXT_integer_dot_product : require" in linear
    assert "dotPacked4x8EXT" in linear
    assert "subgroupClusteredMax" in linear
    assert "shared uint quantized_input" in linear
    assert "subgroupShuffle(quantized" in linear
    assert "binding = 3) readonly buffer Bias" in bias
    assert "binding = 1) readonly buffer ResidualFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert "layout(push_constant) uniform BatchControl" in batch
    assert "batch_control.batch_width" in batch
    assert all("{{" not in source for source in (linear, bias, residual, batch))


def test_compiler_renders_internal_q8_0_parallel_and_fused_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "parallel_linear_2way_q8_0_512x768.comp",
        "parallel_linear_batch16_2way_q8_0_512x768.comp",
        "parallel_linear_silu_multiply_q8_0_512x768.comp",
        "parallel_linear_silu_multiply_batch16_q8_0_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    parallel = (tmp_path / "parallel_linear_2way_q8_0_512x768.comp").read_text()
    fused = (tmp_path / "parallel_linear_silu_multiply_q8_0_512x768.comp").read_text()
    parallel_batch = (
        tmp_path / "parallel_linear_batch16_2way_q8_0_512x768.comp"
    ).read_text()
    fused_batch = (
        tmp_path / "parallel_linear_silu_multiply_batch16_q8_0_512x768.comp"
    ).read_text()
    for source in (parallel, fused, parallel_batch, fused_batch):
        assert "const uint INPUT_SIZE = 512u;" in source
        assert "const uint OUTPUT_SIZE = 768u;" in source
        assert "const uint OUTPUT_TILE_ROWS = 32u;" in source
        assert "#extension GL_EXT_integer_dot_product : require" in source
        assert "dotPacked4x8EXT" in source
        assert "subgroupClusteredMax" in source
        assert "shared uint quantized_input" in source
        assert "{{" not in source
    assert "const uint BRANCH_COUNT = 2u;" in parallel
    assert "buffer OutputA" in parallel
    assert "buffer OutputB" in parallel
    assert "readonly buffer WeightA" in parallel
    assert "readonly buffer WeightB" in parallel
    assert "write_branch_output" in parallel
    assert "rounded_silu(round_bf16(gate)) * round_bf16(up)" in fused
    assert "readonly buffer GateWeight" in fused
    assert "readonly buffer UpWeight" in fused
    assert "const uint BATCH_TILE_WIDTH = 16u;" in parallel_batch
    assert "const uint BATCH_TILE_WIDTH = 16u;" in fused_batch
    assert "batch_control.batch_width" in parallel_batch
    assert "batch_control.batch_width" in fused_batch


def test_compiler_renders_native_block_scaled_fp8_linear_shaders(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_bias_fp8_e4m3_b128x128_5120x17408.comp",
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    expected_tile_rows = {
        "linear_fp8_e4m3_b128x128_5120x17408.comp": 16,
        "linear_bias_fp8_e4m3_b128x128_5120x17408.comp": 16,
        "linear_residual_fp8_e4m3_b128x128_17408x5120.comp": 16,
    }
    for shader_file in shader_files:
        shader = (tmp_path / shader_file).read_text()
        assert "const uint BLOCK_ROWS = 128u;" in shader
        assert "const uint BLOCK_COLUMNS = 128u;" in shader
        assert (
            f"const uint OUTPUT_TILE_ROWS = {expected_tile_rows[shader_file]}u;"
            in shader
        )
        assert "#extension GL_EXT_spirv_intrinsics : require" in shader
        assert "SPV_VALVE_mixed_float_dot_product" in shader
        assert "fp8_dot4_acc32" in shader
        assert "shared fe4m3vec4 quantized_input" in shader
        assert "subgroupClusteredMax" in shader
        assert "word < INPUT_FP8_WORDS" in shader
        assert "WeightScaleInv" in shader
        assert "{{" not in shader
    assert (
        "binding = 4) readonly buffer Bias"
        in (tmp_path / "linear_bias_fp8_e4m3_b128x128_5120x17408.comp").read_text()
    )


def test_compiler_renders_native_auto_gptq_int4_linear_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_int4_gptq_sf16_g128_512x768.comp",
        "linear_bias_int4_gptq_sf16_g128_512x768.comp",
        "linear_residual_int4_gptq_sf16_g128_512x768.comp",
        "linear_batch16_int4_gptq_sf16_g128_512x768.comp",
        "linear_bias_batch16_int4_gptq_sf16_g128_512x768.comp",
        "linear_residual_batch16_int4_gptq_sf16_g128_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_gptq_sf16_g128_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_gptq_sf16_g128_512x768.comp").read_text()
    residual = (
        tmp_path / "linear_residual_int4_gptq_sf16_g128_512x768.comp"
    ).read_text()
    batch = (tmp_path / "linear_batch16_int4_gptq_sf16_g128_512x768.comp").read_text()
    assert "const uint GROUP_SIZE = 128u;" in linear
    assert "const uint INPUT_SIZE = 512u;" in linear
    assert "const uint OUTPUT_SIZE = 768u;" in linear
    assert "const uint OUTPUT_TILE_ROWS = 64u;" in linear
    assert "subgroupAdd(sum)" not in linear
    assert "SPV_KHR_integer_dot_product" not in linear
    assert "int8_dot4" not in linear
    assert "quantized_input" not in linear
    assert "read_inputx4(batch_index, packed_column * 8u)" in linear
    assert "packed_column * OUTPUT_SIZE + row" in linear
    assert "unpackHalf2x16" in linear
    assert "readonly buffer Bias" in bias
    assert "readonly buffer ResidualFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert "batch_control.batch_width" in batch
    assert all("{{" not in source for source in (linear, bias, residual, batch))


def test_compiler_renders_shared_int8_activation_int4_kernel_family(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "quantize_int8_symmetric_pairpacked_b32_h512.comp",
        "quantize_batch16_int8_symmetric_pairpacked_b32_h512.comp",
        "linear_prequant_pairpacked_int4_gptq_sf16_g128_512x768.comp",
        "linear_prequant_pairpacked_batch16_int4_gptq_sf16_g128_512x768.comp",
        "linear_residual_prequant_int4_ct_sbf16_g32_512x768.comp",
        "linear_residual_prequant_batch16_int4_ct_sbf16_g32_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    quantize = (
        tmp_path / "quantize_int8_symmetric_pairpacked_b32_h512.comp"
    ).read_text()
    gptq = (
        tmp_path
        / "linear_prequant_pairpacked_int4_gptq_sf16_g128_512x768.comp"
    ).read_text()
    residual = (
        tmp_path / "linear_residual_prequant_int4_ct_sbf16_g32_512x768.comp"
    ).read_text()
    batch = (
        tmp_path
        / "linear_prequant_pairpacked_batch16_int4_gptq_sf16_g128_512x768.comp"
    ).read_text()

    assert "const uint BLOCK_COLUMNS = 32u;" in quantize
    assert "block_max / 127.0" in quantize
    assert "buffer BlockSum" in quantize
    assert "subgroupAdd(lane_sum)" in quantize
    assert "#extension GL_EXT_integer_dot_product : require" in gptq
    assert "dotPacked4x8EXT" in gptq
    assert "pack_weight_i8x4" not in gptq
    assert "packed & 0x0f0f0f0fu" in gptq
    assert "-8 * input_sums.values" in gptq
    assert "binding = 0) readonly buffer QuantizedInputs" in gptq
    assert "binding = 1) readonly buffer InputScales" in gptq
    assert "binding = 2) readonly buffer InputSums" in gptq
    assert "binding = 3) buffer OutputFrames" in gptq
    assert "binding = 2) readonly buffer ResidualFrames" in residual
    assert "binding = 3) buffer OutputFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert all(
        "{{" not in (tmp_path / shader_file).read_text()
        for shader_file in shader_files
    )
    compile_shader_artifacts(tmp_path)


def test_pairpacked_int8_dot_is_exactly_equivalent_to_signed_int4_dot() -> None:
    activations = [
        -127,
        126,
        -103,
        87,
        -64,
        63,
        -31,
        30,
        -17,
        16,
        -9,
        8,
        -3,
        2,
        -1,
        0,
        1,
        -2,
        3,
        -8,
        9,
        -16,
        17,
        -30,
        31,
        -63,
        64,
        -87,
        103,
        -126,
        127,
        11,
    ]
    raw_nibbles = [
        0,
        15,
        1,
        14,
        2,
        13,
        3,
        12,
        4,
        11,
        5,
        10,
        6,
        9,
        7,
        8,
        8,
        7,
        9,
        6,
        10,
        5,
        11,
        4,
        12,
        3,
        13,
        2,
        14,
        1,
        15,
        0,
    ]

    signed_dot = sum(
        activation * (nibble - 8)
        for activation, nibble in zip(activations, raw_nibbles, strict=True)
    )
    pairpacked_raw_dot = 0
    for offset in range(0, len(activations), 8):
        values = activations[offset : offset + 8]
        nibbles = raw_nibbles[offset : offset + 8]
        pairpacked_raw_dot += sum(
            values[index] * nibbles[index] for index in (0, 2, 4, 6)
        )
        pairpacked_raw_dot += sum(
            values[index] * nibbles[index] for index in (1, 3, 5, 7)
        )
    corrected_dot = pairpacked_raw_dot - 8 * sum(activations)

    assert corrected_dot == signed_dot


def test_compiler_renders_fused_pairpacked_int8_representation_producers(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "rms_norm_quantize_int8_pairpacked_b32_h5120_eps1e-06_offset1.comp",
        (
            "rms_norm_quantize_batch4_int8_pairpacked_b32_"
            "h5120_eps1e-06_offset1.comp"
        ),
        "silu_multiply_quantize_int8_pairpacked_b32_h17408.comp",
        (
            "silu_multiply_quantize_batch4_int8_pairpacked_"
            "b32_h17408.comp"
        ),
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    norm = (
        tmp_path
        / "rms_norm_quantize_int8_pairpacked_b32_h5120_eps1e-06_offset1.comp"
    ).read_text()
    silu = (
        tmp_path / "silu_multiply_quantize_int8_pairpacked_b32_h17408.comp"
    ).read_text()
    for source in (norm, silu):
        assert "layout(local_size_x = 1024" in source
        assert "const uint BLOCK_COLUMNS = 32u;" in source
        assert "buffer QuantizedFrame" in source
        assert "buffer BlockSum" in source
        assert "subgroupMax" in source
        assert "subgroupAdd(lane_sum)" in source
        assert "{{" not in source
    assert "const uint HIDDEN_SIZE = 5120u;" in norm
    assert "readonly buffer Weight" in norm
    assert "shared float reduction[64];" in norm
    assert "if (lane < 64u)" in norm
    assert "uint index = lane;" in norm
    assert "gl_SubgroupSize" in norm
    assert "const uint ELEMENT_COUNT = 17408u;" in silu
    assert "rounded_silu" in silu
    assert all(
        "layout(push_constant) uniform BatchControl"
        in (tmp_path / shader_file).read_text()
        for shader_file in shader_files
        if "batch4" in shader_file
    )
    compile_shader_artifacts(tmp_path)


def test_packed_int4_projection_requests_reusable_int8_input_representation() -> None:
    node = {
        "id": "projection",
        "op": "linear",
        "inputs": ["normalized"],
        "outputs": ["projected"],
        "params": ["weight", "weight_scales"],
    }
    circuit = {
        "parameters": {
            "refs": {
                "weight": {"tensor": "weight"},
                "weight_scales": {"tensor": "scales"},
            }
        }
    }
    tensor_index = {
        "tensors": {
            "weight": {
                "dtype": "I32",
                "shape": [64, 768],
                "logical_shape": [768, 512],
                "layout": ROW_MAJOR_LAYOUT,
                "quantization": {
                    "format": "auto_gptq",
                    "bits": 4,
                    "group_size": 128,
                    "packing_layout": "input_major_packed_columns",
                    "zero_point_encoding": "fixed_8",
                },
            },
            "scales": {
                "dtype": "F16",
                "shape": [4, 768],
                "layout": ROW_MAJOR_LAYOUT,
            },
        }
    }

    assert physical_input_prequantization_spec(circuit, node, tensor_index) == {
        "contract": (
            "bf16_blockwise_symmetric_int8_pairpacked_f32_scale_i32_sum.v1"
        ),
        "input_size": 512,
        "block_columns": 32,
    }
    assert (
        physical_input_prequantization_spec(
            circuit,
            node,
            tensor_index,
            compiler_target={
                "devices": [
                    {
                        "shader_features": [],
                        "subgroup_operations": ["arithmetic"],
                        "subgroup_compute_supported": True,
                    }
                ]
            },
        )
        is None
    )
    lowered = {
        **node,
        "inputs": [
            "normalized_int8_pairpacked",
            "normalized_scale",
            "normalized_sum",
        ],
        "attrs": {
            "physical_input_contract": (
                "bf16_blockwise_symmetric_int8_pairpacked_f32_scale_i32_sum.v1"
            )
        },
    }
    assert shader_file_for_node(
        circuit,
        lowered,
        tensor_index,
        {"hidden_size": 512, "intermediate_size": 2048},
    ) == "linear_prequant_pairpacked_int4_gptq_sf16_g128_512x768.comp"


def test_compiler_renders_native_compressed_tensors_int4_linear_variants(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_int4_ct_sbf16_g32_512x768.comp",
        "linear_bias_int4_ct_sbf16_g32_512x768.comp",
        "linear_residual_int4_ct_sbf16_g32_512x768.comp",
        "linear_batch16_int4_ct_sbf16_g32_512x768.comp",
        "linear_bias_batch16_int4_ct_sbf16_g32_512x768.comp",
        "linear_residual_batch16_int4_ct_sbf16_g32_512x768.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    linear = (tmp_path / "linear_int4_ct_sbf16_g32_512x768.comp").read_text()
    bias = (tmp_path / "linear_bias_int4_ct_sbf16_g32_512x768.comp").read_text()
    residual = (tmp_path / "linear_residual_int4_ct_sbf16_g32_512x768.comp").read_text()
    batch = (tmp_path / "linear_batch16_int4_ct_sbf16_g32_512x768.comp").read_text()
    assert "const uint GROUP_SIZE = 32u;" in linear
    assert "const uint OUTPUT_TILE_ROWS = 16u;" in linear
    assert "row * PACKED_COLUMNS" in linear
    assert "int(packed & 15u) - 8" in linear
    assert "SPV_KHR_integer_dot_product" not in linear
    assert "int8_dot4" not in linear
    assert "quantized_input" not in linear
    assert "read_inputx4(batch_index, packed_column * 8u)" in linear
    assert "subgroupAdd" in linear
    assert "read_bf16_word(scales.words[index >> 1u], index)" in bias
    assert "readonly buffer ResidualFrames" in residual
    assert "const uint BATCH_TILE_WIDTH = 16u;" in batch
    assert "batch_control.batch_width" in batch
    assert all("{{" not in source for source in (linear, bias, residual, batch))


def test_compiler_renders_native_block_scaled_fp8_sparse_experts(
    tmp_path: Path,
) -> None:
    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "moe_route_compact_batch1_i512_k8_t16.comp",
        "moe_route_count_batch1_i512_k8_t32.comp",
        "moe_topk_bf16_e256_k8.comp",
        "moe_topk_batch1_bf16_e256_k8.comp",
        "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_gate_up_prequant_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_gate_up_batch1_prequant_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_down_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "sparse_moe_down_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp",
        "moe_reduce_bf16_h2048_k8_scale1.comp",
        "moe_reduce_batch1_bf16_h2048_k8_scale1.comp",
        "sigmoid_scalar_multiply_bf16_2048.comp",
    }

    copy_shader_templates(shader_source_dir, tmp_path, shader_files)

    gate_up_shader = (
        tmp_path / "sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    down_shader = (
        tmp_path / "sparse_moe_down_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    prequant_gate_up_shader = (
        tmp_path
        / "sparse_moe_gate_up_prequant_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
    ).read_text()
    router_shader = (tmp_path / "moe_topk_bf16_e256_k8.comp").read_text()
    reduce_shader = (tmp_path / "moe_reduce_bf16_h2048_k8_scale1.comp").read_text()
    assert "const uint NUM_EXPERTS = 256u;" in gate_up_shader
    assert "buffer SelectionTelemetry" in router_shader
    assert "atomicAdd(selection_telemetry.counts[expert], 1u);" in router_shader
    assert "const uint EXPERTS_PER_TOKEN = 8u;" in gate_up_shader
    assert "#extension GL_EXT_float_e4m3 : require" in gate_up_shader
    assert "uintBitsToFloate4m3EXT" in gate_up_shader
    assert "buffer DynamicResourceAddresses" in gate_up_shader
    assert "buffer DynamicParameterSlots" in gate_up_shader
    assert "GL_EXT_buffer_reference2" in gate_up_shader
    assert "DynamicU32Buffer expert_input_scale_inv" in gate_up_shader
    assert "DynamicU32Buffer expert_output_scale_inv" in down_shader
    assert "const uint TILE_ROWS = 32u;" in gate_up_shader
    assert "const uint TILE_ROWS = 64u;" in down_shader
    assert "layout(local_size_x = 512" in down_shader
    assert "shared fe4m3vec4 quantized_hidden" in gate_up_shader
    assert "shared fe4m3vec4 quantized_hidden" not in prequant_gate_up_shader
    assert "readonly buffer QuantizedHidden" in prequant_gate_up_shader
    assert "readonly buffer HiddenScales" in prequant_gate_up_shader
    assert "const uint TILE_ROWS = 32u;" in prequant_gate_up_shader
    assert "layout(local_size_x = 512" in prequant_gate_up_shader
    assert "uint local_row = gl_SubgroupID;" in prequant_gate_up_shader
    assert "barrier();" in prequant_gate_up_shader
    assert "shared fe4m3vec4 quantized_intermediate" in down_shader
    assert "SPV_VALVE_mixed_float_dot_product" in gate_up_shader
    assert "fp8_dot4_acc32" in gate_up_shader
    assert "subgroupClusteredMax" in gate_up_shader
    assert "layout(local_size_x = 512" in gate_up_shader
    assert "local_row = gl_SubgroupID" in gate_up_shader
    assert "expert_routes.words[route] = (weight << 16u) | expert;" in router_shader
    assert "route < EXPERTS_PER_TOKEN" in reduce_shader
    route_compaction = (
        tmp_path / "moe_route_compact_batch1_i512_k8_t16.comp"
    ).read_text()
    assert "candidate_expert < source_expert" in route_compaction
    assert "EXPERT_FRAME_WORDS" in route_compaction
    assert all("{{" not in source for source in (gate_up_shader, down_shader))
    assert (
        "gl_WorkGroupID.y"
        in (
            tmp_path
            / "sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"
        ).read_text()
    )
    assert (
        "const uint HIDDEN_SIZE = 2048u;"
        in (tmp_path / "sigmoid_scalar_multiply_bf16_2048.comp").read_text()
    )
    compile_shader_artifacts(tmp_path)
    assert (tmp_path / "moe_route_compact_batch1_i512_k8_t16.spv").is_file()
    assert (tmp_path / "moe_route_count_batch1_i512_k8_t32.spv").is_file()


def test_compiler_parallelizes_only_selected_sparse_expert_routes() -> None:
    attrs = {
        "hidden_size": 2048,
        "intermediate_size": 512,
        "num_experts": 256,
        "experts_per_token": 8,
    }
    circuit = {"parameters": {"refs": {"expert_weight": {"tensor": "expert_weight"}}}}
    fp8_tensor_index = {"tensors": {"expert_weight": {"dtype": "F8_E4M3"}}}
    bf16_tensor_index = {"tensors": {"expert_weight": {"dtype": "BF16"}}}
    gate_up = {
        "op": "sparse_moe_gate_up",
        "attrs": attrs,
        "params": ["expert_weight"],
    }
    down = {
        "op": "sparse_moe_down",
        "attrs": attrs,
        "params": ["expert_weight"],
    }

    assert workgroup_count_x_for_node(circuit, gate_up, fp8_tensor_index) == 128
    assert workgroup_count_x_for_node(circuit, down, fp8_tensor_index) == 256
    assert workgroup_count_x_for_node(circuit, gate_up, bf16_tensor_index) == 2048
    assert workgroup_count_x_for_node(circuit, down, bf16_tensor_index) == 8192

    spec = component_kernel_spec(
        execution_index=0,
        node={
            "id": "sparse_moe_gate_up",
            "op": "sparse_moe_gate_up",
            "inputs": ["hidden", "routes"],
            "outputs": ["intermediates"],
        },
        circuit={},
        shader_file=("sparse_moe_gate_up_fp8_e4m3_b128x128_h2048_i512_e256_k8.comp"),
        local_size_x=512,
        workgroup_count_x=128,
    )
    assert spec["batch_mode"] == "weight_shared"
    assert [
        implementation["lane_tile_width"]
        for implementation in spec["batch_implementations"]
    ] == [1]
    assert spec["execution_domain"] == "decode"
    assert (
        spec["batch_implementations"][0]["execution_domain"]
        == "decode_and_prefill"
    )
    assert spec["batch_implementations"][0]["stages"] == [
        {
            "shader_path": (
                "shaders/moe_route_compact_batch1_i512_k8_t16__pbc31.comp"
            ),
            "local_size_x": 64,
            "workgroup_count_x": 8,
            "descriptor_bindings": [
                {"binding": 1, "source_binding": 1},
                {"binding": 2, "source_binding": 2},
            ],
            "control": {
                "kind": "storage_buffer",
                "byte_count": 28,
                "binding": 31,
                "payload": "width_expert_range_indirect",
                "access": "read_write",
            },
        },
        {
            "shader_path": (
                "shaders/sparse_moe_gate_up_batch1_fp8_e4m3_b128x128_"
                "h2048_i512_e256_k8__pbc31.comp"
            ),
            "local_size_x": 512,
            "workgroup_count_x": 128,
            "control": {
                "kind": "storage_buffer",
                "byte_count": 28,
                "binding": 31,
                "payload": "width_expert_range_indirect",
            },
            "indirect_dispatch_byte_offset": 16,
        }
    ]


def test_compiler_tiles_dense_fp8_dispatch_without_changing_bf16_dispatch() -> None:
    circuit = {
        "parameters": {
            "refs": {
                "weight": {"tensor": "weight"},
                "gate_weight": {"tensor": "gate_weight"},
                "up_weight": {"tensor": "up_weight"},
            }
        }
    }
    fp8_tensor_index = {
        "tensors": {
            "weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
            "gate_weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
            "up_weight": {"dtype": "F8_E4M3", "shape": [17408, 5120]},
        }
    }
    bf16_tensor_index = {
        "tensors": {
            tensor_name: {"dtype": "BF16", "shape": [17408, 5120]}
            for tensor_name in ("weight", "gate_weight", "up_weight")
        }
    }
    linear = {"op": "linear", "params": ["weight"]}
    fused_ffn = {
        "op": "parallel_linear_silu_multiply",
        "params": ["gate_weight", "up_weight"],
    }

    assert workgroup_count_x_for_node(circuit, linear, fp8_tensor_index) == 1088
    assert workgroup_count_x_for_node(circuit, fused_ffn, fp8_tensor_index) == 1088
    assert workgroup_count_x_for_node(circuit, linear, bf16_tensor_index) == 8704
    assert workgroup_count_x_for_node(circuit, fused_ffn, bf16_tensor_index) == 8704


def test_compiler_tiles_int4_dispatch_by_physical_packing_format() -> None:
    circuit = {"parameters": {"refs": {"weight": {"tensor": "weight"}}}}
    node = {"id": "project", "op": "linear", "params": ["weight"]}
    auto_gptq = {
        "tensors": {
            "weight": {
                "dtype": "I32",
                "shape": [640, 17408],
                "logical_shape": [17408, 5120],
                "quantization": {
                    "format": "auto_gptq",
                    "group_size": 128,
                    "packing_layout": "input_major_packed_columns",
                    "zero_point_encoding": "fixed_8",
                },
            }
        }
    }
    compressed_tensors = {
        "tensors": {
            "weight": {
                "dtype": "I32",
                "shape": [16384, 672],
                "logical_shape": [16384, 5376],
                "quantization": {"format": "compressed_tensors_pack_quantized"},
            }
        }
    }
    bf16 = {"tensors": {"weight": {"dtype": "BF16", "shape": [17408, 5120]}}}

    assert workgroup_count_x_for_node(circuit, node, auto_gptq) == 272
    assert workgroup_count_x_for_node(circuit, node, compressed_tensors) == 1024
    assert workgroup_count_x_for_node(circuit, node, bf16) == 8704


def test_compiler_rejects_fp8_sparse_expert_geometry_unsafe_for_native_dot() -> None:
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
        "params": ["moe_input", "moe_input_scale_inv"],
        "attrs": {
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "experts_per_token": 8,
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
                "shape": [256, 8, 32],
                "layout": "row_major",
            },
        }
    }

    with pytest.raises(ModelCompileError, match="requires 128-column blocks"):
        fp8_moe_block_shape_for_stage(
            circuit,
            node,
            tensor_index,
            stage="gate_up",
        )
