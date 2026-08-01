from model_structure_common import *
from model_structure_common import _tensor
from nerve.model_transpiler_types import ModelTranspileError

import pytest


def test_compiles_exact_yarn_frequency_and_attention_scaling() -> None:
    scaling = compile_rope_scaling(
        {
            "rope_type": "yarn",
            "rope_theta": 500_000.0,
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.3465735902799727,
        },
        64,
    )

    assert scaling == {
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 8192,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": True,
        "attention_factor": 1.3465735902799727,
        "correction_low": 9.0,
        "correction_high": 18.0,
    }


def test_discovers_model_owned_greedy_sampling_policy() -> None:
    assert discover_sampling_policy({}) == {
        "method": "greedy",
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }


def test_preserves_model_owned_sampled_generation_filters() -> None:
    assert discover_sampling_policy(
        {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
        }
    ) == {
        "method": "temperature_top_p",
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert discover_sampling_policy(
        {
            "do_sample": True,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
        }
    ) == {
        "method": "temperature_top_k_top_p",
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert discover_sampling_policy(
        {
            "do_sample": True,
            "temperature": 0.1,
            "top_k": 50,
            "min_p": 0.04,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.05,
        }
    ) == {
        "method": "temperature_top_k_top_p",
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": 0.04,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.05,
    }


def test_discovers_dynamic_block_fp8_by_numerical_structure() -> None:
    config = {
        "model_type": "outer_container",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "weight_block_size": [128, 128],
        },
        "text_config": {"model_type": "unrelated_family_name"},
    }

    assert discover_quantization_policy(config) == {
        "weight": {
            "format": "block_scaled_fp8_e4m3",
            "block_shape": [128, 128],
            "per_tensor": False,
        },
        "activation": {
            "format": "dynamic_block_fp8_e4m3",
            "group_size": 128,
            "per_tensor": False,
        },
    }


def test_does_not_invent_dynamic_activation_quantization() -> None:
    assert (
        discover_quantization_policy(
            {
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "static",
                    "weight_block_size": [128, 128],
                }
            }
        )
        is None
    )


def test_discovers_compressed_tensors_channel_fp8_by_numerical_structure() -> None:
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "linear": {
                    "format": "float-quantized",
                    "weights": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "channel",
                        "dynamic": False,
                        "symmetric": True,
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                        "strategy": "token",
                        "dynamic": True,
                        "symmetric": True,
                    },
                }
            },
        }
    }

    assert discover_quantization_policy(config) == {
        "weight": {
            "format": "channel_scaled_fp8_e4m3",
            "channel_axis": 0,
            "per_tensor": False,
        },
        "activation": {
            "format": "dynamic_token_fp8_e4m3",
            "per_tensor": False,
        },
    }


def test_attaches_block_scale_to_fp8_parameter_by_tensor_structure() -> None:
    tensors = {
        "projection.weight": _tensor([256, 512], "F8_E4M3"),
        "projection.weight_scale_inv": _tensor([2, 4]),
    }
    parameters = {"projection": "projection.weight"}

    attach_block_quantization_scales(tensors, parameters)

    assert parameters == {
        "projection": "projection.weight",
        "projection_scale_inv": "projection.weight_scale_inv",
    }


def test_preserves_native_e8m0_block_scale_without_conversion() -> None:
    tensors = {
        "projection.weight": _tensor([256, 512], "F8_E4M3"),
        "projection.scale": _tensor([2, 4], "F8_E8M0"),
    }
    parameters = {"projection": "projection.weight"}

    attach_block_quantization_scales(tensors, parameters)

    assert parameters == {
        "projection": "projection.weight",
        "projection_scale": "projection.scale",
    }


def test_annotates_native_mxfp4_expert_storage_without_expansion(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text('{"text_config":{"expert_dtype":"fp4"}}')
    weight_name = "layers.0.ffn.experts.7.w1.weight"
    scale_name = "layers.0.ffn.experts.7.w1.scale"
    tensors = {
        weight_name: {**_tensor([6, 32], "I8"), "byte_count": 192},
        scale_name: {**_tensor([6, 2], "F8_E8M0"), "byte_count": 12},
    }

    annotate_quantized_linear_tensors(tmp_path, tensors)

    assert tensors[weight_name]["dtype"] == "I8"
    assert tensors[weight_name]["shape"] == [6, 32]
    assert tensors[weight_name]["byte_count"] == 192
    assert tensors[weight_name]["parameter_count"] == 384
    assert tensors[weight_name]["logical_shape"] == [6, 64]
    assert tensors[weight_name]["quantization"] == {
        "format": "mxfp4_e2m1",
        "bits": 4,
        "element_type": "float",
        "values_per_byte": 2,
        "packing_axis": 1,
        "packing_order": "low_nibble_then_high_nibble_along_k",
        "group_size": 32,
        "scales": scale_name,
        "scale_dtype": "F8_E8M0",
        "scale_mode": "power_of_two_per_output_row_k_group",
    }


@pytest.mark.parametrize(
    ("weight_shape", "weight_dtype", "scale_shape", "scale_dtype", "message"),
    [
        ([6, 31], "I8", [6, 2], "F8_E8M0", "logical K must be aligned"),
        ([6, 32], "F8_E4M3", [6, 2], "F8_E8M0", "packed I8 matrix"),
        ([6, 32], "I8", [6, 1], "F8_E8M0", "requires F8_E8M0 scale"),
        ([6, 32], "I8", [6, 2], "BF16", "requires F8_E8M0 scale"),
    ],
)
def test_rejects_invalid_mxfp4_storage_contract(
    tmp_path: Path,
    weight_shape: list[int],
    weight_dtype: str,
    scale_shape: list[int],
    scale_dtype: str,
    message: str,
) -> None:
    (tmp_path / "config.json").write_text('{"expert_dtype":"fp4"}')
    weight_name = "layers.0.ffn.experts.0.w2.weight"
    tensors = {
        weight_name: _tensor(weight_shape, weight_dtype),
        "layers.0.ffn.experts.0.w2.scale": _tensor(scale_shape, scale_dtype),
    }

    with pytest.raises(ModelTranspileError, match=message):
        annotate_quantized_linear_tensors(tmp_path, tensors)


def test_does_not_reinterpret_int8_storage_without_fp4_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text("{}")
    weight_name = "layers.0.ffn.experts.0.w1.weight"
    tensors = {weight_name: _tensor([6, 32], "I8")}

    annotate_quantized_linear_tensors(tmp_path, tensors)

    assert tensors[weight_name] == _tensor([6, 32], "I8")


def test_annotates_auto_gptq_storage_as_logical_packed_linear(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "packing_format": "auto_round:auto_gptq",
            "bits": 4,
            "group_size": 128,
            "sym": true
          }
        }"""
    )
    tensors = {
        "projection.qweight": _tensor([64, 768], "I32"),
        "projection.qzeros": _tensor([4, 96], "I32"),
        "projection.scales": _tensor([4, 768], "F16"),
    }

    annotate_quantized_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.qweight"}
    attach_packed_linear_quantization(tensors, parameters)

    assert tensors["projection.qweight"]["logical_shape"] == [768, 512]
    assert tensors["projection.qweight"]["quantization"] == {
        "format": "auto_gptq",
        "bits": 4,
        "group_size": 128,
        "symmetric": True,
        "packing_layout": "input_major_packed_columns",
        "zero_point_encoding": "fixed_8",
        "execution_zero_point_encoding": "fixed_8",
        "scales": "projection.scales",
    }
    assert parameters == {
        "projection": "projection.qweight",
        "projection_scales": "projection.scales",
    }


def test_annotates_asymmetric_auto_gptq_zero_points_as_compile_only_source_data(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "packing_format": "auto_round:auto_gptq",
            "bits": 4,
            "group_size": 128,
            "sym": false
          }
        }"""
    )
    tensors = {
        "projection.qweight": _tensor([64, 768], "I32"),
        "projection.qzeros": _tensor([4, 96], "I32"),
        "projection.scales": _tensor([4, 768], "F16"),
    }

    annotate_quantized_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.qweight"}
    attach_packed_linear_quantization(tensors, parameters)

    assert tensors["projection.qweight"]["quantization"] == {
        "format": "auto_gptq",
        "bits": 4,
        "group_size": 128,
        "symmetric": False,
        "zero_point_add": 1,
        "packing_layout": "input_major_packed_columns",
        "zero_point_encoding": "packed_per_group_output",
        "execution_zero_point_encoding": "fixed_8",
        "qzeros": "projection.qzeros",
        "scales": "projection.scales",
    }
    assert parameters == {
        "projection": "projection.qweight",
        "projection_scales": "projection.scales",
    }


def test_rejects_non_boolean_auto_gptq_symmetry_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "packing_format": "auto_round:auto_gptq",
            "bits": 4,
            "group_size": 128,
            "sym": "false"
          }
        }"""
    )

    with pytest.raises(ModelTranspileError, match="invalid sym value"):
        annotate_quantized_linear_tensors(
            tmp_path,
            {
                "projection.qweight": _tensor([64, 768], "I32"),
                "projection.qzeros": _tensor([4, 96], "I32"),
                "projection.scales": _tensor([4, 768], "F16"),
            },
        )


def test_annotates_compressed_tensors_int4_storage_by_structure(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "format": "pack-quantized",
            "config_groups": {
              "linear": {
                "format": "pack-quantized",
                "weights": {
                  "type": "int",
                  "num_bits": 4,
                  "group_size": 32,
                  "symmetric": true
                }
              }
            }
          }
        }"""
    )
    tensors = {
        "projection.weight_packed": _tensor([768, 64], "I32"),
        "projection.weight_scale": _tensor([768, 16], "BF16"),
        "projection.weight_shape": _tensor([2], "I64"),
    }

    annotate_quantized_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.weight_packed"}
    attach_packed_linear_quantization(tensors, parameters)

    assert tensors["projection.weight_packed"]["logical_shape"] == [768, 512]
    assert tensors["projection.weight_packed"]["quantization"] == {
        "format": "compressed_tensors_pack_quantized",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "signed_offset": 8,
        "scales": "projection.weight_scale",
    }
    assert parameters == {
        "projection": "projection.weight_packed",
        "projection_scales": "projection.weight_scale",
    }


def test_annotates_compressed_tensors_channel_fp8_as_native_block_grid(
    tmp_path: Path,
) -> None:
    (tmp_path / "config.json").write_text(
        """{
          "quantization_config": {
            "format": "float-quantized",
            "config_groups": {
              "linear": {
                "format": "float-quantized",
                "weights": {
                  "type": "float",
                  "num_bits": 8,
                  "strategy": "channel",
                  "dynamic": false,
                  "symmetric": true
                },
                "input_activations": {
                  "type": "float",
                  "num_bits": 8,
                  "strategy": "token",
                  "dynamic": true,
                  "symmetric": true
                }
              }
            }
          }
        }"""
    )
    tensors = {
        "projection.weight": {
            **_tensor([768, 512], "F8_E4M3"),
            "source_file": "/models/source.safetensors",
            "source_header_bytes": 128,
            "data_offsets": [0, 768 * 512],
        },
        "projection.weight_scale": {
            **_tensor([768, 1], "BF16"),
            "source_file": "/models/source.safetensors",
            "source_header_bytes": 128,
            "data_offsets": [768 * 512, 768 * 512 + 768 * 2],
        },
    }

    annotate_quantized_linear_tensors(tmp_path, tensors)
    parameters = {"projection": "projection.weight"}
    attach_block_quantization_scales(tensors, parameters)
    attach_packed_linear_quantization(tensors, parameters)

    execution_scale = tensors["projection.weight_scale_inv"]
    assert execution_scale["shape"] == [768, 4]
    assert execution_scale["dtype"] == "BF16"
    assert execution_scale["derived"] == {
        "kind": "fp8_channel_scale_to_block_grid",
        "source_tensor": "projection.weight_scale",
        "source_file": "/models/source.safetensors",
        "source_header_bytes": 128,
        "data_offsets": [768 * 512, 768 * 512 + 768 * 2],
        "source_shape": [768, 1],
        "block_columns": 128,
    }
    assert tensors["projection.weight"]["quantization"] == {
        "format": "compressed_tensors_channel_fp8",
        "weight_strategy": "channel",
        "activation_strategy": "dynamic_token",
        "source_scales": "projection.weight_scale",
        "execution_scales": "projection.weight_scale_inv",
        "execution_block_shape": [1, 128],
    }
    assert parameters == {
        "projection": "projection.weight",
        "projection_scale_inv": "projection.weight_scale_inv",
    }
