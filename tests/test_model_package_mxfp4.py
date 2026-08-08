from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from nerve.model_package_batching import (
    frame_parallel_batch_shader_file,
    is_sparse_moe_projection_shader,
    sparse_moe_route_scheduling_shader_file,
)
from nerve.model_package_assets import copy_tensor_package
from nerve.model_package_shader_compiler import compile_shader_artifacts
from nerve.model_package_shader_selection import (
    shader_file_for_node,
    workgroup_count_x_for_node,
)
from nerve.model_package_shader_templates import copy_shader_templates
from nerve.model_package_common import ModelCompileError
from nerve.model_package_tensors import physical_input_prequantization_spec
from nerve.model_transpiler_tensor_index import make_tensor_index


def _payload(path: Path) -> bytes:
    data = path.read_bytes()
    header_bytes = struct.unpack("<Q", data[:8])[0]
    return data[8 + header_bytes :]


def test_packages_mxfp4_experts_as_independent_byte_exact_resources(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "config.json").write_text('{"expert_dtype":"fp4"}')
    tensor_payloads = {
        "layers.0.ffn.experts.0.w1.weight": bytes(range(32)),
        "layers.0.ffn.experts.0.w1.scale": bytes(range(2)),
        "layers.0.ffn.experts.1.w1.weight": bytes(range(32, 64)),
        "layers.0.ffn.experts.1.w1.scale": bytes(range(2, 4)),
    }
    header: dict[str, object] = {}
    payload = bytearray()
    for name, values in tensor_payloads.items():
        start = len(payload)
        payload.extend(values)
        is_scale = name.endswith(".scale")
        header[name] = {
            "dtype": "F8_E8M0" if is_scale else "I8",
            "shape": [1, 2] if is_scale else [1, 32],
            "data_offsets": [start, len(payload)],
        }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    (source_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header_payload)) + header_payload + payload
    )

    source_index = make_tensor_index(source_dir)
    assert source_index["totals"] == {
        "tensor_count": 4,
        "parameter_count": 132,
        "byte_count": 68,
    }
    package_dir = tmp_path / "package"
    packaged = copy_tensor_package(source_index, package_dir)

    packaged_paths: set[str] = set()
    for name, expected_payload in tensor_payloads.items():
        info = packaged["tensors"][name]
        packaged_paths.add(str(info["source_file"]))
        assert _payload(package_dir / str(info["source_file"])) == expected_payload
        if name.endswith(".weight"):
            assert info["dtype"] == "I8"
            assert info["shape"] == [1, 32]
            assert info["logical_shape"] == [1, 64]
            assert info["quantization"]["format"] == "mxfp4_e2m1"
            assert info["quantization"]["scales"] == name.replace(".weight", ".scale")
    assert len(packaged_paths) == len(tensor_payloads)


def test_renders_demand_addressed_native_mxfp4_expert_kernels(
    tmp_path: Path,
) -> None:
    hidden_size = 128
    intermediate_size = 128
    num_experts = 2
    experts_per_token = 1
    refs: dict[str, dict[str, str]] = {}
    tensors: dict[str, dict[str, object]] = {}
    gate_mapping = []
    down_mapping = []

    def add_matrix(expert: int, projection: str, rows: int, columns: int) -> list[str]:
        prefix = f"expert_{expert}.{projection}"
        weight_id = f"expert_{expert}_{projection}"
        scale_id = f"{weight_id}_scale"
        refs[weight_id] = {"tensor": f"{prefix}.weight"}
        refs[scale_id] = {"tensor": f"{prefix}.scale"}
        tensors[f"{prefix}.weight"] = {
            "dtype": "I8",
            "shape": [rows, columns // 2],
            "logical_shape": [rows, columns],
            "layout": "row_major",
            "byte_count": rows * columns // 2,
            "quantization": {
                "format": "mxfp4_e2m1",
                "bits": 4,
                "element_type": "float",
                "values_per_byte": 2,
                "packing_axis": 1,
                "packing_order": "low_nibble_then_high_nibble_along_k",
                "group_size": 32,
                "scales": f"{prefix}.scale",
                "scale_dtype": "F8_E8M0",
                "scale_mode": "power_of_two_per_output_row_k_group",
            },
        }
        tensors[f"{prefix}.scale"] = {
            "dtype": "F8_E8M0",
            "shape": [rows, columns // 32],
            "layout": "row_major",
            "byte_count": rows * columns // 32,
        }
        return [weight_id, scale_id]

    for expert in range(num_experts):
        gate_mapping.append(
            {
                "selector": expert,
                "parameter_ids": [
                    *add_matrix(expert, "w1", intermediate_size, hidden_size),
                    *add_matrix(expert, "w3", intermediate_size, hidden_size),
                ],
            }
        )
        down_mapping.append(
            {
                "selector": expert,
                "parameter_ids": add_matrix(
                    expert, "w2", hidden_size, intermediate_size
                ),
            }
        )

    circuit = {"parameters": {"refs": refs}}
    common_attrs = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "experts_per_token": experts_per_token,
    }
    gate = {
        "id": "gate_up",
        "op": "independent_sparse_moe_gate_up",
        "inputs": ["hidden", "routes"],
        "outputs": ["intermediates"],
        "params": [
            parameter for entry in gate_mapping for parameter in entry["parameter_ids"]
        ],
        "attrs": {
            **common_attrs,
            "swiglu_limit": 10.0,
            "selected_parameter_accesses": [
                {"selection_signal": "routes", "mapping": gate_mapping}
            ],
        },
    }
    down = {
        "id": "down",
        "op": "independent_sparse_moe_down",
        "inputs": ["intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": [
            parameter for entry in down_mapping for parameter in entry["parameter_ids"]
        ],
        "attrs": {
            **common_attrs,
            "selected_parameter_accesses": [
                {"selection_signal": "routes", "mapping": down_mapping}
            ],
        },
    }
    tensor_index = {"tensors": tensors}
    dimensions = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }
    gate_shader = (
        "independent_sparse_moe_gate_up_mxfp4_e2m1_g32_h128_i128_e2_k1_limit10.comp"
    )
    down_shader = "independent_sparse_moe_down_mxfp4_e2m1_g32_h128_i128_e2_k1.comp"
    gate_batch_shader = gate_shader.replace(
        "independent_sparse_moe_gate_up_",
        "independent_sparse_moe_gate_up_batch1_",
    )
    down_batch_shader = down_shader.replace(
        "independent_sparse_moe_down_",
        "independent_sparse_moe_down_batch1_",
    )
    prequant_gate = deepcopy(gate)
    prequant_gate["inputs"] = ["hidden_fp8", "hidden_scale", "routes"]
    prequant_gate["attrs"]["physical_input_contract"] = (
        "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1"
    )
    prequant_gate_shader = gate_shader.replace(
        "independent_sparse_moe_gate_up_",
        "independent_sparse_moe_gate_up_prequant_",
    )
    prequant_gate_batch_shader = prequant_gate_shader.replace(
        "independent_sparse_moe_gate_up_",
        "independent_sparse_moe_gate_up_batch1_",
    )
    resident_gate_shader = gate_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_resident_fp8_e4m3_g32",
    )
    resident_down_shader = down_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_resident_fp8_e4m3_g32",
    )
    adaptive_gate_shader = gate_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_adaptive_fp8_e4m3_g32",
    )
    adaptive_gate_batch_shader = gate_batch_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_adaptive_fp8_e4m3_g32",
    )
    adaptive_down_shader = down_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_adaptive_fp8_e4m3_g32",
    )
    adaptive_down_batch_shader = down_batch_shader.replace(
        "mxfp4_e2m1_g32",
        "mxfp4_e2m1_adaptive_fp8_e4m3_g32",
    )
    assert shader_file_for_node(circuit, gate, tensor_index, dimensions) == gate_shader
    assert shader_file_for_node(circuit, down, tensor_index, dimensions) == down_shader
    assert physical_input_prequantization_spec(
        circuit,
        gate,
        tensor_index,
        activation_quantization={
            "format": "dynamic_block_fp8_e4m3",
            "group_size": 128,
            "scale_format": "e8m0_power_of_two",
        },
    ) == {
        "contract": "bf16_blockwise_fp8_e4m3_e8m0_scale_f32.v1",
        "input_size": hidden_size,
        "block_columns": 128,
    }
    assert (
        shader_file_for_node(circuit, prequant_gate, tensor_index, dimensions)
        == prequant_gate_shader
    )
    assert frame_parallel_batch_shader_file(gate_shader) == gate_batch_shader
    assert frame_parallel_batch_shader_file(down_shader) == down_batch_shader
    assert (
        frame_parallel_batch_shader_file(prequant_gate_shader)
        == prequant_gate_batch_shader
    )
    assert is_sparse_moe_projection_shader(gate_batch_shader)
    assert sparse_moe_route_scheduling_shader_file(gate_shader) == (
        "moe_route_compact_batch1_i128_k1_t4.comp"
    )
    assert sparse_moe_route_scheduling_shader_file(down_shader) == (
        "moe_route_count_batch1_i128_k1_t2.comp"
    )
    assert workgroup_count_x_for_node(circuit, gate, tensor_index) == 4
    assert workgroup_count_x_for_node(circuit, down, tensor_index) == 2

    tensors["expert_0.w1.weight"]["byte_count"] = hidden_size - 1
    with pytest.raises(ModelCompileError, match="incompatible MXFP4"):
        shader_file_for_node(circuit, gate, tensor_index, dimensions)
    tensors["expert_0.w1.weight"]["byte_count"] = (
        intermediate_size * hidden_size // 2
    )

    shader_source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    copy_shader_templates(
        shader_source_dir,
        tmp_path,
        {
            gate_shader,
            gate_batch_shader,
            prequant_gate_shader,
            prequant_gate_batch_shader,
            down_shader,
            down_batch_shader,
            resident_gate_shader,
            resident_down_shader,
            adaptive_gate_shader,
            adaptive_gate_batch_shader,
            adaptive_down_shader,
            adaptive_down_batch_shader,
        },
    )
    gate_source = (tmp_path / gate_shader).read_text()
    gate_batch_source = (tmp_path / gate_batch_shader).read_text()
    prequant_gate_source = (tmp_path / prequant_gate_shader).read_text()
    prequant_gate_batch_source = (tmp_path / prequant_gate_batch_shader).read_text()
    down_source = (tmp_path / down_shader).read_text()
    resident_gate_source = (tmp_path / resident_gate_shader).read_text()
    resident_down_source = (tmp_path / resident_down_shader).read_text()
    adaptive_gate_source = (tmp_path / adaptive_gate_shader).read_text()
    adaptive_gate_batch_source = (tmp_path / adaptive_gate_batch_shader).read_text()
    adaptive_down_source = (tmp_path / adaptive_down_shader).read_text()
    adaptive_down_batch_source = (tmp_path / adaptive_down_batch_shader).read_text()
    assert "const uint DYNAMIC_PARAMETER_COUNT = 4u;" in gate_source
    assert "const uint DYNAMIC_PARAMETER_COUNT = 2u;" in down_source
    assert "const uint MXFP4_E4M3_BITS[8]" in gate_source
    assert "0x00u, 0x30u, 0x38u, 0x3cu, 0x40u, 0x44u, 0x48u, 0x4cu" in gate_source
    assert "uint8_t(mxfp4_e4m3_bits(lo & 0x0fu))" in gate_source
    assert "uintBitsToFloat(scale_byte << 23u)" in gate_source
    assert "float rounded_fp8_scale(float block_max)" in gate_source
    assert "fp8_dot4_acc32" in gate_source
    assert "fe4m3vec4 read_mxfp4x4" in gate_source
    assert "min(gate_value, SWIGLU_LIMIT)" in gate_source
    assert "clamp(up_value, -SWIGLU_LIMIT, SWIGLU_LIMIT)" in gate_source
    assert "expert * DYNAMIC_PARAMETER_COUNT + parameter" in gate_source
    assert "bool dynamic_parameter_record_is_valid" in gate_source
    assert "address_table_slot_count" in gate_source
    assert "address_record_byte_count" in gate_source
    assert "address_record_is_resident" in gate_source
    assert "route_index >= route_capacity" in gate_batch_source
    assert "bool dynamic_parameter_record_is_valid" in down_source
    assert "batch_control.owned_route_count" in gate_batch_source
    assert "batch_index * HIDDEN_WORDS" in gate_batch_source
    assert "readonly buffer QuantizedHidden" in prequant_gate_source
    assert "readonly buffer HiddenScales" in prequant_gate_source
    assert "binding = 5) readonly buffer DynamicParameterSlots" in prequant_gate_source
    assert "#define PREQUANTIZED_INPUT 1" in prequant_gate_source
    assert "#define PREEXPANDED_FP8 0" in gate_source
    assert "#define PREEXPANDED_FP8 1" in resident_gate_source
    assert "#define PREEXPANDED_FP8 1" in resident_down_source
    for adaptive_source in (
        adaptive_gate_source,
        adaptive_gate_batch_source,
        adaptive_down_source,
        adaptive_down_batch_source,
    ):
        assert "#define PREEXPANDED_FP8 0" in adaptive_source
        assert "#define DYNAMIC_WEIGHT_REPRESENTATION 1" in adaptive_source
        assert "dynamic_parameter_representation" in adaptive_source
        assert "read_compact_mxfp4x4" in adaptive_source
        assert "read_expanded_fp8x4" in adaptive_source
    assert "gate_preexpanded == up_preexpanded" in adaptive_gate_source
    assert "gate_preexpanded == up_preexpanded" in adaptive_gate_batch_source
    assert "column) >> 2u" in resident_gate_source
    assert "column) >> 2u" in resident_down_source
    assert "quantized_hidden.words[word]" in prequant_gate_source
    assert "batch_index * HIDDEN_FP8_WORDS + word" in prequant_gate_batch_source
    assert "batch_index * HIDDEN_BLOCKS + activation_block" in " ".join(
        prequant_gate_batch_source.split()
    )
    assert "{{" not in gate_source
    assert "{{" not in gate_batch_source
    assert "{{" not in prequant_gate_source
    assert "{{" not in prequant_gate_batch_source
    assert "{{" not in down_source
    assert "{{" not in resident_gate_source
    assert "{{" not in resident_down_source
    assert "{{" not in adaptive_gate_source
    assert "{{" not in adaptive_gate_batch_source
    assert "{{" not in adaptive_down_source
    assert "{{" not in adaptive_down_batch_source
    compile_shader_artifacts(tmp_path)
