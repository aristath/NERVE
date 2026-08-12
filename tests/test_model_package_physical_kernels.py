from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from nerve.model_package_assets import copy_tensor_package
from nerve.model_package_batching import frame_parallel_batch_shader_file
from nerve.model_package_common import ModelCompileError
from nerve.model_package_derived_tensors import (
    TP_INPUT_BLOCK_COLUMNS,
    derive_tensor_parallel_independent_expert_tensors,
    derive_tensor_parallel_linear_tensors,
    ensure_input_block_major_tensor,
    input_block_major_tensor_name,
    transposed_tensor_name,
)
from nerve.model_package_physical_kernels import (
    local_shard_intermediates_for_node,
    physical_kernel_implementations_for_node,
)
from nerve.model_package_independent_experts import (
    independent_sparse_moe_shader_file,
)
from nerve.model_package_shader_compiler import compile_shader_artifacts
from nerve.model_package_shader_templates import copy_shader_templates
from nerve.model_package_spirv_requirements import required_shader_files
from nerve.model_package_tensors import (
    compiled_safetensors_header,
    write_compiled_derived_matrix_reorder,
)
from nerve.physical_execution_contracts import (
    build_kernel_physical_execution_contracts,
)


class NativeTarget:
    def supports_native_dtype(self, dtype: str) -> bool:
        return dtype in {"BF16", "F8_E4M3"}


def bf16_down_fixture(tmp_path: Path) -> tuple[dict, dict, dict, bytes]:
    values = np.arange(4 * 256, dtype="<u2").reshape(4, 256)
    payload = values.tobytes(order="C")
    source = tmp_path / "down.safetensors"
    header = compiled_safetensors_header(
        "down.weight",
        dtype="BF16",
        shape=[4, 256],
        byte_count=len(payload),
        layout="row_major",
    )
    source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    tensor_index = {
        "tensors": {
            "down.weight": {
                "dtype": "BF16",
                "shape": [4, 256],
                "parameter_count": 4 * 256,
                "byte_count": len(payload),
                "layout": "row_major",
                "source_file": str(source),
                "source_header_bytes": len(header),
                "data_offsets": [0, len(payload)],
            }
        }
    }
    node = {
        "id": "down_residual",
        "op": "linear_residual",
        "inputs": ["activated", "residual"],
        "outputs": ["hidden"],
        "params": ["down"],
    }
    circuit = {
        "nodes": [node],
        "parameters": {"refs": {"down": {"tensor": "down.weight"}}},
    }
    return node, circuit, tensor_index, payload


def fp8_down_fixture(
    tmp_path: Path,
    *,
    output_rows: int = 4,
    input_columns: int = 256,
) -> tuple[dict, dict, dict, np.ndarray]:
    if output_rows % 2 or input_columns % TP_INPUT_BLOCK_COLUMNS:
        raise ValueError("FP8 fixture dimensions must satisfy the physical ABI")
    weight_values = np.arange(
        output_rows * input_columns,
        dtype="u1",
    ).reshape(output_rows, input_columns)
    scale_values = np.arange(
        2 * (input_columns // TP_INPUT_BLOCK_COLUMNS),
        dtype="<u2",
    ).reshape(2, input_columns // TP_INPUT_BLOCK_COLUMNS)
    tensor_index: dict = {"tensors": {}}
    for name, dtype, values in (
        ("down.weight", "F8_E4M3", weight_values),
        ("down.weight_scale_inv", "BF16", scale_values),
    ):
        payload = values.tobytes(order="C")
        source = tmp_path / f"{name}.safetensors"
        header = compiled_safetensors_header(
            name,
            dtype=dtype,
            shape=list(values.shape),
            byte_count=len(payload),
            layout="row_major",
        )
        source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        tensor_index["tensors"][name] = {
            "dtype": dtype,
            "shape": list(values.shape),
            "parameter_count": int(values.size),
            "byte_count": len(payload),
            "layout": "row_major",
            "source_file": str(source),
            "source_header_bytes": len(header),
            "data_offsets": [0, len(payload)],
        }
    node = {
        "id": "down_residual",
        "op": "linear_residual",
        "inputs": ["activated", "residual"],
        "outputs": ["hidden"],
        "params": ["down", "down_scale"],
    }
    circuit = {
        "nodes": [node],
        "parameters": {
            "refs": {
                "down": {"tensor": "down.weight"},
                "down_scale": {"tensor": "down.weight_scale_inv"},
            }
        },
    }
    return node, circuit, tensor_index, scale_values


def test_compiler_derives_contiguous_input_block_major_weights(tmp_path: Path) -> None:
    _, circuit, tensor_index, payload = bf16_down_fixture(tmp_path)
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    lowered_index = {"graph": {"circuits": [{"circuit": "circuit.json"}]}}

    derive_tensor_parallel_linear_tensors(
        lowered_index,
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    name = input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
    info = tensor_index["tensors"][name]
    assert info["shape"] == [2, 4, 128]
    assert info["logical_shape"] == [4, 256]
    assert info["physical_execution_only"] is True

    destination = tmp_path / "input-block-major.safetensors"
    header_bytes, _, _ = write_compiled_derived_matrix_reorder(
        tensor_name=name,
        info=info,
        destination=destination,
        layout="row_major",
    )
    reordered = np.frombuffer(
        destination.read_bytes()[8 + header_bytes :], dtype="<u2"
    ).reshape(2, 4, 128)
    expected = np.frombuffer(payload, dtype="<u2").reshape(4, 2, 128).transpose(1, 0, 2)
    np.testing.assert_array_equal(reordered, expected)


def test_compiler_derives_dense_down_weight_from_the_real_prefusion_region(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    linear = {
        "id": "down",
        "op": "linear",
        "inputs": ["activated"],
        "outputs": ["projected"],
        "params": ["down"],
    }
    residual = {
        "id": "residual",
        "op": "residual_add",
        "inputs": ["residual", "projected"],
        "outputs": ["hidden"],
    }
    circuit["nodes"] = [linear, residual]
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))

    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    assert (
        input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
        in tensor_index["tensors"]
    )


def test_compiler_does_not_derive_dense_down_weight_for_an_unfusible_region(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    circuit["nodes"] = [
        {
            "id": "down",
            "op": "linear",
            "inputs": ["activated"],
            "outputs": ["projected"],
            "params": ["down"],
        },
        {
            "id": "residual",
            "op": "residual_add",
            "inputs": ["residual", "projected"],
            "outputs": ["hidden"],
        },
        {
            "id": "observer",
            "op": "identity",
            "inputs": ["projected"],
            "outputs": ["observed"],
        },
    ]
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))

    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    assert (
        input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
        not in tensor_index["tensors"]
    )


def test_contract_owns_the_input_column_shader_and_physical_weight(
    tmp_path: Path,
) -> None:
    node, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )
    implementation = physical_kernel_implementations_for_node(
        circuit, node, tensor_index
    )[0]
    implementation["shader_path"] = implementation["shader_path"].replace(
        ".comp", ".spv"
    )
    scalar_path = tmp_path / "shaders" / "linear_residual_bf16.spv"
    physical_path = tmp_path / implementation["shader_path"]
    scalar_path.parent.mkdir()
    scalar_path.write_bytes(b"scalar")
    physical_path.write_bytes(b"input columns")
    kernel = {
        "source_node_ids": ["down", "residual"],
        "semantic_module_ids": ["layer.feed_forward.down_residual"],
        "shader_path": "shaders/linear_residual_bf16.spv",
        "local_size_x": 64,
        "workgroup_count_x": 2,
        "batch_implementations": [],
        "physical_implementations": [implementation],
    }

    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel=kernel,
        package_dir=tmp_path,
    )
    distributed = next(
        contract
        for contract in contracts
        if contract["execution_form"] == "partitioned_input_partial_output"
    )
    physical_weight = input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    )
    assert implementation["execution_shape"] == "single_and_multi_lane"
    assert distributed["execution_shape"] == "single_and_multi_lane"
    assert distributed["artifacts"][0]["path"] == implementation["shader_path"]
    assert distributed["artifacts"][0]["path"] != kernel["shader_path"]
    assert distributed["parameter_partitions"] == [
        {
            "binding": 3,
            "resource": physical_weight,
            "dimension": 0,
            "kind": "contiguous",
            "alignment_elements": 1,
            "logical_elements_per_index": 128,
        }
    ]
    assert distributed["outputs"][0]["reduction"]["finalization"] == {
        "kind": "add_bf16_residual_to_bf16",
        "residual_binding": 1,
    }
    assert distributed["phases"] == ["decode", "prefill"]
    assert distributed["equivalence"]["output"] == "absolute_relative_tolerance"
    assert distributed["local_intermediates"] == []


def test_compiler_declares_only_an_executable_local_shard_handoff(
    tmp_path: Path,
) -> None:
    down, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    gate_up = {
        "id": "gate_up",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["normalized"],
        "outputs": ["activated"],
        "params": ["gate", "up"],
    }
    circuit["nodes"].insert(0, gate_up)
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    compiled_gate_up = {**gate_up, "_physical_contract_member_node_ids": ["gate_up"]}
    assert local_shard_intermediates_for_node(
        circuit, compiled_gate_up, tensor_index
    ) == [
        {
            "signal": "activated",
            "producer_binding": 1,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]
    assert physical_kernel_implementations_for_node(circuit, down, tensor_index)[0][
        "local_intermediates"
    ] == [
        {
            "signal": "activated",
            "producer_binding": 1,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]

    circuit["nodes"].append(
        {
            "id": "side_consumer",
            "op": "identity",
            "inputs": ["activated"],
            "outputs": ["side"],
            "params": [],
        }
    )
    assert (
        local_shard_intermediates_for_node(circuit, compiled_gate_up, tensor_index)
        == []
    )
    circuit["nodes"].pop()

    down["inputs"][0] = "another-signal"
    assert local_shard_intermediates_for_node(circuit, gate_up, tensor_index) == []


def test_dense_ffn_pair_emits_one_compatible_local_tensor_parallel_island(
    tmp_path: Path,
) -> None:
    down, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    gate_up = {
        "id": "gate_up",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["normalized"],
        "outputs": ["activated"],
        "params": ["gate", "up"],
    }
    circuit["nodes"].insert(0, gate_up)
    circuit["parameters"]["refs"].update(
        {
            "gate": {"tensor": "gate.weight"},
            "up": {"tensor": "up.weight"},
        }
    )
    for tensor in ("gate.weight", "up.weight"):
        tensor_index["tensors"][tensor] = {
            "dtype": "BF16",
            "shape": [256, 4],
            "parameter_count": 256 * 4,
            "byte_count": 256 * 4 * 2,
            "layout": "row_major",
        }

    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir()
    gate_shader = shader_dir / "gate-up.spv"
    down_shader = shader_dir / "down.spv"
    gate_shader.write_bytes(b"gate-up")
    down_shader.write_bytes(b"down")
    down_implementation = physical_kernel_implementations_for_node(
        circuit, down, tensor_index
    )[0]
    down_implementation["shader_path"] = down_implementation["shader_path"].replace(
        ".comp", ".spv"
    )
    physical_down_shader = tmp_path / down_implementation["shader_path"]
    physical_down_shader.write_bytes(b"input-column-down")

    gate_contracts = build_kernel_physical_execution_contracts(
        node=gate_up,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": ["gate_up"],
            "semantic_module_ids": ["layer.feed_forward.gate_up"],
            "shader_path": str(gate_shader.relative_to(tmp_path)),
            "local_size_x": 64,
            "workgroup_count_x": 128,
            "batch_implementations": [],
            "physical_implementations": [],
        },
        package_dir=tmp_path,
    )
    down_contracts = build_kernel_physical_execution_contracts(
        node=down,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": ["down_residual"],
            "semantic_module_ids": ["layer.feed_forward.down_residual"],
            "shader_path": str(down_shader.relative_to(tmp_path)),
            "local_size_x": 64,
            "workgroup_count_x": 2,
            "batch_implementations": [],
            "physical_implementations": [down_implementation],
        },
        package_dir=tmp_path,
    )

    gate_distributed = next(
        contract
        for contract in gate_contracts
        if contract["strategy"] == "tensor_parallel"
    )
    down_distributed = next(
        contract
        for contract in down_contracts
        if contract["execution_form"] == "partitioned_input_partial_output"
    )
    expected_handoff = [
        {
            "signal": "activated",
            "producer_binding": 1,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]
    assert gate_distributed["execution_form"] == ("replicated_input_partitioned_output")
    assert gate_distributed["local_intermediates"] == expected_handoff
    assert down_distributed["local_intermediates"] == expected_handoff
    assert gate_distributed["outputs"][0]["collection"] == "concatenated"
    assert down_distributed["inputs"][0] == {
        "binding": 0,
        "distribution": "sharded",
        "dimension": 0,
        "alignment_elements": TP_INPUT_BLOCK_COLUMNS,
    }
    assert gate_distributed["partition_extent"]["elements"] == 256
    assert down_distributed["partition_extent"]["elements"] == 256
    assert down_distributed["outputs"][0]["reduction"] == {
        "operation": "sum_f32",
        "dimension_name": "output_rows",
        "finalization": {
            "kind": "add_bf16_residual_to_bf16",
            "residual_binding": 1,
        },
    }
    assert down_distributed["formats"]["accumulation"] == "f32"
    assert down_distributed["phases"] == ["decode", "prefill"]


def test_compiler_declares_expert_intermediate_private_on_both_kernels() -> None:
    gate_up = {
        "id": "expert_gate_up",
        "op": "independent_sparse_moe_gate_up",
        "inputs": ["normalized", "routes"],
        "outputs": ["expert_intermediates"],
        "params": ["gate", "up"],
    }
    down = {
        "id": "expert_down",
        "op": "independent_sparse_moe_down",
        "inputs": ["expert_intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": ["down"],
    }
    circuit = {"nodes": [gate_up, down]}
    expected = [
        {
            "signal": "expert_intermediates",
            "producer_binding": 2,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]

    assert local_shard_intermediates_for_node(circuit, gate_up, {}) == expected
    assert local_shard_intermediates_for_node(circuit, down, {}) == expected

    circuit["nodes"].append(
        {
            "id": "observer",
            "op": "identity",
            "inputs": ["expert_intermediates"],
            "outputs": ["observed"],
            "params": [],
        }
    )
    assert local_shard_intermediates_for_node(circuit, gate_up, {}) == []
    assert local_shard_intermediates_for_node(circuit, down, {}) == []


def test_compiler_derives_fp8_weight_and_scale_physical_resources(
    tmp_path: Path,
) -> None:
    node, circuit, tensor_index, scale_values = fp8_down_fixture(tmp_path)
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    weight = input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
    scale = transposed_tensor_name("down.weight_scale_inv")
    assert tensor_index["tensors"][weight]["shape"] == [2, 4, 128]
    assert tensor_index["tensors"][scale]["shape"] == [2, 2]
    implementation = physical_kernel_implementations_for_node(
        circuit, node, tensor_index
    )[0]
    assert [
        (partition["binding"], partition["resource"])
        for partition in implementation["parameter_partitions"]
    ] == [(3, weight), (4, scale)]

    scale_destination = tmp_path / "scale-transposed.safetensors"
    scale_header_bytes, _, _ = write_compiled_derived_matrix_reorder(
        tensor_name=scale,
        info=tensor_index["tensors"][scale],
        destination=scale_destination,
        layout="row_major",
    )
    actual_scale = np.frombuffer(
        scale_destination.read_bytes()[8 + scale_header_bytes :], dtype="<u2"
    ).reshape(2, 2)
    np.testing.assert_array_equal(actual_scale, scale_values.T)


def test_fp8_dense_ffn_pair_declares_one_local_tp_island_for_decode_and_prefill(
    tmp_path: Path,
) -> None:
    down, circuit, tensor_index, _ = fp8_down_fixture(
        tmp_path,
        output_rows=256,
    )
    gate_up = {
        "id": "gate_up",
        "op": "parallel_linear_silu_multiply",
        "inputs": ["normalized"],
        "outputs": ["activated"],
        "params": ["gate", "gate_scale", "up", "up_scale"],
    }
    circuit["nodes"].insert(0, gate_up)
    circuit["parameters"]["refs"].update(
        {
            "gate": {"tensor": "gate.weight"},
            "gate_scale": {"tensor": "gate.weight_scale_inv"},
            "up": {"tensor": "up.weight"},
            "up_scale": {"tensor": "up.weight_scale_inv"},
        }
    )
    for projection in ("gate", "up"):
        tensor_index["tensors"][f"{projection}.weight"] = {
            "dtype": "F8_E4M3",
            "shape": [256, 256],
            "parameter_count": 256 * 256,
            "byte_count": 256 * 256,
            "layout": "row_major",
        }
        tensor_index["tensors"][f"{projection}.weight_scale_inv"] = {
            "dtype": "BF16",
            "shape": [2, 2],
            "parameter_count": 4,
            "byte_count": 8,
            "layout": "row_major",
        }

    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir()
    gate_shader = shader_dir / "parallel_linear_silu_multiply_fp8_e4m3.spv"
    gate_batch_shader = (
        shader_dir / "parallel_linear_silu_multiply_batch16_fp8_e4m3.spv"
    )
    down_shader = shader_dir / "linear_residual_fp8_e4m3.spv"
    gate_shader.write_bytes(b"gate-up decode")
    gate_batch_shader.write_bytes(b"gate-up prefill")
    down_shader.write_bytes(b"canonical down")
    [down_implementation] = physical_kernel_implementations_for_node(
        circuit,
        down,
        tensor_index,
    )
    down_implementation["shader_path"] = down_implementation["shader_path"].replace(
        ".comp", ".spv"
    )
    physical_down_shader = tmp_path / down_implementation["shader_path"]
    physical_down_shader.write_bytes(b"partitioned input-column down")

    gate_contracts = build_kernel_physical_execution_contracts(
        node=gate_up,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": ["gate_up"],
            "semantic_module_ids": ["layer.feed_forward.gate_up"],
            "shader_path": str(gate_shader.relative_to(tmp_path)),
            "local_size_x": 64,
            "workgroup_count_x": 2,
            "batch_implementations": [
                {
                    "execution_domain": "decode_and_prefill",
                    "lane_tile_width": 16,
                    "stages": [
                        {
                            "shader_path": str(gate_batch_shader.relative_to(tmp_path)),
                            "local_size_x": 64,
                            "workgroup_count_x": 2,
                        }
                    ],
                }
            ],
            "physical_implementations": [],
        },
        package_dir=tmp_path,
    )
    down_contracts = build_kernel_physical_execution_contracts(
        node=down,
        circuit=circuit,
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": ["down_residual"],
            "semantic_module_ids": ["layer.feed_forward.down_residual"],
            "shader_path": str(down_shader.relative_to(tmp_path)),
            "local_size_x": 64,
            "workgroup_count_x": 128,
            "batch_implementations": [],
            "physical_implementations": [down_implementation],
        },
        package_dir=tmp_path,
    )

    gate_distributed = [
        contract
        for contract in gate_contracts
        if contract["strategy"] == "tensor_parallel"
    ]
    [down_distributed] = [
        contract
        for contract in down_contracts
        if contract["execution_form"] == "partitioned_input_partial_output"
    ]
    expected_handoff = [
        {
            "signal": "activated",
            "producer_binding": 1,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]
    assert {
        (tuple(contract["phases"]), contract["execution_shape"])
        for contract in gate_distributed
    } == {
        (("decode",), "single_lane"),
        (("decode",), "multi_lane"),
        (("prefill",), "multi_lane"),
    }
    assert all(
        contract["execution_form"] == "replicated_input_partitioned_output"
        and contract["local_intermediates"] == expected_handoff
        and contract["partition_extent"]["elements"] == 256
        and contract["outputs"][0]["collection"] == "concatenated"
        for contract in gate_distributed
    )
    assert down_distributed["phases"] == ["decode", "prefill"]
    assert down_distributed["execution_shape"] == "single_and_multi_lane"
    assert down_distributed["local_intermediates"] == expected_handoff
    assert down_distributed["partition_extent"] == {
        "dimension_name": "input_columns",
        "elements": 256,
        "alignment_elements": TP_INPUT_BLOCK_COLUMNS,
    }
    assert down_distributed["inputs"][0]["distribution"] == "sharded"
    assert down_distributed["outputs"][0]["reduction"]["operation"] == "sum_f32"
    assert down_distributed["formats"] == {
        "storage": "f8_e4m3+bf16:input_block_major",
        "compute": "fp8_e4m3",
        "accumulation": "f32",
    }


def test_compiler_does_not_relabel_an_unsupported_dense_format_as_tp(
    tmp_path: Path,
) -> None:
    node, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    tensor_index["tensors"]["down.weight"]["dtype"] = "F16"
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))

    class TargetWithNativeF16:
        def supports_native_dtype(self, _dtype: str) -> bool:
            return True

    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=TargetWithNativeF16(),  # type: ignore[arg-type]
    )

    assert (
        input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
        not in tensor_index["tensors"]
    )
    assert (
        physical_kernel_implementations_for_node(
            circuit,
            node,
            tensor_index,
        )
        == []
    )


def test_partitioned_packaging_seals_each_derived_matrix_range(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )
    physical_weight = input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    )

    packaged = copy_tensor_package(
        tensor_index,
        tmp_path / "package",
        partition_counts={physical_weight: 2},
    )

    info = packaged["tensors"][physical_weight]
    integrity = info["partition_integrity"]
    assert integrity["partition_axis"] == 0
    assert integrity["partition_count"] == 2
    assert integrity["partition_byte_count"] == info["byte_count"] // 2
    digest_table = tmp_path / "package" / integrity["digest_table_path"]
    assert digest_table.stat().st_size == 64


@pytest.mark.parametrize(
    ("dtype", "numpy_dtype"),
    [("I8", "i1"), ("F8_E8M0", "u1")],
)
def test_partitioned_byte_matrix_reorders_preserve_exact_physical_ranges(
    tmp_path: Path,
    dtype: str,
    numpy_dtype: str,
) -> None:
    values = np.arange(3 * 8, dtype=numpy_dtype).reshape(3, 8)
    payload = values.tobytes(order="C")
    source = tmp_path / f"{dtype}.safetensors"
    header = compiled_safetensors_header(
        "source",
        dtype=dtype,
        shape=[3, 8],
        byte_count=len(payload),
        layout="row_major",
    )
    source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    tensor_index = {
        "tensors": {
            "source": {
                "dtype": dtype,
                "shape": [3, 8],
                "parameter_count": 24,
                "byte_count": len(payload),
                "layout": "row_major",
                "source_file": str(source),
                "source_header_bytes": len(header),
                "data_offsets": [0, len(payload)],
            }
        }
    }
    derived = ensure_input_block_major_tensor(
        tensor_index,
        source_tensor="source",
        block_columns=2,
    )

    destination = tmp_path / f"{dtype}-block-major.safetensors"
    header_bytes, _, partition_digests = write_compiled_derived_matrix_reorder(
        tensor_name=derived,
        info=tensor_index["tensors"][derived],
        destination=destination,
        layout="row_major",
        partition_count=4,
    )

    actual = np.frombuffer(
        destination.read_bytes()[8 + header_bytes :], dtype=numpy_dtype
    ).reshape(4, 3, 2)
    expected = values.reshape(3, 4, 2).transpose(1, 0, 2)
    np.testing.assert_array_equal(actual, expected)
    assert len(partition_digests) == 4


def independent_mxfp4_expert_pair_fixture(
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict]:
    tensor_index: dict = {"tensors": {}}
    refs: dict = {}
    gate_params: list[str] = []
    down_params: list[str] = []
    gate_mapping: list[dict] = []
    down_mapping: list[dict] = []
    for expert in range(2):
        gate_ids: list[str] = []
        for projection in ("gate", "up"):
            weight_id = f"expert_{expert}_{projection}_weight"
            scale_id = f"expert_{expert}_{projection}_scale"
            weight_name = f"expert.{expert}.{projection}.weight"
            scale_name = f"expert.{expert}.{projection}.scale"
            _write_mxfp4_fixture_tensor_pair(
                tmp_path,
                tensor_index,
                weight_name=weight_name,
                scale_name=scale_name,
                rows=256,
                columns=128,
            )
            refs[weight_id] = {"tensor": weight_name}
            refs[scale_id] = {"tensor": scale_name}
            gate_ids.extend((weight_id, scale_id))
        down_weight_id = f"expert_{expert}_down_weight"
        down_scale_id = f"expert_{expert}_down_scale"
        down_weight_name = f"expert.{expert}.down.weight"
        down_scale_name = f"expert.{expert}.down.scale"
        _write_mxfp4_fixture_tensor_pair(
            tmp_path,
            tensor_index,
            weight_name=down_weight_name,
            scale_name=down_scale_name,
            rows=128,
            columns=256,
        )
        refs[down_weight_id] = {"tensor": down_weight_name}
        refs[down_scale_id] = {"tensor": down_scale_name}
        gate_params.extend(gate_ids)
        down_params.extend((down_weight_id, down_scale_id))
        gate_mapping.append({"selector": expert, "parameter_ids": gate_ids})
        down_mapping.append(
            {
                "selector": expert,
                "parameter_ids": [down_weight_id, down_scale_id],
            }
        )
    common_attrs = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "experts_per_token": 2,
    }
    gate_up = {
        "id": "expert_gate_up",
        "op": "independent_sparse_moe_gate_up",
        "inputs": ["hidden", "routes"],
        "outputs": ["expert_intermediates"],
        "params": gate_params,
        "attrs": {
            **common_attrs,
            "swiglu_limit": 0.0,
            "selected_parameter_accesses": [
                {
                    "selection_signal": "routes",
                    "execution_signal": "routes",
                    "execution_calibration_word_base": 0x3F800000,
                    "mapping": gate_mapping,
                }
            ],
        },
    }
    down = {
        "id": "expert_down",
        "op": "independent_sparse_moe_down",
        "inputs": ["expert_intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": down_params,
        "attrs": {
            **common_attrs,
            "selected_parameter_accesses": [
                {
                    "selection_signal": "routes",
                    "execution_signal": "routes",
                    "execution_calibration_word_base": 0x3F800000,
                    "mapping": down_mapping,
                }
            ],
        },
    }
    circuit = {"nodes": [gate_up, down], "parameters": {"refs": refs}}
    return gate_up, down, circuit, tensor_index


def _write_mxfp4_fixture_tensor_pair(
    tmp_path: Path,
    tensor_index: dict,
    *,
    weight_name: str,
    scale_name: str,
    rows: int,
    columns: int,
) -> None:
    for name, dtype, shape in (
        (weight_name, "I8", [rows, columns // 2]),
        (scale_name, "F8_E8M0", [rows, columns // 32]),
    ):
        numpy_dtype = "i1" if dtype == "I8" else "u1"
        values = np.arange(math.prod(shape), dtype=numpy_dtype).reshape(shape)
        payload = values.tobytes(order="C")
        source = tmp_path / f"{name}.safetensors"
        header = compiled_safetensors_header(
            name,
            dtype=dtype,
            shape=shape,
            byte_count=len(payload),
            layout="row_major",
        )
        source.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        tensor_index["tensors"][name] = {
            "dtype": dtype,
            "shape": shape,
            **({"logical_shape": [rows, columns]} if dtype == "I8" else {}),
            "parameter_count": rows * columns
            if dtype == "I8"
            else rows * columns // 32,
            "byte_count": len(payload),
            "layout": "row_major",
            "source_file": str(source),
            "source_header_bytes": len(header),
            "data_offsets": [0, len(payload)],
        }
    tensor_index["tensors"][weight_name]["quantization"] = {
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


def test_compiler_derives_fragmentable_independent_expert_down_resources(
    tmp_path: Path,
) -> None:
    gate_up, down, circuit, tensor_index = independent_mxfp4_expert_pair_fixture(
        tmp_path
    )
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    circuit_path = lowered_dir / "circuit.json"
    circuit_path.write_text(json.dumps(circuit))

    derive_tensor_parallel_independent_expert_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    weight_name = "expert.0.down.weight"
    scale_name = "expert.0.down.scale"
    weight = input_block_major_tensor_name(weight_name, 64)
    scale = input_block_major_tensor_name(scale_name, 4)
    rewritten = json.loads(circuit_path.read_text())
    assert rewritten["parameters"]["refs"]["expert_0_down_weight"] == {"tensor": weight}
    assert rewritten["parameters"]["refs"]["expert_0_down_scale"] == {"tensor": scale}
    assert tensor_index["tensors"][weight]["shape"] == [2, 128, 64]
    assert tensor_index["tensors"][weight]["logical_shape"] == [128, 256]
    assert tensor_index["tensors"][scale]["shape"] == [2, 128, 4]
    assert tensor_index["tensors"][weight]["source_integrity_partition_count"] == 2
    assert tensor_index["tensors"][scale]["source_integrity_partition_count"] == 2
    assert tensor_index["tensors"][weight]["quantization"]["scales"] == scale
    assert "physical_execution_only" not in tensor_index["tensors"][weight]
    for parameter_id in gate_up["params"]:
        tensor_name = circuit["parameters"]["refs"][parameter_id]["tensor"]
        assert (
            tensor_index["tensors"][tensor_name]["source_integrity_partition_count"]
            == 2
        )

    shader_file = independent_sparse_moe_shader_file(
        rewritten,
        rewritten["nodes"][1],
        tensor_index,
    )
    assert "down_input_block_major_b128_mxfp4" in shader_file
    gate_shader_file = independent_sparse_moe_shader_file(
        rewritten,
        rewritten["nodes"][0],
        tensor_index,
    )
    gate_implementations = physical_kernel_implementations_for_node(
        rewritten, rewritten["nodes"][0], tensor_index
    )
    down_implementations = physical_kernel_implementations_for_node(
        rewritten, rewritten["nodes"][1], tensor_index
    )
    assert [
        implementation["execution_shape"] for implementation in gate_implementations
    ] == [
        "single_lane",
        "multi_lane",
    ]
    assert [
        implementation["execution_shape"] for implementation in down_implementations
    ] == [
        "single_lane",
        "multi_lane",
    ]
    assert [implementation["phases"] for implementation in gate_implementations] == [
        ["decode"],
        ["decode", "prefill"],
    ]
    assert [implementation["phases"] for implementation in down_implementations] == [
        ["decode"],
        ["decode", "prefill"],
    ]
    gate_implementation = gate_implementations[0]
    down_implementation = down_implementations[0]
    assert gate_implementation["strategy"] == "tensor_parallel_expert"
    assert gate_implementation["execution_form"] == (
        "replicated_input_partitioned_output"
    )
    assert gate_implementation["local_intermediates"] == [
        {
            "signal": "expert_intermediates",
            "producer_binding": 2,
            "consumer_binding": 0,
            "format": "bf16:route_major_local_rows",
        }
    ]
    assert [
        partition["parameter_slot"]
        for partition in gate_implementation["selected_resource_partitions"][0][
            "parameter_partitions"
        ]
    ] == [0, 1, 2, 3]
    assert down_implementation["strategy"] == "tensor_parallel_expert"
    assert down_implementation["execution_form"] == "partitioned_input_partial_output"
    assert down_implementation["outputs"][0]["reduction"] == {
        "operation": "sum_f32",
        "dimension_name": "expert_output_elements",
        "finalization": {"kind": "store_f32_to_bf16"},
    }
    batch_shader_file = frame_parallel_batch_shader_file(shader_file)
    assert batch_shader_file is not None
    physical_shader_files = {
        Path(implementation["shader_path"]).name
        for implementation in (*gate_implementations, *down_implementations)
    }
    rendered_dir = tmp_path / "shaders"
    copy_shader_templates(
        Path(__file__).parents[1] / "runtime-rs" / "shaders",
        rendered_dir,
        {
            shader_file,
            gate_shader_file,
            batch_shader_file,
            *physical_shader_files,
        },
    )
    for rendered_shader in rendered_dir.glob("*.comp"):
        source = rendered_shader.read_text()
        assert "{{" not in source
        if "_down" in rendered_shader.name:
            assert "#define INPUT_BLOCK_MAJOR 1" in source
            assert "block * HIDDEN_SIZE + row" in source
        if rendered_shader.name in physical_shader_files:
            assert "#define TENSOR_PARALLEL 1" in source
            if "_batch1_" in rendered_shader.name:
                assert "gl_WorkGroupID.y" in source
                assert "batch_index" in source
                if "_gate_up_" in rendered_shader.name:
                    assert "local_frame_words" in source
                else:
                    assert "InputRange" in source
    compile_shader_artifacts(rendered_dir)

    for node, canonical_shader, implementations in (
        (rewritten["nodes"][0], gate_shader_file, gate_implementations),
        (rewritten["nodes"][1], shader_file, down_implementations),
    ):
        for implementation in implementations:
            implementation["shader_path"] = implementation["shader_path"].replace(
                ".comp", ".spv"
            )
        kernel = {
            "source_node_ids": [node["id"]],
            "semantic_module_ids": [f"layer.feed_forward.{node['id']}"],
            "shader_path": f"shaders/{canonical_shader.replace('.comp', '.spv')}",
            "local_size_x": 512,
            "workgroup_count_x": 1,
            "batch_implementations": [],
            "physical_implementations": implementations,
        }
        contracts = build_kernel_physical_execution_contracts(
            node=node,
            circuit=rewritten,
            tensor_index=tensor_index,
            kernel=kernel,
            package_dir=tmp_path,
        )
        distributed = [
            contract
            for contract in contracts
            if contract["strategy"] == "tensor_parallel_expert"
        ]
        assert len(distributed) == 2
        assert {
            (contract["execution_shape"], tuple(contract["phases"]))
            for contract in distributed
        } == {
            ("single_lane", ("decode",)),
            ("multi_lane", ("decode", "prefill")),
        }
        assert {contract["artifacts"][0]["path"] for contract in distributed} == {
            implementation["shader_path"] for implementation in implementations
        }

    packaged = copy_tensor_package(
        tensor_index,
        tmp_path / "package",
        partition_counts={weight: 2, scale: 2},
    )
    assert packaged["tensors"][weight]["partition_integrity"]["partition_count"] == 2
    assert packaged["tensors"][scale]["partition_integrity"]["partition_count"] == 2


def test_compiler_skips_a_linear_without_legal_physical_geometry(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    tensor_index["tensors"]["down.weight"]["shape"] = [3, 256]
    tensor_index["tensors"]["down.weight"]["parameter_count"] = 3 * 256
    tensor_index["tensors"]["down.weight"]["byte_count"] = 3 * 256 * 2
    tensor_index["tensors"]["down.weight"]["data_offsets"] = [0, 3 * 256 * 2]
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))

    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    assert (
        input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
        not in tensor_index["tensors"]
    )


def test_compiler_rejects_a_collision_with_its_reserved_tensor_name(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    reserved = input_block_major_tensor_name("down.weight", TP_INPUT_BLOCK_COLUMNS)
    tensor_index["tensors"][reserved] = {"dtype": "BF16", "shape": [1]}
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))

    with pytest.raises(ModelCompileError, match="collides with incompatible metadata"):
        derive_tensor_parallel_linear_tensors(
            {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
            lowered_dir,
            tensor_index,
            target=NativeTarget(),  # type: ignore[arg-type]
        )


def test_physical_shader_is_part_of_the_required_shader_set() -> None:
    component_executions = [
        {
            "kernels": [
                {
                    "shader_path": "shaders/canonical.comp",
                    "batch_implementations": [],
                    "physical_implementations": [
                        {"shader_path": "shaders/input_columns.comp"}
                    ],
                }
            ]
        }
    ]

    required = required_shader_files(
        component_executions,
        embedding_shader_file="embedding.comp",
        embedding_batch_shader_file="embedding_batch.comp",
        projection_shader_file="projection.comp",
        projection_batch_shader_file="projection_batch.comp",
        norm_shader_file="norm.comp",
        norm_batch_shader_file="norm_batch.comp",
        sampler_shader_files={"sampler.comp"},
    )
    assert "canonical.comp" in required
    assert "input_columns.comp" in required


def test_independent_experts_compile_selector_partition_contracts(
    tmp_path: Path,
) -> None:
    shader = tmp_path / "shaders" / "independent_sparse_moe_down_mxfp4.spv"
    shader.parent.mkdir()
    shader.write_bytes(b"independent expert shader")
    preparation_shader = tmp_path / "shaders" / "compact_routes.spv"
    preparation_shader.write_bytes(b"route compaction shader")
    batch_shader = tmp_path / "shaders" / "independent_sparse_moe_down_mxfp4_batch.spv"
    batch_shader.write_bytes(b"independent expert batch shader")
    mapping = [
        {
            "selector": expert,
            "parameter_ids": [f"expert_{expert}_weight", f"expert_{expert}_scale"],
        }
        for expert in range(4)
    ]
    node = {
        "id": "expert_down",
        "op": "independent_sparse_moe_down",
        "inputs": ["expert_intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": [
            parameter for entry in mapping for parameter in entry["parameter_ids"]
        ],
        "attrs": {
            "hidden_size": 128,
            "intermediate_size": 128,
            "experts_per_token": 2,
            "selected_parameter_accesses": [
                {
                    "selection_signal": "routes",
                    "execution_signal": "routes",
                    "execution_calibration_word_base": 0x3F800000,
                    "mapping": mapping,
                }
            ],
        },
    }
    refs = {
        parameter: {"tensor": f"tensor.{parameter}"} for parameter in node["params"]
    }
    tensor_index = {
        "tensors": {
            ref["tensor"]: {
                "dtype": "F8_E8M0" if parameter.endswith("scale") else "I8",
                "shape": [128, 4] if parameter.endswith("scale") else [128, 64],
                "layout": "row_major",
            }
            for parameter, ref in refs.items()
        }
    }
    gate_up = {
        "id": "expert_gate_up",
        "op": "independent_sparse_moe_gate_up",
        "inputs": ["normalized", "routes"],
        "outputs": ["expert_intermediates"],
        "params": [],
    }
    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit={"nodes": [gate_up, node], "parameters": {"refs": refs}},
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": ["expert_down"],
            "semantic_module_ids": ["layer.feed_forward.routed_experts"],
            "shader_path": "shaders/independent_sparse_moe_down_mxfp4.spv",
            "local_size_x": 64,
            "workgroup_count_x": 2,
            "batch_implementations": [
                {
                    "execution_domain": "decode_and_prefill",
                    "lane_tile_width": 16,
                    "stages": [
                        {
                            "shader_path": "shaders/compact_routes.spv",
                            "local_size_x": 32,
                            "workgroup_count_x": 1,
                        },
                        {
                            "shader_path": "shaders/independent_sparse_moe_down_mxfp4_batch.spv",
                            "local_size_x": 64,
                            "workgroup_count_x": 2,
                        },
                    ],
                }
            ],
            "physical_implementations": [],
        },
        package_dir=tmp_path,
    )

    distributed = next(
        contract for contract in contracts if contract["strategy"] == "expert_parallel"
    )
    assert distributed["execution_form"] == "whole_expert_ownership"
    assert distributed["partition_extent"] == {
        "dimension_name": "selected_resource_count",
        "elements": 4,
        "alignment_elements": 1,
    }
    assert distributed["parameter_partitions"] == []
    assert distributed["selected_resource_partitions"] == [
        {
            "selection_signal": "routes",
            "address_table_binding": 3,
            "parameter_slots_binding": 4,
            "kind": "expert_range",
            "resource_count": 4,
            "parameters_per_resource": 2,
            "alignment_elements": 1,
            "parameter_partitions": [],
        }
    ]
    assert distributed["inputs"] == [
        {
            "binding": 0,
            "distribution": "routed",
            "dimension": 0,
            "alignment_elements": 1,
        },
        {
            "binding": 1,
            "distribution": "routed",
            "dimension": 0,
            "alignment_elements": 1,
        },
    ]
    assert distributed["outputs"] == [
        {
            "binding": 2,
            "collection": "routed",
            "dimension": 0,
            "alignment_elements": 1,
        }
    ]
    assert distributed["local_intermediates"] == [
        {
            "signal": "expert_intermediates",
            "producer_binding": 2,
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]
    distributed_batches = [
        contract
        for contract in contracts
        if contract["strategy"] == "expert_parallel"
        and contract["execution_shape"] == "multi_lane"
    ]
    assert {tuple(contract["phases"]) for contract in distributed_batches} == {
        ("decode",),
        ("prefill",),
    }
    assert all(
        [artifact["role"] for artifact in contract["artifacts"]]
        == ["preparation", "primary"]
        for contract in distributed_batches
    )
    assert all(
        contract["geometry"]["dimensions"]["workgroup_count_x"] == 2
        for contract in distributed_batches
    )


def test_physical_expert_implementation_preserves_selected_parameter_partitions(
    tmp_path: Path,
) -> None:
    shader_dir = tmp_path / "shaders"
    shader_dir.mkdir()
    (shader_dir / "canonical.spv").write_bytes(b"canonical")
    (shader_dir / "tensor_parallel_expert.spv").write_bytes(b"physical")
    mapping = [
        {
            "selector": expert,
            "parameter_ids": [f"expert_{expert}_weight", f"expert_{expert}_scale"],
        }
        for expert in range(4)
    ]
    node = {
        "id": "expert_down",
        "op": "independent_sparse_moe_down",
        "inputs": ["expert_intermediates", "routes"],
        "outputs": ["expert_outputs"],
        "params": [
            parameter for entry in mapping for parameter in entry["parameter_ids"]
        ],
        "attrs": {
            "selected_parameter_accesses": [
                {
                    "selection_signal": "routes",
                    "execution_signal": "routes",
                    "execution_calibration_word_base": 0x3F800000,
                    "mapping": mapping,
                }
            ]
        },
    }
    refs = {
        parameter: {"tensor": f"tensor.{parameter}"} for parameter in node["params"]
    }
    tensor_index = {
        "tensors": {
            ref["tensor"]: {
                "dtype": "F8_E8M0" if parameter.endswith("scale") else "I8",
                "shape": [128, 4] if parameter.endswith("scale") else [128, 64],
                "layout": "row_major",
            }
            for parameter, ref in refs.items()
        }
    }
    selected_resource_partition = {
        "selection_signal": "routes",
        "address_table_binding": 3,
        "parameter_slots_binding": 4,
        "kind": "expert_range",
        "resource_count": 4,
        "parameters_per_resource": 2,
        "alignment_elements": 32,
        "parameter_partitions": [
            {
                "parameter_slot": slot,
                "dimension": 0,
                "kind": "contiguous",
                "alignment_elements": 32,
                "logical_elements_per_index": 1,
            }
            for slot in range(2)
        ],
    }
    implementation = {
        "shader_path": "shaders/tensor_parallel_expert.spv",
        "local_size_x": 64,
        "workgroup_count_x": 2,
        "phases": ["decode", "prefill"],
        "execution_shape": "single_and_multi_lane",
        "formats": {
            "storage": "mxfp4_e2m1",
            "compute": "mxfp4_e2m1",
            "accumulation": "f32",
        },
        "geometry_dimensions": {"hidden_size": 128, "intermediate_size": 128},
        "strategy": "tensor_parallel_expert",
        "execution_form": "partitioned_input_partial_output",
        "partition_extent": {
            "dimension_name": "intermediate_size",
            "elements": 128,
            "alignment_elements": 32,
        },
        "partition_launch": {
            "workgroup_x": "repeated",
            "origin": "push_constant_u32",
            "origin_push_constant": "input_start",
            "count_push_constant": "input_count",
        },
        "parameter_partitions": [],
        "selected_resource_partitions": [selected_resource_partition],
        "inputs": [
            {
                "binding": 0,
                "distribution": "sharded",
                "dimension": 0,
                "alignment_elements": 32,
            },
            {
                "binding": 1,
                "distribution": "routed",
                "dimension": 0,
                "alignment_elements": 1,
            },
        ],
        "outputs": [
            {
                "binding": 2,
                "collection": "reduced",
                "reduction": {
                    "operation": "sum_f32",
                    "dimension_name": "hidden_size",
                    "finalization": {"kind": "store_f32"},
                },
            }
        ],
        "local_intermediates": [],
        "resources": [
            {
                "resource": ref["tensor"],
                "kind": "lazy_resource",
                "residency": "demand",
                "access": "read",
            }
            for ref in refs.values()
        ],
        "equivalence": {
            "output": "absolute_relative_tolerance",
            "state": "bit_exact",
            "absolute_tolerance": 0.01,
            "relative_tolerance": 0.01,
        },
    }

    contracts = build_kernel_physical_execution_contracts(
        node=node,
        circuit={"nodes": [node], "parameters": {"refs": refs}},
        tensor_index=tensor_index,
        kernel={
            "source_node_ids": [node["id"]],
            "semantic_module_ids": ["layer.feed_forward.routed_experts"],
            "shader_path": "shaders/canonical.spv",
            "local_size_x": 64,
            "workgroup_count_x": 2,
            "batch_implementations": [],
            "physical_implementations": [implementation],
        },
        package_dir=tmp_path,
    )

    physical = next(
        contract
        for contract in contracts
        if contract["strategy"] == "tensor_parallel_expert"
    )
    assert physical["selected_resource_partitions"] == [selected_resource_partition]


def test_input_column_physical_shaders_render_and_compile(tmp_path: Path) -> None:
    source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_residual_input_columns_bf16_b128_256x4.comp": (
            "batch_index * TOTAL_INPUT_WORDS"
        ),
        "linear_residual_input_columns_fp8_e4m3_b2x128_256x4.comp": (
            "batch_index * (TOTAL_INPUT_SIZE / 2u)"
        ),
    }
    copy_shader_templates(source_dir, tmp_path, shader_files)
    for shader_file, full_frame_stride in shader_files.items():
        source = (tmp_path / shader_file).read_text()
        assert "PartitionControl" in source
        assert "binding = 2, std430" in source
        assert "gl_WorkGroupID.y" in source
        # Every participant owns a full-frame-strided private buffer. Its
        # descriptor starts at the shard offset, while successive lanes remain
        # separated by the complete logical activation width.
        assert full_frame_stride in source
        assert "batch_index * OUTPUT_SIZE" in source
        assert "{{" not in source

    compile_shader_artifacts(tmp_path)
    assert {path.name for path in tmp_path.glob("*.spv")} == {
        name.replace(".comp", ".spv") for name in shader_files
    }
