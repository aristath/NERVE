from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from nerve.model_package_assets import copy_tensor_package
from nerve.model_package_common import ModelCompileError
from nerve.model_package_derived_tensors import (
    TP_INPUT_BLOCK_COLUMNS,
    derive_tensor_parallel_linear_tensors,
    input_block_major_tensor_name,
    transposed_tensor_name,
)
from nerve.model_package_physical_kernels import (
    local_shard_intermediates_for_node,
    physical_kernel_implementations_for_node,
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

    name = input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    )
    info = tensor_index["tensors"][name]
    assert info["shape"] == [2, 4, 128]
    assert info["logical_shape"] == [4, 256]
    assert info["physical_execution_only"] is True

    destination = tmp_path / "input-block-major.safetensors"
    header_bytes, _ = write_compiled_derived_matrix_reorder(
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
    assert physical_kernel_implementations_for_node(
        circuit, down, tensor_index
    )[0]["local_intermediates"] == [
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
        local_shard_intermediates_for_node(
            circuit, compiled_gate_up, tensor_index
        )
        == []
    )
    circuit["nodes"].pop()

    down["inputs"][0] = "another-signal"
    assert (
        local_shard_intermediates_for_node(circuit, gate_up, tensor_index)
        == []
    )


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
    weight_values = np.arange(4 * 256, dtype="u1").reshape(4, 256)
    scale_values = np.arange(4, dtype="<u2").reshape(2, 2)
    tensor_index = {"tensors": {}}
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
    lowered_dir = tmp_path / "lowered"
    lowered_dir.mkdir()
    (lowered_dir / "circuit.json").write_text(json.dumps(circuit))
    derive_tensor_parallel_linear_tensors(
        {"graph": {"circuits": [{"circuit": "circuit.json"}]}},
        lowered_dir,
        tensor_index,
        target=NativeTarget(),  # type: ignore[arg-type]
    )

    weight = input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    )
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
    scale_header_bytes, _ = write_compiled_derived_matrix_reorder(
        tensor_name=scale,
        info=tensor_index["tensors"][scale],
        destination=scale_destination,
        layout="row_major",
    )
    actual_scale = np.frombuffer(
        scale_destination.read_bytes()[8 + scale_header_bytes :], dtype="<u2"
    ).reshape(2, 2)
    np.testing.assert_array_equal(actual_scale, scale_values.T)


def test_partitioned_packaging_rejects_a_derived_matrix_reorder(
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

    with pytest.raises(
        ModelCompileError,
        match="matrix reorder that does not preserve independently verifiable partitions",
    ):
        copy_tensor_package(
            tensor_index,
            tmp_path / "package",
            partition_counts={physical_weight: 2},
        )


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

    assert input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    ) not in tensor_index["tensors"]


def test_compiler_rejects_a_collision_with_its_reserved_tensor_name(
    tmp_path: Path,
) -> None:
    _, circuit, tensor_index, _ = bf16_down_fixture(tmp_path)
    reserved = input_block_major_tensor_name(
        "down.weight", TP_INPUT_BLOCK_COLUMNS
    )
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
            parameter
            for entry in mapping
            for parameter in entry["parameter_ids"]
        ],
        "attrs": {
            "hidden_size": 128,
            "intermediate_size": 128,
            "experts_per_token": 2,
            "selected_parameter_accesses": [
                {"selection_signal": "routes", "mapping": mapping}
            ],
        },
    }
    refs = {
        parameter: {"tensor": f"tensor.{parameter}"}
        for parameter in node["params"]
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
        contract
        for contract in contracts
        if contract["strategy"] == "expert_parallel"
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
