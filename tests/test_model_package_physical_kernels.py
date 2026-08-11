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
    assert distributed["equivalence"]["output"] == "absolute_relative_tolerance"


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


def test_input_column_physical_shaders_render_and_compile(tmp_path: Path) -> None:
    source_dir = Path(__file__).parents[1] / "runtime-rs" / "shaders"
    shader_files = {
        "linear_residual_input_columns_bf16_b128_256x4.comp",
        "linear_residual_input_columns_fp8_e4m3_b2x128_256x4.comp",
    }
    copy_shader_templates(source_dir, tmp_path, shader_files)
    for shader_file in shader_files:
        source = (tmp_path / shader_file).read_text()
        assert "PartitionControl" in source
        assert "binding = 2, std430" in source
        assert "{{" not in source

    compile_shader_artifacts(tmp_path)
    assert {path.name for path in tmp_path.glob("*.spv")} == {
        name.replace(".comp", ".spv") for name in shader_files
    }
