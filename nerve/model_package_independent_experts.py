from __future__ import annotations

import math

from nerve.model_package_common import (
    ModelCompileError,
    ROW_MAJOR_LAYOUT,
    Json,
    shader_float_token,
)
from nerve.physical_representations import (
    FP8_E8M0_PREQUANTIZATION_CONTRACT,
    FP8_PREQUANTIZATION_CONTRACT,
)


INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS = 32
INDEPENDENT_MXFP4_DOWN_TILE_ROWS = 64
INDEPENDENT_MXFP4_TP_COLUMNS = 128
INDEPENDENT_NATIVE_FP8_BLOCK = 128
MXFP4_EXPERT_FORMAT = "mxfp4_e2m1_g32"
NATIVE_FP8_EXPERT_FORMAT = "fp8_e4m3_e8m0_b128"


def independent_sparse_moe_shader_file(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> str:
    attrs = node.get("attrs", {})
    operation = str(node.get("op", ""))
    if operation not in {
        "independent_sparse_moe_gate_up",
        "independent_sparse_moe_down",
    }:
        raise ModelCompileError(
            f"node {node.get('id')!r} is not an independent sparse expert projection"
        )
    stage = "gate_up" if operation.endswith("gate_up") else "down"
    hidden_size = int(attrs.get("hidden_size", 0))
    intermediate_size = int(attrs.get("intermediate_size", 0))
    experts_per_token = int(attrs.get("experts_per_token", 0))
    accesses = attrs.get("selected_parameter_accesses")
    physical_input = attrs.get("physical_input_contract")
    prequantized_input = stage == "gate_up" and physical_input in {
        FP8_PREQUANTIZATION_CONTRACT,
        FP8_E8M0_PREQUANTIZATION_CONTRACT,
    }
    expected_input_count = 3 if prequantized_input else 2
    if (
        hidden_size <= 0
        or hidden_size % 128
        or intermediate_size <= 0
        or intermediate_size % 128
        or not isinstance(accesses, list)
        or len(accesses) != 1
        or len(node.get("inputs", [])) != expected_input_count
        or accesses[0].get("selection_signal") != node["inputs"][-1]
        or len(node.get("outputs", [])) != 1
    ):
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} has an invalid interface"
        )
    mapping = accesses[0].get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} has no expert mapping"
        )
    num_experts = len(mapping)
    parameters_per_expert = 4 if stage == "gate_up" else 2
    expected_parameters: list[str] = []
    input_block_major_layout: bool | None = None
    resource_formats: list[str] = []
    for expert, entry in enumerate(mapping):
        parameter_ids = entry.get("parameter_ids") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or int(entry.get("selector", -1)) != expert
            or not isinstance(parameter_ids, list)
            or len(parameter_ids) != parameters_per_expert
            or not all(isinstance(value, str) and value for value in parameter_ids)
        ):
            raise ModelCompileError(
                f"independent sparse expert node {node['id']!r} has a malformed "
                f"mapping for expert {expert}"
            )
        expected_parameters.extend(parameter_ids)
        matrix_pairs = (
            ((parameter_ids[0], parameter_ids[1]), (parameter_ids[2], parameter_ids[3]))
            if stage == "gate_up"
            else ((parameter_ids[0], parameter_ids[1]),)
        )
        rows = intermediate_size if stage == "gate_up" else hidden_size
        columns = hidden_size if stage == "gate_up" else intermediate_size
        matrix_formats: list[str] = []
        for weight_id, scale_id in matrix_pairs:
            matrix_format, matrix_is_input_block_major = _validate_expert_matrix(
                circuit,
                tensor_index,
                weight_id,
                scale_id,
                rows=rows,
                columns=columns,
                node_id=str(node["id"]),
            )
            matrix_formats.append(matrix_format)
            if matrix_format == MXFP4_EXPERT_FORMAT and input_block_major_layout is None:
                input_block_major_layout = matrix_is_input_block_major
            elif (
                matrix_format == MXFP4_EXPERT_FORMAT
                and input_block_major_layout != matrix_is_input_block_major
            ):
                raise ModelCompileError(
                    f"independent sparse expert node {node['id']!r} mixes "
                    "incompatible physical matrix layouts"
                )
        if len(set(matrix_formats)) != 1:
            raise ModelCompileError(
                f"independent sparse expert node {node['id']!r} resource {expert} "
                "mixes incompatible matrix representations"
            )
        resource_formats.append(matrix_formats[0])
    if node.get("params") != expected_parameters:
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} does not preserve its "
            "selector-ordered parameter mapping"
        )
    if not 0 < experts_per_token <= num_experts:
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} has invalid routing "
            f"geometry e{num_experts} k{experts_per_token}"
        )
    native_fp8_start = next(
        (
            index
            for index, resource_format in enumerate(resource_formats)
            if resource_format == NATIVE_FP8_EXPERT_FORMAT
        ),
        num_experts,
    )
    if resource_formats != [MXFP4_EXPERT_FORMAT] * native_fp8_start + [
        NATIVE_FP8_EXPERT_FORMAT
    ] * (num_experts - native_fp8_start):
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} requires one "
            "selector-ordered compact-to-native representation boundary"
        )
    if native_fp8_start == 0:
        raise ModelCompileError(
            f"independent sparse expert node {node['id']!r} has no compact "
            "resource prefix for the mixed MXFP4/native FP8 kernel"
        )
    suffix = f"h{hidden_size}_i{intermediate_size}_e{num_experts}_k{experts_per_token}"
    if stage == "gate_up":
        limit = float(attrs.get("swiglu_limit", 0.0))
        if not math.isfinite(limit) or limit < 0.0:
            raise ModelCompileError(
                f"independent sparse expert node {node['id']!r} has invalid "
                f"SwiGLU limit {limit}"
            )
        suffix += f"_limit{shader_float_token(limit)}"
    representation = "_prequant" if prequantized_input else ""
    if input_block_major_layout:
        if stage != "down":
            raise ModelCompileError(
                f"independent sparse expert node {node['id']!r} cannot use "
                "input-block-major gate/up parameters"
            )
        representation += "_input_block_major_b128"
    mixed_suffix = (
        f"_native_fp8_e4m3_se8m0_b128_nf{native_fp8_start}"
        if native_fp8_start < num_experts
        else ""
    )
    return (
        f"independent_sparse_moe_{stage}{representation}_"
        f"mxfp4_e2m1_g32{mixed_suffix}_{suffix}.comp"
    )


def _validate_expert_matrix(
    circuit: Json,
    tensor_index: Json,
    weight_id: str,
    scale_id: str,
    *,
    rows: int,
    columns: int,
    node_id: str,
) -> tuple[str, bool]:
    refs = circuit.get("parameters", {}).get("refs", {})
    weight_ref = refs.get(weight_id)
    scale_ref = refs.get(scale_id)
    tensors = tensor_index.get("tensors", {})
    weight = (
        tensors.get(weight_ref.get("tensor")) if isinstance(weight_ref, dict) else None
    )
    scale = (
        tensors.get(scale_ref.get("tensor")) if isinstance(scale_ref, dict) else None
    )
    quantization = weight.get("quantization") if isinstance(weight, dict) else None
    expected_scale_tensor = (
        scale_ref.get("tensor") if isinstance(scale_ref, dict) else None
    )
    if not isinstance(weight, dict) or not isinstance(scale, dict):
        raise ModelCompileError(
            f"independent sparse expert node {node_id!r} has missing MXFP4 "
            f"parameters {weight_id!r} and {scale_id!r}"
        )
    valid_mxfp4_quantization = {
        "format": "mxfp4_e2m1",
        "bits": 4,
        "element_type": "float",
        "values_per_byte": 2,
        "packing_axis": 1,
        "packing_order": "low_nibble_then_high_nibble_along_k",
        "group_size": 32,
        "scales": expected_scale_tensor,
        "scale_dtype": "F8_E8M0",
        "scale_mode": "power_of_two_per_output_row_k_group",
    }
    expected_weight_bytes = rows * columns // 2
    expected_scale_bytes = rows * columns // 32
    weight_layout = (
        weight.get("physical_layout") if isinstance(weight, dict) else None
    )
    scale_layout = (
        scale.get("physical_layout") if isinstance(scale, dict) else None
    )
    input_block_major = (
        weight_layout == {"kind": "input_block_major", "block_columns": 64}
        and scale_layout == {"kind": "input_block_major", "block_columns": 4}
        and weight.get("shape") == [columns // 128, rows, 64]
        and weight.get("logical_shape") == [rows, columns]
        and scale.get("shape") == [columns // 128, rows, 4]
        and scale.get("logical_shape") == [rows, columns // 32]
    )
    row_major = (
        weight_layout is None
        and scale_layout is None
        and weight.get("shape") == [rows, columns // 2]
        and weight.get("logical_shape") == [rows, columns]
        and scale.get("shape") == [rows, columns // 32]
    )
    valid_mxfp4 = not (
        weight.get("dtype") != "I8"
        or not (row_major or input_block_major)
        or weight.get("layout", ROW_MAJOR_LAYOUT) != ROW_MAJOR_LAYOUT
        or quantization != valid_mxfp4_quantization
        or scale.get("dtype") != "F8_E8M0"
        or scale.get("layout", ROW_MAJOR_LAYOUT) != ROW_MAJOR_LAYOUT
        or weight.get("byte_count") != expected_weight_bytes
        or scale.get("byte_count") != expected_scale_bytes
    )
    if valid_mxfp4:
        return MXFP4_EXPERT_FORMAT, input_block_major

    native_scale_shape = [
        (rows + INDEPENDENT_NATIVE_FP8_BLOCK - 1) // INDEPENDENT_NATIVE_FP8_BLOCK,
        (columns + INDEPENDENT_NATIVE_FP8_BLOCK - 1)
        // INDEPENDENT_NATIVE_FP8_BLOCK,
    ]
    valid_native_fp8 = (
        weight.get("dtype") == "F8_E4M3"
        and weight.get("shape") == [rows, columns]
        and weight.get("logical_shape", [rows, columns]) == [rows, columns]
        and weight.get("physical_layout") is None
        and weight.get("layout", ROW_MAJOR_LAYOUT) == ROW_MAJOR_LAYOUT
        and weight.get("byte_count") == rows * columns
        and quantization is None
        and scale.get("dtype") == "F8_E8M0"
        and scale.get("shape") == native_scale_shape
        and scale.get("physical_layout") is None
        and scale.get("layout", ROW_MAJOR_LAYOUT) == ROW_MAJOR_LAYOUT
        and scale.get("byte_count") == native_scale_shape[0] * native_scale_shape[1]
    )
    if valid_native_fp8:
        return NATIVE_FP8_EXPERT_FORMAT, False
    raise ModelCompileError(
        f"independent sparse expert node {node_id!r} has incompatible MXFP4 or "
        "native FP8 expert "
        f"matrix parameters {weight_id!r} and {scale_id!r}"
    )
