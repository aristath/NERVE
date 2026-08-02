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
        for weight_id, scale_id in matrix_pairs:
            _validate_mxfp4_matrix(
                circuit,
                tensor_index,
                weight_id,
                scale_id,
                rows=rows,
                columns=columns,
                node_id=str(node["id"]),
            )
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
    return (
        f"independent_sparse_moe_{stage}{representation}_"
        f"mxfp4_e2m1_g32_{suffix}.comp"
    )


def _validate_mxfp4_matrix(
    circuit: Json,
    tensor_index: Json,
    weight_id: str,
    scale_id: str,
    *,
    rows: int,
    columns: int,
    node_id: str,
) -> None:
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
    valid_quantization = {
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
    if (
        not isinstance(weight, dict)
        or not isinstance(scale, dict)
        or weight.get("dtype") != "I8"
        or weight.get("shape") != [rows, columns // 2]
        or weight.get("logical_shape") != [rows, columns]
        or weight.get("layout", ROW_MAJOR_LAYOUT) != ROW_MAJOR_LAYOUT
        or quantization != valid_quantization
        or scale.get("dtype") != "F8_E8M0"
        or scale.get("shape") != [rows, columns // 32]
        or scale.get("layout", ROW_MAJOR_LAYOUT) != ROW_MAJOR_LAYOUT
        or weight.get("byte_count") != expected_weight_bytes
        or scale.get("byte_count") != expected_scale_bytes
    ):
        raise ModelCompileError(
            f"independent sparse expert node {node_id!r} has incompatible MXFP4 "
            f"parameters {weight_id!r} and {scale_id!r}"
        )
