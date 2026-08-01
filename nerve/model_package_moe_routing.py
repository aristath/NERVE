from __future__ import annotations

import math

from nerve.model_package_common import (
    ModelCompileError,
    ROW_MAJOR_LAYOUT,
    Json,
    shader_float_token,
)
from nerve.model_package_tensors import (
    parameter_dtype_for_id,
    parameter_layout_for_id,
    parameter_shape_for_id,
)


def independent_moe_route_shader_file(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> str:
    attrs = node.get("attrs", {})
    num_experts = int(attrs.get("num_experts", 0))
    experts_per_token = int(attrs.get("experts_per_token", 0))
    selection = str(attrs.get("selection", ""))
    activation = str(attrs.get("activation", ""))
    normalize_selected = bool(attrs.get("normalize_selected"))
    routed_scale = float(attrs.get("routed_scaling_factor", 0.0))

    if not 0 < experts_per_token <= num_experts <= 4096:
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} has invalid routing "
            f"geometry e{num_experts} k{experts_per_token}"
        )
    if activation not in {"sigmoid", "softmax", "sqrtsoftplus"}:
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} has unsupported "
            f"activation {activation!r}"
        )
    if not math.isfinite(routed_scale) or routed_scale <= 0.0:
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} has invalid routed "
            f"scale {routed_scale}"
        )
    if len(node.get("outputs", [])) != 1:
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} must have one route output"
        )

    geometry = f"{activation}_bf16_e{num_experts}_k{experts_per_token}"
    policy = (
        f"norm{int(normalize_selected)}_scale{shader_float_token(routed_scale)}"
    )
    if selection == "score_topk":
        if len(node.get("inputs", [])) != 1 or len(node.get("params", [])) != 1:
            raise ModelCompileError(
                f"score-selected MoE router node {node['id']!r} requires router "
                "logits and one selection-bias parameter"
            )
        bias_id = node["params"][0]
        bias_dtype = parameter_dtype_for_id(circuit, bias_id, tensor_index)
        if (
            parameter_shape_for_id(circuit, bias_id, tensor_index) != [num_experts]
            or bias_dtype not in {"F32", "BF16"}
            or parameter_layout_for_id(circuit, bias_id, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"score-selected MoE router node {node['id']!r} has an "
                "incompatible selection bias"
            )
        return (
            f"moe_router_score_topk_{geometry}_{policy}_"
            f"bias{bias_dtype.lower()}.comp"
        )

    if selection == "token_id_table":
        if activation == "softmax":
            raise ModelCompileError(
                f"token-table MoE router node {node['id']!r} requires an "
                "independently evaluable activation"
            )
        if len(node.get("inputs", [])) != 2 or len(node.get("params", [])) != 1:
            raise ModelCompileError(
                f"token-table MoE router node {node['id']!r} requires router "
                "logits, a token ID, and one route-table parameter"
            )
        table_id = node["params"][0]
        table_shape = parameter_shape_for_id(circuit, table_id, tensor_index)
        table_dtype = parameter_dtype_for_id(circuit, table_id, tensor_index)
        if (
            len(table_shape) != 2
            or int(table_shape[0]) <= 0
            or int(table_shape[1]) != experts_per_token
            or table_dtype not in {"I32", "I64"}
            or parameter_layout_for_id(circuit, table_id, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"token-table MoE router node {node['id']!r} has an incompatible "
                "route table"
            )
        return (
            f"moe_router_token_table_{geometry}_v{int(table_shape[0])}_"
            f"{policy}_table{table_dtype.lower()}.comp"
        )

    raise ModelCompileError(
        f"independent MoE router node {node['id']!r} has unsupported selection "
        f"mode {selection!r}"
    )
