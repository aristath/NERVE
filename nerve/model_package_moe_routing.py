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
    routed_resource_count = int(attrs.get("routed_resource_count", 0))
    routed_selection_count = int(attrs.get("routed_selection_count", 0))
    always_selected = attrs.get("always_selected_resources")
    selection = str(attrs.get("selection", ""))
    activation = str(attrs.get("activation", ""))
    normalize_selected = bool(attrs.get("normalize_selected"))
    routed_scale = float(attrs.get("routed_scaling_factor", 0.0))

    if (
        not 0 < routed_selection_count <= routed_resource_count <= 4096
        or not isinstance(always_selected, list)
        or not always_selected
    ):
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} has invalid routing "
            f"geometry r{routed_resource_count} k{routed_selection_count}"
        )
    always_weights: list[float] = []
    for offset, resource in enumerate(always_selected):
        expected_index = routed_resource_count + offset
        weight = (
            float(resource.get("weight", 0.0))
            if isinstance(resource, dict)
            else 0.0
        )
        if (
            not isinstance(resource, dict)
            or set(resource) != {"resource_index", "weight"}
            or resource.get("resource_index") != expected_index
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ModelCompileError(
                f"independent MoE router node {node['id']!r} has a malformed "
                f"always-selected resource at offset {offset}"
            )
        always_weights.append(weight)
    if len(set(always_weights)) != 1:
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} requires one common "
            "always-selected resource weight"
        )
    always_count = len(always_selected)
    total_resource_count = routed_resource_count + always_count
    total_selection_count = routed_selection_count + always_count
    selection_domain = attrs.get("selection_domain")
    expected_selection_domain = {
        "id": "experts",
        "resource_count": total_resource_count,
        "selection_signal": node["outputs"][0] if len(node.get("outputs", [])) == 1 else "",
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": total_selection_count,
            "index_shift": 0,
            "index_mask": (1 << (total_resource_count - 1).bit_length()) - 1,
        },
    }
    if (
        total_resource_count > 4096
        or int(attrs.get("experts_per_token", 0)) != total_selection_count
        or selection_domain != expected_selection_domain
    ):
        raise ModelCompileError(
            f"independent MoE router node {node['id']!r} has an inconsistent "
            "total selection-domain contract"
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

    geometry = (
        f"{activation}_bf16_r{routed_resource_count}_k{routed_selection_count}_"
        f"a{always_count}w{shader_float_token(always_weights[0])}"
    )
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
            parameter_shape_for_id(circuit, bias_id, tensor_index)
            != [routed_resource_count]
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
            or int(table_shape[1]) != routed_selection_count
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
