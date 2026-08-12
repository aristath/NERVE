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
    (
        routed_resource_count,
        routed_selection_count,
        always_count,
        always_weight,
        total_resource_count,
        total_selection_count,
    ) = _routing_geometry(node)
    selection = str(attrs.get("selection", ""))
    activation = str(attrs.get("activation", ""))
    normalize_selected = bool(attrs.get("normalize_selected"))
    routed_scale = float(attrs.get("routed_scaling_factor", 0.0))
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
        f"a{always_count}w{shader_float_token(always_weight)}"
    )
    policy = f"norm{int(normalize_selected)}_scale{shader_float_token(routed_scale)}"
    if selection == "score_topk":
        _validate_selection_domain(
            node,
            total_resource_count=total_resource_count,
            total_selection_count=total_selection_count,
        )
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
            f"moe_router_score_topk_{geometry}_{policy}_bias{bias_dtype.lower()}.comp"
        )

    if selection == "preselected_resource_indices":
        if activation == "softmax":
            raise ModelCompileError(
                f"preselected MoE router node {node['id']!r} requires an "
                "independently evaluable activation"
            )
        if len(node.get("inputs", [])) != 2 or node.get("params"):
            raise ModelCompileError(
                f"preselected MoE router node {node['id']!r} requires router "
                "logits and exact resource indices without parameters"
            )
        return f"moe_router_preselected_{geometry}_{policy}.comp"

    raise ModelCompileError(
        f"independent MoE router node {node['id']!r} has unsupported selection "
        f"mode {selection!r}"
    )


def parameter_table_resource_preselection_shader_file(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> str:
    attrs = node.get("attrs", {})
    (
        routed_resource_count,
        routed_selection_count,
        always_count,
        _always_weight,
        total_resource_count,
        total_selection_count,
    ) = _routing_geometry(node)
    _validate_selection_domain(
        node,
        total_resource_count=total_resource_count,
        total_selection_count=total_selection_count,
    )
    dependency = attrs.get("predictable_dependency")
    expected_dependency = {
        "schema": "nerve.predictable_resource_selection.v1",
        "kind": "parameter_table_lookup",
        "key_signal": node["inputs"][0] if len(node.get("inputs", [])) == 1 else "",
        "table_parameter": node["params"][0]
        if len(node.get("params", [])) == 1
        else "",
        "selection_semantics": "exact",
    }
    if (
        len(node.get("inputs", [])) != 1
        or len(node.get("params", [])) != 1
        or dependency != expected_dependency
    ):
        raise ModelCompileError(
            f"resource preselection node {node['id']!r} has an invalid "
            "predictable dependency contract"
        )
    table_id = node["params"][0]
    table_shape = parameter_shape_for_id(circuit, table_id, tensor_index)
    table_dtype = parameter_dtype_for_id(circuit, table_id, tensor_index)
    if (
        len(table_shape) != 2
        or int(table_shape[0]) <= 0
        or int(table_shape[1]) != routed_selection_count
        or table_dtype not in {"I32", "I64"}
        or parameter_layout_for_id(circuit, table_id, tensor_index) != ROW_MAJOR_LAYOUT
    ):
        raise ModelCompileError(
            f"resource preselection node {node['id']!r} has an incompatible table"
        )
    return (
        f"resource_preselect_table_r{routed_resource_count}_k{routed_selection_count}_"
        f"a{always_count}_v{int(table_shape[0])}_table{table_dtype.lower()}.comp"
    )


def _routing_geometry(node: Json) -> tuple[int, int, int, float, int, int]:
    attrs = node.get("attrs", {})
    routed_resource_count = int(attrs.get("routed_resource_count", 0))
    routed_selection_count = int(attrs.get("routed_selection_count", 0))
    always_selected = attrs.get("always_selected_resources")
    if (
        not 0 < routed_selection_count <= routed_resource_count <= 4096
        or not isinstance(always_selected, list)
    ):
        raise ModelCompileError(
            f"independent MoE routing node {node['id']!r} has invalid routing "
            f"geometry r{routed_resource_count} k{routed_selection_count}"
        )
    always_weights: list[float] = []
    for offset, resource in enumerate(always_selected):
        weight = (
            float(resource.get("weight", 0.0)) if isinstance(resource, dict) else 0.0
        )
        if (
            not isinstance(resource, dict)
            or set(resource) != {"resource_index", "weight"}
            or resource.get("resource_index") != routed_resource_count + offset
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ModelCompileError(
                f"independent MoE routing node {node['id']!r} has a malformed "
                f"always-selected resource at offset {offset}"
            )
        always_weights.append(weight)
    if always_weights and len(set(always_weights)) != 1:
        raise ModelCompileError(
            f"independent MoE routing node {node['id']!r} requires one common "
            "always-selected resource weight"
        )
    always_count = len(always_selected)
    total_resource_count = routed_resource_count + always_count
    total_selection_count = routed_selection_count + always_count
    if (
        total_resource_count > 4096
        or int(attrs.get("experts_per_token", 0)) != total_selection_count
        or len(node.get("outputs", [])) != 1
    ):
        raise ModelCompileError(
            f"independent MoE routing node {node['id']!r} has inconsistent total geometry"
        )
    return (
        routed_resource_count,
        routed_selection_count,
        always_count,
        always_weights[0] if always_weights else 1.0,
        total_resource_count,
        total_selection_count,
    )


def _validate_selection_domain(
    node: Json,
    *,
    total_resource_count: int,
    total_selection_count: int,
) -> None:
    expected = {
        "id": "experts",
        "resource_count": total_resource_count,
        "selection_signal": node["outputs"][0],
        "encoding": {
            "element_type": "u32",
            "selection_count_per_activation": total_selection_count,
            "index_shift": 0,
            "index_mask": (1 << (total_resource_count - 1).bit_length()) - 1,
            "calibration_word_base": 0,
        },
    }
    if node.get("attrs", {}).get("selection_domain") != expected:
        raise ModelCompileError(
            f"independent MoE routing node {node['id']!r} has an inconsistent "
            "total selection-domain contract"
        )
