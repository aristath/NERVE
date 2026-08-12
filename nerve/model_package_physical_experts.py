from __future__ import annotations

from copy import deepcopy
from typing import Any

from nerve.compilation import Json
from nerve.model_package_common import shader_float_token
from nerve.model_package_independent_experts import (
    INDEPENDENT_MXFP4_DOWN_TILE_ROWS,
    INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS,
    INDEPENDENT_MXFP4_TP_COLUMNS,
    independent_sparse_moe_shader_file,
)


def tensor_parallel_independent_expert_pair(
    circuit: Json,
    node: Json,
) -> tuple[Json, Json] | None:
    """Return an adjacent, exclusive gate/up -> down expert dataflow pair."""

    nodes = circuit.get("nodes", [])
    node_id = node.get("id")
    matches = [
        index
        for index, candidate in enumerate(nodes)
        if isinstance(candidate, dict) and candidate.get("id") == node_id
    ]
    if len(matches) != 1:
        return None
    index = matches[0]
    if node.get("op") == "independent_sparse_moe_gate_up":
        if index + 1 >= len(nodes):
            return None
        gate_up, down = node, nodes[index + 1]
    elif node.get("op") == "independent_sparse_moe_down":
        if index == 0:
            return None
        gate_up, down = nodes[index - 1], node
    else:
        return None
    if (
        not isinstance(gate_up, dict)
        or not isinstance(down, dict)
        or gate_up.get("op") != "independent_sparse_moe_gate_up"
        or down.get("op") != "independent_sparse_moe_down"
    ):
        return None
    gate_outputs = gate_up.get("outputs", [])
    down_inputs = down.get("inputs", [])
    if (
        len(gate_outputs) != 1
        or len(down_inputs) < 2
        or down_inputs[0] != gate_outputs[0]
        or gate_up.get("inputs", [])[-1:] != down_inputs[-1:]
        or sum(
            candidate.get("inputs", []).count(gate_outputs[0])
            for candidate in nodes
            if isinstance(candidate, dict)
        )
        != 1
    ):
        return None
    gate_attrs = gate_up.get("attrs", {})
    down_attrs = down.get("attrs", {})
    geometry = (
        int(gate_attrs.get("hidden_size", 0)),
        int(gate_attrs.get("intermediate_size", 0)),
        int(gate_attrs.get("experts_per_token", 0)),
    )
    if geometry != (
        int(down_attrs.get("hidden_size", 0)),
        int(down_attrs.get("intermediate_size", 0)),
        int(down_attrs.get("experts_per_token", 0)),
    ):
        return None
    hidden_size, intermediate_size, experts_per_token = geometry
    if (
        hidden_size <= 0
        or hidden_size % INDEPENDENT_MXFP4_DOWN_TILE_ROWS
        or intermediate_size <= 0
        or intermediate_size % INDEPENDENT_MXFP4_TP_COLUMNS
        or experts_per_token <= 0
    ):
        return None
    gate_mapping = _selected_mapping(gate_up)
    down_mapping = _selected_mapping(down)
    if (
        gate_mapping is None
        or down_mapping is None
        or [entry.get("selector") for entry in gate_mapping]
        != [entry.get("selector") for entry in down_mapping]
    ):
        return None
    return gate_up, down


def independent_expert_physical_implementations(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    *,
    local_intermediates: list[Json],
) -> list[Json]:
    pair = tensor_parallel_independent_expert_pair(circuit, node)
    if pair is None or not local_intermediates:
        return []
    gate_up, down = pair
    source_shader_files = (
        independent_sparse_moe_shader_file(circuit, gate_up, tensor_index),
        independent_sparse_moe_shader_file(circuit, down, tensor_index),
    )
    if any("_native_fp8_e4m3_se8m0_b128_" in path for path in source_shader_files):
        # The current intra-expert partition ABI describes one physical matrix
        # geometry for every selected resource. A heterogeneous selector bank
        # remains eligible for whole-expert parallelism, but must not advertise
        # an invalid homogeneous tensor-parallel implementation.
        return []
    attrs = node["attrs"]
    hidden_size = int(attrs["hidden_size"])
    intermediate_size = int(attrs["intermediate_size"])
    experts_per_token = int(attrs["experts_per_token"])
    mapping = _selected_mapping(node)
    assert mapping is not None
    resource_count = len(mapping)
    input_count = len(node["inputs"])
    output_binding = input_count
    dynamic_binding_base = output_binding + 1
    stage = (
        "gate_up"
        if node.get("op") == "independent_sparse_moe_gate_up"
        else "down"
    )
    parameters_per_resource = 4 if stage == "gate_up" else 2
    refs = circuit["parameters"]["refs"]
    resources = [
        {
            "resource": refs[parameter_id]["tensor"],
            "kind": "lazy_resource",
            "residency": "demand",
            "access": "read",
        }
        for entry in mapping
        for parameter_id in entry["parameter_ids"]
    ]
    selected_partition = {
        "selection_signal": node["attrs"]["selected_parameter_accesses"][0][
            "selection_signal"
        ],
        "address_table_binding": dynamic_binding_base,
        "parameter_slots_binding": dynamic_binding_base + 1,
        "kind": "expert_range",
        "resource_count": resource_count,
        "parameters_per_resource": parameters_per_resource,
        "alignment_elements": INDEPENDENT_MXFP4_TP_COLUMNS,
        "parameter_partitions": [
            {
                "parameter_slot": slot,
                "dimension": 0,
                "kind": "contiguous",
                "alignment_elements": (
                    INDEPENDENT_MXFP4_TP_COLUMNS if stage == "gate_up" else 1
                ),
                "logical_elements_per_index": (
                    1 if stage == "gate_up" else INDEPENDENT_MXFP4_TP_COLUMNS
                ),
            }
            for slot in range(parameters_per_resource)
        ],
    }
    physical_intermediates = deepcopy(local_intermediates)
    for intermediate in physical_intermediates:
        intermediate["format"] = "bf16:route_major_local_rows"
    common: Json = {
        "local_size_x": 512,
        "phases": ["decode"],
        "execution_shape": "single_lane",
        "formats": {
            "storage": "mxfp4_e2m1+f8_e8m0",
            "compute": "fp8_e4m3",
            "accumulation": "f32",
        },
        "geometry_dimensions": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "experts_per_token": experts_per_token,
            "expert_output_elements": hidden_size * experts_per_token,
        },
        "strategy": "tensor_parallel_expert",
        "partition_extent": {
            "dimension_name": "intermediate_size",
            "elements": intermediate_size,
            "alignment_elements": INDEPENDENT_MXFP4_TP_COLUMNS,
        },
        "parameter_partitions": [],
        "selected_resource_partitions": [selected_partition],
        "local_intermediates": physical_intermediates,
        "resources": resources,
        "equivalence": {
            "output": "absolute_relative_tolerance",
            "state": "bit_exact",
            "absolute_tolerance": 0.01,
            "relative_tolerance": 0.01,
        },
    }
    if stage == "gate_up":
        prequantized = len(node["inputs"]) == 3
        shader_file = (
            "independent_sparse_moe_gate_up_tensor_parallel"
            f"{'_prequant' if prequantized else ''}_mxfp4_e2m1_g32_"
            f"h{hidden_size}_i{intermediate_size}_e{resource_count}_"
            f"k{experts_per_token}_limit"
            f"{shader_float_token(float(attrs.get('swiglu_limit', 0.0)))}.comp"
        )
        return [
            {
                **common,
                "shader_path": f"shaders/{shader_file}",
                "workgroup_count_x": (
                    intermediate_size // INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS
                ),
                "execution_form": "replicated_input_partitioned_output",
                "partition_launch": {
                    "workgroup_x": "proportional",
                    "origin": "local_zero",
                },
                "inputs": [
                    *[
                        {"binding": binding, "distribution": "replicated"}
                        for binding in range(input_count - 1)
                    ],
                    {
                        "binding": input_count - 1,
                        "distribution": "routed",
                        "dimension": 0,
                        "alignment_elements": 1,
                    },
                ],
                "outputs": [
                    {
                        "binding": output_binding,
                        "collection": "concatenated",
                        "dimension": 0,
                        "alignment_elements": INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS,
                    }
                ],
            }
        ]
    shader_file = (
        "independent_sparse_moe_down_tensor_parallel_input_block_major_b128_"
        f"mxfp4_e2m1_g32_h{hidden_size}_i{intermediate_size}_"
        f"e{resource_count}_k{experts_per_token}.comp"
    )
    return [
        {
            **common,
            "shader_path": f"shaders/{shader_file}",
            "workgroup_count_x": hidden_size // INDEPENDENT_MXFP4_DOWN_TILE_ROWS,
            "execution_form": "partitioned_input_partial_output",
            "partition_launch": {
                "workgroup_x": "repeated",
                "origin": "push_constant_u32",
                "origin_push_constant": "input_start",
                "count_push_constant": "input_count",
            },
            "inputs": [
                {
                    "binding": 0,
                    "distribution": "sharded",
                    "dimension": 0,
                    "alignment_elements": INDEPENDENT_MXFP4_TP_COLUMNS,
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
                    "binding": output_binding,
                    "collection": "reduced",
                    "reduction": {
                        "operation": "sum_f32",
                        "dimension_name": "expert_output_elements",
                        "finalization": {"kind": "store_f32_to_bf16"},
                    },
                }
            ],
        }
    ]


def _selected_mapping(node: Json) -> list[Json] | None:
    accesses = node.get("attrs", {}).get("selected_parameter_accesses")
    if not isinstance(accesses, list) or len(accesses) != 1:
        return None
    mapping: Any = accesses[0].get("mapping")
    if not isinstance(mapping, list) or not mapping:
        return None
    return mapping
