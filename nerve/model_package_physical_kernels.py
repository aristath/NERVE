from nerve.model_package_common import *
from nerve.model_package_derived_tensors import (
    TP_INPUT_BLOCK_COLUMNS,
    input_block_major_tensor_name,
    transposed_tensor_name,
)
from nerve.model_package_tensors import tensor_dtype, tensor_shape


def local_output_shard_intermediates_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> list[Json]:
    """Describe a legal device-local handoff to the next physical kernel."""

    if node.get("op") != "parallel_linear_silu_multiply":
        return []
    outputs = node.get("outputs", [])
    nodes = circuit.get("nodes", [])
    node_id = node.get("id")
    producer_indices = [
        index
        for index, candidate in enumerate(nodes)
        if isinstance(candidate, dict) and candidate.get("id") == node_id
    ]
    if len(outputs) != 1 or len(producer_indices) != 1:
        return []
    if sum(
        candidate.get("inputs", []).count(outputs[0])
        for candidate in nodes
        if isinstance(candidate, dict)
    ) != 1:
        return []
    producer_index = producer_indices[0]
    if producer_index + 1 >= len(nodes):
        return []
    consumer = nodes[producer_index + 1]
    if (
        consumer.get("op") != "linear_residual"
        or not consumer.get("inputs")
        or consumer["inputs"][0] != outputs[0]
        or not physical_kernel_implementations_for_node(
            circuit, consumer, tensor_index
        )
    ):
        return []
    return [
        {
            "signal": outputs[0],
            "producer_binding": len(node.get("inputs", [])),
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]


def _local_input_shard_intermediates_for_node(
    circuit: Json,
    node: Json,
) -> list[Json]:
    inputs = node.get("inputs", [])
    nodes = circuit.get("nodes", [])
    node_id = node.get("id")
    consumer_indices = [
        index
        for index, candidate in enumerate(nodes)
        if isinstance(candidate, dict) and candidate.get("id") == node_id
    ]
    if len(inputs) != 2 or len(consumer_indices) != 1 or consumer_indices[0] == 0:
        return []
    producer = nodes[consumer_indices[0] - 1]
    if (
        not isinstance(producer, dict)
        or producer.get("op") != "parallel_linear_silu_multiply"
        or producer.get("outputs") != [inputs[0]]
    ):
        return []
    return [
        {
            "signal": inputs[0],
            "producer_binding": len(producer.get("inputs", [])),
            "consumer_binding": 0,
            "format": "bf16",
        }
    ]


def physical_kernel_implementations_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> list[Json]:
    """Return legal compiler-owned physical implementations for one node."""

    if (
        node.get("op") != "linear_residual"
        or len(node.get("inputs", [])) != 2
        or len(node.get("outputs", [])) != 1
    ):
        return []
    params = node.get("params", [])
    refs = circuit.get("parameters", {}).get("refs", {})
    if not params or not isinstance(refs.get(params[0]), dict):
        return []
    source_weight = refs[params[0]].get("tensor")
    if not isinstance(source_weight, str):
        return []
    source_shape = tensor_shape(tensor_index, source_weight)
    if (
        len(source_shape) != 2
        or any(dimension <= 0 for dimension in source_shape)
        or source_shape[1] % TP_INPUT_BLOCK_COLUMNS
        or source_shape[0] % 2
    ):
        return []
    output_rows, input_columns = source_shape
    weight = input_block_major_tensor_name(
        source_weight, TP_INPUT_BLOCK_COLUMNS
    )
    if weight not in tensor_index["tensors"]:
        return []
    weight_dtype = tensor_dtype(tensor_index, source_weight)
    parameter_partitions = [
        {
            "binding": 3,
            "resource": weight,
            "dimension": 0,
            "kind": "contiguous",
            "alignment_elements": 1,
            "logical_elements_per_index": TP_INPUT_BLOCK_COLUMNS,
        }
    ]
    resources = [
        {
            "resource": weight,
            "kind": "persistent_parameter",
            "residency": "permanent",
            "access": "read",
            "binding": 3,
        }
    ]
    if weight_dtype == "BF16" and len(params) == 1:
        shader_file = (
            "linear_residual_input_columns_bf16_"
            f"b{TP_INPUT_BLOCK_COLUMNS}_{input_columns}x{output_rows}.comp"
        )
        local_size_x = 64
        output_tile_rows = 2
        storage = "bf16:input_block_major"
        compute = "bf16"
    elif weight_dtype == "F8_E4M3" and len(params) == 2:
        scale_ref = refs.get(params[1])
        source_scale = (
            scale_ref.get("tensor") if isinstance(scale_ref, dict) else None
        )
        if (
            not isinstance(source_scale, str)
            or tensor_dtype(tensor_index, source_scale) != "BF16"
        ):
            return []
        scale_shape = tensor_shape(tensor_index, source_scale)
        if (
            len(scale_shape) != 2
            or any(dimension <= 0 for dimension in scale_shape)
            or scale_shape[0] % 2
            or scale_shape[1] != input_columns // TP_INPUT_BLOCK_COLUMNS
            or output_rows % scale_shape[0]
        ):
            return []
        block_rows = output_rows // scale_shape[0]
        scale = transposed_tensor_name(source_scale)
        if scale not in tensor_index["tensors"]:
            return []
        parameter_partitions.append(
            {
                "binding": 4,
                "resource": scale,
                "dimension": 0,
                "kind": "contiguous",
                "alignment_elements": 1,
                "logical_elements_per_index": TP_INPUT_BLOCK_COLUMNS,
            }
        )
        resources.append(
            {
                "resource": scale,
                "kind": "persistent_parameter",
                "residency": "permanent",
                "access": "read",
                "binding": 4,
            }
        )
        shader_file = (
            "linear_residual_input_columns_fp8_e4m3_"
            f"b{block_rows}x{TP_INPUT_BLOCK_COLUMNS}_"
            f"{input_columns}x{output_rows}.comp"
        )
        local_size_x = 1024
        output_tile_rows = FP8_LINEAR_TILE_ROWS[-1]
        storage = "f8_e4m3+bf16:input_block_major"
        compute = "fp8_e4m3"
    else:
        return []

    return [
        {
            "shader_path": f"shaders/{shader_file}",
            "local_size_x": local_size_x,
            "workgroup_count_x": (output_rows + output_tile_rows - 1)
            // output_tile_rows,
            "phases": ["decode"],
            "formats": {
                "storage": storage,
                "compute": compute,
                "accumulation": "f32",
            },
            "geometry_dimensions": {
                "input_columns": input_columns,
                "output_rows": output_rows,
            },
            "strategy": "tensor_parallel",
            "execution_form": "partitioned_input_partial_output",
            "partition_extent": {
                "dimension_name": "input_columns",
                "elements": input_columns,
                "alignment_elements": TP_INPUT_BLOCK_COLUMNS,
            },
            "partition_launch": {
                "workgroup_x": "repeated",
                "origin": "push_constant_u32",
                "origin_push_constant": "input_start",
                "count_push_constant": "input_count",
            },
            "parameter_partitions": parameter_partitions,
            "inputs": [
                {
                    "binding": 0,
                    "distribution": "sharded",
                    "dimension": 0,
                    "alignment_elements": TP_INPUT_BLOCK_COLUMNS,
                },
                {"binding": 1, "distribution": "replicated"},
            ],
            "outputs": [
                {
                    "binding": 2,
                    "collection": "reduced",
                    "reduction": {
                        "operation": "sum_f32",
                        "dimension_name": "output_rows",
                        "finalization": {
                            "kind": "add_bf16_residual_to_bf16",
                            "residual_binding": 1,
                        },
                    },
                }
            ],
            "local_intermediates": _local_input_shard_intermediates_for_node(
                circuit, node
            ),
            "resources": resources,
            "equivalence": {
                "output": "absolute_relative_tolerance",
                "state": "bit_exact",
                "absolute_tolerance": 0.01,
                "relative_tolerance": 0.01,
            },
        }
    ]
