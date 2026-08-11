from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from collections.abc import Callable
from typing import Any

from nerve.physical_representations import (
    ATTENTION_PARTIALS_CONTRACT,
    physical_representation_contract,
)


Json = dict[str, Any]


def optimize_circuit_for_vulkan(
    circuit: Json,
    *,
    can_fuse_linear_split: Callable[[Json], bool] | None = None,
    can_fuse_parallel_linears: Callable[[list[Json]], bool] | None = None,
    can_fuse_parallel_linear_silu_multiply: (
        Callable[[Json, Json], bool] | None
    ) = None,
    can_fuse_parallel_head_norm_rope: (
        Callable[[list[tuple[Json, Json]]], bool] | None
    ) = None,
    can_fuse_parallel_mixed_head_norm_rope: (
        Callable[[tuple[Json, Json], tuple[Json, Json]], bool] | None
    ) = None,
    can_fuse_multiply_rolling_depthwise: (
        Callable[[Json, Json, Json], bool] | None
    ) = None,
    can_fuse_recurrent_output_gate: Callable[[Json, Json], bool] | None = None,
    can_fuse_linear_split_recurrent: Callable[[Json, Json], bool] | None = None,
    can_fuse_append_attention: Callable[[Json, Json], bool] | None = None,
    can_fuse_mixed_precision_parallel_linears: (
        Callable[[Json, Json], bool] | None
    ) = None,
    can_fuse_contiguous_linear_swiglu: (
        Callable[[Json, Json, Json], bool] | None
    ) = None,
    can_fuse_linear_sigmoid_scalar_multiply: (
        Callable[[Json, Json], bool] | None
    ) = None,
    can_fuse_hyper_connection_rms_norm: (
        Callable[[Json, Json], bool] | None
    ) = None,
    prequantization_spec: Callable[[Json], Json | None] | None = None,
    can_emit_representation: Callable[[Json, Json], bool] | None = None,
    attention_partition_count: int | None = None,
) -> Json:
    """Compile discoverable node regions without changing the component boundary."""
    optimized = deepcopy(circuit)
    nodes = _fuse_hyper_connection_pre_regions(optimized["nodes"])
    nodes = _fuse_hyper_connection_post_pre_regions(nodes)
    nodes = _fuse_parallel_head_norm_rope_regions(
        nodes, can_fuse_parallel_head_norm_rope
    )
    nodes = _fuse_parallel_mixed_head_norm_rope_regions(
        nodes, can_fuse_parallel_mixed_head_norm_rope
    )
    consumer_counts = Counter(
        signal
        for node in nodes
        for signal in node.get("inputs", [])
    )

    compiled_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        parallel_fusion = _fuse_parallel_linears(
            nodes,
            index,
            can_fuse_parallel_linears,
        )
        if parallel_fusion is not None:
            fused, consumed_node_count = parallel_fusion
            compiled_nodes.append(fused)
            index += consumed_node_count
            continue

        current = nodes[index]
        following = nodes[index + 1] if index + 1 < len(nodes) else None

        fused = _fuse_linear_split(
            current,
            following,
            consumer_counts,
            can_fuse_linear_split,
        )
        if fused is None:
            fused = _fuse_silu_multiply(current, following, consumer_counts)
        if fused is None:
            fused = _fuse_linear_residual(current, following, consumer_counts)
        if fused is None:
            fused = _fuse_linear_sigmoid_scalar_multiply(
                current,
                following,
                consumer_counts,
                can_fuse_linear_sigmoid_scalar_multiply,
            )
        if fused is not None:
            compiled_nodes.append(fused)
            index += 2
            continue

        compiled_nodes.append(deepcopy(current))
        index += 1

    compiled_nodes = _fuse_linear_scalar_gate_residual_chains(compiled_nodes)
    compiled_nodes = _fuse_multiply_rolling_depthwise_regions(
        compiled_nodes,
        can_fuse_multiply_rolling_depthwise,
    )
    compiled_nodes = _fuse_recurrent_output_gate_regions(
        compiled_nodes,
        can_fuse_recurrent_output_gate,
    )
    compiled_nodes = _fuse_parallel_linear_silu_multiply_regions(
        compiled_nodes,
        can_fuse_parallel_linear_silu_multiply,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    compiled_nodes = _fuse_linear_split_recurrent_regions(
        compiled_nodes,
        can_fuse_linear_split_recurrent,
    )
    compiled_nodes = _fuse_append_attention_regions(
        compiled_nodes,
        can_fuse_append_attention,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    compiled_nodes = _lower_partitioned_attention(
        compiled_nodes,
        attention_partition_count,
    )
    compiled_nodes = _fuse_contiguous_linear_swiglu_regions(
        compiled_nodes,
        consumer_counts,
        can_fuse_contiguous_linear_swiglu,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    lowered_nodes = _lower_prequantized_inputs(
        compiled_nodes,
        prequantization_spec,
        can_emit_representation,
    )
    transaction_nodes = _fuse_hyper_connection_rms_norm_regions(
        lowered_nodes,
        can_fuse_hyper_connection_rms_norm,
        {
            output.get("source", output["id"])
            for output in optimized.get("boundary", {}).get("outputs", [])
        },
    )
    optimized["nodes"] = _fuse_mixed_precision_parallel_linears(
        transaction_nodes,
        can_fuse_mixed_precision_parallel_linears,
    )
    return optimized


def _fuse_hyper_connection_rms_norm_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
    boundary_outputs: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    fused_nodes: list[Json] = []
    provider_rewrites: dict[str, str] = {}
    index = 0
    while index < len(nodes):
        hyper = nodes[index]
        norm = nodes[index + 1] if index + 1 < len(nodes) else None
        hyper_op = hyper.get("op")
        reduced_output_index = 0 if hyper_op == "hyper_connection_pre" else 1
        hyper_outputs = hyper.get("outputs", [])
        reduced_output = (
            hyper_outputs[reduced_output_index]
            if reduced_output_index < len(hyper_outputs)
            else None
        )
        if (
            norm is None
            or hyper_op
            not in {"hyper_connection_pre", "hyper_connection_post_pre"}
            or reduced_output is None
            or norm.get("op") != "rms_norm"
            or norm.get("inputs") != [reduced_output]
            or len(norm.get("outputs", [])) < 1
            or len(norm.get("params", [])) != 1
            or norm.get("state_reads")
            or norm.get("state_writes")
            or consumer_counts[reduced_output] != 1
            or reduced_output in boundary_outputs
            or not can_fuse(hyper, norm)
        ):
            fused_nodes.append(deepcopy(hyper))
            index += 1
            continue

        hyper_attrs = deepcopy(hyper.get("attrs", {}))
        norm_attrs = deepcopy(norm.get("attrs", {}))
        hyper_source_ids = _source_node_ids(hyper)
        norm_source_ids = _source_node_ids(norm)
        hyper_element_bytes = list(
            hyper_attrs.get("output_element_bytes", [2] * len(hyper_outputs))
        )
        norm_element_bytes = list(
            norm_attrs.get(
                "output_element_bytes", [2] * len(norm.get("outputs", []))
            )
        )
        representation_outputs = {
            output
            for representation in norm_attrs.get(
                "physical_output_representations", []
            )
            if isinstance(representation, dict)
            for output in representation.get("outputs", [])
        }
        norm_output_bytes = dict(
            zip(norm.get("outputs", []), norm_element_bytes, strict=True)
        )
        logical_norm_outputs = [
            output
            for output in norm.get("outputs", [])
            if output not in representation_outputs
        ]
        physical_norm_outputs = [
            output
            for output in norm.get("outputs", [])
            if output in representation_outputs
        ]
        if hyper_op == "hyper_connection_pre":
            outputs = [
                *deepcopy(logical_norm_outputs),
                *deepcopy(hyper_outputs[1:]),
                *deepcopy(physical_norm_outputs),
            ]
            output_element_bytes = [
                *(norm_output_bytes[output] for output in logical_norm_outputs),
                *hyper_element_bytes[1:],
                *(norm_output_bytes[output] for output in physical_norm_outputs),
            ]
        else:
            outputs = [
                hyper_outputs[0],
                *deepcopy(logical_norm_outputs),
                *deepcopy(hyper_outputs[2:]),
                *deepcopy(physical_norm_outputs),
            ]
            output_element_bytes = [
                hyper_element_bytes[0],
                *(norm_output_bytes[output] for output in logical_norm_outputs),
                *hyper_element_bytes[2:],
                *(norm_output_bytes[output] for output in physical_norm_outputs),
            ]
        attrs = {
            **hyper_attrs,
            "compiled_from": [*hyper_source_ids, *norm_source_ids],
            "rms_norm_eps": norm_attrs.get("eps"),
            "rms_norm_weight_offset": norm_attrs.get("weight_offset", 0.0),
            "rms_norm_intermediate_rounding": "BF16",
            "output_element_bytes": output_element_bytes,
        }
        if "physical_output_representations" in norm_attrs:
            attrs["physical_output_representations"] = deepcopy(
                norm_attrs["physical_output_representations"]
            )
        fused_id = f"{hyper['id']}__{norm['id']}"
        provider_rewrites[str(norm["id"])] = fused_id
        fused_nodes.append(
            {
                "id": fused_id,
                "op": f"{hyper_op}_rms_norm",
                "inputs": deepcopy(hyper.get("inputs", [])),
                "outputs": outputs,
                "params": [
                    *deepcopy(hyper.get("params", [])),
                    *deepcopy(norm["params"]),
                ],
                "attrs": attrs,
            }
        )
        index += 2
    for node in fused_nodes:
        attrs = node.get("attrs", {})
        provider_id = attrs.get("physical_input_provider_id")
        if provider_id in provider_rewrites:
            attrs["physical_input_provider_id"] = provider_rewrites[provider_id]
    return fused_nodes


def _fuse_hyper_connection_pre_regions(nodes: list[Json]) -> list[Json]:
    fused_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        function = nodes[index]
        sinkhorn = nodes[index + 1] if index + 1 < len(nodes) else None
        reduce = nodes[index + 2] if index + 2 < len(nodes) else None
        function_attrs = function.get("attrs", {})
        sinkhorn_attrs = sinkhorn.get("attrs", {}) if sinkhorn is not None else {}
        reduce_attrs = reduce.get("attrs", {}) if reduce is not None else {}
        function_inputs = function.get("inputs", [])
        function_outputs = function.get("outputs", [])
        sinkhorn_outputs = sinkhorn.get("outputs", []) if sinkhorn is not None else []
        if (
            sinkhorn is None
            or reduce is None
            or function.get("op") != "normalized_linear"
            or sinkhorn.get("op") != "hyper_connection_sinkhorn"
            or reduce.get("op") != "hyper_connection_reduce"
            or len(function_inputs) != 1
            or len(function_outputs) != 1
            or sinkhorn.get("inputs") != function_outputs
            or len(sinkhorn_outputs) != 3
            or reduce.get("inputs") != [function_inputs[0], sinkhorn_outputs[0]]
            or len(reduce.get("outputs", [])) != 1
            or len(function.get("params", [])) != 1
            or len(sinkhorn.get("params", [])) != 2
            or function.get("state_reads")
            or function.get("state_writes")
            or sinkhorn.get("state_reads")
            or sinkhorn.get("state_writes")
            or reduce.get("params")
            or reduce.get("state_reads")
            or reduce.get("state_writes")
            or function_attrs.get("normalization") != "root_mean_square"
            or int(function_attrs.get("multiplicity", 0)) <= 0
            or int(function_attrs.get("multiplicity", 0))
            != int(sinkhorn_attrs.get("multiplicity", 0))
            or int(function_attrs.get("multiplicity", 0))
            != int(reduce_attrs.get("multiplicity", 0))
            or int(sinkhorn_attrs.get("sinkhorn_iterations", 0)) <= 0
            or float(function_attrs.get("normalization_epsilon", 0.0)) <= 0.0
            or float(sinkhorn_attrs.get("epsilon", 0.0)) <= 0.0
        ):
            fused_nodes.append(deepcopy(function))
            index += 1
            continue
        fused_nodes.append(
            {
                "id": f"{function['id']}__{sinkhorn['id']}__{reduce['id']}",
                "op": "hyper_connection_pre",
                "inputs": deepcopy(function_inputs),
                "outputs": [
                    reduce["outputs"][0],
                    sinkhorn_outputs[1],
                    sinkhorn_outputs[2],
                ],
                "params": [*function["params"], *sinkhorn["params"]],
                "attrs": {
                    "compiled_from": [
                        *_source_node_ids(function),
                        *_source_node_ids(sinkhorn),
                        *_source_node_ids(reduce),
                    ],
                    "multiplicity": int(function_attrs["multiplicity"]),
                    "normalization_epsilon": float(
                        function_attrs["normalization_epsilon"]
                    ),
                    "sinkhorn_iterations": int(
                        sinkhorn_attrs["sinkhorn_iterations"]
                    ),
                    "epsilon": float(sinkhorn_attrs["epsilon"]),
                    "intermediate_rounding": "BF16",
                    "output_element_bytes": [2, 4, 4],
                },
            }
        )
        index += 3
    return fused_nodes


def _fuse_hyper_connection_post_pre_regions(nodes: list[Json]) -> list[Json]:
    fused_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        post = nodes[index]
        pre = nodes[index + 1] if index + 1 < len(nodes) else None
        post_outputs = post.get("outputs", [])
        post_attrs = post.get("attrs", {})
        pre_attrs = pre.get("attrs", {}) if pre is not None else {}
        if (
            pre is None
            or post.get("op") != "hyper_connection_post"
            or pre.get("op") != "hyper_connection_pre"
            or len(post.get("inputs", [])) != 4
            or len(post_outputs) != 1
            or pre.get("inputs") != post_outputs
            or len(pre.get("outputs", [])) != 3
            or post.get("params")
            or post.get("state_reads")
            or post.get("state_writes")
            or len(pre.get("params", [])) != 3
            or pre.get("state_reads")
            or pre.get("state_writes")
            or int(post_attrs.get("multiplicity", 0)) <= 0
            or int(post_attrs.get("multiplicity", 0))
            != int(pre_attrs.get("multiplicity", 0))
        ):
            fused_nodes.append(deepcopy(post))
            index += 1
            continue
        fused_nodes.append(
            {
                "id": f"{post['id']}__{pre['id']}",
                "op": "hyper_connection_post_pre",
                "inputs": deepcopy(post["inputs"]),
                # The next hyper-connection post consumes this residual stream,
                # so the fused kernel must preserve it as well as feeding the
                # following pre-reduction directly.
                "outputs": [post_outputs[0], *deepcopy(pre["outputs"])],
                "params": deepcopy(pre["params"]),
                "attrs": {
                    **deepcopy(pre_attrs),
                    "compiled_from": [
                        *_source_node_ids(post),
                        *_source_node_ids(pre),
                    ],
                    "post_rounding": "BF16",
                    "output_element_bytes": [2, 2, 4, 4],
                },
            }
        )
        index += 2
    return fused_nodes


def _lower_partitioned_attention(
    nodes: list[Json],
    partition_count: int | None,
) -> list[Json]:
    if partition_count is None or partition_count <= 1:
        return nodes
    compiled: list[Json] = []
    for source in nodes:
        node = deepcopy(source)
        attrs = node.get("attrs", {})
        attention = attrs.get("attention", {})
        inputs = node.get("inputs", [])
        state_reads = node.get("state_reads", [])
        if (
            node.get("op") != "append_scaled_dot_product_attention"
            or len(inputs) != 4
            or len(node.get("outputs", [])) != 1
            or len(state_reads) != 1
            or node.get("state_writes") != state_reads
            or not isinstance(attention, dict)
            or attention.get("causal") is not True
            or not isinstance(attention.get("query_heads"), int)
            or attention["query_heads"] <= 0
            or not isinstance(attention.get("key_value_heads"), int)
            or attention["key_value_heads"] <= 0
            or attention["query_heads"] % attention["key_value_heads"]
            or not isinstance(attention.get("head_width"), int)
            or attention["head_width"] <= 0
            or attention["head_width"] % 2
        ):
            compiled.append(node)
            continue

        helper_id = f"{node['id']}__partition_partials"
        partials_signal = f"{node['id']}__attention_partials_f32"
        metadata = {
            "query_heads": attention["query_heads"],
            "key_value_heads": attention["key_value_heads"],
            "head_width": attention["head_width"],
            "partition_count": partition_count,
            "scale": attention["scale"],
            "window_size": attention.get("window_size"),
            "attention_sinks": bool(attention.get("attention_sinks")),
        }
        compiled.append(
            {
                "id": helper_id,
                "op": "attention_partition_partials",
                "inputs": deepcopy(inputs),
                "outputs": [partials_signal],
                "params": [],
                "state_reads": deepcopy(state_reads),
                "state_writes": [],
                "attrs": {
                    "physical_representation_contract": (
                        ATTENTION_PARTIALS_CONTRACT
                    ),
                    "consumer_node_ids": [node["id"]],
                    "semantic_source_node_ids": _source_node_ids(node),
                    **metadata,
                    "output_element_bytes": [4],
                },
            }
        )
        node["inputs"] = [partials_signal, *deepcopy(inputs[1:])]
        node_attrs = node.setdefault("attrs", {})
        node_attrs.update(
            {
                "physical_input_contract": ATTENTION_PARTIALS_CONTRACT,
                "physical_input_provider_id": helper_id,
                "physical_input_source_node_ids": _source_node_ids(node),
                "physical_logical_inputs": deepcopy(inputs),
                "physical_passthrough_inputs": deepcopy(inputs[1:]),
                "attention_partition_count": partition_count,
                "output_element_bytes": [2],
            }
        )
        compiled.append(node)
    return compiled


def _fuse_contiguous_linear_swiglu_regions(
    nodes: list[Json],
    consumer_counts: Counter[str],
    can_fuse: Callable[[Json, Json, Json], bool] | None,
    boundary_outputs: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    fused_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        projection = nodes[index]
        split = nodes[index + 1] if index + 1 < len(nodes) else None
        activation = nodes[index + 2] if index + 2 < len(nodes) else None
        if (
            split is None
            or activation is None
            or projection.get("op") != "linear"
            or split.get("op") != "split"
            or activation.get("op") != "silu_multiply"
            or len(projection.get("inputs", [])) != 1
            or len(projection.get("outputs", [])) != 1
            or len(split.get("inputs", [])) != 1
            or len(split.get("outputs", [])) != 2
            or len(activation.get("inputs", [])) != 2
            or len(activation.get("outputs", [])) != 1
            or projection.get("state_reads")
            or projection.get("state_writes")
            or split.get("params")
            or split.get("state_reads")
            or split.get("state_writes")
            or activation.get("params")
            or activation.get("state_reads")
            or activation.get("state_writes")
        ):
            fused_nodes.append(deepcopy(projection))
            index += 1
            continue

        projection_output = projection["outputs"][0]
        split_outputs = split["outputs"]
        split_attrs = split.get("attrs", {})
        if split_attrs.get("part_widths") is not None:
            part_widths = [int(width) for width in split_attrs["part_widths"]]
        else:
            part_width = split_attrs.get("part_width")
            part_widths = [int(part_width)] * 2 if part_width is not None else []
        if (
            split["inputs"] != [projection_output]
            or activation["inputs"] != split_outputs
            or consumer_counts[projection_output] != 1
            or any(consumer_counts[signal] != 1 for signal in split_outputs)
            or projection_output in boundary_outputs
            or any(signal in boundary_outputs for signal in split_outputs)
            or split_attrs.get("layout") not in {None, "contiguous"}
            or len(part_widths) != 2
            or part_widths[0] != part_widths[1]
            or part_widths[0] <= 0
            or part_widths[0] % 2
            or int(activation.get("attrs", {}).get("element_count", 0))
            != part_widths[0]
            or not can_fuse(projection, split, activation)
        ):
            fused_nodes.append(deepcopy(projection))
            index += 1
            continue

        fused_nodes.append(
            {
                "id": "__".join(
                    (
                        projection["id"],
                        split["id"],
                        activation["id"],
                    )
                ),
                "op": "contiguous_linear_swiglu",
                "inputs": deepcopy(projection["inputs"]),
                "outputs": deepcopy(activation["outputs"]),
                "params": deepcopy(projection["params"]),
                "attrs": {
                    "compiled_from": [
                        *_source_node_ids(projection),
                        *_source_node_ids(split),
                        *_source_node_ids(activation),
                    ],
                    "part_width": part_widths[0],
                    "weight_partition": "contiguous_gate_up",
                    "intermediate_rounding": "BF16",
                },
            }
        )
        index += 3
    return fused_nodes


def _fuse_mixed_precision_parallel_linears(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    fused_nodes: list[Json] = []
    index = 0
    while index < len(nodes):
        first = nodes[index]
        second = nodes[index + 1] if index + 1 < len(nodes) else None
        first_attrs = first.get("attrs", {})
        if (
            second is None
            or first.get("op") != "parallel_linear_2way"
            or second.get("op") != "parallel_linear_2way"
            or first_attrs.get("physical_input_contract")
            != "bf16_blockwise_fp8_e4m3_f32_scale.v1"
            or first_attrs.get("physical_logical_inputs") != second.get("inputs")
            or len(first.get("inputs", [])) != 2
            or len(second.get("inputs", [])) != 1
            or len(first.get("outputs", [])) != 2
            or len(second.get("outputs", [])) != 2
            or first.get("state_reads")
            or first.get("state_writes")
            or second.get("state_reads")
            or second.get("state_writes")
            or not can_fuse(first, second)
        ):
            fused_nodes.append(deepcopy(first))
            index += 1
            continue

        fused_nodes.append(
            {
                "id": first["id"],
                "op": "mixed_parallel_linear_4way",
                "inputs": [
                    *deepcopy(first["inputs"]),
                    *deepcopy(second["inputs"]),
                ],
                "outputs": [
                    *deepcopy(first["outputs"]),
                    *deepcopy(second["outputs"]),
                ],
                "params": [
                    *deepcopy(first["params"]),
                    *deepcopy(second["params"]),
                ],
                "attrs": {
                    "compiled_from": [
                        *_source_node_ids(first),
                        *_source_node_ids(second),
                    ],
                    "branch_count": 4,
                    "branch_parameter_counts": [2, 2, 1, 1],
                    "branch_dtypes": ["F8_E4M3", "F8_E4M3", "BF16", "BF16"],
                    "output_element_bytes": [
                        *first_attrs.get("output_element_bytes", [2, 2]),
                        *second.get("attrs", {}).get(
                            "output_element_bytes", [2, 2]
                        ),
                    ],
                    "physical_input_contract": first_attrs[
                        "physical_input_contract"
                    ],
                    "physical_input_provider_id": first_attrs[
                        "physical_input_provider_id"
                    ],
                    "physical_input_source_node_ids": deepcopy(
                        first_attrs["physical_input_source_node_ids"]
                    ),
                    "physical_logical_inputs": deepcopy(
                        first_attrs["physical_logical_inputs"]
                    ),
                    "physical_passthrough_inputs": deepcopy(second["inputs"]),
                },
            }
        )
        index += 2
    return fused_nodes


def _lower_prequantized_inputs(
    nodes: list[Json],
    describe: Callable[[Json], Json | None] | None,
    can_emit: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if describe is None:
        return nodes
    prepared: list[tuple[Json, Json | None]] = []
    scopes: dict[tuple[str, str, int, int], Json] = {}
    for source in nodes:
        node = deepcopy(source)
        node_attrs = node.setdefault("attrs", {})
        node_attrs.setdefault(
            "output_element_bytes",
            [2] * len(node.get("outputs", [])),
        )
        spec = describe(node)
        prepared.append((node, spec))
        if spec is None:
            continue

        inputs = node.get("inputs", [])
        contract_id = str(spec.get("contract", ""))
        contract = physical_representation_contract(contract_id)
        input_size = int(spec["input_size"])
        block_columns = int(spec["block_columns"])
        metadata = {
            field: (
                input_size
                if field == "element_count"
                else deepcopy(spec[field])
            )
            for field in contract.metadata_fields
        }
        if (
            not inputs
            or input_size <= 0
            or block_columns <= 0
            or input_size % block_columns
        ):
            raise ValueError(
                f"node {node.get('id')!r} has an invalid prequantization description"
            )
        key = (
            contract_id,
            str(inputs[0]),
            tuple((field, metadata[field]) for field in contract.metadata_fields),
        )
        output_signals = [
            f"{node['id']}__input_{suffix}"
            for suffix in contract.output_signal_suffixes
        ]
        scope = scopes.setdefault(
            key,
            {
                "helper_id": f"{node['id']}__quantize_input",
                "contract": contract_id,
                "helper_op": contract.helper_op,
                "output_signals": output_signals,
                "output_element_bytes": list(contract.output_element_bytes),
                "consumer_node_ids": [],
                "semantic_source_node_ids": [],
                "logical_signal": key[1],
                "input_size": input_size,
                "block_columns": block_columns,
                "metadata": metadata,
            },
        )
        scope["consumer_node_ids"].append(node["id"])
        scope["semantic_source_node_ids"] = list(
            dict.fromkeys(
                [
                    *scope["semantic_source_node_ids"],
                    *_source_node_ids(node),
                ]
            )
        )

    producer_by_signal = {
        str(signal): node
        for node, _spec in prepared
        for signal in node.get("outputs", [])
    }
    producer_scopes: dict[str, list[Json]] = defaultdict(list)
    for key, scope in scopes.items():
        producer = producer_by_signal.get(key[1])
        if producer is None or can_emit is None or not can_emit(producer, scope):
            scope["provider_id"] = (
                scope["helper_id"]
                if scope["helper_op"] is not None
                else None
            )
            continue
        scope["provider_id"] = producer["id"]
        producer_scopes[producer["id"]].append(scope)

    compiled: list[Json] = []
    emitted_scopes: set[tuple[Any, ...]] = set()
    for node, spec in prepared:
        emitted = producer_scopes.get(node["id"], [])
        if emitted:
            representations = []
            for scope in emitted:
                node["outputs"].extend(scope["output_signals"])
                node["attrs"]["output_element_bytes"].extend(
                    scope["output_element_bytes"]
                )
                representations.append(
                    {
                        "contract": scope["contract"],
                        "logical_signal": scope["logical_signal"],
                        "outputs": scope["output_signals"],
                        "consumer_node_ids": scope["consumer_node_ids"],
                        **deepcopy(scope["metadata"]),
                    }
                )
            node["attrs"]["physical_output_representations"] = representations
        if spec is None:
            compiled.append(node)
            continue
        node_attrs = node.setdefault("attrs", {})
        inputs = node["inputs"]
        contract = physical_representation_contract(str(spec["contract"]))
        metadata = {
            field: (
                int(spec["input_size"])
                if field == "element_count"
                else deepcopy(spec[field])
            )
            for field in contract.metadata_fields
        }
        key = (
            str(spec["contract"]),
            str(inputs[0]),
            tuple((field, metadata[field]) for field in contract.metadata_fields),
        )
        scope = scopes[key]
        if scope["provider_id"] is None:
            compiled.append(node)
            continue
        if key not in emitted_scopes:
            if scope["provider_id"] == scope["helper_id"]:
                compiled.append(
                    {
                        "id": scope["helper_id"],
                        "op": scope["helper_op"],
                        "inputs": [inputs[0]],
                        "outputs": scope["output_signals"],
                        "attrs": {
                            "physical_representation_contract": scope["contract"],
                            "consumer_node_ids": scope["consumer_node_ids"],
                            "semantic_source_node_ids": scope[
                                "semantic_source_node_ids"
                            ],
                            **deepcopy(scope["metadata"]),
                            "output_element_bytes": scope["output_element_bytes"],
                        },
                    }
                )
            emitted_scopes.add(key)
        node["inputs"] = [
            *scope["output_signals"],
            *inputs[1:],
        ]
        node_attrs["physical_input_contract"] = scope["contract"]
        node_attrs["physical_input_provider_id"] = scope["provider_id"]
        node_attrs["physical_input_source_node_ids"] = _source_node_ids(node)
        node_attrs["physical_logical_inputs"] = inputs
        compiled.append(node)
    return compiled


def _source_node_ids(node: Json) -> list[str]:
    compiled_from = node.get("attrs", {}).get("compiled_from")
    if isinstance(compiled_from, list) and compiled_from:
        return [str(node_id) for node_id in compiled_from]
    return [str(node["id"])]


def _fuse_parallel_linear_silu_multiply_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
    protected_signals: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        projection = nodes[index]
        activation = nodes[index + 1] if index + 1 < len(nodes) else None
        consumed_node_count = 2
        projection_sources = projection.get("attrs", {}).get("compiled_from")

        if (
            projection.get("op") == "linear"
            and activation is not None
            and activation.get("op") == "linear"
            and index + 2 < len(nodes)
        ):
            second_projection = activation
            activation = nodes[index + 2]
            shared_inputs = projection.get("inputs", [])
            projection_outputs = [
                *projection.get("outputs", []),
                *second_projection.get("outputs", []),
            ]
            if (
                len(shared_inputs) == 1
                and second_projection.get("inputs") == shared_inputs
                and len(projection.get("outputs", [])) == 1
                and len(second_projection.get("outputs", [])) == 1
                and projection.get("params")
                and second_projection.get("params")
                and not projection.get("state_reads")
                and not projection.get("state_writes")
                and not second_projection.get("state_reads")
                and not second_projection.get("state_writes")
            ):
                projection = {
                    "id": f"{projection['id']}__{second_projection['id']}",
                    "op": "parallel_linear_2way",
                    "inputs": deepcopy(shared_inputs),
                    "outputs": projection_outputs,
                    "params": [
                        *projection["params"],
                        *second_projection["params"],
                    ],
                    "attrs": {
                        "compiled_from": [
                            projection["id"],
                            second_projection["id"],
                        ],
                        "branch_count": 2,
                    },
                }
                projection_sources = projection["attrs"]["compiled_from"]
                consumed_node_count = 3

        outputs = projection.get("outputs", [])
        if (
            activation is None
            or projection.get("op") != "parallel_linear_2way"
            or activation.get("op") != "silu_multiply"
            or len(projection.get("inputs", [])) != 1
            or len(outputs) != 2
            or not projection.get("params")
            or projection.get("state_reads")
            or projection.get("state_writes")
            or activation.get("inputs") != outputs
            or len(activation.get("outputs", [])) != 1
            or activation.get("params")
            or activation.get("state_reads")
            or activation.get("state_writes")
            or any(consumer_counts[output] != 1 for output in outputs)
            or any(output in protected_signals for output in outputs)
            or not can_fuse(projection, activation)
        ):
            compiled.append(deepcopy(nodes[index]))
            index += 1
            continue

        activation_sources = activation.get("attrs", {}).get("compiled_from")
        if not isinstance(projection_sources, list) or not isinstance(
            activation_sources, list
        ):
            compiled.append(deepcopy(nodes[index]))
            index += 1
            continue
        compiled.append(
            {
                "id": f"{projection['id']}__{activation['id']}",
                "op": "parallel_linear_silu_multiply",
                "inputs": deepcopy(projection["inputs"]),
                "outputs": deepcopy(activation["outputs"]),
                "params": deepcopy(projection["params"]),
                "attrs": {
                    "compiled_from": [*projection_sources, *activation_sources],
                    "branch_count": 2,
                    "intermediate_rounding": "BF16",
                    "element_count": activation.get("attrs", {}).get(
                        "element_count"
                    ),
                },
            }
        )
        index += consumed_node_count
    return compiled


def _fuse_append_attention_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
    protected_signals: set[str],
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        append = nodes[index]
        attention = nodes[index + 1] if index + 1 < len(nodes) else None
        append_outputs = append.get("outputs", [])
        if (
            attention is None
            or append.get("op") != "append_state_update"
            or attention.get("op") != "scaled_dot_product_attention"
            or len(append.get("inputs", [])) != 3
            or len(append_outputs) != 2
            or append.get("params")
            or len(append.get("state_reads", [])) != 1
            or append.get("state_reads") != append.get("state_writes")
            or len(attention.get("inputs", [])) != 3
            or attention["inputs"][1:] != append_outputs
            or len(attention.get("outputs", [])) != 1
            or attention.get("state_reads")
            or attention.get("state_writes")
            or any(consumer_counts[output] != 1 for output in append_outputs)
            or any(output in protected_signals for output in append_outputs)
            or not can_fuse(append, attention)
        ):
            compiled.append(deepcopy(append))
            index += 1
            continue

        compiled.append(
            {
                "id": f"{append['id']}__{attention['id']}",
                "op": "append_scaled_dot_product_attention",
                "inputs": [
                    attention["inputs"][0],
                    append["inputs"][0],
                    append["inputs"][1],
                    append["inputs"][2],
                ],
                "outputs": deepcopy(attention["outputs"]),
                "params": deepcopy(attention.get("params", [])),
                "state_reads": deepcopy(append["state_reads"]),
                "state_writes": deepcopy(append["state_writes"]),
                "attrs": {
                    "compiled_from": [append["id"], attention["id"]],
                    "append": deepcopy(append.get("attrs", {})),
                    "attention": deepcopy(attention.get("attrs", {})),
                    "current_kv_source": "direct_bf16_input",
                },
            }
        )
        index += 2
    return compiled


def _fuse_linear_split_recurrent_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        projection = nodes[index]
        recurrent = nodes[index + 1] if index + 1 < len(nodes) else None
        projection_outputs = projection.get("outputs", [])
        recurrent_inputs = recurrent.get("inputs", []) if recurrent is not None else []
        state_reads = recurrent.get("state_reads", []) if recurrent is not None else []
        if (
            recurrent is None
            or projection.get("op") != "linear_split_3way"
            or recurrent.get("op") != "multiply_rolling_depthwise_gate"
            or len(projection.get("inputs", [])) != 1
            or len(projection_outputs) != 3
            or len(projection.get("params", [])) != 1
            or projection.get("state_reads")
            or projection.get("state_writes")
            or len(recurrent_inputs) != 4
            or len(recurrent.get("outputs", [])) != 1
            or len(recurrent.get("params", [])) != 1
            or len(state_reads) != 1
            or recurrent.get("state_writes") != state_reads
            or recurrent_inputs[2] != state_reads[0]
            or set([recurrent_inputs[0], recurrent_inputs[1], recurrent_inputs[3]])
            != set(projection_outputs)
            or any(consumer_counts[output] != 1 for output in projection_outputs)
            or not can_fuse(projection, recurrent)
        ):
            compiled.append(deepcopy(projection))
            index += 1
            continue

        input_gate_indices = [
            projection_outputs.index(recurrent_inputs[0]),
            projection_outputs.index(recurrent_inputs[1]),
        ]
        output_gate_index = projection_outputs.index(recurrent_inputs[3])
        projection_attrs = deepcopy(projection.get("attrs", {}))
        recurrent_attrs = deepcopy(recurrent.get("attrs", {}))
        compiled.append(
            {
                "id": f"{projection['id']}__{recurrent['id']}",
                "op": "linear_split_recurrent_depthwise_gate",
                "inputs": [projection["inputs"][0], state_reads[0]],
                "outputs": deepcopy(recurrent["outputs"]),
                "params": [projection["params"][0], recurrent["params"][0]],
                "state_reads": deepcopy(state_reads),
                "state_writes": deepcopy(state_reads),
                "attrs": {
                    "compiled_from": [
                        *projection_attrs.get("compiled_from", [projection["id"]]),
                        *recurrent_attrs.get("compiled_from", [recurrent["id"]]),
                    ],
                    "projection": projection_attrs,
                    "recurrent": recurrent_attrs,
                    "input_gate_branch_indices": input_gate_indices,
                    "output_gate_branch_index": output_gate_index,
                    "projection_rounding": "BF16",
                },
            }
        )
        index += 2
    return compiled


def _fuse_recurrent_output_gate_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        recurrent = nodes[index]
        gate = nodes[index + 1] if index + 1 < len(nodes) else None
        recurrent_outputs = recurrent.get("outputs", [])
        if (
            gate is None
            or recurrent.get("op") != "multiply_rolling_depthwise"
            or gate.get("op") != "multiply"
            or len(recurrent.get("inputs", [])) != 3
            or len(recurrent_outputs) != 1
            or len(recurrent.get("params", [])) != 1
            or len(recurrent.get("state_reads", [])) != 1
            or recurrent.get("state_reads") != recurrent.get("state_writes")
            or len(gate.get("inputs", [])) != 2
            or gate["inputs"].count(recurrent_outputs[0]) != 1
            or len(gate.get("outputs", [])) != 1
            or gate.get("params")
            or gate.get("state_reads")
            or gate.get("state_writes")
            or consumer_counts[recurrent_outputs[0]] != 1
            or not can_fuse(recurrent, gate)
        ):
            compiled.append(deepcopy(recurrent))
            index += 1
            continue

        output_gate = next(
            signal for signal in gate["inputs"] if signal != recurrent_outputs[0]
        )
        attrs = deepcopy(recurrent.get("attrs", {}))
        attrs["compiled_from"] = [
            *attrs.get("compiled_from", [recurrent["id"]]),
            gate["id"],
        ]
        attrs["output_gate_rounding"] = "BF16"
        compiled.append(
            {
                "id": f"{recurrent['id']}__{gate['id']}",
                "op": "multiply_rolling_depthwise_gate",
                "inputs": [*deepcopy(recurrent["inputs"]), output_gate],
                "outputs": deepcopy(gate["outputs"]),
                "params": deepcopy(recurrent["params"]),
                "state_reads": deepcopy(recurrent["state_reads"]),
                "state_writes": deepcopy(recurrent["state_writes"]),
                "attrs": attrs,
            }
        )
        index += 2
    return compiled


def _fuse_multiply_rolling_depthwise_regions(
    nodes: list[Json],
    can_fuse: Callable[[Json, Json, Json], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        multiply = nodes[index]
        rolling = nodes[index + 1] if index + 1 < len(nodes) else None
        depthwise = nodes[index + 2] if index + 2 < len(nodes) else None
        multiply_outputs = multiply.get("outputs", [])
        rolling_outputs = rolling.get("outputs", []) if rolling is not None else []
        if (
            rolling is None
            or depthwise is None
            or multiply.get("op") != "multiply"
            or rolling.get("op") != "rolling_state_update"
            or depthwise.get("op") != "depthwise_conv1d"
            or len(multiply.get("inputs", [])) != 2
            or len(multiply_outputs) != 1
            or multiply.get("params")
            or multiply.get("state_reads")
            or multiply.get("state_writes")
            or len(rolling.get("inputs", [])) != 2
            or rolling["inputs"].count(multiply_outputs[0]) != 1
            or len(rolling_outputs) != 1
            or rolling.get("params")
            or len(rolling.get("state_reads", [])) != 1
            or len(rolling.get("state_writes", [])) != 1
            or rolling["state_reads"] != rolling["state_writes"]
            or depthwise.get("inputs") != rolling_outputs
            or len(depthwise.get("outputs", [])) != 1
            or len(depthwise.get("params", [])) != 1
            or depthwise.get("state_reads")
            or depthwise.get("state_writes")
            or consumer_counts[multiply_outputs[0]] != 1
            or consumer_counts[rolling_outputs[0]] != 1
            or not can_fuse(multiply, rolling, depthwise)
        ):
            compiled.append(deepcopy(multiply))
            index += 1
            continue

        state_input = next(
            signal for signal in rolling["inputs"] if signal != multiply_outputs[0]
        )
        compiled.append(
            {
                "id": f"{multiply['id']}__{rolling['id']}__{depthwise['id']}",
                "op": "multiply_rolling_depthwise",
                "inputs": [*deepcopy(multiply["inputs"]), state_input],
                "outputs": deepcopy(depthwise["outputs"]),
                "params": deepcopy(depthwise["params"]),
                "state_reads": deepcopy(rolling["state_reads"]),
                "state_writes": deepcopy(rolling["state_writes"]),
                "attrs": {
                    "compiled_from": [
                        multiply["id"],
                        rolling["id"],
                        depthwise["id"],
                    ],
                    "multiply": deepcopy(multiply.get("attrs", {})),
                    "rolling": deepcopy(rolling.get("attrs", {})),
                    "depthwise": deepcopy(depthwise.get("attrs", {})),
                    "intermediate_rounding": "BF16",
                },
            }
        )
        index += 3
    return compiled


def _fuse_parallel_head_norm_rope_regions(
    nodes: list[Json],
    can_fuse: Callable[[list[tuple[Json, Json]]], bool] | None,
) -> list[Json]:
    if can_fuse is None:
        return nodes

    consumers: dict[str, list[tuple[int, Json]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for signal in node.get("inputs", []):
            consumers[signal].append((index, node))

    skipped: set[int] = set()
    compiled: list[Json] = []
    for index, node in enumerate(nodes):
        if index in skipped:
            continue
        following_index = index + 1
        if following_index >= len(nodes) or following_index in skipped:
            compiled.append(deepcopy(node))
            continue

        first = _head_norm_rope_branch(nodes, index, consumers)
        second = _head_norm_rope_branch(nodes, following_index, consumers)
        branches = [first, second] if first is not None and second is not None else []
        if (
            len(branches) != 2
            or branches[0][1] == branches[1][1]
            or any(rope_index <= following_index for _, rope_index, _, _ in branches)
            or branches[1][2].get("inputs") == branches[0][2].get("outputs")
            or not can_fuse([(norm, rope) for _, _, norm, rope in branches])
        ):
            compiled.append(deepcopy(node))
            continue

        first_norm = branches[0][2]
        first_rope = branches[0][3]
        second_norm = branches[1][2]
        second_rope = branches[1][3]
        compiled.append(
            {
                "id": "__".join(
                    item["id"]
                    for item in (first_norm, first_rope, second_norm, second_rope)
                ),
                "op": "parallel_head_norm_rope_2way",
                "inputs": [first_norm["inputs"][0], second_norm["inputs"][0]],
                "outputs": [first_rope["outputs"][0], second_rope["outputs"][0]],
                "params": [first_norm["params"][0], second_norm["params"][0]],
                "attrs": {
                    "compiled_from": [
                        item["id"]
                        for item in (first_norm, first_rope, second_norm, second_rope)
                    ],
                    "branches": [
                        {
                            "norm": deepcopy(first_norm.get("attrs", {})),
                            "rope": deepcopy(first_rope.get("attrs", {})),
                        },
                        {
                            "norm": deepcopy(second_norm.get("attrs", {})),
                            "rope": deepcopy(second_rope.get("attrs", {})),
                        },
                    ],
                    "intermediate_rounding": "BF16",
                },
            }
        )
        skipped.update(
            {
                following_index,
                branches[0][1],
                branches[1][1],
            }
        )

    return compiled


def _fuse_parallel_mixed_head_norm_rope_regions(
    nodes: list[Json],
    can_fuse: Callable[[tuple[Json, Json], tuple[Json, Json]], bool] | None,
) -> list[Json]:
    """Fuse independent unscaled-query and weighted-KV norm/RoPE branches.

    The two branches need not be adjacent: their projections commonly appear
    between the query and KV normalization nodes. The fused transaction is
    emitted only after both branch inputs have been produced and only when all
    discarded intermediates have exactly one consumer.
    """
    if can_fuse is None:
        return nodes

    consumers: dict[str, list[tuple[int, Json]]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for signal in node.get("inputs", []):
            consumers[signal].append((index, node))

    branches: list[tuple[int, int, Json, Json]] = []
    for norm_index, norm in enumerate(nodes):
        norm_outputs = norm.get("outputs", [])
        if (
            norm.get("op") not in {"rms_norm", "rms_norm_per_head_unscaled"}
            or len(norm.get("inputs", [])) != 1
            or len(norm_outputs) != 1
            or norm.get("state_reads")
            or norm.get("state_writes")
        ):
            continue
        output_consumers = consumers.get(norm_outputs[0], [])
        if len(output_consumers) != 1:
            continue
        rope_index, rope = output_consumers[0]
        if (
            rope_index <= norm_index
            or rope.get("op") != "rotary_position_embedding"
            or rope.get("inputs") != norm_outputs
            or len(rope.get("outputs", [])) != 1
            or rope.get("params")
            or rope.get("state_reads")
            or rope.get("state_writes")
        ):
            continue
        branches.append((norm_index, rope_index, norm, rope))

    consumed: set[int] = set()
    insertions: dict[int, Json] = {}
    for query in branches:
        if query[0] in consumed or query[2].get("op") != "rms_norm_per_head_unscaled":
            continue
        for key_value in branches:
            if (
                key_value[0] in consumed
                or key_value[2].get("op") != "rms_norm"
                or set(query[:2]) & set(key_value[:2])
                or not can_fuse((query[2], query[3]), (key_value[2], key_value[3]))
            ):
                continue
            insertion_index = max(query[1], key_value[1])
            branch_outputs = (
                query[3]["outputs"][0],
                key_value[3]["outputs"][0],
            )
            if any(
                consumer_index <= insertion_index
                for output in branch_outputs
                for consumer_index, _consumer in consumers.get(output, [])
            ):
                continue
            attrs = {
                "compiled_from": [
                    *_source_node_ids(query[2]),
                    *_source_node_ids(query[3]),
                    *_source_node_ids(key_value[2]),
                    *_source_node_ids(key_value[3]),
                ],
                "branches": [
                    {
                        "norm_op": query[2]["op"],
                        "norm": deepcopy(query[2].get("attrs", {})),
                        "rope": deepcopy(query[3].get("attrs", {})),
                    },
                    {
                        "norm_op": key_value[2]["op"],
                        "norm": deepcopy(key_value[2].get("attrs", {})),
                        "rope": deepcopy(key_value[3].get("attrs", {})),
                    },
                ],
                "branch_parameter_counts": [
                    len(query[2].get("params", [])),
                    len(key_value[2].get("params", [])),
                ],
                "intermediate_rounding": "BF16",
                "output_element_bytes": [2, 2],
            }
            insertions[insertion_index] = {
                "id": "__".join(
                    node["id"]
                    for node in (query[2], query[3], key_value[2], key_value[3])
                ),
                "op": "parallel_mixed_head_norm_rope_2way",
                "inputs": [query[2]["inputs"][0], key_value[2]["inputs"][0]],
                "outputs": [query[3]["outputs"][0], key_value[3]["outputs"][0]],
                "params": [
                    *deepcopy(query[2].get("params", [])),
                    *deepcopy(key_value[2].get("params", [])),
                ],
                "attrs": attrs,
            }
            consumed.update({query[0], query[1], key_value[0], key_value[1]})
            break

    compiled: list[Json] = []
    for index, node in enumerate(nodes):
        if index in insertions:
            compiled.append(insertions[index])
        if index not in consumed:
            compiled.append(deepcopy(node))
    return compiled


def _head_norm_rope_branch(
    nodes: list[Json],
    norm_index: int,
    consumers: dict[str, list[tuple[int, Json]]],
) -> tuple[int, int, Json, Json] | None:
    norm = nodes[norm_index]
    if (
        norm.get("op") != "rms_norm_per_head"
        or len(norm.get("inputs", [])) != 1
        or len(norm.get("outputs", [])) != 1
        or len(norm.get("params", [])) != 1
        or norm.get("state_reads")
        or norm.get("state_writes")
    ):
        return None
    norm_output = norm["outputs"][0]
    output_consumers = consumers.get(norm_output, [])
    if len(output_consumers) != 1:
        return None
    rope_index, rope = output_consumers[0]
    if (
        rope_index <= norm_index
        or rope.get("op") != "rotary_position_embedding"
        or rope.get("inputs") != [norm_output]
        or len(rope.get("outputs", [])) != 1
        or rope.get("params")
        or rope.get("state_reads")
        or rope.get("state_writes")
    ):
        return None
    return norm_index, rope_index, norm, rope


def _fuse_parallel_linears(
    nodes: list[Json],
    start: int,
    can_fuse: Callable[[list[Json]], bool] | None,
) -> tuple[Json, int] | None:
    if can_fuse is None:
        return None
    candidates = nodes[start : start + 3]
    for count in range(min(3, len(candidates)), 1, -1):
        group = candidates[:count]
        shared_inputs = group[0].get("inputs", [])
        if (
            len(shared_inputs) != 1
            or any(
                node.get("op") != "linear"
                or node.get("inputs") != shared_inputs
                or len(node.get("outputs", [])) != 1
                or not _linear_params_are_fusible(node.get("params", []))
                or node.get("state_reads")
                or node.get("state_writes")
                for node in group
            )
            or not can_fuse(group)
        ):
            continue
        branch_parameter_counts = [len(node["params"]) for node in group]
        attrs = {
            "compiled_from": [node["id"] for node in group],
            "branch_count": count,
        }
        if any(parameter_count != 1 for parameter_count in branch_parameter_counts):
            attrs["branch_parameter_counts"] = branch_parameter_counts
        return (
            {
                "id": "__".join(node["id"] for node in group),
                "op": f"parallel_linear_{count}way",
                "inputs": deepcopy(shared_inputs),
                "outputs": [node["outputs"][0] for node in group],
                "params": [
                    parameter_id for node in group for parameter_id in node["params"]
                ],
                "attrs": attrs,
            },
            count,
        )
    return None


def _fuse_linear_split(
    linear: Json,
    split: Json | None,
    consumer_counts: Counter[str],
    can_fuse: Callable[[Json], bool] | None,
) -> Json | None:
    if (
        split is None
        or can_fuse is None
        or linear.get("op") != "linear"
        or split.get("op") != "split"
        or not can_fuse(linear)
    ):
        return None
    if (
        len(linear.get("inputs", [])) != 1
        or len(linear.get("outputs", [])) != 1
        or len(linear.get("params", [])) != 1
        or linear.get("state_reads")
        or linear.get("state_writes")
    ):
        return None
    linear_output = linear["outputs"][0]
    split_attrs = split.get("attrs", {})
    split_outputs = split.get("outputs", [])
    if (
        split.get("inputs") != [linear_output]
        or consumer_counts[linear_output] != 1
        or len(split_outputs) != 3
        or split_attrs.get("layout") not in {None, "contiguous"}
        or split.get("params")
        or split.get("state_reads")
        or split.get("state_writes")
    ):
        return None
    if split_attrs.get("part_widths") is not None:
        part_widths = [int(width) for width in split_attrs["part_widths"]]
    else:
        part_width = split_attrs.get("part_width")
        if part_width is None:
            return None
        part_widths = [int(part_width)] * 3
    if len(part_widths) != 3 or any(width <= 0 or width % 2 for width in part_widths):
        return None

    attrs = deepcopy(split_attrs)
    attrs["part_widths"] = part_widths
    attrs["compiled_from"] = [linear["id"], split["id"]]
    attrs["intermediate_rounding"] = "BF16"
    return {
        "id": f"{linear['id']}__{split['id']}",
        "op": "linear_split_3way",
        "inputs": deepcopy(linear["inputs"]),
        "outputs": deepcopy(split_outputs),
        "params": deepcopy(linear["params"]),
        "attrs": attrs,
    }


def _fuse_silu_multiply(
    activation: Json,
    multiply: Json | None,
    consumer_counts: Counter[str],
) -> Json | None:
    if multiply is None or activation.get("op") != "silu" or multiply.get("op") != "multiply":
        return None
    if not _plain_single_input_output_node(activation):
        return None
    element_count = activation.get("attrs", {}).get("element_count")
    if not isinstance(element_count, int) or element_count <= 0:
        return None
    activation_output = activation["outputs"][0]
    multiply_inputs = multiply.get("inputs", [])
    if (
        len(multiply_inputs) != 2
        or multiply_inputs.count(activation_output) != 1
        or consumer_counts[activation_output] != 1
        or multiply.get("params")
        or multiply.get("state_reads")
        or multiply.get("state_writes")
    ):
        return None

    other_input = next(signal for signal in multiply_inputs if signal != activation_output)
    return {
        "id": f"{activation['id']}__{multiply['id']}",
        "op": "silu_multiply",
        "inputs": [activation["inputs"][0], other_input],
        "outputs": deepcopy(multiply.get("outputs", [])),
        "attrs": {
            "compiled_from": [activation["id"], multiply["id"]],
            "intermediate_rounding": "BF16",
            "element_count": element_count,
        },
    }


def _fuse_linear_residual(
    linear: Json,
    residual: Json | None,
    consumer_counts: Counter[str],
) -> Json | None:
    if residual is None or linear.get("op") != "linear" or residual.get("op") != "residual_add":
        return None
    if (
        len(linear.get("inputs", [])) != 1
        or len(linear.get("outputs", [])) != 1
        or not _linear_params_are_fusible(linear.get("params", []))
        or linear.get("state_reads")
        or linear.get("state_writes")
    ):
        return None
    linear_output = linear["outputs"][0]
    residual_inputs = residual.get("inputs", [])
    if (
        len(residual_inputs) != 2
        or residual_inputs.count(linear_output) != 1
        or consumer_counts[linear_output] != 1
        or residual.get("params")
        or residual.get("state_reads")
        or residual.get("state_writes")
    ):
        return None

    residual_input = next(signal for signal in residual_inputs if signal != linear_output)
    return {
        "id": f"{linear['id']}__{residual['id']}",
        "op": "linear_residual",
        "inputs": [linear["inputs"][0], residual_input],
        "outputs": deepcopy(residual.get("outputs", [])),
        "params": deepcopy(linear["params"]),
        "attrs": {
            "compiled_from": [linear["id"], residual["id"]],
            "intermediate_rounding": "BF16",
        },
    }


def _fuse_linear_sigmoid_scalar_multiply(
    linear: Json,
    multiply: Json | None,
    consumer_counts: Counter[str],
    can_fuse: Callable[[Json, Json], bool] | None,
) -> Json | None:
    if (
        can_fuse is None
        or multiply is None
        or linear.get("op") != "linear"
        or multiply.get("op") != "sigmoid_scalar_multiply"
        or not can_fuse(linear, multiply)
    ):
        return None
    if (
        len(linear.get("inputs", [])) != 1
        or len(linear.get("outputs", [])) != 1
        or len(linear.get("params", [])) != 1
        or linear.get("state_reads")
        or linear.get("state_writes")
    ):
        return None
    linear_output = linear["outputs"][0]
    multiply_inputs = multiply.get("inputs", [])
    if (
        len(multiply_inputs) != 2
        or multiply_inputs.count(linear_output) != 1
        or consumer_counts[linear_output] != 1
        or multiply.get("params")
        or multiply.get("state_reads")
        or multiply.get("state_writes")
    ):
        return None

    value_input = next(signal for signal in multiply_inputs if signal != linear_output)
    return {
        "id": f"{linear['id']}__{multiply['id']}",
        "op": "linear_sigmoid_scalar_multiply",
        "inputs": [linear["inputs"][0], value_input],
        "outputs": deepcopy(multiply.get("outputs", [])),
        "params": deepcopy(linear["params"]),
        "attrs": {
            "compiled_from": [linear["id"], multiply["id"]],
            "intermediate_rounding": "BF16",
        },
    }


def _fuse_linear_scalar_gate_residual_chains(nodes: list[Json]) -> list[Json]:
    consumer_counts = Counter(
        signal for node in nodes for signal in node.get("inputs", [])
    )
    compiled: list[Json] = []
    index = 0
    while index < len(nodes):
        if index + 2 >= len(nodes):
            compiled.extend(deepcopy(node) for node in nodes[index:])
            break
        gate, first_add, second_add = nodes[index : index + 3]
        gate_outputs = gate.get("outputs", [])
        first_outputs = first_add.get("outputs", [])
        gate_output = gate_outputs[0] if len(gate_outputs) == 1 else None
        first_output = first_outputs[0] if len(first_outputs) == 1 else None
        first_inputs = first_add.get("inputs", [])
        second_inputs = second_add.get("inputs", [])
        if (
            gate.get("op") != "linear_sigmoid_scalar_multiply"
            or first_add.get("op") != "residual_add"
            or second_add.get("op") != "residual_add"
            or len(gate.get("inputs", [])) != 2
            or len(gate_outputs) != 1
            or len(gate.get("params", [])) != 1
            or len(first_inputs) != 2
            or len(first_outputs) != 1
            or len(second_inputs) != 2
            or len(second_add.get("outputs", [])) != 1
            or first_inputs.count(gate_output) != 1
            or second_inputs.count(first_output) != 1
            or consumer_counts[gate_output] != 1
            or consumer_counts[first_output] != 1
            or any(
                node.get("params")
                or node.get("state_reads")
                or node.get("state_writes")
                for node in (first_add, second_add)
            )
        ):
            compiled.append(deepcopy(gate))
            index += 1
            continue
        first_residual = next(signal for signal in first_inputs if signal != gate_output)
        second_residual = next(
            signal for signal in second_inputs if signal != first_output
        )
        compiled.append(
            {
                "id": f"{gate['id']}__{first_add['id']}__{second_add['id']}",
                "op": "linear_sigmoid_scalar_multiply_residual2",
                "inputs": [
                    *deepcopy(gate["inputs"]),
                    first_residual,
                    second_residual,
                ],
                "outputs": deepcopy(second_add["outputs"]),
                "params": deepcopy(gate["params"]),
                "attrs": {
                    "compiled_from": [
                        *_source_node_ids(gate),
                        first_add["id"],
                        second_add["id"],
                    ],
                    "intermediate_rounding": "BF16",
                },
            }
        )
        index += 3
    return compiled


def _plain_single_input_output_node(node: Json) -> bool:
    return (
        len(node.get("inputs", [])) == 1
        and len(node.get("outputs", [])) == 1
        and not node.get("params")
        and not node.get("state_reads")
        and not node.get("state_writes")
    )


def _linear_params_are_fusible(parameters: list[str]) -> bool:
    # Parameter identifiers are compiled-model metadata, not an execution
    # contract. A linear owns either one matrix parameter or a matrix plus
    # representation metadata; the target callback validates their concrete
    # dtypes, layouts, shapes, and relationship before fusion is accepted.
    return len(parameters) in {1, 2}
