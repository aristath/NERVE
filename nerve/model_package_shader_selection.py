from nerve.model_package_common import *
from nerve.model_package_assets import stream_control_binding_for_node
from nerve.model_package_independent_experts import (
    INDEPENDENT_MXFP4_DOWN_TILE_ROWS,
    INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS,
    independent_sparse_moe_shader_file,
)
from nerve.model_package_moe_routing import independent_moe_route_shader_file
from nerve.model_package_latent_compression import latent_compression_shader_file
from nerve.model_package_tensors import *


def feed_forward_intermediate_size(circuit: Json) -> int:
    for node in circuit.get("nodes", []):
        if node.get("id") == "ffn_gate_activation":
            width = int(node.get("attrs", {}).get("element_count", 0))
            if width > 0:
                return width
    raise ModelCompileError(
        f"circuit {circuit.get('id')!r} does not describe its feed-forward width"
    )


def shader_file_for_node(
    circuit: Json,
    node: Json,
    tensor_index: Json,
    dimensions: Json,
) -> str:
    hidden_size = int(dimensions["hidden_size"])
    op = node["op"]

    if op in {
        "learned_gated_kv_pool",
        "compressed_kv_finalize",
        "conditional_append_state_update",
        "index_vector_transform",
        "compressed_index_kv_finalize",
        "learned_index_scores",
        "radix_topk_index",
        "chronological_compressed_index",
    }:
        return latent_compression_shader_file(circuit, node, tensor_index)

    if op in {"hyper_connection_pre", "hyper_connection_post_pre"}:
        (
            multiplicity,
            sinkhorn_iterations,
            normalization_epsilon,
            sinkhorn_epsilon,
        ) = hyper_connection_geometry_for_node(
            circuit,
            node,
            tensor_index,
            hidden_size=hidden_size,
        )
        return (
            f"{op}_m{multiplicity}_h{hidden_size}_i{sinkhorn_iterations}_"
            f"neps{shader_float_token(normalization_epsilon)}_"
            f"heps{shader_float_token(sinkhorn_epsilon)}.comp"
        )
    if op == "hyper_connection_post":
        attrs = node.get("attrs", {})
        multiplicity = int(attrs.get("multiplicity", 0))
        if (
            multiplicity <= 0
            or hidden_size <= 0
            or hidden_size % 2
            or len(node.get("inputs", [])) != 4
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or attrs.get("output_element_bytes") != [2]
        ):
            raise ModelCompileError(
                f"hyper-connection post node {node['id']!r} has an invalid contract"
            )
        return f"hyper_connection_post_m{multiplicity}_h{hidden_size}.comp"

    if op == "anchor_noise_embedding_block":
        attrs = node.get("attrs", {})
        block_size = int(attrs.get("block_size", 0))
        minimum_block_size = int(attrs.get("minimum_block_size", 0))
        multiplicity = int(attrs.get("stream_multiplicity", 0))
        node_hidden_size = int(attrs.get("hidden_size", 0))
        noise_token_id = int(attrs.get("noise_token_id", -1))
        embedding_shape = parameter_shape_for_node(circuit, node, tensor_index)
        if (
            node.get("inputs") != ["anchor_token_id"]
            or node.get("outputs") != ["query_frames"]
            or len(node.get("params", [])) != 1
            or attrs.get("output_layout") != "block_stream_hidden"
            or attrs.get("anchor_position") != 0
            or attrs.get("runtime_extensible") is not False
            or attrs.get("runtime_selectable_prefix") is not True
            or block_size < minimum_block_size
            or minimum_block_size <= 0
            or multiplicity <= 0
            or node_hidden_size != hidden_size
            or hidden_size <= 0
            or hidden_size % 2
            or len(embedding_shape) != 2
            or int(embedding_shape[1]) != hidden_size
            or noise_token_id < 0
            or noise_token_id >= int(embedding_shape[0])
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"anchor/noise embedding node {node['id']!r} has an invalid contract"
            )
        return (
            f"anchor_noise_embedding_block_b{block_size}_m{multiplicity}_"
            f"h{hidden_size}_noise{noise_token_id}.comp"
        )

    if op == "repeat_stream_lanes":
        attrs = node.get("attrs", {})
        multiplicity = int(attrs.get("multiplicity", 0))
        node_hidden_size = int(attrs.get("hidden_size", 0))
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or multiplicity <= 0
            or node_hidden_size != hidden_size
            or hidden_size <= 0
            or hidden_size % 2
            or attrs.get("input_shape") != [hidden_size]
            or attrs.get("output_shape") != [multiplicity, hidden_size]
        ):
            raise ModelCompileError(
                f"stream-repeat node {node['id']!r} has an invalid contract"
            )
        return f"repeat_stream_lanes_bf16_m{multiplicity}_h{hidden_size}.comp"

    if op == "mean_stream_lanes":
        attrs = node.get("attrs", {})
        multiplicity = int(attrs.get("multiplicity", 0))
        node_hidden_size = int(attrs.get("hidden_size", 0))
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or multiplicity <= 0
            or node_hidden_size != hidden_size
            or hidden_size <= 0
            or hidden_size % 2
            or attrs.get("input_shape") != [multiplicity, hidden_size]
            or attrs.get("output_shape") != [hidden_size]
            or attrs.get("output_element_bytes") != [2]
        ):
            raise ModelCompileError(
                f"stream-mean node {node['id']!r} has an invalid contract"
            )
        return f"mean_stream_lanes_bf16_m{multiplicity}_h{hidden_size}.comp"

    if op == "sinkhorn_hyper_connection_head":
        attrs = node.get("attrs", {})
        block_width = int(attrs.get("block_width", 0))
        multiplicity = int(attrs.get("multiplicity", 0))
        node_hidden_size = int(attrs.get("hidden_size", 0))
        epsilon = float(attrs.get("epsilon", 0.0))
        parameter_shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node.get("params", [])
        ]
        parameter_dtypes = [
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node.get("params", [])
        ]
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or node.get("params") != ["head_function", "head_scale", "head_base"]
            or block_width <= 0
            or multiplicity <= 0
            or node_hidden_size != hidden_size
            or hidden_size <= 0
            or hidden_size % 2
            or not math.isfinite(epsilon)
            or epsilon <= 0.0
            or parameter_shapes
            != [
                [multiplicity, multiplicity * hidden_size],
                [1],
                [multiplicity],
            ]
            or parameter_dtypes != ["F32", "F32", "F32"]
            or attrs.get("output_element_bytes") != [2]
        ):
            raise ModelCompileError(
                f"hyper head node {node['id']!r} has an invalid block contract"
            )
        return (
            f"hyper_head_block_b{block_width}_m{multiplicity}_h{hidden_size}_"
            f"eps{shader_float_token(epsilon)}.comp"
        )

    if op == "markov_argmax_partials":
        attrs = node.get("attrs", {})
        block_width = int(attrs.get("block_width", 0))
        position = int(attrs.get("position", -1))
        vocabulary_size = int(attrs.get("vocabulary_size", 0))
        rank = int(attrs.get("rank", 0))
        tile_width = int(attrs.get("vocabulary_tile_width", 0))
        parameter_shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node.get("params", [])
        ]
        parameter_dtypes = [
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node.get("params", [])
        ]
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 2
            or position < 0
            or position >= block_width
            or vocabulary_size <= 0
            or rank <= 0
            or tile_width != 256
            or parameter_shapes != [[vocabulary_size, rank], [vocabulary_size, rank]]
            or parameter_dtypes != ["BF16", "BF16"]
            or attrs.get("sampling") != "greedy"
            or attrs.get("dependency") != "previous_sampled_token"
            or attrs.get("output_element_bytes") != [4, 2]
        ):
            raise ModelCompileError(
                f"Markov proposal node {node['id']!r} has an invalid contract"
            )
        return (
            f"markov_argmax_partials_b{block_width}_p{position}_v{vocabulary_size}_"
            f"r{rank}_t{tile_width}.comp"
        )

    if op == "argmax_candidate_reduce":
        candidate_count = int(node.get("attrs", {}).get("candidate_count", 0))
        if (
            len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or candidate_count <= 0
            or candidate_count > 1024
            or node["attrs"].get("tie_break") != "lowest_token_id"
            or node["attrs"].get("output_element_bytes") != [4]
        ):
            raise ModelCompileError(
                f"argmax reduction node {node['id']!r} has an invalid contract"
            )
        return f"argmax_candidate_reduce_c{candidate_count}.comp"

    if op == "pack_token_block":
        block_width = int(node.get("attrs", {}).get("block_width", 0))
        if (
            block_width <= 0
            or len(node.get("inputs", [])) != block_width
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or node["attrs"].get("output_element_bytes") != [4]
        ):
            raise ModelCompileError(
                f"token-pack node {node['id']!r} has an invalid contract"
            )
        return f"pack_token_block_b{block_width}.comp"

    if op in {"quantize_fp8_e4m3", "quantize_fp8_e4m3_e8m0"}:
        scale_suffix = "_spow2" if op.endswith("_e8m0") else ""
        return (
            f"quantize_fp8_e4m3{scale_suffix}_b{int(node['attrs']['block_columns'])}"
            f"_h{int(node['attrs']['element_count'])}.comp"
        )
    if op == "quantize_int8_symmetric":
        return (
            f"quantize_int8_symmetric_b{int(node['attrs']['block_columns'])}"
            f"_h{int(node['attrs']['element_count'])}.comp"
        )
    if op == "quantize_int8_symmetric_pairpacked":
        return (
            "quantize_int8_symmetric_pairpacked_"
            f"b{int(node['attrs']['block_columns'])}"
            f"_h{int(node['attrs']['element_count'])}.comp"
        )
    if op == "rms_norm":
        attrs = node.get("attrs", {})
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        node_hidden_size = (
            int(parameter_shape[0])
            if isinstance(parameter_shape, list) and len(parameter_shape) == 1
            else 0
        )
        eps = float(attrs.get("eps", 0.0))
        weight_offset = float(attrs.get("weight_offset", 0.0))
        if (
            node_hidden_size <= 0
            or node_hidden_size % 2
            or len(node.get("inputs", [])) != 1
            or len(node.get("params", [])) != 1
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or not math.isfinite(eps)
            or eps <= 0.0
            or not math.isfinite(weight_offset)
        ):
            raise ModelCompileError(
                f"RMS normalization node {node['id']!r} has an invalid contract"
            )
        block_width = int(attrs.get("block_width", 0))
        if block_width > 0:
            if (
                int(attrs.get("hidden_size", 0)) != node_hidden_size
                or len(node.get("outputs", [])) != 1
                or attrs.get("output_element_bytes") != [2]
            ):
                raise ModelCompileError(
                    f"block RMS norm node {node['id']!r} has an invalid contract"
                )
            return (
                f"rms_norm_block_b{block_width}_bf16_h{node_hidden_size}_"
                f"eps{shader_float_token(eps)}_offset{shader_float_token(weight_offset)}.comp"
            )
        representations = node.get("attrs", {}).get("physical_output_representations")
        if representations:
            valid_common = (
                len(representations) != 1
                or representations[0].get("logical_signal")
                != node.get("outputs", [None])[0]
                or int(representations[0].get("element_count", 0))
                != node_hidden_size
            )
            contract = representations[0].get("contract")
            block_columns = int(representations[0].get("block_columns", 0))
            if (
                valid_common
                or (
                    contract
                    in {
                        FP8_PREQUANTIZATION_CONTRACT,
                        FP8_E8M0_PREQUANTIZATION_CONTRACT,
                    }
                    and block_columns != 128
                )
                or (
                    contract == PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT
                    and block_columns != 32
                )
                or contract
                not in {
                    FP8_PREQUANTIZATION_CONTRACT,
                    FP8_E8M0_PREQUANTIZATION_CONTRACT,
                    PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT,
                }
            ):
                raise ModelCompileError(
                    f"RMS normalization node {node['id']!r} has an invalid "
                    "physical output representation"
                )
            representation_token = (
                "fp8_e4m3"
                if contract
                in {
                    FP8_PREQUANTIZATION_CONTRACT,
                    FP8_E8M0_PREQUANTIZATION_CONTRACT,
                }
                else "int8_pairpacked"
            )
            scale_suffix = (
                "_spow2"
                if contract == FP8_E8M0_PREQUANTIZATION_CONTRACT
                else ""
            )
            return (
                f"rms_norm_quantize_{representation_token}{scale_suffix}_b{block_columns}_"
                f"h{node_hidden_size}_eps{shader_float_token(eps)}"
                f"_offset{shader_float_token(weight_offset)}.comp"
            )
        if len(node.get("outputs", [])) != 1:
            raise ModelCompileError(
                f"RMS normalization node {node['id']!r} has an invalid output contract"
            )
        return rms_norm_shader_file(
            node_hidden_size,
            eps,
            weight_offset,
        )
    if (
        op == "linear_projection"
        and int(node.get("attrs", {}).get("block_width", 0)) > 0
    ):
        attrs = node["attrs"]
        block_width = int(attrs["block_width"])
        input_size = int(attrs.get("input_size", 0))
        output_size = int(attrs.get("output_size", 0))
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if (
            block_width <= 0
            or input_size != hidden_size
            or output_size <= 0
            or parameter_shape != [output_size, input_size]
            or parameter_dtype != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
            or attrs.get("output_element_bytes") != [4]
        ):
            raise ModelCompileError(
                f"block output projection node {node['id']!r} has an invalid contract"
            )
        return (
            f"linear_projection_block_b{block_width}_bf16_"
            f"{input_size}x{output_size}_f32.comp"
        )
    if op == "confidence_projection_block":
        attrs = node["attrs"]
        block_width = int(attrs["block_width"])
        input_size = int(attrs.get("input_size", 0))
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if (
            block_width <= 0
            or input_size <= 0
            or input_size % 2
            or len(node.get("inputs", [])) != block_width + 1
            or parameter_shape != [1, input_size]
            or parameter_dtype != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
            or attrs.get("output_element_bytes") != [4]
        ):
            raise ModelCompileError(
                f"block confidence projection node {node['id']!r} has an invalid contract"
            )
        rank = input_size - hidden_size
        if rank <= 0 or rank % 2:
            raise ModelCompileError(
                f"block confidence projection node {node['id']!r} has an invalid rank"
            )
        return (
            f"confidence_projection_block_b{block_width}_bf16_"
            f"h{hidden_size}_r{rank}.comp"
        )
    if op == "linear":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
                has_bias = len(node.get("params", [])) == 3
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
                has_bias = len(node.get("params", [])) == 3
            else:
                raise ModelCompileError(
                    f"linear node {node['id']!r} has unsupported packed format "
                    f"{quantization_format!r}"
                )
            scale_dtype = packed_int4_scale_dtype_for_node(
                circuit, node, tensor_index
            ).lower()
            prefix = "linear_bias" if has_bias else "linear"
            if uses_pairpacked_int8_input(node):
                prefix += "_prequant_pairpacked"
            elif uses_prequantized_int8_input(node):
                prefix += "_prequant"
            return (
                f"{prefix}_int4_{format_token}_s{scale_dtype}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            scale_suffix = fp8_scale_shader_suffix_for_node(circuit, node, tensor_index)
            has_bias = len(node.get("params", [])) == 3
            prefix = "linear_bias" if has_bias else "linear"
            if uses_prequantized_fp8_input(node):
                prefix += "_prequant"
            return (
                f"{prefix}_fp8_e4m3{scale_suffix}_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "Q8_0":
            out_features, in_features = q8_0_linear_shape_for_node(
                circuit, node, tensor_index
            )
            has_bias = len(node.get("params", [])) == 2
            prefix = "linear_bias" if has_bias else "linear"
            return f"{prefix}_q8_0_{in_features}x{out_features}.comp"
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear node {node['id']!r} has unsupported layout {layout!r}"
            )
        has_bias = len(node.get("params", [])) == 2
        prefix = "linear"
        if has_bias:
            prefix += "_bias"
        prefix += "_bf16"
        return f"{prefix}_{in_features}x{out_features}.comp"
    if op == "grouped_linear":
        attrs = node.get("attrs", {})
        groups = int(attrs.get("groups", 0))
        rank_per_group = int(attrs.get("rank_per_group", 0))
        out_features, group_input_features = parameter_shape_for_node(
            circuit, node, tensor_index
        )
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if (
            groups <= 0
            or rank_per_group <= 0
            or rank_per_group % 2
            or int(out_features) != groups * rank_per_group
            or int(group_input_features) <= 0
            or int(group_input_features) % 128
            or parameter_dtype != "F8_E4M3"
            or len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 2
            or node.get("state_reads")
            or node.get("state_writes")
        ):
            raise ModelCompileError(
                f"grouped linear node {node['id']!r} has an invalid contract"
            )
        block_rows, block_columns = fp8_block_shape_for_node(
            circuit, node, tensor_index
        )
        scale_suffix = fp8_scale_shader_suffix_for_node(circuit, node, tensor_index)
        total_input_features = groups * int(group_input_features)
        return (
            f"grouped_linear_fp8_e4m3{scale_suffix}_b{block_rows}x{block_columns}_"
            f"g{groups}_{total_input_features}x{out_features}.comp"
        )
    if op == "mixed_parallel_linear_4way":
        attrs = node.get("attrs", {})
        if (
            len(node.get("inputs", [])) != 3
            or len(node.get("outputs", [])) != 4
            or len(node.get("params", [])) != 6
            or attrs.get("branch_count") != 4
            or attrs.get("branch_parameter_counts") != [2, 2, 1, 1]
            or attrs.get("branch_dtypes") != ["F8_E4M3", "F8_E4M3", "BF16", "BF16"]
            or not uses_prequantized_fp8_input(node)
        ):
            raise ModelCompileError(
                f"mixed parallel-linear node {node['id']!r} has invalid bindings"
            )
        parameter_groups = [
            node["params"][:2],
            node["params"][2:4],
            node["params"][4:5],
            node["params"][5:6],
        ]
        shapes = [
            parameter_shape_for_id(circuit, group[0], tensor_index)
            for group in parameter_groups
        ]
        dtypes = [
            parameter_dtype_for_id(circuit, group[0], tensor_index)
            for group in parameter_groups
        ]
        layouts = {
            parameter_layout_for_id(circuit, group[0], tensor_index)
            for group in parameter_groups
        }
        block_shapes = {
            fp8_block_shape_for_node(
                circuit,
                {
                    "id": f"{node['id']}__branch_{index}",
                    "params": parameter_groups[index],
                },
                tensor_index,
            )
            for index in range(2)
        }
        if (
            dtypes != ["F8_E4M3", "F8_E4M3", "BF16", "BF16"]
            or layouts != {ROW_MAJOR_LAYOUT}
            or len(block_shapes) != 1
            or any(len(shape) != 2 for shape in shapes)
            or len({int(shape[1]) for shape in shapes}) != 1
        ):
            raise ModelCompileError(
                f"mixed parallel-linear node {node['id']!r} has incompatible "
                f"projection shapes {shapes}"
            )
        block_rows, block_columns = block_shapes.pop()
        input_width = int(shapes[0][1])
        output_widths = [int(shape[0]) for shape in shapes]
        return (
            "mixed_parallel_linear_4way_prequant_fp8_e4m3_"
            f"b{block_rows}x{block_columns}_bf16_{input_width}x"
            + "_".join(map(str, output_widths))
            + ".comp"
        )
    if op == "contiguous_linear_swiglu":
        attrs = node.get("attrs", {})
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 2
            or attrs.get("weight_partition") != "contiguous_gate_up"
            or attrs.get("intermediate_rounding") != "BF16"
            or not uses_prequantized_fp8_input(node)
        ):
            raise ModelCompileError(
                f"contiguous SwiGLU node {node['id']!r} has invalid bindings"
            )
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        part_width = int(attrs.get("part_width", 0))
        if (
            len(parameter_shape) != 2
            or part_width <= 0
            or part_width % 2
            or int(parameter_shape[0]) != 2 * part_width
            or int(parameter_shape[1]) <= 0
            or parameter_dtype_for_node(circuit, node, tensor_index) != "F8_E4M3"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"contiguous SwiGLU node {node['id']!r} has incompatible "
                f"projection shape {parameter_shape}"
            )
        block_rows, block_columns = fp8_block_shape_for_node(
            circuit,
            node,
            tensor_index,
        )
        input_width = int(parameter_shape[1])
        return (
            "contiguous_linear_swiglu_prequant_fp8_e4m3_"
            f"b{block_rows}x{block_columns}_{input_width}x{part_width}.comp"
        )
    if op in {"parallel_linear_2way", "parallel_linear_3way"}:
        expected_branch_count = 2 if op == "parallel_linear_2way" else 3
        branch_count = int(node["attrs"]["branch_count"])
        branch_parameter_counts = [
            int(count)
            for count in node["attrs"].get(
                "branch_parameter_counts", [1] * branch_count
            )
        ]
        if (
            branch_count != expected_branch_count
            or len(branch_parameter_counts) != branch_count
            or sum(branch_parameter_counts) != len(node["params"])
            or branch_count != len(node["outputs"])
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has inconsistent branch metadata"
            )
        branch_params = []
        offset = 0
        for count in branch_parameter_counts:
            branch_params.append(node["params"][offset : offset + count])
            offset += count
        shapes = [
            parameter_shape_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        ]
        input_widths = {int(shape[1]) for shape in shapes if len(shape) == 2}
        dtypes = {
            parameter_dtype_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        }
        if (
            len(shapes) != branch_count
            or any(len(shape) != 2 for shape in shapes)
            or len(input_widths) != 1
            or dtypes not in ({"BF16"}, {"F8_E4M3"}, {"Q8_0"})
        ):
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has incompatible shapes {shapes}"
            )
        output_widths = [int(shape[0]) for shape in shapes]
        layouts = {
            parameter_layout_for_id(circuit, parameter_ids[0], tensor_index)
            for parameter_ids in branch_params
        }
        if layouts != {ROW_MAJOR_LAYOUT}:
            raise ModelCompileError(
                f"parallel-linear node {node['id']!r} has unsupported layouts "
                f"{sorted(layouts)}"
            )
        if dtypes == {"F8_E4M3"}:
            block_shapes = {
                fp8_block_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_{index}",
                        "params": parameter_ids,
                    },
                    tensor_index,
                )
                for index, parameter_ids in enumerate(branch_params)
            }
            if len(block_shapes) != 1 or any(
                len(parameter_ids) != 2 for parameter_ids in branch_params
            ):
                raise ModelCompileError(
                    f"parallel-linear FP8 node {node['id']!r} has incompatible block scales"
                )
            block_rows, block_columns = block_shapes.pop()
            input_width = input_widths.pop()
            return (
                f"parallel_linear_{branch_count}way"
                f"{'_prequant' if uses_prequantized_fp8_input(node) else ''}"
                "_fp8_e4m3_"
                f"b{block_rows}x{block_columns}_{input_width}x"
                + "_".join(map(str, output_widths))
                + ".comp"
            )
        if dtypes == {"Q8_0"}:
            for parameter_ids in branch_params:
                q8_0_linear_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_q8",
                        "params": [parameter_ids[0]],
                    },
                    tensor_index,
                )
            if len(set(output_widths)) != 1:
                raise ModelCompileError(
                    f"parallel-linear Q8_0 node {node['id']!r} requires equal output widths"
                )
            input_width = input_widths.pop()
            return (
                f"parallel_linear_{branch_count}way_q8_0_"
                f"{input_width}x{output_widths[0]}.comp"
            )
        input_width = input_widths.pop()
        return (
            f"parallel_linear_{branch_count}way_bf16_{input_width}x"
            + "_".join(map(str, output_widths))
            + ".comp"
        )
    if op == "parallel_linear_silu_multiply":
        params = node.get("params", [])
        expected_input_count = 2 if uses_prequantized_fp8_input(node) else 1
        if (
            len(node.get("inputs", [])) != expected_input_count
            or len(node.get("outputs", [])) != 1
            or int(node.get("attrs", {}).get("branch_count", 0)) != 2
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid bindings"
            )
        if len(params) == 2:
            weight_ids = params
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            dtypes = {
                parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            layouts = {
                parameter_layout_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or dtypes not in ({"BF16"}, {"Q8_0"})
                or layouts != {ROW_MAJOR_LAYOUT}
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            block_shape = None
            q8_shape = dtypes == {"Q8_0"}
            if q8_shape:
                for parameter_id in weight_ids:
                    q8_0_linear_shape_for_node(
                        circuit,
                        {
                            "id": f"{node['id']}__q8",
                            "params": [parameter_id],
                        },
                        tensor_index,
                    )
        elif len(params) == 4:
            weight_ids = [params[0], params[2]]
            shapes = [
                parameter_shape_for_id(circuit, parameter_id, tensor_index)
                for parameter_id in weight_ids
            ]
            branch_params = [params[:2], params[2:]]
            block_shapes = {
                fp8_block_shape_for_node(
                    circuit,
                    {
                        "id": f"{node['id']}__branch_{index}",
                        "params": parameter_ids,
                    },
                    tensor_index,
                )
                for index, parameter_ids in enumerate(branch_params)
            }
            if (
                len(shapes) != 2
                or shapes[0] != shapes[1]
                or len(shapes[0]) != 2
                or len(block_shapes) != 1
                or any(
                    parameter_dtype_for_id(circuit, parameter_id, tensor_index)
                    != "F8_E4M3"
                    or parameter_layout_for_id(circuit, parameter_id, tensor_index)
                    != ROW_MAJOR_LAYOUT
                    for parameter_id in weight_ids
                )
            ):
                raise ModelCompileError(
                    f"fused FFN projection node {node['id']!r} has incompatible "
                    f"parameters {shapes}"
                )
            block_shape = block_shapes.pop()
            q8_shape = False
        else:
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid parameter count "
                f"{len(params)}"
            )
        output_width, input_width = map(int, shapes[0])
        if (
            input_width <= 0
            or input_width % 2
            or output_width <= 0
            or output_width % 2
            or int(node["attrs"].get("element_count", 0)) != output_width
            or node["attrs"].get("intermediate_rounding") != "BF16"
        ):
            raise ModelCompileError(
                f"fused FFN projection node {node['id']!r} has invalid geometry"
            )
        if block_shape is not None:
            block_rows, block_columns = block_shape
            return (
                "parallel_linear_silu_multiply"
                f"{'_prequant' if uses_prequantized_fp8_input(node) else ''}"
                "_fp8_e4m3_"
                f"b{block_rows}x{block_columns}_{input_width}x{output_width}.comp"
            )
        if q8_shape:
            return (
                f"parallel_linear_silu_multiply_q8_0_{input_width}x{output_width}.comp"
            )
        return f"parallel_linear_silu_multiply_bf16_{input_width}x{output_width}.comp"
    if op == "linear_split_3way":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = map(int, parameter_shape)
        if parameter_dtype_for_node(circuit, node, tensor_index) != "BF16":
            raise ModelCompileError(
                f"linear-split node {node['id']!r} requires BF16 weights"
            )
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            len(part_widths) != 3
            or any(width <= 0 or width % 2 for width in part_widths)
            or sum(part_widths) != out_features
        ):
            raise ModelCompileError(
                f"linear-split node {node['id']!r} cannot partition {out_features} "
                f"outputs into {part_widths}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear-split node {node['id']!r} has unsupported layout {layout!r}"
            )
        return (
            f"linear_split_3way_bf16_{in_features}x"
            + "_".join(map(str, part_widths))
            + ".comp"
        )
    if op == "linear_residual":
        parameter_shape = parameter_shape_for_node(circuit, node, tensor_index)
        out_features, in_features = parameter_shape
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            if quantization_format == "auto_gptq":
                group_size = packed_int4_linear_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "gptq"
            elif quantization_format == "compressed_tensors_pack_quantized":
                group_size = compressed_tensors_int4_group_size_for_node(
                    circuit, node, tensor_index
                )
                format_token = "ct"
            else:
                raise ModelCompileError(
                    f"linear-residual node {node['id']!r} has unsupported packed "
                    f"format {quantization_format!r}"
                )
            scale_dtype = packed_int4_scale_dtype_for_node(
                circuit, node, tensor_index
            ).lower()
            prefix = (
                "linear_residual_prequant"
                if uses_prequantized_int8_input(node)
                else "linear_residual"
            )
            return (
                f"{prefix}_int4_{format_token}_s{scale_dtype}_g{group_size}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_block_shape_for_node(
                circuit, node, tensor_index
            )
            return (
                "linear_residual"
                f"{'_prequant' if uses_prequantized_fp8_input(node) else ''}"
                f"_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"{in_features}x{out_features}.comp"
            )
        if parameter_dtype == "Q8_0":
            out_features, in_features = q8_0_linear_shape_for_node(
                circuit, node, tensor_index
            )
            return f"linear_residual_q8_0_{in_features}x{out_features}.comp"
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"linear-residual node {node['id']!r} has unsupported weight dtype "
                f"{parameter_dtype}"
            )
        layout = parameter_layout_for_node(circuit, node, tensor_index)
        if layout != ROW_MAJOR_LAYOUT:
            raise ModelCompileError(
                f"linear-residual node {node['id']!r} has unsupported layout {layout!r}"
            )
        return f"linear_residual_bf16_{in_features}x{out_features}.comp"
    if op == "split":
        if node["attrs"].get("layout") == "per_head_interleaved":
            return (
                f"split_bf16_2x{node['attrs']['blocks']}x{node['attrs']['block_part_width']}"
                "_head_interleaved.comp"
            )
        if node["attrs"].get("part_widths") is not None:
            part_widths = [int(width) for width in node["attrs"]["part_widths"]]
            if len(part_widths) != 3:
                raise ModelCompileError(
                    f"split node {node['id']!r} has unsupported unequal part widths {part_widths}"
                )
            return "split_bf16_3x" + "_".join(map(str, part_widths)) + ".comp"
        part_width = int(node["attrs"]["part_width"])
        return f"split_bf16_{len(node['outputs'])}x{part_width}.comp"
    if op == "concatenate":
        part_widths = [int(width) for width in node["attrs"]["part_widths"]]
        if (
            node["attrs"].get("axis") != "channel"
            or len(node.get("inputs", [])) != len(part_widths)
            or len(node.get("outputs", [])) != 1
            or any(width <= 0 or width % 2 for width in part_widths)
        ):
            raise ModelCompileError(
                f"concatenate node {node['id']!r} has unsupported geometry"
            )
        return "concatenate_bf16_" + "_".join(map(str, part_widths)) + ".comp"
    if op == "multiply":
        element_count = int(
            node.get("attrs", {}).get(
                "element_count",
                (
                    feed_forward_intermediate_size(circuit)
                    if node["id"] == "ffn_gate_multiply"
                    else hidden_size
                ),
            )
        )
        return f"multiply_bf16_{element_count}.comp"
    if op == "scalar_multiply":
        return f"scalar_multiply_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "rolling_state_update":
        state_reads = node.get("state_reads", [])
        attrs = node.get("attrs", {})
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or len(state_reads) != 1
            or state_reads != node.get("state_writes")
            or node["inputs"][1] != state_reads[0]
            or attrs.get("update") not in {"ring_append", "shift_append"}
        ):
            raise ModelCompileError(
                f"rolling-state node {node['id']!r} has an invalid contract"
            )
        temporal_memory = state_port(circuit, state_reads[0])
        if "shape" in temporal_memory:
            frames, state_hidden = map(int, temporal_memory["shape"])
        else:
            per_token_shape = list(map(int, temporal_memory.get("shape_per_token", [])))
            frames = int(temporal_memory.get("capacity", 0))
            state_hidden = math.prod(per_token_shape) if per_token_shape else 0
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames <= 0
            or state_hidden <= 0
            or state_hidden % 2
            or int(attrs.get("capacity", frames)) != frames
        ):
            raise ModelCompileError(
                f"rolling-state node {node['id']!r} has incompatible state geometry"
            )
        if attrs["update"] == "ring_append":
            binding = stream_control_binding_for_node(circuit, node)
            return (
                f"rolling_state_ring_append_bf16_{frames}x{state_hidden}"
                f"__sc{binding}.comp"
            )
        return f"rolling_state_update_bf16_{frames}x{state_hidden}.comp"
    if op == "depthwise_conv1d":
        temporal_memory = state_port(circuit, "temporal_memory")
        frames, state_hidden = temporal_memory["shape"]
        return f"depthwise_conv1d_bf16_{frames}x{state_hidden}.comp"
    if op in {"multiply_rolling_depthwise", "multiply_rolling_depthwise_gate"}:
        expected_input_count = 4 if op.endswith("_gate") else 3
        if (
            len(node.get("inputs", [])) != expected_input_count
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 1
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, state_hidden = map(int, temporal_memory["shape"])
        kernel_shape = parameter_shape_for_node(circuit, node, tensor_index)
        supported_kernel_shapes = ([state_hidden, frames], [state_hidden, 1, frames])
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or state_hidden <= 0
            or state_hidden % 2
            or kernel_shape not in supported_kernel_shapes
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"fused recurrent convolution node {node['id']!r} has incompatible "
                f"state {temporal_memory.get('shape')} or kernel {kernel_shape}"
            )
        shader_prefix = (
            "multiply_rolling_depthwise_gate"
            if op.endswith("_gate")
            else "multiply_rolling_depthwise"
        )
        return f"{shader_prefix}_bf16_{frames}x{state_hidden}.comp"
    if op == "linear_split_recurrent_depthwise_gate":
        if (
            len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 2
            or len(node.get("state_reads", [])) != 1
            or node.get("state_reads") != node.get("state_writes")
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has invalid bindings"
            )
        temporal_memory = state_port(circuit, node["state_reads"][0])
        frames, hidden_size = map(int, temporal_memory["shape"])
        projection_shape = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        kernel_shape = parameter_shape_for_id(circuit, node["params"][1], tensor_index)
        part_widths = [
            int(width) for width in node["attrs"]["projection"]["part_widths"]
        ]
        input_gate_indices = [
            int(index) for index in node["attrs"]["input_gate_branch_indices"]
        ]
        output_gate_index = int(node["attrs"]["output_gate_branch_index"])
        projection_layout = parameter_layout_for_id(
            circuit, node["params"][0], tensor_index
        )
        if (
            temporal_memory.get("dtype") != "BF16"
            or frames < 2
            or hidden_size <= 0
            or hidden_size % 2
            or len(projection_shape) != 2
            or projection_shape[0] != 3 * hidden_size
            or projection_shape[1] <= 0
            or projection_shape[1] % 2
            or part_widths != [hidden_size] * 3
            or sorted([*input_gate_indices, output_gate_index]) != [0, 1, 2]
            or kernel_shape not in ([hidden_size, frames], [hidden_size, 1, frames])
            or any(
                parameter_dtype_for_id(circuit, parameter_id, tensor_index) != "BF16"
                for parameter_id in node["params"]
            )
            or projection_layout != ROW_MAJOR_LAYOUT
            or parameter_layout_for_id(circuit, node["params"][1], tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"projected recurrent convolution node {node['id']!r} has "
                f"incompatible projection {projection_shape}, state "
                f"{temporal_memory.get('shape')}, or kernel {kernel_shape}"
            )
        return (
            "linear_split_recurrent_depthwise_gate_bf16_"
            f"{projection_shape[1]}x{hidden_size}_k{frames}"
            f"_ig{input_gate_indices[0]}_{input_gate_indices[1]}"
            f"_og{output_gate_index}.comp"
        )
    if op == "residual_add":
        return f"add_bf16_{hidden_size}.comp"
    if op == "scaled_residual_add":
        return (
            f"scaled_add_bf16_{hidden_size}"
            f"_scale{shader_float_token(float(node['attrs']['scale']))}.comp"
        )
    if op == "silu":
        return f"silu_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "gelu_tanh":
        return f"gelu_tanh_bf16_{int(node['attrs']['element_count'])}.comp"
    if op == "silu_multiply":
        element_count = int(node["attrs"]["element_count"])
        representations = node.get("attrs", {}).get("physical_output_representations")
        if representations:
            if (
                len(representations) != 1
                or representations[0].get("contract")
                != PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT
                or representations[0].get("logical_signal")
                != node.get("outputs", [None])[0]
                or int(representations[0].get("element_count", 0)) != element_count
                or int(representations[0].get("block_columns", 0)) != 32
            ):
                raise ModelCompileError(
                    f"SiLU-multiply node {node['id']!r} has an invalid "
                    "physical output representation"
                )
            return f"silu_multiply_quantize_int8_pairpacked_b32_h{element_count}.comp"
        return f"silu_multiply_bf16_{element_count}.comp"
    if op == "bounded_silu_multiply":
        attrs = node.get("attrs", {})
        element_count = int(attrs.get("element_count", 0))
        limit = float(attrs.get("limit", 0.0))
        if (
            element_count <= 0
            or element_count % 2
            or not math.isfinite(limit)
            or limit <= 0.0
            or len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or node.get("state_reads")
            or node.get("state_writes")
        ):
            raise ModelCompileError(
                f"bounded SiLU-multiply node {node['id']!r} has an invalid contract"
            )
        return (
            f"bounded_silu_multiply_bf16_{element_count}_"
            f"limit{shader_float_token(limit)}.comp"
        )
    if op == "sigmoid_multiply":
        representations = node.get("attrs", {}).get("physical_output_representations")
        if representations:
            if (
                len(representations) != 1
                or representations[0].get("contract")
                not in {
                    FP8_PREQUANTIZATION_CONTRACT,
                    FP8_E8M0_PREQUANTIZATION_CONTRACT,
                }
                or representations[0].get("logical_signal")
                != node.get("outputs", [None])[0]
                or int(representations[0].get("element_count", 0)) <= 0
                or int(representations[0].get("block_columns", 0)) != 128
            ):
                raise ModelCompileError(
                    f"sigmoid-multiply node {node['id']!r} has an invalid "
                    "physical FP8 output representation"
                )
            return (
                "sigmoid_multiply_quantize_fp8_e4m3"
                f"{'_spow2' if representations[0]['contract'] == FP8_E8M0_PREQUANTIZATION_CONTRACT else ''}"
                "_b128_"
                f"h{int(representations[0]['element_count'])}.comp"
            )
        return "sigmoid_multiply_bf16.comp"
    if op == "softplus_multiply":
        attrs = node.get("attrs", {})
        query_heads = int(attrs.get("query_heads", 0))
        head_width = int(attrs.get("head_width", 0))
        if query_heads <= 0 or head_width <= 0 or head_width % 2:
            raise ModelCompileError(
                f"softplus gate node {node['id']!r} has invalid attention geometry"
            )
        mode = "per_head" if attrs.get("per_head") else "per_element"
        return f"softplus_multiply_bf16_q{query_heads}_d{head_width}_{mode}.comp"
    if op == "sigmoid_scalar_multiply":
        return f"sigmoid_scalar_multiply_bf16_{hidden_size}.comp"
    if op == "linear_sigmoid_scalar_multiply":
        out_features, in_features = parameter_shape_for_node(
            circuit, node, tensor_index
        )
        if (
            int(out_features) != 1
            or int(in_features) <= 0
            or int(in_features) % 2
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"linear scalar-gate node {node['id']!r} has unsupported geometry"
            )
        return f"linear_sigmoid_scalar_multiply_bf16_{in_features}x{hidden_size}.comp"
    if op == "linear_sigmoid_scalar_multiply_residual2":
        out_features, in_features = parameter_shape_for_node(
            circuit, node, tensor_index
        )
        if (
            int(out_features) != 1
            or int(in_features) <= 0
            or int(in_features) % 2
            or int(hidden_size) <= 0
            or int(hidden_size) % 2
            or parameter_dtype_for_node(circuit, node, tensor_index) != "BF16"
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"linear scalar-gate residual node {node['id']!r} has "
                "unsupported geometry"
            )
        return (
            "linear_sigmoid_scalar_multiply_residual2_bf16_"
            f"{in_features}x{hidden_size}.comp"
        )
    if op == "rms_norm_per_head":
        return (
            f"rms_norm_per_head_bf16_{node['attrs']['head_count']}x"
            f"{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}"
            f"_offset{shader_float_token(float(node['attrs']['weight_offset']))}.comp"
        )
    if op == "rms_norm_per_head_unscaled":
        return (
            f"rms_norm_per_head_unscaled_bf16_{node['attrs']['head_count']}"
            f"x{node['attrs']['head_width']}"
            f"_eps{shader_float_token(float(node['attrs']['eps']))}.comp"
        )
    if op == "parallel_head_norm_rope_2way":
        branches = node.get("attrs", {}).get("branches", [])
        if (
            len(branches) != 2
            or len(node.get("inputs", [])) != 2
            or len(node.get("outputs", [])) != 2
            or len(node.get("params", [])) != 2
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has invalid branch metadata"
            )
        norms = [branch.get("norm", {}) for branch in branches]
        ropes = [branch.get("rope", {}) for branch in branches]
        head_counts = [int(norm["head_count"]) for norm in norms]
        common_fields = {
            "head_width": {int(norm["head_width"]) for norm in norms}
            | {int(rope["head_width"]) for rope in ropes},
            "eps": {float(norm["eps"]) for norm in norms},
            "weight_offset": {float(norm["weight_offset"]) for norm in norms},
            "rotary_width": {int(rope["rotary_width"]) for rope in ropes},
            "theta": {float(rope["theta"]) for rope in ropes},
            "rope_type": {str(rope.get("rope_type", "default")) for rope in ropes},
            "interleaved": {bool(rope["interleaved"]) for rope in ropes},
        }
        if (
            any(len(values) != 1 for values in common_fields.values())
            or any(
                int(norm["head_count"]) != int(rope["head_count"])
                for norm, rope in zip(norms, ropes, strict=True)
            )
            or ropes[0].get("scaling") != ropes[1].get("scaling")
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} mixes incompatible branch geometry"
            )
        parameter_dtypes = {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        }
        parameter_shapes = [
            parameter_shape_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in node["params"]
        ]
        head_width = common_fields["head_width"].pop()
        if parameter_dtypes != {"BF16"} or any(
            list(map(int, shape)) != [head_width] for shape in parameter_shapes
        ):
            raise ModelCompileError(
                f"parallel head-norm/rope node {node['id']!r} has incompatible "
                f"normalization parameters {parameter_shapes}"
            )
        rope_attrs = {
            "theta": common_fields["theta"].pop(),
            "rope_type": common_fields["rope_type"].pop(),
            "interleaved": common_fields["interleaved"].pop(),
            "scaling": ropes[0].get("scaling"),
        }
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"parallel_head_norm_rope_2way_bf16_h{head_counts[0]}_{head_counts[1]}"
            f"_d{head_width}_r{common_fields['rotary_width'].pop()}"
            f"_eps{shader_float_token(common_fields['eps'].pop())}"
            f"_offset{shader_float_token(common_fields['weight_offset'].pop())}"
            f"_{rope_shader_suffix(rope_attrs)}"
            f"__sc{binding}.comp"
        )
    if op == "per_layer_embedding":
        attrs = node["attrs"]
        token_shape = parameter_shape_for_id(circuit, "token_embedding", tensor_index)
        vocab_size = int(token_shape[0])
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"per_layer_embedding_bf16_v{vocab_size}_h{attrs['hidden_size']}"
            f"_p{attrs['per_layer_width']}_l{attrs['layer_index']}of{attrs['layer_count']}"
            f"_c{attrs['embedding_chunk_count']}r{attrs['embedding_chunk_rows']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}"
            f"_tes{shader_float_token(float(attrs['token_embedding_scale']))}"
            f"_pes{shader_float_token(float(attrs['per_layer_embedding_scale']))}"
            f"_mps{shader_float_token(float(attrs['model_projection_scale']))}"
            f"_cs{shader_float_token(float(attrs['combination_scale']))}__sc{binding}.comp"
        )
    if op in {"rotary_position_embedding", "inverse_rotary_position_embedding"}:
        attrs = node["attrs"]
        head_count = int(attrs.get("head_count", 0))
        head_width = int(attrs.get("head_width", 0))
        rotary_width = int(attrs.get("rotary_width", 0))
        theta = float(attrs.get("theta", 0.0))
        if (
            head_count <= 0
            or head_width <= 0
            or head_width % 2
            or rotary_width <= 0
            or rotary_width % 2
            or rotary_width > head_width
            or not math.isfinite(theta)
            or theta <= 0.0
            or attrs.get("position_source") != "stream_tick"
            or len(node.get("inputs", [])) != 1
            or len(node.get("outputs", [])) != 1
            or node.get("params")
            or node.get("state_reads")
            or node.get("state_writes")
        ):
            raise ModelCompileError(
                f"rotary node {node['id']!r} has an invalid contract"
            )
        binding = stream_control_binding_for_node(circuit, node)
        prefix = "inverse_rotary" if op.startswith("inverse_") else "rotary"
        position_offset = int(attrs.get("position_offset", 0))
        rotary_scope = str(attrs.get("rotary_scope", "prefix"))
        if rotary_scope not in {"prefix", "tail"}:
            raise ModelCompileError(
                f"rotary node {node['id']!r} has an invalid rotary scope"
            )
        scope_suffix = "_tail" if rotary_scope == "tail" else ""
        activation_quantization = attrs.get("activation_quantization")
        quantization_suffix = ""
        if activation_quantization is not None:
            block_columns = int(activation_quantization.get("block_columns", 0))
            scale_format = activation_quantization.get("scale_format")
            if (
                op != "rotary_position_embedding"
                or activation_quantization.get("format") != "fp8_e4m3"
                or activation_quantization.get("scope")
                != "non_rotary_dimensions"
                or activation_quantization.get("mode")
                != "quantize_dequantize"
                or scale_format not in {"f32_exact", "e8m0_power_of_two"}
                or block_columns <= 0
                or (head_width - rotary_width) <= 0
                or (head_width - rotary_width) % block_columns
                or rotary_scope != "tail"
                or str(attrs.get("rope_type", "default")) == "proportional"
            ):
                raise ModelCompileError(
                    f"rotary node {node['id']!r} has an invalid activation quantization contract"
                )
            scale_token = (
                "spow2" if scale_format == "e8m0_power_of_two" else "sexact"
            )
            prefix = "rotary_qdq_fp8_e4m3"
            quantization_suffix = f"_{scale_token}_b{block_columns}"
        return (
            f"{prefix}{quantization_suffix}_bf16_{head_count}x"
            f"{head_width}"
            f"_r{rotary_width}"
            f"_{rope_shader_suffix(attrs)}"
            f"{scope_suffix}"
            f"{f'_po{position_offset}' if position_offset else ''}"
            f"__sc{binding}.comp"
        )
    if op == "append_state_update":
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"append_kv_state_bf16_{node['attrs']['key_value_heads']}"
            f"x{node['attrs']['head_width']}__sc{binding}.comp"
        )
    if op == "indexed_sparse_attention":
        attrs = node.get("attrs", {})
        inputs = node.get("inputs", [])
        query_heads = int(attrs.get("query_heads", 0))
        key_value_heads = int(attrs.get("key_value_heads", 0))
        head_width = int(attrs.get("head_width", 0))
        window_size = int(attrs.get("window_size", 0))
        scale = float(attrs.get("scale", 0.0))
        if (
            len(node.get("outputs", [])) != 1
            or len(node.get("params", [])) != 1
            or node.get("state_reads")
            or node.get("state_writes")
            or query_heads <= 0
            or key_value_heads <= 0
            or query_heads % key_value_heads
            or head_width <= 0
            or head_width % 64
            or head_width > 1024
            or window_size <= 0
            or not math.isfinite(scale)
            or scale <= 0.0
            or parameter_dtype_for_node(circuit, node, tensor_index) != "F32"
            or parameter_shape_for_node(circuit, node, tensor_index) != [query_heads]
            or parameter_layout_for_node(circuit, node, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"indexed sparse-attention node {node['id']!r} has an invalid contract"
            )
        binding = stream_control_binding_for_node(circuit, node)
        common = f"q{query_heads}_kv{key_value_heads}_d{head_width}_w{window_size}_"
        if attrs.get("causal") is False:
            if (
                len(inputs) != 3
                or attrs.get("intra_block_visibility") != "all"
                or attrs.get("query_state") != "transient"
            ):
                raise ModelCompileError(
                    f"indexed sparse-attention node {node['id']!r} has an invalid contract"
                )
            return (
                f"indexed_sparse_attention_bf16_{common}"
                f"scale{shader_float_token(scale)}__sc{binding}.comp"
            )
        if (
            attrs.get("causal") is not True
            or len(inputs) not in {2, 4}
            or attrs.get("intra_block_visibility") is not None
            or attrs.get("query_state") is not None
        ):
            raise ModelCompileError(
                f"indexed sparse-attention node {node['id']!r} has an invalid contract"
            )
        compression_ratio = 0
        max_compressed_indices = 0
        if len(inputs) == 4:
            index_signal = inputs[3]
            index_producer = next(
                (
                    producer
                    for producer in circuit.get("nodes", [])
                    if index_signal in producer.get("outputs", [])
                ),
                None,
            )
            if index_producer is None:
                raise ModelCompileError(
                    f"indexed sparse-attention node {node['id']!r} has no index producer"
                )
            if index_producer.get("op") == "radix_topk_index":
                max_compressed_indices = int(
                    index_producer.get("attrs", {}).get("top_k", 0)
                )
                compressor = next(
                    (
                        producer
                        for producer in circuit.get("nodes", [])
                        if producer.get("op") == "learned_gated_kv_pool"
                    ),
                    None,
                )
                compression_ratio = int(
                    compressor.get("attrs", {}).get("ratio", 0)
                    if compressor is not None
                    else 0
                )
            elif index_producer.get("op") == "chronological_compressed_index":
                compression_ratio = int(index_producer.get("attrs", {}).get("ratio", 0))
                max_context = int(dimensions.get("max_position_embeddings", 0))
                max_compressed_indices = (
                    (max_context + compression_ratio - 1) // compression_ratio
                    if compression_ratio > 0
                    else 0
                )
            if compression_ratio <= 0 or max_compressed_indices <= 0:
                raise ModelCompileError(
                    f"indexed sparse-attention node {node['id']!r} has an invalid compressed-index contract"
                )
        return (
            f"indexed_sparse_attention_main_bf16_{common}"
            f"r{compression_ratio}_k{max_compressed_indices}_"
            f"scale{shader_float_token(scale)}__sc{binding}.comp"
        )
    if op == "scaled_dot_product_attention":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "gqa_attention_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "attention_partition_partials":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        name = (
            "attention_partition_partials_bf16_"
            f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}"
            f"_d{attrs['head_width']}_s{attrs['partition_count']}"
            f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        return f"{name}__sc{binding}.comp"
    if op == "append_scaled_dot_product_attention":
        attrs = node["attrs"]["attention"]
        binding = stream_control_binding_for_node(circuit, node)
        partition_count = node["attrs"].get("attention_partition_count")
        name = (
            (
                "append_gqa_attention_partition_reduce_bf16_"
                if partition_count is not None
                else "append_gqa_attention_bf16_"
            )
            + f"q{attrs['query_heads']}_kv{attrs['key_value_heads']}_d{attrs['head_width']}"
            + (f"_s{int(partition_count)}" if partition_count is not None else "")
            + f"_scale{shader_float_token(float(attrs['scale']))}"
        )
        if attrs.get("window_size") is not None:
            name += f"_w{int(attrs['window_size'])}"
        if attrs.get("attention_sinks"):
            name += "_sinks"
        return f"{name}__sc{binding}.comp"
    if op == "causal_conv1d_silu":
        return (
            f"causal_conv1d_silu_bf16_c{node['attrs']['channels']}"
            f"_k{node['attrs']['kernel_width']}.comp"
        )
    if op == "gated_delta_step":
        attrs = node["attrs"]
        dtype_tokens = {"F32": "f32", "BF16": "bf16"}
        parameter_tokens: dict[str, str] = {}
        for parameter_id in ("delta_a_log", "delta_dt_bias", "delta_norm"):
            actual_dtype = parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            if actual_dtype not in dtype_tokens:
                raise ModelCompileError(
                    f"gated-delta parameter {parameter_id} has dtype {actual_dtype}; "
                    "expected F32 or BF16"
                )
            parameter_tokens[parameter_id] = dtype_tokens[actual_dtype]
        shader_file = (
            f"gated_delta_step_k{attrs['key_heads']}x{attrs['key_head_width']}"
            f"_v{attrs['value_heads']}x{attrs['value_head_width']}"
            f"_a{parameter_tokens['delta_a_log']}"
            f"_dt{parameter_tokens['delta_dt_bias']}"
            f"_n{parameter_tokens['delta_norm']}"
            f"_eps{shader_float_token(float(attrs['norm_eps']))}"
        )
        representations = attrs.get("physical_output_representations")
        if representations:
            if (
                len(representations) != 1
                or representations[0].get("contract")
                != "bf16_blockwise_fp8_e4m3_f32_scale.v1"
                or representations[0].get("logical_signal")
                != node.get("outputs", [None])[0]
                or int(representations[0].get("element_count", 0))
                != int(attrs["value_heads"]) * int(attrs["value_head_width"])
                or int(representations[0].get("block_columns", 0))
                != int(attrs["value_head_width"])
            ):
                raise ModelCompileError(
                    f"gated-delta node {node['id']!r} has an invalid physical "
                    "FP8 output representation"
                )
            shader_file += f"_qfp8b{int(representations[0]['block_columns'])}"
        return f"{shader_file}.comp"
    if op == "rg_lru_step":
        attrs = node["attrs"]
        binding = stream_control_binding_for_node(circuit, node)
        return (
            f"rg_lru_step_bf16_h{attrs['width']}_b{attrs['heads']}x{attrs['block_width']}"
            f"_k{attrs['conv_kernel_width']}__sc{binding}.comp"
        )
    if op == "moe_topk":
        attrs = node["attrs"]
        activation = str(attrs.get("activation"))
        normalize_selected = bool(attrs.get("normalize_selected"))
        logit_softcap = float(attrs.get("logit_softcap"))
        has_bias = bool(attrs.get("selection_bias"))
        bias_dtype = None
        if activation not in {"sigmoid", "softmax"}:
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has unsupported activation {activation!r}"
            )
        if not math.isfinite(logit_softcap) or logit_softcap < 0.0:
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has invalid logit softcap {logit_softcap}"
            )
        if has_bias:
            if len(node.get("params", [])) != 1:
                raise ModelCompileError(
                    f"MoE router node {node['id']!r} is missing its selection bias"
                )
            bias_id = node["params"][0]
            if (
                parameter_shape_for_id(circuit, bias_id, tensor_index)
                != [int(attrs["num_experts"])]
                or parameter_dtype_for_id(circuit, bias_id, tensor_index)
                not in {"F32", "BF16"}
                or parameter_layout_for_id(circuit, bias_id, tensor_index)
                != ROW_MAJOR_LAYOUT
            ):
                raise ModelCompileError(
                    f"MoE router node {node['id']!r} has incompatible selection bias"
                )
            bias_dtype = parameter_dtype_for_id(circuit, bias_id, tensor_index)
        elif node.get("params"):
            raise ModelCompileError(
                f"MoE router node {node['id']!r} has an undeclared selection bias"
            )
        if (
            activation == "softmax"
            and normalize_selected
            and logit_softcap == 0.0
            and not has_bias
        ):
            return f"moe_topk_bf16_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        return (
            f"moe_topk_{activation}_bf16_e{attrs['num_experts']}_"
            f"k{attrs['experts_per_token']}_norm{int(normalize_selected)}_"
            f"cap{shader_float_token(logit_softcap)}_"
            f"{'bias' + bias_dtype.lower() if bias_dtype else 'nobias'}.comp"
        )
    if op == "moe_route":
        return independent_moe_route_shader_file(circuit, node, tensor_index)
    if op in {
        "independent_sparse_moe_gate_up",
        "independent_sparse_moe_down",
    }:
        return independent_sparse_moe_shader_file(circuit, node, tensor_index)
    if op in {"sparse_moe_gate_up", "sparse_moe_down"}:
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        stage = "gate_up" if op == "sparse_moe_gate_up" else "down"
        if parameter_dtype == "F8_E4M3":
            block_rows, block_columns = fp8_moe_block_shape_for_stage(
                circuit, node, tensor_index, stage=stage
            )
            emits_intermediate = (
                op == "sparse_moe_gate_up" and emits_sparse_moe_fp8_intermediate(node)
            )
            return (
                f"sparse_moe_{stage}"
                f"{'_prequant' if uses_prequantized_fp8_input(node) else ''}"
                f"{'_emit_intermediate' if emits_intermediate else ''}"
                f"_fp8_e4m3_b{block_rows}x{block_columns}_"
                f"h{attrs['hidden_size']}_i{attrs['intermediate_size']}_"
                f"e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
            )
        if parameter_dtype == "I32":
            group_size, scale_dtype = compressed_tensors_int4_moe_shape_for_stage(
                circuit, node, tensor_index, stage=stage
            )
            return (
                f"sparse_moe_{stage}_int4_ct_s{scale_dtype.lower()}_g{group_size}_"
                f"h{attrs['hidden_size']}_i{attrs['intermediate_size']}_"
                f"e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
            )
        if parameter_dtype != "BF16":
            raise ModelCompileError(
                f"sparse MoE {stage} node {node['id']!r} has unsupported expert dtype "
                f"{parameter_dtype}"
            )
        return (
            f"sparse_moe_{stage}_bf16_h{attrs['hidden_size']}_i{attrs['intermediate_size']}"
            f"_e{attrs['num_experts']}_k{attrs['experts_per_token']}.comp"
        )
    if op == "moe_reduce":
        attrs = node["attrs"]
        routed_scale = float(attrs["routed_scaling_factor"])
        if not math.isfinite(routed_scale) or routed_scale <= 0.0:
            raise ModelCompileError(
                f"MoE reduction node {node['id']!r} has invalid routed scale {routed_scale}"
            )
        return (
            f"moe_reduce_bf16_h{attrs['hidden_size']}"
            f"_k{attrs['experts_per_token']}"
            f"_scale{shader_float_token(routed_scale)}.comp"
        )

    raise ModelCompileError(
        f"no Vulkan shader selector for op {op!r} in node {node['id']!r}"
    )


def workgroup_count_x_for_node(circuit: Json, node: Json, tensor_index: Json) -> int:
    if node["op"] == "repeat_stream_lanes":
        attrs = node["attrs"]
        output_words = int(attrs["multiplicity"]) * int(attrs["hidden_size"]) // 2
        return (output_words + 63) // 64
    if node["op"] == "mean_stream_lanes":
        output_words = int(node["attrs"]["hidden_size"]) // 2
        return (output_words + 63) // 64
    if (
        node["op"]
        in {
            "sinkhorn_hyper_connection_head",
            "rms_norm",
        }
        and int(node.get("attrs", {}).get("block_width", 0)) > 0
    ):
        return int(node["attrs"]["block_width"])
    if (
        node["op"] == "linear_projection"
        and int(node.get("attrs", {}).get("block_width", 0)) > 0
    ):
        output_size = int(node["attrs"]["output_size"])
        return (output_size + 1) // 2
    if node["op"] == "markov_argmax_partials":
        attrs = node["attrs"]
        return (
            int(attrs["vocabulary_size"]) + int(attrs["vocabulary_tile_width"]) - 1
        ) // int(attrs["vocabulary_tile_width"])
    if node["op"] in {
        "argmax_candidate_reduce",
        "confidence_projection_block",
        "pack_token_block",
    }:
        return 1
    if node["op"] in {
        "hyper_connection_pre",
        "hyper_connection_post_pre",
        "hyper_connection_post",
    }:
        return 1
    if node["op"] == "anchor_noise_embedding_block":
        attrs = node["attrs"]
        output_words = (
            int(attrs["block_size"])
            * int(attrs["stream_multiplicity"])
            * int(attrs["hidden_size"])
            // 2
        )
        return (output_words + 63) // 64
    if node["op"] == "causal_conv1d_silu":
        channels = int(node["attrs"]["channels"])
        if channels <= 0 or channels % 2 != 0:
            raise ModelCompileError(
                "packed BF16 causal convolution requires a positive even channel "
                f"count, got {channels}"
            )
        channel_pairs = channels // 2
        return (channel_pairs + 63) // 64
    if node["op"] in {
        "quantize_fp8_e4m3",
        "quantize_fp8_e4m3_e8m0",
        "quantize_int8_symmetric",
        "quantize_int8_symmetric_pairpacked",
    }:
        return int(node["attrs"]["element_count"]) // int(
            node["attrs"]["block_columns"]
        )
    if node["op"] == "linear_split_recurrent_depthwise_gate":
        hidden_size = int(state_port(circuit, node["state_reads"][0])["shape"][1])
        return hidden_size // 2
    if node["op"] == "parallel_head_norm_rope_2way":
        return sum(
            int(branch["norm"]["head_count"]) for branch in node["attrs"]["branches"]
        )
    if node["op"] == "mixed_parallel_linear_4way":
        output_sizes = [
            int(
                parameter_shape_for_id(circuit, node["params"][offset], tensor_index)[0]
            )
            for offset in (0, 2)
        ]
        return max(
            (output_size + FP8_PREQUANT_TILE_ROWS - 1) // FP8_PREQUANT_TILE_ROWS
            for output_size in output_sizes
        )
    if node["op"] == "contiguous_linear_swiglu":
        part_width = int(node.get("attrs", {}).get("part_width", 0))
        if part_width <= 0 or part_width % 8:
            raise ModelCompileError(
                f"contiguous SwiGLU node {node['id']!r} has invalid output width "
                f"{part_width}"
            )
        return part_width // 8
    if (
        node["op"] == "split"
        and node.get("attrs", {}).get("layout") == "per_head_interleaved"
    ):
        attrs = node["attrs"]
        output_words = (int(attrs["blocks"]) * int(attrs["block_part_width"]) + 1) // 2
        return (
            output_words + HEAD_INTERLEAVED_SPLIT_LOCAL_SIZE - 1
        ) // HEAD_INTERLEAVED_SPLIT_LOCAL_SIZE
    if node["op"] in {"parallel_linear_2way", "parallel_linear_3way"}:
        branch_count = int(node["attrs"]["branch_count"])
        branch_parameter_counts = [
            int(count)
            for count in node["attrs"].get(
                "branch_parameter_counts", [1] * branch_count
            )
        ]
        branch_weight_ids = []
        offset = 0
        for count in branch_parameter_counts:
            branch_weight_ids.append(node["params"][offset])
            offset += count
        if {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in branch_weight_ids
        } == {"F8_E4M3"}:
            output_sizes = [
                int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0])
                for parameter_id in branch_weight_ids
            ]
            return max(
                (
                    output_size
                    + (
                        FP8_PREQUANT_TILE_ROWS
                        if uses_prequantized_fp8_input(node)
                        else fp8_linear_tile_rows(output_size)
                    )
                    - 1
                )
                // (
                    FP8_PREQUANT_TILE_ROWS
                    if uses_prequantized_fp8_input(node)
                    else fp8_linear_tile_rows(output_size)
                )
                for output_size in output_sizes
            )
        if {
            parameter_dtype_for_id(circuit, parameter_id, tensor_index)
            for parameter_id in branch_weight_ids
        } == {"Q8_0"}:
            output_sizes = [
                int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0])
                for parameter_id in branch_weight_ids
            ]
            return max(
                (output_size + Q8_0_OUTPUT_TILE_ROWS - 1) // Q8_0_OUTPUT_TILE_ROWS
                for output_size in output_sizes
            )
        return sum(
            (int(parameter_shape_for_id(circuit, parameter_id, tensor_index)[0]) + 1)
            // 2
            for parameter_id in branch_weight_ids
        )
    if node["op"] == "parallel_linear_silu_multiply":
        out_features, _ = parameter_shape_for_id(
            circuit, node["params"][0], tensor_index
        )
        if (
            parameter_dtype_for_id(circuit, node["params"][0], tensor_index)
            == "F8_E4M3"
        ):
            return (
                int(out_features)
                + (
                    FP8_PREQUANT_TILE_ROWS
                    if uses_prequantized_fp8_input(node)
                    else FP8_FUSED_FFN_TILE_ROWS
                )
                - 1
            ) // (
                FP8_PREQUANT_TILE_ROWS
                if uses_prequantized_fp8_input(node)
                else FP8_FUSED_FFN_TILE_ROWS
            )
        if parameter_dtype_for_id(circuit, node["params"][0], tensor_index) == "Q8_0":
            return (
                int(out_features) + Q8_0_OUTPUT_TILE_ROWS - 1
            ) // Q8_0_OUTPUT_TILE_ROWS
        return (int(out_features) + 1) // 2
    if node["op"] in {"linear", "linear_residual", "linear_split_3way"}:
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "I32":
            quantization_format = packed_linear_quantization_format_for_node(
                circuit, node, tensor_index
            )
            tile_rows = (
                INT4_GPTQ_OUTPUT_TILE_ROWS
                if quantization_format == "auto_gptq"
                else INT4_CT_OUTPUT_TILE_ROWS
                if quantization_format == "compressed_tensors_pack_quantized"
                else 0
            )
            if tile_rows == 0:
                raise ModelCompileError(
                    f"packed linear node {node['id']!r} has unsupported format "
                    f"{quantization_format!r}"
                )
            return (int(out_features) + tile_rows - 1) // tile_rows
        if parameter_dtype == "F8_E4M3":
            tile_rows = (
                FP8_PREQUANT_TILE_ROWS
                if uses_prequantized_fp8_input(node)
                else fp8_linear_tile_rows(int(out_features))
            )
            return (int(out_features) + tile_rows - 1) // tile_rows
        if parameter_dtype == "Q8_0":
            return (
                int(out_features) + Q8_0_OUTPUT_TILE_ROWS - 1
            ) // Q8_0_OUTPUT_TILE_ROWS
        # One workgroup collaboratively computes and packs two BF16 output rows.
        return (int(out_features) + 1) // 2
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
        "indexed_sparse_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return int(attrs["query_heads"])
    if node["op"] == "attention_partition_partials":
        attrs = node["attrs"]
        return int(attrs["query_heads"]) * int(attrs["partition_count"])
    if node["op"] == "gated_delta_step":
        return int(node["attrs"]["value_heads"])
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["heads"])
    if node["op"] == "learned_gated_kv_pool":
        return int(node["attrs"]["head_width"])
    if node["op"] == "compressed_kv_finalize":
        return 1
    if node["op"] == "conditional_append_state_update":
        return 1
    if node["op"] == "index_vector_transform":
        return int(node["attrs"]["head_count"])
    if node["op"] == "compressed_index_kv_finalize":
        return 1
    if node["op"] == "learned_index_scores":
        maximum = int(node["attrs"]["max_compressed_positions"])
        return (maximum + 255) // 256
    if node["op"] == "radix_topk_index":
        return 1
    if node["op"] == "chronological_compressed_index":
        return 1
    if node["op"] == "independent_sparse_moe_gate_up":
        attrs = node["attrs"]
        return int(attrs["experts_per_token"]) * (
            (int(attrs["intermediate_size"]) + INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS - 1)
            // INDEPENDENT_MXFP4_GATE_UP_TILE_ROWS
        )
    if node["op"] == "independent_sparse_moe_down":
        attrs = node["attrs"]
        return int(attrs["experts_per_token"]) * (
            (int(attrs["hidden_size"]) + INDEPENDENT_MXFP4_DOWN_TILE_ROWS - 1)
            // INDEPENDENT_MXFP4_DOWN_TILE_ROWS
        )
    if node["op"] == "sparse_moe_gate_up":
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "F8_E4M3":
            tile_rows = (
                FP8_SPARSE_GATE_UP_REPRESENTATION_TILE_ROWS
                if emits_sparse_moe_fp8_intermediate(node)
                else FP8_SPARSE_PREQUANT_GATE_UP_TILE_ROWS
                if uses_prequantized_fp8_input(node)
                else FP8_SPARSE_GATE_UP_TILE_ROWS
            )
            return int(attrs["experts_per_token"]) * (
                (int(attrs["intermediate_size"]) + tile_rows - 1) // tile_rows
            )
        if parameter_dtype == "I32":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["intermediate_size"]) + INT4_CT_OUTPUT_TILE_ROWS - 1)
                // INT4_CT_OUTPUT_TILE_ROWS
            )
        return int(attrs["experts_per_token"]) * (
            (int(attrs["intermediate_size"]) + 1) // 2
        )
    if node["op"] == "sparse_moe_down":
        attrs = node["attrs"]
        parameter_dtype = parameter_dtype_for_node(circuit, node, tensor_index)
        if parameter_dtype == "F8_E4M3":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["hidden_size"]) + FP8_SPARSE_DOWN_TILE_ROWS - 1)
                // FP8_SPARSE_DOWN_TILE_ROWS
            )
        if parameter_dtype == "I32":
            return int(attrs["experts_per_token"]) * (
                (int(attrs["hidden_size"]) + INT4_CT_OUTPUT_TILE_ROWS - 1)
                // INT4_CT_OUTPUT_TILE_ROWS
            )
        return int(attrs["experts_per_token"]) * ((int(attrs["hidden_size"]) + 1) // 2)
    if node["op"] == "moe_reduce":
        hidden_words = (int(node["attrs"]["hidden_size"]) + 1) // 2
        return (hidden_words + MOE_REDUCE_LOCAL_SIZE - 1) // MOE_REDUCE_LOCAL_SIZE
    if node["op"] in {
        "rms_norm_per_head",
        "rms_norm_per_head_unscaled",
        "rotary_position_embedding",
        "inverse_rotary_position_embedding",
    }:
        return int(node["attrs"]["head_count"])
    if node["op"] == "grouped_linear":
        out_features, _ = parameter_shape_for_node(circuit, node, tensor_index)
        tile_rows = fp8_linear_tile_rows(int(out_features))
        return (int(out_features) + tile_rows - 1) // tile_rows
    return 1


def local_size_x_for_node(node: Json) -> int:
    # The tiled attention kernel maps sixteen 64-wide head reductions onto one
    # workgroup. This execution geometry belongs to the compiled model package.
    if node["op"] in {
        "scaled_dot_product_attention",
        "append_scaled_dot_product_attention",
    }:
        attrs = (
            node["attrs"]["attention"]
            if node["op"] == "append_scaled_dot_product_attention"
            else node["attrs"]
        )
        return attention_workgroup_shape(int(attrs["head_width"]))[0]
    if node["op"] == "indexed_sparse_attention":
        return int(node["attrs"]["head_width"])
    if node["op"] == "gated_delta_step":
        attrs = node["attrs"]
        key_head_width = int(attrs["key_head_width"])
        value_head_width = int(attrs["value_head_width"])
        return value_head_width * gated_delta_lanes_per_value(
            key_head_width,
            value_head_width,
        )
    if node["op"] == "rg_lru_step":
        return int(node["attrs"]["block_width"])
    if node["op"] == "learned_gated_kv_pool":
        return 64
    if node["op"] == "compressed_kv_finalize":
        return int(node["attrs"]["head_width"])
    if node["op"] == "conditional_append_state_update":
        return 64
    if node["op"] in {"index_vector_transform", "compressed_index_kv_finalize"}:
        return int(node["attrs"]["head_width"])
    if node["op"] in {"learned_index_scores", "radix_topk_index"}:
        return 1024
    if node["op"] == "chronological_compressed_index":
        return 1024
    if node["op"] == "moe_reduce":
        return MOE_REDUCE_LOCAL_SIZE
    if (
        node["op"] == "split"
        and node.get("attrs", {}).get("layout") == "per_head_interleaved"
    ):
        return HEAD_INTERLEAVED_SPLIT_LOCAL_SIZE
    return 64


def local_size_x_for_shader_file(shader_file: str, node: Json) -> int:
    if shader_file.startswith(("markov_argmax_partials_", "argmax_candidate_reduce_")):
        return 256
    if (
        shader_file.startswith("independent_sparse_moe_")
        and "_mxfp4_e2m1_" in shader_file
    ):
        return 512
    if shader_file.startswith("attention_partition_partials_bf16_"):
        return attention_workgroup_shape(int(node["attrs"]["head_width"]))[0]
    if shader_file.startswith("append_gqa_attention_partition_reduce_bf16_"):
        return int(node["attrs"]["attention"]["head_width"])
    if shader_file.startswith(
        (
            "sparse_moe_gate_up_prequant_fp8_",
            "sparse_moe_gate_up_prequant_emit_intermediate_fp8_",
            "sparse_moe_down_prequant_fp8_",
        )
    ):
        return 512
    if shader_file.startswith(("sparse_moe_gate_up_fp8_", "sparse_moe_down_fp8_")):
        return 512
    if shader_file.startswith(
        (
            "rms_norm_quantize_",
            "sigmoid_multiply_quantize_",
            "silu_multiply_quantize_",
        )
    ):
        return 1024
    if shader_file.startswith(
        (
            "quantize_fp8_e4m3_",
            "quantize_int8_symmetric_",
            "quantize_int8_symmetric_pairpacked_",
        )
    ):
        return 32
    if "_prequant_fp8_e4m3_" in shader_file:
        return 1024
    if (
        re.fullmatch(
            r"(linear|linear_residual)_fp8_e4m3_(?:se8m0_)?b\d+x\d+_\d+x\d+\.comp",
            shader_file,
        )
        or re.fullmatch(
            r"parallel_linear_silu_multiply_fp8_e4m3_b\d+x\d+_\d+x\d+\.comp",
            shader_file,
        )
        or shader_file.startswith("grouped_linear_fp8_e4m3_")
    ):
        return 1024
    return local_size_x_for_node(node)


def uses_prequantized_fp8_input(node: Json) -> bool:
    return node.get("attrs", {}).get("physical_input_contract") in {
        FP8_PREQUANTIZATION_CONTRACT,
        FP8_E8M0_PREQUANTIZATION_CONTRACT,
        SPARSE_MOE_FP8_INTERMEDIATE_CONTRACT,
    }


def emits_sparse_moe_fp8_intermediate(node: Json) -> bool:
    representations = node.get("attrs", {}).get("physical_output_representations")
    return isinstance(representations, list) and any(
        isinstance(representation, dict)
        and representation.get("contract") == SPARSE_MOE_FP8_INTERMEDIATE_CONTRACT
        for representation in representations
    )


def uses_prequantized_int8_input(node: Json) -> bool:
    return node.get("attrs", {}).get("physical_input_contract") in {
        INT8_PREQUANTIZATION_CONTRACT,
        PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT,
    }


def uses_pairpacked_int8_input(node: Json) -> bool:
    return (
        node.get("attrs", {}).get("physical_input_contract")
        == PAIRPACKED_INT8_PREQUANTIZATION_CONTRACT
    )


def fp8_linear_tile_rows(output_size: int) -> int:
    if output_size <= 0:
        raise ModelCompileError("FP8 linear output width must be positive")
    for tile_rows in reversed(FP8_LINEAR_TILE_ROWS):
        if (output_size + tile_rows - 1) // tile_rows >= FP8_LINEAR_MIN_WORKGROUPS:
            return tile_rows
    return FP8_LINEAR_TILE_ROWS[0]


def validate_native_int4_shader_shape(
    shader_file: str, group_size: int, input_size: int, output_size: int
) -> None:
    if (
        group_size <= 0
        or group_size % INT4_VALUES_PER_PACKED_WORD != 0
        or input_size <= 0
        or input_size % group_size != 0
        or output_size <= 0
        or output_size % 2 != 0
    ):
        raise ModelCompileError(f"invalid native INT4 shader shape {shader_file!r}")


def int4_shader_replacements(
    *,
    operation: str,
    quantization_format: str,
    scale_dtype: str,
    batch_tile_width: int | None,
    prequantized_input: bool = False,
    pairpacked_input: bool = False,
) -> dict[str, str]:
    if operation not in {"linear", "linear_bias", "linear_residual"}:
        raise ModelCompileError(f"unsupported native INT4 operation {operation!r}")
    if quantization_format not in {"gptq", "ct"}:
        raise ModelCompileError(
            f"unsupported native INT4 quantization format {quantization_format!r}"
        )
    if scale_dtype == "f16":
        read_scale_body = (
            "    vec2 values = unpackHalf2x16(scales.words[index >> 1u]);\n"
            "    return (index & 1u) == 0u ? values.x : values.y;"
        )
    elif scale_dtype == "bf16":
        read_scale_body = "    return read_bf16_word(scales.words[index >> 1u], index);"
    else:
        raise ModelCompileError(f"unsupported native INT4 scale dtype {scale_dtype!r}")

    has_residual = operation == "linear_residual"
    has_bias = operation == "linear_bias"
    if pairpacked_input and not prequantized_input:
        raise ModelCompileError("pair-packed INT8 input must be prequantized")
    input_binding_count = 3 if pairpacked_input else 2 if prequantized_input else 1
    output_binding = input_binding_count + (1 if has_residual else 0)
    qweight_binding = output_binding + 1
    scales_binding = qweight_binding + 1
    auxiliary_binding = scales_binding + 1 if has_bias else None

    if has_residual:
        auxiliary_buffer = (
            f"layout(set = 0, binding = {input_binding_count}) "
            "readonly buffer ResidualFrames { "
            "uint words[]; } residual_frames;"
        )
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    uint index = batch_index * OUTPUT_WORDS + (row >> 1u);\n"
            "    return read_bf16_word(residual_frames.words[index], row) + value;\n"
            "}"
        )
    elif has_bias:
        auxiliary_buffer = (
            f"layout(set = 0, binding = {auxiliary_binding}) readonly buffer Bias {{ "
            "uint words[]; } bias;"
        )
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    return read_bf16_word(bias.words[row >> 1u], row) + value;\n"
            "}"
        )
    else:
        auxiliary_buffer = ""
        finalize_output = (
            "float finalize_output(uint batch_index, uint row, float value) {\n"
            "    return value;\n"
            "}"
        )

    if batch_tile_width is None:
        batch_control = ""
        batch_tile_width = 1
        batch_start = "0u"
        batch_width = "1u"
    else:
        batch_control = (
            "layout(push_constant) uniform BatchControl { uint batch_width; } "
            "batch_control;"
        )
        batch_start = "gl_WorkGroupID.y * BATCH_TILE_WIDTH"
        batch_width = "batch_control.batch_width"

    replacements = {
        "OUTPUT_BINDING": str(output_binding),
        "QWEIGHT_BINDING": str(qweight_binding),
        "SCALES_BINDING": str(scales_binding),
        "AUXILIARY_BUFFER": auxiliary_buffer,
        "FINALIZE_OUTPUT_FUNCTION": finalize_output,
        "BATCH_CONTROL": batch_control,
        "BATCH_TILE_WIDTH": str(batch_tile_width),
        "BATCH_START": batch_start,
        "BATCH_WIDTH": batch_width,
        "READ_SCALE_BODY": read_scale_body,
    }
    return replacements


def attention_workgroup_shape(head_width: int) -> tuple[int, int]:
    padded_head_width = ((head_width + 63) // 64) * 64
    physical_tile_tokens = 1024 // padded_head_width
    if physical_tile_tokens == 0:
        return 0, 0
    return (
        padded_head_width * physical_tile_tokens,
        attention_tile_token_width(head_width),
    )


def rms_norm_shader_file(hidden_size: int, eps: float, weight_offset: float) -> str:
    return (
        f"rms_norm_bf16_h{hidden_size}_eps{shader_float_token(eps)}"
        f"_offset{shader_float_token(weight_offset)}.comp"
    )


def rope_shader_suffix(attrs: Json) -> str:
    rope_type = str(attrs.get("rope_type", "default"))
    layout = (
        "proportional"
        if rope_type == "proportional"
        else "interleaved"
        if attrs.get("interleaved")
        else "half"
    )
    theta = float(attrs["theta"])
    scaling = attrs.get("scaling")
    if rope_type == "yarn":
        if not isinstance(scaling, dict) or scaling.get("type") != "yarn":
            raise ModelCompileError("YaRN RoPE node has no compiled scaling profile")
        return (
            f"theta{shader_float_token(theta)}_yarn"
            f"_f{shader_float_token(float(scaling['factor']))}"
            f"_lo{shader_float_token(float(scaling['correction_low']))}"
            f"_hi{shader_float_token(float(scaling['correction_high']))}"
            f"_a{shader_float_token(float(scaling['attention_factor']))}_{layout}"
        )
    if scaling is not None:
        raise ModelCompileError(
            f"RoPE type {rope_type!r} unexpectedly declares a scaling profile"
        )
    return f"theta{shader_float_token(theta)}_{layout}"
