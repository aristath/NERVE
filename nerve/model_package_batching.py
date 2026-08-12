from nerve.model_package_common import *


def is_sparse_moe_projection_shader(shader_file: str) -> bool:
    return shader_file.startswith(("sparse_moe_", "independent_sparse_moe_"))


def persistent_batch_control_shader_file(shader_file: str, *, binding: int) -> str:
    return shader_file.removesuffix(".comp") + f"__pbc{binding}.comp"


def persistent_batch_control_stage(
    shader_file: str,
    local_size_x: int,
    workgroup_count_x: int,
    *,
    payload: str = "width",
    binding: int = 31,
    descriptor_bindings: list[Json] | None = None,
    state_snapshot_binding: int | None = None,
    state_snapshot_source_binding: int | None = None,
    control_access: str = "read",
    indirect_dispatch_byte_offset: int | None = None,
    dispatch_y_from_batch_width: bool = False,
) -> Json:
    byte_count = {
        "width": 4,
        "width_state_snapshots": 8,
        "width_expert_start": 8,
        "width_expert_range_indirect": 28,
        "temporal": 16,
        "temporal_state_snapshots": 20,
    }[payload]
    packaged_shader_file = persistent_batch_control_shader_file(
        shader_file,
        binding=binding,
    )
    control = {
        "kind": "storage_buffer",
        "byte_count": byte_count,
        "binding": binding,
        "payload": payload,
    }
    if control_access != "read":
        control["access"] = control_access
    stage = {
        "shader_path": f"shaders/{packaged_shader_file}",
        "local_size_x": local_size_x,
        "workgroup_count_x": workgroup_count_x,
        "control": control,
    }
    if descriptor_bindings is not None:
        stage["descriptor_bindings"] = descriptor_bindings
    if state_snapshot_binding is not None:
        stage["state_snapshot_binding"] = state_snapshot_binding
    if state_snapshot_source_binding is not None:
        stage["state_snapshot_source_binding"] = state_snapshot_source_binding
    if indirect_dispatch_byte_offset is not None:
        stage["indirect_dispatch_byte_offset"] = indirect_dispatch_byte_offset
    if dispatch_y_from_batch_width:
        stage["dispatch_y_from_batch_width"] = True
    return stage


def hyper_connection_rms_norm_batch_implementations(
    shader_file: str,
) -> list[Json] | None:
    match = re.fullmatch(
        r"(hyper_connection_pre|hyper_connection_post_pre)_rms_norm_quantize_"
        r"fp8_e4m3_spow2_b(\d+)_m(\d+)_h(\d+)_i(\d+)_"
        r"neps([^_]+)_heps([^_]+)_reps([^_]+)_roffset([^_]+)\.comp",
        shader_file,
    )
    if match is None:
        return None
    (
        operation,
        block_columns,
        multiplicity,
        hidden_size,
        sinkhorn_iterations,
        normalization_epsilon,
        sinkhorn_epsilon,
        rms_epsilon,
        rms_weight_offset,
    ) = match.groups()
    post_pre = operation == "hyper_connection_post_pre"
    hyper_scalar = (
        f"{operation}_m{multiplicity}_h{hidden_size}_i{sinkhorn_iterations}_"
        f"neps{normalization_epsilon}_heps{sinkhorn_epsilon}.comp"
    )
    # The component descriptor namespace is inputs, then outputs, then parameters.
    # These maps preserve the unfused hyper kernel's interface while its reduced
    # BF16 output becomes the in-place scratch/final output for the RMS stage.
    hyper_source_bindings = (
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
        if post_pre
        else [0, 1, 2, 3, 6, 7, 8]
    )
    rms_source_bindings = (
        [5, 8, 9, 13]
        if post_pre
        else [1, 4, 5, 9]
    )
    implementations = []
    for tile_width in EXACT_BATCH_LANE_TILE_WIDTHS:
        hyper_batch = weight_shared_batch_shader_file(
            hyper_scalar,
            tile_width=tile_width,
        )
        if hyper_batch is None:
            raise ModelCompileError(
                f"fused hyper/RMS shader {shader_file!r} lost hyper batch support"
            )
        rms_batch = (
            f"rms_norm_quantize_in_place_batch{tile_width}_fp8_e4m3_spow2_"
            f"b{block_columns}_h{hidden_size}_eps{rms_epsilon}_"
            f"offset{rms_weight_offset}.comp"
        )
        implementations.append(
            {
                "execution_domain": "decode_and_prefill",
                "lane_tile_width": tile_width,
                "selection_priority": 0,
                "independent_candidate_compatible": True,
                "causal_sequence_compatible": True,
                "parallel_block_compatible": True,
                "device_requirements": {
                    "vulkan_device_extensions": [],
                    "vulkan_features": [],
                    "subgroup_operations": [],
                },
                "stages": [
                    persistent_batch_control_stage(
                        hyper_batch,
                        1024,
                        tile_width,
                        descriptor_bindings=[
                            {"binding": binding, "source_binding": source}
                            for binding, source in enumerate(hyper_source_bindings)
                        ],
                    ),
                    persistent_batch_control_stage(
                        rms_batch,
                        1024,
                        1,
                        descriptor_bindings=[
                            {"binding": binding, "source_binding": source}
                            for binding, source in enumerate(rms_source_bindings)
                        ],
                    ),
                ],
            }
        )
    return implementations


def frame_parallel_batch_shader_file(shader_file: str) -> str | None:
    if re.fullmatch(
        r"independent_sparse_moe_(?:gate_up|down)(?:_prequant)?"
        r"(?:_input_block_major_b128)?_mxfp4_e2m1"
        r"(?:_(?:resident|adaptive)_fp8_e4m3)?_g32"
        r"(?:_native_fp8_e4m3_se8m0_b128_nf\d+)?_"
        r"h\d+_i\d+_e\d+_k\d+(?:_limit[0-9eE+.-]+)?\.comp",
        shader_file,
    ):
        return re.sub(
            r"^(independent_sparse_moe_(?:gate_up|down))_",
            r"\1_batch1_",
            shader_file,
            count=1,
        )
    if re.fullmatch(
        r"moe_router_(?:score_topk|preselected)_"
        r"(?:sigmoid|softmax|sqrtsoftplus)_bf16_r\d+_k\d+_"
        r"a\d+w[0-9eE+.-]+_"
        r"norm[01]_scale[0-9eE+.-]+"
        r"(?:_bias(?:f32|bf16))?\.comp",
        shader_file,
    ):
        return shader_file.replace("moe_router_", "moe_router_batch1_", 1)
    if re.fullmatch(
        r"resource_preselect_table_r\d+_k\d+_a\d+_v\d+_tablei(?:32|64)\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "resource_preselect_", "resource_preselect_batch1_", 1
        )
    if re.fullmatch(r"moe_topk_bf16_e\d+_k\d+\.comp", shader_file):
        return shader_file.replace("moe_topk_", "moe_topk_batch1_", 1)
    if re.fullmatch(
        r"moe_topk_(?:sigmoid|softmax)_bf16_e\d+_k\d+_norm[01]_"
        r"cap[0-9eE+.-]+_(?:biasf32|biasbf16|nobias)\.comp",
        shader_file,
    ):
        return shader_file.replace("moe_topk_", "moe_topk_batch1_", 1)
    if re.fullmatch(
        r"sparse_moe_(?:gate_up|down)_(?:bf16|"
        r"(?:prequant_)?(?:emit_intermediate_)?fp8_e4m3_b\d+x\d+|"
        r"int4_ct_s(?:f16|bf16)_g\d+)_"
        r"h\d+_i\d+_e\d+_k\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"^(sparse_moe_(?:gate_up|down))_",
            r"\1_batch1_",
            shader_file,
            count=1,
        )
    if re.fullmatch(
        r"parallel_linear_[23]way_fp8_e4m3(?:_se8m0)?_"
        r"b\d+x\d+_\d+x\d+_\d+(?:_\d+)?\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "parallel_linear_",
            "parallel_linear_batch1_",
            1,
        )
    if re.fullmatch(r"moe_reduce_bf16_h\d+_k\d+_scale[0-9eE+.-]+\.comp", shader_file):
        return shader_file.replace("moe_reduce_", "moe_reduce_batch1_", 1)
    if re.fullmatch(
        r"rms_norm_batch\d+_bf16_h\d+_eps[0-9eE+.-]+_offset[0-9eE+.-]+\.comp",
        shader_file,
    ) or re.fullmatch(
        r"split_batch\d+_bf16_2x\d+x\d+_head_interleaved\.comp",
        shader_file,
    ):
        return re.sub(r"_batch\d+_", "_batch1_", shader_file, count=1)
    return None


def parallel_block_attention_stages(
    shader_file: str,
    local_size_x: int,
    workgroup_count_x: int,
) -> list[Json] | None:
    match = re.fullmatch(
        r"indexed_sparse_attention_bf16_q\d+_kv\d+_d\d+_w\d+_"
        r"scale[0-9eE+.-]+__sc(\d+)\.comp",
        shader_file,
    )
    if match is None:
        return None
    control_binding = int(match.group(1))
    parallel_shader = re.sub(
        r"^indexed_sparse_attention_",
        "indexed_sparse_attention_parallel_",
        shader_file,
        count=1,
    )
    parallel_shader = re.sub(r"__sc\d+\.comp$", ".comp", parallel_shader)
    return [
        persistent_batch_control_stage(
            parallel_shader,
            local_size_x,
            workgroup_count_x,
            payload="temporal",
            binding=control_binding,
            dispatch_y_from_batch_width=True,
        )
    ]


def sparse_moe_route_scheduling_shader_file(shader_file: str) -> str | None:
    independent = re.fullmatch(
        r"independent_sparse_moe_(gate_up|down)(?:_prequant)?"
        r"(?:_input_block_major_b128)?_mxfp4_e2m1"
        r"(?:_(?:resident|adaptive)_fp8_e4m3)?_g32"
        r"(?:_native_fp8_e4m3_se8m0_b128_nf\d+)?_"
        r"h(\d+)_i(\d+)_e\d+_k(\d+)(?:_limit[0-9eE+.-]+)?\.comp",
        shader_file,
    )
    if independent is not None:
        stage, hidden_size, intermediate_size, experts_per_token = independent.groups()
        output_rows = int(intermediate_size) if stage == "gate_up" else int(hidden_size)
        tile_rows = 32 if stage == "gate_up" else 64
        operation = "compact" if stage == "gate_up" else "count"
        parameters_per_resource = 4 if stage == "gate_up" else 2
        return (
            f"moe_route_{operation}_selected_p{parameters_per_resource}_batch1_"
            f"i{intermediate_size}_"
            f"k{experts_per_token}_t{(output_rows + tile_rows - 1) // tile_rows}.comp"
        )
    match = re.fullmatch(
        r"sparse_moe_(gate_up|down)_(?:bf16|"
        r"(?:prequant_)?(?:emit_intermediate_)?fp8_e4m3_b\d+x\d+|"
        r"int4_ct_s(?:f16|bf16)_g\d+)_"
        r"h(\d+)_i(\d+)_e\d+_k(\d+)\.comp",
        shader_file,
    )
    if match is None:
        return None
    stage, hidden_size, intermediate_size, experts_per_token = match.groups()
    output_rows = int(intermediate_size) if stage == "gate_up" else int(hidden_size)
    tile_rows = (
        FP8_SPARSE_GATE_UP_REPRESENTATION_TILE_ROWS
        if stage == "gate_up" and "_emit_intermediate_fp8_" in shader_file
        else FP8_SPARSE_PREQUANT_GATE_UP_TILE_ROWS
        if stage == "gate_up" and "_prequant_fp8_" in shader_file
        else FP8_SPARSE_GATE_UP_TILE_ROWS
        if stage == "gate_up" and "_fp8_" in shader_file
        else FP8_SPARSE_DOWN_TILE_ROWS
        if stage == "down" and "_fp8_" in shader_file
        else INT4_CT_OUTPUT_TILE_ROWS
        if "_int4_ct_" in shader_file
        else 2
    )
    tiles_per_route = (output_rows + tile_rows - 1) // tile_rows
    operation = "compact" if stage == "gate_up" else "count"
    return (
        f"moe_route_{operation}_batch1_i{intermediate_size}_"
        f"k{experts_per_token}_t{tiles_per_route}.comp"
    )


def sparse_moe_route_scheduling_workgroup_count_x(shader_file: str) -> int:
    match = re.fullmatch(
        r"moe_route_(compact|count)(?:_selected_p\d+)?_batch1_"
        r"i\d+_k(\d+)_t\d+\.comp",
        shader_file,
    )
    if match is None:
        raise ModelCompileError(
            f"shader {shader_file!r} is not a sparse route-scheduling kernel"
        )
    return int(match.group(2)) if match.group(1) == "compact" else 1


def sparse_moe_route_scheduling_descriptor_bindings(node: Json) -> list[Json]:
    if node.get("op") not in {
        "sparse_moe_gate_up",
        "sparse_moe_down",
        "independent_sparse_moe_gate_up",
        "independent_sparse_moe_down",
    }:
        raise ModelCompileError(
            "sparse route scheduling requires a sparse MoE projection node"
        )
    inputs = node.get("inputs")
    outputs = node.get("outputs")
    if (
        not isinstance(inputs, list)
        or len(inputs) < 2
        or not all(isinstance(signal, str) and signal for signal in inputs)
        or not isinstance(outputs, list)
        or not outputs
        or not all(isinstance(signal, str) and signal for signal in outputs)
    ):
        raise ModelCompileError(
            f"sparse gate node {node.get('id')!r} has no valid descriptor interface"
        )
    bindings = [
        {
            "binding": 1,
            "source_binding": len(inputs) - 1,
        },
    ]
    if node["op"] in {"sparse_moe_gate_up", "independent_sparse_moe_gate_up"}:
        bindings.append(
            {
                "binding": 2,
                "source_binding": len(inputs),
            }
        )
    if node["op"] in {
        "independent_sparse_moe_gate_up",
        "independent_sparse_moe_down",
    }:
        # Independent resources are not required to form contiguous owner
        # ranges after runtime placement. The scheduling stage consumes the
        # same device-local dynamic tables as the projection and filters the
        # routed worklist by exact resident ownership.
        bindings.extend(
            [
                {"binding": 3, "source_binding": len(inputs) + 2},
            ]
        )
    return bindings


def causal_scan_batch_stages(shader_file: str, local_size_x: int) -> list[Json] | None:
    causal_scan_shader = causal_scan_batch_shader_file(shader_file)
    if causal_scan_shader is not None:
        captures_static_state = causal_scan_shader.startswith(
            "causal_conv1d_silu_temporal_"
        ) or causal_scan_shader.startswith(
            (
                "gated_delta_scan_",
                "rolling_state_ring_append_temporal_",
                "learned_gated_kv_pool_temporal_",
            )
        )
        reads_static_state_snapshot = causal_scan_shader.startswith(
            (
                "indexed_sparse_attention_main_temporal_",
                "indexed_sparse_attention_main_score_pipeline_temporal_",
                "indexed_sparse_attention_main_tile_overlap_temporal_",
            )
        )
        temporal_state_snapshot_control = (
            captures_static_state
            and causal_scan_shader.startswith(
                (
                    "rolling_state_ring_append_temporal_",
                    "learned_gated_kv_pool_temporal_",
                )
            )
        ) or reads_static_state_snapshot
        source_stream_control = re.search(r"__sc(\d+)\.comp$", shader_file)
        temporal_binding = (
            5
            if causal_scan_shader.startswith(
                "parallel_mixed_head_norm_rope_2way_qdq_fp8_e4m3_temporal_"
            )
            else
            6
            if causal_scan_shader.startswith("parallel_head_norm_rope_2way_temporal_")
            else 2
            if causal_scan_shader.startswith(
                (
                    "rotary_temporal_",
                    "inverse_rotary_temporal_",
                    "rotary_qdq_fp8_e4m3_",
                )
            )
            else None
        )
        if temporal_binding is None and source_stream_control is not None:
            temporal_binding = int(source_stream_control.group(1))
        return [
            persistent_batch_control_stage(
                causal_scan_shader,
                local_size_x,
                causal_scan_workgroup_count_x(shader_file),
                payload=(
                    "temporal_state_snapshots"
                    if temporal_state_snapshot_control
                    else "temporal"
                ),
                binding=temporal_binding,
                state_snapshot_binding=(
                    30 if captures_static_state or reads_static_state_snapshot else None
                ),
                state_snapshot_source_binding=(
                    1 if reads_static_state_snapshot else None
                ),
                dispatch_y_from_batch_width=causal_scan_shader.startswith(
                    (
                        "indexed_sparse_attention_main_temporal_parallel_",
                        "indexed_sparse_attention_main_score_pipeline_temporal_parallel_",
                        "indexed_sparse_attention_main_tile_overlap_temporal_parallel_",
                    )
                ),
            )
            if temporal_binding is not None
            else persistent_batch_control_stage(
                causal_scan_shader,
                local_size_x,
                causal_scan_workgroup_count_x(shader_file),
                payload=("width_state_snapshots" if captures_static_state else "width"),
                state_snapshot_binding=30 if captures_static_state else None,
            )
        ]

    partition_partials = re.fullmatch(
        r"attention_partition_partials_bf16_q(\d+)_kv(\d+)_d(\d+)_s(\d+)"
        r"_scale([0-9eE+.-]+)(?:_w(\d+))?__sc\d+\.comp",
        shader_file,
    )
    if partition_partials is not None:
        query_heads, _kv_heads, _head_width, partition_count = map(
            int, partition_partials.groups()[:4]
        )
        stem = re.sub(r"__sc\d+\.comp$", ".comp", shader_file).replace(
            "attention_partition_partials_bf16_",
            "attention_partition_partials_temporal_bf16_",
            1,
        )
        return [
            persistent_batch_control_stage(
                stem,
                local_size_x,
                query_heads * partition_count,
                payload="temporal",
                binding=6,
                dispatch_y_from_batch_width=True,
            )
        ]

    partition_reduce = re.fullmatch(
        r"append_gqa_attention_partition_reduce_bf16_"
        r"q(\d+)_kv(\d+)_d(\d+)_s(\d+)_scale([0-9eE+.-]+)"
        r"(?:_w(\d+))?(_sinks)?__sc\d+\.comp",
        shader_file,
    )
    if partition_reduce is not None:
        query_heads, kv_heads, head_width, _partition_count = map(
            int, partition_reduce.groups()[:4]
        )
        stem = re.sub(r"__sc\d+\.comp$", ".comp", shader_file).replace(
            "append_gqa_attention_partition_reduce_bf16_",
            "append_gqa_attention_partition_reduce_temporal_bf16_",
            1,
        )
        sinks = "_sinks" if partition_reduce.group(7) else ""
        attention_window = partition_reduce.group(6) or "0"
        control_binding = 8 if sinks else 7
        return [
            persistent_batch_control_stage(
                stem,
                head_width,
                query_heads,
                payload="temporal",
                binding=control_binding,
                dispatch_y_from_batch_width=True,
            ),
            persistent_batch_control_stage(
                (
                    "append_kv_temporal_commit_bf16_"
                    f"kv{kv_heads}_d{head_width}_w{attention_window}{sinks}.comp"
                ),
                64,
                kv_heads,
                payload="temporal",
                binding=control_binding,
            ),
        ]

    attention = re.fullmatch(
        r"append_gqa_attention_bf16_q(\d+)_kv(\d+)_d(\d+)_scale([0-9eE+.-]+)"
        r"(?:_w(\d+))?(_sinks)?__sc\d+\.comp",
        shader_file,
    )
    if attention is None:
        return None
    query_heads, kv_heads, head_width = map(int, attention.groups()[:3])
    stem = re.sub(r"__sc\d+\.comp$", ".comp", shader_file).replace(
        "append_gqa_attention_bf16_",
        "append_gqa_attention_temporal_read_bf16_",
        1,
    )
    sinks = "_sinks" if attention.group(6) else ""
    attention_window = attention.group(5) or "0"
    return [
        persistent_batch_control_stage(
            stem,
            local_size_x,
            query_heads * CAUSAL_SCAN_LANE_TILE_WIDTH,
            payload="temporal",
            binding=8 if sinks else 7,
        ),
        persistent_batch_control_stage(
            (
                "append_kv_temporal_commit_bf16_"
                f"kv{kv_heads}_d{head_width}_w{attention_window}{sinks}.comp"
            ),
            64,
            kv_heads,
            payload="temporal",
            binding=8 if sinks else 7,
        ),
    ]


def cooperative_bfloat16_batch_shader_file(shader_file: str) -> str | None:
    linear = re.fullmatch(
        r"(linear|linear_residual)_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if linear is not None:
        operation, input_size, output_size = linear.groups()
        if (
            int(input_size) <= 0
            or int(input_size) % 2
            or int(output_size) <= 0
            or int(output_size) % 2
        ):
            return None
        return f"{operation}_batch64_cooperative_bf16_{input_size}x{output_size}.comp"
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_\d+x.+\.comp",
        shader_file,
    )
    if parallel is not None:
        return shader_file.replace(
            "parallel_linear_",
            "parallel_linear_batch64_cooperative_",
            1,
        )
    fused = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_\d+x\d+\.comp",
        shader_file,
    )
    if fused is not None:
        return shader_file.replace(
            "parallel_linear_silu_multiply_",
            "parallel_linear_silu_multiply_batch64_cooperative_",
            1,
        )
    int4 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_int4_(gptq|ct)_"
        r"s(f16|bf16)_g(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if int4 is not None:
        (
            operation,
            quantization_format,
            scale_dtype,
            group_size,
            input_size,
            output_size,
        ) = int4.groups()
        if int(group_size) % COOPERATIVE_BFLOAT16_SHAPE[2] or int(input_size) % int(
            group_size
        ):
            return None
        return (
            f"{operation}_batch64_cooperative_int4_{quantization_format}_"
            f"s{scale_dtype}_g{group_size}_{input_size}x{output_size}.comp"
        )
    return None


def cooperative_bfloat16_workgroup_count_x(shader_file: str) -> int:
    linear = re.fullmatch(
        r"(?:linear|linear_residual)_bf16_\d+x(\d+)\.comp",
        shader_file,
    )
    if linear is not None:
        return (
            int(linear.group(1)) + COOPERATIVE_OUTPUT_TILE_WIDTH - 1
        ) // COOPERATIVE_OUTPUT_TILE_WIDTH
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_\d+x"
        r"(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel is not None:
        return sum(
            (int(width) + COOPERATIVE_OUTPUT_TILE_WIDTH - 1)
            // COOPERATIVE_OUTPUT_TILE_WIDTH
            for width in parallel.groups()
            if width is not None
        )
    fused = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_"
        r"\d+x(\d+)\.comp",
        shader_file,
    )
    if fused is not None:
        return (
            int(fused.group(1)) + COOPERATIVE_FUSED_OUTPUT_TILE_WIDTH - 1
        ) // COOPERATIVE_FUSED_OUTPUT_TILE_WIDTH
    int4 = re.fullmatch(
        r"(?:linear|linear_bias|linear_residual)_int4_(?:gptq|ct)_"
        r"s(?:f16|bf16)_g\d+_\d+x(\d+)\.comp",
        shader_file,
    )
    if int4 is not None:
        return (
            int(int4.group(1)) + COOPERATIVE_OUTPUT_TILE_WIDTH - 1
        ) // COOPERATIVE_OUTPUT_TILE_WIDTH
    raise ModelCompileError(
        f"shader {shader_file!r} has no cooperative BF16 batch geometry"
    )


def cooperative_float8_e4m3_batch_shader_file(
    shader_file: str,
    *,
    shape: tuple[int, int, int],
    batch_tile_multiplier: int = 4,
) -> str | None:
    linear = re.fullmatch(
        r"(linear|linear_residual)_prequant_fp8_e4m3(_se8m0)?_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    m, n, k = shape
    if min(shape) <= 0:
        return None
    if batch_tile_multiplier not in {1, 4}:
        raise ModelCompileError(
            "cooperative FP8 batch tile multiplier must be 1 or 4"
        )
    batch_tile_width = batch_tile_multiplier * n
    if linear is not None:
        (
            operation,
            scale_suffix,
            block_rows,
            block_columns,
            input_size,
            output_size,
        ) = linear.groups()
        if int(block_columns) % k or int(input_size) % int(block_columns):
            return None
        return (
            f"{operation}_prequant_batch{batch_tile_width}_cooperative_"
            f"fp8_e4m3{scale_suffix or ''}_m{m}n{n}k{k}_"
            f"b{block_rows}x{block_columns}_"
            f"{input_size}x{output_size}.comp"
        )

    parallel = re.fullmatch(
        r"parallel_linear_([23])way_prequant_fp8_e4m3(_se8m0)?_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)_(\d+)(?:_(\d+))?\.comp",
        shader_file,
    )
    if parallel is not None:
        (
            branch_count,
            scale_suffix,
            block_rows,
            block_columns,
            input_size,
            *output_sizes,
        ) = parallel.groups()
        output_sizes = [size for size in output_sizes if size is not None]
        if (
            len(output_sizes) != int(branch_count)
            or int(block_columns) % k
            or int(input_size) % int(block_columns)
        ):
            return None
        return (
            f"parallel_linear_batch{batch_tile_width}_{branch_count}way_"
            f"prequant_cooperative_fp8_e4m3{scale_suffix or ''}_m{m}n{n}k{k}_"
            f"b{block_rows}x{block_columns}_{input_size}x"
            f"{'_'.join(output_sizes)}.comp"
        )

    fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn is not None:
        block_rows, block_columns, input_size, output_size = fused_ffn.groups()
        if int(block_columns) % k or int(input_size) % int(block_columns):
            return None
        return (
            "parallel_linear_silu_multiply_prequant_"
            f"batch{batch_tile_width}_cooperative_fp8_e4m3_"
            f"m{m}n{n}k{k}_b{block_rows}x{block_columns}_"
            f"{input_size}x{output_size}.comp"
        )

    contiguous_swiglu = re.fullmatch(
        r"contiguous_linear_swiglu_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if contiguous_swiglu is not None:
        block_rows, block_columns, input_size, output_size = contiguous_swiglu.groups()
        if (
            int(block_columns) % k
            or int(input_size) % int(block_columns)
            or int(output_size) % (2 * m)
        ):
            return None
        return (
            "contiguous_linear_swiglu_prequant_"
            f"batch{batch_tile_width}_cooperative_fp8_e4m3_"
            f"m{m}n{n}k{k}_b{block_rows}x{block_columns}_"
            f"{input_size}x{output_size}.comp"
        )

    return None


COMPACT_COOPERATIVE_FP8_MIN_WORKGROUP_COUNT_X = 256


def compact_cooperative_float8_e4m3_batch_shader_file(
    shader_file: str,
    *,
    shape: tuple[int, int, int],
) -> str | None:
    """Return the exact small-batch matrix path only at proven occupancy.

    A compact matrix tile reuses each staged weight across an entire verifier
    block, but its fixed cooperative-matrix setup loses to the vector kernel
    when the output grid cannot occupy the device.  The conservative grid
    floor is deliberately expressed in physical workgroups rather than model
    names or layer roles; both sides of the crossover are hardware-tested.
    """

    if re.fullmatch(
        r"(?:linear|linear_residual)_prequant_fp8_e4m3(?:_se8m0)?_"
        r"b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    ) is None:
        return None
    if (
        cooperative_float8_e4m3_workgroup_count_x(shader_file, shape=shape)
        < COMPACT_COOPERATIVE_FP8_MIN_WORKGROUP_COUNT_X
    ):
        return None
    return cooperative_float8_e4m3_batch_shader_file(
        shader_file,
        shape=shape,
        batch_tile_multiplier=1,
    )


def cooperative_float8_e4m3_workgroup_count_x(
    shader_file: str,
    *,
    shape: tuple[int, int, int],
) -> int:
    contiguous_swiglu_geometry = False
    linear = re.fullmatch(
        r"(?:linear|linear_residual)_prequant_fp8_e4m3(?:_se8m0)?_"
        r"b\d+x\d+_\d+x(\d+)\.comp",
        shader_file,
    )
    output_sizes: list[int]
    if linear is not None:
        output_sizes = [int(linear.group(1))]
    else:
        parallel = re.fullmatch(
            r"parallel_linear_[23]way_prequant_fp8_e4m3(?:_se8m0)?_"
            r"b\d+x\d+_\d+x(\d+)_(\d+)(?:_(\d+))?\.comp",
            shader_file,
        )
        if parallel is not None:
            output_sizes = [int(size) for size in parallel.groups() if size is not None]
        else:
            fused_ffn = re.fullmatch(
                r"parallel_linear_silu_multiply_prequant_fp8_e4m3_"
                r"b\d+x\d+_\d+x(\d+)\.comp",
                shader_file,
            )
            if fused_ffn is not None:
                output_sizes = [int(fused_ffn.group(1))]
            else:
                contiguous_swiglu = re.fullmatch(
                    r"contiguous_linear_swiglu_prequant_fp8_e4m3_"
                    r"b\d+x\d+_\d+x(\d+)\.comp",
                    shader_file,
                )
                if contiguous_swiglu is None:
                    raise ModelCompileError(
                        f"shader {shader_file!r} has no cooperative FP8 batch geometry"
                    )
                output_sizes = [int(contiguous_swiglu.group(1))]
                contiguous_swiglu_geometry = True
    if not output_sizes:
        raise ModelCompileError(
            f"shader {shader_file!r} has no cooperative FP8 batch geometry"
        )
    output_tile = (2 if contiguous_swiglu_geometry else 4) * shape[0]
    output_size = max(output_sizes)
    return (output_size + output_tile - 1) // output_tile


def causal_scan_batch_shader_file(shader_file: str) -> str | None:
    if re.fullmatch(r"causal_conv1d_silu_bf16_c\d+_k\d+\.comp", shader_file):
        return shader_file.replace(
            "causal_conv1d_silu_bf16_",
            "causal_conv1d_silu_temporal_bf16_",
            1,
        )
    if re.fullmatch(
        r"gated_delta_step_k\d+x\d+_v\d+x\d+"
        r"_a(?:f32|bf16)_dt(?:f32|bf16)_n(?:f32|bf16)_eps[0-9eE+.-]+"
        r"(?:_qfp8b\d+)?\.comp",
        shader_file,
    ):
        return shader_file.replace("gated_delta_step_", "gated_delta_scan_", 1)
    if re.fullmatch(
        r"parallel_mixed_head_norm_rope_2way_qdq_fp8_e4m3_(?:spow2|sexact)_b\d+_"
        r"bf16_h\d+_\d+_d\d+_r\d+_qeps[0-9eE+.-]+_keps[0-9eE+.-]+"
        r"_koffset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved)_tail__sc\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"__sc\d+\.comp$",
            ".comp",
            shader_file.replace("_bf16_", "_temporal_bf16_", 1),
        )
    if re.fullmatch(
        r"parallel_head_norm_rope_2way_bf16_h\d+_\d+_d\d+_r\d+"
        r"_eps[0-9eE+.-]+_offset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)__sc\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"__sc\d+\.comp$",
            ".comp",
            shader_file.replace(
                "parallel_head_norm_rope_2way_",
                "parallel_head_norm_rope_2way_temporal_",
                1,
            ),
        )
    if re.fullmatch(
        r"rotary_qdq_fp8_e4m3_(?:spow2|sexact)_b\d+_bf16_"
        r"\d+x\d+_r\d+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved)_tail(?:_po-?\d+)?__sc\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"__sc\d+\.comp$",
            ".comp",
            shader_file.replace("_bf16_", "_temporal_bf16_", 1),
        )
    if re.fullmatch(
        r"(?:inverse_)?rotary_bf16_\d+x\d+_r\d+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)(?:_tail)?(?:_po-?\d+)?__sc\d+\.comp",
        shader_file,
    ):
        return re.sub(
            r"__sc\d+\.comp$",
            ".comp",
            shader_file.replace("rotary_bf16_", "rotary_temporal_bf16_", 1),
        )
    temporal_latent_patterns = (
        (
            r"rolling_state_ring_append_bf16_\d+x\d+__sc\d+\.comp",
            "rolling_state_ring_append_",
            "rolling_state_ring_append_temporal_",
        ),
        (
            r"learned_gated_kv_pool_bf16_f32_h\d+_d\d+_r\d+_c[12]__sc\d+\.comp",
            "learned_gated_kv_pool_",
            "learned_gated_kv_pool_temporal_",
        ),
        (
            r"compressed_kv_finalize_f32_bf16_.+__sc\d+\.comp",
            "compressed_kv_finalize_",
            "compressed_kv_finalize_temporal_",
        ),
        (
            r"conditional_append_state_bf16_d\d+_p\d+__sc\d+\.comp",
            "conditional_append_state_",
            "conditional_append_state_temporal_",
        ),
        (
            r"index_vector_transform_bf16_.+__sc\d+\.comp",
            "index_vector_transform_",
            "index_vector_transform_temporal_",
        ),
        (
            r"compressed_index_kv_finalize_f32_bf16_.+__sc\d+\.comp",
            "compressed_index_kv_finalize_",
            "compressed_index_kv_finalize_temporal_",
        ),
        (
            r"learned_index_scores_bf16_f32_.+__sc\d+\.comp",
            "learned_index_scores_",
            "learned_index_scores_temporal_",
        ),
        (
            r"radix_topk_index_f32_u32_.+__sc\d+\.comp",
            "radix_topk_index_",
            "radix_topk_index_temporal_",
        ),
        (
            r"chronological_compressed_index_u32_.+__sc\d+\.comp",
            "chronological_compressed_index_",
            "chronological_compressed_index_temporal_",
        ),
        (
            r"indexed_sparse_attention_main_tile_overlap_bf16_q\d+_kv1_.+__sc\d+\.comp",
            "indexed_sparse_attention_main_tile_overlap_",
            "indexed_sparse_attention_main_tile_overlap_temporal_parallel_",
        ),
        (
            r"indexed_sparse_attention_main_score_pipeline_bf16_q\d+_kv1_.+__sc\d+\.comp",
            "indexed_sparse_attention_main_score_pipeline_",
            "indexed_sparse_attention_main_score_pipeline_temporal_parallel_",
        ),
        (
            r"indexed_sparse_attention_main_bf16_q\d+_kv1_.+__sc\d+\.comp",
            "indexed_sparse_attention_main_",
            "indexed_sparse_attention_main_temporal_parallel_",
        ),
    )
    for pattern, source_prefix, temporal_prefix in temporal_latent_patterns:
        if re.fullmatch(pattern, shader_file):
            return re.sub(r"__sc\d+\.comp$", ".comp", shader_file).replace(
                source_prefix,
                temporal_prefix,
                1,
            )
    return None


def causal_scan_workgroup_count_x(shader_file: str) -> int:
    causal_conv = re.fullmatch(
        r"causal_conv1d_silu_bf16_c(\d+)_k\d+\.comp", shader_file
    )
    if causal_conv is not None:
        channels = int(causal_conv.group(1))
        return (channels + 127) // 128
    gated_delta = re.fullmatch(
        r"gated_delta_step_k\d+x\d+_v(\d+)x\d+"
        r"_a(?:f32|bf16)_dt(?:f32|bf16)_n(?:f32|bf16)_eps[0-9eE+.-]+"
        r"(?:_qfp8b\d+)?\.comp",
        shader_file,
    )
    if gated_delta is not None:
        return int(gated_delta.group(1))
    mixed_head_norm_rope = re.fullmatch(
        r"parallel_mixed_head_norm_rope_2way_qdq_fp8_e4m3_(?:spow2|sexact)_b\d+_"
        r"bf16_h(\d+)_(\d+)_d\d+_r\d+_qeps[0-9eE+.-]+_keps[0-9eE+.-]+"
        r"_koffset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved)_tail__sc\d+\.comp",
        shader_file,
    )
    if mixed_head_norm_rope is not None:
        return int(mixed_head_norm_rope.group(1)) + int(
            mixed_head_norm_rope.group(2)
        )
    head_norm_rope = re.fullmatch(
        r"parallel_head_norm_rope_2way_bf16_h(\d+)_(\d+)_d\d+_r\d+"
        r"_eps[0-9eE+.-]+_offset[0-9eE+.-]+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)__sc\d+\.comp",
        shader_file,
    )
    if head_norm_rope is not None:
        return int(head_norm_rope.group(1)) + int(head_norm_rope.group(2))
    rotary = re.fullmatch(
        r"(?:(?:inverse_)?rotary_bf16_|rotary_qdq_fp8_e4m3_(?:spow2|sexact)_b\d+_bf16_)"
        r"(\d+)x\d+_r\d+_theta[0-9eE+.-]+"
        r"(?:_yarn_f[0-9eE+.-]+_lo[0-9eE+.-]+_hi[0-9eE+.-]+_a[0-9eE+.-]+)?"
        r"_(?:half|interleaved|proportional)(?:_tail)?(?:_po-?\d+)?__sc\d+\.comp",
        shader_file,
    )
    if rotary is not None:
        return int(rotary.group(1))
    rolling = re.fullmatch(
        r"rolling_state_ring_append_bf16_\d+x\d+__sc\d+\.comp",
        shader_file,
    )
    if rolling is not None:
        return 1
    pool = re.fullmatch(
        r"learned_gated_kv_pool_bf16_f32_h\d+_d(\d+)_r\d+_c[12]__sc\d+\.comp",
        shader_file,
    )
    if pool is not None:
        return int(pool.group(1))
    index_transform = re.fullmatch(
        r"index_vector_transform_bf16_h(\d+)_d\d+_.+__sc\d+\.comp",
        shader_file,
    )
    if index_transform is not None:
        return int(index_transform.group(1))
    index_scores = re.fullmatch(
        r"learned_index_scores_bf16_f32_h\d+_d\d+_r\d+_m(\d+)_c(\d+)_"
        r"scale[0-9eE+.-]+__sc\d+\.comp",
        shader_file,
    )
    if index_scores is not None:
        maximum, chunk = map(int, index_scores.groups())
        return (maximum + chunk - 1) // chunk
    indexed_attention = re.fullmatch(
        r"indexed_sparse_attention_main(?:_score_pipeline|_tile_overlap)?_bf16_"
        r"q(\d+)_kv1_d\d+_.+__sc\d+\.comp",
        shader_file,
    )
    if indexed_attention is not None:
        return int(indexed_attention.group(1))
    if any(
        re.fullmatch(pattern, shader_file)
        for pattern in (
            r"compressed_kv_finalize_f32_bf16_.+__sc\d+\.comp",
            r"conditional_append_state_bf16_d\d+_p\d+__sc\d+\.comp",
            r"compressed_index_kv_finalize_f32_bf16_.+__sc\d+\.comp",
            r"radix_topk_index_f32_u32_.+__sc\d+\.comp",
            r"chronological_compressed_index_u32_.+__sc\d+\.comp",
        )
    ):
        return 1
    raise ModelCompileError(f"shader {shader_file!r} is not a causal scan kernel")


def weight_shared_batch_shader_file(
    shader_file: str, *, tile_width: int = SCALAR_BATCH_LANE_TILE_WIDTH
) -> str | None:
    if tile_width <= 0:
        raise ValueError("batch tile width must be positive")
    tile = tile_width
    hyper_pre = re.fullmatch(
        r"(hyper_connection_pre|hyper_connection_post_pre)_m\d+_h\d+_i\d+_"
        r"neps[0-9eE+.-]+_heps[0-9eE+.-]+\.comp",
        shader_file,
    )
    if hyper_pre is not None:
        return shader_file.replace(
            hyper_pre.group(1) + "_",
            hyper_pre.group(1) + f"_batch{tile}_",
            1,
        )
    if re.fullmatch(r"hyper_connection_post_m\d+_h\d+\.comp", shader_file):
        return shader_file.replace(
            "hyper_connection_post_",
            f"hyper_connection_post_batch{tile}_",
            1,
        )
    if re.fullmatch(
        r"rms_norm_per_head_unscaled_bf16_\d+x\d+_eps[0-9eE+.-]+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "rms_norm_per_head_unscaled_bf16_",
            f"rms_norm_per_head_unscaled_batch{tile}_bf16_",
            1,
        )
    if re.fullmatch(
        r"grouped_linear_fp8_e4m3_(?:se8m0_)?b\d+x\d+_g\d+_\d+x\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "grouped_linear_fp8_e4m3_",
            f"grouped_linear_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"bounded_silu_multiply_bf16_\d+_limit[0-9eE+.-]+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "bounded_silu_multiply_bf16_",
            f"bounded_silu_multiply_batch{tile}_bf16_",
            1,
        )
    if re.fullmatch(r"quantize_int8_symmetric_b32_h\d+\.comp", shader_file):
        return shader_file.replace(
            "quantize_int8_symmetric_",
            f"quantize_batch{tile}_int8_symmetric_",
            1,
        )
    if re.fullmatch(r"quantize_int8_symmetric_pairpacked_b32_h\d+\.comp", shader_file):
        return shader_file.replace(
            "quantize_int8_symmetric_pairpacked_",
            f"quantize_batch{tile}_int8_symmetric_pairpacked_",
            1,
        )
    if re.fullmatch(r"quantize_fp8_e4m3(?:_spow2)?_b128_h\d+\.comp", shader_file):
        return shader_file.replace(
            "quantize_fp8_e4m3_",
            f"quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"rms_norm_quantize_fp8_e4m3(?:_spow2)?_b128_h\d+_"
        r"eps[0-9eE+.-]+_offset[0-9eE+.-]+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "rms_norm_quantize_fp8_e4m3_",
            f"rms_norm_quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"rms_norm_quantize_int8_pairpacked_b32_h\d+_"
        r"eps[0-9eE+.-]+_offset[0-9eE+.-]+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "rms_norm_quantize_int8_pairpacked_",
            f"rms_norm_quantize_batch{tile}_int8_pairpacked_",
            1,
        )
    if re.fullmatch(
        r"silu_multiply_quantize_int8_pairpacked_b32_h\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "silu_multiply_quantize_int8_pairpacked_",
            f"silu_multiply_quantize_batch{tile}_int8_pairpacked_",
            1,
        )
    if re.fullmatch(
        r"sigmoid_multiply_quantize_fp8_e4m3(?:_spow2)?_b128_h\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "sigmoid_multiply_quantize_fp8_e4m3_",
            f"sigmoid_multiply_quantize_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(r"add_bf16_\d+\.comp", shader_file):
        return shader_file.replace("add_bf16_", f"add_batch{tile}_bf16_", 1)
    if re.fullmatch(r"silu_multiply_bf16_\d+\.comp", shader_file):
        return shader_file.replace(
            "silu_multiply_bf16_",
            f"silu_multiply_batch{tile}_bf16_",
            1,
        )
    if re.fullmatch(r"sigmoid_scalar_multiply_bf16_\d+\.comp", shader_file):
        return shader_file.replace(
            "sigmoid_scalar_multiply_bf16_",
            f"sigmoid_scalar_multiply_batch{tile}_bf16_",
            1,
        )
    if re.fullmatch(
        r"linear_sigmoid_scalar_multiply_bf16_\d+x\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "linear_sigmoid_scalar_multiply_bf16_",
            f"linear_sigmoid_scalar_multiply_batch{tile}_bf16_",
            1,
        )
    if re.fullmatch(
        r"linear_sigmoid_scalar_multiply_residual2_bf16_\d+x\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "linear_sigmoid_scalar_multiply_residual2_bf16_",
            f"linear_sigmoid_scalar_multiply_residual2_batch{tile}_bf16_",
            1,
        )
    contiguous_split = re.fullmatch(r"split_bf16_2x(\d+)\.comp", shader_file)
    if contiguous_split is not None and int(contiguous_split.group(1)) % 2 == 0:
        return shader_file.replace("split_bf16_", f"split_batch{tile}_bf16_", 1)
    prequant_fp8 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_prequant_fp8_e4m3_"
        r"(?:se8m0_)?b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if prequant_fp8 is not None:
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    mixed_parallel = re.fullmatch(
        r"mixed_parallel_linear_4way_prequant_fp8_e4m3_"
        r"b\d+x\d+_bf16_\d+x\d+_\d+_\d+_\d+\.comp",
        shader_file,
    )
    if mixed_parallel is not None:
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    if re.fullmatch(
        r"contiguous_linear_swiglu_prequant_fp8_e4m3_"
        r"b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    ):
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    prequant_parallel_fp8 = re.fullmatch(
        r"parallel_linear_[23]way_prequant_fp8_e4m3(?:_se8m0)?_"
        r"b\d+x\d+_\d+x\d+_\d+(?:_\d+)?\.comp",
        shader_file,
    )
    if prequant_parallel_fp8 is not None:
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    prequant_fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_prequant_fp8_e4m3_"
        r"b\d+x\d+_\d+x\d+\.comp",
        shader_file,
    )
    if prequant_fused_ffn is not None:
        return shader_file.replace(
            "_prequant_fp8_e4m3_",
            f"_prequant_batch{tile}_fp8_e4m3_",
            1,
        )
    prequant_int4 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_prequant_int4_"
        r"(?:gptq|ct)_s(?:f16|bf16)_g\d+_\d+x\d+\.comp",
        shader_file,
    )
    if prequant_int4 is not None:
        return shader_file.replace(
            "_prequant_int4_",
            f"_prequant_batch{tile}_int4_",
            1,
        )
    pairpacked_int4 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_prequant_pairpacked_int4_"
        r"gptq_s(?:f16|bf16)_g\d+_\d+x\d+\.comp",
        shader_file,
    )
    if pairpacked_int4 is not None:
        return shader_file.replace(
            "_prequant_pairpacked_int4_",
            f"_prequant_pairpacked_batch{tile}_int4_",
            1,
        )
    if re.fullmatch(r"split_bf16_2x\d+x\d+_head_interleaved\.comp", shader_file):
        return shader_file.replace("split_bf16_", f"split_batch{tile}_bf16_", 1)
    if shader_file == "sigmoid_multiply_bf16.comp":
        return f"sigmoid_multiply_batch{tile}_bf16.comp"
    attention_gate = re.fullmatch(
        r"softplus_multiply_bf16_q(\d+)_d(\d+)_(per_head|per_element)\.comp",
        shader_file,
    )
    if attention_gate is not None:
        return shader_file.replace(
            "softplus_multiply_bf16_",
            f"softplus_multiply_batch{tile}_bf16_",
            1,
        )
    rms_norm = re.fullmatch(
        r"rms_norm_bf16_h(\d+)_eps([0-9eE+.-]+)_offset([0-9eE+.-]+)\.comp",
        shader_file,
    )
    if rms_norm is not None and int(rms_norm.group(1)) % 2 == 0:
        return shader_file.replace("rms_norm_bf16_", f"rms_norm_batch{tile}_bf16_", 1)
    fp8 = re.fullmatch(
        r"(linear|linear_residual)_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fp8 is not None:
        operation, block_rows, block_columns, input_size, _ = fp8.groups()
        if (
            int(block_rows) % 2 == 0
            and int(block_columns) % 4 == 0
            and int(input_size) % 4 == 0
        ):
            return shader_file.replace(
                f"{operation}_fp8_e4m3_",
                f"{operation}_batch{tile}_fp8_e4m3_",
                1,
            )
    q8 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if q8 is not None:
        operation, input_size, output_size = q8.groups()
        if int(input_size) % Q8_0_GROUP_SIZE == 0 and int(output_size) % 2 == 0:
            return shader_file.replace(
                f"{operation}_q8_0_",
                f"{operation}_batch{tile}_q8_0_",
                1,
            )
    int4 = re.fullmatch(
        r"(linear|linear_bias|linear_residual)_int4_(gptq|ct)_s(?:f16|bf16)_"
        r"g(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if int4 is not None:
        operation, _, group_size, input_size, output_size = int4.groups()
        if (
            int(group_size) % INT4_VALUES_PER_PACKED_WORD == 0
            and int(input_size) % int(group_size) == 0
            and int(output_size) % 2 == 0
        ):
            return shader_file.replace(
                f"{operation}_int4_",
                f"{operation}_batch{tile}_int4_",
                1,
            )
    bf16 = re.fullmatch(
        r"(linear|linear_residual)_bf16_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if bf16 is not None:
        operation, input_size, output_size = bf16.groups()
        if (
            int(input_size) % 2 == 0
            and int(output_size) > 0
            and (operation == "linear" or int(output_size) % 2 == 0)
        ):
            return f"{operation}_batch{tile}_bf16_{input_size}x{output_size}.comp"
    parallel = re.fullmatch(
        r"parallel_linear_[23]way_bf16_(\d+)x.+\.comp",
        shader_file,
    )
    if parallel is not None and int(parallel.group(1)) % 2 == 0:
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    fused_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_fp8_e4m3_b(\d+)x(\d+)_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_ffn is not None:
        block_rows, block_columns, input_size, _ = map(int, fused_ffn.groups())
        if block_rows % 2 == 0 and block_columns % 4 == 0 and input_size % 4 == 0:
            return shader_file.replace(
                "parallel_linear_silu_multiply_fp8_e4m3_",
                f"parallel_linear_silu_multiply_batch{tile}_fp8_e4m3_",
                1,
            )
    parallel_fp8 = re.fullmatch(
        r"parallel_linear_(?P<branches>[23])way_fp8_e4m3(?:_se8m0)?_"
        r"b(?P<block_rows>\d+)x(?P<block_columns>\d+)_"
        r"(?P<input>\d+)x(?P<output_a>\d+)_(?P<output_b>\d+)"
        r"(?:_(?P<output_c>\d+))?\.comp",
        shader_file,
    )
    if parallel_fp8 is not None:
        branch_count = int(parallel_fp8["branches"])
        block_rows = int(parallel_fp8["block_rows"])
        output_sizes = [
            int(parallel_fp8[name])
            for name in ("output_a", "output_b", "output_c")
            if parallel_fp8[name] is not None
        ]
        if len(output_sizes) != branch_count:
            return None
        tile = tile_width or min(
            SCALAR_BATCH_LANE_TILE_WIDTH,
            max(1, FP8_LINEAR_MIN_WORKGROUPS // block_rows),
        )
        if output_sizes:
            tile = min(tile, max(output_sizes))
        if tile <= 1:
            return None
        return shader_file.replace(
            "parallel_linear_",
            f"parallel_linear_batch{tile}_",
            1,
        )
    parallel_q8 = re.fullmatch(
        r"parallel_linear_([23])way_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if parallel_q8 is not None:
        _branch_count, input_size, output_size = map(int, parallel_q8.groups())
        if input_size % Q8_0_GROUP_SIZE == 0 and output_size % 2 == 0:
            return shader_file.replace(
                "parallel_linear_",
                f"parallel_linear_batch{tile}_",
                1,
            )
    fused_q8_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_q8_0_(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_q8_ffn is not None:
        input_size, output_size = map(int, fused_q8_ffn.groups())
        if input_size % Q8_0_GROUP_SIZE == 0 and output_size % 2 == 0:
            return shader_file.replace(
                "parallel_linear_silu_multiply_",
                f"parallel_linear_silu_multiply_batch{tile}_",
                1,
            )
    fused_bf16_ffn = re.fullmatch(
        r"parallel_linear_silu_multiply_bf16_"
        r"(\d+)x(\d+)\.comp",
        shader_file,
    )
    if fused_bf16_ffn is not None:
        input_size, output_size = fused_bf16_ffn.groups()
        if int(input_size) % 2 == 0 and int(output_size) % 2 == 0:
            return (
                f"parallel_linear_silu_multiply_batch{tile}_bf16_"
                f"{input_size}x{output_size}.comp"
            )
    return None


def mixed_parallel_projection_source_shader_files(
    shader_file: str,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"mixed_parallel_linear_4way_prequant_fp8_e4m3_"
        r"b(\d+)x(\d+)_bf16_(\d+)x(\d+)_(\d+)_(\d+)_(\d+)\.comp",
        shader_file,
    )
    if match is None:
        return None
    (
        block_rows,
        block_columns,
        input_size,
        output_a,
        output_b,
        output_c,
        output_d,
    ) = match.groups()
    return (
        "parallel_linear_2way_prequant_fp8_e4m3_"
        f"b{block_rows}x{block_columns}_{input_size}x{output_a}_{output_b}.comp",
        f"parallel_linear_2way_bf16_{input_size}x{output_c}_{output_d}.comp",
    )


def mixed_parallel_projection_batch_implementations(
    shader_file: str,
    *,
    local_size_x: int,
    workgroup_count_x: int,
    cooperative_float8_e4m3_shapes: tuple[tuple[int, int, int], ...],
) -> list[Json]:
    sources = mixed_parallel_projection_source_shader_files(shader_file)
    if sources is None:
        raise ModelCompileError(
            f"shader {shader_file!r} is not a mixed parallel projection"
        )
    fp8_shader_file, bf16_shader_file = sources
    implementations: list[Json] = []
    bf16_cooperative = cooperative_bfloat16_batch_shader_file(bf16_shader_file)
    for shape in cooperative_float8_e4m3_shapes:
        fp8_cooperative = cooperative_float8_e4m3_batch_shader_file(
            fp8_shader_file,
            shape=shape,
        )
        if fp8_cooperative is None or bf16_cooperative is None:
            continue
        implementations.append(
            {
                "execution_domain": "prefill",
                "lane_tile_width": 4 * shape[1],
                "selection_priority": 0,
                "independent_candidate_compatible": False,
                "causal_sequence_compatible": True,
                "parallel_block_compatible": False,
                "device_requirements": {
                    "vulkan_device_extensions": [],
                    "vulkan_features": [],
                    "subgroup_operations": [],
                    "cooperative_float8_e4m3_shape": list(shape),
                    "cooperative_bfloat16_shape": COOPERATIVE_BFLOAT16_SHAPE,
                    "subgroup_size": 64,
                },
                "stages": [
                    persistent_batch_control_stage(
                        fp8_cooperative,
                        256,
                        cooperative_float8_e4m3_workgroup_count_x(
                            fp8_shader_file,
                            shape=shape,
                        ),
                        descriptor_bindings=[
                            {"binding": 0, "source_binding": 0},
                            {"binding": 1, "source_binding": 1},
                            {"binding": 2, "source_binding": 3},
                            {"binding": 3, "source_binding": 4},
                            {"binding": 4, "source_binding": 7},
                            {"binding": 5, "source_binding": 8},
                            {"binding": 6, "source_binding": 9},
                            {"binding": 7, "source_binding": 10},
                        ],
                    ),
                    persistent_batch_control_stage(
                        bf16_cooperative,
                        256,
                        cooperative_bfloat16_workgroup_count_x(bf16_shader_file),
                        descriptor_bindings=[
                            {"binding": 0, "source_binding": 2},
                            {"binding": 1, "source_binding": 5},
                            {"binding": 2, "source_binding": 6},
                            {"binding": 3, "source_binding": 11},
                            {"binding": 4, "source_binding": 12},
                        ],
                    ),
                ],
            }
        )
    for tile_width in EXACT_BATCH_LANE_TILE_WIDTHS:
        batch_shader_file = weight_shared_batch_shader_file(
            shader_file,
            tile_width=tile_width,
        )
        if batch_shader_file is None:
            raise ModelCompileError(
                f"mixed projection {shader_file!r} lost batch width {tile_width}"
            )
        implementations.append(
            {
                "execution_domain": "decode_and_prefill",
                "lane_tile_width": tile_width,
                "selection_priority": 0,
                "independent_candidate_compatible": True,
                "causal_sequence_compatible": True,
                "parallel_block_compatible": True,
                "device_requirements": {
                    "vulkan_device_extensions": [],
                    "vulkan_features": [],
                    "subgroup_operations": [],
                },
                "stages": [
                    persistent_batch_control_stage(
                        batch_shader_file,
                        local_size_x,
                        workgroup_count_x,
                    )
                ],
            }
        )
    return implementations


def weight_shared_batch_workgroup_count_x(
    shader_file: str,
    *,
    tile_width: int,
    scalar_workgroup_count_x: int,
) -> int:
    if tile_width <= 0 or scalar_workgroup_count_x <= 0:
        raise ValueError("batch and scalar workgroup counts must be positive")
    lane_parallel = (
        re.fullmatch(r"add_bf16_\d+\.comp", shader_file) is not None
        or re.fullmatch(r"silu_multiply_bf16_\d+\.comp", shader_file) is not None
        or re.fullmatch(
            r"bounded_silu_multiply_bf16_\d+_limit[0-9eE+.-]+\.comp",
            shader_file,
        )
        is not None
        or re.fullmatch(
            r"(?:hyper_connection_pre|hyper_connection_post_pre)_m\d+_h\d+_i\d+_"
            r"neps[0-9eE+.-]+_heps[0-9eE+.-]+\.comp",
            shader_file,
        )
        is not None
        or re.fullmatch(
            r"sigmoid_scalar_multiply_bf16_\d+\.comp",
            shader_file,
        )
        is not None
        or re.fullmatch(
            r"linear_sigmoid_scalar_multiply_bf16_\d+x\d+\.comp",
            shader_file,
        )
        is not None
        or re.fullmatch(
            r"linear_sigmoid_scalar_multiply_residual2_bf16_\d+x\d+\.comp",
            shader_file,
        )
        is not None
        or re.fullmatch(r"split_bf16_2x\d+\.comp", shader_file) is not None
    )
    bf16_linear = re.fullmatch(
        r"linear_bf16_\d+x(\d+)\.comp",
        shader_file,
    )
    lane_parallel = lane_parallel or (
        bf16_linear is not None and int(bf16_linear.group(1)) % 2 == 1
    )
    return (
        scalar_workgroup_count_x * tile_width
        if lane_parallel
        else scalar_workgroup_count_x
    )
