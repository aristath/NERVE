from nerve.model_package_common import *
from nerve.model_package_shaders import *
from nerve.model_package_tensors import *


def render_sparse_moe_projection_shader(
    source_dir: Path,
    shader_file: str,
    render_template,
) -> str | None:
    sparse_moe_int4_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?_int4_ct_"
        r"s(f16|bf16)_g(\d+)_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_int4_shape is not None:
        (
            stage,
            batch_tile,
            scale_dtype,
            group_size,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = sparse_moe_int4_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "INT4 sparse experts support only frame-parallel batch tiles"
            )
        (
            group_size,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = map(
            int,
            (
                group_size,
                hidden_size,
                intermediate_size,
                num_experts,
                experts_per_token,
            ),
        )
        input_width = hidden_size if stage == "gate_up" else intermediate_size
        output_width = intermediate_size if stage == "gate_up" else hidden_size
        if (
            group_size <= 0
            or group_size % INT4_VALUES_PER_PACKED_WORD != 0
            or input_width % group_size != 0
            or input_width % 8 != 0
            or output_width % 2 != 0
            or not 0 < experts_per_token <= num_experts <= 4096
        ):
            raise ModelCompileError(
                f"invalid INT4 sparse expert geometry {shader_file!r}"
            )
        read_scale_body = (
            "    vec2 values = unpackHalf2x16("
            "expert_scales.words[index >> 1u]);\n"
            "    return (index & 1u) == 0u ? values.x : values.y;"
            if scale_dtype == "f16"
            else (
                "    return read_bf16_word("
                "expert_scales.words[index >> 1u], index);"
            )
        )
        return render_template(
            source_dir,
            f"sparse_moe_{stage}_int4_ct.comp.template",
            {
                "GROUP_SIZE": str(group_size),
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "TILE_ROWS": str(INT4_CT_OUTPUT_TILE_ROWS),
                "BATCH_CONTROL": (
                    "layout(push_constant) uniform DispatchControl { uint "
                    "expert_start; } dispatch_control;"
                    if batch_tile is None
                    else "layout(push_constant) uniform BatchControl { uint "
                    "batch_width; uint expert_start; uint expert_count; uint "
                    "owned_route_count; uint dispatch_x; uint dispatch_y; uint "
                    "dispatch_z; } batch_control;"
                ),
                "EXPERT_START": (
                    "dispatch_control.expert_start"
                    if batch_tile is None
                    else "batch_control.expert_start"
                ),
                "BATCH_INDEX": (
                    "0u"
                    if batch_tile is None
                    else (
                        "batch_control.expert_count == 0u ? gl_WorkGroupID.y : "
                        "(route / EXPERTS_PER_TOKEN)"
                    )
                ),
                "BATCH_WIDTH": (
                    "1u"
                    if batch_tile is None
                    else (
                        "(batch_control.expert_count == 0u ? "
                        "batch_control.batch_width : "
                        "((batch_control.owned_route_count + EXPERTS_PER_TOKEN - 1u) "
                        "/ EXPERTS_PER_TOKEN))"
                    )
                ),
                "ROUTE_LIMIT": (
                    "EXPERTS_PER_TOKEN"
                    if batch_tile is None
                    else (
                        "(batch_control.expert_count == 0u ? EXPERTS_PER_TOKEN "
                        ": batch_control.owned_route_count)"
                    )
                ),
                "EXPERT_COUNT": (
                    "expert_input_weight.words.length() / EXPERT_INPUT_WORDS"
                    if batch_tile is None and stage == "gate_up"
                    else "expert_output_weight.words.length() / EXPERT_OUTPUT_WORDS"
                    if batch_tile is None
                    else (
                        "(batch_control.expert_count == 0u "
                        "? expert_input_weight.words.length() / EXPERT_INPUT_WORDS "
                        ": batch_control.expert_count)"
                        if stage == "gate_up"
                        else (
                            "(batch_control.expert_count == 0u "
                            "? expert_output_weight.words.length() / EXPERT_OUTPUT_WORDS "
                            ": batch_control.expert_count)"
                        )
                    )
                ),
                "ROUTE_MAPPING": (
                    ""
                    if batch_tile is None
                    else (
                        "    uint compact_batch = batch_control.expert_count == 0u\n"
                        "        ? batch_index\n"
                        "        : route / EXPERTS_PER_TOKEN;\n"
                        "    uint compact_route = batch_control.expert_count == 0u\n"
                        "        ? route\n"
                        "        : route % EXPERTS_PER_TOKEN;\n"
                        "    uint compact_index = expert_intermediates.words[\n"
                        "        compact_batch * EXPERT_FRAME_WORDS\n"
                        "            + EXPERT_DATA_WORDS\n"
                        "            + compact_route\n"
                        "    ];\n"
                        "    batch_index = compact_index / EXPERTS_PER_TOKEN;\n"
                        "    route = compact_index % EXPERTS_PER_TOKEN;\n"
                        "    tile = group % TILES_PER_ROUTE;"
                    )
                ),
                "INTERMEDIATE_OUTPUT_OFFSET": (
                    "(route_offset + route) * INTERMEDIATE_WORDS"
                    if batch_tile is None
                    else (
                        "batch_index * EXPERT_FRAME_WORDS"
                        " + route * INTERMEDIATE_WORDS"
                    )
                ),
                "INTERMEDIATE_INPUT_OFFSET": (
                    "(batch_index * EXPERTS_PER_TOKEN + route) * INTERMEDIATE_WORDS"
                    if batch_tile is None
                    else (
                        "batch_index * EXPERT_FRAME_WORDS"
                        " + route * INTERMEDIATE_WORDS"
                    )
                ),
                "READ_SCALE_BODY": read_scale_body,
            },
        )

    sparse_moe_fp8_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?(?:_(prequant))?_fp8_e4m3_"
        r"b(\d+)x(\d+)_h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_fp8_shape is not None:
        (
            stage,
            batch_tile,
            prequant,
            block_rows,
            block_columns,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = sparse_moe_fp8_shape.groups()
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "FP8 sparse experts support only frame-parallel batch tiles"
            )
        (
            block_rows,
            block_columns,
            hidden_size,
            intermediate_size,
            num_experts,
            experts_per_token,
        ) = map(
            int,
            (
                block_rows,
                block_columns,
                hidden_size,
                intermediate_size,
                num_experts,
                experts_per_token,
            ),
        )
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 activations for FP8 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        if prequant is not None and stage != "gate_up":
            raise ModelCompileError(
                "only the sparse gate/up stage accepts a shared prequantized input"
            )
        if prequant is not None:
            return render_template(
                source_dir,
                (
                    "sparse_moe_gate_up_prequant_fp8_e4m3.comp.template"
                    if batch_tile is None
                    else "sparse_moe_gate_up_prequant_batch1_fp8_e4m3.comp.template"
                ),
                {
                    "BLOCK_ROWS": str(block_rows),
                    "BLOCK_COLUMNS": str(block_columns),
                    "HIDDEN_SIZE": str(hidden_size),
                    "INTERMEDIATE_SIZE": str(intermediate_size),
                    "NUM_EXPERTS": str(num_experts),
                    "EXPERTS_PER_TOKEN": str(experts_per_token),
                    "LOCAL_SIZE_X": "512",
                    "TILE_ROWS": str(
                        FP8_SPARSE_PREQUANT_GATE_UP_TILE_ROWS
                    ),
                },
            )
        return render_template(
            source_dir,
            (
                f"sparse_moe_{stage}_fp8_e4m3.comp.template"
                if batch_tile is None
                else f"sparse_moe_{stage}_batch1_fp8_e4m3.comp.template"
            ),
            {
                "BLOCK_ROWS": str(block_rows),
                "BLOCK_COLUMNS": str(block_columns),
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
                "LOCAL_SIZE_X": "512",
                "TILE_ROWS": str(
                    FP8_SPARSE_GATE_UP_TILE_ROWS
                    if stage == "gate_up"
                    else FP8_SPARSE_DOWN_TILE_ROWS
                ),
            },
        )

    sparse_moe_shape = re.fullmatch(
        r"sparse_moe_(gate_up|down)(?:_batch(\d+))?_bf16_"
        r"h(\d+)_i(\d+)_e(\d+)_k(\d+)\.comp",
        shader_file,
    )
    if sparse_moe_shape is not None:
        stage = sparse_moe_shape.group(1)
        batch_tile = sparse_moe_shape.group(2)
        if batch_tile not in {None, "1"}:
            raise ModelCompileError(
                "BF16 sparse experts support only frame-parallel batch tiles"
            )
        hidden_size, intermediate_size, num_experts, experts_per_token = map(
            int, sparse_moe_shape.groups()[2:]
        )
        if hidden_size % 2 or intermediate_size % 2:
            raise ModelCompileError(
                "packed BF16 sparse experts require even dimensions"
            )
        if not 0 < experts_per_token <= num_experts <= 4096:
            raise ModelCompileError(
                f"invalid sparse expert routing e{num_experts} k{experts_per_token}"
            )
        return render_template(
            source_dir,
            (
                f"sparse_moe_{stage}_bf16.comp.template"
                if batch_tile is None
                else f"sparse_moe_{stage}_batch1_bf16.comp.template"
            ),
            {
                "HIDDEN_SIZE": str(hidden_size),
                "INTERMEDIATE_SIZE": str(intermediate_size),
                "NUM_EXPERTS": str(num_experts),
                "EXPERTS_PER_TOKEN": str(experts_per_token),
            },
        )

    return None
