from __future__ import annotations

import math
import re
from pathlib import Path

from nerve.model_package_assets import stream_control_binding_for_node
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


POOL_SHADER_PATTERN = re.compile(
    r"learned_gated_kv_pool_bf16_f32_h(\d+)_d(\d+)_r(\d+)_c([12])\.comp"
)
TEMPORAL_POOL_SHADER_PATTERN = re.compile(
    r"learned_gated_kv_pool_temporal_bf16_f32_h(\d+)_d(\d+)_r(\d+)_c([12])\.comp"
)
FINALIZE_SHADER_PATTERN = re.compile(
    r"compressed_kv_finalize_f32_bf16_d(\d+)_r(\d+)_eps([0-9eE+.-]+)_"
    r"theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved|proportional)_po(-?\d+)_qfp8e4m3b(\d+)\.comp"
)
TEMPORAL_FINALIZE_SHADER_PATTERN = re.compile(
    r"compressed_kv_finalize_temporal_f32_bf16_d(\d+)_r(\d+)_eps([0-9eE+.-]+)_"
    r"theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved|proportional)_po(-?\d+)_qfp8e4m3b(\d+)\.comp"
)
CONDITIONAL_APPEND_SHADER_PATTERN = re.compile(
    r"conditional_append_state_bf16_d(\d+)_p(\d+)\.comp"
)
TEMPORAL_CONDITIONAL_APPEND_SHADER_PATTERN = re.compile(
    r"conditional_append_state_temporal_bf16_d(\d+)_p(\d+)\.comp"
)
INDEX_TRANSFORM_SHADER_PATTERN = re.compile(
    r"index_vector_transform_bf16_h(\d+)_d(\d+)_r(\d+)_theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved)_qfp4e2m1b(\d+)\.comp"
)
TEMPORAL_INDEX_TRANSFORM_SHADER_PATTERN = re.compile(
    r"index_vector_transform_temporal_bf16_h(\d+)_d(\d+)_r(\d+)_theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved)_qfp4e2m1b(\d+)\.comp"
)
INDEX_FINALIZE_SHADER_PATTERN = re.compile(
    r"compressed_index_kv_finalize_f32_bf16_d(\d+)_r(\d+)_eps([0-9eE+.-]+)_"
    r"theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved)_po(-?\d+)_qfp4e2m1b(\d+)\.comp"
)
TEMPORAL_INDEX_FINALIZE_SHADER_PATTERN = re.compile(
    r"compressed_index_kv_finalize_temporal_f32_bf16_d(\d+)_r(\d+)_eps([0-9eE+.-]+)_"
    r"theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved)_po(-?\d+)_qfp4e2m1b(\d+)\.comp"
)
INDEX_SCORES_SHADER_PATTERN = re.compile(
    r"learned_index_scores_bf16_f32_h(\d+)_d(\d+)_r(\d+)_m(\d+)_c(\d+)_"
    r"scale([0-9eE+.-]+)\.comp"
)
TEMPORAL_INDEX_SCORES_SHADER_PATTERN = re.compile(
    r"learned_index_scores_temporal_bf16_f32_h(\d+)_d(\d+)_r(\d+)_m(\d+)_c(\d+)_"
    r"scale([0-9eE+.-]+)\.comp"
)
RADIX_TOPK_SHADER_PATTERN = re.compile(
    r"radix_topk_index_f32_u32_m(\d+)_k(\d+)_r(\d+)_o(\d+)\.comp"
)
TEMPORAL_RADIX_TOPK_SHADER_PATTERN = re.compile(
    r"radix_topk_index_temporal_f32_u32_m(\d+)_k(\d+)_r(\d+)_o(\d+)\.comp"
)
CHRONOLOGICAL_INDEX_SHADER_PATTERN = re.compile(
    r"chronological_compressed_index_u32_m(\d+)_r(\d+)_o(\d+)\.comp"
)
TEMPORAL_CHRONOLOGICAL_INDEX_SHADER_PATTERN = re.compile(
    r"chronological_compressed_index_temporal_u32_m(\d+)_r(\d+)_o(\d+)\.comp"
)


def latent_compression_shader_file(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> str:
    operation = str(node.get("op", ""))
    if operation == "learned_gated_kv_pool":
        return _pool_shader_file(circuit, node, tensor_index)
    if operation == "compressed_kv_finalize":
        return _finalize_shader_file(circuit, node, tensor_index)
    if operation == "conditional_append_state_update":
        return _conditional_append_shader_file(circuit, node)
    if operation in {"index_vector_transform", "compressed_index_kv_finalize"}:
        return _index_transform_shader_file(circuit, node, tensor_index)
    if operation == "learned_index_scores":
        return _index_scores_shader_file(circuit, node)
    if operation == "radix_topk_index":
        return _radix_topk_shader_file(node)
    if operation == "chronological_compressed_index":
        return _chronological_index_shader_file(circuit, node)
    raise ModelCompileError(
        f"node {node.get('id')!r} is not a latent-memory compression component"
    )


def _pool_shader_file(circuit: Json, node: Json, tensor_index: Json) -> str:
    attrs = node.get("attrs", {})
    hidden_size = int(attrs.get("hidden_size", 0))
    head_width = int(attrs.get("head_width", 0))
    ratio = int(attrs.get("ratio", 0))
    coefficient = int(attrs.get("lane_coefficient", 0))
    overlap = attrs.get("overlap")
    params = node.get("params", [])
    states = node.get("state_reads", [])
    state_writes = node.get("state_writes", [])
    expected_state_shape = [
        2,
        coefficient * ratio,
        coefficient * head_width,
    ]
    state_port = next(
        (
            port
            for port in circuit.get("state_ports", [])
            if states and port.get("id") == states[0]
        ),
        None,
    )
    if (
        hidden_size <= 0
        or hidden_size % 2
        or head_width <= 0
        or head_width > 1024
        or ratio <= 0
        or coefficient not in {1, 2}
        or overlap is not (coefficient == 2)
        or attrs.get("pooling") != "learned_position_biased_softmax"
        or attrs.get("output_element_bytes") != [4]
        or len(node.get("inputs", [])) != 2
        or len(node.get("outputs", [])) != 1
        or len(params) != 3
        or len(states) != 1
        or state_writes != states
        or not isinstance(state_port, dict)
        or state_port.get("dtype") != "F32"
        or state_port.get("shape") != expected_state_shape
        or state_port.get("update") != "position_biased_softmax_pool"
    ):
        raise ModelCompileError(
            f"learned compressor pool node {node['id']!r} has an invalid contract"
        )
    expected_parameters = (
        (params[0], "F32", [ratio, coefficient * head_width]),
        (params[1], "BF16", [coefficient * head_width, hidden_size]),
        (params[2], "BF16", [coefficient * head_width, hidden_size]),
    )
    for parameter_id, dtype, shape in expected_parameters:
        if (
            parameter_dtype_for_id(circuit, parameter_id, tensor_index) != dtype
            or parameter_shape_for_id(circuit, parameter_id, tensor_index) != shape
            or parameter_layout_for_id(circuit, parameter_id, tensor_index)
            != ROW_MAJOR_LAYOUT
        ):
            raise ModelCompileError(
                f"learned compressor pool node {node['id']!r} has incompatible "
                f"parameter {parameter_id!r}"
            )
    binding = stream_control_binding_for_node(circuit, node)
    return (
        f"learned_gated_kv_pool_bf16_f32_h{hidden_size}_d{head_width}_"
        f"r{ratio}_c{coefficient}__sc{binding}.comp"
    )


def _finalize_shader_file(circuit: Json, node: Json, tensor_index: Json) -> str:
    attrs = node.get("attrs", {})
    head_width = int(attrs.get("head_width", 0))
    rotary_width = int(attrs.get("rotary_width", 0))
    epsilon = float(attrs.get("normalization_epsilon", 0.0))
    theta = float(attrs.get("theta", 0.0))
    position_offset = int(attrs.get("position_offset", 0))
    quantization = attrs.get("activation_quantization")
    block_columns = (
        int(quantization.get("block_columns", 0))
        if isinstance(quantization, dict)
        else 0
    )
    non_rotary_width = head_width - rotary_width
    params = node.get("params", [])
    if (
        head_width <= 0
        or head_width > 1024
        or head_width & (head_width - 1)
        or rotary_width <= 0
        or rotary_width > head_width
        or rotary_width % 2
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not math.isfinite(theta)
        or theta <= 0.0
        or attrs.get("position_source") != "stream_tick"
        or attrs.get("rotary_scope") != "tail"
        or int(attrs.get("head_count", 0)) != 1
        or quantization
        != {
            "format": "fp8_e4m3",
            "scale_format": "e8m0_power_of_two",
            "block_columns": 64,
            "scope": "non_rotary_dimensions",
            "mode": "quantize_dequantize",
        }
        or non_rotary_width <= 0
        or non_rotary_width % block_columns
        or attrs.get("output_element_bytes") != [2]
        or len(node.get("inputs", [])) != 1
        or len(node.get("outputs", [])) != 1
        or len(params) != 1
        or node.get("state_reads")
        or node.get("state_writes")
        or parameter_dtype_for_id(circuit, params[0], tensor_index) != "BF16"
        or parameter_shape_for_id(circuit, params[0], tensor_index) != [head_width]
        or parameter_layout_for_id(circuit, params[0], tensor_index) != ROW_MAJOR_LAYOUT
    ):
        raise ModelCompileError(
            f"compressed KV finalizer node {node['id']!r} has an invalid contract"
        )
    binding = stream_control_binding_for_node(circuit, node)
    return (
        f"compressed_kv_finalize_f32_bf16_d{head_width}_r{rotary_width}_"
        f"eps{shader_float_token(epsilon)}_{_rope_suffix(attrs)}_"
        f"po{position_offset}_qfp8e4m3b{block_columns}__sc{binding}.comp"
    )


def _rope_suffix(attrs: Json) -> str:
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
            raise ModelCompileError(
                "YaRN compressor RoPE has no compiled scaling profile"
            )
        return (
            f"theta{shader_float_token(theta)}_yarn"
            f"_f{shader_float_token(float(scaling['factor']))}"
            f"_lo{shader_float_token(float(scaling['correction_low']))}"
            f"_hi{shader_float_token(float(scaling['correction_high']))}"
            f"_a{shader_float_token(float(scaling['attention_factor']))}_{layout}"
        )
    if scaling is not None:
        raise ModelCompileError(
            f"compressor RoPE type {rope_type!r} unexpectedly declares scaling"
        )
    return f"theta{shader_float_token(theta)}_{layout}"


def _conditional_append_shader_file(circuit: Json, node: Json) -> str:
    attrs = node.get("attrs", {})
    period = int(attrs.get("period", 0))
    states = node.get("state_reads", [])
    state_writes = node.get("state_writes", [])
    state_port = next(
        (
            port
            for port in circuit.get("state_ports", [])
            if states and port.get("id") == states[0]
        ),
        None,
    )
    shape_per_token = (
        state_port.get("shape_per_token") if isinstance(state_port, dict) else None
    )
    width = (
        int(shape_per_token[0])
        if isinstance(shape_per_token, list) and len(shape_per_token) == 1
        else 0
    )
    if (
        period <= 0
        or width <= 0
        or width % 2
        or len(node.get("inputs", [])) != 2
        or len(node.get("outputs", [])) != 1
        or node.get("params")
        or len(states) != 1
        or state_writes != states
        or not isinstance(state_port, dict)
        or state_port.get("type")
        not in {"append_only_attention_memory", "append_only_index_memory"}
        or state_port.get("dtype") != "BF16"
        or state_port.get("growth")
        not in {f"one_per_{period}_activations", "with_compressed_kv_memory"}
    ):
        raise ModelCompileError(
            f"conditional state append node {node['id']!r} has an invalid contract"
        )
    binding = stream_control_binding_for_node(circuit, node)
    return f"conditional_append_state_bf16_d{width}_p{period}__sc{binding}.comp"


def _index_transform_shader_file(
    circuit: Json,
    node: Json,
    tensor_index: Json,
) -> str:
    attrs = node.get("attrs", {})
    operation = str(node["op"])
    head_count = int(attrs.get("head_count", 0))
    head_width = int(attrs.get("head_width", 0))
    rotary_width = int(attrs.get("rotary_width", 0))
    theta = float(attrs.get("theta", 0.0))
    quantization = attrs.get("activation_quantization")
    params = node.get("params", [])
    epsilon = float(attrs.get("normalization_epsilon", 0.0))
    if (
        head_count <= 0
        or (operation == "compressed_index_kv_finalize" and head_count != 1)
        or head_width <= 0
        or head_width > 1024
        or head_width & (head_width - 1)
        or rotary_width <= 0
        or rotary_width > head_width
        or rotary_width % 2
        or not math.isfinite(theta)
        or theta <= 0.0
        or attrs.get("position_source") != "stream_tick"
        or attrs.get("rotary_scope") != "tail"
        or attrs.get("rotation") != "hadamard"
        or quantization
        != {
            "format": "fp4_e2m1",
            "scale_format": "e8m0_power_of_two",
            "block_columns": 32,
            "mode": "quantize_dequantize",
        }
        or head_width % 32
        or attrs.get("output_element_bytes") != [2]
        or len(node.get("inputs", [])) != 1
        or len(node.get("outputs", [])) != 1
        or node.get("state_reads")
        or node.get("state_writes")
    ):
        raise ModelCompileError(
            f"index vector transform node {node['id']!r} has an invalid contract"
        )
    rope_suffix = _rope_suffix(attrs)
    binding = stream_control_binding_for_node(circuit, node)
    if operation == "index_vector_transform":
        if params:
            raise ModelCompileError(
                f"index query transform node {node['id']!r} unexpectedly has parameters"
            )
        return (
            f"index_vector_transform_bf16_h{head_count}_d{head_width}_r{rotary_width}_"
            f"{rope_suffix}_qfp4e2m1b32__sc{binding}.comp"
        )
    if (
        len(params) != 1
        or not math.isfinite(epsilon)
        or epsilon <= 0.0
        or parameter_dtype_for_id(circuit, params[0], tensor_index) != "BF16"
        or parameter_shape_for_id(circuit, params[0], tensor_index) != [head_width]
        or parameter_layout_for_id(circuit, params[0], tensor_index) != ROW_MAJOR_LAYOUT
    ):
        raise ModelCompileError(
            f"compressed index finalizer node {node['id']!r} has an invalid norm"
        )
    position_offset = int(attrs.get("position_offset", 0))
    return (
        f"compressed_index_kv_finalize_f32_bf16_d{head_width}_r{rotary_width}_"
        f"eps{shader_float_token(epsilon)}_{rope_suffix}_po{position_offset}_"
        f"qfp4e2m1b32__sc{binding}.comp"
    )


def _index_scores_shader_file(circuit: Json, node: Json) -> str:
    attrs = node.get("attrs", {})
    heads = int(attrs.get("heads", 0))
    head_width = int(attrs.get("head_width", 0))
    ratio = int(attrs.get("ratio", 0))
    maximum = int(attrs.get("max_compressed_positions", 0))
    scale = float(attrs.get("score_scale", 0.0))
    chunk = 256
    if (
        heads <= 0
        or heads % 16
        or head_width <= 0
        or head_width % 16
        or heads * head_width > 16_384
        or ratio <= 0
        or maximum <= 0
        or not math.isfinite(scale)
        or scale <= 0.0
        or attrs.get("score_activation") != "relu_then_head_weighted_sum"
        or attrs.get("output_element_bytes") != [4]
        or len(node.get("inputs", [])) != 3
        or len(node.get("outputs", [])) != 1
        or node.get("params")
        or node.get("state_reads")
        or node.get("state_writes")
    ):
        raise ModelCompileError(
            f"learned index score node {node['id']!r} has an invalid contract"
        )
    binding = stream_control_binding_for_node(circuit, node)
    return (
        f"learned_index_scores_bf16_f32_h{heads}_d{head_width}_r{ratio}_"
        f"m{maximum}_c{chunk}_scale{shader_float_token(scale)}__sc{binding}.comp"
    )


def _radix_topk_shader_file(node: Json) -> str:
    attrs = node.get("attrs", {})
    maximum = int(attrs.get("max_scores", 0))
    top_k = int(attrs.get("top_k", 0))
    ratio = int(attrs.get("ratio", 0))
    offset = int(attrs.get("index_offset", -1))
    if (
        maximum <= 0
        or top_k <= 0
        or top_k > maximum
        or ratio <= 0
        or offset < 0
        or attrs.get("ordering") != "descending_float_score"
        or attrs.get("output_element_bytes") != [4]
        or len(node.get("inputs", [])) != 1
        or len(node.get("outputs", [])) != 1
        or node.get("params")
        or node.get("state_reads")
        or node.get("state_writes")
    ):
        raise ModelCompileError(
            f"radix top-k node {node['id']!r} has an invalid contract"
        )
    return f"radix_topk_index_f32_u32_m{maximum}_k{top_k}_r{ratio}_o{offset}__sc2.comp"


def _chronological_index_shader_file(circuit: Json, node: Json) -> str:
    attrs = node.get("attrs", {})
    maximum = int(attrs.get("max_indices", 0))
    ratio = int(attrs.get("ratio", 0))
    offset = int(attrs.get("index_offset", -1))
    if (
        maximum <= 0
        or ratio <= 0
        or offset < 0
        or attrs.get("causal") is not True
        or attrs.get("output_element_bytes") != [4]
        or len(node.get("inputs", [])) != 1
        or len(node.get("outputs", [])) != 1
        or node.get("params")
        or node.get("state_reads")
        or node.get("state_writes")
    ):
        raise ModelCompileError(
            f"chronological compressed-index node {node['id']!r} has an invalid contract"
        )
    binding = stream_control_binding_for_node(circuit, node)
    return (
        f"chronological_compressed_index_u32_m{maximum}_r{ratio}_o{offset}"
        f"__sc{binding}.comp"
    )


def render_latent_compression_shader(
    source_dir: Path,
    shader_file: str,
) -> str | None:
    temporal_pool = TEMPORAL_POOL_SHADER_PATTERN.fullmatch(shader_file)
    pool = temporal_pool or POOL_SHADER_PATTERN.fullmatch(shader_file)
    if pool is not None:
        hidden_size, head_width, ratio, coefficient = map(int, pool.groups())
        if (
            hidden_size <= 0
            or hidden_size % 2
            or head_width <= 0
            or head_width > 1024
            or ratio <= 0
            or coefficient not in {1, 2}
        ):
            raise ModelCompileError(
                f"invalid learned compressor pool shape {shader_file!r}"
            )
        return _render_template(
            source_dir
            / (
                "learned_gated_kv_pool_temporal.comp.template"
                if temporal_pool is not None
                else "learned_gated_kv_pool.comp.template"
            ),
            {
                "HIDDEN_SIZE": str(hidden_size),
                "HEAD_WIDTH": str(head_width),
                "COMPRESSION_RATIO": str(ratio),
                "LANE_COEFFICIENT": str(coefficient),
            },
        )
    temporal_conditional_append = TEMPORAL_CONDITIONAL_APPEND_SHADER_PATTERN.fullmatch(
        shader_file
    )
    conditional_append = temporal_conditional_append or (
        CONDITIONAL_APPEND_SHADER_PATTERN.fullmatch(shader_file)
    )
    if conditional_append is not None:
        width, period = map(int, conditional_append.groups())
        if width <= 0 or width % 2 or period <= 0:
            raise ModelCompileError(
                f"invalid conditional state append shape {shader_file!r}"
            )
        return _render_template(
            source_dir
            / (
                "conditional_append_state_temporal_bf16.comp.template"
                if temporal_conditional_append is not None
                else "conditional_append_state_bf16.comp.template"
            ),
            {"ELEMENT_COUNT": str(width), "PERIOD": str(period)},
        )
    temporal_index_transform = TEMPORAL_INDEX_TRANSFORM_SHADER_PATTERN.fullmatch(
        shader_file
    )
    index_transform = (
        temporal_index_transform
        or INDEX_TRANSFORM_SHADER_PATTERN.fullmatch(shader_file)
    )
    if index_transform is not None:
        return _render_index_transform(
            source_dir,
            index_transform,
            finalizer=False,
            temporal=temporal_index_transform is not None,
        )
    temporal_index_finalize = TEMPORAL_INDEX_FINALIZE_SHADER_PATTERN.fullmatch(
        shader_file
    )
    index_finalize = temporal_index_finalize or INDEX_FINALIZE_SHADER_PATTERN.fullmatch(
        shader_file
    )
    if index_finalize is not None:
        return _render_index_transform(
            source_dir,
            index_finalize,
            finalizer=True,
            temporal=temporal_index_finalize is not None,
        )
    temporal_index_scores = TEMPORAL_INDEX_SCORES_SHADER_PATTERN.fullmatch(shader_file)
    index_scores = temporal_index_scores or INDEX_SCORES_SHADER_PATTERN.fullmatch(
        shader_file
    )
    if index_scores is not None:
        heads, width, ratio, maximum, chunk = map(int, index_scores.groups()[:5])
        scale = index_scores.group(6)
        if (
            heads <= 0
            or heads % 16
            or width <= 0
            or width % 16
            or ratio <= 0
            or maximum <= 0
            or chunk != 256
        ):
            raise ModelCompileError(
                f"invalid learned index score shape {shader_file!r}"
            )
        return _render_template(
            source_dir
            / (
                "learned_index_scores_temporal.comp.template"
                if temporal_index_scores is not None
                else "learned_index_scores.comp.template"
            ),
            {
                "HEAD_COUNT": str(heads),
                "HEAD_WIDTH": str(width),
                "COMPRESSION_RATIO": str(ratio),
                "MAX_COMPRESSED_POSITIONS": str(maximum),
                "CHUNK_POSITIONS": str(chunk),
                "SCORE_SCALE": scale,
            },
        )
    temporal_radix_topk = TEMPORAL_RADIX_TOPK_SHADER_PATTERN.fullmatch(shader_file)
    radix_topk = temporal_radix_topk or RADIX_TOPK_SHADER_PATTERN.fullmatch(shader_file)
    if radix_topk is not None:
        maximum, top_k, ratio, offset = map(int, radix_topk.groups())
        sort_capacity = 1 << (top_k - 1).bit_length()
        if (
            maximum <= 0
            or top_k <= 0
            or top_k > maximum
            or sort_capacity > 1024
            or ratio <= 0
        ):
            raise ModelCompileError(f"invalid radix top-k shape {shader_file!r}")
        return _render_template(
            source_dir
            / (
                "radix_topk_index_temporal.comp.template"
                if temporal_radix_topk is not None
                else "radix_topk_index.comp.template"
            ),
            {
                "MAX_SCORES": str(maximum),
                "TOP_K": str(top_k),
                "SORT_CAPACITY": str(sort_capacity),
                "COMPRESSION_RATIO": str(ratio),
                "INDEX_OFFSET": str(offset),
            },
        )
    temporal_chronological = TEMPORAL_CHRONOLOGICAL_INDEX_SHADER_PATTERN.fullmatch(
        shader_file
    )
    chronological = (
        temporal_chronological
        or CHRONOLOGICAL_INDEX_SHADER_PATTERN.fullmatch(shader_file)
    )
    if chronological is not None:
        maximum, ratio, offset = map(int, chronological.groups())
        if maximum <= 0 or ratio <= 0:
            raise ModelCompileError(
                f"invalid chronological compressed-index shape {shader_file!r}"
            )
        return _render_template(
            source_dir
            / (
                "chronological_compressed_index_temporal.comp.template"
                if temporal_chronological is not None
                else "chronological_compressed_index.comp.template"
            ),
            {
                "MAX_INDICES": str(maximum),
                "COMPRESSION_RATIO": str(ratio),
                "INDEX_OFFSET": str(offset),
            },
        )
    temporal_finalize = TEMPORAL_FINALIZE_SHADER_PATTERN.fullmatch(shader_file)
    finalize = temporal_finalize or FINALIZE_SHADER_PATTERN.fullmatch(shader_file)
    if finalize is None:
        return None
    (
        head_width_token,
        rotary_width_token,
        epsilon,
        theta,
        yarn_factor,
        correction_low,
        correction_high,
        attention_factor,
        layout,
        position_offset,
        block_columns_token,
    ) = finalize.groups()
    head_width = int(head_width_token)
    rotary_width = int(rotary_width_token)
    block_columns = int(block_columns_token)
    if (
        head_width <= 0
        or head_width > 1024
        or head_width & (head_width - 1)
        or rotary_width <= 0
        or rotary_width > head_width
        or rotary_width % 2
        or (head_width - rotary_width) % block_columns
        or block_columns != 64
    ):
        raise ModelCompileError(
            f"invalid compressed KV finalizer shape {shader_file!r}"
        )
    return _render_template(
        source_dir
        / (
            "compressed_kv_finalize_temporal.comp.template"
            if temporal_finalize is not None
            else "compressed_kv_finalize.comp.template"
        ),
        {
            "HEAD_WIDTH": str(head_width),
            "ROTARY_WIDTH": str(rotary_width),
            "NON_ROTARY_WIDTH": str(head_width - rotary_width),
            "NORM_EPS": epsilon,
            "ROPE_THETA": theta,
            "ROPE_YARN": "true" if yarn_factor is not None else "false",
            "ROPE_FACTOR": yarn_factor or "1.0",
            "ROPE_CORRECTION_LOW": correction_low or "0.0",
            "ROPE_CORRECTION_HIGH": correction_high or "1.0",
            "ROPE_ATTENTION_FACTOR": attention_factor or "1.0",
            "ROPE_INTERLEAVED": "true" if layout == "interleaved" else "false",
            "POSITION_OFFSET": position_offset,
            "QUANT_BLOCK_COLUMNS": str(block_columns),
        },
    )


def _render_template(path: Path, replacements: dict[str, str]) -> str:
    rendered = path.read_text()
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    unresolved = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", rendered)))
    if unresolved:
        raise ModelCompileError(
            f"shader template {path.name!r} has unresolved values {unresolved}"
        )
    return rendered


def _render_index_transform(
    source_dir: Path,
    match: re.Match[str],
    *,
    finalizer: bool,
    temporal: bool = False,
) -> str:
    groups = match.groups()
    if finalizer:
        (
            head_width_token,
            rotary_width_token,
            epsilon,
            theta,
            yarn_factor,
            correction_low,
            correction_high,
            attention_factor,
            layout,
            position_offset,
            block_columns_token,
        ) = groups
        head_count = 1
    else:
        (
            head_count_token,
            head_width_token,
            rotary_width_token,
            theta,
            yarn_factor,
            correction_low,
            correction_high,
            attention_factor,
            layout,
            block_columns_token,
        ) = groups
        head_count = int(head_count_token)
        epsilon = "0.0"
        position_offset = "0"
    head_width = int(head_width_token)
    rotary_width = int(rotary_width_token)
    block_columns = int(block_columns_token)
    if (
        head_count <= 0
        or head_width <= 0
        or head_width > 1024
        or head_width & (head_width - 1)
        or rotary_width <= 0
        or rotary_width > head_width
        or rotary_width % 2
        or block_columns != 32
        or head_width % block_columns
    ):
        raise ModelCompileError(
            f"invalid index vector transform shape {match.group(0)!r}"
        )
    template_name = (
        "compressed_index_kv_finalize_temporal.comp.template"
        if finalizer and temporal
        else "compressed_index_kv_finalize.comp.template"
        if finalizer
        else "index_vector_transform_temporal.comp.template"
        if temporal
        else "index_vector_transform.comp.template"
    )
    return _render_template(
        source_dir / template_name,
        {
            "HEAD_COUNT": str(head_count),
            "HEAD_WIDTH": str(head_width),
            "ROTARY_WIDTH": str(rotary_width),
            "NON_ROTARY_WIDTH": str(head_width - rotary_width),
            "NORM_EPS": epsilon,
            "ROPE_THETA": theta,
            "ROPE_YARN": "true" if yarn_factor is not None else "false",
            "ROPE_FACTOR": yarn_factor or "1.0",
            "ROPE_CORRECTION_LOW": correction_low or "0.0",
            "ROPE_CORRECTION_HIGH": correction_high or "1.0",
            "ROPE_ATTENTION_FACTOR": attention_factor or "1.0",
            "ROPE_INTERLEAVED": "true" if layout == "interleaved" else "false",
            "POSITION_OFFSET": position_offset,
            "QUANT_BLOCK_COLUMNS": str(block_columns),
        },
    )
