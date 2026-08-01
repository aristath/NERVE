from __future__ import annotations

import math
import re
from pathlib import Path

from nerve.model_package_assets import stream_control_binding_for_node
from nerve.model_package_common import ModelCompileError, ROW_MAJOR_LAYOUT, Json, shader_float_token
from nerve.model_package_tensors import (
    parameter_dtype_for_id,
    parameter_layout_for_id,
    parameter_shape_for_id,
)


POOL_SHADER_PATTERN = re.compile(
    r"learned_gated_kv_pool_bf16_f32_h(\d+)_d(\d+)_r(\d+)_c([12])\.comp"
)
FINALIZE_SHADER_PATTERN = re.compile(
    r"compressed_kv_finalize_f32_bf16_d(\d+)_r(\d+)_eps([0-9eE+.-]+)_"
    r"theta([0-9eE+.-]+)"
    r"(?:_yarn_f([0-9eE+.-]+)_lo([0-9eE+.-]+)_hi([0-9eE+.-]+)_a([0-9eE+.-]+))?"
    r"_(half|interleaved|proportional)_po(-?\d+)_qfp8e4m3b(\d+)\.comp"
)
CONDITIONAL_APPEND_SHADER_PATTERN = re.compile(
    r"conditional_append_state_bf16_d(\d+)_p(\d+)\.comp"
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
        or parameter_layout_for_id(circuit, params[0], tensor_index)
        != ROW_MAJOR_LAYOUT
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
            raise ModelCompileError("YaRN compressor RoPE has no compiled scaling profile")
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
        or state_port.get("type") != "append_only_attention_memory"
        or state_port.get("dtype") != "BF16"
        or state_port.get("growth") != f"one_per_{period}_activations"
    ):
        raise ModelCompileError(
            f"conditional state append node {node['id']!r} has an invalid contract"
        )
    binding = stream_control_binding_for_node(circuit, node)
    return f"conditional_append_state_bf16_d{width}_p{period}__sc{binding}.comp"


def render_latent_compression_shader(
    source_dir: Path,
    shader_file: str,
) -> str | None:
    pool = POOL_SHADER_PATTERN.fullmatch(shader_file)
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
            raise ModelCompileError(f"invalid learned compressor pool shape {shader_file!r}")
        return _render_template(
            source_dir / "learned_gated_kv_pool.comp.template",
            {
                "HIDDEN_SIZE": str(hidden_size),
                "HEAD_WIDTH": str(head_width),
                "COMPRESSION_RATIO": str(ratio),
                "LANE_COEFFICIENT": str(coefficient),
            },
        )
    conditional_append = CONDITIONAL_APPEND_SHADER_PATTERN.fullmatch(shader_file)
    if conditional_append is not None:
        width, period = map(int, conditional_append.groups())
        if width <= 0 or width % 2 or period <= 0:
            raise ModelCompileError(
                f"invalid conditional state append shape {shader_file!r}"
            )
        return _render_template(
            source_dir / "conditional_append_state_bf16.comp.template",
            {"ELEMENT_COUNT": str(width), "PERIOD": str(period)},
        )
    finalize = FINALIZE_SHADER_PATTERN.fullmatch(shader_file)
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
        raise ModelCompileError(f"invalid compressed KV finalizer shape {shader_file!r}")
    return _render_template(
        source_dir / "compressed_kv_finalize.comp.template",
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
