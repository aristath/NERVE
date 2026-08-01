from __future__ import annotations

import math
from collections.abc import Iterable

from nerve.model_transpiler_types import Json, ModelTranspileError


HEAD_TENSORS = {
    "function": "hc_head_fn",
    "base": "hc_head_base",
    "scale": "hc_head_scale",
}

LAYER_TENSOR_SUFFIXES = {
    "attention": {
        "function": "hc_attn_fn",
        "base": "hc_attn_base",
        "scale": "hc_attn_scale",
    },
    "feed_forward": {
        "function": "hc_ffn_fn",
        "base": "hc_ffn_base",
        "scale": "hc_ffn_scale",
    },
}


def discover_stream_mixer(
    tensors: dict[str, Json], config: Json, *, hidden_size: int, prefix: str = ""
) -> Json | None:
    names = {
        role: f"{prefix}.{name}" if prefix else name
        for role, name in HEAD_TENSORS.items()
    }
    present = {role: name for role, name in names.items() if name in tensors}
    if not present:
        return None
    if present.keys() != HEAD_TENSORS.keys():
        raise ModelTranspileError(
            "incomplete hyper-connection head tensor set: expected "
            f"{sorted(names.values())}, found {sorted(present.values())}"
        )

    function_shape = _shape(tensors, names["function"])
    if len(function_shape) != 2 or function_shape[1] % hidden_size:
        raise ModelTranspileError(
            "hyper-connection head function must be a matrix whose input width is "
            "a positive multiple of hidden_size"
        )
    multiplicity = function_shape[1] // hidden_size
    if multiplicity <= 1 or function_shape[0] != multiplicity:
        raise ModelTranspileError(
            "hyper-connection head function must map N hidden lanes to N mixing weights"
        )
    configured_multiplicity = config.get("hc_mult")
    if (
        configured_multiplicity is not None
        and int(configured_multiplicity) != multiplicity
    ):
        raise ModelTranspileError(
            "hyper-connection multiplicity disagrees with the head function shape"
        )
    _require_shape(tensors, names["base"], [multiplicity])
    _require_shape(tensors, names["scale"], [1])
    _require_f32(tensors, names.values())

    sinkhorn_iterations = int(config.get("hc_sinkhorn_iters", 20))
    epsilon = float(config.get("hc_eps", 1e-6))
    if sinkhorn_iterations <= 0:
        raise ModelTranspileError(
            "hyper-connection Sinkhorn iteration count must be positive"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ModelTranspileError(
            "hyper-connection epsilon must be finite and positive"
        )
    return {
        "type": "sinkhorn_hyper_connection",
        "multiplicity": multiplicity,
        "sinkhorn_iterations": sinkhorn_iterations,
        "epsilon": epsilon,
        "head": names,
    }


def discover_layer_residual_mixer(
    tensors: dict[str, Json],
    *,
    prefix: str,
    hidden_size: int,
    stream_mixer: Json | None,
) -> tuple[Json | None, dict[str, str]]:
    discovered = {
        stage: {
            role: f"{prefix}.{suffix}"
            for role, suffix in roles.items()
            if f"{prefix}.{suffix}" in tensors
        }
        for stage, roles in LAYER_TENSOR_SUFFIXES.items()
    }
    present_names = [name for stage in discovered.values() for name in stage.values()]
    if stream_mixer is None:
        if present_names:
            raise ModelTranspileError(
                f"layer prefix {prefix!r} has hyper-connection tensors without a "
                "complete model head contract"
            )
        return None, {}

    expected = {
        stage: {role: f"{prefix}.{suffix}" for role, suffix in roles.items()}
        for stage, roles in LAYER_TENSOR_SUFFIXES.items()
    }
    if discovered != expected:
        expected_names = sorted(
            name for stage in expected.values() for name in stage.values()
        )
        raise ModelTranspileError(
            f"incomplete hyper-connection tensor set for layer prefix {prefix!r}: "
            f"expected {expected_names}, found {sorted(present_names)}"
        )

    multiplicity = int(stream_mixer["multiplicity"])
    mix_width = multiplicity * (2 + multiplicity)
    lane_width = multiplicity * hidden_size
    for stage in expected.values():
        _require_shape(tensors, stage["function"], [mix_width, lane_width])
        _require_shape(tensors, stage["base"], [mix_width])
        _require_shape(tensors, stage["scale"], [3])
        _require_f32(tensors, stage.values())

    parameters = {
        f"hyper_{stage}_{role}": name
        for stage, roles in expected.items()
        for role, name in roles.items()
    }
    return {
        "type": "sinkhorn_hyper_connection",
        "multiplicity": multiplicity,
        "sinkhorn_iterations": int(stream_mixer["sinkhorn_iterations"]),
        "epsilon": float(stream_mixer["epsilon"]),
        "attention": expected["attention"],
        "feed_forward": expected["feed_forward"],
    }, parameters


def _shape(tensors: dict[str, Json], name: str) -> list[int]:
    return [int(value) for value in tensors[name].get("shape", [])]


def _require_shape(tensors: dict[str, Json], name: str, expected: list[int]) -> None:
    actual = _shape(tensors, name)
    if actual != expected:
        raise ModelTranspileError(
            f"hyper-connection tensor {name!r} has shape {actual}, expected {expected}"
        )


def _require_f32(tensors: dict[str, Json], names: Iterable[str]) -> None:
    for name in names:
        if tensors[name].get("dtype") != "F32":
            raise ModelTranspileError(
                f"hyper-connection tensor {name!r} must use F32 parameters"
            )
