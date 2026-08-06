from __future__ import annotations

from typing import Any

from nerve.compilation import Json, ModelCompileError


MXFP4_TO_FP8_RESIDENT_DERIVATION = "mxfp4_e2m1_to_fp8_e4m3"
RESIDENT_DERIVATION_SCHEMA = "nerve.resident_derivation.v1"
MXFP4_TO_FP8_REQUIRED_FEATURES = (
    "shader_float8",
    "shader_int8",
    "shader_mixed_float_dot_product_float8_acc_float32",
)


def target_supports_mxfp4_to_fp8_residency(compiler_target: Json) -> bool:
    devices = compiler_target.get("devices")
    required = set(MXFP4_TO_FP8_REQUIRED_FEATURES)
    return bool(devices) and all(
        isinstance(device, dict)
        and required <= set(device.get("shader_features", []))
        for device in devices
    )


def mxfp4_to_fp8_resident_derivation(
    tensor: Any,
    compiler_target: Json,
) -> Json | None:
    if not target_supports_mxfp4_to_fp8_residency(compiler_target):
        return None
    if not isinstance(tensor, dict):
        return None
    quantization = tensor.get("quantization")
    source_byte_count = tensor.get("byte_count")
    if (
        tensor.get("dtype") != "I8"
        or not isinstance(quantization, dict)
        or quantization.get("format") != "mxfp4_e2m1"
        or quantization.get("bits") != 4
        or quantization.get("element_type") != "float"
        or quantization.get("values_per_byte") != 2
        or quantization.get("packing_axis") != 1
        or quantization.get("packing_order")
        != "low_nibble_then_high_nibble_along_k"
        or not isinstance(source_byte_count, int)
        or isinstance(source_byte_count, bool)
        or source_byte_count <= 0
    ):
        return None
    return {
        "schema": RESIDENT_DERIVATION_SCHEMA,
        "kind": MXFP4_TO_FP8_RESIDENT_DERIVATION,
        "source_byte_count": source_byte_count,
        "resident_byte_count": source_byte_count * 2,
        "required_features": list(MXFP4_TO_FP8_REQUIRED_FEATURES),
    }


def validate_resident_derivation(
    value: Any,
    *,
    source_byte_count: int,
    label: str,
) -> Json:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "source_byte_count",
        "resident_byte_count",
        "required_features",
    }:
        raise ModelCompileError(f"{label} has an invalid resident derivation contract")
    if value["schema"] != RESIDENT_DERIVATION_SCHEMA:
        raise ModelCompileError(f"{label} has an unsupported resident derivation schema")
    if value["kind"] != MXFP4_TO_FP8_RESIDENT_DERIVATION:
        raise ModelCompileError(f"{label} has an unsupported resident derivation kind")
    if (
        not isinstance(source_byte_count, int)
        or isinstance(source_byte_count, bool)
        or source_byte_count <= 0
    ):
        raise ModelCompileError(f"{label} has an invalid source byte count")
    if (
        not isinstance(value["source_byte_count"], int)
        or isinstance(value["source_byte_count"], bool)
        or value["source_byte_count"] <= 0
        or not isinstance(value["resident_byte_count"], int)
        or isinstance(value["resident_byte_count"], bool)
        or value["resident_byte_count"] <= 0
    ):
        raise ModelCompileError(f"{label} has invalid resident derivation sizes")
    if value["source_byte_count"] != source_byte_count:
        raise ModelCompileError(f"{label} resident derivation source size is inconsistent")
    if value["resident_byte_count"] != source_byte_count * 2:
        raise ModelCompileError(f"{label} resident derivation output size is inconsistent")
    if value["required_features"] != list(MXFP4_TO_FP8_REQUIRED_FEATURES):
        raise ModelCompileError(f"{label} resident derivation features are inconsistent")
    return value
