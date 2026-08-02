AUTO_GPTQ_INPUT_MAJOR_PACKING = "input_major_packed_columns"
AUTO_GPTQ_PER_GROUP_ZERO = "packed_per_group_output"
AUTO_GPTQ_FIXED_ZERO_8 = "fixed_8"
MXFP4_FORMAT = "mxfp4_e2m1"
MXFP4_GROUP_SIZE = 32
MXFP4_VALUES_PER_BYTE = 2
MXFP4_PACKING_ORDER = "low_nibble_then_high_nibble_along_k"


def auto_gptq_packing(info: dict[str, object]) -> str:
    quantization = info.get("quantization")
    if not isinstance(quantization, dict):
        return ""
    return str(
        quantization.get(
            "packing_layout",
            AUTO_GPTQ_INPUT_MAJOR_PACKING,
        )
    )


def auto_gptq_zero_encoding(info: dict[str, object]) -> str:
    quantization = info.get("quantization")
    if not isinstance(quantization, dict):
        return ""
    return str(
        quantization.get(
            "zero_point_encoding",
            AUTO_GPTQ_PER_GROUP_ZERO,
        )
    )
