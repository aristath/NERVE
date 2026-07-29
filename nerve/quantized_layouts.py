AUTO_GPTQ_INPUT_MAJOR_PACKING = "input_major_packed_columns"
AUTO_GPTQ_PER_GROUP_ZERO = "packed_per_group_output"
AUTO_GPTQ_FIXED_ZERO_8 = "fixed_8"


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
