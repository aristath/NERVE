from nerve.model_package_common import *
from nerve.quantized_layouts import (
    AUTO_GPTQ_FIXED_ZERO_8,
    AUTO_GPTQ_INPUT_MAJOR_PACKING,
    AUTO_GPTQ_PER_GROUP_ZERO,
    auto_gptq_packing,
    auto_gptq_zero_encoding,
)

import numpy as np


def write_compiled_auto_gptq_fixed_zero_8(
    *,
    tensor_name: str,
    info: Json,
    zero_info: Json,
    source: Path,
    zero_source: Path,
    destination: Path,
    layout: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[int, str]:
    if layout != ROW_MAJOR_LAYOUT:
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} requires row-major storage"
        )
    if str(info.get("dtype")) != "I32" or str(zero_info.get("dtype")) != "I32":
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} requires I32 weight and zero storage"
        )
    quantization = info.get("quantization")
    if (
        not isinstance(quantization, dict)
        or quantization.get("format") != "auto_gptq"
        or int(quantization.get("bits") or 0) != 4
        or quantization.get("symmetric") is not False
        or int(quantization.get("zero_point_add") or 0) != 1
        or auto_gptq_packing(info) != AUTO_GPTQ_INPUT_MAJOR_PACKING
        or auto_gptq_zero_encoding(info) != AUTO_GPTQ_PER_GROUP_ZERO
    ):
        raise ModelCompileError(
            f"tensor {tensor_name!r} is not source-layout AutoGPTQ INT4"
        )
    logical_shape = [
        int(value) for value in info.get("logical_shape", info.get("shape", []))
    ]
    storage_shape = [int(value) for value in info.get("shape", [])]
    if len(logical_shape) != 2:
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} requires a rank-2 logical shape"
        )
    output_features, input_features = logical_shape
    group_size = int(quantization.get("group_size") or 0)
    if (
        input_features % 8
        or group_size <= 0
        or input_features % group_size
        or group_size % 8
        or storage_shape != [input_features // 8, output_features]
    ):
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} storage shape {storage_shape} and "
            f"group size {group_size} do not encode logical shape {logical_shape}"
        )
    group_count = input_features // group_size
    zero_shape = [int(value) for value in zero_info.get("shape", [])]
    if zero_shape != [group_count, (output_features + 7) // 8]:
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} zero shape {zero_shape} does not "
            f"encode {output_features} outputs across {group_count} groups"
        )
    byte_count = int(info["byte_count"])
    if byte_count != math.prod(storage_shape) * 4:
        raise ModelCompileError(
            f"AutoGPTQ tensor {tensor_name!r} byte count {byte_count} does not "
            f"match storage shape {storage_shape}"
        )

    header = {
        "__metadata__": {
            "format": "nerve",
            "layout": layout,
            "packing_layout": AUTO_GPTQ_INPUT_MAJOR_PACKING,
            "zero_point_encoding": AUTO_GPTQ_FIXED_ZERO_8,
        },
        tensor_name: {
            "dtype": "I32",
            "shape": storage_shape,
            "data_offsets": [0, byte_count],
        },
    }
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    source_header_bytes = int(
        info.get("source_header_bytes") or read_safetensors_header(source)[0]
    )
    zero_header_bytes = int(
        zero_info.get("source_header_bytes")
        or read_safetensors_header(zero_source)[0]
    )
    source_start = 8 + source_header_bytes + int(info["data_offsets"][0])
    zero_start = 8 + zero_header_bytes + int(zero_info["data_offsets"][0])
    source_words = np.memmap(
        source,
        mode="r",
        dtype="<u4",
        offset=source_start,
        shape=tuple(storage_shape),
        order="C",
    )
    zero_words = np.memmap(
        zero_source,
        mode="r",
        dtype="<u4",
        offset=zero_start,
        shape=tuple(zero_shape),
        order="C",
    )
    output_indices = np.arange(output_features, dtype=np.uint32)
    packed_columns_per_group = group_size // 8
    data_digest = sha256()

    with destination.open("wb") as destination_handle:
        destination_handle.write(struct.pack("<Q", len(header_payload)))
        destination_handle.write(header_payload)
        for group in range(group_count):
            check_compile_cancelled(cancel_requested)
            packed_zero = zero_words[group, output_indices // 8]
            effective_zero = (
                (packed_zero >> ((output_indices % 8) * 4)) & np.uint32(15)
            ).astype(np.int16) + int(quantization.get("zero_point_add") or 0)
            source_block = np.asarray(
                source_words[
                    group
                    * packed_columns_per_group : (group + 1)
                    * packed_columns_per_group
                ],
                dtype=np.uint32,
            )
            if np.all(effective_zero == 8):
                canonical_block = source_block
            else:
                delta = (8 - effective_zero).astype(np.int16)
                canonical_block = np.zeros_like(source_block)
                for shift in range(0, 32, 4):
                    values = ((source_block >> shift) & np.uint32(15)).astype(
                        np.int16
                    )
                    values += delta[None, :]
                    if np.any(values < 0) or np.any(values > 15):
                        raise ModelCompileError(
                            f"AutoGPTQ tensor {tensor_name!r} cannot be represented "
                            "exactly with fixed zero point 8"
                        )
                    canonical_block |= values.astype(np.uint32) << np.uint32(shift)
            payload = canonical_block.astype("<u4", copy=False).tobytes()
            destination_handle.write(payload)
            data_digest.update(payload)
    del zero_words
    del source_words
    return len(header_payload), data_digest.hexdigest()
