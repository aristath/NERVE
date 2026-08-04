from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.quantized_layouts import (
    AUTO_GPTQ_FIXED_ZERO_8,
    AUTO_GPTQ_INPUT_MAJOR_PACKING,
    AUTO_GPTQ_PER_GROUP_ZERO,
    MXFP4_FORMAT,
    MXFP4_PACKING_ORDER,
    auto_gptq_packing,
    auto_gptq_zero_encoding,
)


@dataclass(frozen=True)
class TensorObservation:
    tensor_name: str
    values: np.ndarray
    logical_shape: tuple[int, ...]
    storage_dtype: str
    exhaustive: bool
    sample_indices: tuple[tuple[int, ...], ...]
    effective_values: bool

    @property
    def logical_element_count(self) -> int:
        return math.prod(self.logical_shape)


class TensorRepository(Protocol):
    def metadata(self, tensor_name: str) -> Json: ...

    def observe(
        self,
        tensor_name: str,
        *,
        exhaustive_element_limit: int | None,
        sampled_element_limit: int,
    ) -> TensorObservation: ...


class InMemoryTensorRepository:
    def __init__(
        self,
        tensors: dict[str, np.ndarray],
        *,
        metadata: dict[str, Json] | None = None,
    ) -> None:
        self._tensors = {}
        for name, values in tensors.items():
            immutable = np.array(values, copy=True)
            immutable.flags.writeable = False
            self._tensors[name] = immutable
        self._metadata = metadata or {}

    def metadata(self, tensor_name: str) -> Json:
        values = self._tensors[tensor_name]
        return {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "logical_shape": list(values.shape),
            **self._metadata.get(tensor_name, {}),
        }

    def observe(
        self,
        tensor_name: str,
        *,
        exhaustive_element_limit: int | None,
        sampled_element_limit: int,
    ) -> TensorObservation:
        values = self._tensors[tensor_name]
        exhaustive = (
            exhaustive_element_limit is None or values.size <= exhaustive_element_limit
        )
        if exhaustive:
            observed = np.asarray(values, dtype=np.float32)
            indices: tuple[tuple[int, ...], ...] = ()
        else:
            indices = _grid_indices(values.shape, sampled_element_limit)
            observed = values[np.ix_(*indices)].astype(np.float32, copy=False)
        observed = np.asarray(observed)
        observed.flags.writeable = False
        return TensorObservation(
            tensor_name=tensor_name,
            values=np.asarray(observed),
            logical_shape=tuple(int(value) for value in values.shape),
            storage_dtype=str(values.dtype),
            exhaustive=exhaustive,
            sample_indices=indices,
            effective_values=True,
        )


class PackageTensorRepository:
    def __init__(self, package_dir: Path, tensor_index: Json | None = None) -> None:
        self.package_dir = package_dir
        self.tensor_index = tensor_index or read_json(package_dir / "tensors.json")
        raw_tensors = self.tensor_index.get("tensors")
        if not isinstance(raw_tensors, dict):
            raise ModelCompileError("compiled tensor index has no tensor map")
        self._tensors: dict[str, Json] = raw_tensors
        source = self.tensor_index.get("source", {})
        raw_files = source.get("weights_files", []) if isinstance(source, dict) else []
        self._header_bytes = {
            str(record["path"]): int(record["safetensors_header_bytes"])
            for record in raw_files
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("safetensors_header_bytes"), int)
        }
        self._observations: dict[
            tuple[str, int | None, int],
            TensorObservation,
        ] = {}

    def metadata(self, tensor_name: str) -> Json:
        try:
            return self._tensors[tensor_name]
        except KeyError as error:
            raise ModelCompileError(
                f"compiled package has no tensor {tensor_name!r}"
            ) from error

    def observe(
        self,
        tensor_name: str,
        *,
        exhaustive_element_limit: int | None,
        sampled_element_limit: int,
    ) -> TensorObservation:
        key = (
            tensor_name,
            exhaustive_element_limit,
            sampled_element_limit,
        )
        cached = self._observations.get(key)
        if cached is not None:
            return cached
        info = self.metadata(tensor_name)
        logical_shape = tuple(
            int(value) for value in info.get("logical_shape", info.get("shape", []))
        )
        if not logical_shape or any(value <= 0 for value in logical_shape):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has invalid logical shape"
            )
        logical_count = math.prod(logical_shape)
        exhaustive = (
            exhaustive_element_limit is None
            or logical_count <= exhaustive_element_limit
        )
        indices = (
            ()
            if exhaustive
            else _grid_indices(
                logical_shape,
                sampled_element_limit,
            )
        )
        values, effective = self._decode(
            tensor_name,
            info,
            logical_shape,
            indices,
        )
        observed_values = np.asarray(values, dtype=np.float32)
        observed_values.flags.writeable = False
        observation = TensorObservation(
            tensor_name=tensor_name,
            values=observed_values,
            logical_shape=logical_shape,
            storage_dtype=str(info.get("dtype", "")),
            exhaustive=exhaustive,
            sample_indices=indices,
            effective_values=effective,
        )
        self._observations[key] = observation
        return observation

    def _decode(
        self,
        tensor_name: str,
        info: Json,
        logical_shape: tuple[int, ...],
        indices: tuple[tuple[int, ...], ...],
    ) -> tuple[np.ndarray, bool]:
        dtype = str(info.get("dtype"))
        if dtype == "Q8_0":
            values = self._decode_q8_0(info, logical_shape)
            return (_select(values, indices), True)
        quantization = info.get("quantization")
        if (
            dtype == "I8"
            and isinstance(quantization, dict)
            and quantization.get("format") == MXFP4_FORMAT
        ):
            return (
                self._decode_mxfp4(
                    info,
                    logical_shape,
                    quantization,
                    indices,
                ),
                True,
            )
        if (
            dtype == "I32"
            and isinstance(quantization, dict)
            and quantization.get("format") == "compressed_tensors_pack_quantized"
            and int(quantization.get("bits", 0)) == 4
        ):
            return (
                self._decode_compressed_int4(
                    info,
                    logical_shape,
                    quantization,
                    indices,
                ),
                True,
            )
        if (
            dtype == "I32"
            and isinstance(quantization, dict)
            and quantization.get("format") == "auto_gptq"
            and int(quantization.get("bits", 0)) == 4
        ):
            return (
                self._decode_auto_gptq_int4(
                    info,
                    logical_shape,
                    quantization,
                    indices,
                ),
                True,
            )

        storage_shape = tuple(int(value) for value in info.get("shape", []))
        if storage_shape != logical_shape:
            raise ModelCompileError(
                f"analysis has no decoder for dtype {dtype!r}, quantization "
                f"{quantization!r}, storage shape {storage_shape}, and logical "
                f"shape {logical_shape}"
            )
        raw = self._memmap(info, dtype, storage_shape)
        selected = _select(raw, indices)
        if dtype == "BF16":
            values = _bf16_to_f32(selected)
        elif dtype == "F8_E4M3":
            values = _e4m3fn_to_f32(selected)
        else:
            values = np.asarray(selected, dtype=np.float32)

        effective = True
        scale = self._fp8_scale(tensor_name, info, logical_shape)
        if dtype == "F8_E4M3" and scale is not None:
            scale_name, scale_info = scale
            scale_shape = tuple(int(value) for value in scale_info["shape"])
            scale_raw = self._memmap(scale_info, str(scale_info["dtype"]), scale_shape)
            scale_dtype = str(scale_info["dtype"])
            scales = (
                _e8m0_to_f32(scale_raw)
                if scale_dtype == "F8_E8M0"
                else _to_f32(scale_raw, scale_dtype)
            )
            scale_indices = _scale_indices(
                logical_shape,
                scale_shape,
                indices,
            )
            values = values * _select(scales, scale_indices)
        elif dtype == "F8_E4M3":
            effective = False
            raise ModelCompileError(
                f"FP8 tensor {tensor_name!r} has no scale tensor; "
                "effective algebraic values are unavailable"
            )
        return values, effective

    def _fp8_scale(
        self,
        tensor_name: str,
        info: Json,
        logical_shape: tuple[int, ...],
    ) -> tuple[str, Json] | None:
        quantization = info.get("quantization")
        declared = []
        if isinstance(quantization, dict):
            for key in ("execution_scales", "scales", "source_scales"):
                value = quantization.get(key)
                if isinstance(value, str) and value:
                    declared.append(value)
        declared.append(f"{tensor_name}_scale_inv")
        if tensor_name.endswith(".weight"):
            declared.append(tensor_name.removesuffix(".weight") + ".scale")
        candidates = tuple(dict.fromkeys(declared))
        for scale_name in candidates:
            scale_info = self._tensors.get(scale_name)
            if not isinstance(scale_info, dict):
                continue
            scale_dtype = str(scale_info.get("dtype"))
            scale_shape = tuple(int(value) for value in scale_info.get("shape", []))
            if (
                len(scale_shape) != len(logical_shape)
                or any(value <= 0 for value in scale_shape)
                or any(
                    scale_dimension > logical_dimension
                    for logical_dimension, scale_dimension in zip(
                        logical_shape,
                        scale_shape,
                        strict=True,
                    )
                )
            ):
                continue
            if scale_dtype == "F8_E8M0":
                expected = (
                    (*logical_shape[:-2],)
                    + (
                        math.ceil(logical_shape[-2] / 128),
                        math.ceil(logical_shape[-1] / 128),
                    )
                    if len(logical_shape) >= 2
                    else ()
                )
                if scale_shape != expected:
                    continue
            elif scale_dtype not in {"BF16", "F16", "F32", "F64"}:
                continue
            return scale_name, scale_info
        return None

    def _memmap(
        self,
        info: Json,
        dtype: str,
        shape: tuple[int, ...],
    ) -> np.memmap:
        source_ref = info.get("source_file")
        if not isinstance(source_ref, str) or not source_ref:
            raise ModelCompileError("compiled tensor has no source file")
        relative = Path(source_ref)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelCompileError("compiled tensor source must stay inside package")
        header_bytes = self._header_bytes.get(source_ref)
        if header_bytes is None:
            raise ModelCompileError(
                f"compiled tensor source {source_ref!r} has no header metadata"
            )
        offsets = [int(value) for value in info.get("data_offsets", [])]
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ModelCompileError("compiled tensor has invalid data offsets")
        numpy_dtype = {
            "BF16": np.dtype("<u2"),
            "F16": np.dtype("<f2"),
            "F32": np.dtype("<f4"),
            "F64": np.dtype("<f8"),
            "F8_E4M3": np.dtype("u1"),
            "F8_E8M0": np.dtype("u1"),
            "I8": np.dtype("i1"),
            "U8": np.dtype("u1"),
            "I16": np.dtype("<i2"),
            "U16": np.dtype("<u2"),
            "I32": np.dtype("<i4"),
            "U32": np.dtype("<u4"),
            "I64": np.dtype("<i8"),
            "U64": np.dtype("<u8"),
        }.get(dtype)
        if numpy_dtype is None:
            raise ModelCompileError(f"analysis cannot decode tensor dtype {dtype!r}")
        expected = math.prod(shape) * numpy_dtype.itemsize
        if expected != offsets[1] - offsets[0]:
            raise ModelCompileError(
                f"compiled tensor storage shape and byte count disagree for {source_ref}"
            )
        return np.memmap(
            self.package_dir / relative,
            mode="r",
            dtype=numpy_dtype,
            offset=8 + header_bytes + offsets[0],
            shape=shape,
            order="C",
        )

    def _decode_q8_0(
        self,
        info: Json,
        logical_shape: tuple[int, ...],
    ) -> np.ndarray:
        if len(logical_shape) != 2 or logical_shape[1] % 32:
            raise ModelCompileError("Q8_0 analysis requires aligned rank-2 tensor")
        raw = self._memmap_bytes(info)
        blocks = np.asarray(raw).reshape(-1, 36)
        scales = _bf16_to_f32(blocks[:, :2].copy().view("<u2")).reshape(-1, 1)
        quantized = blocks[:, 4:].copy().view(np.int8).astype(np.float32)
        return (quantized * scales).reshape(logical_shape)

    def _decode_compressed_int4(
        self,
        info: Json,
        logical_shape: tuple[int, ...],
        quantization: Json,
        indices: tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        if len(logical_shape) < 2:
            raise ModelCompileError(
                "compressed-tensors INT4 analysis requires a rank-2 or higher tensor"
            )
        storage_shape = tuple(int(value) for value in info["shape"])
        expected_storage_shape = (
            *logical_shape[:-1],
            (logical_shape[-1] + 7) // 8,
        )
        if storage_shape != expected_storage_shape:
            raise ModelCompileError(
                "compressed-tensors INT4 storage shape "
                f"{storage_shape} does not encode logical shape {logical_shape}"
            )
        logical_indices = _logical_indices(logical_shape, indices)
        packed_indices = (
            *logical_indices[:-1],
            logical_indices[-1] // 8,
        )
        packed = np.asarray(
            self._memmap(info, "I32", storage_shape)[np.ix_(*packed_indices)],
            dtype=np.uint32,
        )
        shifts = (logical_indices[-1] % 8) * np.uint32(4)
        shifts = shifts.reshape((1,) * (len(logical_shape) - 1) + (-1,))
        values = ((packed >> shifts) & np.uint32(0x0F)).astype(np.int16)
        values -= int(quantization.get("signed_offset", 8))
        scale_name = quantization.get("scales")
        if not isinstance(scale_name, str) or scale_name not in self._tensors:
            raise ModelCompileError(
                "compressed-tensors INT4 analysis requires its scale tensor"
            )
        scale_info = self._tensors[scale_name]
        scale_shape = tuple(int(value) for value in scale_info["shape"])
        group_size = int(quantization.get("group_size", 0))
        expected_scale_shape = (
            *logical_shape[:-1],
            (logical_shape[-1] + group_size - 1) // group_size,
        ) if group_size > 0 else ()
        if group_size <= 0 or scale_shape != expected_scale_shape:
            raise ModelCompileError(
                "compressed-tensors INT4 scale shape "
                f"{scale_shape} is incompatible with logical shape "
                f"{logical_shape} and group size {group_size}"
            )
        raw_scales = self._memmap(
            scale_info,
            str(scale_info["dtype"]),
            scale_shape,
        )
        scale_indices = (
            *logical_indices[:-1],
            logical_indices[-1] // group_size,
        )
        scales = _to_f32(
            raw_scales[np.ix_(*scale_indices)],
            str(scale_info["dtype"]),
        )
        return values.astype(np.float32) * scales

    def _decode_mxfp4(
        self,
        info: Json,
        logical_shape: tuple[int, ...],
        quantization: Json,
        indices: tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        if len(logical_shape) < 2:
            raise ModelCompileError("MXFP4 analysis requires a rank-2 or higher tensor")
        storage_shape = tuple(int(value) for value in info["shape"])
        expected_storage_shape = (*logical_shape[:-1], logical_shape[-1] // 2)
        if (
            logical_shape[-1] % 2
            or storage_shape != expected_storage_shape
            or int(quantization.get("bits", 0)) != 4
            or int(quantization.get("values_per_byte", 0)) != 2
            or int(quantization.get("packing_axis", -1)) != len(logical_shape) - 1
            or quantization.get("packing_order") != MXFP4_PACKING_ORDER
        ):
            raise ModelCompileError(
                f"MXFP4 storage shape {storage_shape} or packing contract does not "
                f"encode logical shape {logical_shape}"
            )
        group_size = int(quantization.get("group_size", 0))
        scale_name = quantization.get("scales")
        if (
            group_size <= 0
            or logical_shape[-1] % group_size
            or not isinstance(scale_name, str)
            or scale_name not in self._tensors
            or quantization.get("scale_dtype") != "F8_E8M0"
            or quantization.get("scale_mode")
            != "power_of_two_per_output_row_k_group"
        ):
            raise ModelCompileError("MXFP4 analysis requires its E8M0 group-scale contract")
        scale_info = self._tensors[scale_name]
        scale_shape = tuple(int(value) for value in scale_info["shape"])
        expected_scale_shape = (
            *logical_shape[:-1],
            logical_shape[-1] // group_size,
        )
        if (
            str(scale_info.get("dtype")) != "F8_E8M0"
            or scale_shape != expected_scale_shape
        ):
            raise ModelCompileError(
                f"MXFP4 scale shape {scale_shape} is incompatible with logical "
                f"shape {logical_shape} and group size {group_size}"
            )

        logical_indices = _logical_indices(logical_shape, indices)
        packed_indices = (*logical_indices[:-1], logical_indices[-1] // 2)
        packed = np.asarray(
            self._memmap(info, "I8", storage_shape)[np.ix_(*packed_indices)],
            dtype=np.uint8,
        )
        shifts = ((logical_indices[-1] % 2) * np.uint8(4)).reshape(
            (1,) * (len(logical_shape) - 1) + (-1,)
        )
        nibbles = (packed >> shifts) & np.uint8(0x0F)
        magnitudes = np.array(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=np.float32,
        )[nibbles & np.uint8(7)]
        values = np.where(nibbles & np.uint8(8), -magnitudes, magnitudes)

        scale_indices = (
            *logical_indices[:-1],
            logical_indices[-1] // group_size,
        )
        raw_scales = self._memmap(
            scale_info,
            "F8_E8M0",
            scale_shape,
        )[np.ix_(*scale_indices)]
        return values * _e8m0_to_f32(raw_scales)

    def _decode_auto_gptq_int4(
        self,
        info: Json,
        logical_shape: tuple[int, ...],
        quantization: Json,
        indices: tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        if len(logical_shape) != 2:
            raise ModelCompileError("AutoGPTQ INT4 analysis requires a rank-2 tensor")
        output_features, input_features = logical_shape
        storage_shape = tuple(int(value) for value in info["shape"])
        packing_layout = auto_gptq_packing(info)
        expected_storage_shape = (
            ((input_features + 7) // 8, output_features)
            if packing_layout == AUTO_GPTQ_INPUT_MAJOR_PACKING
            else ()
        )
        if storage_shape != expected_storage_shape:
            raise ModelCompileError(
                f"AutoGPTQ INT4 storage shape {storage_shape} does not encode "
                f"logical shape {logical_shape}"
            )
        group_size = int(quantization.get("group_size", 0))
        if group_size <= 0:
            raise ModelCompileError("AutoGPTQ INT4 group size must be positive")
        group_count = (input_features + group_size - 1) // group_size
        qzeros_name = quantization.get("qzeros")
        scales_name = quantization.get("scales")
        zero_encoding = auto_gptq_zero_encoding(info)
        if not isinstance(scales_name, str) or scales_name not in self._tensors:
            raise ModelCompileError(
                "AutoGPTQ INT4 analysis requires a scale tensor"
            )
        qzeros_info = (
            self._tensors.get(qzeros_name)
            if isinstance(qzeros_name, str)
            else None
        )
        if (
            zero_encoding == AUTO_GPTQ_PER_GROUP_ZERO
            and not isinstance(qzeros_info, dict)
        ):
            raise ModelCompileError(
                "per-group AutoGPTQ INT4 analysis requires a zero-point tensor"
            )
        scales_info = self._tensors[scales_name]
        scales_shape = tuple(int(value) for value in scales_info["shape"])
        expected_qzeros_shape = (group_count, (output_features + 7) // 8)
        expected_scales_shape = (group_count, output_features)
        qzeros_shape = (
            tuple(int(value) for value in qzeros_info["shape"])
            if isinstance(qzeros_info, dict)
            else None
        )
        if scales_shape != expected_scales_shape or (
            zero_encoding == AUTO_GPTQ_PER_GROUP_ZERO
            and (
                str(qzeros_info.get("dtype")) != "I32"
                or qzeros_shape != expected_qzeros_shape
            )
        ):
            raise ModelCompileError(
                "AutoGPTQ INT4 auxiliary tensors are incompatible with logical "
                f"shape {logical_shape}: qzeros={qzeros_shape}, scales={scales_shape}"
            )

        output_indices, input_indices = _logical_indices(logical_shape, indices)
        input_groups = input_indices // group_size
        packed = np.asarray(
            self._memmap(info, "I32", storage_shape)[
                np.ix_(input_indices // 8, output_indices)
            ].T,
            dtype=np.uint32,
        )
        weight_shifts = ((input_indices % 8) * np.uint32(4))[None, :]
        quantized = ((packed >> weight_shifts) & np.uint32(0x0F)).astype(np.int16)

        if zero_encoding == AUTO_GPTQ_FIXED_ZERO_8:
            zero_points = np.full(
                (len(output_indices), len(input_indices)),
                8,
                dtype=np.int16,
            )
        elif zero_encoding == AUTO_GPTQ_PER_GROUP_ZERO:
            assert isinstance(qzeros_info, dict)
            qzeros = np.asarray(
                self._memmap(qzeros_info, "I32", qzeros_shape)[
                    np.ix_(input_groups, output_indices // 8)
                ],
                dtype=np.uint32,
            )
            zero_shifts = ((output_indices % 8) * np.uint32(4))[None, :]
            zero_points = (
                ((qzeros >> zero_shifts) & np.uint32(0x0F)).astype(np.int16)
                + int(quantization.get("zero_point_add", 0))
            ).T
        else:
            raise ModelCompileError(
                f"AutoGPTQ INT4 analysis does not support zero encoding "
                f"{zero_encoding!r}"
            )

        raw_scales = self._memmap(
            scales_info,
            str(scales_info["dtype"]),
            scales_shape,
        )[np.ix_(input_groups, output_indices)]
        scales = _to_f32(raw_scales, str(scales_info["dtype"]))
        return (quantized - zero_points).astype(np.float32) * scales.T

    def _memmap_bytes(self, info: Json) -> np.memmap:
        source_ref = info.get("source_file")
        if not isinstance(source_ref, str) or not source_ref:
            raise ModelCompileError("compiled tensor has no source file")
        relative = Path(source_ref)
        if relative.is_absolute() or ".." in relative.parts:
            raise ModelCompileError("compiled tensor source must stay inside package")
        header_bytes = self._header_bytes.get(source_ref)
        if header_bytes is None:
            raise ModelCompileError(
                f"compiled tensor source {source_ref!r} has no header metadata"
            )
        offsets = [int(value) for value in info.get("data_offsets", [])]
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ModelCompileError("compiled tensor has invalid data offsets")
        return np.memmap(
            self.package_dir / relative,
            mode="r",
            dtype=np.uint8,
            offset=8 + header_bytes + offsets[0],
            shape=(offsets[1] - offsets[0],),
        )


def _grid_indices(
    shape: tuple[int, ...],
    maximum_elements: int,
) -> tuple[tuple[int, ...], ...]:
    rank = len(shape)
    per_axis = max(1, int(maximum_elements ** (1.0 / rank)))
    counts = [min(dimension, per_axis) for dimension in shape]
    while math.prod(counts) < maximum_elements:
        candidates = [
            index
            for index, (count, dimension) in enumerate(zip(counts, shape, strict=True))
            if count < dimension
        ]
        if not candidates:
            break
        axis = max(candidates, key=lambda index: shape[index] / counts[index])
        proposed = counts.copy()
        proposed[axis] += 1
        if math.prod(proposed) > maximum_elements:
            break
        counts = proposed
    return tuple(
        tuple(
            int(value) for value in np.linspace(0, dimension - 1, count, dtype=np.int64)
        )
        for dimension, count in zip(shape, counts, strict=True)
    )


def _select(
    values: np.ndarray,
    indices: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    if not indices:
        return np.asarray(values)
    return np.asarray(values[np.ix_(*indices)])


def _logical_indices(
    logical_shape: tuple[int, ...],
    indices: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, ...]:
    if indices and len(indices) != len(logical_shape):
        raise ModelCompileError(
            "analysis sample index rank does not match the logical tensor rank"
        )
    selected = indices or tuple(
        tuple(range(dimension)) for dimension in logical_shape
    )
    return tuple(np.asarray(axis, dtype=np.int64) for axis in selected)


def _scale_indices(
    logical_shape: tuple[int, ...],
    scale_shape: tuple[int, ...],
    indices: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if len(logical_shape) != len(scale_shape):
        raise ModelCompileError("FP8 weight and scale tensors have incompatible ranks")
    logical_indices = indices or tuple(
        tuple(range(dimension)) for dimension in logical_shape
    )
    result = []
    for dimension, scale_dimension, selected in zip(
        logical_shape,
        scale_shape,
        logical_indices,
        strict=True,
    ):
        block = math.ceil(dimension / scale_dimension)
        result.append(
            tuple(min(value // block, scale_dimension - 1) for value in selected)
        )
    return tuple(result)


def _to_f32(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        return _bf16_to_f32(values)
    if dtype in {"F16", "F32", "F64"}:
        return np.asarray(values, dtype=np.float32)
    raise ModelCompileError(
        f"analysis cannot decode quantization scale dtype {dtype!r}"
    )


def _e8m0_to_f32(values: np.ndarray) -> np.ndarray:
    encoded = np.asarray(values, dtype=np.uint8)
    bits = encoded.astype(np.uint32) << np.uint32(23)
    bits = np.where(encoded == 0, np.uint32(0x00400000), bits)
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _e4m3fn_to_f32(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.uint8)
    sign = np.where((raw & np.uint8(0x80)) != 0, -1.0, 1.0).astype(np.float32)
    exponent = ((raw >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    mantissa = (raw & np.uint8(0x07)).astype(np.float32)
    decoded = np.zeros(raw.shape, dtype=np.float32)
    normal = exponent != 0
    decoded[normal] = (1.0 + mantissa[normal] / 8.0) * np.exp2(
        exponent[normal].astype(np.float32) - 7.0
    )
    decoded[~normal] = mantissa[~normal] / 512.0
    result = decoded * sign
    result[(exponent == 15) & (mantissa == 7)] = np.nan
    return result
