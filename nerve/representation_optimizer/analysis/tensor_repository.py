from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from nerve.compilation import Json, ModelCompileError, read_json


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
        self._tensors = {name: np.asarray(values) for name, values in tensors.items()}
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
            observed = values.astype(np.float32, copy=False)
            indices: tuple[tuple[int, ...], ...] = ()
        else:
            indices = _grid_indices(values.shape, sampled_element_limit)
            observed = values[np.ix_(*indices)].astype(np.float32, copy=False)
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
            dtype == "I32"
            and isinstance(quantization, dict)
            and quantization.get("format") == "compressed_tensors_pack_quantized"
            and int(quantization.get("bits", 0)) == 4
        ):
            values = self._decode_int4(info, logical_shape, quantization)
            return (_select(values, indices), True)

        storage_shape = tuple(int(value) for value in info.get("shape", []))
        raw = self._memmap(info, dtype, storage_shape)
        selected = _select(raw, indices)
        if dtype == "BF16":
            values = _bf16_to_f32(selected)
        elif dtype == "F8_E4M3":
            values = _e4m3fn_to_f32(selected)
        else:
            values = np.asarray(selected, dtype=np.float32)

        effective = True
        scale_name = f"{tensor_name}_scale_inv"
        if dtype == "F8_E4M3" and scale_name in self._tensors:
            scale_info = self._tensors[scale_name]
            scale_shape = tuple(int(value) for value in scale_info["shape"])
            scale_raw = self._memmap(scale_info, str(scale_info["dtype"]), scale_shape)
            scales = (
                _bf16_to_f32(scale_raw)
                if scale_info["dtype"] == "BF16"
                else np.asarray(scale_raw, dtype=np.float32)
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
            "I8": np.dtype("i1"),
            "U8": np.dtype("u1"),
            "I16": np.dtype("<i2"),
            "U16": np.dtype("<u2"),
            "I32": np.dtype("<i4"),
            "U32": np.dtype("<u4"),
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

    def _decode_int4(
        self,
        info: Json,
        logical_shape: tuple[int, ...],
        quantization: Json,
    ) -> np.ndarray:
        storage_shape = tuple(int(value) for value in info["shape"])
        packed = np.asarray(self._memmap(info, "I32", storage_shape), dtype=np.uint32)
        shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
        unpacked = ((packed[..., None] >> shifts) & np.uint32(0x0F)).astype(np.int16)
        unpacked -= int(quantization.get("signed_offset", 8))
        values = unpacked.reshape(logical_shape).astype(np.float32)
        scale_name = quantization.get("scales")
        if not isinstance(scale_name, str) or scale_name not in self._tensors:
            return values
        scale_info = self._tensors[scale_name]
        scale_shape = tuple(int(value) for value in scale_info["shape"])
        raw_scales = self._memmap(
            scale_info,
            str(scale_info["dtype"]),
            scale_shape,
        )
        scales = (
            _bf16_to_f32(raw_scales)
            if scale_info["dtype"] == "BF16"
            else np.asarray(raw_scales, dtype=np.float32)
        )
        group_size = int(quantization["group_size"])
        expanded = np.repeat(scales, group_size, axis=-1)
        return values * expanded[..., : logical_shape[-1]]

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
