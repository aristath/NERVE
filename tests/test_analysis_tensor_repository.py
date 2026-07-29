from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.analysis.tensor_repository import (
    PackageTensorRepository,
)


def _bf16(values: np.ndarray) -> bytes:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (bits >> 16).astype("<u2").tobytes()


def _package(
    root: Path,
    records: list[tuple[str, str, list[int], bytes, dict]],
) -> PackageTensorRepository:
    header = {}
    payload = bytearray()
    tensors = {}
    for name, dtype, shape, data, extensions in records:
        start = len(payload)
        payload.extend(data)
        stop = len(payload)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, stop],
        }
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "logical_shape": extensions.pop("logical_shape", shape),
            "source_file": "weights/test.safetensors",
            "data_offsets": [start, stop],
            **extensions,
        }
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
    ).encode()
    source = root / "weights" / "test.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    index = {
        "tensors": tensors,
        "source": {
            "weights_files": [
                {
                    "path": "weights/test.safetensors",
                    "safetensors_header_bytes": len(header_bytes),
                }
            ]
        },
    }
    return PackageTensorRepository(root, index)


def test_package_repository_decodes_bf16_and_block_scaled_fp8(tmp_path: Path):
    repository = _package(
        tmp_path,
        [
            (
                "bf16",
                "BF16",
                [2],
                _bf16(np.array([1.0, -2.0], dtype=np.float32)),
                {},
            ),
            (
                "fp8",
                "F8_E4M3",
                [1, 2],
                bytes((0x38, 0x40)),
                {},
            ),
            (
                "fp8_scale_inv",
                "BF16",
                [1, 1],
                _bf16(np.array([[2.0]], dtype=np.float32)),
                {},
            ),
        ],
    )
    bf16 = repository.observe(
        "bf16",
        exhaustive_element_limit=None,
        sampled_element_limit=4,
    )
    np.testing.assert_array_equal(bf16.values, [1.0, -2.0])
    fp8 = repository.observe(
        "fp8",
        exhaustive_element_limit=None,
        sampled_element_limit=4,
    )
    np.testing.assert_array_equal(fp8.values, [[2.0, 4.0]])
    assert fp8.effective_values is True


def test_package_repository_rejects_unscaled_fp8_as_ineffective(
    tmp_path: Path,
):
    repository = _package(
        tmp_path,
        [("fp8", "F8_E4M3", [1], bytes((0x38,)), {})],
    )

    with pytest.raises(ModelCompileError, match="effective algebraic values"):
        repository.observe(
            "fp8",
            exhaustive_element_limit=None,
            sampled_element_limit=1,
        )


def test_package_repository_decodes_native_q8_blocks(tmp_path: Path):
    block = bytearray(36)
    block[:2] = _bf16(np.array([0.5], dtype=np.float32))
    block[4:] = np.arange(-16, 16, dtype=np.int8).tobytes()
    repository = _package(
        tmp_path,
        [
            (
                "q8",
                "Q8_0",
                [1, 1, 9],
                bytes(block),
                {
                    "logical_shape": [1, 32],
                    "quantization": {
                        "format": "nerve_q8_0",
                        "group_size": 32,
                        "block_byte_count": 36,
                    },
                },
            )
        ],
    )
    observed = repository.observe(
        "q8",
        exhaustive_element_limit=None,
        sampled_element_limit=32,
    )
    np.testing.assert_array_equal(
        observed.values,
        np.arange(-16, 16, dtype=np.float32)[None, :] * 0.5,
    )


def test_package_repository_decodes_group_scaled_int4(tmp_path: Path):
    values = np.array([[-8, -1, 0, 1, 2, 3, 6, 7]], dtype=np.int16)
    nibbles = (values + 8).astype(np.uint32)
    packed = np.sum(
        nibbles * (np.uint32(1) << (np.arange(8, dtype=np.uint32) * 4)),
        axis=-1,
        dtype=np.uint32,
    )
    repository = _package(
        tmp_path,
        [
            (
                "int4",
                "I32",
                [1, 1],
                packed.astype("<u4").tobytes(),
                {
                    "logical_shape": [1, 8],
                    "quantization": {
                        "format": "compressed_tensors_pack_quantized",
                        "bits": 4,
                        "group_size": 4,
                        "symmetric": True,
                        "signed_offset": 8,
                        "scales": "int4_scale",
                    },
                },
            ),
            (
                "int4_scale",
                "BF16",
                [1, 2],
                _bf16(np.array([[0.5, 2.0]], dtype=np.float32)),
                {},
            ),
        ],
    )
    observed = repository.observe(
        "int4",
        exhaustive_element_limit=None,
        sampled_element_limit=8,
    )
    expected = values.astype(np.float32)
    expected[:, :4] *= 0.5
    expected[:, 4:] *= 2
    np.testing.assert_array_equal(observed.values, expected)


def test_package_repository_decodes_and_samples_autogptq_int4(tmp_path: Path):
    output_features = 10
    input_features = 16
    group_size = 4
    group_count = input_features // group_size
    quantized = np.fromfunction(
        lambda output, input_: (output * 3 + input_ * 5) % 16,
        (output_features, input_features),
        dtype=int,
    ).astype(np.uint32)
    zero_points = np.fromfunction(
        lambda group, output: 8 + (group + output) % 2,
        (group_count, output_features),
        dtype=int,
    ).astype(np.uint32)
    for group in range(group_count):
        outputs_with_zero_9 = zero_points[group] == 9
        quantized[
            outputs_with_zero_9,
            group * group_size : (group + 1) * group_size,
        ] = np.maximum(
            quantized[
                outputs_with_zero_9,
                group * group_size : (group + 1) * group_size,
            ],
            1,
        )
    scales = np.fromfunction(
        lambda group, output: (group + 1) * (output + 1) / 16,
        (group_count, output_features),
        dtype=float,
    ).astype(np.float16)

    qweight = np.zeros((input_features // 8, output_features), dtype=np.uint32)
    for input_index in range(input_features):
        qweight[input_index // 8] |= (
            quantized[:, input_index]
            << np.uint32((input_index % 8) * 4)
        )
    qzeros = np.zeros(
        (group_count, (output_features + 7) // 8),
        dtype=np.uint32,
    )
    for output_index in range(output_features):
        qzeros[:, output_index // 8] |= (
            (zero_points[:, output_index] - 1)
            << np.uint32((output_index % 8) * 4)
        )

    repository = _package(
        tmp_path,
        [
            (
                "qweight",
                "I32",
                list(qweight.shape),
                qweight.astype("<u4").tobytes(),
                {
                    "logical_shape": [output_features, input_features],
                    "quantization": {
                        "format": "auto_gptq",
                        "bits": 4,
                        "group_size": group_size,
                        "symmetric": True,
                        "zero_point_add": 1,
                        "qzeros": "qzeros",
                        "scales": "scales",
                    },
                },
            ),
            (
                "qzeros",
                "I32",
                list(qzeros.shape),
                qzeros.astype("<u4").tobytes(),
                {},
            ),
            (
                "scales",
                "F16",
                list(scales.shape),
                scales.astype("<f2").tobytes(),
                {},
            ),
        ],
    )
    expected = np.empty((output_features, input_features), dtype=np.float32)
    for input_index in range(input_features):
        group = input_index // group_size
        expected[:, input_index] = (
            quantized[:, input_index].astype(np.float32)
            - zero_points[group].astype(np.float32)
        ) * scales[group].astype(np.float32)

    exhaustive = repository.observe(
        "qweight",
        exhaustive_element_limit=None,
        sampled_element_limit=160,
    )
    np.testing.assert_array_equal(exhaustive.values, expected)

    sampled = repository.observe(
        "qweight",
        exhaustive_element_limit=1,
        sampled_element_limit=15,
    )
    assert sampled.exhaustive is False
    np.testing.assert_array_equal(
        sampled.values,
        expected[np.ix_(*sampled.sample_indices)],
    )

    canonical_quantized = quantized.copy()
    for input_index in range(input_features):
        canonical_quantized[:, input_index] += (
            8 - zero_points[input_index // group_size]
        )
    canonical_qweight = np.zeros_like(qweight)
    for input_index in range(input_features):
        canonical_qweight[input_index // 8] |= (
            canonical_quantized[:, input_index]
            << np.uint32((input_index % 8) * 4)
        )

    fixed_zero_repository = _package(
        tmp_path / "fixed-zero",
        [
            (
                "qweight",
                "I32",
                list(canonical_qweight.shape),
                canonical_qweight.astype("<u4").tobytes(),
                {
                    "logical_shape": [output_features, input_features],
                    "quantization": {
                        "format": "auto_gptq",
                        "bits": 4,
                        "group_size": group_size,
                        "symmetric": True,
                        "zero_point_add": 1,
                        "packing_layout": "input_major_packed_columns",
                        "zero_point_encoding": "fixed_8",
                        "qzeros": "qzeros",
                        "scales": "scales",
                    },
                },
            ),
            (
                "qzeros",
                "I32",
                list(qzeros.shape),
                qzeros.astype("<u4").tobytes(),
                {},
            ),
            (
                "scales",
                "F16",
                list(scales.shape),
                scales.astype("<f2").tobytes(),
                {},
            ),
        ],
    )
    fixed_zero = fixed_zero_repository.observe(
        "qweight",
        exhaustive_element_limit=None,
        sampled_element_limit=160,
    )
    np.testing.assert_array_equal(fixed_zero.values, expected)


def test_package_repository_sampling_is_deterministic_and_declared(tmp_path: Path):
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    repository = _package(
        tmp_path,
        [("large", "F32", [8, 8], values.astype("<f4").tobytes(), {})],
    )
    first = repository.observe(
        "large",
        exhaustive_element_limit=16,
        sampled_element_limit=9,
    )
    second = repository.observe(
        "large",
        exhaustive_element_limit=16,
        sampled_element_limit=9,
    )
    assert first.exhaustive is False
    assert first is second
    assert first.values.flags.writeable is False
    assert first.sample_indices == second.sample_indices
    np.testing.assert_array_equal(first.values, second.values)
    different_budget = repository.observe(
        "large",
        exhaustive_element_limit=16,
        sampled_element_limit=4,
    )
    assert different_budget is not first
    assert different_budget.values.size <= 4
