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
