from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.resident_representations import (
    MXFP4_TO_FP8_REQUIRED_FEATURES,
    mxfp4_to_fp8_resident_derivation,
    target_supports_mxfp4_to_fp8_residency,
    validate_resident_derivation,
)
from nerve.resource_residency import compiled_immutable_resource, resource_identity


def _target(*, features: tuple[str, ...] = MXFP4_TO_FP8_REQUIRED_FEATURES) -> dict:
    return {"devices": [{"shader_features": list(features)}]}


def _mxfp4_tensor() -> dict:
    return {
        "dtype": "I8",
        "byte_count": 128,
        "quantization": {
            "format": "mxfp4_e2m1",
            "bits": 4,
            "element_type": "float",
            "values_per_byte": 2,
            "packing_axis": 1,
            "packing_order": "low_nibble_then_high_nibble_along_k",
        },
    }


def test_constructs_exact_fp8_candidate_only_when_every_target_device_supports_it() -> None:
    target = _target()
    target["devices"].append(
        {"shader_features": list(MXFP4_TO_FP8_REQUIRED_FEATURES)}
    )

    assert target_supports_mxfp4_to_fp8_residency(target)
    assert mxfp4_to_fp8_resident_derivation(_mxfp4_tensor(), target) == {
        "schema": "nerve.resident_derivation.v1",
        "kind": "mxfp4_e2m1_to_fp8_e4m3",
        "source_byte_count": 128,
        "resident_byte_count": 256,
        "required_features": list(MXFP4_TO_FP8_REQUIRED_FEATURES),
    }


def test_rejects_partial_hardware_support_and_non_mxfp4_storage() -> None:
    partial = _target(features=MXFP4_TO_FP8_REQUIRED_FEATURES[:-1])
    tensor = _mxfp4_tensor()

    assert not target_supports_mxfp4_to_fp8_residency(partial)
    assert mxfp4_to_fp8_resident_derivation(tensor, partial) is None
    tensor["quantization"]["packing_order"] = "high_nibble_first"
    assert mxfp4_to_fp8_resident_derivation(tensor, _target()) is None


def test_derivation_contract_fails_closed_on_size_or_feature_drift() -> None:
    derivation = mxfp4_to_fp8_resident_derivation(_mxfp4_tensor(), _target())
    assert derivation is not None
    validate_resident_derivation(
        derivation,
        source_byte_count=128,
        label="fixture",
    )

    corrupt = dict(derivation)
    corrupt["resident_byte_count"] = 255
    with pytest.raises(ModelCompileError, match="output size"):
        validate_resident_derivation(
            corrupt,
            source_byte_count=128,
            label="fixture",
        )

    corrupt = dict(derivation)
    corrupt["required_features"] = ["shader_float8"]
    with pytest.raises(ModelCompileError, match="features"):
        validate_resident_derivation(
            corrupt,
            source_byte_count=128,
            label="fixture",
        )


def test_hardware_capability_does_not_promote_a_resident_representation(
    tmp_path: Path,
) -> None:
    payload = bytes([0x10, 0x32, 0x54, 0x76])
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"H" * 16 + payload)
    tensor = {
        **_mxfp4_tensor(),
        "byte_count": len(payload),
        "source_file": artifact.name,
        "data_offsets": [0, len(payload)],
        "safetensors_header_bytes": 8,
        "data_sha256": sha256(payload).hexdigest(),
    }

    resource = compiled_immutable_resource(
        package_dir=tmp_path,
        tensor_index={"tensors": {"expert.weight": tensor}},
        tensor_name="expert.weight",
        lifetime="dynamic",
        source_headers={},
        artifact_byte_counts={artifact.name: artifact.stat().st_size},
    )

    assert resource["ranges"][0]["byte_count"] == len(payload)
    assert "resident_derivation" not in resource
    assert resource["compatibility"]["required_features"] == []
    assert resource["id"] == resource_identity(resource)
