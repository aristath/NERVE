from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileError
from nerve.representation_optimizer.providers import (
    PackageSourceArtifactResolver,
    ProviderProblem,
)
from nerve.representation_optimizer.staging.contracts import staged_artifact_digest
from tests.test_representation_optimizer_contracts import (
    contract_fixtures,
    hardware_profile_contract,
)


def _package(root: Path) -> tuple[Path, str, bytes]:
    tensor_name = "component.norm.weight"
    payload = struct.pack("<2H", 0x3F80, 0x4000)
    header = json.dumps(
        {
            tensor_name: {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    storage = root / "weights" / "norm.safetensors"
    storage.parent.mkdir(parents=True)
    storage.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (root / "tensors.json").write_text(
        json.dumps(
            {
                "schema": "nerve.tensor_index.v1",
                "source": {
                    "weights_files": [
                        {
                            "path": "weights/norm.safetensors",
                            "safetensors_header_bytes": len(header),
                            "metadata": {
                                "format": "nerve",
                                "layout": "row_major",
                            },
                        }
                    ]
                },
                "tensors": {
                    tensor_name: {
                        "dtype": "BF16",
                        "shape": [2],
                        "data_offsets": [0, len(payload)],
                        "parameter_count": 2,
                        "byte_count": len(payload),
                        "source_file": "weights/norm.safetensors",
                        "data_sha256": sha256(payload).hexdigest(),
                        "layout": "row_major",
                    }
                },
            }
        )
    )
    return root, tensor_name, payload


def test_package_source_artifacts_hash_lazily_and_cache_unchanged_files(
    tmp_path: Path,
):
    package, tensor_name, _ = _package(tmp_path / "package")
    hashed: list[str] = []

    def digest(path: Path) -> str:
        hashed.append(path.relative_to(package).as_posix())
        return staged_artifact_digest(path.read_bytes())

    resolver = PackageSourceArtifactResolver(package, file_digester=digest)
    assert hashed == []

    first = resolver.resolve_tensor(tensor_name)
    assert hashed == ["weights/norm.safetensors", "tensors.json"]
    assert first.source_inputs == (
        {
            "path": "tensors.json",
            "digest": staged_artifact_digest((package / "tensors.json").read_bytes()),
        },
        {
            "path": "weights/norm.safetensors",
            "digest": staged_artifact_digest(
                (package / "weights/norm.safetensors").read_bytes()
            ),
        },
    )
    assert resolver.resolve_tensor(tensor_name) == first
    assert hashed == ["weights/norm.safetensors", "tensors.json"]


def test_package_source_artifacts_read_exact_storage_and_copy_out_metadata(
    tmp_path: Path,
):
    package, tensor_name, payload = _package(tmp_path / "package")
    resolver = PackageSourceArtifactResolver(package)

    tensor = resolver.resolve_tensor(tensor_name)
    metadata = tensor.metadata
    metadata["dtype"] = "mutated"

    assert tensor.metadata["dtype"] == "BF16"
    assert tensor.payload_byte_count == len(payload)
    assert resolver.read_tensor_storage(tensor_name) == payload


def test_package_source_artifacts_rehash_drift_and_reject_corrupt_tensor(
    tmp_path: Path,
):
    package, tensor_name, _ = _package(tmp_path / "package")
    resolver = PackageSourceArtifactResolver(package)
    first = resolver.resolve_tensor(tensor_name)
    storage_path = package / first.storage.path
    mutated = bytearray(storage_path.read_bytes())
    mutated[-1] ^= 0x01
    storage_path.write_bytes(mutated)

    second = resolver.resolve_tensor(tensor_name)
    assert second.storage.digest != first.storage.digest
    with pytest.raises(ModelCompileError, match="data digest disagrees"):
        resolver.read_tensor_storage(tensor_name)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../outside",
        "/absolute",
        "./tensors.json",
        "weights/../tensors.json",
    ),
)
def test_package_source_artifacts_reject_unsafe_paths(
    tmp_path: Path,
    unsafe_path: str,
):
    package, _, _ = _package(tmp_path / "package")
    resolver = PackageSourceArtifactResolver(package)
    with pytest.raises(ModelCompileError, match="path is unsafe"):
        resolver.resolve_path(unsafe_path)


def test_package_source_artifacts_reject_symlink_even_inside_package(tmp_path: Path):
    package, _, _ = _package(tmp_path / "package")
    (package / "alias.json").symlink_to(package / "tensors.json")
    resolver = PackageSourceArtifactResolver(package)
    with pytest.raises(ModelCompileError, match="confined regular file"):
        resolver.resolve_path("alias.json")


def test_provider_context_exposes_only_explicit_source_artifact_authority(
    tmp_path: Path,
):
    package, tensor_name, payload = _package(tmp_path / "package")
    resolver = PackageSourceArtifactResolver(package)
    fixtures = contract_fixtures()
    problem = ProviderProblem.from_documents(
        package_id="fixture_package",
        scopes=[fixtures[0]],
        source_contracts=[fixtures[1]],
        evidence=[fixtures[2]],
        hardware_profile=hardware_profile_contract(),
        source_artifacts=resolver,
    )
    from nerve.representation_optimizer.descriptor_registry import (
        load_builtin_representation_descriptors,
    )

    descriptor = load_builtin_representation_descriptors().descriptors[0]
    context = problem.bind_descriptor(descriptor)
    assert context.source_artifacts.resolve_tensor(tensor_name).metadata["dtype"] == "BF16"
    assert context.source_artifacts.read_tensor_storage(tensor_name) == payload


def test_provider_context_reports_missing_source_artifact_authority():
    fixtures = contract_fixtures()
    problem = ProviderProblem.from_documents(
        package_id="fixture_package",
        scopes=[fixtures[0]],
        source_contracts=[fixtures[1]],
        evidence=[fixtures[2]],
        hardware_profile=hardware_profile_contract(),
    )
    from nerve.representation_optimizer.descriptor_registry import (
        load_builtin_representation_descriptors,
    )

    descriptor = load_builtin_representation_descriptors().descriptors[0]
    context = problem.bind_descriptor(descriptor)
    with pytest.raises(ModelCompileError, match="no source artifact resolver"):
        _ = context.source_artifacts
