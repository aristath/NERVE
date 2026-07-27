from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Protocol

from nerve.compilation import Json, ModelCompileError, read_json
from nerve.representation_optimizer.staging.contracts import staged_file_digest


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    digest: str
    byte_count: int

    def source_input(self) -> Json:
        return {"path": self.path, "digest": self.digest}

    def to_json(self) -> Json:
        return {
            "path": self.path,
            "digest": self.digest,
            "byte_count": self.byte_count,
        }


@dataclass(frozen=True)
class SourceTensorArtifact:
    tensor_name: str
    _metadata: Json
    tensor_index: SourceArtifact
    storage: SourceArtifact
    safetensors_header_bytes: int
    payload_byte_offset: int
    payload_byte_count: int

    @property
    def metadata(self) -> Json:
        return deepcopy(self._metadata)

    @property
    def source_inputs(self) -> tuple[Json, ...]:
        return tuple(
            artifact.source_input()
            for artifact in sorted(
                (self.tensor_index, self.storage),
                key=lambda item: item.path,
            )
        )

    def to_json(self) -> Json:
        return {
            "tensor_name": self.tensor_name,
            "metadata": self.metadata,
            "tensor_index": self.tensor_index.to_json(),
            "storage": self.storage.to_json(),
            "safetensors_header_bytes": self.safetensors_header_bytes,
            "payload_byte_offset": self.payload_byte_offset,
            "payload_byte_count": self.payload_byte_count,
        }


class SourceArtifactResolver(Protocol):
    def resolve_path(self, package_relative_path: str) -> SourceArtifact: ...

    def resolve_tensor(self, tensor_name: str) -> SourceTensorArtifact: ...

    def read_tensor_storage(self, tensor_name: str) -> bytes: ...


class PackageSourceArtifactResolver:
    """Lazy, package-confined authority over provider source artifacts."""

    def __init__(
        self,
        package_dir: Path,
        *,
        file_digester: Callable[[Path], str] = staged_file_digest,
    ) -> None:
        root = package_dir.resolve()
        if not root.is_dir():
            raise ModelCompileError(
                f"provider source package is not a directory: {root}"
            )
        self._root = root
        self._file_digester = file_digester
        self._digest_cache: dict[str, tuple[tuple[int, ...], SourceArtifact]] = {}
        self._tensor_index = read_json(root / "tensors.json")
        raw_tensors = self._tensor_index.get("tensors")
        if not isinstance(raw_tensors, dict):
            raise ModelCompileError("compiled tensor index has no tensor map")
        self._tensors: dict[str, Json] = raw_tensors
        self._header_bytes: dict[str, int] | None = None

    def resolve_path(self, package_relative_path: str) -> SourceArtifact:
        normalized, path = self._package_file(package_relative_path)
        signature = _file_signature(path)
        cached = self._digest_cache.get(normalized)
        if cached is not None and cached[0] == signature:
            return cached[1]
        artifact = SourceArtifact(
            path=normalized,
            digest=self._file_digester(path),
            byte_count=signature[2],
        )
        self._digest_cache[normalized] = (signature, artifact)
        return artifact

    def resolve_tensor(self, tensor_name: str) -> SourceTensorArtifact:
        try:
            metadata = deepcopy(self._tensors[tensor_name])
        except KeyError as error:
            raise ModelCompileError(
                f"compiled package has no tensor {tensor_name!r}"
            ) from error
        if not isinstance(metadata, dict):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} metadata must be an object"
            )
        source_file = metadata.get("source_file")
        if not isinstance(source_file, str) or not source_file:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has no source file"
            )
        if self._header_bytes is None:
            self._header_bytes = _source_headers(self._tensor_index)
        header_bytes = self._header_bytes.get(source_file)
        if header_bytes is None:
            raise ModelCompileError(
                f"compiled tensor source {source_file!r} has no header metadata"
            )
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has invalid data offsets"
            )
        payload_offset = 8 + header_bytes + offsets[0]
        payload_bytes = offsets[1] - offsets[0]
        storage = self.resolve_path(source_file)
        if payload_offset + payload_bytes > storage.byte_count:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} exceeds its storage artifact"
            )
        data_digest = metadata.get("data_sha256")
        if not _sha256_hex(data_digest):
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} has invalid data_sha256"
            )
        return SourceTensorArtifact(
            tensor_name=tensor_name,
            _metadata=metadata,
            tensor_index=self.resolve_path("tensors.json"),
            storage=storage,
            safetensors_header_bytes=header_bytes,
            payload_byte_offset=payload_offset,
            payload_byte_count=payload_bytes,
        )

    def read_tensor_storage(self, tensor_name: str) -> bytes:
        tensor = self.resolve_tensor(tensor_name)
        _, path = self._package_file(tensor.storage.path)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            try:
                payload = os.pread(
                    descriptor,
                    tensor.payload_byte_count,
                    tensor.payload_byte_offset,
                )
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} cannot be read safely"
            ) from error
        if len(payload) != tensor.payload_byte_count:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} storage was truncated"
            )
        if sha256(payload).hexdigest() != tensor.metadata["data_sha256"]:
            raise ModelCompileError(
                f"compiled tensor {tensor_name!r} data digest disagrees"
            )
        return payload

    def _package_file(self, value: str) -> tuple[str, Path]:
        if not isinstance(value, str) or not value:
            raise ModelCompileError(
                "provider source artifact path must be a non-empty string"
            )
        relative = Path(value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or relative.as_posix() != value
        ):
            raise ModelCompileError(
                f"provider source artifact path is unsafe: {value!r}"
            )
        path = self._root / relative
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ModelCompileError(
                f"provider source artifact does not exist: {value!r}"
            ) from error
        if (
            path.is_symlink()
            or not path.is_file()
            or not resolved.is_relative_to(self._root)
        ):
            raise ModelCompileError(
                f"provider source artifact is not a confined regular file: {value!r}"
            )
        return relative.as_posix(), path


def _source_headers(tensor_index: Json) -> dict[str, int]:
    source = tensor_index.get("source")
    raw_files = source.get("weights_files") if isinstance(source, dict) else None
    if not isinstance(raw_files, list):
        raise ModelCompileError("compiled tensor index has no weights_files")
    headers: dict[str, int] = {}
    for index, record in enumerate(raw_files):
        if not isinstance(record, dict):
            raise ModelCompileError(
                f"compiled weights_files[{index}] must be an object"
            )
        path = record.get("path")
        header_bytes = record.get("safetensors_header_bytes")
        if (
            not isinstance(path, str)
            or not path
            or isinstance(header_bytes, bool)
            or not isinstance(header_bytes, int)
            or header_bytes <= 0
            or path in headers
        ):
            raise ModelCompileError(
                f"compiled weights_files[{index}] is invalid or duplicated"
            )
        headers[path] = header_bytes
    return headers


def _file_signature(path: Path) -> tuple[int, ...]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ModelCompileError(
            f"provider source artifact cannot be inspected: {path}"
        ) from error
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
