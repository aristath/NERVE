from __future__ import annotations

import os
import time
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable, Iterator

from nerve.compilation import Json, ModelCompileCancelled, ModelCompileError
from nerve.representation_optimizer.contracts import canonical_json_bytes
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import (
    CONSTRUCTION_PHASES,
    STAGED_ARTIFACT_DIGEST_SCHEMA,
    CandidateBuildPlan,
    staged_file_digest,
)


class CandidateConstructionContext:
    """Capability-limited interface exposed to candidate construction phases."""

    def __init__(
        self,
        *,
        package_dir: Path,
        staging_dir: Path,
        candidate: Json,
        representation_graph: Json,
        target_lowering: Json,
        build_plan: CandidateBuildPlan,
        source_artifacts: PackageSourceArtifactResolver,
        started_ns: int,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        self._package_dir = package_dir.resolve()
        self._staging_dir = staging_dir.resolve()
        self._candidate = deepcopy(candidate)
        self._representation_graph = deepcopy(representation_graph)
        self._target_lowering = deepcopy(target_lowering)
        self._build_plan = build_plan
        self._source_artifacts = source_artifacts
        self._started_ns = started_ns
        self._cancel_requested = cancel_requested
        self._phase: str | None = None
        self._declared_sources = {
            record["path"]: record["digest"] for record in build_plan.source_inputs
        }
        self._read_sources: set[str] = set()
        self._declared_outputs = {
            record["path"]: record for record in build_plan.outputs
        }
        self._written_outputs: set[str] = set()
        self._staging_bytes = 0
        self._peak_staging_bytes = 0
        self._transient_bytes = 0
        self._peak_transient_bytes = 0

    @property
    def candidate(self) -> Json:
        return deepcopy(self._candidate)

    @property
    def representation_graph(self) -> Json:
        return deepcopy(self._representation_graph)

    @property
    def target_lowering(self) -> Json:
        return deepcopy(self._target_lowering)

    @property
    def build_plan(self) -> CandidateBuildPlan:
        return self._build_plan

    @property
    def phase(self) -> str:
        if self._phase is None:
            raise ModelCompileError("candidate construction phase is not active")
        return self._phase

    @property
    def staging_bytes(self) -> int:
        return self._staging_bytes

    @property
    def peak_staging_bytes(self) -> int:
        return self._peak_staging_bytes

    @property
    def peak_transient_bytes(self) -> int:
        return self._peak_transient_bytes

    def begin_phase(self, phase: str) -> None:
        if phase not in CONSTRUCTION_PHASES:
            raise ModelCompileError(f"unknown candidate construction phase {phase!r}")
        self._phase = phase
        self.checkpoint()

    def end_phase(self) -> None:
        self.checkpoint()
        self._phase = None

    def checkpoint(self) -> None:
        if self._cancel_requested is not None and self._cancel_requested():
            raise ModelCompileCancelled("candidate construction cancelled")
        self._enforce_limits()

    def _enforce_limits(self) -> None:
        limits = self._build_plan.to_json()["resource_limits"]
        elapsed = time.monotonic_ns() - self._started_ns
        _enforce_limit(
            elapsed,
            limits["maximum_construction_time_ns"],
            "candidate construction time",
        )
        _enforce_limit(
            self._peak_transient_bytes,
            limits["maximum_temporary_bytes"],
            "candidate temporary memory",
        )
        _enforce_limit(
            self._peak_staging_bytes,
            limits["maximum_staging_bytes"],
            "candidate staging bytes",
        )

    def account_transient_bytes(self, current_bytes: int) -> None:
        if isinstance(current_bytes, bool) or current_bytes < 0:
            raise ModelCompileError(
                "candidate transient-memory accounting must be non-negative"
            )
        self._transient_bytes = current_bytes
        self._peak_transient_bytes = max(
            self._peak_transient_bytes,
            current_bytes,
        )
        self.checkpoint()

    def observe_total_staging_bytes(self, total_bytes: int) -> None:
        """Account for engine-owned files such as the integrity manifest."""
        if isinstance(total_bytes, bool) or total_bytes < self._staging_bytes:
            raise ModelCompileError("candidate total staging bytes are inconsistent")
        self._peak_staging_bytes = max(
            self._peak_staging_bytes,
            total_bytes,
        )
        self.checkpoint()

    def read_source_artifact(self, relative_path: str) -> bytes:
        return b"".join(self.iter_source_artifact(relative_path))

    def read_source_artifact_regions(
        self,
        relative_path: str,
        ranges: Iterable[tuple[int, int]],
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> tuple[bytes, ...]:
        requested = tuple(ranges)
        for index, region in enumerate(requested):
            if (
                not isinstance(region, tuple)
                or len(region) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in region
                )
                or region[0] < 0
                or region[1] < 0
            ):
                raise ModelCompileError(
                    f"candidate source region {index} must contain "
                    "non-negative integer offset and byte count"
                )
        if not requested:
            raise ModelCompileError(
                "candidate source region read requires at least one region"
            )
        expected = self._declared_sources.get(relative_path)
        if expected is None:
            raise ModelCompileError(
                f"candidate attempted to read undeclared source artifact "
                f"{relative_path!r}"
            )
        artifact = self._source_artifacts.resolve_path(relative_path)
        if artifact.digest != expected:
            raise ModelCompileError(
                f"candidate source artifact digest mismatch: {relative_path!r}"
            )
        self.checkpoint()
        payloads = self._source_artifacts.read_path_regions(
            relative_path,
            requested,
        )
        self._read_sources.add(relative_path)
        self.checkpoint()
        return payloads

    def read_source_artifact_region(
        self,
        relative_path: str,
        offset: int,
        byte_count: int,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> bytes:
        return self.read_source_artifact_regions(
            relative_path,
            ((offset, byte_count),),
            chunk_bytes=chunk_bytes,
        )[0]

    def iter_source_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterator[bytes]:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
            raise ModelCompileError(
                "candidate source chunk size must be a positive integer"
            )
        if chunk_bytes <= 0:
            raise ModelCompileError(
                "candidate source chunk size must be a positive integer"
            )
        expected = self._declared_sources.get(relative_path)
        if expected is None:
            raise ModelCompileError(
                f"candidate attempted to read undeclared source artifact "
                f"{relative_path!r}"
            )
        path = _contained_path(self._package_dir, relative_path, "source artifact")
        if path.is_symlink() or not path.is_file():
            raise ModelCompileError(
                f"candidate source artifact is not a regular file: {relative_path!r}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        digest = sha256()
        self.checkpoint()
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            while chunk := stream.read(chunk_bytes):
                digest.update(chunk)
                self.checkpoint()
                yield chunk
        actual = f"{STAGED_ARTIFACT_DIGEST_SCHEMA}:{digest.hexdigest()}"
        if actual != expected:
            raise ModelCompileError(
                f"candidate source artifact digest mismatch: {relative_path!r}"
            )
        self._read_sources.add(relative_path)
        self.checkpoint()

    def write_artifact(self, relative_path: str, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise ModelCompileError("candidate artifact payload must be bytes")
        self.write_artifact_stream(relative_path, (payload,))

    def artifact_reference(self, relative_path: str) -> str:
        """Return the candidate-root-relative reference for one output."""
        if relative_path not in self._declared_outputs:
            raise ModelCompileError(
                f"candidate referenced undeclared artifact {relative_path!r}"
            )
        return relative_path

    def write_artifact_stream(
        self,
        relative_path: str,
        chunks: Iterable[bytes],
    ) -> None:
        self.write_artifact_streams(
            (relative_path,),
            ((chunk,) for chunk in chunks),
        )

    def write_artifact_streams(
        self,
        relative_paths: tuple[str, ...],
        chunks: Iterable[tuple[bytes, ...]],
    ) -> None:
        if not relative_paths or len(set(relative_paths)) != len(relative_paths):
            raise ModelCompileError(
                "candidate artifact stream paths must be non-empty and unique"
            )
        paths = []
        for relative_path in relative_paths:
            declaration = self._declared_outputs.get(relative_path)
            if declaration is None:
                raise ModelCompileError(
                    "candidate attempted to write undeclared artifact "
                    f"{relative_path!r}"
                )
            if declaration["producer_phase"] != self.phase:
                raise ModelCompileError(
                    f"candidate artifact {relative_path!r} belongs to phase "
                    f"{declaration['producer_phase']!r}, not {self.phase!r}"
                )
            if relative_path in self._written_outputs:
                raise ModelCompileError(
                    f"candidate artifact {relative_path!r} was written more than once"
                )
            path = _contained_path(
                self._staging_dir,
                relative_path,
                "candidate artifact",
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            paths.append(path)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptors: list[int] = []
        streams = []
        created_paths: list[Path] = []
        try:
            for path in paths:
                descriptor = os.open(path, flags, 0o644)
                created_paths.append(path)
                descriptors.append(descriptor)
                streams.append(os.fdopen(descriptor, "wb", closefd=False))
            for index, payloads in enumerate(chunks):
                if not isinstance(payloads, tuple) or len(payloads) != len(streams):
                    raise ModelCompileError(
                        "candidate multi-artifact stream chunk "
                        f"{index} must contain {len(streams)} byte payloads"
                    )
                for stream, payload in zip(streams, payloads, strict=True):
                    if not isinstance(payload, bytes):
                        raise ModelCompileError(
                            "candidate multi-artifact stream chunk "
                            f"{index} contains a non-bytes payload"
                        )
                    stream.write(payload)
                    self._staging_bytes += len(payload)
                self._peak_staging_bytes = max(
                    self._peak_staging_bytes,
                    self._staging_bytes,
                )
                self.checkpoint()
            for stream in streams:
                stream.flush()
                os.fsync(stream.fileno())
            for parent in {path.parent for path in paths}:
                _fsync_directory(parent)
        except BaseException:
            for stream in streams:
                stream.close()
            streams.clear()
            for descriptor in descriptors:
                os.close(descriptor)
            descriptors.clear()
            for path in created_paths:
                path.unlink(missing_ok=True)
            for parent in {path.parent for path in paths}:
                _fsync_directory(parent)
            raise
        finally:
            for stream in streams:
                stream.close()
            for descriptor in descriptors:
                os.close(descriptor)
        self._written_outputs.update(relative_paths)
        self.checkpoint()

    def write_json_artifact(self, relative_path: str, document: Json) -> None:
        self.write_artifact(
            relative_path,
            canonical_json_bytes(document) + b"\n",
        )

    def write_internal_contract(self, relative_path: str, document: Json) -> None:
        if self._phase is not None:
            raise ModelCompileError(
                "internal candidate contracts are controlled by the staging engine"
            )
        path = _contained_path(
            self._staging_dir / "contracts",
            relative_path,
            "candidate internal contract",
        )
        payload = canonical_json_bytes(document) + b"\n"
        _write_durable(path, payload)
        self._staging_bytes += len(payload)
        self._peak_staging_bytes = max(self._peak_staging_bytes, self._staging_bytes)
        self._enforce_limits()

    def validate_complete(self) -> None:
        unread = sorted(set(self._declared_sources) - self._read_sources)
        if unread:
            raise ModelCompileError(
                f"candidate did not consume declared source artifacts: {unread}"
            )
        missing = sorted(set(self._declared_outputs) - self._written_outputs)
        if missing:
            raise ModelCompileError(
                f"candidate did not produce declared artifacts: {missing}"
            )
        actual = {
            path.relative_to(self._staging_dir).as_posix()
            for path in self._staging_dir.rglob("*")
            if path.is_file()
            and path.relative_to(self._staging_dir).parts[0] != "contracts"
        }
        if actual != self._written_outputs:
            extra = sorted(actual - self._written_outputs)
            raise ModelCompileError(
                f"candidate staging contains undeclared artifacts: {extra}"
            )
        self.checkpoint()

    def artifact_records(
        self,
        validation_results: dict[str, Json],
    ) -> list[Json]:
        records = []
        for relative_path, declaration in sorted(self._declared_outputs.items()):
            path = _contained_path(
                self._staging_dir, relative_path, "candidate artifact"
            )
            byte_count = path.stat().st_size
            records.append(
                {
                    "path": relative_path,
                    "digest": staged_file_digest(path),
                    "byte_count": byte_count,
                    "kind": declaration["kind"],
                    "lifetime": declaration["lifetime"],
                    "producer_phase": declaration["producer_phase"],
                    "resident_bytes": declaration["resident_bytes"],
                    "validation": deepcopy(validation_results[relative_path]),
                }
            )
        return records


def _enforce_limit(value: int, limit: int | None, label: str) -> None:
    if limit is not None and value > limit:
        raise ModelCompileError(f"{label} exceeded declared limit {limit}")


def _contained_path(root: Path, relative_path: str, label: str) -> Path:
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ModelCompileError(f"{label} path is unsafe: {relative_path!r}")
    path = root / relative
    resolved_parent = path.parent.resolve()
    if not resolved_parent.is_relative_to(root.resolve()):
        raise ModelCompileError(f"{label} path escapes its root: {relative_path!r}")
    cursor = root
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise ModelCompileError(
                f"{label} path crosses a symbolic link: {relative_path!r}"
            )
    return path


def _write_durable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ModelCompileError(f"candidate artifact already exists: {path}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
