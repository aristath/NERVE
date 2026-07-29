from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO

from nerve.compilation import ModelCompileError


_SHA256_BYTES = 32
_VERIFY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ConcreteResourceInterval:
    artifact_path: str
    byte_offset: int
    byte_count: int
    resource_id: str


@dataclass(frozen=True)
class PartitionRangeSeries:
    template_id: str
    resource_identity_seed: str
    artifact_path: str
    base_byte_offset: int
    stride_bytes: int
    byte_count: int
    partition_count: int
    digest_table_path: str
    digest_table_byte_offset: int
    digest_stride_bytes: int
    table_sha256: str


@dataclass(frozen=True)
class ResolvedResourceRange:
    artifact_path: str
    byte_offset: int
    byte_count: int
    alignment_bytes: int
    sha256: str


def validate_partition_range_storage(
    package_dir: Path,
    *,
    concrete_intervals: list[ConcreteResourceInterval],
    partition_series: list[PartitionRangeSeries],
) -> None:
    """Validate physical overlap and exact digest-table coverage.

    Identical dynamic series may be reused by multiple selectors or atomic
    templates. Any partial overlap, conflicting identity, digest alias, gap, or
    unreferenced table suffix fails closed.
    """

    artifact_intervals: dict[
        str, list[tuple[int, int, tuple[object, ...]]]
    ] = {}
    for interval in concrete_intervals:
        artifact_intervals.setdefault(interval.artifact_path, []).append(
            (
                interval.byte_offset,
                interval.byte_offset + interval.byte_count,
                ("concrete", interval.resource_id),
            )
        )

    table_contracts: dict[str, str] = {}
    digest_intervals: dict[
        str, list[tuple[int, int, tuple[object, ...]]]
    ] = {}
    for series in partition_series:
        if series.stride_bytes < series.byte_count:
            raise ModelCompileError(
                "partition range stride overlaps adjacent resources"
            )
        existing_table_digest = table_contracts.setdefault(
            series.digest_table_path, series.table_sha256
        )
        if existing_table_digest != series.table_sha256:
            raise ModelCompileError(
                f"partition digest table {series.digest_table_path!r} "
                "has conflicting integrity contracts"
            )
        for partition_index in range(series.partition_count):
            byte_offset = (
                series.base_byte_offset
                + partition_index * series.stride_bytes
            )
            digest_offset = (
                series.digest_table_byte_offset
                + partition_index * series.digest_stride_bytes
            )
            identity = (
                "dynamic",
                series.resource_identity_seed,
                series.artifact_path,
                byte_offset,
                series.byte_count,
                series.digest_table_path,
                digest_offset,
            )
            artifact_intervals.setdefault(series.artifact_path, []).append(
                (byte_offset, byte_offset + series.byte_count, identity)
            )
            digest_intervals.setdefault(
                series.digest_table_path, []
            ).append(
                (digest_offset, digest_offset + _SHA256_BYTES, identity)
            )

    for artifact_path, intervals in artifact_intervals.items():
        _reject_conflicting_intervals(
            intervals,
            f"compiled resource ranges overlap in artifact {artifact_path!r}",
        )

    for table_path, expected_sha256 in sorted(table_contracts.items()):
        path = package_dir / table_path
        actual_sha256, table_bytes = _file_sha256_and_size(
            path, "partition digest table"
        )
        if actual_sha256 != expected_sha256:
            raise ModelCompileError(
                f"partition digest table {table_path!r} does not match "
                "its SHA-256 contract"
            )
        covered = _deduplicated_sorted_intervals(
            digest_intervals.get(table_path, []),
            f"partition digest entries overlap in table {table_path!r}",
        )
        cursor = 0
        for start, end, _identity in covered:
            if start != cursor:
                raise ModelCompileError(
                    f"partition digest table {table_path!r} has a coverage "
                    f"gap at byte {cursor}"
                )
            cursor = end
        if cursor != table_bytes:
            raise ModelCompileError(
                f"partition digest table {table_path!r} covers {cursor} "
                f"of {table_bytes} bytes"
            )


def resolve_partition_range(
    package_dir: Path,
    *,
    artifact_path: str,
    base_byte_offset: int,
    stride_bytes: int,
    byte_count: int,
    alignment_bytes: int,
    digest_table_path: str,
    digest_table_byte_offset: int,
    digest_stride_bytes: int,
    partition_index: int,
) -> ResolvedResourceRange:
    _validate_relative_path(artifact_path, "partition artifact")
    _validate_relative_path(digest_table_path, "partition digest table")
    integer_fields = (
        partition_index,
        base_byte_offset,
        stride_bytes,
        byte_count,
        alignment_bytes,
        digest_table_byte_offset,
        digest_stride_bytes,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in integer_fields
    ):
        raise ModelCompileError("partition range resolution input is invalid")
    if (
        partition_index < 0
        or base_byte_offset < 0
        or stride_bytes < byte_count
        or byte_count <= 0
        or alignment_bytes <= 0
        or alignment_bytes & (alignment_bytes - 1)
        or base_byte_offset % alignment_bytes
        or stride_bytes % alignment_bytes
        or digest_table_byte_offset < 0
        or digest_table_byte_offset % _SHA256_BYTES
        or digest_stride_bytes != _SHA256_BYTES
    ):
        raise ModelCompileError("partition range resolution input is invalid")
    byte_offset = base_byte_offset + partition_index * stride_bytes
    if byte_offset % alignment_bytes:
        raise ModelCompileError(
            "resolved partition byte offset violates alignment"
        )
    digest_offset = (
        digest_table_byte_offset + partition_index * digest_stride_bytes
    )
    digest_path = package_dir / digest_table_path
    try:
        with digest_path.open("rb") as digest_file:
            digest_file.seek(digest_offset)
            digest = digest_file.read(_SHA256_BYTES)
    except OSError as error:
        raise ModelCompileError(
            f"cannot resolve partition digest from {digest_table_path!r}: {error}"
        ) from error
    if len(digest) != _SHA256_BYTES:
        raise ModelCompileError(
            f"partition digest at {digest_table_path}:{digest_offset} is truncated"
        )
    return ResolvedResourceRange(
        artifact_path=artifact_path,
        byte_offset=byte_offset,
        byte_count=byte_count,
        alignment_bytes=alignment_bytes,
        sha256=digest.hex(),
    )


def read_verified_resource_range(
    package_dir: Path, byte_range: ResolvedResourceRange
) -> bytes:
    _validate_relative_path(
        byte_range.artifact_path, "resolved resource artifact"
    )
    integer_fields = (
        byte_range.byte_offset,
        byte_range.byte_count,
        byte_range.alignment_bytes,
    )
    if (
        any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields
        )
        or byte_range.byte_offset < 0
        or byte_range.byte_count <= 0
        or byte_range.alignment_bytes <= 0
        or byte_range.alignment_bytes
        & (byte_range.alignment_bytes - 1)
        or byte_range.byte_offset % byte_range.alignment_bytes
        or len(byte_range.sha256) != _SHA256_BYTES * 2
        or any(
            character not in "0123456789abcdef"
            for character in byte_range.sha256
        )
    ):
        raise ModelCompileError("resolved resource range is invalid")
    path = package_dir / byte_range.artifact_path
    try:
        with path.open("rb") as source:
            source.seek(byte_range.byte_offset)
            payload = source.read(byte_range.byte_count)
    except OSError as error:
        raise ModelCompileError(
            f"cannot read compiled resource range "
            f"{byte_range.artifact_path}:{byte_range.byte_offset}: {error}"
        ) from error
    if len(payload) != byte_range.byte_count:
        raise ModelCompileError(
            f"compiled resource range {byte_range.artifact_path}:"
            f"{byte_range.byte_offset}+{byte_range.byte_count} is truncated"
        )
    actual = sha256(payload).hexdigest()
    if actual != byte_range.sha256:
        raise ModelCompileError(
            f"compiled resource range {byte_range.artifact_path}:"
            f"{byte_range.byte_offset}+{byte_range.byte_count} failed SHA-256"
        )
    return payload


def _file_sha256_and_size(path: Path, label: str) -> tuple[str, int]:
    digest = sha256()
    size = 0
    try:
        with path.open("rb") as source:
            _hash_chunks(source, digest)
            size = source.tell()
    except OSError as error:
        raise ModelCompileError(
            f"{label} {path!s} cannot be read: {error}"
        ) from error
    return digest.hexdigest(), size


def _hash_chunks(source: BinaryIO, digest) -> None:
    while payload := source.read(_VERIFY_CHUNK_BYTES):
        digest.update(payload)


def _validate_relative_path(value: str, label: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ModelCompileError(f"{label} must stay inside the compiled package")


def _reject_conflicting_intervals(
    intervals: list[tuple[int, int, tuple[object, ...]]],
    message: str,
) -> None:
    _deduplicated_sorted_intervals(intervals, message)


def _deduplicated_sorted_intervals(
    intervals: list[tuple[int, int, tuple[object, ...]]],
    message: str,
) -> list[tuple[int, int, tuple[object, ...]]]:
    ordered = sorted(set(intervals))
    previous: tuple[int, int, tuple[object, ...]] | None = None
    for current in ordered:
        if previous is not None and current[0] < previous[1]:
            raise ModelCompileError(message)
        previous = current
    return ordered
