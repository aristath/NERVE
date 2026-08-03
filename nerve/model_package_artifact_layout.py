from __future__ import annotations

import json
import os
import struct
from contextlib import ExitStack
from hashlib import blake2s, sha256
from pathlib import Path
from typing import Callable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled
from nerve.model_package_common import ROW_MAJOR_LAYOUT, package_artifact_path
from nerve.model_package_tensors import copy_exact_bytes
from nerve.model_transpiler import read_safetensors_header


def validate_artifact_affinity_groups(
    tensors: dict[str, Json], raw_groups: list[list[str]] | None
) -> list[list[str]]:
    return _validated_affinity_groups(
        tensors, [] if raw_groups is None else raw_groups
    )


def write_direct_tensor_affinity_bank(
    *,
    package_dir: Path,
    tensor_names: list[str],
    tensors: dict[str, Json],
    cancel_requested: Callable[[], bool] | None = None,
) -> tuple[Json, dict[str, Json]]:
    """Stream untransformed source tensors directly into one compiled bank."""

    relative_destination = _affinity_bank_path(tensor_names)
    destination = package_artifact_path(
        package_dir, relative_destination, "artifact affinity bank"
    )
    header_payload, offsets = _affinity_bank_header(tensor_names, tensors)
    results: dict[str, Json] = {}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with ExitStack() as source_stack, temporary.open("wb") as destination_handle:
            source_handles = {}
            destination_handle.write(struct.pack("<Q", len(header_payload)))
            destination_handle.write(header_payload)
            for tensor_name in tensor_names:
                check_compile_cancelled(cancel_requested)
                info = tensors[tensor_name]
                source = Path(str(info.get("source_file", "")))
                if not source.is_file():
                    raise ModelCompileError(
                        f"tensor source file does not exist: {source}"
                    )
                source_offsets = info.get("data_offsets")
                byte_count = info.get("byte_count")
                if (
                    not isinstance(source_offsets, list)
                    or len(source_offsets) != 2
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or value < 0
                        for value in source_offsets
                    )
                    or not isinstance(byte_count, int)
                    or isinstance(byte_count, bool)
                    or byte_count <= 0
                    or source_offsets[1] - source_offsets[0] != byte_count
                ):
                    raise ModelCompileError(
                        f"source tensor {tensor_name!r} cannot be affinity packed"
                    )
                source_header_bytes = info.get("source_header_bytes")
                if (
                    not isinstance(source_header_bytes, int)
                    or isinstance(source_header_bytes, bool)
                    or source_header_bytes <= 0
                ):
                    source_header_bytes = read_safetensors_header(source)[0]
                digest = sha256()
                source_handle = source_handles.get(source)
                if source_handle is None:
                    source_handle = source_stack.enter_context(source.open("rb"))
                    source_handles[source] = source_handle
                source_handle.seek(8 + source_header_bytes + int(source_offsets[0]))
                copy_exact_bytes(
                    source_handle,
                    destination_handle,
                    byte_count,
                    digest=digest,
                )
                results[tensor_name] = {
                    "source_file": relative_destination,
                    "data_offsets": offsets[tensor_name],
                    "data_sha256": digest.hexdigest(),
                    "safetensors_header_bytes": len(header_payload),
                }
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return (
        {
            "path": relative_destination,
            "safetensors_header_bytes": len(header_payload),
            "metadata": {
                "format": "nerve",
                "layout": ROW_MAJOR_LAYOUT,
                "storage_affinity": "physical_coaccess",
            },
        },
        results,
    )


def pack_tensor_artifacts_by_affinity(
    *,
    package_dir: Path,
    tensors: dict[str, Json],
    compiled_sources: list[Json],
    affinity_groups: list[list[str]] | None,
    cancel_requested: Callable[[], bool] | None = None,
) -> list[Json]:
    """Pack co-accessed compiled tensors into deterministic Safetensors banks."""

    groups = validate_artifact_affinity_groups(tensors, affinity_groups)
    if not groups:
        return compiled_sources
    source_records = _compiled_source_records(compiled_sources)
    removed_paths: set[str] = set()
    emitted_records: list[Json] = []

    for tensor_names in groups:
        check_compile_cancelled(cancel_requested)
        source_paths = {
            _compiled_tensor_source(
                tensor_name,
                tensors[tensor_name],
                source_records,
            )[0]
            for tensor_name in tensor_names
        }
        relative_destination = _affinity_bank_path(tensor_names)
        destination = package_artifact_path(
            package_dir, relative_destination, "artifact affinity bank"
        )
        header_bytes, offsets = _write_affinity_bank(
            package_dir=package_dir,
            destination=destination,
            tensor_names=tensor_names,
            tensors=tensors,
            source_records=source_records,
            cancel_requested=cancel_requested,
        )
        for tensor_name in tensor_names:
            info = tensors[tensor_name]
            info["source_file"] = relative_destination
            info["data_offsets"] = offsets[tensor_name]
            info["safetensors_header_bytes"] = header_bytes
        removed_paths.update(source_paths)
        emitted_records.append(
            {
                "path": relative_destination,
                "safetensors_header_bytes": header_bytes,
                "metadata": {
                    "format": "nerve",
                    "layout": ROW_MAJOR_LAYOUT,
                    "storage_affinity": "physical_coaccess",
                },
            }
        )

    referenced_paths = {
        info.get("source_file")
        for info in tensors.values()
        if isinstance(info, dict) and info.get("compile_only") is not True
    }
    removable_paths = removed_paths.difference(referenced_paths)
    for relative_path in sorted(removable_paths):
        path = package_artifact_path(
            package_dir, relative_path, "obsolete tensor artifact"
        )
        try:
            path.unlink()
        except OSError as error:
            raise ModelCompileError(
                f"cannot remove obsolete tensor artifact {relative_path!r}: {error}"
            ) from error

    retained_records = [
        record
        for record in compiled_sources
        if record.get("path") not in removable_paths
    ]
    return sorted([*retained_records, *emitted_records], key=lambda item: item["path"])


def _validated_affinity_groups(
    tensors: dict[str, Json], raw_groups: list[list[str]]
) -> list[list[str]]:
    if not isinstance(raw_groups, list):
        raise ModelCompileError("artifact affinity plan must be a list")
    groups: list[list[str]] = []
    seen: set[str] = set()
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, list):
            raise ModelCompileError(
                f"artifact affinity group {group_index} must be a list"
            )
        group: list[str] = []
        local_seen: set[str] = set()
        for tensor_name in raw_group:
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ModelCompileError(
                    f"artifact affinity group {group_index} has an invalid tensor"
                )
            if tensor_name not in tensors:
                raise ModelCompileError(
                    f"artifact affinity group references unknown tensor {tensor_name!r}"
                )
            if tensors[tensor_name].get("compile_only") is True:
                raise ModelCompileError(
                    f"artifact affinity group references compile-only tensor {tensor_name!r}"
                )
            if tensor_name in local_seen:
                raise ModelCompileError(
                    f"artifact affinity group repeats tensor {tensor_name!r}"
                )
            if tensor_name in seen:
                raise ModelCompileError(
                    f"tensor {tensor_name!r} appears in multiple artifact affinity groups"
                )
            local_seen.add(tensor_name)
            seen.add(tensor_name)
            group.append(tensor_name)
        if len(group) > 1:
            groups.append(group)
    return groups


def _compiled_source_records(compiled_sources: list[Json]) -> dict[str, Json]:
    records: dict[str, Json] = {}
    for index, record in enumerate(compiled_sources):
        if not isinstance(record, dict):
            raise ModelCompileError(f"compiled source record {index} is invalid")
        path = record.get("path")
        header_bytes = record.get("safetensors_header_bytes")
        if (
            not isinstance(path, str)
            or not path
            or path in records
            or not isinstance(header_bytes, int)
            or isinstance(header_bytes, bool)
            or header_bytes <= 0
        ):
            raise ModelCompileError(f"compiled source record {index} is ambiguous")
        records[path] = record
    return records


def _compiled_tensor_source(
    tensor_name: str,
    info: Json,
    source_records: dict[str, Json],
) -> tuple[str, int, int, int]:
    source_file = info.get("source_file")
    offsets = info.get("data_offsets")
    byte_count = info.get("byte_count")
    if (
        not isinstance(source_file, str)
        or source_file not in source_records
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in offsets
        )
        or not isinstance(byte_count, int)
        or isinstance(byte_count, bool)
        or byte_count <= 0
        or offsets[1] - offsets[0] != byte_count
    ):
        raise ModelCompileError(
            f"compiled tensor {tensor_name!r} cannot be affinity packed"
        )
    header_bytes = info.get(
        "safetensors_header_bytes",
        source_records[source_file]["safetensors_header_bytes"],
    )
    if (
        not isinstance(header_bytes, int)
        or isinstance(header_bytes, bool)
        or header_bytes <= 0
    ):
        raise ModelCompileError(
            f"compiled tensor {tensor_name!r} has no source header size"
        )
    return source_file, header_bytes, offsets[0], byte_count


def _write_affinity_bank(
    *,
    package_dir: Path,
    destination: Path,
    tensor_names: list[str],
    tensors: dict[str, Json],
    source_records: dict[str, Json],
    cancel_requested: Callable[[], bool] | None,
) -> tuple[int, dict[str, list[int]]]:
    header_payload, offsets = _affinity_bank_header(tensor_names, tensors)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as destination_handle:
            destination_handle.write(struct.pack("<Q", len(header_payload)))
            destination_handle.write(header_payload)
            for tensor_name in tensor_names:
                check_compile_cancelled(cancel_requested)
                info = tensors[tensor_name]
                source_file, source_header_bytes, source_offset, byte_count = (
                    _compiled_tensor_source(tensor_name, info, source_records)
                )
                source = package_artifact_path(
                    package_dir, source_file, f"tensor {tensor_name!r} source"
                )
                digest = sha256()
                with source.open("rb") as source_handle:
                    source_handle.seek(8 + source_header_bytes + source_offset)
                    copy_exact_bytes(
                        source_handle,
                        destination_handle,
                        byte_count,
                        digest=digest,
                    )
                expected_digest = info.get("data_sha256")
                if digest.hexdigest() != expected_digest:
                    raise ModelCompileError(
                        f"compiled tensor {tensor_name!r} failed affinity-pack integrity"
                    )
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(header_payload), offsets


def _affinity_bank_path(tensor_names: list[str]) -> str:
    bank_digest = blake2s(
        json.dumps(tensor_names, separators=(",", ":")).encode("utf-8"),
        digest_size=12,
    ).hexdigest()
    return f"weights/bank_{bank_digest}.safetensors"


def _affinity_bank_header(
    tensor_names: list[str], tensors: dict[str, Json]
) -> tuple[bytes, dict[str, list[int]]]:
    offsets: dict[str, list[int]] = {}
    cursor = 0
    header: Json = {
        "__metadata__": {
            "format": "nerve",
            "layout": ROW_MAJOR_LAYOUT,
            "storage_affinity": "physical_coaccess",
        }
    }
    for tensor_name in tensor_names:
        info = tensors[tensor_name]
        byte_count = int(info["byte_count"])
        offsets[tensor_name] = [cursor, cursor + byte_count]
        header[tensor_name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": offsets[tensor_name],
        }
        cursor += byte_count
    header_payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_payload += b" " * (-len(header_payload) % 8)
    return header_payload, offsets
