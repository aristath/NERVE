from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    RELOWERING_REQUEST_SCHEMA,
    ContractDocument,
    canonical_json_bytes,
)
from nerve.representation_optimizer.representation_ir import (
    RepresentationGraphDocument,
)
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    staged_artifact_digest,
)


STAGED_CANDIDATE_INTEGRITY_SCHEMA = "nerve.optimizer.staged_candidate_integrity.v1"
STAGED_CANDIDATE_INTEGRITY_FILE = "integrity.json"


def write_staged_candidate_integrity(
    root: Path,
    *,
    candidate_id: str,
    construction_id: str,
) -> Json:
    if (root / STAGED_CANDIDATE_INTEGRITY_FILE).exists():
        raise ModelCompileError("candidate integrity manifest already exists")
    files = _file_records(root)
    document = {
        "schema": STAGED_CANDIDATE_INTEGRITY_SCHEMA,
        "candidate_id": candidate_id,
        "construction_id": construction_id,
        "files": files,
    }
    payload = canonical_json_bytes(document) + b"\n"
    path = root / STAGED_CANDIDATE_INTEGRITY_FILE
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(root)
    return document


def validate_staged_candidate(
    root: Path,
    *,
    expected_candidate_id: str | None = None,
    expected_build_plan: CandidateBuildPlan | None = None,
) -> Json:
    root = root.resolve()
    manifest_path = root / STAGED_CANDIDATE_INTEGRITY_FILE
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"candidate integrity manifest is unreadable: {error}"
        ) from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema",
        "candidate_id",
        "construction_id",
        "files",
    }:
        raise ModelCompileError("candidate integrity manifest fields are invalid")
    if manifest["schema"] != STAGED_CANDIDATE_INTEGRITY_SCHEMA:
        raise ModelCompileError("candidate integrity manifest schema is unsupported")
    if (
        expected_candidate_id is not None
        and manifest["candidate_id"] != expected_candidate_id
    ):
        raise ModelCompileError("staged candidate identity does not match")
    records = manifest["files"]
    if not isinstance(records, list):
        raise ModelCompileError("candidate integrity files must be a list")
    paths = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "byte_count",
            "sha256",
        }:
            raise ModelCompileError(
                f"candidate integrity record {index} is malformed"
            )
        relative = _safe_relative(record["path"])
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ModelCompileError(
                f"candidate integrity artifact is not a regular file: {relative}"
            )
        try:
            byte_count = path.stat().st_size
            digest = _file_sha256(path)
        except OSError as error:
            raise ModelCompileError(
                f"candidate artifact cannot be read: {relative}"
            ) from error
        if (
            record["byte_count"] != byte_count
            or record["sha256"] != digest
        ):
            raise ModelCompileError(
                f"candidate artifact failed integrity validation: {relative}"
            )
        paths.append(relative.as_posix())
    if paths != sorted(set(paths)):
        raise ModelCompileError(
            "candidate integrity paths must be sorted and unique"
        )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != STAGED_CANDIDATE_INTEGRITY_FILE
    }
    if set(paths) != actual:
        raise ModelCompileError(
            "candidate integrity manifest does not cover every staged artifact"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ModelCompileError("staged candidate must not contain symbolic links")

    graph = _read_object(root / "contracts" / "representation_graph.json")
    RepresentationGraphDocument.from_json(graph)
    candidate = _read_object(root / "contracts" / "candidate.json")
    if candidate.get("candidate_id") != manifest["candidate_id"]:
        raise ModelCompileError("staged candidate contract identity mismatch")
    target_lowering = _read_object(root / "contracts" / "target_lowering.json")
    if not isinstance(target_lowering.get("schema"), str):
        raise ModelCompileError("staged target lowering has no schema")
    ContractDocument.from_json(
        _read_object(root / "contracts" / "relowering_request.json"),
        expected_schema=RELOWERING_REQUEST_SCHEMA,
    )
    build_plan = CandidateBuildPlan.from_json(
        _read_object(root / "contracts" / "build_plan.json")
    )
    if expected_build_plan is not None and build_plan != expected_build_plan:
        raise ModelCompileError("staged candidate build plan changed")
    declared_paths = set(build_plan.output_paths)
    actual_output_paths = {
        path
        for path in actual
        if not path.startswith("contracts/")
    }
    if actual_output_paths != declared_paths:
        raise ModelCompileError(
            "staged candidate output artifacts do not match its build plan"
        )
    return manifest


def integrity_evidence(manifest: Json) -> Json:
    payload = canonical_json_bytes(manifest)
    return {
        "schema": manifest["schema"],
        "digest": staged_artifact_digest(payload),
        "file_count": len(manifest["files"]),
    }


def _file_records(root: Path) -> list[Json]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ModelCompileError(
                f"candidate staging contains a symbolic link: {path}"
            )
        if not path.is_file():
            continue
        try:
            byte_count = path.stat().st_size
            digest = _file_sha256(path)
        except OSError as error:
            raise ModelCompileError(
                f"candidate staging artifact cannot be read: {path}"
            ) from error
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "byte_count": byte_count,
                "sha256": digest,
            }
        )
    return records


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"staged contract is unreadable: {path}") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"staged contract must be an object: {path}")
    return document


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError("candidate integrity path must be non-empty")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
    ):
        raise ModelCompileError(f"candidate integrity path is unsafe: {value!r}")
    return relative


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
