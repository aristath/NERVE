from __future__ import annotations

import json
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import contract_digest
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.stage import load_optimizer_stage
from nerve.representation_optimizer.staging.contracts import (
    CandidateBuildPlan,
    SOURCE_PACKAGE_SEAL_SCHEMA,
    staged_artifact_digest,
    validate_source_package_seal,
)


PACKAGE_MANIFEST_FILE = "vulkan_resident_package.json"


def seal_source_package(
    package_dir: Path,
    build_plan: CandidateBuildPlan,
    source_artifacts: PackageSourceArtifactResolver,
) -> Json:
    package_dir = package_dir.resolve()
    if source_artifacts.package_root != package_dir:
        raise ModelCompileError(
            "source package seal authority belongs to another package"
        )
    identity = _source_package_identity(package_dir)
    source_inputs = {}
    for declaration in build_plan.source_inputs:
        record = source_artifacts.source_seal_record(
            declaration["path"]
        )
        if record["digest"] != declaration["digest"]:
            raise ModelCompileError(
                f"candidate source input digest mismatch: "
                f"{declaration['path']!r}"
            )
        source_inputs[declaration["path"]] = record
    return {
        "schema": SOURCE_PACKAGE_SEAL_SCHEMA,
        **identity,
        "source_inputs": source_inputs,
    }


def _source_package_identity(package_dir: Path) -> Json:
    manifest_path = package_dir / PACKAGE_MANIFEST_FILE
    manifest_payload = _read_bytes(manifest_path, "package manifest")
    manifest = _json_object(manifest_payload, "package manifest")
    package_id = manifest.get("package_id")
    if not isinstance(package_id, str) or not package_id:
        raise ModelCompileError("source package manifest has no package_id")
    optimization_ref = manifest.get("representation_optimization_path")
    optimization_path = _package_path(
        package_dir,
        optimization_ref,
        "representation optimization",
    )
    stage_payload = _read_bytes(optimization_path, "representation optimizer stage")
    stage = load_optimizer_stage(optimization_path, package_dir=package_dir)

    integrity = manifest.get("artifact_integrity")
    return {
        "package_id": package_id,
        "manifest_digest": staged_artifact_digest(manifest_payload),
        "optimizer_stage_digest": staged_artifact_digest(stage_payload),
        "exact_baseline_digest": stage["exact_baseline"]["contract_digest"],
        "scope_catalog_digest": stage["scope_catalog"]["contract_digest"],
        "package_integrity_contract_digest": contract_digest(
            integrity if isinstance(integrity, dict) else {}
        ),
    }


def verify_source_package_seal(
    package_dir: Path,
    build_plan: CandidateBuildPlan,
    expected: Json,
) -> Json:
    package_dir = package_dir.resolve()
    validate_source_package_seal(expected, build_plan)
    if (
        {
            key: value
            for key, value in expected.items()
            if key not in {"schema", "source_inputs"}
        }
        != _source_package_identity(package_dir)
    ):
        raise ModelCompileError(
            "source package changed during isolated candidate construction"
        )
    declarations = {
        item["path"]: item["digest"]
        for item in build_plan.source_inputs
    }
    records = expected.get("source_inputs")
    if not isinstance(records, dict) or set(records) != set(declarations):
        raise ModelCompileError(
            "source package seal does not cover its exact source inputs"
        )
    for relative_path, expected_digest in declarations.items():
        record = records.get(relative_path)
        if (
            not isinstance(record, dict)
            or set(record) != {"digest", "signature"}
            or record["digest"] != expected_digest
            or record["signature"]
            != _file_signature(
                _package_path(
                    package_dir,
                    relative_path,
                    "candidate source input",
                )
            )
        ):
            raise ModelCompileError(
                "source package changed during isolated candidate "
                f"construction: {relative_path!r}"
            )
    return expected


def _package_path(package_dir: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
    ):
        raise ModelCompileError(f"{label} path is unsafe: {value!r}")
    path = package_dir / relative
    if not path.resolve().is_relative_to(package_dir):
        raise ModelCompileError(f"{label} path escapes the source package")
    return path


def _read_bytes(path: Path, label: str) -> bytes:
    _require_regular_file(path, label)
    return path.read_bytes()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ModelCompileError(f"{label} is not a regular file: {path}")


def _file_signature(path: Path) -> Json:
    _require_regular_file(path, "candidate source input")
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "byte_count": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "changed_ns": stat.st_ctime_ns,
    }


def _json_object(payload: bytes, label: str) -> Json:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"{label} must be a JSON object")
    return document
