from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import canonical_json_bytes

from .contracts import (
    CALIBRATION_MANIFEST_SCHEMA,
    validate_calibration_plan,
    validate_calibration_run,
    validate_calibration_summary,
)


def publish_calibration(
    destination: Path,
    *,
    plan: Json,
    run: Json,
    summary: Json,
    artifact_directory: Path | None = None,
) -> Path:
    validate_calibration_plan(plan)
    validate_calibration_run(run)
    validate_calibration_summary(summary)
    if run["plan_id"] != plan["plan_id"] or summary["plan_id"] != plan["plan_id"]:
        raise ValueError("calibration artifacts do not share one plan")
    if summary["run_id"] != run["run_id"]:
        raise ValueError("calibration summary does not reference the supplied run")

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        documents = {
            "plan.json": plan,
            "run.json": run,
            "summary.json": summary,
            "hardware_profile.json": summary["hardware_profile"],
        }
        files: list[Json] = []
        for name, document in documents.items():
            payload = canonical_json_bytes(document) + b"\n"
            path = staging / name
            _write_durable(path, payload)
            files.append(
                {
                    "path": name,
                    "byte_length": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                }
            )
        artifact_records = [
            artifact
            for workload in run["workloads"]
            for artifact in workload["artifacts"]
        ]
        if artifact_records and artifact_directory is None:
            raise ValueError(
                "calibration run references physical artifacts but no artifact directory was supplied"
            )
        artifact_paths: set[str] = set()
        for record in artifact_records:
            relative = Path(record["relative_path"])
            normalized = relative.as_posix()
            if normalized in artifact_paths:
                raise ValueError(f"duplicate calibration artifact path {normalized!r}")
            artifact_paths.add(normalized)
            assert artifact_directory is not None
            source = artifact_directory / relative
            payload = source.read_bytes()
            expected = record["digest"].removeprefix(
                "nerve.calibration_artifact_sha256.v1:"
            )
            if len(payload) != record["byte_length"]:
                raise ValueError(
                    f"calibration artifact length mismatch before publication: {relative}"
                )
            if sha256(payload).hexdigest() != expected:
                raise ValueError(
                    f"calibration artifact digest mismatch before publication: {relative}"
                )
            published_relative = Path("artifacts") / relative
            destination_path = staging / published_relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            _write_durable(destination_path, payload)
            files.append(
                {
                    "path": published_relative.as_posix(),
                    "byte_length": len(payload),
                    "sha256": expected,
                }
            )
        manifest: Json = {
            "schema": CALIBRATION_MANIFEST_SCHEMA,
            "plan_id": plan["plan_id"],
            "run_id": run["run_id"],
            "summary_id": summary["summary_id"],
            "hardware_profile_id": summary["hardware_profile"]["profile_id"],
            "files": sorted(files, key=lambda file: file["path"]),
        }
        manifest_payload = canonical_json_bytes(manifest) + b"\n"
        _write_durable(staging / "manifest.json", manifest_payload)
        _fsync_directory(staging)

        previous = destination.with_name(f".{destination.name}.previous")
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            destination.replace(previous)
        try:
            staging.replace(destination)
            _fsync_directory(destination.parent)
        except BaseException:
            if previous.exists() and not destination.exists():
                previous.replace(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_published_calibration(destination: Path) -> Json:
    destination = destination.resolve()
    manifest = _read_json(destination / "manifest.json")
    required = {
        "schema",
        "plan_id",
        "run_id",
        "summary_id",
        "hardware_profile_id",
        "files",
    }
    if set(manifest) != required:
        raise ValueError("calibration manifest fields are incomplete or unknown")
    if manifest["schema"] != CALIBRATION_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported calibration manifest {manifest['schema']!r}")
    paths: list[str] = []
    for record in manifest["files"]:
        if set(record) != {"path", "byte_length", "sha256"}:
            raise ValueError("calibration manifest file record is malformed")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("calibration manifest contains an unsafe path")
        path = destination / relative
        payload = path.read_bytes()
        if len(payload) != record["byte_length"]:
            raise ValueError(f"calibration artifact length mismatch: {relative}")
        if sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"calibration artifact digest mismatch: {relative}")
        paths.append(record["path"])
    if paths != sorted(set(paths)):
        raise ValueError("calibration manifest paths must be sorted and unique")

    plan = _read_json(destination / "plan.json")
    run = _read_json(destination / "run.json")
    summary = _read_json(destination / "summary.json")
    profile = _read_json(destination / "hardware_profile.json")
    validate_calibration_plan(plan)
    validate_calibration_run(run)
    validate_calibration_summary(summary)
    if profile != summary["hardware_profile"]:
        raise ValueError("published hardware profile differs from its summary")
    if (
        manifest["plan_id"] != plan["plan_id"]
        or manifest["run_id"] != run["run_id"]
        or manifest["summary_id"] != summary["summary_id"]
        or manifest["hardware_profile_id"] != profile["profile_id"]
    ):
        raise ValueError("calibration manifest identities do not match its artifacts")
    return manifest


def _write_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Json:
    document = json.loads(path.read_bytes())
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document
