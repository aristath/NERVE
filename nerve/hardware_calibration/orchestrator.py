from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Sequence

from nerve.compilation import Json, ModelCompileCancelled, ModelCompileError
from nerve.compiler_target import CompilerTarget, discover_compiler_target
from nerve.representation_optimizer.contracts import canonical_json_bytes

from .contracts import validate_calibration_run
from .planning import CalibrationPolicy, build_calibration_plan
from .publication import publish_calibration, validate_published_calibration
from .statistics import summarize_calibration_run


CALIBRATION_COLLECTION_SCHEMA = "nerve.optimizer.hardware_calibration_collection.v1"
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class CalibrationCollectionReport:
    destination: Path
    profile_count: int
    source_profile_ids: tuple[str, ...]
    calibrated_profile_ids: tuple[str, ...]

    def to_json(self) -> Json:
        return {
            "destination": str(self.destination),
            "profile_count": self.profile_count,
            "source_profile_ids": list(self.source_profile_ids),
            "calibrated_profile_ids": list(self.calibrated_profile_ids),
        }


def calibrate_hardware(
    destination: Path,
    *,
    runtime_bin: Path | None = None,
    calibrator_bin: Path | None = None,
    selected_devices: Sequence[str] = (),
    policy: CalibrationPolicy | None = None,
    cancel_requested: CancelCheck | None = None,
    target: CompilerTarget | None = None,
) -> CalibrationCollectionReport:
    cancelled = cancel_requested or (lambda: False)
    if cancelled():
        raise ModelCompileCancelled("hardware calibration was cancelled")
    compiler_target = target or discover_compiler_target(runtime_bin=runtime_bin)
    profiles = _selected_profiles(compiler_target, selected_devices)
    command = _calibrator_command(calibrator_bin)
    fingerprint = _calibrator_fingerprint(command, cancelled)

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    source_profile_ids: list[str] = []
    calibrated_profile_ids: list[str] = []
    entries: list[Json] = []
    try:
        for profile_document in profiles:
            if cancelled():
                raise ModelCompileCancelled("hardware calibration was cancelled")
            profile = profile_document.to_json()
            source_profile_ids.append(profile["profile_id"])
            plan = build_calibration_plan(
                profile,
                implementation_fingerprint=fingerprint,
                policy=policy,
            )
            profile_root = staging / "profiles" / profile["profile_id"]
            working_root = staging / ".working" / profile["profile_id"]
            working_root.mkdir(parents=True)
            plan_path = working_root / "plan.json"
            run_path = working_root / "run.json"
            artifact_directory = working_root / "artifacts"
            plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")
            invocation = [
                *command,
                "--plan",
                str(plan_path),
                "--output",
                str(run_path),
                "--artifacts",
                str(artifact_directory),
            ]
            device_index = _vulkan_device_index(profile)
            if device_index is not None:
                invocation.extend(["--vulkan-device-index", str(device_index)])
            _run_cancellable(invocation, cancelled)
            run = _read_json(run_path)
            validate_calibration_run(run)
            summary = summarize_calibration_run(profile, plan, run)
            if summary["coverage"]["missing_processes"]:
                raise ModelCompileError(
                    "hardware calibration produced an incomplete profile for "
                    f"{profile['hardware_identity']['name']}: "
                    f"{summary['coverage']['missing_processes']}"
                )
            publish_calibration(
                profile_root,
                plan=plan,
                run=run,
                summary=summary,
                artifact_directory=artifact_directory,
            )
            manifest = validate_published_calibration(profile_root)
            calibrated_profile_ids.append(summary["hardware_profile"]["profile_id"])
            manifest_path = profile_root / "manifest.json"
            entries.append(
                {
                    "source_profile_id": profile["profile_id"],
                    "calibrated_profile_id": summary["hardware_profile"]["profile_id"],
                    "relative_path": manifest_path.relative_to(staging).as_posix(),
                    "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                    "manifest": manifest,
                }
            )

        collection: Json = {
            "schema": CALIBRATION_COLLECTION_SCHEMA,
            "calibrator_fingerprint": fingerprint,
            "profiles": entries,
        }
        (staging / "collection.json").write_bytes(
            canonical_json_bytes(collection) + b"\n"
        )
        shutil.rmtree(staging / ".working", ignore_errors=True)
        _replace_directory_atomically(staging, destination)
        return CalibrationCollectionReport(
            destination=destination,
            profile_count=len(entries),
            source_profile_ids=tuple(source_profile_ids),
            calibrated_profile_ids=tuple(calibrated_profile_ids),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_calibration_collection(destination: Path) -> Json:
    destination = destination.resolve()
    collection = _read_json(destination / "collection.json")
    if set(collection) != {"schema", "calibrator_fingerprint", "profiles"}:
        raise ValueError("hardware-calibration collection fields are malformed")
    if collection["schema"] != CALIBRATION_COLLECTION_SCHEMA:
        raise ValueError(
            f"unsupported hardware-calibration collection {collection['schema']!r}"
        )
    relative_paths: list[str] = []
    source_ids: list[str] = []
    for entry in collection["profiles"]:
        if set(entry) != {
            "source_profile_id",
            "calibrated_profile_id",
            "relative_path",
            "manifest_sha256",
            "manifest",
        }:
            raise ValueError("hardware-calibration collection entry is malformed")
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("hardware-calibration collection path is unsafe")
        manifest_path = destination / relative
        if sha256(manifest_path.read_bytes()).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("hardware-calibration profile manifest digest mismatch")
        profile_root = manifest_path.parent
        manifest = validate_published_calibration(profile_root)
        if manifest != entry["manifest"]:
            raise ValueError("hardware-calibration embedded manifest differs from disk")
        if manifest["hardware_profile_id"] != entry["calibrated_profile_id"]:
            raise ValueError("hardware-calibration profile identity mismatch")
        relative_paths.append(entry["relative_path"])
        source_ids.append(entry["source_profile_id"])
    if relative_paths != sorted(set(relative_paths)):
        raise ValueError("hardware-calibration collection paths are not sorted and unique")
    if source_ids != sorted(set(source_ids)):
        raise ValueError("hardware-calibration source profiles are not sorted and unique")
    return collection


def _selected_profiles(
    target: CompilerTarget,
    selected_devices: Sequence[str],
) -> tuple[object, ...]:
    profiles = sorted(
        target.hardware_profiles,
        key=lambda profile: profile.to_json()["profile_id"],
    )
    if not selected_devices:
        if not profiles:
            raise ModelCompileError("hardware discovery returned no profiles to calibrate")
        return tuple(profiles)
    requested = set(selected_devices)
    selected = [
        profile
        for profile in profiles
        if profile.to_json()["profile_id"] in requested
        or profile.to_json()["hardware_identity"]["stable_device_id"] in requested
    ]
    matched = {
        selector
        for selector in requested
        if any(
            selector
            in {
                profile.to_json()["profile_id"],
                profile.to_json()["hardware_identity"]["stable_device_id"],
            }
            for profile in selected
        )
    }
    missing = sorted(requested - matched)
    if missing:
        raise ModelCompileError(f"unknown hardware calibration devices: {missing}")
    return tuple(selected)


def _calibrator_command(calibrator_bin: Path | None) -> list[str]:
    configured = calibrator_bin or _path_from_env("NERVE_CALIBRATOR_BIN")
    if configured is not None:
        return [str(configured)]
    repo_root = Path(__file__).resolve().parents[2]
    manifest = repo_root / "runtime-rs" / "Cargo.toml"
    if manifest.is_file():
        return [
            "cargo",
            "run",
            "--release",
            "--quiet",
            "--manifest-path",
            str(manifest),
            "--features",
            "vulkan",
            "--bin",
            "nerve-calibrate",
            "--",
        ]
    installed = shutil.which("nerve-calibrate")
    if installed:
        return [installed]
    raise ModelCompileError("could not find the NERVE hardware calibrator")


def _calibrator_fingerprint(command: list[str], cancelled: CancelCheck) -> str:
    completed = _run_cancellable([*command, "--fingerprint"], cancelled)
    fingerprint = completed.stdout.strip()
    if not fingerprint.startswith("nerve.hardware_calibrator_sha256.v1:"):
        raise ModelCompileError(
            f"hardware calibrator returned invalid fingerprint {fingerprint!r}"
        )
    return fingerprint


def _run_cancellable(
    command: list[str],
    cancelled: CancelCheck,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while process.poll() is None:
            if cancelled():
                process.terminate()
                process.wait()
                raise ModelCompileCancelled("hardware calibration was cancelled")
            time.sleep(0.05)
        stdout, stderr = process.communicate()
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout,
        stderr,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise ModelCompileError(
            "hardware calibrator failed"
            + (f": {diagnostic}" if diagnostic else "")
        )
    return completed


def _vulkan_device_index(profile: Json) -> int | None:
    binding = profile.get("runtime_bindings", {}).get("vulkan_runtime_binding")
    if binding is None:
        return None
    index = binding.get("physical_device_index")
    if not isinstance(index, int) or index < 0:
        raise ModelCompileError("Vulkan hardware profile has no valid runtime index")
    return index


def _read_json(path: Path) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"could not read calibration result {path}: {error}") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"calibration result {path} is not a JSON object")
    return document


def _replace_directory_atomically(staging: Path, destination: Path) -> None:
    previous = destination.with_name(f".{destination.name}.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.replace(previous)
    try:
        staging.replace(destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if previous.exists() and not destination.exists():
            previous.replace(destination)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None
