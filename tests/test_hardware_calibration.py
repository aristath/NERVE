from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from nerve.compilation import ModelCompileCancelled, ModelCompileError
from nerve.compiler_target import CompilerTarget
from nerve.hardware_calibration import orchestrator
from nerve.hardware_calibration.contracts import (
    CALIBRATION_RUN_SCHEMA,
    CalibrationContractError,
    validate_calibration_plan,
    validate_calibration_run,
)
from nerve.hardware_calibration.planning import (
    CalibrationPolicy,
    build_calibration_plan,
)
from nerve.hardware_calibration.publication import (
    publish_calibration,
    validate_published_calibration,
)
from nerve.hardware_calibration.statistics import summarize_calibration_run
from nerve.representation_optimizer.contracts import (
    ContractDocument,
    stable_contract_id,
)


FINGERPRINT = "nerve.hardware_calibrator_sha256.v1:" + "a" * 64


GPU_PROCESS_CATEGORIES = {
    "acceleration_structure_construction": "ray_traversal",
    "blending": "graphics",
    "command_queues": "scheduling",
    "cooperative_matrix": "arithmetic",
    "copy_engines": "transfer",
    "depth_stencil": "graphics",
    "device_cache_hierarchy": "memory",
    "device_generated_commands": "scheduling",
    "device_memory_bandwidth": "memory",
    "execution_graphs": "scheduling",
    "fixed_function_interpolation": "graphics",
    "indirect_work_generation": "scheduling",
    "occupancy_constraints": "scheduling",
    "packed_dot_product": "arithmetic",
    "parallel_collective_algorithms": "arithmetic",
    "rasterization": "graphics",
    "ray_traversal": "ray_traversal",
    "register_file": "memory",
    "resident_command_replay": "scheduling",
    "shader_atomics": "synchronization",
    "shader_scalar": "arithmetic",
    "shader_vector": "arithmetic",
    "subgroup_collectives": "control_flow",
    "synchronization": "synchronization",
    "texture_sampling": "sampling",
    "video_decode": "media",
    "video_encode": "media",
    "workgroup_shared_memory": "memory",
}


def hardware_profile(*, include_unknown: bool = False) -> dict[str, object]:
    processes = []
    for name, category in GPU_PROCESS_CATEGORIES.items():
        unavailable = name == "execution_graphs"
        process = {
            "name": name,
            "category": category,
            "availability": (
                "unavailable"
                if unavailable
                else "opaque"
                if name == "occupancy_constraints"
                else "available"
            ),
            "programmability": "none" if unavailable else "indirect" if name in {
                "device_cache_hierarchy",
                "device_memory_bandwidth",
                "occupancy_constraints",
                "register_file",
            } else "direct",
            "api": "vulkan",
            "operations": ["probe"],
            "numeric_formats": (
                ["bf16", "f16", "f32", "f8_e4m3"]
                if name
                in {
                    "cooperative_matrix",
                    "packed_dot_product",
                    "shader_scalar",
                    "shader_vector",
                }
                else []
            ),
            "required_extensions": [],
            "required_features": [],
            "limits": {},
            "properties": (
                {
                    "bfloat16_shapes": "16x16x16",
                    "float16_shapes": "16x16x16",
                    "float8_e4m3_shapes": "16x16x16",
                }
                if name == "cooperative_matrix"
                else {}
            ),
        }
        processes.append(process)
    if include_unknown:
        processes.append(
            {
                "name": "future_photonic_unit",
                "category": "arithmetic",
                "availability": "available",
                "programmability": "direct",
                "api": "vulkan",
                "operations": ["phase_shift"],
                "numeric_formats": [],
                "required_extensions": [],
                "required_features": [],
                "limits": {},
                "properties": {},
            }
        )
    processes.sort(key=lambda process: process["name"])
    identity = {
        "device_kind": "gpu",
        "vendor_id": "0x1002",
        "device_id": "0x7551",
        "stable_device_id": "vulkan-uuid:" + "1" * 32,
        "name": "Synthetic AMD GPU",
        "architecture": "gfx1201",
        "physical_location": "pci:0000:07:00.0",
    }
    memory_domains = [
        {
            "name": "vulkan_memory_type_000",
            "kind": "device_local_memory_type",
            "capacity_bytes": 32 * 1024 * 1024 * 1024,
            "host_visible": False,
            "device_local": True,
            "coherent": False,
            "cached": False,
            "minimum_alignment_bytes": 256,
            "properties": {
                "capacity_scope": "shared_vulkan_heap",
                "heap_index": "0",
            },
        }
    ]
    provenance = {
        "api": "vulkan",
        "api_version": "1.4.354",
        "driver": "radv",
        "driver_version": "26.1.5",
        "compiler": "nerve-test",
        "operating_system": "linux",
        "discovery_backend": "synthetic",
    }
    capability_extensions: dict[str, object] = {}
    identity_extensions: dict[str, object] = {}
    capability_class = stable_contract_id(
        "hardware_capability",
        {
            "device_kind": "gpu",
            "architecture": "gfx1201",
            "processes": processes,
            "memory_domains": memory_domains,
            "interconnects": [],
            "api": "vulkan",
            "api_version": "1.4.354",
            "capability_extensions": capability_extensions,
        },
    )
    profile_id = stable_contract_id(
        "hardware_profile",
        [
            identity,
            capability_class,
            provenance,
            identity_extensions,
            [],
        ],
    )
    return {
        "schema": "nerve.optimizer.hardware_process_profile.v1",
        "profile_id": profile_id,
        "hardware_identity": identity,
        "capability_class": capability_class,
        "processes": processes,
        "memory_domains": memory_domains,
        "interconnects": [],
        "measurements": [],
        "provenance": provenance,
        "capability_extensions": capability_extensions,
        "identity_extensions": identity_extensions,
        "runtime_bindings": {
            "vulkan_runtime_binding": {"physical_device_index": 2}
        },
    }


def calibration_policy() -> CalibrationPolicy:
    return CalibrationPolicy(
        warmup_iterations=1,
        steady_iterations=5,
        minimum_sample_duration_ns=1,
        sustained_window_duration_ms=1,
        sustained_window_count=2,
        confidence_level_ppm=950_000,
        maximum_relative_ci_width_ppm=200_000,
    )


def completed_run(plan: dict[str, object]) -> dict[str, object]:
    started_at = "2026-07-26T12:00:00Z"
    run_id = stable_contract_id(
        "calibration_run",
        plan["plan_id"],
        plan["hardware_profile_id"],
        started_at,
    )
    workloads = []
    for workload_number, workload in enumerate(plan["workloads"]):
        base = 1_000_000 + workload_number * 100
        samples = [
            {
                "sample_index": 0,
                "phase": "warmup",
                "duration_ns": base + 50_000,
                "device_duration_ns": None,
                "iterations": 1,
                "window_index": None,
                "thermal_millidegrees_celsius": None,
                "valid": True,
            }
        ]
        for index, offset in enumerate((-1000, -500, 0, 500, 1000), start=1):
            samples.append(
                {
                    "sample_index": index,
                    "phase": "steady",
                    "duration_ns": base + offset,
                    "device_duration_ns": base + offset - 100,
                    "iterations": 1,
                    "window_index": None,
                    "thermal_millidegrees_celsius": 50_000,
                    "valid": True,
                }
            )
        samples.extend(
            [
                {
                    "sample_index": 6,
                    "phase": "sustained",
                    "duration_ns": base,
                    "device_duration_ns": base - 100,
                    "iterations": 1,
                    "window_index": 0,
                    "thermal_millidegrees_celsius": 51_000,
                    "valid": True,
                },
                {
                    "sample_index": 7,
                    "phase": "sustained",
                    "duration_ns": base + 10_000,
                    "device_duration_ns": base + 9_900,
                    "iterations": 1,
                    "window_index": 1,
                    "thermal_millidegrees_celsius": 52_000,
                    "valid": True,
                },
            ]
        )
        workloads.append(
            {
                "workload_id": workload["workload_id"],
                "status": "completed",
                "construction_duration_ns": 10_000 + workload_number,
                "artifacts": [
                    {
                        "name": artifact["name"],
                        "kind": artifact["kind"],
                        "digest": (
                            "nerve.calibration_artifact_sha256.v1:"
                            + hashlib.sha256(b"x").hexdigest()
                        ),
                        "byte_length": 1,
                        "relative_path": (
                            f"{artifact['name']}_{workload['workload_id']}.bin"
                        ),
                    }
                    for artifact in workload["artifacts"]
                ],
                "samples": samples,
                "validation": {
                    "status": "passed",
                    "observed_digest": (
                        "nerve.calibration_output_sha256.v1:" + "b" * 64
                    ),
                    "maximum_error_ppm": 0,
                },
                "counters": {"queue_submits": 8},
                "diagnostics": [],
            }
        )
    return {
        "schema": CALIBRATION_RUN_SCHEMA,
        "run_id": run_id,
        "plan_id": plan["plan_id"],
        "hardware_profile_id": plan["hardware_profile_id"],
        "status": "completed",
        "started_at": started_at,
        "finished_at": "2026-07-26T12:05:00Z",
        "workloads": sorted(
            workloads,
            key=lambda workload: workload["workload_id"],
        ),
        "diagnostics": [],
    }


def test_plan_covers_every_exposed_process_and_is_deterministic() -> None:
    profile = hardware_profile()
    first = build_calibration_plan(
        profile,
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    second = build_calibration_plan(
        deepcopy(profile),
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )

    assert first == second
    covered = {
        name for workload in first["workloads"] for name in workload["process_names"]
    }
    assert covered == set(GPU_PROCESS_CATEGORIES) - {"execution_graphs"}
    assert first["excluded_processes"] == [
        {"process_name": "execution_graphs", "reason": "unavailable"}
    ]
    assert {workload["executor"] for workload in first["workloads"]} == {
        "vulkan_compute",
        "vulkan_dgc",
        "vulkan_graphics",
        "vulkan_ray",
        "vulkan_synchronization",
        "vulkan_transfer",
        "vulkan_video",
    }
    assert {
        workload["operation"] for workload in first["workloads"]
    }.issuperset(
        {
            "bitfield_mix",
            "sparse_compaction",
        }
    )
    assert {
        int(workload["regime"]["working_set_bytes"])
        for workload in first["workloads"]
        if workload["operation"] == "sequential_copy"
    }.issuperset({4_096, 32_768, 262_144, 2_097_152, 16_777_216, 134_217_728})
    assert {
        int(workload["regime"]["bytes"])
        for workload in first["workloads"]
        if workload["operation"] == "buffer_copy"
    } == {4_096, 1_048_576, 268_435_456}
    assert {
        int(workload["regime"]["round_trips"])
        for workload in first["workloads"]
        if workload["operation"] == "synchronization_round_trip"
    } == {1, 64, 4_096}
    validate_calibration_plan(first)


def test_plan_fails_closed_for_new_exposed_hardware_process() -> None:
    with pytest.raises(ValueError, match="no calibration provider"):
        build_calibration_plan(
            hardware_profile(include_unknown=True),
            implementation_fingerprint=FINGERPRINT,
            policy=calibration_policy(),
        )


def test_plan_identity_detects_workload_mutation() -> None:
    plan = build_calibration_plan(
        hardware_profile(),
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    plan["workloads"][0]["work"]["operations_per_iteration"] += 1

    with pytest.raises(CalibrationContractError, match="workload_id"):
        validate_calibration_plan(plan)


def test_completed_run_requires_validated_complete_samples() -> None:
    plan = build_calibration_plan(
        hardware_profile(),
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    run = completed_run(plan)
    validate_calibration_run(run)

    run["workloads"][0]["validation"]["status"] = "failed"
    with pytest.raises(CalibrationContractError, match="without passing validation"):
        validate_calibration_run(run)


def test_summary_preserves_raw_statistics_and_detects_sustained_decay() -> None:
    profile = hardware_profile()
    plan = build_calibration_plan(
        profile,
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    run = completed_run(plan)
    summary = summarize_calibration_run(profile, plan, run)

    assert summary["coverage"]["missing_processes"] == []
    assert summary["coverage"]["calibrated_processes"] == sorted(
        set(GPU_PROCESS_CATEGORIES) - {"execution_graphs"}
    )
    assert summary["hardware_profile"]["profile_id"] != profile["profile_id"]
    assert summary["hardware_profile"]["capability_class"] == profile["capability_class"]
    assert summary["hardware_profile"]["measurements"]
    measurement_units = {
        measurement["unit"]
        for measurement in summary["hardware_profile"]["measurements"]
    }
    assert {
        "host_nanoseconds_per_iteration",
        "device_nanoseconds_per_iteration",
        "millidegrees_celsius",
        "nanoseconds",
    } <= measurement_units
    assert all(workload["reliable"] for workload in summary["workloads"])
    assert any(
        workload["sustained"]["throughput_slope_ppm_per_window"] < 0
        for workload in summary["workloads"]
    )


def test_unreliable_measurements_cannot_become_a_calibrated_profile() -> None:
    profile = hardware_profile()
    plan = build_calibration_plan(
        profile,
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    run = completed_run(plan)
    workload = run["workloads"][0]
    workload["samples"] = [
        sample for sample in workload["samples"] if sample["phase"] != "steady"
    ]
    for index, sample in enumerate(workload["samples"]):
        sample["sample_index"] = index

    with pytest.raises(CalibrationContractError, match="no valid samples"):
        summarize_calibration_run(profile, plan, run)


def test_requested_confidence_level_changes_the_statistical_interval() -> None:
    profile = hardware_profile()
    intervals: list[int] = []
    for confidence in (900_000, 990_000):
        policy = CalibrationPolicy(
            warmup_iterations=1,
            steady_iterations=5,
            minimum_sample_duration_ns=1,
            sustained_window_duration_ms=1,
            sustained_window_count=2,
            confidence_level_ppm=confidence,
            maximum_relative_ci_width_ppm=900_000,
        )
        plan = build_calibration_plan(
            profile,
            implementation_fingerprint=FINGERPRINT,
            policy=policy,
        )
        summary = summarize_calibration_run(profile, plan, completed_run(plan))
        distribution = summary["workloads"][0]["steady"]
        intervals.append(
            distribution["confidence_interval_high_ns"]
            - distribution["confidence_interval_low_ns"]
        )
    assert intervals[1] > intervals[0]


def test_publication_is_atomic_and_detects_corruption(tmp_path: Path) -> None:
    profile = hardware_profile()
    plan = build_calibration_plan(
        profile,
        implementation_fingerprint=FINGERPRINT,
        policy=calibration_policy(),
    )
    run = completed_run(plan)
    summary = summarize_calibration_run(profile, plan, run)
    destination = tmp_path / "calibration"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for workload in run["workloads"]:
        for artifact in workload["artifacts"]:
            (artifacts / artifact["relative_path"]).write_bytes(b"x")

    publish_calibration(
        destination,
        plan=plan,
        run=run,
        summary=summary,
        artifact_directory=artifacts,
    )
    manifest = validate_published_calibration(destination)
    assert manifest["hardware_profile_id"] == summary["hardware_profile"]["profile_id"]

    (destination / "run.json").write_text("{}\n")
    with pytest.raises(ValueError, match="length mismatch"):
        validate_published_calibration(destination)


def test_orchestrator_runs_profiles_sequentially_and_publishes_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = hardware_profile()
    target = CompilerTarget(
        devices=(),
        hardware_profiles=(ContractDocument.from_json(profile),),
    )
    active = 0
    maximum_active = 0

    def fake_run(
        command: list[str],
        _cancelled: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal active, maximum_active
        if command[-1] == "--fingerprint":
            return subprocess.CompletedProcess(command, 0, FINGERPRINT + "\n", "")
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            plan_path = Path(command[command.index("--plan") + 1])
            run_path = Path(command[command.index("--output") + 1])
            artifact_directory = Path(command[command.index("--artifacts") + 1])
            plan = json.loads(plan_path.read_bytes())
            run = completed_run(plan)
            artifact_directory.mkdir(parents=True)
            for workload in run["workloads"]:
                for artifact in workload["artifacts"]:
                    (artifact_directory / artifact["relative_path"]).write_bytes(b"x")
            run_path.write_text(json.dumps(run))
            return subprocess.CompletedProcess(command, 0, "ok\n", "")
        finally:
            active -= 1

    monkeypatch.setattr(orchestrator, "_calibrator_command", lambda _path: ["fake"])
    monkeypatch.setattr(orchestrator, "_run_cancellable", fake_run)
    destination = tmp_path / "collection"
    report = orchestrator.calibrate_hardware(
        destination,
        target=target,
        policy=calibration_policy(),
    )

    assert report.profile_count == 1
    assert maximum_active == 1
    collection = orchestrator.validate_calibration_collection(destination)
    assert len(collection["profiles"]) == 1

    profile_manifest = destination / collection["profiles"][0]["relative_path"]
    profile_manifest.write_text("{}\n")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        orchestrator.validate_calibration_collection(destination)


def test_cancellable_subprocess_drains_large_output_without_pipe_deadlock() -> None:
    completed = orchestrator._run_cancellable(
        [
            sys.executable,
            "-c",
            (
                "import sys;"
                "sys.stdout.write('o' * 2000000);"
                "sys.stdout.flush();"
                "sys.stderr.write('e' * 2000000);"
                "sys.stderr.flush()"
            ),
        ],
        lambda: False,
    )

    assert completed.returncode == 0
    assert completed.stdout.startswith("[earlier subprocess output truncated]")
    assert completed.stderr.startswith("[earlier subprocess output truncated]")
    assert completed.stdout.endswith("o" * 1024)
    assert completed.stderr.endswith("e" * 1024)
    assert len(completed.stdout) < 70_000
    assert len(completed.stderr) < 70_000


def test_orchestrator_cancellation_leaves_no_partial_collection(
    tmp_path: Path,
) -> None:
    profile = hardware_profile()
    target = CompilerTarget(
        devices=(),
        hardware_profiles=(ContractDocument.from_json(profile),),
    )
    destination = tmp_path / "cancelled"
    with pytest.raises(ModelCompileCancelled, match="cancelled"):
        orchestrator.calibrate_hardware(
            destination,
            target=target,
            cancel_requested=lambda: True,
        )
    assert not destination.exists()


def test_orchestrator_preserves_raw_failed_run_for_reliability_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = hardware_profile()
    target = CompilerTarget(
        devices=(),
        hardware_profiles=(ContractDocument.from_json(profile),),
    )

    def fake_run(
        command: list[str],
        _cancelled: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--fingerprint":
            return subprocess.CompletedProcess(command, 0, FINGERPRINT + "\n", "")
        plan_path = Path(command[command.index("--plan") + 1])
        run_path = Path(command[command.index("--output") + 1])
        artifact_directory = Path(command[command.index("--artifacts") + 1])
        plan = json.loads(plan_path.read_bytes())
        run = completed_run(plan)
        steady = [
            sample
            for sample in run["workloads"][0]["samples"]
            if sample["phase"] == "steady"
        ]
        for index, sample in enumerate(steady):
            sample["duration_ns"] = 100 if index % 2 == 0 else 10_000_000
            sample["device_duration_ns"] = sample["duration_ns"]
        artifact_directory.mkdir(parents=True)
        for workload in run["workloads"]:
            for artifact in workload["artifacts"]:
                (artifact_directory / artifact["relative_path"]).write_bytes(b"x")
        run_path.write_text(json.dumps(run))
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(orchestrator, "_calibrator_command", lambda _path: ["fake"])
    monkeypatch.setattr(orchestrator, "_run_cancellable", fake_run)
    destination = tmp_path / "calibration"

    with pytest.raises(ModelCompileError, match="raw failed calibration preserved"):
        orchestrator.calibrate_hardware(
            destination,
            target=target,
            policy=calibration_policy(),
        )

    failure = tmp_path / ".calibration.failed"
    assert not destination.exists()
    assert (failure / "failure.json").is_file()
    assert list((failure / ".working").rglob("plan.json"))
    assert list((failure / ".working").rglob("run.json"))
