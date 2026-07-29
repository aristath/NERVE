from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled
from nerve.compiler_target import CompilerTarget, discover_compiler_target
from nerve.representation_optimizer.automation.device_state import (
    LinuxAmdDeviceStateProbe,
    declared_idle_state_digest,
)
from nerve.representation_optimizer.automation.residency_planner import (
    RuntimeResidencyPlanningCase,
    plan_runtime_residency_cases,
)
from nerve.representation_optimizer.automation.target import (
    OptimizationTarget,
    VerifiedDeviceLeaseManager,
)
from nerve.representation_optimizer.benchmarking.executor_adapter import (
    ResidentComponentExecutionAdapter,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.providers.builtin import (
    BuiltinCandidateToolchainResolver,
)
from nerve.representation_optimizer.providers.codebook import (
    ExactCodebookProofVerifier,
    ExactEmbeddedParameterProgramProofVerifier,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.validation.executor_adapter import (
    ResidentBehavioralValidationAdapter,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)

RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA = (
    "nerve.runtime_implementation_sha256.v1"
)


@dataclass(frozen=True)
class RuntimeOptimizationPolicy:
    model_residency_fraction_ppm: int = 850_000
    component_quantum_wait_ns: int = 1_000_000_000
    context_lifecycle_required_observations: int = 2
    maximum_context_lifecycle_attempts: int = 5

    def __post_init__(self) -> None:
        if not 0 < self.model_residency_fraction_ppm <= 1_000_000:
            raise ModelCompileError(
                "model residency fraction must be in (0, 1000000] ppm"
            )
        if self.component_quantum_wait_ns <= 0:
            raise ModelCompileError("component execution quantum wait must be positive")
        if self.context_lifecycle_required_observations < 2:
            raise ModelCompileError(
                "context lifecycle requires at least two matching endpoints"
            )
        if (
            self.maximum_context_lifecycle_attempts
            < self.context_lifecycle_required_observations
        ):
            raise ModelCompileError(
                "context lifecycle attempt bound must cover required endpoints"
            )


@dataclass(frozen=True)
class PreparedOptimizationTargets:
    targets: tuple[OptimizationTarget, ...]
    source_artifacts: PackageSourceArtifactResolver
    selected_devices: tuple[Json, ...]
    excluded_devices: tuple[Json, ...]
    parameter_bytes: int
    residency_plans: tuple[Json, ...]
    vulkan_driver_files: tuple[Path, ...]

    def to_json(self) -> Json:
        return {
            "target_ids": [target.target_id for target in self.targets],
            "selected_devices": [dict(item) for item in self.selected_devices],
            "excluded_devices": [dict(item) for item in self.excluded_devices],
            "parameter_bytes": self.parameter_bytes,
            "residency_plans": [dict(item) for item in self.residency_plans],
            "vulkan_driver_files": [str(path) for path in self.vulkan_driver_files],
        }


@dataclass(frozen=True)
class _SelectedCapabilityGroup:
    profiles: tuple[Json, ...]
    placement: dict[str, str]
    residency_plan: Json
    safe_device_capacity_bytes: dict[str, int]


def prepare_runtime_optimization_targets(
    *,
    package_manifest: Path,
    run_root: Path,
    runtime_bin: Path | None = None,
    component_executor_bin: Path | None = None,
    validation_executor_bin: Path | None = None,
    residency_planner_bin: Path | None = None,
    selected_device_ids: Iterable[str] = (),
    vulkan_driver_files: Iterable[Path] = (),
    idle_probe: LinuxAmdDeviceStateProbe | None = None,
    live_target: CompilerTarget | None = None,
    policy: RuntimeOptimizationPolicy = RuntimeOptimizationPolicy(),
    lease_root: Path | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> PreparedOptimizationTargets:
    check_compile_cancelled(cancel_requested)
    package_manifest = package_manifest.resolve()
    source_artifacts = PackageSourceArtifactResolver(
        package_manifest.parent
    )
    manifest = _read_json(package_manifest, "compiled package manifest")
    component_command = runtime_executor_command(
        "nerve-optimizer-executor",
        explicit=component_executor_bin,
        features=("vulkan",),
    )
    validation_command = runtime_executor_command(
        "nerve-validation-executor",
        explicit=validation_executor_bin,
        features=("vulkan", "tokenizers"),
    )
    residency_planner_command = runtime_executor_command(
        "nerve-residency-planner",
        explicit=residency_planner_bin,
        features=("vulkan",),
    )
    runtime_command = (
        runtime_executor_command(
            "nerve-runtime",
            explicit=runtime_bin,
            features=("vulkan", "tokenizers"),
        )
        if live_target is None
        else None
    )
    executor_commands = tuple(
        (
            label,
            command,
        )
        for label, command in (
            ("runtime", runtime_command),
            ("component executor", component_command),
            ("validation executor", validation_command),
            ("residency planner", residency_planner_command),
        )
        if command is not None
    )
    runtime_fingerprint = (
        _require_current_runtime_implementation_fingerprints(
            commands=executor_commands,
            runtime_root=Path(__file__).resolve().parents[3]
            / "runtime-rs",
            cancel_requested=cancel_requested,
        )
    )
    package_target = CompilerTarget.from_json(
        _required_object(manifest, "compiler_target")
    )
    package_profiles = tuple(
        profile.to_json()
        for profile in package_target.hardware_profiles
        if _is_amd_vulkan_gpu(profile.to_json())
    )
    if not package_profiles:
        raise ModelCompileError(
            "compiled package declares no AMD Vulkan optimization target"
        )
    requested = tuple(sorted(set(selected_device_ids)))
    if any(not value.startswith("vulkan-uuid:") for value in requested):
        raise ModelCompileError(
            "optimizer device selection requires stable Vulkan identities"
        )
    by_id = {
        str(profile["hardware_identity"]["stable_device_id"]): profile
        for profile in package_profiles
    }
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ModelCompileError(
            f"optimizer devices are absent from the compiled package: {missing}"
        )

    probe = idle_probe or LinuxAmdDeviceStateProbe()
    eligible = (
        tuple(by_id[device_id] for device_id in requested)
        if requested
        else (package_profiles)
    )
    idle_profiles: list[Json] = []
    selected_records: list[Json] = []
    excluded_records: list[Json] = []
    for profile in eligible:
        check_compile_cancelled(cancel_requested)
        device_id = str(profile["hardware_identity"]["stable_device_id"])
        try:
            observation = probe.require_idle((profile,))[0]
        except ModelCompileError as error:
            if requested:
                raise
            excluded_records.append(
                {
                    "device_id": device_id,
                    "reason": str(error),
                }
            )
            continue
        idle_profiles.append(profile)
        selected_records.append(observation)
    if not idle_profiles:
        raise ModelCompileError("no package-compatible AMD GPU is verified idle")

    parameter_bytes = _package_parameter_bytes(
        package_manifest.parent,
        manifest,
    )
    check_compile_cancelled(cancel_requested)
    selected_groups = _select_capability_groups(
        tuple(idle_profiles),
        package_manifest=package_manifest,
        manifest=manifest,
        residency_planner_command=residency_planner_command,
        explicit_selection=bool(requested),
        policy=policy,
        cancel_requested=cancel_requested,
    )
    selected_ids = tuple(
        sorted(
            str(profile["hardware_identity"]["stable_device_id"])
            for group in selected_groups
            for profile in group.profiles
        )
    )
    selected_records = [
        record for record in selected_records if record["device_id"] in selected_ids
    ]
    for profile in idle_profiles:
        device_id = str(profile["hardware_identity"]["stable_device_id"])
        if device_id not in selected_ids:
            excluded_records.append(
                {
                    "device_id": device_id,
                    "reason": (
                        "verified idle but not required by minimum safe "
                        "model-residency placement"
                    ),
                }
            )

    drivers = discover_amd_vulkan_driver_files(vulkan_driver_files)
    check_compile_cancelled(cancel_requested)
    if live_target is None:
        environment = amd_vulkan_environment(drivers)
        live_target, selected_records = _discover_stable_context_lifecycle(
            runtime_bin=(
                Path(runtime_command[0])
                if runtime_command is not None and len(runtime_command) == 1
                else runtime_bin
            ),
            selected_ids=selected_ids,
            package_profiles=by_id,
            idle_probe=probe,
            environment=environment,
            policy=policy,
            cancel_requested=cancel_requested,
        )
    else:
        live_profiles_for_baseline = {
            str(profile.to_json()["hardware_identity"]["stable_device_id"]): (
                profile.to_json()
            )
            for profile in live_target.hardware_profiles
            if _is_amd_vulkan_gpu(profile.to_json())
        }
        selected_records = list(
            probe.capture_stable_idle_baseline(
                tuple(
                    live_profiles_for_baseline[device_id] for device_id in selected_ids
                )
            )
        )
    check_compile_cancelled(cancel_requested)
    live_profiles = {
        str(profile.to_json()["hardware_identity"]["stable_device_id"]): (
            profile.to_json()
        )
        for profile in live_target.hardware_profiles
        if _is_amd_vulkan_gpu(profile.to_json())
    }
    if set(live_profiles) != set(selected_ids):
        raise ModelCompileError(
            "live AMD discovery did not return exactly the verified idle devices"
        )
    _require_live_identity_match(by_id, live_profiles, selected_ids)
    live_groups = tuple(
        (
            group,
            tuple(
                live_profiles[_device_id(profile)]
                for profile in group.profiles
            ),
        )
        for group in selected_groups
    )

    targets = tuple(
        _build_target(
            package_manifest=package_manifest,
            run_root=run_root,
            profiles=profiles,
            placement=group.placement,
            residency_plan=group.residency_plan,
            safe_device_capacity_bytes=group.safe_device_capacity_bytes,
            idle_probe=probe,
            selected_observations=tuple(
                record
                for record in selected_records
                if record["device_id"] in {_device_id(profile) for profile in profiles}
            ),
            driver_files=drivers,
            component_command=component_command,
            validation_command=validation_command,
            runtime_implementation_fingerprint=runtime_fingerprint,
            policy=policy,
            lease_root=lease_root or default_device_lease_root(),
            source_artifacts=source_artifacts,
        )
        for group, profiles in live_groups
    )
    check_compile_cancelled(cancel_requested)
    return PreparedOptimizationTargets(
        targets=targets,
        source_artifacts=source_artifacts,
        selected_devices=tuple(
            sorted(selected_records, key=lambda item: item["device_id"])
        ),
        excluded_devices=tuple(
            sorted(excluded_records, key=lambda item: item["device_id"])
        ),
        parameter_bytes=parameter_bytes,
        residency_plans=tuple(
            group.residency_plan for group in selected_groups
        ),
        vulkan_driver_files=drivers,
    )


def discover_amd_vulkan_driver_files(
    configured: Iterable[Path] = (),
) -> tuple[Path, ...]:
    paths = tuple(Path(path).expanduser().resolve() for path in configured)
    if not paths:
        raw = os.environ.get("NERVE_AMD_VULKAN_DRIVER_FILES")
        if raw:
            paths = tuple(
                Path(value).expanduser().resolve()
                for value in raw.split(os.pathsep)
                if value
            )
    if not paths:
        architecture = platform.machine().lower()
        suffix = {
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
        }.get(architecture, architecture)
        preferred = Path(f"/usr/share/vulkan/icd.d/radeon_icd.{suffix}.json")
        if preferred.is_file():
            paths = (preferred.resolve(),)
    if not paths:
        raise ModelCompileError(
            "AMD Vulkan driver manifest is unavailable; set "
            "NERVE_AMD_VULKAN_DRIVER_FILES"
        )
    for path in paths:
        document = _read_json(path, "Vulkan ICD manifest")
        library = document.get("ICD", {}).get("library_path")
        if (
            not path.is_file()
            or not isinstance(library, str)
            or "radeon" not in library.lower()
        ):
            raise ModelCompileError(
                f"Vulkan ICD manifest is not an AMD Radeon driver: {path}"
            )
    return tuple(sorted(set(paths)))


def amd_vulkan_environment(driver_files: tuple[Path, ...]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["VK_DRIVER_FILES"] = os.pathsep.join(str(path) for path in driver_files)
    environment.pop("VK_ICD_FILENAMES", None)
    return environment


def runtime_implementation_fingerprint(runtime_root: Path) -> str:
    runtime_root = runtime_root.resolve()
    inputs = [
        *(
            (relative, runtime_root / relative)
            for relative in ("Cargo.lock", "Cargo.toml", "build.rs")
        ),
        *(
            (
                path.relative_to(runtime_root).as_posix(),
                path,
            )
            for path in (runtime_root / "src").rglob("*.rs")
            if path.is_file()
        ),
    ]
    missing = [
        relative
        for relative, path in inputs
        if not path.is_file()
    ]
    if missing:
        raise ModelCompileError(
            "runtime implementation fingerprint inputs are missing: "
            f"{sorted(missing)}"
        )
    digest = sha256()
    for relative, path in sorted(inputs):
        relative_bytes = relative.encode("utf-8")
        source = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
    return (
        f"{RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA}:"
        f"{digest.hexdigest()}"
    )


def _require_current_runtime_implementation_fingerprints(
    *,
    commands: tuple[tuple[str, tuple[str, ...]], ...],
    runtime_root: Path,
    cancel_requested: Callable[[], bool] | None,
) -> str:
    expected = runtime_implementation_fingerprint(runtime_root)
    mismatches = []
    for label, command in commands:
        check_compile_cancelled(cancel_requested)
        observed = _query_runtime_implementation_fingerprint(
            command,
            cancel_requested=cancel_requested,
        )
        if observed != expected:
            mismatches.append(f"{label} reports {observed}")
    if mismatches:
        raise ModelCompileError(
            "optimizer executables are stale relative to runtime source "
            f"{expected}: "
            + "; ".join(mismatches)
            + "; rebuild every executable before starting optimization"
        )
    return expected


def _query_runtime_implementation_fingerprint(
    command: tuple[str, ...],
    *,
    cancel_requested: Callable[[], bool] | None,
) -> str:
    invocation = [
        *command,
        "--runtime-implementation-fingerprint",
    ]
    try:
        process = subprocess.Popen(
            invocation,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise ModelCompileError(
            f"could not start optimizer executable {command[0]!r}: {error}"
        ) from error
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            try:
                check_compile_cancelled(cancel_requested)
            except BaseException:
                process.kill()
                process.communicate()
                raise
    check_compile_cancelled(cancel_requested)
    if process.returncode != 0:
        diagnostic = stderr.strip() or stdout.strip()
        raise ModelCompileError(
            f"optimizer executable {command[0]!r} could not report its "
            "runtime implementation fingerprint"
            + (f": {diagnostic}" if diagnostic else "")
        )
    fingerprint = stdout.strip()
    if (
        not fingerprint.startswith(
            f"{RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA}:"
        )
        or "\n" in fingerprint
    ):
        raise ModelCompileError(
            f"optimizer executable {command[0]!r} returned an invalid "
            f"runtime implementation fingerprint {fingerprint!r}"
        )
    return fingerprint


def _discover_stable_context_lifecycle(
    *,
    runtime_bin: Path | None,
    selected_ids: tuple[str, ...],
    package_profiles: dict[str, Json],
    idle_probe: LinuxAmdDeviceStateProbe,
    environment: dict[str, str],
    policy: RuntimeOptimizationPolicy,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[CompilerTarget, list[Json]]:
    previous_endpoint: tuple[tuple[str, int], ...] | None = None
    consecutive_endpoints = 0
    observed_endpoints: list[tuple[tuple[str, int], ...]] = []
    last_target: CompilerTarget | None = None
    last_records: list[Json] = []
    for _attempt in range(policy.maximum_context_lifecycle_attempts):
        check_compile_cancelled(cancel_requested)
        target = discover_compiler_target(
            runtime_bin=runtime_bin,
            allowed_physical_device_ids=selected_ids,
            environment=environment,
            initialize_device_contexts=True,
            cancel_requested=cancel_requested,
        )
        profiles = {
            str(profile.to_json()["hardware_identity"]["stable_device_id"]): (
                profile.to_json()
            )
            for profile in target.hardware_profiles
            if _is_amd_vulkan_gpu(profile.to_json())
        }
        if set(profiles) != set(selected_ids):
            raise ModelCompileError(
                "live AMD discovery did not return exactly the verified idle devices"
            )
        _require_live_identity_match(
            package_profiles,
            profiles,
            selected_ids,
        )
        records = list(
            idle_probe.capture_stable_idle_baseline(
                tuple(profiles[device_id] for device_id in selected_ids)
            )
        )
        endpoint = tuple(
            (str(record["device_id"]), int(record["vram_used_bytes"]))
            for record in records
        )
        observed_endpoints.append(endpoint)
        if endpoint == previous_endpoint:
            consecutive_endpoints += 1
        else:
            previous_endpoint = endpoint
            consecutive_endpoints = 1
        last_target = target
        last_records = records
        if consecutive_endpoints >= policy.context_lifecycle_required_observations:
            return last_target, last_records
    raise ModelCompileError(
        "independent Vulkan context lifecycles did not converge on a "
        "repeatable idle residency endpoint within "
        f"{policy.maximum_context_lifecycle_attempts} attempts: "
        f"{observed_endpoints}"
    )


def runtime_executor_command(
    binary_name: str,
    *,
    explicit: Path | None,
    features: tuple[str, ...],
    repo_root: Path | None = None,
) -> tuple[str, ...]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ModelCompileError(f"{binary_name} executable is unavailable: {path}")
        return (str(path),)
    environment_name = binary_name.upper().replace("-", "_") + "_BIN"
    configured = os.environ.get(environment_name)
    if configured:
        return runtime_executor_command(
            binary_name,
            explicit=Path(configured),
            features=features,
            repo_root=repo_root,
        )
    repo_root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else repo_root.resolve()
    )
    manifest = repo_root / "runtime-rs" / "Cargo.toml"
    if manifest.is_file():
        return _build_current_repo_executor(
            binary_name=binary_name,
            manifest=manifest,
            features=features,
        )
    installed = shutil.which(binary_name)
    if installed:
        return (installed,)
    raise ModelCompileError(
        f"could not find or build required executor {binary_name!r}"
    )


def _build_current_repo_executor(
    *,
    binary_name: str,
    manifest: Path,
    features: tuple[str, ...],
) -> tuple[str, ...]:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise ModelCompileError(
            f"cannot build current {binary_name!r}: cargo is unavailable"
        )
    command = [
        cargo,
        "build",
        "--release",
        "--manifest-path",
        str(manifest),
        "--features",
        ",".join(features),
        "--bin",
        binary_name,
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise ModelCompileError(
            f"could not build current optimizer executable {binary_name!r}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    binary = (
        manifest.parent
        / "target"
        / "release"
        / binary_name
    ).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ModelCompileError(
            f"cargo did not produce optimizer executable {binary}"
        )
    return (str(binary),)


def balanced_component_placement(
    package_root: Path,
    manifest: Json,
    device_ids: tuple[str, ...],
) -> dict[str, str]:
    if not device_ids:
        raise ModelCompileError("component placement requires devices")
    components = manifest.get("circuit_graph", {}).get("components")
    if not isinstance(components, list) or not components:
        raise ModelCompileError("compiled package has no component graph")
    tensor_index = _tensor_index(package_root, manifest)
    tensor_sizes = {
        str(name): int(metadata["byte_count"])
        for name, metadata in tensor_index["tensors"].items()
    }
    weighted: list[tuple[str, int]] = []
    for component in components:
        if not isinstance(component, dict):
            raise ModelCompileError("compiled component graph is malformed")
        component_id = str(component.get("component_id", ""))
        runtime_role = component.get("runtime_role")
        refs = component.get("params", {}).get("refs", {})
        if (
            not component_id
            or not isinstance(runtime_role, str)
            or not runtime_role
            or not isinstance(refs, dict)
        ):
            raise ModelCompileError(
                "compiled component placement contract is malformed"
            )
        if runtime_role != "signal_processor":
            continue
        names = {
            value.get("tensor")
            for value in refs.values()
            if isinstance(value, dict) and isinstance(value.get("tensor"), str)
        }
        missing = sorted(name for name in names if name not in tensor_sizes)
        if missing:
            raise ModelCompileError(
                f"component {component_id!r} references unknown tensors: {missing}"
            )
        weighted.append((component_id, sum(tensor_sizes[name] for name in names)))
    if not weighted:
        raise ModelCompileError(
            "compiled package has no independently placeable signal processors"
        )
    if len(device_ids) == 1:
        return {component_id: device_ids[0] for component_id, _ in weighted}
    return _contiguous_weighted_partition(weighted, device_ids)


def _build_target(
    *,
    package_manifest: Path,
    run_root: Path,
    profiles: tuple[Json, ...],
    placement: dict[str, str],
    residency_plan: Json,
    safe_device_capacity_bytes: dict[str, int],
    idle_probe: LinuxAmdDeviceStateProbe,
    selected_observations: tuple[Json, ...],
    driver_files: tuple[Path, ...],
    component_command: tuple[str, ...],
    validation_command: tuple[str, ...],
    runtime_implementation_fingerprint: str,
    policy: RuntimeOptimizationPolicy,
    lease_root: Path,
    source_artifacts: PackageSourceArtifactResolver,
) -> OptimizationTarget:
    profiles = tuple(sorted(profiles, key=lambda item: _device_id(item)))
    capability_classes = {str(profile["capability_class"]) for profile in profiles}
    if len(capability_classes) != 1:
        raise ModelCompileError(
            "one optimization execution target must have one capability class"
        )
    device_ids = tuple(_device_id(profile) for profile in profiles)
    if set(device_ids) != set(safe_device_capacity_bytes):
        raise ModelCompileError(
            "residency admission capacities do not match live target devices"
        )
    manifest = _read_json(package_manifest, "compiled package manifest")
    idle_digest = declared_idle_state_digest(
        profiles,
        idle_probe.policy,
    )
    matched_conditions = {
        "devices": sorted(
            (
                {
                    "device_id": _device_id(profile),
                    "hardware_profile_digest": contract_digest(profile),
                    "capability_class": profile["capability_class"],
                    "api": profile["provenance"]["api"],
                }
                for profile in profiles
            ),
            key=lambda item: item["device_id"],
        ),
        "placement": placement,
        "controls": {
            "scheduler": "normal",
            "maximum_quantum_wait_ns": policy.component_quantum_wait_ns,
        },
        "environment": {
            "runtime_implementation_fingerprint": (
                runtime_implementation_fingerprint
            ),
            "vulkan_driver_manifests": [str(path) for path in driver_files],
            "device_idle_policy": idle_probe.policy.to_json(),
            "context_prepared_idle_observations": [
                dict(item)
                for item in sorted(
                    selected_observations,
                    key=lambda item: item["device_id"],
                )
            ],
            "residency_admission": {
                "plan": residency_plan,
                "safe_device_capacity_bytes": dict(
                    sorted(safe_device_capacity_bytes.items())
                ),
                "residency_fraction_ppm": (
                    policy.model_residency_fraction_ppm
                ),
            },
        },
        "idle_device_state_digest": idle_digest,
        "exclusive_residency": True,
    }
    target_id = stable_contract_id(
        "optimization_target",
        manifest["package_id"],
        sorted(capability_classes),
        list(device_ids),
        placement,
    )
    candidate_workspace = run_root / "workspaces" / "candidates"
    benchmark_adapter = ResidentComponentExecutionAdapter(
        package_manifest=package_manifest,
        candidate_workspace=candidate_workspace,
        trace_root=run_root / "workspaces" / "benchmark" / "adapter-traces",
        executor_command=component_command,
        vulkan_driver_files=driver_files,
    )
    validation_adapter = ResidentBehavioralValidationAdapter(
        package_manifest=package_manifest,
        candidate_workspace=candidate_workspace,
        trace_root=run_root / "workspaces" / "validation" / "adapter-traces",
        component_executor_command=component_command,
        whole_model_executor_command=validation_command,
        vulkan_driver_files=driver_files,
    )
    return OptimizationTarget(
        target_id=target_id,
        synthesis_profile=profiles[0],
        hardware_profiles=profiles,
        matched_conditions=matched_conditions,
        requires_device_lease=True,
        toolchains=BuiltinCandidateToolchainResolver(),
        benchmark_adapter=benchmark_adapter,
        validation_adapter=validation_adapter,
        proof_verifiers=ProofVerifierRegistry.from_verifiers(
            (
                ExactCodebookProofVerifier(
                    source_artifacts=source_artifacts,
                    candidate_workspace_root=candidate_workspace,
                ),
                ExactEmbeddedParameterProgramProofVerifier(
                    source_artifacts=source_artifacts,
                    candidate_workspace_root=candidate_workspace,
                ),
            )
        ),
        lease_manager=VerifiedDeviceLeaseManager(
            lock_root=lease_root,
            probe_idle_state_digest=idle_probe.target_idle_state_digest,
        ),
        estimate_execution_nanoseconds=lambda _plan, _policy: None,
    )


def _select_capability_groups(
    profiles: tuple[Json, ...],
    *,
    package_manifest: Path,
    manifest: Json,
    residency_planner_command: tuple[str, ...],
    explicit_selection: bool,
    policy: RuntimeOptimizationPolicy,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[_SelectedCapabilityGroup, ...]:
    by_capability: dict[str, list[Json]] = defaultdict(list)
    for profile in profiles:
        by_capability[str(profile["capability_class"])].append(profile)
    groups: list[_SelectedCapabilityGroup] = []
    failures: list[str] = []
    max_context = manifest.get("max_context_activations")
    if (
        isinstance(max_context, bool)
        or not isinstance(max_context, int)
        or max_context <= 0
    ):
        raise ModelCompileError(
            "compiled package max_context_activations is invalid"
        )
    mount_speculative_decoders = bool(manifest.get("speculative_decoders"))
    for capability, raw_group in sorted(by_capability.items()):
        group = sorted(raw_group, key=_device_id)
        candidate_groups = (
            (tuple(group),)
            if explicit_selection
            else tuple(
                tuple(group[:device_count])
                for device_count in range(1, len(group) + 1)
            )
        )
        cases = []
        placements: dict[str, dict[str, str]] = {}
        profiles_by_case: dict[str, tuple[Json, ...]] = {}
        for index, candidate_profiles in enumerate(candidate_groups, start=1):
            device_ids = tuple(
                _device_id(profile) for profile in candidate_profiles
            )
            placement = balanced_component_placement(
                package_manifest.parent,
                manifest,
                device_ids,
            )
            case_id = f"{capability}:{index}"
            placements[case_id] = placement
            profiles_by_case[case_id] = candidate_profiles
            cases.append(
                RuntimeResidencyPlanningCase(
                    case_id=case_id,
                    default_device_id=device_ids[0],
                    component_placement=placement,
                    context_capacity_activations=max_context,
                    mount_speculative_decoders=(
                        mount_speculative_decoders
                    ),
                )
            )
        plans = plan_runtime_residency_cases(
            command=residency_planner_command,
            package_manifest=package_manifest,
            cases=cases,
            cancel_requested=cancel_requested,
        )
        selected_group = None
        case_failures = []
        for case in cases:
            candidate_profiles = profiles_by_case[case.case_id]
            capacities = {
                _device_id(profile): _safe_profile_capacity(
                    profile,
                    policy=policy,
                )
                for profile in candidate_profiles
            }
            plan = plans[case.case_id]
            planned_devices = {
                str(device["device_id"]): int(
                    device["total_device_resident_bytes"]
                )
                for device in plan["device_plans"]
            }
            if set(planned_devices) != set(capacities):
                raise ModelCompileError(
                    "runtime residency plan devices do not match the "
                    "requested placement topology"
                )
            oversized = {
                device_id: {
                    "planned": planned_devices[device_id],
                    "safe_capacity": capacities[device_id],
                }
                for device_id in sorted(capacities)
                if planned_devices[device_id] > capacities[device_id]
            }
            if oversized:
                case_failures.append(
                    f"{len(candidate_profiles)} device(s): {oversized}"
                )
                continue
            selected_group = _SelectedCapabilityGroup(
                profiles=candidate_profiles,
                placement=placements[case.case_id],
                residency_plan=plan,
                safe_device_capacity_bytes=capacities,
            )
            break
        if selected_group is None:
            failures.append(
                f"{capability} cannot safely host the planned runtime "
                f"working set ({'; '.join(case_failures)})"
            )
            continue
        groups.append(selected_group)
    if not groups:
        raise ModelCompileError(
            "no AMD capability class can host the compiled package: "
            + "; ".join(failures)
        )
    return tuple(groups)


def _safe_profile_capacity(
    profile: Json,
    *,
    policy: RuntimeOptimizationPolicy,
) -> int:
    heaps: dict[tuple[str, str], int] = {}
    for domain in profile["memory_domains"]:
        if not domain.get("device_local"):
            continue
        properties = domain.get("properties", {})
        key = (
            str(properties.get("capacity_scope", domain["name"])),
            str(properties.get("heap_index", domain["name"])),
        )
        heaps[key] = max(heaps.get(key, 0), int(domain["capacity_bytes"]))
    if not heaps:
        raise ModelCompileError(
            f"AMD profile {_device_id(profile)!r} has no device-local memory heap"
        )
    return sum(heaps.values()) * policy.model_residency_fraction_ppm // 1_000_000


def _package_parameter_bytes(package_root: Path, manifest: Json) -> int:
    tensor_index = _tensor_index(package_root, manifest)
    totals = tensor_index.get("totals")
    if not isinstance(totals, dict):
        raise ModelCompileError("compiled tensor index has no totals")
    value = totals.get("byte_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelCompileError("compiled tensor byte count is invalid")
    return value


def _tensor_index(package_root: Path, manifest: Json) -> Json:
    raw = manifest.get("tensor_index_path")
    if not isinstance(raw, str) or not raw:
        raise ModelCompileError("compiled package has no tensor index")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelCompileError("compiled tensor index path is unsafe")
    document = _read_json(package_root / relative, "compiled tensor index")
    if document.get("schema") != "nerve.tensor_index.v1" or not isinstance(
        document.get("tensors"), dict
    ):
        raise ModelCompileError("compiled tensor index is malformed")
    return document


def _contiguous_weighted_partition(
    components: list[tuple[str, int]],
    device_ids: tuple[str, ...],
) -> dict[str, str]:
    placement: dict[str, str] = {}
    cursor = 0
    remaining_weight = sum(weight for _, weight in components)
    for device_index, device_id in enumerate(device_ids):
        remaining_devices = len(device_ids) - device_index
        if device_index == len(device_ids) - 1:
            for component_id, _ in components[cursor:]:
                placement[component_id] = device_id
            break
        current_weight = 0
        while cursor < len(components):
            components_after = len(components) - (cursor + 1)
            devices_after = remaining_devices - 1
            if components_after < devices_after:
                break
            component_id, weight = components[cursor]
            if current_weight > 0 and abs(
                remaining_weight - current_weight * remaining_devices
            ) <= abs(remaining_weight - (current_weight + weight) * remaining_devices):
                break
            placement[component_id] = device_id
            current_weight += weight
            cursor += 1
        if current_weight == 0:
            component_id, weight = components[cursor]
            placement[component_id] = device_id
            current_weight = weight
            cursor += 1
        remaining_weight -= current_weight
    if set(placement) != {component_id for component_id, _ in components}:
        raise ModelCompileError("balanced placement omitted compiled components")
    return placement


def _require_live_identity_match(
    package_profiles: dict[str, Json],
    live_profiles: dict[str, Json],
    selected_ids: tuple[str, ...],
) -> None:
    for device_id in selected_ids:
        packaged = package_profiles[device_id]
        live = live_profiles[device_id]
        for field in (
            "vendor_id",
            "device_id",
            "architecture",
            "physical_location",
        ):
            if packaged["hardware_identity"].get(field) != live[
                "hardware_identity"
            ].get(field):
                raise ModelCompileError(
                    f"live device {device_id!r} no longer matches package "
                    f"hardware identity field {field!r}"
                )


def _is_amd_vulkan_gpu(profile: Json) -> bool:
    identity = profile.get("hardware_identity", {})
    provenance = profile.get("provenance", {})
    return (
        identity.get("device_kind") == "gpu"
        and str(identity.get("vendor_id", "")).lower() == "0x1002"
        and provenance.get("api") == "vulkan"
    )


def _device_id(profile: Json) -> str:
    return str(profile["hardware_identity"]["stable_device_id"])


def _read_json(path: Path, label: str) -> Json:
    try:
        document = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCompileError(f"{label} is unavailable or invalid: {path}") from error
    if not isinstance(document, dict):
        raise ModelCompileError(f"{label} must contain a JSON object")
    return document


def _required_object(document: Json, field: str) -> Json:
    value = document.get(field)
    if not isinstance(value, dict):
        raise ModelCompileError(f"compiled package {field!r} is missing")
    return value


def default_device_lease_root() -> Path:
    runtime_root = os.environ.get("XDG_RUNTIME_DIR")
    base = (
        Path(runtime_root).expanduser()
        if runtime_root
        else Path("/tmp") / f"nerve-{os.getuid()}"
    )
    return base / "device-leases"
