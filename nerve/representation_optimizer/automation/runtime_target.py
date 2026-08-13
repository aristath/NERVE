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
    DeviceCapacityPolicy,
    LinuxDrmDeviceCapacityProbe,
    RuntimeVulkanDeviceCapacityProbe,
    declared_capacity_reservation_digest,
    normalize_pci_address,
)
from nerve.representation_optimizer.automation.residency_planner import (
    RuntimeResidencyPlanningCase,
    plan_runtime_residency_cases,
)
from nerve.representation_optimizer.automation.target import (
    OptimizationTarget,
    VerifiedCapacityLeaseManager,
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
from nerve.representation_optimizer.providers.resident_expansion import (
    ExactResidentExpansionProofVerifier,
)
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.validation.executor_adapter import (
    ResidentBehavioralValidationAdapter,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)

RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA = "nerve.runtime_implementation_sha256.v1"


@dataclass(frozen=True)
class RuntimeOptimizationPolicy:
    component_quantum_wait_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.component_quantum_wait_ns <= 0:
            raise ModelCompileError("component execution quantum wait must be positive")


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
            "qualification_regimes": {
                target.target_id: target.qualification_regime.to_json()
                for target in self.targets
            },
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
    available_device_capacity_bytes: dict[str, int]
    reserved_device_capacity_bytes: dict[str, int]


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
    speculative_draft_tokens: int = 0,
    residency_policy: str = "demand_retained",
    capacity_probe: LinuxDrmDeviceCapacityProbe | None = None,
    live_target: CompilerTarget | None = None,
    policy: RuntimeOptimizationPolicy = RuntimeOptimizationPolicy(),
    lease_root: Path | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> PreparedOptimizationTargets:
    check_compile_cancelled(cancel_requested)
    qualification_regime = QualificationRegime(
        speculative_draft_tokens=speculative_draft_tokens,
    )
    if residency_policy not in {
        "demand_paged",
        "demand_retained",
        "eager",
    }:
        raise ModelCompileError(
            f"unsupported optimizer residency policy {residency_policy!r}"
        )
    package_manifest = package_manifest.resolve()
    source_artifacts = PackageSourceArtifactResolver(package_manifest.parent)
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
    runtime_fingerprint = _require_current_runtime_implementation_fingerprints(
        commands=executor_commands,
        runtime_root=Path(__file__).resolve().parents[3] / "runtime-rs",
        cancel_requested=cancel_requested,
    )
    runtime_capacity_policy = _require_consistent_runtime_device_local_memory_policy(
        commands=executor_commands,
        cancel_requested=cancel_requested,
    )
    package_target = CompilerTarget.from_json(
        _required_object(manifest, "compiler_target")
    )
    package_profiles = tuple(
        profile.to_json()
        for profile in package_target.hardware_profiles
        if _is_vulkan_gpu(profile.to_json())
    )
    if not package_profiles:
        raise ModelCompileError(
            "compiled package declares no Vulkan GPU optimization target"
        )
    requested = tuple(sorted(set(selected_device_ids)))
    if any(not value.startswith("vulkan-uuid:") for value in requested):
        raise ModelCompileError(
            "optimizer device selection requires stable Vulkan identities"
        )
    parameter_bytes = _package_parameter_bytes(
        package_manifest.parent,
        manifest,
    )
    drivers = discover_vulkan_driver_files(vulkan_driver_files)
    check_compile_cancelled(cancel_requested)
    if live_target is None:
        environment = vulkan_environment(drivers)
        live_target = discover_compiler_target(
            runtime_bin=(
                Path(runtime_command[0])
                if runtime_command is not None and len(runtime_command) == 1
                else runtime_bin
            ),
            allowed_physical_device_ids=requested,
            environment=environment,
            initialize_device_contexts=True,
            cancel_requested=cancel_requested,
        )
    check_compile_cancelled(cancel_requested)
    all_live_gpu_profiles = {
        str(profile.to_json()["hardware_identity"]["stable_device_id"]): (
            profile.to_json()
        )
        for profile in live_target.hardware_profiles
        if _is_vulkan_gpu(profile.to_json())
    }
    package_capability_classes = {
        str(profile["capability_class"]) for profile in package_profiles
    }
    live_profiles = {
        device_id: profile
        for device_id, profile in all_live_gpu_profiles.items()
        if profile["capability_class"] in package_capability_classes
    }
    missing = sorted(set(requested) - set(live_profiles))
    if missing:
        raise ModelCompileError(
            "optimizer devices are not live Vulkan targets compatible with the "
            f"compiled execution capabilities: {missing}"
        )
    eligible = (
        tuple(live_profiles[device_id] for device_id in requested)
        if requested
        else tuple(live_profiles.values())
    )
    probe = capacity_probe or RuntimeVulkanDeviceCapacityProbe(
        policy=runtime_capacity_policy,
        executor_command=component_command,
        vulkan_driver_files=drivers,
        cancel_requested=cancel_requested,
    )
    capacity_profiles: list[Json] = []
    selected_records: list[Json] = []
    excluded_records: list[Json] = [
        {
            "device_id": device_id,
            "reason": ("live Vulkan device does not match a compiled capability class"),
        }
        for device_id in sorted(set(all_live_gpu_profiles) - set(live_profiles))
    ]
    for profile in eligible:
        check_compile_cancelled(cancel_requested)
        device_id = _device_id(profile)
        try:
            observation = probe.require_capacity((profile,))[0]
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
        capacity_profiles.append(profile)
        selected_records.append(observation)
    if not capacity_profiles:
        raise ModelCompileError(
            "no package-compatible Vulkan GPU has measurable reservable VRAM capacity"
        )
    live_eligible_profiles = tuple(sorted(capacity_profiles, key=_device_topology_key))
    available_capacity_by_device = {
        str(record["device_id"]): int(record["reservable_vram_bytes"])
        for record in selected_records
    }
    selected_groups = _select_capability_groups(
        live_eligible_profiles,
        available_capacity_by_device=available_capacity_by_device,
        package_manifest=package_manifest,
        manifest=manifest,
        residency_planner_command=residency_planner_command,
        speculative_draft_tokens=(qualification_regime.speculative_draft_tokens),
        residency_policy=residency_policy,
        explicit_selection=bool(requested),
        cancel_requested=cancel_requested,
    )
    selected_ids = tuple(
        sorted(
            _device_id(profile)
            for group in selected_groups
            for profile in group.profiles
        )
    )
    selected_records = [
        record for record in selected_records if record["device_id"] in selected_ids
    ]
    for profile in capacity_profiles:
        device_id = _device_id(profile)
        if device_id not in selected_ids:
            excluded_records.append(
                {
                    "device_id": device_id,
                    "reason": (
                        "compatible capacity not required by the smallest "
                        "admissible contiguous placement"
                    ),
                }
            )
    live_groups = tuple(
        (
            group,
            tuple(live_profiles[_device_id(profile)] for profile in group.profiles),
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
            available_device_capacity_bytes=group.available_device_capacity_bytes,
            reserved_device_capacity_bytes=group.reserved_device_capacity_bytes,
            capacity_probe=probe,
            selected_observations=tuple(
                record
                for record in selected_records
                if record["device_id"] in {_device_id(profile) for profile in profiles}
            ),
            driver_files=drivers,
            component_command=component_command,
            validation_command=validation_command,
            runtime_implementation_fingerprint=runtime_fingerprint,
            qualification_regime=qualification_regime,
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
        residency_plans=tuple(group.residency_plan for group in selected_groups),
        vulkan_driver_files=drivers,
    )


def discover_vulkan_driver_files(
    configured: Iterable[Path] = (),
) -> tuple[Path, ...]:
    paths = tuple(Path(path).expanduser().resolve() for path in configured)
    if not paths:
        raw = (
            os.environ.get("NERVE_VULKAN_DRIVER_FILES")
            or os.environ.get("VK_DRIVER_FILES")
            or os.environ.get("VK_ICD_FILENAMES")
        )
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
        manifest_root = Path("/usr/share/vulkan/icd.d")
        paths = tuple(
            path.resolve()
            for path in sorted(manifest_root.glob(f"*.{suffix}.json"))
            if path.is_file()
        )
    if not paths:
        raise ModelCompileError(
            "Vulkan driver manifests are unavailable; set "
            "NERVE_VULKAN_DRIVER_FILES"
        )
    for path in paths:
        if not path.is_file():
            raise ModelCompileError(
                f"Vulkan ICD manifest is unavailable: {path}"
            )
        document = _read_json(path, "Vulkan ICD manifest")
        icd = document.get("ICD")
        library = icd.get("library_path") if isinstance(icd, dict) else None
        if (
            not isinstance(document.get("file_format_version"), str)
            or not isinstance(library, str)
            or not library
        ):
            raise ModelCompileError(
                f"Vulkan ICD manifest is malformed: {path}"
            )
    return tuple(sorted(set(paths)))


def vulkan_environment(driver_files: tuple[Path, ...]) -> dict[str, str]:
    environment = dict(os.environ)
    environment["VK_DRIVER_FILES"] = os.pathsep.join(str(path) for path in driver_files)
    environment.pop("VK_ICD_FILENAMES", None)
    return environment


def runtime_implementation_fingerprint(runtime_root: Path) -> str:
    runtime_root = runtime_root.resolve()
    inputs = [
        *(
            (relative, runtime_root / relative)
            for relative in (
                "Cargo.lock",
                "Cargo.toml",
                "build.rs",
                "shaders/gpu_residency_gate.comp",
            )
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
    missing = [relative for relative, path in inputs if not path.is_file()]
    if missing:
        raise ModelCompileError(
            f"runtime implementation fingerprint inputs are missing: {sorted(missing)}"
        )
    digest = sha256()
    for relative, path in sorted(inputs):
        relative_bytes = relative.encode("utf-8")
        source = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "little"))
        digest.update(relative_bytes)
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
    return f"{RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA}:{digest.hexdigest()}"


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
    stdout = _query_runtime_introspection(
        command,
        argument="--runtime-implementation-fingerprint",
        subject="runtime implementation fingerprint",
        cancel_requested=cancel_requested,
    )
    fingerprint = stdout.strip()
    if (
        not fingerprint.startswith(f"{RUNTIME_IMPLEMENTATION_FINGERPRINT_SCHEMA}:")
        or "\n" in fingerprint
    ):
        raise ModelCompileError(
            f"optimizer executable {command[0]!r} returned an invalid "
            f"runtime implementation fingerprint {fingerprint!r}"
        )
    return fingerprint


def _require_consistent_runtime_device_local_memory_policy(
    *,
    commands: tuple[tuple[str, tuple[str, ...]], ...],
    cancel_requested: Callable[[], bool] | None,
) -> DeviceCapacityPolicy:
    observed: list[tuple[str, DeviceCapacityPolicy]] = []
    for label, command in commands:
        check_compile_cancelled(cancel_requested)
        stdout = _query_runtime_introspection(
            command,
            argument="--runtime-device-local-memory-policy",
            subject="runtime device-local memory policy",
            cancel_requested=cancel_requested,
        )
        try:
            document = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise ModelCompileError(
                f"optimizer executable {command[0]!r} returned invalid JSON "
                "for its runtime device-local memory policy"
            ) from error
        if not isinstance(document, dict):
            raise ModelCompileError(
                f"optimizer executable {command[0]!r} returned a non-object "
                "runtime device-local memory policy"
            )
        observed.append((label, DeviceCapacityPolicy.from_runtime_policy(document)))
    if not observed:
        raise ModelCompileError(
            "optimizer requires at least one runtime executable memory policy"
        )
    expected = observed[0][1]
    mismatches = [label for label, policy in observed[1:] if policy != expected]
    if mismatches:
        raise ModelCompileError(
            "optimizer executables report inconsistent runtime device-local "
            f"memory policies: {mismatches}"
        )
    return expected


def _query_runtime_introspection(
    command: tuple[str, ...],
    *,
    argument: str,
    subject: str,
    cancel_requested: Callable[[], bool] | None,
) -> str:
    invocation = [*command, argument]
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
            f"optimizer executable {command[0]!r} could not report its {subject}"
            + (f": {diagnostic}" if diagnostic else "")
        )
    return stdout


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
    binary = (manifest.parent / "target" / "release" / binary_name).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ModelCompileError(f"cargo did not produce optimizer executable {binary}")
    return (str(binary),)


def capacity_packed_component_placement(
    package_root: Path,
    manifest: Json,
    device_capacity_bytes: dict[str, int],
) -> dict[str, str]:
    device_ids = tuple(device_capacity_bytes)
    if not device_ids:
        raise ModelCompileError("component placement requires devices")
    if any(
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0
        for capacity in device_capacity_bytes.values()
    ):
        raise ModelCompileError(
            "component placement requires positive device capacities"
        )
    weighted, output_transducer_ids = _weighted_signal_components(
        package_root,
        manifest,
    )
    if not weighted:
        raise ModelCompileError(
            "compiled package has no independently placeable signal processors"
        )
    if len(device_ids) == 1:
        placement = {component_id: device_ids[0] for component_id, _ in weighted}
    else:
        placement = _contiguous_capacity_packed_partition(
            weighted,
            device_capacity_bytes,
        )
    output_device = placement[weighted[-1][0]]
    placement.update(
        (component_id, output_device) for component_id in output_transducer_ids
    )
    return placement


def _weighted_signal_components(
    package_root: Path,
    manifest: Json,
) -> tuple[list[tuple[str, int]], list[str]]:
    components = manifest.get("circuit_graph", {}).get("components")
    if not isinstance(components, list) or not components:
        raise ModelCompileError("compiled package has no component graph")
    tensor_index = _tensor_index(package_root, manifest)
    tensor_sizes = {
        str(name): int(metadata["byte_count"])
        for name, metadata in tensor_index["tensors"].items()
    }
    weighted: list[tuple[str, int]] = []
    output_transducer_ids: list[str] = []
    charged_tensors: set[str] = set()
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
        if runtime_role == "output_transducer":
            output_transducer_ids.append(component_id)
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
        newly_charged = names - charged_tensors
        charged_tensors.update(names)
        weighted.append(
            (component_id, sum(tensor_sizes[name] for name in newly_charged))
        )
    return weighted, output_transducer_ids


def _build_target(
    *,
    package_manifest: Path,
    run_root: Path,
    profiles: tuple[Json, ...],
    placement: dict[str, str],
    residency_plan: Json,
    available_device_capacity_bytes: dict[str, int],
    reserved_device_capacity_bytes: dict[str, int],
    capacity_probe: LinuxDrmDeviceCapacityProbe,
    selected_observations: tuple[Json, ...],
    driver_files: tuple[Path, ...],
    component_command: tuple[str, ...],
    validation_command: tuple[str, ...],
    runtime_implementation_fingerprint: str,
    qualification_regime: QualificationRegime,
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
    if set(device_ids) != set(available_device_capacity_bytes) or set(
        device_ids
    ) != set(reserved_device_capacity_bytes):
        raise ModelCompileError(
            "residency admission capacities do not match live target devices"
        )
    manifest = _read_json(package_manifest, "compiled package manifest")
    capacity_digest = declared_capacity_reservation_digest(
        profiles,
        reserved_device_capacity_bytes,
        capacity_probe.policy,
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
            "speculative_draft_tokens": (qualification_regime.speculative_draft_tokens),
        },
        "environment": {
            "runtime_implementation_fingerprint": (runtime_implementation_fingerprint),
            "vulkan_driver_manifests": [str(path) for path in driver_files],
            "device_capacity_policy": capacity_probe.policy.to_json(),
            "capacity_observations": [
                dict(item)
                for item in sorted(
                    selected_observations,
                    key=lambda item: item["device_id"],
                )
            ],
            "residency_admission": {
                "plan": residency_plan,
                "available_device_capacity_bytes": dict(
                    sorted(available_device_capacity_bytes.items())
                ),
                "reserved_device_capacity_bytes": dict(
                    sorted(reserved_device_capacity_bytes.items())
                ),
            },
        },
        "capacity_reservation_digest": capacity_digest,
        "residency_scope": "capacity_partition",
    }
    target_id = stable_contract_id(
        "optimization_target",
        manifest["package_id"],
        sorted(capability_classes),
        list(device_ids),
        placement,
        qualification_regime.to_json(),
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
        qualification_regime=qualification_regime,
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
                ExactResidentExpansionProofVerifier(
                    source_artifacts=source_artifacts,
                    candidate_workspace_root=candidate_workspace,
                ),
            )
        ),
        lease_manager=VerifiedCapacityLeaseManager(
            lock_root=lease_root,
            probe_capacity_reservation_state=(
                capacity_probe.target_capacity_reservation_state
            ),
        ),
        estimate_execution_nanoseconds=lambda _plan, _policy: None,
    )


def _select_capability_groups(
    profiles: tuple[Json, ...],
    *,
    available_capacity_by_device: dict[str, int],
    package_manifest: Path,
    manifest: Json,
    residency_planner_command: tuple[str, ...],
    speculative_draft_tokens: int,
    residency_policy: str,
    explicit_selection: bool,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[_SelectedCapabilityGroup, ...]:
    if {_device_id(profile) for profile in profiles} != set(
        available_capacity_by_device
    ):
        raise ModelCompileError(
            "live device capacities do not match optimization profiles"
        )
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
        raise ModelCompileError("compiled package max_context_activations is invalid")
    components = manifest.get("circuit_graph", {}).get("components")
    if not isinstance(components, list):
        raise ModelCompileError("compiled package has no component graph")
    placeable_component_count = sum(
        isinstance(component, dict)
        and component.get("runtime_role") == "signal_processor"
        for component in components
    )
    for capability, raw_group in sorted(by_capability.items()):
        group = sorted(raw_group, key=_device_topology_key)
        ranked = sorted(
            group,
            key=lambda profile: (
                -available_capacity_by_device[_device_id(profile)],
                _device_id(profile),
            ),
        )
        candidate_groups = (
            (tuple(group),)
            if explicit_selection
            else tuple(
                tuple(ranked[:device_count])
                for device_count in range(
                    1,
                    min(len(group), placeable_component_count) + 1,
                )
            )
        )
        selected_group = None
        case_failures = []
        weighted_components, _ = _weighted_signal_components(
            package_manifest.parent,
            manifest,
        )
        for index, candidate_profiles in enumerate(candidate_groups, start=1):
            device_ids = tuple(_device_id(profile) for profile in candidate_profiles)
            capacities = {
                device_id: available_capacity_by_device[device_id]
                for device_id in device_ids
            }
            effective_capacities = dict(capacities)
            previous_placement = None
            refinement_failure = None
            for refinement in range(
                1,
                placeable_component_count + len(device_ids) + 1,
            ):
                placement = capacity_packed_component_placement(
                    package_manifest.parent,
                    manifest,
                    effective_capacities,
                )
                if placement == previous_placement:
                    refinement_failure = (
                        "exact residency correction converged to an "
                        "over-capacity placement"
                    )
                    break
                case_id = f"{capability}:{index}:{refinement}"
                case = RuntimeResidencyPlanningCase(
                    case_id=case_id,
                    default_device_id=device_ids[0],
                    component_placement=placement,
                    context_capacity_activations=max_context,
                    speculative_draft_tokens=speculative_draft_tokens,
                    residency_policy=residency_policy,
                )
                plan = plan_runtime_residency_cases(
                    command=residency_planner_command,
                    package_manifest=package_manifest,
                    cases=(case,),
                    cancel_requested=cancel_requested,
                )[case_id]
                planned_devices = _capacity_admission_bytes_by_device(plan)
                if set(planned_devices) != set(capacities):
                    raise ModelCompileError(
                        "runtime residency plan devices do not match the "
                        "requested placement topology"
                    )
                oversized = {
                    device_id: {
                        "planned": planned_devices[device_id],
                        "available_capacity": capacities[device_id],
                    }
                    for device_id in capacities
                    if planned_devices[device_id] > capacities[device_id]
                }
                if not oversized:
                    selected_group = _SelectedCapabilityGroup(
                        profiles=candidate_profiles,
                        placement=placement,
                        residency_plan=plan,
                        available_device_capacity_bytes=capacities,
                        # Demand-retained resources are admitted against their
                        # complete eventual set; reserve the measured tier that
                        # was proven safe so later expert loads cannot fail.
                        reserved_device_capacity_bytes=capacities,
                    )
                    break
                weights_by_device = {
                    device_id: sum(
                        weight
                        for component_id, weight in weighted_components
                        if placement[component_id] == device_id
                    )
                    for device_id in device_ids
                }
                corrected = {
                    device_id: capacities[device_id]
                    - max(
                        0,
                        planned_devices[device_id] - weights_by_device[device_id],
                    )
                    for device_id in device_ids
                }
                if any(capacity <= 0 for capacity in corrected.values()):
                    refinement_failure = (
                        f"fixed runtime residency exhausts capacity: {oversized}"
                    )
                    break
                previous_placement = placement
                effective_capacities = corrected
            if selected_group is not None:
                break
            case_failures.append(
                f"{len(candidate_profiles)} device(s): "
                + (refinement_failure or "residency correction did not converge")
            )
        if selected_group is None:
            failures.append(
                f"{capability} cannot safely host the planned runtime "
                f"working set ({'; '.join(case_failures)})"
            )
            continue
        groups.append(selected_group)
    if not groups:
        raise ModelCompileError(
            "no Vulkan capability class can host the compiled package: "
            + "; ".join(failures)
        )
    return tuple(groups)


def _capacity_admission_bytes_by_device(plan: Json) -> dict[str, int]:
    device_plans = plan.get("device_plans")
    if not isinstance(device_plans, list) or not device_plans:
        raise ModelCompileError("runtime residency plan has no device plans")
    retained: dict[str, int] = {}
    for device in device_plans:
        if not isinstance(device, dict):
            raise ModelCompileError("runtime residency device plan is malformed")
        device_id = device.get("device_id")
        parameters = device.get("parameter_residency")
        resource_store = device.get("resource_store")
        working_set = device.get("working_set")
        if (
            not isinstance(device_id, str)
            or not device_id
            or not isinstance(parameters, dict)
            or not isinstance(resource_store, dict)
            or not isinstance(working_set, dict)
        ):
            raise ModelCompileError("runtime residency device plan is malformed")
        fields = (
            parameters.get("maximum_addressable_bytes"),
            resource_store.get("metadata_device_bytes"),
            resource_store.get("transfer_staging_device_bytes"),
            resource_store.get("maximum_dynamic_allocation_padding_bytes"),
            working_set.get("transient_state_bytes"),
            working_set.get("activation_headroom_bytes"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in fields
        ):
            raise ModelCompileError(
                f"runtime residency device {device_id!r} has invalid retained-byte accounting"
            )
        if device_id in retained:
            raise ModelCompileError(
                f"runtime residency plan repeats device {device_id!r}"
            )
        if plan.get("residency_policy") == "demand_paged":
            initial = device.get("initial_device_resident_bytes")
            if isinstance(initial, bool) or not isinstance(initial, int) or initial < 0:
                raise ModelCompileError(
                    f"runtime residency device {device_id!r} has invalid "
                    "initial-byte accounting"
                )
            retained[device_id] = (
                initial
                + resource_store["maximum_load_wave_payload_bytes"]
                + resource_store["maximum_dynamic_allocation_padding_bytes"]
            )
        else:
            retained[device_id] = sum(fields)
    return retained


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


def _contiguous_capacity_packed_partition(
    components: list[tuple[str, int]],
    device_capacity_bytes: dict[str, int],
) -> dict[str, str]:
    # Mapping order is the runtime's device preference order. Pack each device
    # before crossing a physical boundary; sorting identifiers here would turn
    # an execution-cost decision back into an arbitrary naming decision.
    device_ids = tuple(device_capacity_bytes)
    if len(components) < len(device_ids):
        raise ModelCompileError(
            "contiguous placement cannot assign more devices than components"
        )
    placement: dict[str, str] = {}
    cursor = 0
    for device_index, device_id in enumerate(device_ids):
        remaining_devices = len(device_ids) - device_index
        device_capacity = device_capacity_bytes[device_id]
        if device_index == len(device_ids) - 1:
            remaining = components[cursor:]
            for component_id, _ in remaining:
                placement[component_id] = device_id
            break
        current_weight = 0
        while cursor < len(components):
            components_after = len(components) - (cursor + 1)
            devices_after = remaining_devices - 1
            if components_after < devices_after:
                break
            component_id, weight = components[cursor]
            if current_weight + weight > device_capacity:
                break
            placement[component_id] = device_id
            current_weight += weight
            cursor += 1
        if current_weight == 0:
            component_id, weight = components[cursor]
            placement[component_id] = device_id
            cursor += 1
    if set(placement) != {component_id for component_id, _ in components}:
        raise ModelCompileError("capacity-packed placement omitted compiled components")
    return placement


def _is_vulkan_gpu(profile: Json) -> bool:
    identity = profile.get("hardware_identity", {})
    provenance = profile.get("provenance", {})
    return (
        identity.get("device_kind") == "gpu"
        and provenance.get("api") == "vulkan"
    )


def _device_id(profile: Json) -> str:
    return str(profile["hardware_identity"]["stable_device_id"])


def _device_topology_key(profile: Json) -> tuple[int, int, int, int, str]:
    location = str(profile["hardware_identity"].get("physical_location", ""))
    if not location.startswith("pci:"):
        return (1 << 16, 1 << 8, 1 << 8, 8, _device_id(profile))
    domain_bus, device_function = normalize_pci_address(
        location.removeprefix("pci:")
    ).rsplit(":", 1)
    domain, bus = domain_bus.split(":", 1)
    device, function = device_function.split(".", 1)
    return (
        int(domain, 16),
        int(bus, 16),
        int(device, 16),
        int(function),
        _device_id(profile),
    )


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
