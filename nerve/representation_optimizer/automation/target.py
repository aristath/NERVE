from __future__ import annotations

import fcntl
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.planning import BenchmarkPolicy
from nerve.representation_optimizer.benchmarking.protocols import (
    NormalExecutionAdapter,
)
from nerve.representation_optimizer.contracts import (
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    contract_digest,
    require_device_state_digest,
    validate_contract,
)
from nerve.representation_optimizer.providers.types import ProviderCandidatePlan
from nerve.representation_optimizer.qualification import QualificationRegime
from nerve.representation_optimizer.staging.artifact_validation import (
    ArtifactValidatorRegistry,
)
from nerve.representation_optimizer.staging.protocols import (
    CandidateOrdinaryRelowerer,
    CandidatePhysicalOptimizer,
    CandidateSemanticConstructor,
)
from nerve.representation_optimizer.validation.proofs import (
    ProofVerifierRegistry,
)
from nerve.representation_optimizer.validation.protocols import (
    BehavioralValidationAdapter,
)


@dataclass(frozen=True)
class CandidateToolchain:
    semantic_constructor: CandidateSemanticConstructor
    ordinary_relowerer: CandidateOrdinaryRelowerer
    physical_optimizer: CandidatePhysicalOptimizer
    artifact_validators: ArtifactValidatorRegistry | None = None


class CandidateToolchainResolver(Protocol):
    def resolve(self, plan: ProviderCandidatePlan) -> CandidateToolchain:
        """Return the generic construction toolchain for a provider plan."""


class DeviceLeaseManager(Protocol):
    def acquire(
        self,
        target: OptimizationTarget,
    ) -> AbstractContextManager[None]:
        """Exclusively lease and verify every device used by the target."""


class NoDeviceLeaseManager:
    """Valid only for targets whose execution never touches an accelerator."""

    def acquire(
        self,
        target: OptimizationTarget,
    ) -> AbstractContextManager[None]:
        if target.requires_device_lease:
            raise ModelCompileError(
                f"target {target.target_id!r} requires a real device lease manager"
            )
        return nullcontext()


@dataclass(frozen=True)
class VerifiedDeviceLeaseManager:
    """Cross-process device locks plus attested idle-state probes."""

    lock_root: Path
    probe_idle_state_digest: Callable[["OptimizationTarget"], str]

    @contextmanager
    def acquire(self, target: OptimizationTarget) -> Iterator[None]:
        expected = str(target.matched_conditions["idle_device_state_digest"])
        if self.lock_root.is_symlink():
            raise ModelCompileError("device lease root must not be a symlink")
        root = self.lock_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        descriptors: list[int] = []
        try:
            for device_id in sorted(
                str(device["device_id"])
                for device in target.matched_conditions["devices"]
            ):
                lock_name = stable_device_lock_name(device_id)
                path = root / f"{lock_name}.lock"
                flags = os.O_RDWR | os.O_CREAT
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags, 0o644)
                try:
                    fcntl.flock(
                        descriptor,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError as error:
                    os.close(descriptor)
                    raise ModelCompileError(
                        f"device {device_id!r} already has an optimizer lease"
                    ) from error
                except BaseException:
                    os.close(descriptor)
                    raise
                descriptors.append(descriptor)
            before = self.probe_idle_state_digest(target)
            if before != expected:
                raise ModelCompileError(
                    f"target {target.target_id!r} is not at its declared idle "
                    "device baseline before execution"
                )
            try:
                yield
            finally:
                after = self.probe_idle_state_digest(target)
                if after != expected:
                    raise ModelCompileError(
                        f"target {target.target_id!r} did not return to its "
                        "declared idle device baseline"
                    )
        finally:
            for descriptor in reversed(descriptors):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def stable_device_lock_name(device_id: str) -> str:
    from hashlib import sha256

    if not device_id:
        raise ModelCompileError("device lease requires a stable device identity")
    return sha256(device_id.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OptimizationTarget:
    target_id: str
    synthesis_profile: Json
    hardware_profiles: tuple[Json, ...]
    matched_conditions: Json
    qualification_regime: QualificationRegime
    requires_device_lease: bool
    toolchains: CandidateToolchainResolver
    benchmark_adapter: NormalExecutionAdapter
    validation_adapter: BehavioralValidationAdapter
    proof_verifiers: ProofVerifierRegistry
    lease_manager: DeviceLeaseManager
    estimate_execution_nanoseconds: Callable[
        [ProviderCandidatePlan, BenchmarkPolicy], int | None
    ]
    benchmark_policy: BenchmarkPolicy = BenchmarkPolicy()

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ModelCompileError("optimization target requires target_id")
        if not self.hardware_profiles:
            raise ModelCompileError(
                "optimization target requires at least one hardware profile"
            )
        profiles = tuple(dict(profile) for profile in self.hardware_profiles)
        for profile in profiles:
            validate_contract(
                profile,
                expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
            )
        validate_contract(
            self.synthesis_profile,
            expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA,
        )
        synthesis_id = self.synthesis_profile["profile_id"]
        if synthesis_id not in {profile["profile_id"] for profile in profiles}:
            raise ModelCompileError(
                "optimization synthesis profile is not part of target hardware"
            )
        expected_devices = sorted(
            (
                {
                    "device_id": profile["hardware_identity"]["stable_device_id"],
                    "hardware_profile_digest": contract_digest(profile),
                    "capability_class": profile["capability_class"],
                    "api": profile["provenance"]["api"],
                }
                for profile in profiles
            ),
            key=lambda item: item["device_id"],
        )
        if self.matched_conditions.get("devices") != expected_devices:
            raise ModelCompileError(
                "optimization target matched devices do not match its profiles"
            )
        if self.matched_conditions.get("exclusive_residency") is not True:
            raise ModelCompileError(
                "optimization target must require exclusive residency"
            )
        controls = self.matched_conditions.get("controls")
        if (
            not isinstance(controls, dict)
            or controls.get("speculative_draft_tokens")
            != self.qualification_regime.speculative_draft_tokens
        ):
            raise ModelCompileError(
                "optimization target matched controls do not match its "
                "qualification regime"
            )
        require_device_state_digest(
            self.matched_conditions.get("idle_device_state_digest"),
            "optimization target idle_device_state_digest",
        )

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(str(profile["profile_id"]) for profile in self.hardware_profiles)
        )
