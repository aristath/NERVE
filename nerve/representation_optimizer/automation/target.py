from __future__ import annotations

import fcntl
import os
import time
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
        """Reserve target capacity while serializing NERVE optimizers."""


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
class CapacityLeaseState:
    """Live device state associated with one declared capacity reservation."""

    reservation_digest: str
    observations: tuple[Json, ...]
    release_vram_tolerance_bytes: int
    release_settle_timeout_ns: int
    release_poll_interval_ns: int

    def __post_init__(self) -> None:
        require_device_state_digest(
            self.reservation_digest,
            "capacity lease reservation_digest",
        )
        if (
            isinstance(self.release_vram_tolerance_bytes, bool)
            or not isinstance(self.release_vram_tolerance_bytes, int)
            or self.release_vram_tolerance_bytes < 0
        ):
            raise ModelCompileError(
                "capacity lease release tolerance must be a nonnegative integer"
            )
        if (
            isinstance(self.release_settle_timeout_ns, bool)
            or not isinstance(self.release_settle_timeout_ns, int)
            or self.release_settle_timeout_ns < 0
        ):
            raise ModelCompileError(
                "capacity lease release settlement timeout must be a nonnegative integer"
            )
        if (
            isinstance(self.release_poll_interval_ns, bool)
            or not isinstance(self.release_poll_interval_ns, int)
            or self.release_poll_interval_ns <= 0
        ):
            raise ModelCompileError(
                "capacity lease release poll interval must be a positive integer"
            )
        _capacity_observations_by_device(self.observations)


@dataclass(frozen=True)
class VerifiedCapacityLeaseManager:
    """NERVE-only locks plus live capacity checks before and after execution."""

    lock_root: Path
    probe_capacity_reservation_state: Callable[
        ["OptimizationTarget"], CapacityLeaseState
    ]
    monotonic_ns: Callable[[], int] = time.monotonic_ns
    sleep: Callable[[float], None] = time.sleep

    @contextmanager
    def acquire(self, target: OptimizationTarget) -> Iterator[None]:
        expected = str(target.matched_conditions["capacity_reservation_digest"])
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
            before = self.probe_capacity_reservation_state(target)
            if before.reservation_digest != expected:
                raise ModelCompileError(
                    f"target {target.target_id!r} does not satisfy its declared "
                    "device-capacity reservation before execution"
                )
            try:
                yield
            finally:
                _wait_for_capacity_release(
                    before=before,
                    expected_reservation_digest=expected,
                    target=target,
                    probe=self.probe_capacity_reservation_state,
                    monotonic_ns=self.monotonic_ns,
                    sleep=self.sleep,
                )
        finally:
            for descriptor in reversed(descriptors):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _wait_for_capacity_release(
    *,
    before: CapacityLeaseState,
    expected_reservation_digest: str,
    target: OptimizationTarget,
    probe: Callable[[OptimizationTarget], CapacityLeaseState],
    monotonic_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> None:
    deadline = monotonic_ns() + before.release_settle_timeout_ns
    last_error: ModelCompileError | None = None
    while True:
        try:
            after = probe(target)
            if after.reservation_digest != expected_reservation_digest:
                raise ModelCompileError(
                    f"target {target.target_id!r} did not restore its "
                    "declared device-capacity reservation"
                )
            _require_capacity_released(before, after, target.target_id)
            return
        except ModelCompileError as error:
            last_error = error
        now = monotonic_ns()
        if now >= deadline:
            assert last_error is not None
            raise last_error
        sleep_ns = min(before.release_poll_interval_ns, deadline - now)
        sleep(sleep_ns / 1_000_000_000)


def _require_capacity_released(
    before: CapacityLeaseState,
    after: CapacityLeaseState,
    target_id: str,
) -> None:
    if (
        before.release_vram_tolerance_bytes
        != after.release_vram_tolerance_bytes
    ):
        raise ModelCompileError(
            f"target {target_id!r} changed its device-release tolerance"
        )
    if (
        before.release_settle_timeout_ns != after.release_settle_timeout_ns
        or before.release_poll_interval_ns != after.release_poll_interval_ns
    ):
        raise ModelCompileError(
            f"target {target_id!r} changed its device-release settlement policy"
        )
    before_by_id = _capacity_observations_by_device(before.observations)
    after_by_id = _capacity_observations_by_device(after.observations)
    if set(before_by_id) != set(after_by_id):
        raise ModelCompileError(
            f"target {target_id!r} did not restore the same device set"
        )
    tolerance = before.release_vram_tolerance_bytes
    failures = []
    for device_id in sorted(before_by_id):
        prior = before_by_id[device_id]
        current = after_by_id[device_id]
        if current["vram_total_bytes"] != prior["vram_total_bytes"]:
            failures.append(f"{device_id} changed total VRAM")
            continue
        growth = current["vram_used_bytes"] - prior["vram_used_bytes"]
        if growth > tolerance:
            failures.append(
                f"{device_id} retained {growth} VRAM bytes above its "
                f"pre-execution allocation (tolerance {tolerance})"
            )
        prior_pids = {
            process["pid"]
            for process in prior["resident_processes"]
        }
        current_pids = {
            process["pid"]
            for process in current["resident_processes"]
        }
        missing = sorted(prior_pids - current_pids)
        if missing:
            failures.append(
                f"{device_id} lost pre-existing resident process(es) {missing}"
            )
    if failures:
        raise ModelCompileError(
            f"target {target_id!r} did not restore its pre-execution AMD "
            "device state: " + "; ".join(failures)
        )


def _capacity_observations_by_device(
    observations: tuple[Json, ...],
) -> dict[str, Json]:
    by_device: dict[str, Json] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ModelCompileError("capacity lease observation must be an object")
        device_id = observation.get("device_id")
        total = observation.get("vram_total_bytes")
        used = observation.get("vram_used_bytes")
        processes = observation.get("resident_processes")
        if (
            not isinstance(device_id, str)
            or not device_id
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or isinstance(used, bool)
            or not isinstance(used, int)
            or used < 0
            or used > total
            or not isinstance(processes, list)
            or any(
                not isinstance(process, dict)
                or isinstance(process.get("pid"), bool)
                or not isinstance(process.get("pid"), int)
                or process["pid"] < 0
                for process in processes
            )
            or len({process["pid"] for process in processes}) != len(processes)
            or device_id in by_device
        ):
            raise ModelCompileError("capacity lease observation is invalid")
        by_device[device_id] = observation
    if not by_device:
        raise ModelCompileError("capacity lease requires device observations")
    return by_device


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
        if self.matched_conditions.get("residency_scope") != "capacity_partition":
            raise ModelCompileError(
                "optimization target must declare capacity-partition residency"
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
            self.matched_conditions.get("capacity_reservation_digest"),
            "optimization target capacity_reservation_digest",
        )

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(str(profile["profile_id"]) for profile in self.hardware_profiles)
        )
