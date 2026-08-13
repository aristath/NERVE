from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import device_state_digest
from nerve.representation_optimizer.automation.target import CapacityLeaseState


RUNTIME_DEVICE_LOCAL_MEMORY_POLICY_SCHEMA = (
    "nerve.runtime.device_local_memory_policy.v1"
)
_PCI_ADDRESS = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


@dataclass(frozen=True)
class DeviceCapacityPolicy:
    """Conservative share of currently free VRAM that NERVE may reserve."""

    reservable_free_vram_fraction_ppm: int
    minimum_reservable_vram_bytes: int = 1
    material_process_vram_bytes: int = 64 * 1024 * 1024
    material_process_gtt_bytes: int = 64 * 1024 * 1024
    admission_vram_tolerance_bytes: int = 16 * 1024 * 1024
    release_vram_tolerance_bytes: int = 16 * 1024 * 1024
    release_settle_timeout_ns: int = 5_000_000_000
    release_poll_interval_ns: int = 50_000_000

    @classmethod
    def from_runtime_policy(cls, document: Json) -> DeviceCapacityPolicy:
        expected = {
            "schema",
            "capacity_parts_per_million",
            "protected_headroom_fraction_ppm",
            "reservable_free_vram_fraction_ppm",
        }
        if set(document) != expected:
            raise ModelCompileError(
                "runtime device-local memory policy fields are invalid"
            )
        if document.get("schema") != RUNTIME_DEVICE_LOCAL_MEMORY_POLICY_SCHEMA:
            raise ModelCompileError(
                "runtime device-local memory policy schema is unsupported"
            )
        parts = document.get("capacity_parts_per_million")
        protected = document.get("protected_headroom_fraction_ppm")
        reservable = document.get("reservable_free_vram_fraction_ppm")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (parts, protected, reservable)
        ) or parts != 1_000_000 or protected + reservable != parts:
            raise ModelCompileError(
                "runtime device-local memory policy fractions are invalid"
            )
        return cls(reservable_free_vram_fraction_ppm=reservable)

    def __post_init__(self) -> None:
        if not 0 < self.reservable_free_vram_fraction_ppm <= 1_000_000:
            raise ModelCompileError(
                "reservable free-VRAM fraction must be in (0, 1000000] ppm"
            )
        for field in (
            "minimum_reservable_vram_bytes",
            "material_process_vram_bytes",
            "material_process_gtt_bytes",
            "admission_vram_tolerance_bytes",
            "release_vram_tolerance_bytes",
        ):
            value = getattr(self, field)
            if value < 0:
                raise ModelCompileError(f"device-capacity {field} must not be negative")
        if (
            isinstance(self.release_settle_timeout_ns, bool)
            or not isinstance(self.release_settle_timeout_ns, int)
            or self.release_settle_timeout_ns < 0
        ):
            raise ModelCompileError(
                "device-capacity release_settle_timeout_ns must be a "
                "nonnegative integer"
            )
        if (
            isinstance(self.release_poll_interval_ns, bool)
            or not isinstance(self.release_poll_interval_ns, int)
            or self.release_poll_interval_ns <= 0
        ):
            raise ModelCompileError(
                "device-capacity release_poll_interval_ns must be a positive integer"
            )

    def reservable_vram_bytes(self, *, total: int, used: int) -> int:
        if total <= 0 or used < 0 or used > total:
            raise ModelCompileError("device returned invalid VRAM capacity counters")
        return (
            (total - used)
            * self.reservable_free_vram_fraction_ppm
            // 1_000_000
        )

    def to_json(self) -> Json:
        return {
            "reservable_free_vram_fraction_ppm": (
                self.reservable_free_vram_fraction_ppm
            ),
            "minimum_reservable_vram_bytes": self.minimum_reservable_vram_bytes,
            "material_process_vram_bytes": self.material_process_vram_bytes,
            "material_process_gtt_bytes": self.material_process_gtt_bytes,
            "admission_vram_tolerance_bytes": (
                self.admission_vram_tolerance_bytes
            ),
            "release_vram_tolerance_bytes": (
                self.release_vram_tolerance_bytes
            ),
            "release_settle_timeout_ns": self.release_settle_timeout_ns,
            "release_poll_interval_ns": self.release_poll_interval_ns,
            "concurrent_workload_policy": "preserve_and_share_unreserved_capacity",
            "release_policy": "declared_capacity_restored_after_driver_settlement",
        }


@dataclass(frozen=True)
class DeviceCapacityObservation:
    device_id: str
    pci_address: str
    drm_card: str
    vram_total_bytes: int
    vram_used_bytes: int
    vram_free_bytes: int
    reservable_vram_bytes: int
    busy_percent: int
    resident_processes: tuple[Json, ...]

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "pci_address": self.pci_address,
            "drm_card": self.drm_card,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_used_bytes": self.vram_used_bytes,
            "vram_free_bytes": self.vram_free_bytes,
            "reservable_vram_bytes": self.reservable_vram_bytes,
            "busy_percent": self.busy_percent,
            "resident_processes": [dict(item) for item in self.resident_processes],
        }


class LinuxDrmSysfsDeviceCapacityProbe:
    """Measure shareable VRAM through driver-published Linux DRM counters.

    Selection is capability-driven rather than vendor-driven. A driver is
    eligible when its DRM device publishes exact total/used VRAM and current
    activity counters; drivers without that telemetry fail closed instead of
    receiving an invented capacity.
    """

    def __init__(
        self,
        *,
        policy: DeviceCapacityPolicy,
        sysfs_drm_root: Path = Path("/sys/class/drm"),
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.sysfs_drm_root = sysfs_drm_root.resolve()
        self.proc_root = proc_root.resolve()
        self.policy = policy

    def observe(self, profile: Json) -> DeviceCapacityObservation:
        identity = profile.get("hardware_identity")
        if not isinstance(identity, dict):
            raise ModelCompileError("hardware profile has no identity")
        provenance = profile.get("provenance")
        if (
            identity.get("device_kind") != "gpu"
            or not isinstance(provenance, dict)
            or provenance.get("api") != "vulkan"
        ):
            raise ModelCompileError(
                "NERVE optimizer device probe accepts only Vulkan GPU profiles"
            )
        device_id = _required_text(identity, "stable_device_id")
        pci_address = _profile_pci_address(profile)
        card = self._drm_card(pci_address)
        device_root = card / "device"
        total = _read_nonnegative_integer(
            device_root / "mem_info_vram_total",
            "total VRAM",
        )
        used = _read_nonnegative_integer(
            device_root / "mem_info_vram_used",
            "used VRAM",
        )
        busy = _read_nonnegative_integer(
            device_root / "gpu_busy_percent",
            "GPU busy percentage",
        )
        if total <= 0 or used > total or busy > 100:
            raise ModelCompileError(
                f"device {device_id!r} returned invalid residency counters"
            )
        processes = self._material_resident_processes(pci_address)
        return DeviceCapacityObservation(
            device_id=device_id,
            pci_address=pci_address,
            drm_card=card.name,
            vram_total_bytes=total,
            vram_used_bytes=used,
            vram_free_bytes=total - used,
            reservable_vram_bytes=self.policy.reservable_vram_bytes(
                total=total,
                used=used,
            ),
            busy_percent=busy,
            resident_processes=processes,
        )

    def require_capacity(
        self,
        profiles: tuple[Json, ...],
        required_capacity_bytes: dict[str, int] | None = None,
    ) -> tuple[Json, ...]:
        if not profiles:
            raise ModelCompileError("device-capacity probe requires hardware profiles")
        observations = tuple(
            sorted(
                (self.observe(profile) for profile in profiles),
                key=lambda item: item.device_id,
            )
        )
        requirements = required_capacity_bytes or {
            observation.device_id: self.policy.minimum_reservable_vram_bytes
            for observation in observations
        }
        observed_ids = {observation.device_id for observation in observations}
        if set(requirements) != observed_ids:
            raise ModelCompileError(
                "capacity reservation does not match the probed devices"
            )
        failures = []
        for observation in observations:
            required = requirements[observation.device_id]
            if isinstance(required, bool) or not isinstance(required, int) or required < 0:
                raise ModelCompileError(
                    f"device {observation.device_id!r} has an invalid VRAM reservation"
                )
            shortfall = max(0, required - observation.reservable_vram_bytes)
            if shortfall > self.policy.admission_vram_tolerance_bytes:
                failures.append(
                    f"{observation.device_id} offers "
                    f"{observation.reservable_vram_bytes} reservable VRAM bytes, "
                    f"requires {required} (shortfall {shortfall} exceeds the "
                    f"{self.policy.admission_vram_tolerance_bytes}-byte "
                    "capacity-observation tolerance)"
                )
        if failures:
            raise ModelCompileError(
                "device has insufficient unreserved VRAM capacity: "
                + "; ".join(failures)
            )
        return tuple(item.to_json() for item in observations)

    def target_capacity_reservation_state(
        self,
        target: object,
    ) -> CapacityLeaseState:
        raw_profiles = getattr(target, "hardware_profiles", None)
        if not isinstance(raw_profiles, tuple):
            raise ModelCompileError(
                "device-state probe received an invalid optimization target"
            )
        profiles = tuple(dict(profile) for profile in raw_profiles)
        matched = getattr(target, "matched_conditions", None)
        if not isinstance(matched, dict):
            raise ModelCompileError(
                "device-capacity probe received invalid target conditions"
            )
        environment = matched.get("environment")
        admission = (
            environment.get("residency_admission")
            if isinstance(environment, dict)
            else None
        )
        reservations = (
            admission.get("reserved_device_capacity_bytes")
            if isinstance(admission, dict)
            else None
        )
        if not isinstance(reservations, dict):
            raise ModelCompileError(
                "optimization target has no device-capacity reservation"
            )
        baseline_observations = (
            environment.get("capacity_observations")
            if isinstance(environment, dict)
            else None
        )
        if not isinstance(baseline_observations, list):
            raise ModelCompileError(
                "optimization target has no pre-execution capacity observations"
            )
        required = {}
        for device_id, value in reservations.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelCompileError(
                    "optimization target has an invalid device-capacity reservation"
                )
            required[str(device_id)] = value
        current_observations = self.require_capacity(profiles, required)
        self._require_preexisting_residents(
            baseline_observations,
            current_observations,
        )
        return CapacityLeaseState(
            reservation_digest=declared_capacity_reservation_digest(
                profiles,
                required,
                self.policy,
            ),
            observations=current_observations,
            release_vram_tolerance_bytes=(
                self.policy.release_vram_tolerance_bytes
            ),
            release_settle_timeout_ns=self.policy.release_settle_timeout_ns,
            release_poll_interval_ns=self.policy.release_poll_interval_ns,
        )

    def _require_preexisting_residents(
        self,
        baseline_observations: list[object],
        current_observations: tuple[Json, ...],
    ) -> None:
        baseline_by_id = {
            str(observation.get("device_id", "")): observation
            for observation in baseline_observations
            if isinstance(observation, dict)
        }
        current_by_id = {
            str(observation["device_id"]): observation
            for observation in current_observations
        }
        if set(baseline_by_id) != set(current_by_id):
            raise ModelCompileError(
                "capacity observations do not match the reserved devices"
            )
        missing = []
        for device_id, baseline in baseline_by_id.items():
            baseline_processes = baseline.get("resident_processes")
            if not isinstance(baseline_processes, list):
                raise ModelCompileError(
                    "capacity observation has invalid resident-process evidence"
                )
            current_processes = current_by_id[device_id]["resident_processes"]
            current_pids = {
                process.get("pid")
                for process in current_processes
                if isinstance(process, dict)
            }
            for process in baseline_processes:
                if not isinstance(process, dict) or not isinstance(
                    process.get("pid"),
                    int,
                ):
                    raise ModelCompileError(
                        "capacity observation has invalid resident-process evidence"
                    )
                if process["pid"] not in current_pids:
                    missing.append(
                        f"{device_id}:{process['pid']}:{process.get('command', 'unknown')}"
                    )
        if missing:
            raise ModelCompileError(
                "pre-existing resident allocations are no longer present: "
                + ", ".join(missing)
            )

    def _drm_card(self, pci_address: str) -> Path:
        candidates = []
        for card in sorted(self.sysfs_drm_root.glob("card[0-9]*")):
            if "-" in card.name or not (card / "device").exists():
                continue
            try:
                resolved = (card / "device").resolve(strict=True)
            except OSError:
                continue
            if normalize_pci_address(resolved.name) == pci_address:
                candidates.append(card)
        if len(candidates) != 1:
            raise ModelCompileError(
                f"PCI device {pci_address!r} does not map to one DRM card"
            )
        return candidates[0]

    def _material_resident_processes(
        self,
        pci_address: str,
    ) -> tuple[Json, ...]:
        """Return DRM clients too large to be display/driver background state.

        DRM engine counters are cumulative over a client's lifetime and cannot
        establish current activity. Process records are attribution only: they
        never exclude a device whose remaining capacity satisfies the requested
        reservation. File descriptors duplicated from one DRM client share an
        inode and must be counted once rather than multiplying reported memory.
        """

        residents: dict[int, Json] = {}
        for process in sorted(self.proc_root.glob("[0-9]*")):
            try:
                pid = int(process.name)
            except ValueError:
                continue
            fdinfo_root = process / "fdinfo"
            if not fdinfo_root.is_dir():
                continue
            try:
                fdinfo_files = tuple(fdinfo_root.iterdir())
            except (OSError, PermissionError):
                # Material hidden residency is still caught by the global
                # device VRAM and activity counters. Process attribution is
                # necessarily limited to procfs entries visible to this user.
                continue
            vram_bytes = 0
            gtt_bytes = 0
            engine_time_ns = 0
            seen_clients: set[tuple[int, int] | tuple[str, str]] = set()
            for fdinfo in fdinfo_files:
                try:
                    fd_stat = (process / "fd" / fdinfo.name).stat()
                    client_identity: tuple[int, int] | tuple[str, str] = (
                        fd_stat.st_dev,
                        fd_stat.st_ino,
                    )
                except (OSError, PermissionError):
                    # Synthetic procfs fixtures and kernels which hide an fd
                    # still get a deterministic, conservative identity.
                    client_identity = ("fdinfo", fdinfo.name)
                if client_identity in seen_clients:
                    continue
                seen_clients.add(client_identity)
                try:
                    fields = _fdinfo_fields(fdinfo.read_text(errors="replace"))
                except (OSError, PermissionError):
                    continue
                raw_pci = fields.get("drm-pdev") or fields.get("drm-pci")
                if raw_pci is None:
                    continue
                try:
                    current_pci = normalize_pci_address(raw_pci)
                except ModelCompileError:
                    continue
                if current_pci != pci_address:
                    continue
                vram_bytes += _memory_bytes(
                    fields.get("drm-memory-vram", "0")
                )
                gtt_bytes += _memory_bytes(
                    fields.get("drm-memory-gtt", "0")
                )
                engine_time_ns += sum(
                    _duration_nanoseconds(value)
                    for key, value in fields.items()
                    if key.startswith("drm-engine-")
                )
            if (
                vram_bytes
                <= self.policy.material_process_vram_bytes
                and gtt_bytes
                <= self.policy.material_process_gtt_bytes
            ):
                continue
            try:
                command = (process / "comm").read_text(errors="replace").strip()
            except (OSError, PermissionError):
                command = "unknown"
            residents[pid] = {
                "pid": pid,
                "command": command or "unknown",
                "vram_bytes": vram_bytes,
                "gtt_bytes": gtt_bytes,
                "engine_time_ns": engine_time_ns,
            }
        return tuple(residents[pid] for pid in sorted(residents))


def declared_capacity_reservation_digest(
    profiles: tuple[Json, ...],
    reserved_capacity_bytes: dict[str, int],
    policy: DeviceCapacityPolicy,
) -> str:
    devices = sorted(
        (
            {
                "device_id": profile["hardware_identity"]["stable_device_id"],
                "pci_address": _profile_pci_address(profile),
            }
            for profile in profiles
        ),
        key=lambda item: item["device_id"],
    )
    return device_state_digest(
        {
            "schema": "nerve.optimizer.device_capacity_reservation.v1",
            "devices": devices,
            "reserved_capacity_bytes": dict(
                sorted(reserved_capacity_bytes.items())
            ),
            "policy": policy.to_json(),
            "state": "capacity_reserved",
        }
    )


def normalize_pci_address(value: str) -> str:
    match = _PCI_ADDRESS.fullmatch(value.strip())
    if match is None:
        raise ModelCompileError(f"invalid PCI address {value!r}")
    return (
        f"{(match.group('domain') or '0000').lower()}:"
        f"{match.group('bus').lower()}:"
        f"{match.group('device').lower()}."
        f"{match.group('function')}"
    )


def _profile_pci_address(profile: Json) -> str:
    identity = profile["hardware_identity"]
    location = str(identity.get("physical_location", ""))
    if location.startswith("pci:"):
        return normalize_pci_address(location.removeprefix("pci:"))
    binding = profile.get("runtime_bindings", {}).get(
        "vulkan_runtime_binding",
        {},
    )
    if isinstance(binding, dict) and isinstance(binding.get("pci_address"), str):
        return normalize_pci_address(binding["pci_address"])
    raise ModelCompileError(
        f"device {identity.get('stable_device_id')!r} has no PCI identity"
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        raise ModelCompileError(
            f"required device state file is unavailable: {path}"
        ) from error


def _read_nonnegative_integer(path: Path, label: str) -> int:
    try:
        value = int(_read_text(path))
    except ValueError as error:
        raise ModelCompileError(f"{label} is not an integer") from error
    if value < 0:
        raise ModelCompileError(f"{label} must not be negative")
    return value


def _fdinfo_fields(payload: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in payload.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _memory_bytes(value: str) -> int:
    amount, unit = _quantity(value)
    multiplier = {
        "": 1,
        "B": 1,
        "KiB": 1024,
        "MiB": 1024 * 1024,
        "GiB": 1024 * 1024 * 1024,
    }.get(unit)
    if multiplier is None:
        raise ModelCompileError(f"unsupported DRM memory unit {unit!r}")
    return amount * multiplier


def _duration_nanoseconds(value: str) -> int:
    amount, unit = _quantity(value)
    multiplier = {
        "": 1,
        "ns": 1,
        "us": 1_000,
        "ms": 1_000_000,
        "s": 1_000_000_000,
    }.get(unit)
    if multiplier is None:
        raise ModelCompileError(f"unsupported DRM engine-time unit {unit!r}")
    return amount * multiplier


def _quantity(value: str) -> tuple[int, str]:
    match = re.match(r"^\s*(\d+)(?:\s+([A-Za-z]+))?", value)
    if match is None:
        return 0, ""
    return int(match.group(1)), match.group(2) or ""


def _required_text(document: Json, field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"hardware identity {field!r} is missing")
    return value
