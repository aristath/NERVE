from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import contract_digest


AMD_PCI_VENDOR_ID = "0x1002"
_PCI_ADDRESS = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


@dataclass(frozen=True)
class DeviceIdlePolicy:
    """Observable conditions required before accelerator residency is leased."""

    maximum_vram_fraction_ppm: int = 10_000
    maximum_busy_percent: int = 0
    maximum_driver_context_vram_bytes: int = 1 * 1024 * 1024
    maximum_driver_context_gtt_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 0 <= self.maximum_vram_fraction_ppm <= 1_000_000:
            raise ModelCompileError(
                "idle-device VRAM fraction must be between 0 and 1000000 ppm"
            )
        if not 0 <= self.maximum_busy_percent <= 100:
            raise ModelCompileError(
                "idle-device busy percentage must be between 0 and 100"
            )
        for field in (
            "maximum_driver_context_vram_bytes",
            "maximum_driver_context_gtt_bytes",
        ):
            value = getattr(self, field)
            if value < 0:
                raise ModelCompileError(
                    f"idle-device {field} must not be negative"
                )

    def to_json(self) -> Json:
        return {
            "maximum_vram_fraction_ppm": self.maximum_vram_fraction_ppm,
            "maximum_busy_percent": self.maximum_busy_percent,
            "maximum_driver_context_vram_bytes": (
                self.maximum_driver_context_vram_bytes
            ),
            "maximum_driver_context_gtt_bytes": (
                self.maximum_driver_context_gtt_bytes
            ),
            "resident_process_policy": (
                "no_engine_activity_or_process_above_driver_context_envelope"
            ),
        }


@dataclass(frozen=True)
class DeviceIdleObservation:
    device_id: str
    pci_address: str
    drm_card: str
    vram_total_bytes: int
    vram_used_bytes: int
    busy_percent: int
    resident_processes: tuple[Json, ...]

    def to_json(self) -> Json:
        return {
            "device_id": self.device_id,
            "pci_address": self.pci_address,
            "drm_card": self.drm_card,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_used_bytes": self.vram_used_bytes,
            "busy_percent": self.busy_percent,
            "resident_processes": [dict(item) for item in self.resident_processes],
        }


class LinuxAmdDeviceStateProbe:
    """Fail-closed AMD residency probe using PCI, DRM, procfs, and sysfs."""

    def __init__(
        self,
        *,
        sysfs_drm_root: Path = Path("/sys/class/drm"),
        proc_root: Path = Path("/proc"),
        policy: DeviceIdlePolicy = DeviceIdlePolicy(),
    ) -> None:
        self.sysfs_drm_root = sysfs_drm_root.resolve()
        self.proc_root = proc_root.resolve()
        self.policy = policy

    def observe(self, profile: Json) -> DeviceIdleObservation:
        identity = profile.get("hardware_identity")
        if not isinstance(identity, dict):
            raise ModelCompileError("hardware profile has no identity")
        if (
            identity.get("device_kind") != "gpu"
            or str(identity.get("vendor_id", "")).lower() != AMD_PCI_VENDOR_ID
        ):
            raise ModelCompileError(
                "NERVE optimizer device probe accepts only AMD GPU profiles"
            )
        device_id = _required_text(identity, "stable_device_id")
        pci_address = _profile_pci_address(profile)
        card = self._drm_card(pci_address)
        device_root = card / "device"
        vendor = _read_text(device_root / "vendor").lower()
        if vendor != AMD_PCI_VENDOR_ID:
            raise ModelCompileError(
                f"DRM device {card.name!r} is not an AMD GPU"
            )
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
                f"AMD device {device_id!r} returned invalid residency counters"
            )
        processes = self._resident_processes(pci_address)
        return DeviceIdleObservation(
            device_id=device_id,
            pci_address=pci_address,
            drm_card=card.name,
            vram_total_bytes=total,
            vram_used_bytes=used,
            busy_percent=busy,
            resident_processes=processes,
        )

    def require_idle(self, profiles: tuple[Json, ...]) -> tuple[Json, ...]:
        if not profiles:
            raise ModelCompileError("idle-device probe requires hardware profiles")
        observations = tuple(
            sorted(
                (self.observe(profile) for profile in profiles),
                key=lambda item: item.device_id,
            )
        )
        failures: list[str] = []
        for observation in observations:
            used_fraction_ppm = (
                observation.vram_used_bytes * 1_000_000
                // observation.vram_total_bytes
            )
            if observation.resident_processes:
                owners = ", ".join(
                    f"{item['pid']}:{item['command']}"
                    for item in observation.resident_processes
                )
                failures.append(
                    f"{observation.device_id} has resident DRM consumers ({owners})"
                )
            if used_fraction_ppm > self.policy.maximum_vram_fraction_ppm:
                failures.append(
                    f"{observation.device_id} uses "
                    f"{observation.vram_used_bytes}/{observation.vram_total_bytes} "
                    "VRAM bytes"
                )
            if observation.busy_percent > self.policy.maximum_busy_percent:
                failures.append(
                    f"{observation.device_id} is {observation.busy_percent}% busy"
                )
        if failures:
            raise ModelCompileError(
                "AMD device is not at an idle residency baseline: "
                + "; ".join(failures)
            )
        return tuple(item.to_json() for item in observations)

    def idle_state_digest(self, profiles: tuple[Json, ...]) -> str:
        """Attest current idleness and return its stable semantic identity."""

        self.require_idle(profiles)
        return declared_idle_state_digest(profiles, self.policy)

    def target_idle_state_digest(self, target: object) -> str:
        raw_profiles = getattr(target, "hardware_profiles", None)
        if not isinstance(raw_profiles, tuple):
            raise ModelCompileError(
                "device-state probe received an invalid optimization target"
            )
        profiles = tuple(dict(profile) for profile in raw_profiles)
        observations = self.require_idle(profiles)
        matched = getattr(target, "matched_conditions", None)
        if not isinstance(matched, dict):
            raise ModelCompileError(
                "device-state probe received target conditions without "
                "an idle baseline"
            )
        environment = matched.get("environment")
        baselines = (
            environment.get("initial_idle_observations")
            if isinstance(environment, dict)
            else None
        )
        if not isinstance(baselines, list):
            raise ModelCompileError(
                "optimization target has no initial device-idle observations"
            )
        baseline_by_id = {
            str(item.get("device_id", "")): item
            for item in baselines
            if isinstance(item, dict)
        }
        if set(baseline_by_id) != {
            str(item["device_id"]) for item in observations
        }:
            raise ModelCompileError(
                "optimization target idle baseline does not match its devices"
            )
        for observation in observations:
            baseline = baseline_by_id[str(observation["device_id"])]
            baseline_used = baseline.get("vram_used_bytes")
            if (
                isinstance(baseline_used, bool)
                or not isinstance(baseline_used, int)
                or observation["vram_used_bytes"] > baseline_used
            ):
                raise ModelCompileError(
                    f"device {observation['device_id']!r} did not return to "
                    "its initial VRAM residency baseline"
                )
        return declared_idle_state_digest(profiles, self.policy)

    def _drm_card(self, pci_address: str) -> Path:
        candidates = []
        for card in sorted(self.sysfs_drm_root.glob("card[0-9]*")):
            if "-" in card.name or not (card / "device").exists():
                continue
            try:
                resolved = (card / "device").resolve(strict=True)
                vendor = _read_text(card / "device" / "vendor").lower()
            except ModelCompileError:
                continue
            if (
                normalize_pci_address(resolved.name) == pci_address
                and vendor == AMD_PCI_VENDOR_ID
            ):
                candidates.append(card)
        if len(candidates) != 1:
            raise ModelCompileError(
                f"AMD PCI device {pci_address!r} does not map to one DRM card"
            )
        return candidates[0]

    def _resident_processes(self, pci_address: str) -> tuple[Json, ...]:
        residents: dict[int, Json] = {}
        for process in sorted(self.proc_root.glob("[0-9]*")):
            try:
                pid = int(process.name)
            except ValueError:
                continue
            fdinfo_root = process / "fdinfo"
            if not fdinfo_root.is_dir():
                continue
            vram_bytes = 0
            gtt_bytes = 0
            engine_time_ns = 0
            for fdinfo in fdinfo_root.iterdir():
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
                engine_time_ns == 0
                and vram_bytes
                <= self.policy.maximum_driver_context_vram_bytes
                and gtt_bytes
                <= self.policy.maximum_driver_context_gtt_bytes
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


def declared_idle_state_digest(
    profiles: tuple[Json, ...],
    policy: DeviceIdlePolicy,
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
    return contract_digest(
        {
            "schema": "nerve.optimizer.device_idle_attestation.v1",
            "devices": devices,
            "policy": policy.to_json(),
            "state": "idle",
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
        f"AMD device {identity.get('stable_device_id')!r} has no PCI identity"
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError as error:
        raise ModelCompileError(
            f"required AMD device state file is unavailable: {path}"
        ) from error


def _read_nonnegative_integer(path: Path, label: str) -> int:
    try:
        value = int(_read_text(path))
    except ValueError as error:
        raise ModelCompileError(f"AMD {label} is not an integer") from error
    if value < 0:
        raise ModelCompileError(f"AMD {label} must not be negative")
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
