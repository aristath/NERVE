from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA = (
    "nerve.gate.external_device_reservation_snapshot.v1"
)
EXTERNAL_DEVICE_RESERVATION_REPORT_SCHEMA = (
    "nerve.gate.external_device_reservation_report.v1"
)
DEFAULT_EXTERNAL_RESERVATION_TOLERANCE_BYTES = 16 * 1024 * 1024
_PCI_ADDRESS = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)
_MEMORY_VALUE = re.compile(
    r"^(?P<value>[0-9]+)(?:\s*(?P<unit>B|KiB|MiB|GiB))?$",
    re.IGNORECASE,
)
_MEMORY_MULTIPLIERS = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
}


class DeviceReservationGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DrmProcessReservation:
    pid: int
    start_time_ticks: int
    command: str
    vram_bytes: int
    shared_bytes: int

    @property
    def allocation_bytes(self) -> int:
        return self.vram_bytes + self.shared_bytes


@dataclass(frozen=True)
class DrmDeviceReservation:
    pci_address: str
    drm_card: str
    vram_total_bytes: int | None
    vram_used_bytes: int | None
    busy_percent: int | None
    resident_processes: tuple[DrmProcessReservation, ...]


@dataclass(frozen=True)
class ExternalDeviceReservationSnapshot:
    schema: str
    devices: tuple[DrmDeviceReservation, ...]


@dataclass(frozen=True)
class ExternalDeviceReservationDeviceReport:
    pci_address: str
    restored: bool
    before: DrmDeviceReservation | None
    after: DrmDeviceReservation | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ExternalDeviceReservationReport:
    schema: str
    complete: bool
    selected_device_count: int
    restored_device_count: int
    devices: tuple[ExternalDeviceReservationDeviceReport, ...]
    errors: tuple[str, ...]


class ExternalDeviceReservationProbe(Protocol):
    def capture(self) -> ExternalDeviceReservationSnapshot: ...


class LinuxDrmExternalDeviceReservationProbe:
    def __init__(
        self,
        *,
        sysfs_drm_root: Path = Path("/sys/class/drm"),
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.sysfs_drm_root = sysfs_drm_root.resolve()
        self.proc_root = proc_root.resolve()

    def capture(self) -> ExternalDeviceReservationSnapshot:
        cards = self._drm_cards()
        processes = self._resident_processes(set(cards))
        devices = tuple(
            DrmDeviceReservation(
                pci_address=pci_address,
                drm_card=card.name,
                vram_total_bytes=capacity[0],
                vram_used_bytes=capacity[1],
                busy_percent=capacity[2],
                resident_processes=tuple(
                    sorted(
                        processes.get(pci_address, {}).values(),
                        key=lambda process: (process.pid, process.start_time_ticks),
                    )
                ),
            )
            for pci_address, (card, capacity) in sorted(cards.items())
        )
        if not devices:
            raise DeviceReservationGateError(
                "external device reservation probe found no Linux DRM devices"
            )
        return ExternalDeviceReservationSnapshot(
            schema=EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA,
            devices=devices,
        )

    def _drm_cards(
        self,
    ) -> dict[
        str,
        tuple[Path, tuple[int | None, int | None, int | None]],
    ]:
        cards: dict[
            str,
            tuple[Path, tuple[int | None, int | None, int | None]],
        ] = {}
        for card in sorted(self.sysfs_drm_root.glob("card[0-9]*")):
            if "-" in card.name or not (card / "device").exists():
                continue
            try:
                device_root = (card / "device").resolve(strict=True)
                pci_address = normalize_pci_address(device_root.name)
            except (OSError, DeviceReservationGateError):
                continue
            if pci_address in cards:
                raise DeviceReservationGateError(
                    f"PCI device {pci_address!r} maps to multiple DRM cards"
                )
            total_path = device_root / "mem_info_vram_total"
            used_path = device_root / "mem_info_vram_used"
            if total_path.exists() != used_path.exists():
                raise DeviceReservationGateError(
                    f"DRM card {card.name!r} exposes incomplete VRAM accounting"
                )
            busy_path = device_root / "gpu_busy_percent"
            busy_percent = (
                _nonnegative_integer_file(busy_path, "GPU activity")
                if busy_path.exists()
                else None
            )
            if busy_percent is not None and busy_percent > 100:
                raise DeviceReservationGateError(
                    f"DRM card {card.name!r} exposes invalid GPU activity"
                )
            capacity: tuple[int | None, int | None, int | None]
            if total_path.exists():
                total = _nonnegative_integer_file(total_path, "VRAM total")
                used = _nonnegative_integer_file(used_path, "VRAM used")
                if total <= 0 or used > total:
                    raise DeviceReservationGateError(
                        f"DRM card {card.name!r} exposes invalid VRAM accounting"
                    )
                capacity = (total, used, busy_percent)
            else:
                capacity = (None, None, busy_percent)
            cards[pci_address] = (card, capacity)
        return cards

    def _resident_processes(
        self,
        known_pci_addresses: set[str],
    ) -> dict[str, dict[tuple[int, int], DrmProcessReservation]]:
        allocations: dict[
            str,
            dict[tuple[int, int], tuple[str, int, int]],
        ] = {pci_address: {} for pci_address in known_pci_addresses}
        for process in sorted(self.proc_root.glob("[0-9]*")):
            try:
                pid = int(process.name)
            except ValueError:
                continue
            fdinfo_root = process / "fdinfo"
            if not fdinfo_root.is_dir():
                continue
            try:
                fdinfo_files = tuple(sorted(fdinfo_root.iterdir()))
            except (OSError, PermissionError):
                continue
            per_device: dict[str, list[int]] = {}
            seen_clients: set[
                tuple[str, str, str] | tuple[str, int, int] | tuple[str, str]
            ] = set()
            for fdinfo in fdinfo_files:
                try:
                    fields = _fdinfo_fields(fdinfo.read_text(errors="replace"))
                except (OSError, PermissionError):
                    continue
                raw_pci = fields.get("drm-pdev") or fields.get("drm-pci")
                if raw_pci is None:
                    continue
                try:
                    pci_address = normalize_pci_address(raw_pci)
                except DeviceReservationGateError:
                    continue
                if pci_address not in known_pci_addresses:
                    continue
                client_id = fields.get("drm-client-id")
                client_identity: (
                    tuple[str, str, str] | tuple[str, int, int] | tuple[str, str]
                )
                if client_id is not None:
                    if not client_id.isdecimal():
                        raise DeviceReservationGateError(
                            f"invalid DRM client identity {client_id!r} in {fdinfo}"
                        )
                    client_identity = ("drm-client", pci_address, client_id)
                else:
                    # Older drivers may omit drm-client-id. File identity is a
                    # conservative fallback: it removes duplicated descriptors
                    # without pretending distinct observed client IDs are one.
                    try:
                        fd_stat = (process / "fd" / fdinfo.name).stat()
                        client_identity = (
                            "inode",
                            fd_stat.st_dev,
                            fd_stat.st_ino,
                        )
                    except (OSError, PermissionError):
                        client_identity = ("fdinfo", fdinfo.name)
                if client_identity in seen_clients:
                    continue
                seen_clients.add(client_identity)
                vram_bytes = _local_memory_bytes(fields)
                shared_bytes = _shared_memory_bytes(fields)
                if vram_bytes == 0 and shared_bytes == 0:
                    continue
                totals = per_device.setdefault(pci_address, [0, 0])
                totals[0] += vram_bytes
                totals[1] += shared_bytes
            if not per_device:
                continue
            start_time_ticks = _process_start_time_ticks(process / "stat")
            if start_time_ticks is None:
                # The client exited after its fdinfo was sampled, so it is no
                # longer part of the pre/post reservation boundary.
                continue
            try:
                command = (process / "comm").read_text(errors="replace").strip()
            except (OSError, PermissionError):
                command = "unknown"
            command = command or "unknown"
            for pci_address, (vram_bytes, shared_bytes) in per_device.items():
                allocations[pci_address][(pid, start_time_ticks)] = (
                    command,
                    vram_bytes,
                    shared_bytes,
                )
        return {
            pci_address: {
                identity: DrmProcessReservation(
                    pid=identity[0],
                    start_time_ticks=identity[1],
                    command=values[0],
                    vram_bytes=values[1],
                    shared_bytes=values[2],
                )
                for identity, values in by_process.items()
            }
            for pci_address, by_process in allocations.items()
        }


def verify_external_device_reservations_restored(
    selected_pci_addresses: tuple[str, ...],
    before: ExternalDeviceReservationSnapshot,
    after: ExternalDeviceReservationSnapshot,
    *,
    tolerance_bytes: int = DEFAULT_EXTERNAL_RESERVATION_TOLERANCE_BYTES,
) -> ExternalDeviceReservationReport:
    errors = []
    if (
        isinstance(tolerance_bytes, bool)
        or not isinstance(tolerance_bytes, int)
        or tolerance_bytes < 0
    ):
        errors.append("external device reservation tolerance is invalid")
        tolerance_bytes = 0
    if before.schema != EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA or (
        after.schema != EXTERNAL_DEVICE_RESERVATION_SNAPSHOT_SCHEMA
    ):
        errors.append("external device reservation snapshot schema is unsupported")
    selected = []
    for raw_pci_address in selected_pci_addresses:
        try:
            selected.append(normalize_pci_address(raw_pci_address))
        except DeviceReservationGateError as error:
            errors.append(str(error))
    if not selected:
        errors.append("external device reservation proof has no selected devices")
    if len(set(selected)) != len(selected):
        errors.append("external device reservation proof repeats a selected device")
    selected = sorted(set(selected))
    before_by_pci = _reservation_devices_by_pci("before", before.devices, errors)
    after_by_pci = _reservation_devices_by_pci("after", after.devices, errors)
    devices = []
    for pci_address in selected:
        prior = before_by_pci.get(pci_address)
        current = after_by_pci.get(pci_address)
        device_errors = []
        if prior is None:
            device_errors.append("device was absent from the pre-workload snapshot")
        if current is None:
            device_errors.append("device was absent from the post-workload snapshot")
        if prior is not None and current is not None:
            _verify_external_device_reservation(
                prior,
                current,
                tolerance_bytes,
                device_errors,
            )
        devices.append(
            ExternalDeviceReservationDeviceReport(
                pci_address=pci_address,
                restored=not device_errors,
                before=prior,
                after=current,
                errors=tuple(device_errors),
            )
        )
        errors.extend(f"{pci_address}: {error}" for error in device_errors)
    restored_device_count = sum(device.restored for device in devices)
    complete = not errors and bool(devices) and restored_device_count == len(devices)
    return ExternalDeviceReservationReport(
        schema=EXTERNAL_DEVICE_RESERVATION_REPORT_SCHEMA,
        complete=complete,
        selected_device_count=len(selected),
        restored_device_count=restored_device_count,
        devices=tuple(devices),
        errors=tuple(errors),
    )


def _verify_external_device_reservation(
    before: DrmDeviceReservation,
    after: DrmDeviceReservation,
    tolerance_bytes: int,
    errors: list[str],
) -> None:
    if before.drm_card != after.drm_card:
        errors.append("DRM card identity changed")
    if before.vram_total_bytes != after.vram_total_bytes:
        errors.append("physical VRAM total changed")
    if before.vram_used_bytes is not None and after.vram_used_bytes is not None:
        retained = after.vram_used_bytes - before.vram_used_bytes
        process_vram_growth = max(
            0,
            sum(process.vram_bytes for process in after.resident_processes)
            - sum(process.vram_bytes for process in before.resident_processes),
        )
        unattributed_retained = retained - process_vram_growth
        if unattributed_retained > tolerance_bytes:
            errors.append(
                f"retained {unattributed_retained} unattributed aggregate VRAM bytes "
                "above the pre-workload "
                f"reservation (tolerance {tolerance_bytes})"
            )
    current_by_identity = {
        (process.pid, process.start_time_ticks): process
        for process in after.resident_processes
    }
    for prior in before.resident_processes:
        identity = (prior.pid, prior.start_time_ticks)
        current = current_by_identity.get(identity)
        if current is None:
            errors.append(
                f"lost pre-existing process {prior.pid}:{prior.start_time_ticks}:"
                f"{prior.command}"
            )
            continue
        if current.allocation_bytes + tolerance_bytes < prior.allocation_bytes:
            errors.append(
                f"pre-existing process {prior.pid}:{prior.start_time_ticks}:"
                f"{prior.command} lost {prior.allocation_bytes - current.allocation_bytes} "
                "DRM allocation bytes"
            )


def _reservation_devices_by_pci(
    stage: str,
    devices: tuple[DrmDeviceReservation, ...],
    errors: list[str],
) -> dict[str, DrmDeviceReservation]:
    by_pci = {}
    for device in devices:
        if not isinstance(device, DrmDeviceReservation):
            errors.append(
                f"{stage} external reservation snapshot has an invalid device"
            )
            continue
        try:
            pci_address = normalize_pci_address(device.pci_address)
        except DeviceReservationGateError as error:
            errors.append(f"{stage} snapshot: {error}")
            continue
        if pci_address in by_pci:
            errors.append(
                f"{stage} external reservation snapshot repeats {pci_address!r}"
            )
        _validate_reservation_device(stage, device, errors)
        by_pci[pci_address] = device
    return by_pci


def _validate_reservation_device(
    stage: str,
    device: DrmDeviceReservation,
    errors: list[str],
) -> None:
    prefix = f"{stage} external reservation snapshot {device.pci_address!r}"
    if not isinstance(device.drm_card, str) or not device.drm_card:
        errors.append(f"{prefix} has an invalid DRM card identity")
    total = device.vram_total_bytes
    used = device.vram_used_bytes
    busy = device.busy_percent
    if (total is None) != (used is None):
        errors.append(f"{prefix} has incomplete aggregate VRAM accounting")
    elif total is not None and (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(used, bool)
        or not isinstance(used, int)
        or used < 0
        or used > total
    ):
        errors.append(f"{prefix} has invalid aggregate VRAM accounting")
    if busy is not None and (
        isinstance(busy, bool) or not isinstance(busy, int) or not 0 <= busy <= 100
    ):
        errors.append(f"{prefix} has invalid GPU activity")
    process_identities = set()
    for process in device.resident_processes:
        if not isinstance(process, DrmProcessReservation):
            errors.append(f"{prefix} has an invalid resident process")
            continue
        values = (
            process.pid,
            process.start_time_ticks,
            process.vram_bytes,
            process.shared_bytes,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            )
            or process.pid == 0
        ):
            errors.append(f"{prefix} has invalid resident-process counters")
        if not isinstance(process.command, str) or not process.command:
            errors.append(f"{prefix} has an invalid resident-process command")
        identity = (process.pid, process.start_time_ticks)
        if identity in process_identities:
            errors.append(f"{prefix} repeats resident process {identity}")
        process_identities.add(identity)


def normalize_pci_address(value: str) -> str:
    if not isinstance(value, str):
        raise DeviceReservationGateError("PCI address must be a string")
    match = _PCI_ADDRESS.fullmatch(value.strip())
    if match is None:
        raise DeviceReservationGateError(f"invalid PCI address {value!r}")
    return (
        f"{(match.group('domain') or '0000').lower()}:"
        f"{match.group('bus').lower()}:"
        f"{match.group('device').lower()}."
        f"{match.group('function')}"
    )


def _fdinfo_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _memory_bytes(value: str) -> int:
    match = _MEMORY_VALUE.fullmatch(value.strip())
    if match is None:
        raise DeviceReservationGateError(
            f"invalid DRM memory allocation value {value!r}"
        )
    amount = int(match.group("value"))
    unit = (match.group("unit") or "B").casefold()
    return amount * _MEMORY_MULTIPLIERS[unit]


def _local_memory_bytes(fields: dict[str, str]) -> int:
    alias = fields.get("drm-memory-vram")
    if alias is not None:
        return _memory_bytes(alias)
    return sum(
        _memory_bytes(value)
        for key, value in fields.items()
        if key.startswith("drm-resident-vram")
    )


def _shared_memory_bytes(fields: dict[str, str]) -> int:
    gtt_alias = fields.get("drm-memory-gtt")
    gtt = (
        _memory_bytes(gtt_alias)
        if gtt_alias is not None
        else _memory_bytes(fields.get("drm-resident-gtt", "0"))
    )
    return gtt + sum(
        _memory_bytes(fields.get(key, "0"))
        for key in ("drm-resident-system", "drm-resident-stolen")
    )


def _nonnegative_integer_file(path: Path, label: str) -> int:
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError) as error:
        raise DeviceReservationGateError(
            f"could not read {label} from {path}"
        ) from error
    if value < 0:
        raise DeviceReservationGateError(f"{label} must not be negative")
    return value


def _process_start_time_ticks(path: Path) -> int | None:
    try:
        value = path.read_text(errors="replace").strip()
    except FileNotFoundError:
        return None
    except (OSError, PermissionError) as error:
        raise DeviceReservationGateError(
            f"could not read DRM client process identity from {path}"
        ) from error
    command_end = value.rfind(")")
    if command_end < 0:
        raise DeviceReservationGateError(
            f"DRM client process identity in {path} is malformed"
        )
    fields = value[command_end + 1 :].split()
    try:
        start_time_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise DeviceReservationGateError(
            f"DRM client process identity in {path} is malformed"
        ) from error
    if start_time_ticks < 0:
        raise DeviceReservationGateError(
            f"DRM client process identity in {path} is invalid"
        )
    return start_time_ticks
