from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

from nerve.text_quality import repeated_segment


WARMUP_PROMPT = "hi"
MEASURED_PROMPTS = (
    "Who are you?",
    "what is the capital of Greece?",
    'How many cities named "Corinth" are there?',
    "What is your knowledge cutoff date?",
    "I asked you earlier to tell me the capital of a country. Which country was that?",
)
CANONICAL_CONVERSATION_PROMPTS = (WARMUP_PROMPT, *MEASURED_PROMPTS)
CANONICAL_OUTPUT_TOKEN_ALLOWANCE = 65_536
_PROMPT_MARKER = b"you> "
_RESPONSE_PREFIX = "llm> "
_TURN_START = _PROMPT_MARKER.decode() + _RESPONSE_PREFIX
_TURN_ERROR_MARKER = b"\nturn_error: "
_NEW_CONVERSATION_COMMAND = "/new"
_SESSION_RESET_MARKER = b"session_reset: "
_STATS_MARKER = "\nstats:\n"
_EXECUTION_MARKER = "\nexecution:\n"
_DETERMINISM_MARKER = "\ndeterminism:\n"
_STAT_LINE = re.compile(r"^  ([a-z][a-z0-9_]*)=(.+)$")
_RESIDENCY_POLICY = re.compile(r"^  policy=([^ ]+)", re.MULTILINE)
_RESIDENCY_COUNTER_LINE = re.compile(r"^  ([a-z_]+)\(([^)]+)\)=([^\n]+)$", re.MULTILINE)
_RESIDENCY_PAYLOAD_AND_UNITS_LINE = re.compile(
    r"^  payload_bytes\(initial/current/high_water/maximum\)="
    r"(\d+)/(\d+)/(\d+)/(\d+) "
    r"units\(initial/current/high_water/addressable\)="
    r"(\d+)/(\d+)/(\d+)/(\d+)$",
    re.MULTILINE,
)
_PHYSICAL_EXECUTION_SUMMARY_START = (
    "physical_execution=VulkanMountedPhysicalExecutionSummary {"
)
_PHYSICAL_EXECUTION_SUMMARY = re.compile(
    re.escape(_PHYSICAL_EXECUTION_SUMMARY_START) + r"(?P<body>[^{}\n]*)}"
)
_SHUTDOWN_MARKER = re.compile(r"(?:^|\n)you> shutdown:\n")
_DEVICE_RESTORATION_MARKER = re.compile(r"(?:^|\n)device_restoration:\n")
_DEVICE_RESTORATION_SCHEMA = "nerve.runtime.device_local_memory_restoration.v1"
_SHUTDOWN_SUMMARY_LINE = re.compile(
    r"^  complete=(true|false) streams=(\d+) packages=(\d+) "
    r"scheduler_in_flight=(\d+)$"
)
_SHUTDOWN_TOTALS_LINE = re.compile(
    r"^  physical_devices_acknowledged=(\d+)/(\d+) released_units=(\d+) "
    r"released_payload_bytes=(\d+) cancelled_loads=(\d+)$"
)
_SHUTDOWN_PACKAGE_LINE = re.compile(
    r"^  package=(\S+) scope=(\S+) physical_devices_acknowledged=(\d+)/(\d+)$"
)
_SHUTDOWN_DEVICE_LINE = re.compile(
    r"^  store=(\S+) physical_device=(\S+) acknowledged=(true|false) "
    r"remaining_units=(\d+) remaining_payload_bytes=(\d+) error=(.+)$"
)
_PHYSICAL_EXECUTION_FIELDS = (
    "tensor_parallel_island_count",
    "whole_expert_parallel_island_count",
    "intra_expert_tensor_parallel_island_count",
    "hybrid_island_count",
    "selected_resource_placement_count",
)
_DETERMINISM_FIELDS = (
    "generated_tokens",
    "selection_counters",
    "resident_state",
)
_DETERMINISM_DIGEST_PREFIXES = {
    "generated_tokens": "nerve.runtime.token_ids_sha256.v1:",
    "selection_counters": "nerve.runtime.selection_counters_sha256.v1:",
    "resident_state": "nerve.optimizer.artifact_sha256.v1:",
}
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

_CUMULATIVE_RESIDENCY_COUNTER_GROUPS = {
    "gpu_accesses",
    "residency_requests",
    "residency_eviction",
    "transfers",
}
_RESIDENCY_GAUGE_GROUPS = {"memory_tiers"}
_FULLY_WARM_ZERO_COUNTERS = (
    "gpu_accesses.misses",
    "residency_requests.load_required",
    "residency_requests.succeeded",
    "residency_requests.failed",
    "residency_requests.cancelled",
    "residency_eviction.cycles",
    "residency_eviction.units",
    "residency_eviction.payload_bytes",
    "residency_eviction.device_bytes",
    "residency_eviction.reloads",
    "transfers.reads",
    "transfers.source_bytes",
    "transfers.resident_bytes",
    "transfers.uploaded_bytes",
    "transfers.read_ms",
    "transfers.derivation_ms",
    "transfers.upload_ms",
    "transfers.blocking_ms",
)
_MAXIMUM_RESIDENT_CONVERSATION_SETS = 16


class ConversationGateError(RuntimeError):
    pass


class ResidentConversationError(ConversationGateError):
    def __init__(self, message: str, transcript: str) -> None:
        super().__init__(message)
        self.transcript = transcript


@dataclass(frozen=True)
class PhysicalExecutionSummary:
    tensor_parallel_island_count: int
    whole_expert_parallel_island_count: int
    intra_expert_tensor_parallel_island_count: int
    hybrid_island_count: int
    selected_resource_placement_count: int

    @property
    def total_tensor_parallel_island_count(self) -> int:
        return (
            self.tensor_parallel_island_count
            + self.intra_expert_tensor_parallel_island_count
            + self.hybrid_island_count
        )


@dataclass(frozen=True)
class ShutdownDeviceReport:
    store_id: str
    physical_device_id: str
    acknowledged: bool
    remaining_units: int
    remaining_payload_bytes: int
    error: str | None


@dataclass(frozen=True)
class ShutdownPackageReport:
    package_id: str
    execution_scope: str
    acknowledged_device_count: int
    physical_device_count: int
    devices: list[ShutdownDeviceReport]


@dataclass(frozen=True)
class ShutdownReport:
    complete: bool
    stream_count: int
    package_count: int
    scheduler_in_flight_activation_count: int
    acknowledged_device_count: int
    physical_device_count: int
    released_unit_count: int
    released_payload_bytes: int
    cancelled_load_count: int
    packages: list[ShutdownPackageReport]


@dataclass(frozen=True)
class DeviceRestorationDeviceReport:
    physical_device_id: str
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass(frozen=True)
class DeviceRestorationReport:
    schema: str
    complete: bool
    physical_device_count: int
    restored_device_count: int
    devices: list[DeviceRestorationDeviceReport]


@dataclass(frozen=True)
class ConversationTurn:
    prompt: str
    response: str
    stats: dict[str, int | float | str]
    residency_policy: str | None = None
    residency_counters: dict[str, int | float] | None = None
    residency_gauges: dict[str, int | float] | None = None
    execution_counters: dict[str, int | float | str] | None = None
    determinism_digests: dict[str, str] | None = None

    @property
    def decode_tokens_per_second(self) -> float:
        value = self.stats.get("decode_tokens_per_second")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ConversationGateError(
                f"turn {self.prompt!r} reported invalid decode_tokens_per_second: "
                f"{value!r}"
            )
        return float(value)

    @property
    def prefill_tokens_per_second(self) -> float:
        value = self.stats.get("prefill_tokens_per_second")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ConversationGateError(
                f"turn {self.prompt!r} reported invalid prefill_tokens_per_second: "
                f"{value!r}"
            )
        return float(value)


@dataclass(frozen=True)
class ConversationSetReport:
    warmup: ConversationTurn
    turns: list[ConversationTurn]
    mean_decode_tokens_per_second: float
    mean_prefill_tokens_per_second: float
    residency_policy: str | None
    residency_start: dict[str, int | float] | None
    residency_end: dict[str, int | float] | None
    residency_delta: dict[str, int | float] | None
    residency_gauges_start: dict[str, int | float] | None
    residency_gauges_end: dict[str, int | float] | None


@dataclass(frozen=True)
class ConversationSeedReport:
    seed: int
    command: list[str]
    transcript_sha256: str
    physical_execution: PhysicalExecutionSummary | None
    shutdown: ShutdownReport
    device_restoration: DeviceRestorationReport
    discarded_warmup_sets: list[ConversationSetReport]
    measured_set: ConversationSetReport


@dataclass(frozen=True)
class ConversationGateReport:
    ok: bool
    minimum_decode_tokens_per_second: float
    minimum_tensor_parallel_islands: int
    require_thinking: bool
    warmup_conversation_sets: int
    maximum_conversation_sets: int
    package: dict[str, Any]
    runs: list[ConversationSeedReport]


def _parse_scalar(raw: str) -> int | float | str:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _residency_metrics(
    report: str,
) -> tuple[
    str | None,
    dict[str, int | float] | None,
    dict[str, int | float] | None,
]:
    marker = "\nresource_residency:\n"
    if marker not in report:
        return None, None, None
    block = report.split(marker, 1)[1]
    policy_match = _RESIDENCY_POLICY.search(block)
    counters: dict[str, int | float] = {}
    gauges: dict[str, int | float] = {}
    payload_and_units = _RESIDENCY_PAYLOAD_AND_UNITS_LINE.findall(block)
    if len(payload_and_units) > 1:
        raise ConversationGateError(
            "runtime reported more than one resource payload/unit gauge line per turn"
        )
    if payload_and_units:
        values = tuple(int(value) for value in payload_and_units[0])
        for label, value in zip(
            (
                "payload_bytes.initial",
                "payload_bytes.current",
                "payload_bytes.high_water",
                "payload_bytes.maximum",
                "units.initial",
                "units.current",
                "units.high_water",
                "units.addressable",
            ),
            values,
            strict=True,
        ):
            gauges[label] = value
    for group, raw_labels, raw_values in _RESIDENCY_COUNTER_LINE.findall(block):
        if group not in (
            _CUMULATIVE_RESIDENCY_COUNTER_GROUPS | _RESIDENCY_GAUGE_GROUPS
        ):
            continue
        labels = raw_labels.split("/")
        values = raw_values.split("/")
        if len(labels) != len(values):
            raise ConversationGateError(
                f"resource residency group {group!r} has {len(labels)} labels "
                f"but {len(values)} values"
            )
        for label, value in zip(labels, values, strict=True):
            parsed = _parse_scalar(value)
            if (
                isinstance(parsed, bool)
                or not isinstance(parsed, (int, float))
                or not math.isfinite(parsed)
            ):
                raise ConversationGateError(
                    f"resource residency counter {group}.{label} is not numeric: "
                    f"{value!r}"
                )
            destination = (
                counters if group in _CUMULATIVE_RESIDENCY_COUNTER_GROUPS else gauges
            )
            destination[f"{group}.{label}"] = parsed
    return (
        policy_match.group(1) if policy_match is not None else None,
        counters or None,
        gauges or None,
    )


def _execution_metrics(report: str) -> dict[str, int | float | str] | None:
    if _EXECUTION_MARKER not in report:
        return None
    block = report.split(_EXECUTION_MARKER, 1)[1]
    counters: dict[str, int | float | str] = {}
    for line in block.splitlines():
        if line and not line.startswith("  "):
            break
        match = _STAT_LINE.match(line)
        if match is None:
            if counters:
                break
            continue
        counters[match.group(1)] = _parse_scalar(match.group(2))
    return counters


def _determinism_metrics(report: str) -> dict[str, str] | None:
    marker_count = report.count(_DETERMINISM_MARKER)
    if marker_count == 0:
        return None
    if marker_count != 1:
        raise ConversationGateError(
            "runtime reported more than one determinism evidence block for a turn"
        )
    block = report.split(_DETERMINISM_MARKER, 1)[1]
    digests: dict[str, str] = {}
    for line in block.splitlines():
        if line and not line.startswith("  "):
            break
        match = _STAT_LINE.match(line)
        if match is None:
            if digests:
                break
            continue
        key, value = match.groups()
        if key not in _DETERMINISM_FIELDS or key in digests or not value.strip():
            raise ConversationGateError(
                "runtime determinism report has an unknown, duplicate, or empty field"
            )
        digest = value.strip()
        prefix = _DETERMINISM_DIGEST_PREFIXES[key]
        payload = digest.removeprefix(prefix)
        if payload == digest or _SHA256_HEX.fullmatch(payload) is None:
            raise ConversationGateError(
                f"runtime determinism report field {key!r} is not a canonical digest"
            )
        digests[key] = digest
    if set(digests) != set(_DETERMINISM_FIELDS):
        raise ConversationGateError(
            "runtime determinism report is missing field(s): "
            + ", ".join(sorted(set(_DETERMINISM_FIELDS).difference(digests)))
        )
    return digests


def parse_physical_execution_summary(
    transcript: str,
    *,
    minimum_tensor_parallel_islands: int = 0,
) -> PhysicalExecutionSummary | None:
    if minimum_tensor_parallel_islands < 0:
        raise ConversationGateError(
            "minimum tensor-parallel island count cannot be negative"
        )
    marker_count = transcript.count(_PHYSICAL_EXECUTION_SUMMARY_START)
    if marker_count == 0:
        if minimum_tensor_parallel_islands > 0:
            raise ConversationGateError(
                "runtime did not report mounted physical execution"
            )
        return None
    if marker_count != 1:
        raise ConversationGateError(
            "runtime reported more than one physical execution summary"
        )
    match = _PHYSICAL_EXECUTION_SUMMARY.search(transcript)
    if match is None:
        raise ConversationGateError("runtime physical execution summary is malformed")

    values: dict[str, int] = {}
    for entry in match.group("body").split(","):
        parts = entry.strip().split(":", 1)
        if len(parts) != 2:
            raise ConversationGateError(
                "runtime physical execution summary is malformed"
            )
        key, raw_value = (part.strip() for part in parts)
        if key not in _PHYSICAL_EXECUTION_FIELDS or key in values:
            raise ConversationGateError(
                "runtime physical execution summary has an unknown or duplicate field"
            )
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ConversationGateError(
                f"runtime physical execution summary field {key!r} is not an integer"
            ) from error
        if value < 0:
            raise ConversationGateError(
                f"runtime physical execution summary field {key!r} is negative"
            )
        values[key] = value
    missing = set(_PHYSICAL_EXECUTION_FIELDS).difference(values)
    if missing:
        raise ConversationGateError(
            "runtime physical execution summary is missing field(s): "
            + ", ".join(sorted(missing))
        )

    summary = PhysicalExecutionSummary(**values)
    if summary.total_tensor_parallel_island_count < minimum_tensor_parallel_islands:
        raise ConversationGateError(
            "runtime mounted "
            f"{summary.total_tensor_parallel_island_count} tensor-parallel island(s); "
            f"required at least {minimum_tensor_parallel_islands}"
        )
    return summary


def parse_shutdown_report(transcript: str) -> ShutdownReport:
    markers = list(_SHUTDOWN_MARKER.finditer(transcript))
    if not markers:
        raise ConversationGateError("runtime did not report structured shutdown")
    if len(markers) != 1:
        raise ConversationGateError(
            "runtime reported structured shutdown more than once"
        )
    shutdown_end = len(transcript)
    restoration_markers = list(_DEVICE_RESTORATION_MARKER.finditer(transcript))
    if restoration_markers:
        if (
            len(restoration_markers) != 1
            or restoration_markers[0].start() < markers[0].end()
        ):
            raise ConversationGateError(
                "runtime device restoration report is misplaced or repeated"
            )
        shutdown_end = restoration_markers[0].start()
    lines = transcript[markers[0].end() : shutdown_end].splitlines()
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) < 2:
        raise ConversationGateError("runtime structured shutdown report is incomplete")
    if any(not line for line in lines):
        raise ConversationGateError("runtime structured shutdown report is malformed")

    summary = _SHUTDOWN_SUMMARY_LINE.fullmatch(lines[0])
    totals = _SHUTDOWN_TOTALS_LINE.fullmatch(lines[1])
    if summary is None or totals is None:
        raise ConversationGateError("runtime structured shutdown summary is malformed")
    complete = summary.group(1) == "true"
    stream_count, package_count, scheduler_in_flight = (
        int(value) for value in summary.groups()[1:]
    )
    (
        acknowledged_device_count,
        physical_device_count,
        released_unit_count,
        released_payload_bytes,
        cancelled_load_count,
    ) = (int(value) for value in totals.groups())

    packages: list[ShutdownPackageReport] = []
    current_package: tuple[str, str, int, int] | None = None
    current_devices: list[ShutdownDeviceReport] = []

    def finish_package() -> None:
        nonlocal current_package, current_devices
        if current_package is None:
            return
        package_id, execution_scope, acknowledged, physical = current_package
        packages.append(
            ShutdownPackageReport(
                package_id=package_id,
                execution_scope=execution_scope,
                acknowledged_device_count=acknowledged,
                physical_device_count=physical,
                devices=current_devices,
            )
        )
        current_package = None
        current_devices = []

    for line in lines[2:]:
        package_match = _SHUTDOWN_PACKAGE_LINE.fullmatch(line)
        if package_match is not None:
            finish_package()
            current_package = (
                package_match.group(1),
                package_match.group(2),
                int(package_match.group(3)),
                int(package_match.group(4)),
            )
            continue
        device_match = _SHUTDOWN_DEVICE_LINE.fullmatch(line)
        if device_match is None or current_package is None:
            raise ConversationGateError(
                "runtime structured shutdown package/device report is malformed"
            )
        raw_error = device_match.group(6)
        current_devices.append(
            ShutdownDeviceReport(
                store_id=device_match.group(1),
                physical_device_id=device_match.group(2),
                acknowledged=device_match.group(3) == "true",
                remaining_units=int(device_match.group(4)),
                remaining_payload_bytes=int(device_match.group(5)),
                error=None if raw_error == "None" else raw_error,
            )
        )
    finish_package()

    report = ShutdownReport(
        complete=complete,
        stream_count=stream_count,
        package_count=package_count,
        scheduler_in_flight_activation_count=scheduler_in_flight,
        acknowledged_device_count=acknowledged_device_count,
        physical_device_count=physical_device_count,
        released_unit_count=released_unit_count,
        released_payload_bytes=released_payload_bytes,
        cancelled_load_count=cancelled_load_count,
        packages=packages,
    )
    if not report.complete:
        raise ConversationGateError("runtime reported incomplete shutdown")
    if report.stream_count != 1 or report.package_count != 1:
        raise ConversationGateError(
            "conversation gate shutdown must cover exactly one stream and one package"
        )
    if report.scheduler_in_flight_activation_count != 0:
        raise ConversationGateError("runtime shutdown retained scheduler activations")
    if report.acknowledged_device_count != report.physical_device_count:
        raise ConversationGateError(
            "runtime shutdown was not acknowledged by every physical resource device"
        )
    if len(report.packages) != report.package_count:
        raise ConversationGateError(
            "runtime shutdown package report count does not match its summary"
        )
    if sum(package.physical_device_count for package in report.packages) != (
        report.physical_device_count
    ) or sum(package.acknowledged_device_count for package in report.packages) != (
        report.acknowledged_device_count
    ):
        raise ConversationGateError(
            "runtime shutdown package device totals do not match its summary"
        )
    package_identities = {
        (package.package_id, package.execution_scope) for package in report.packages
    }
    if len(package_identities) != len(report.packages):
        raise ConversationGateError("runtime shutdown repeats a package identity")
    for package in report.packages:
        if len(package.devices) != package.physical_device_count:
            raise ConversationGateError(
                "runtime shutdown package device count does not match its summary"
            )
        if sum(device.acknowledged for device in package.devices) != (
            package.acknowledged_device_count
        ):
            raise ConversationGateError(
                "runtime shutdown package acknowledgement count does not match its devices"
            )
        store_ids = {device.store_id for device in package.devices}
        if len(store_ids) != len(package.devices):
            raise ConversationGateError(
                "runtime shutdown repeats a physical resource store"
            )
        for device in package.devices:
            if (
                not device.acknowledged
                or device.remaining_units != 0
                or device.remaining_payload_bytes != 0
                or device.error is not None
            ):
                raise ConversationGateError(
                    "runtime shutdown left a physical resource store incomplete"
                )
    return report


def _required_nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversationGateError(
            f"runtime device restoration field {path} must be a non-negative integer"
        )
    return value


def _required_exact_object(value: Any, keys: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ConversationGateError(
            f"runtime device restoration field {path} has an invalid schema"
        )
    return value


def _validated_device_restoration_snapshot(
    value: Any,
    path: str,
) -> dict[str, Any]:
    snapshot = _required_exact_object(
        value,
        {
            "physical_device_id",
            "device_name",
            "pci_address",
            "api_version",
            "driver_version",
            "memory_budget",
            "memory_accounting",
            "memory_pressure",
        },
        path,
    )
    for field in ("physical_device_id", "device_name"):
        if not isinstance(snapshot[field], str) or not snapshot[field]:
            raise ConversationGateError(
                f"runtime device restoration field {path}.{field} must be a non-empty string"
            )
    if snapshot["pci_address"] is not None and (
        not isinstance(snapshot["pci_address"], str) or not snapshot["pci_address"]
    ):
        raise ConversationGateError(
            f"runtime device restoration field {path}.pci_address is invalid"
        )
    for field in ("api_version", "driver_version"):
        _required_nonnegative_integer(snapshot[field], f"{path}.{field}")

    budget = _required_exact_object(
        snapshot["memory_budget"],
        {
            "baseline_available_bytes",
            "reservable_bytes",
            "protected_headroom_bytes",
            "counter_tolerance_bytes",
        },
        f"{path}.memory_budget",
    )
    accounting = _required_exact_object(
        snapshot["memory_accounting"],
        {
            "baseline_available_bytes",
            "currently_available_bytes",
            "reservable_bytes",
            "tracked_allocation_bytes",
            "pending_reservation_bytes",
            "untracked_acquired_bytes",
            "remaining_bytes",
            "admissible_remaining_bytes",
        },
        f"{path}.memory_accounting",
    )
    pressure = _required_exact_object(
        snapshot["memory_pressure"],
        {
            "active",
            "episode",
            "observed_available_bytes",
            "current_deficit_bytes",
            "peak_deficit_bytes",
        },
        f"{path}.memory_pressure",
    )
    for field, field_value in budget.items():
        _required_nonnegative_integer(field_value, f"{path}.memory_budget.{field}")
    for field, field_value in accounting.items():
        _required_nonnegative_integer(field_value, f"{path}.memory_accounting.{field}")
    if not isinstance(pressure["active"], bool):
        raise ConversationGateError(
            f"runtime device restoration field {path}.memory_pressure.active must be boolean"
        )
    for field in (
        "episode",
        "observed_available_bytes",
        "current_deficit_bytes",
        "peak_deficit_bytes",
    ):
        _required_nonnegative_integer(
            pressure[field], f"{path}.memory_pressure.{field}"
        )
    if accounting["baseline_available_bytes"] != budget["baseline_available_bytes"]:
        raise ConversationGateError(
            f"runtime device restoration field {path} has inconsistent baseline accounting"
        )
    if accounting["reservable_bytes"] != budget["reservable_bytes"]:
        raise ConversationGateError(
            f"runtime device restoration field {path} has inconsistent reservable accounting"
        )
    return snapshot


def _validate_device_restoration_pair(
    physical_device_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    if before["physical_device_id"] != physical_device_id or (
        after["physical_device_id"] != physical_device_id
    ):
        raise ConversationGateError(
            "runtime device restoration snapshot identity does not match its device report"
        )
    identity_fields = (
        "physical_device_id",
        "device_name",
        "pci_address",
        "api_version",
        "driver_version",
    )
    if any(before[field] != after[field] for field in identity_fields):
        raise ConversationGateError(
            f"runtime device restoration changed physical identity for {physical_device_id!r}"
        )
    if before["memory_budget"] != after["memory_budget"]:
        raise ConversationGateError(
            f"runtime device restoration changed the memory budget for {physical_device_id!r}"
        )
    tolerance = before["memory_budget"]["counter_tolerance_bytes"]
    accounting_tolerances = {
        "baseline_available_bytes": 0,
        "reservable_bytes": 0,
        "tracked_allocation_bytes": 0,
        "pending_reservation_bytes": 0,
        "untracked_acquired_bytes": tolerance,
        "currently_available_bytes": tolerance,
        "remaining_bytes": tolerance,
        "admissible_remaining_bytes": tolerance,
    }
    for field, allowed_difference in accounting_tolerances.items():
        before_value = before["memory_accounting"][field]
        after_value = after["memory_accounting"][field]
        if abs(before_value - after_value) > allowed_difference:
            raise ConversationGateError(
                f"runtime device restoration did not restore {field} for "
                f"{physical_device_id!r}"
            )
    for field in ("active", "episode"):
        if before["memory_pressure"][field] != after["memory_pressure"][field]:
            raise ConversationGateError(
                f"runtime device restoration changed memory pressure for "
                f"{physical_device_id!r}"
            )


def parse_device_restoration_report(transcript: str) -> DeviceRestorationReport:
    markers = list(_DEVICE_RESTORATION_MARKER.finditer(transcript))
    if not markers:
        raise ConversationGateError("runtime did not report device restoration")
    if len(markers) != 1:
        raise ConversationGateError(
            "runtime reported device restoration more than once"
        )
    lines = transcript[markers[0].end() :].splitlines()
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) != 1 or not lines[0].startswith("  "):
        raise ConversationGateError("runtime device restoration report is malformed")
    try:
        payload = json.loads(lines[0][2:])
    except json.JSONDecodeError as error:
        raise ConversationGateError(
            "runtime device restoration report is not valid JSON"
        ) from error
    payload = _required_exact_object(
        payload,
        {
            "schema",
            "complete",
            "physical_device_count",
            "restored_device_count",
            "devices",
            "errors",
        },
        "report",
    )
    if payload["schema"] != _DEVICE_RESTORATION_SCHEMA:
        raise ConversationGateError(
            "runtime device restoration report has an unsupported schema"
        )
    if payload["complete"] is not True:
        raise ConversationGateError("runtime reported incomplete device restoration")
    physical_device_count = _required_nonnegative_integer(
        payload["physical_device_count"], "report.physical_device_count"
    )
    restored_device_count = _required_nonnegative_integer(
        payload["restored_device_count"], "report.restored_device_count"
    )
    if physical_device_count == 0 or restored_device_count != physical_device_count:
        raise ConversationGateError(
            "runtime device restoration did not restore every selected physical device"
        )
    if payload["errors"] != []:
        raise ConversationGateError("runtime device restoration reported global errors")
    if not isinstance(payload["devices"], list) or len(payload["devices"]) != (
        physical_device_count
    ):
        raise ConversationGateError(
            "runtime device restoration device count does not match its summary"
        )

    devices = []
    for index, raw_device in enumerate(payload["devices"]):
        device = _required_exact_object(
            raw_device,
            {"physical_device_id", "restored", "before", "after", "errors"},
            f"report.devices[{index}]",
        )
        physical_device_id = device["physical_device_id"]
        if not isinstance(physical_device_id, str) or not physical_device_id:
            raise ConversationGateError(
                "runtime device restoration has an invalid physical device identity"
            )
        if device["restored"] is not True or device["errors"] != []:
            raise ConversationGateError(
                f"runtime device restoration left {physical_device_id!r} incomplete"
            )
        before = _validated_device_restoration_snapshot(
            device["before"], f"report.devices[{index}].before"
        )
        after = _validated_device_restoration_snapshot(
            device["after"], f"report.devices[{index}].after"
        )
        _validate_device_restoration_pair(physical_device_id, before, after)
        devices.append(
            DeviceRestorationDeviceReport(
                physical_device_id=physical_device_id,
                before=before,
                after=after,
            )
        )
    if len({device.physical_device_id for device in devices}) != len(devices):
        raise ConversationGateError(
            "runtime device restoration repeats a physical device identity"
        )
    return DeviceRestorationReport(
        schema=payload["schema"],
        complete=True,
        physical_device_count=physical_device_count,
        restored_device_count=restored_device_count,
        devices=devices,
    )


def _validate_shutdown_device_restoration(
    shutdown: ShutdownReport,
    device_restoration: DeviceRestorationReport,
) -> None:
    shutdown_physical_device_ids = {
        device.physical_device_id
        for package_report in shutdown.packages
        for device in package_report.devices
    }
    restoration_physical_device_ids = {
        device.physical_device_id for device in device_restoration.devices
    }
    if not shutdown_physical_device_ids.issubset(restoration_physical_device_ids):
        raise ConversationGateError(
            "runtime device restoration omits a physical resource device"
        )


def parse_conversation_transcript(
    transcript: str,
    *,
    warmup_conversation_sets: int = 0,
) -> list[ConversationTurn]:
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    turns = _parse_all_conversation_turns(transcript)
    expected_turn_count = len(CANONICAL_CONVERSATION_PROMPTS) * (
        warmup_conversation_sets + 1
    )
    if len(turns) != expected_turn_count:
        raise ConversationGateError(
            "chat transcript contains "
            f"{len(turns)} completed turn(s); expected {expected_turn_count}"
        )
    return turns


def _parse_all_conversation_turns(transcript: str) -> list[ConversationTurn]:
    completed_sections = []
    for section in transcript.split(_TURN_START)[1:]:
        if _STATS_MARKER not in section:
            continue
        response, report = section.rsplit(_STATS_MARKER, 1)
        stats: dict[str, int | float | str] = {}
        for line in report.splitlines():
            match = _STAT_LINE.match(line)
            if match is None:
                if stats:
                    break
                continue
            stats[match.group(1)] = _parse_scalar(match.group(2))
        policy, counters, gauges = _residency_metrics(report)
        execution_counters = _execution_metrics(report)
        determinism_digests = _determinism_metrics(report)
        completed_sections.append(
            (
                response.rstrip(),
                stats,
                policy,
                counters,
                gauges,
                execution_counters,
                determinism_digests,
            )
        )

    if not completed_sections or (
        len(completed_sections) % len(CANONICAL_CONVERSATION_PROMPTS)
    ):
        raise ConversationGateError(
            "chat transcript contains "
            f"{len(completed_sections)} completed turn(s); expected a non-zero multiple "
            f"of {len(CANONICAL_CONVERSATION_PROMPTS)}"
        )
    expected_prompts = CANONICAL_CONVERSATION_PROMPTS * (
        len(completed_sections) // len(CANONICAL_CONVERSATION_PROMPTS)
    )
    return [
        ConversationTurn(
            prompt=prompt,
            response=response,
            stats=stats,
            residency_policy=policy,
            residency_counters=counters,
            residency_gauges=gauges,
            execution_counters=execution_counters,
            determinism_digests=determinism_digests,
        )
        for prompt, (
            response,
            stats,
            policy,
            counters,
            gauges,
            execution_counters,
            determinism_digests,
        ) in zip(expected_prompts, completed_sections, strict=True)
    ]


def _conversation_sets_from_transcript(
    transcript: str,
) -> list[list[ConversationTurn]]:
    turns = _parse_all_conversation_turns(transcript)
    width = len(CANONICAL_CONVERSATION_PROMPTS)
    return [turns[index : index + width] for index in range(0, len(turns), width)]


def _required_execution_counter(turn: ConversationTurn, key: str) -> int:
    counters = turn.execution_counters
    if counters is None or key not in counters:
        raise ConversationGateError(
            f"turn {turn.prompt!r} did not report execution counter {key!r}"
        )
    value = counters[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversationGateError(
            f"turn {turn.prompt!r} reported invalid execution counter {key!r}: {value!r}"
        )
    return value


def _validate_tensor_parallel_turn_execution(turn: ConversationTurn) -> None:
    for phase in ("decode", "prefill"):
        prefix = f"distributed_{phase}_"
        island_submissions = _required_execution_counter(
            turn, f"{prefix}island_submissions"
        )
        shard_submissions = _required_execution_counter(
            turn, f"{prefix}shard_submissions"
        )
        tensor_parallel = _required_execution_counter(
            turn, f"{prefix}tensor_parallel_island_submissions"
        )
        whole_expert = _required_execution_counter(
            turn, f"{prefix}whole_expert_parallel_island_submissions"
        )
        intra_expert = _required_execution_counter(
            turn, f"{prefix}intra_expert_tensor_parallel_island_submissions"
        )
        hybrid = _required_execution_counter(turn, f"{prefix}hybrid_island_submissions")
        classified_islands = tensor_parallel + whole_expert + intra_expert + hybrid
        if island_submissions != classified_islands:
            raise ConversationGateError(
                f"turn {turn.prompt!r} reported {island_submissions} distributed {phase} "
                f"island submission(s), but classified {classified_islands}"
            )
        if shard_submissions < island_submissions * 2:
            raise ConversationGateError(
                f"turn {turn.prompt!r} reported {shard_submissions} distributed {phase} "
                f"shard submission(s) for {island_submissions} island(s)"
            )
        if tensor_parallel + intra_expert + hybrid == 0:
            raise ConversationGateError(
                f"turn {turn.prompt!r} did not submit a tensor-parallel {phase} island"
            )


def _final_answer(response: str, require_thinking: bool) -> str:
    closing_count = response.count("</think>")
    opening_count = response.count("<think>")
    if require_thinking:
        if closing_count == 1:
            if opening_count > 1:
                raise ConversationGateError(
                    "thinking response contains more than one <think> boundary"
                )
            answer = response.rsplit("</think>", 1)[1].strip()
        elif (
            closing_count == 0
            and opening_count == 0
            and response.startswith(("thought\n", "analysis\n"))
        ):
            # Channel-based templates can decode the channel label while their
            # special delimiters are intentionally omitted by the tokenizer.
            # The model may not emit a second visible delimiter before its
            # answer, so retain the complete, validated channel stream.
            answer = response.split("\n", 1)[1].strip()
        else:
            raise ConversationGateError(
                "thinking response must contain one </think> boundary or begin "
                "with a decoded thought/analysis channel"
            )
    else:
        if closing_count > 1 or opening_count > 1:
            raise ConversationGateError(
                "response contains malformed thinking boundaries"
            )
        answer = response.rsplit("</think>", 1)[-1].strip()
    if not answer:
        raise ConversationGateError("response terminated without a final answer")
    return answer


def validate_conversation_turns(
    turns: Sequence[ConversationTurn],
    *,
    require_thinking: bool,
    minimum_decode_tokens_per_second: float,
    require_tensor_parallel_execution: bool = False,
) -> tuple[float, float]:
    if (
        isinstance(minimum_decode_tokens_per_second, bool)
        or not isinstance(minimum_decode_tokens_per_second, (int, float))
        or not math.isfinite(minimum_decode_tokens_per_second)
        or minimum_decode_tokens_per_second < 0
    ):
        raise ConversationGateError(
            "minimum decode throughput must be a finite non-negative number"
        )
    if len(turns) != len(MEASURED_PROMPTS):
        raise ConversationGateError(
            f"expected {len(MEASURED_PROMPTS)} measured turns; found {len(turns)}"
        )

    answers = []
    for expected_prompt, turn in zip(MEASURED_PROMPTS, turns, strict=True):
        if turn.prompt != expected_prompt:
            raise ConversationGateError(
                f"expected prompt {expected_prompt!r}; found {turn.prompt!r}"
            )
        answer = _final_answer(turn.response, require_thinking)
        repeated = repeated_segment(turn.response)
        if repeated is not None:
            raise ConversationGateError(
                f"turn {turn.prompt!r} ends in a repeated segment: {repeated!r}"
            )
        if require_tensor_parallel_execution:
            _validate_tensor_parallel_turn_execution(turn)
        answers.append(answer)

    if "athens" not in answers[1].casefold():
        raise ConversationGateError("capital-of-Greece turn did not answer Athens")
    if "corinth" not in answers[2].casefold():
        raise ConversationGateError(
            "Corinth turn did not answer the question about Corinth"
        )
    if "greece" not in answers[4].casefold():
        raise ConversationGateError(
            "conversation-recall turn did not identify Greece from prior history"
        )

    decode_rates = [turn.decode_tokens_per_second for turn in turns]
    prefill_rates = [turn.prefill_tokens_per_second for turn in turns]
    mean_decode = fmean(decode_rates)
    mean_prefill = fmean(prefill_rates)
    if mean_decode < minimum_decode_tokens_per_second:
        raise ConversationGateError(
            f"mean decode throughput {mean_decode:.3f} tok/s is below "
            f"the {minimum_decode_tokens_per_second:.3f} tok/s gate"
        )
    return mean_decode, mean_prefill


def _option_value(command: Sequence[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if not positions:
        return None
    index = positions[-1]
    if index + 1 >= len(command):
        raise ConversationGateError(f"{option} is missing its value")
    return command[index + 1]


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    replaced: list[str] = []
    cursor = 0
    while cursor < len(command):
        if command[cursor] == option:
            if cursor + 1 >= len(command):
                raise ConversationGateError(f"{option} is missing its value")
            cursor += 2
            continue
        replaced.append(command[cursor])
        cursor += 1
    replaced.extend((option, value))
    return replaced


def canonical_runtime_command(command: Sequence[str], seed: int) -> list[str]:
    if not command:
        raise ConversationGateError("runtime command must not be empty")
    if "--chat" not in command:
        raise ConversationGateError(
            "conversation gate requires the normal --chat runtime mode"
        )
    if "--prompt" in command:
        raise ConversationGateError(
            "conversation gate owns the canonical warmup and measured prompts"
        )
    if "--json" in command or "--generated-only" in command:
        raise ConversationGateError(
            "conversation gate requires the normal default chat output and statistics"
        )
    raw_limit = _option_value(command, "--max-new-tokens")
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise ConversationGateError(
                f"invalid --max-new-tokens value {raw_limit!r}"
            ) from error
        if limit != CANONICAL_OUTPUT_TOKEN_ALLOWANCE:
            raise ConversationGateError(
                "conversation gate requires --max-new-tokens "
                f"{CANONICAL_OUTPUT_TOKEN_ALLOWANCE}; found {limit}"
            )
    else:
        command = (*command, "--max-new-tokens", str(CANONICAL_OUTPUT_TOKEN_ALLOWANCE))
    return _replace_option(command, "--seed", str(seed))


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _fully_warm_residency_delta(
    delta: dict[str, int | float] | None,
    *,
    label: str,
) -> bool:
    if delta is None:
        raise ConversationGateError(
            f"{label} did not report cumulative resource residency counters"
        )
    missing = [key for key in _FULLY_WARM_ZERO_COUNTERS if key not in delta]
    if missing:
        raise ConversationGateError(
            f"{label} cannot prove full residency; missing counter(s): "
            + ", ".join(missing)
        )
    return all(delta[key] == 0 for key in _FULLY_WARM_ZERO_COUNTERS)


def _conversation_set_warmth(
    turns: Sequence[ConversationTurn],
    prior_residency: dict[str, int | float] | None,
    *,
    set_index: int,
) -> tuple[bool, dict[str, int | float]]:
    if len(turns) != len(CANONICAL_CONVERSATION_PROMPTS):
        raise ConversationGateError(
            f"conversation set {set_index} contains {len(turns)} turn(s); "
            f"expected {len(CANONICAL_CONVERSATION_PROMPTS)}"
        )
    residency_end = turns[-1].residency_counters
    if residency_end is None:
        raise ConversationGateError(
            f"conversation set {set_index} did not report cumulative resource "
            "residency counters"
        )
    delta = _residency_delta(prior_residency, residency_end)
    return (
        _fully_warm_residency_delta(
            delta,
            label=f"conversation set {set_index}",
        ),
        residency_end,
    )


def _validate_repeated_conversation_equivalence(
    reference: ConversationSetReport,
    measured: ConversationSetReport,
) -> None:
    reference_turns = [reference.warmup, *reference.turns]
    measured_turns = [measured.warmup, *measured.turns]
    if len(reference_turns) != len(measured_turns):
        raise ConversationGateError(
            "fully warm conversation sets have different turn counts"
        )
    for turn_index, (reference_turn, measured_turn) in enumerate(
        zip(reference_turns, measured_turns, strict=True),
        start=1,
    ):
        if reference_turn.prompt != measured_turn.prompt:
            raise ConversationGateError(
                f"fully warm conversation turn {turn_index} changed its prompt"
            )
        reference_digests = reference_turn.determinism_digests
        measured_digests = measured_turn.determinism_digests
        if reference_digests is None or measured_digests is None:
            raise ConversationGateError(
                f"fully warm conversation turn {turn_index} lacks determinism evidence"
            )
        for field in _DETERMINISM_FIELDS:
            if reference_digests[field] != measured_digests[field]:
                raise ConversationGateError(
                    f"fully warm conversation turn {turn_index} changed its "
                    f"{field.replace('_', ' ')} digest"
                )
        if reference_turn.response != measured_turn.response:
            raise ConversationGateError(
                f"fully warm conversation turn {turn_index} changed its decoded response"
            )


def run_resident_conversation(
    command: Sequence[str],
    *,
    warmup_conversation_sets: int = 0,
    warm_until_fully_resident: bool = False,
    maximum_conversation_sets: int = _MAXIMUM_RESIDENT_CONVERSATION_SETS,
) -> tuple[str, int]:
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    if maximum_conversation_sets < 1:
        raise ConversationGateError("maximum conversation set count must be positive")
    minimum_set_count = warmup_conversation_sets + 1
    if warm_until_fully_resident:
        minimum_set_count = max(minimum_set_count, 2)
    if maximum_conversation_sets < minimum_set_count:
        raise ConversationGateError(
            "maximum conversation set count cannot satisfy the requested warmup"
        )
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:
        _terminate(process)
        raise ConversationGateError("could not open runtime process pipes")

    prompts: list[str] = list(CANONICAL_CONVERSATION_PROMPTS)
    if not warm_until_fully_resident:
        for _set_index in range(warmup_conversation_sets):
            prompts.extend((_NEW_CONVERSATION_COMMAND, *CANONICAL_CONVERSATION_PROMPTS))
        prompts.append("/exit")
    transcript = bytearray()
    search_from = 0
    accepted_marker_end = 0
    sent = 0
    checked_response_bytes = 0
    live_error: str | None = None
    completed_set_count = 0
    consecutive_fully_warm_sets = 0
    prior_residency: dict[str, int | float] | None = None
    try:
        while True:
            chunk = os.read(process.stdout.fileno(), 65_536)
            if not chunk:
                break
            transcript.extend(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

            turn_error = transcript.find(_TURN_ERROR_MARKER, accepted_marker_end)
            if turn_error >= 0:
                detail_start = turn_error + len(_TURN_ERROR_MARKER)
                detail_end = transcript.find(b"\n", detail_start)
                if detail_end >= 0:
                    detail = transcript[detail_start:detail_end].decode(
                        errors="replace"
                    )
                    live_error = f"runtime rejected a recoverable chat turn: {detail}"
                    break

            while True:
                marker = transcript.find(_PROMPT_MARKER, search_from)
                if marker < 0:
                    search_from = max(0, len(transcript) - len(_PROMPT_MARKER) + 1)
                    break
                search_from = marker + len(_PROMPT_MARKER)
                if sent > 0:
                    previous_output = transcript[accepted_marker_end:marker]
                    if prompts[sent - 1] == _NEW_CONVERSATION_COMMAND:
                        if _SESSION_RESET_MARKER not in previous_output:
                            live_error = (
                                "runtime did not acknowledge the required "
                                "new-conversation reset"
                            )
                            break
                    elif _STATS_MARKER.encode() not in previous_output:
                        continue
                if warm_until_fully_resident and sent == len(prompts):
                    try:
                        conversation_sets = _conversation_sets_from_transcript(
                            transcript.decode(errors="replace")
                        )
                        if len(conversation_sets) != completed_set_count + 1:
                            raise ConversationGateError(
                                "resident runtime completed an unexpected number of "
                                "conversation sets"
                            )
                        completed_set_count += 1
                        fully_warm, prior_residency = _conversation_set_warmth(
                            conversation_sets[-1],
                            prior_residency,
                            set_index=completed_set_count,
                        )
                    except ConversationGateError as error:
                        live_error = str(error)
                        break
                    consecutive_fully_warm_sets = (
                        consecutive_fully_warm_sets + 1 if fully_warm else 0
                    )
                    if (
                        completed_set_count >= minimum_set_count
                        and consecutive_fully_warm_sets >= 2
                    ):
                        prompts.append("/exit")
                    elif completed_set_count >= maximum_conversation_sets:
                        live_error = (
                            "runtime did not produce two consecutive fully warm "
                            f"conversation sets within {maximum_conversation_sets} sets"
                        )
                        break
                    else:
                        prompts.extend(
                            (_NEW_CONVERSATION_COMMAND, *CANONICAL_CONVERSATION_PROMPTS)
                        )
                if sent >= len(prompts):
                    live_error = (
                        "runtime requested more chat turns than the gate supplied"
                    )
                    break
                process.stdin.write((prompts[sent] + "\n").encode())
                process.stdin.flush()
                accepted_marker_end = marker + len(_PROMPT_MARKER)
                sent += 1
            if live_error is not None:
                break

            latest_prompt = transcript.rfind(_PROMPT_MARKER)
            response_start = transcript.find(
                _RESPONSE_PREFIX.encode(),
                latest_prompt + len(_PROMPT_MARKER),
            )
            stats_start = transcript.find(
                _STATS_MARKER.encode(),
                response_start + len(_RESPONSE_PREFIX) if response_start >= 0 else 0,
            )
            if (
                response_start >= 0
                and stats_start < 0
                and len(transcript) - checked_response_bytes >= 16_384
            ):
                response = transcript[response_start + len(_RESPONSE_PREFIX) :].decode(
                    errors="replace"
                )
                repeated = repeated_segment(response)
                checked_response_bytes = len(transcript)
                if repeated is not None:
                    live_error = (
                        "runtime response entered a repeated segment before termination: "
                        f"{repeated!r}"
                    )
                    break
        if live_error is not None:
            _terminate(process)
        return_code = process.wait()
    except BaseException:
        _terminate(process)
        raise
    finally:
        process.stdin.close()
        process.stdout.close()

    decoded_transcript = transcript.decode(errors="replace")
    if live_error is not None:
        raise ResidentConversationError(live_error, decoded_transcript)
    if return_code != 0:
        output_tail = bytes(transcript[-4_096:]).decode(errors="replace").strip()
        detail = f"; output tail:\n{output_tail}" if output_tail else ""
        raise ResidentConversationError(
            f"runtime exited with status {return_code}{detail}",
            decoded_transcript,
        )
    if sent != len(prompts):
        raise ResidentConversationError(
            f"runtime accepted {sent} scripted input(s); expected {len(prompts)}",
            decoded_transcript,
        )
    return decoded_transcript, return_code


def _package_metadata(command: Sequence[str]) -> dict[str, Any]:
    raw_path = _option_value(command, "--package")
    if raw_path is None:
        raise ConversationGateError("runtime command is missing --package")
    path = Path(raw_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConversationGateError(
            f"could not read compiled package manifest {path}: {error}"
        ) from error
    shader_paths = {
        stage["shader_path"]
        for component in manifest.get("component_executions", [])
        for kernel in component.get("kernels", [])
        for implementation in (
            [{"stages": [{"shader_path": kernel.get("shader_path")}]}]
            + kernel.get("batch_implementations", [])
        )
        for stage in implementation.get("stages", [])
        if stage.get("shader_path")
    }
    return {
        "manifest": str(path),
        "package_id": manifest.get("package_id"),
        "package_schema": manifest.get("schema"),
        "compiler_target": manifest.get("compiler_target"),
        "compiled_shader_variant_count": len(shader_paths),
    }


def run_conversation_gate(
    command: Sequence[str],
    *,
    seeds: Sequence[int],
    minimum_decode_tokens_per_second: float,
    require_thinking: bool,
    minimum_tensor_parallel_islands: int = 0,
    warmup_conversation_sets: int = 0,
    maximum_conversation_sets: int = _MAXIMUM_RESIDENT_CONVERSATION_SETS,
    transcript_dir: Path | None = None,
) -> ConversationGateReport:
    if not seeds:
        raise ConversationGateError("at least one fixed sampler seed is required")
    if len(seeds) != 1:
        raise ConversationGateError(
            "run exactly one seed per invocation so GPU residency can be verified "
            "between model loads"
        )
    if warmup_conversation_sets < 0:
        raise ConversationGateError("warmup conversation set count cannot be negative")
    if maximum_conversation_sets < 1:
        raise ConversationGateError("maximum conversation set count must be positive")
    if maximum_conversation_sets < warmup_conversation_sets + 1:
        raise ConversationGateError(
            "maximum conversation set count cannot satisfy the requested warmup"
        )
    if minimum_tensor_parallel_islands < 0:
        raise ConversationGateError(
            "minimum tensor-parallel island count cannot be negative"
        )
    package = _package_metadata(command)
    runs = []
    for seed in seeds:
        seeded_command = canonical_runtime_command(command, seed)
        try:
            transcript, _ = run_resident_conversation(
                seeded_command,
                warmup_conversation_sets=warmup_conversation_sets,
                warm_until_fully_resident=warmup_conversation_sets > 0,
                maximum_conversation_sets=maximum_conversation_sets,
            )
        except ResidentConversationError as error:
            if transcript_dir is not None:
                transcript_dir.mkdir(parents=True, exist_ok=True)
                (transcript_dir / f"conversation-seed-{seed}-failed.log").write_text(
                    error.transcript
                )
            raise
        if transcript_dir is not None:
            transcript_dir.mkdir(parents=True, exist_ok=True)
            (transcript_dir / f"conversation-seed-{seed}.log").write_text(transcript)
        parsed = _parse_all_conversation_turns(transcript)
        physical_execution = parse_physical_execution_summary(
            transcript,
            minimum_tensor_parallel_islands=minimum_tensor_parallel_islands,
        )
        conversation_sets = [
            parsed[index : index + len(CANONICAL_CONVERSATION_PROMPTS)]
            for index in range(0, len(parsed), len(CANONICAL_CONVERSATION_PROMPTS))
        ]
        if warmup_conversation_sets > 0 and len(conversation_sets) < 2:
            raise ConversationGateError(
                "fully warm measurement requires at least two resident conversation sets"
            )
        reports: list[ConversationSetReport] = []
        prior_residency: dict[str, int | float] | None = None
        prior_residency_gauges: dict[str, int | float] | None = None
        for set_index, conversation_set in enumerate(conversation_sets):
            warmup, turns = conversation_set[0], conversation_set[1:]
            _final_answer(warmup.response, require_thinking)
            if repeated_segment(warmup.response) is not None:
                raise ConversationGateError(
                    f"seed {seed} conversation set {set_index + 1} "
                    "warmup ended in repetition"
                )
            if minimum_tensor_parallel_islands > 0:
                _validate_tensor_parallel_turn_execution(warmup)
            set_minimum = (
                minimum_decode_tokens_per_second
                if set_index == len(conversation_sets) - 1
                else 0.0
            )
            mean_decode, mean_prefill = validate_conversation_turns(
                turns,
                require_thinking=require_thinking,
                minimum_decode_tokens_per_second=set_minimum,
                require_tensor_parallel_execution=minimum_tensor_parallel_islands > 0,
            )
            residency_end = conversation_set[-1].residency_counters
            residency_gauges_end = conversation_set[-1].residency_gauges
            residency_delta = _residency_delta(prior_residency, residency_end)
            residency_policies = {
                turn.residency_policy
                for turn in conversation_set
                if turn.residency_policy is not None
            }
            if len(residency_policies) > 1:
                raise ConversationGateError(
                    "resource residency policy changed within one conversation set"
                )
            reports.append(
                ConversationSetReport(
                    warmup=warmup,
                    turns=list(turns),
                    mean_decode_tokens_per_second=mean_decode,
                    mean_prefill_tokens_per_second=mean_prefill,
                    residency_policy=(
                        next(iter(residency_policies)) if residency_policies else None
                    ),
                    residency_start=prior_residency,
                    residency_end=residency_end,
                    residency_delta=residency_delta,
                    residency_gauges_start=prior_residency_gauges,
                    residency_gauges_end=residency_gauges_end,
                )
            )
            prior_residency = residency_end
            prior_residency_gauges = residency_gauges_end
        if warmup_conversation_sets > 0:
            if len(reports) - 1 < warmup_conversation_sets:
                raise ConversationGateError(
                    "runtime completed fewer discarded conversation sets than requested"
                )
            for report_index, report in enumerate(reports[-2:], start=len(reports) - 1):
                if not _fully_warm_residency_delta(
                    report.residency_delta,
                    label=f"conversation set {report_index}",
                ):
                    raise ConversationGateError(
                        "measured conversation was not preceded by two consecutive "
                        "fully warm conversation sets"
                    )
            _validate_repeated_conversation_equivalence(reports[-2], reports[-1])
        session_policies = {
            report.residency_policy
            for report in reports
            if report.residency_policy is not None
        }
        if len(session_policies) > 1:
            raise ConversationGateError(
                "resource residency policy changed within one resident session"
            )
        if not session_policies:
            raise ConversationGateError(
                "runtime did not report resource residency for shutdown reconciliation"
            )
        shutdown = parse_shutdown_report(transcript)
        device_restoration = parse_device_restoration_report(transcript)
        if shutdown.packages[0].package_id != package["package_id"]:
            raise ConversationGateError(
                "runtime shutdown package identity does not match the compiled package"
            )
        final_gauges = reports[-1].residency_gauges_end
        required_gauges = ("payload_bytes.current", "units.current")
        missing_gauges = [
            key
            for key in required_gauges
            if final_gauges is None or key not in final_gauges
        ]
        if missing_gauges:
            raise ConversationGateError(
                "runtime cannot reconcile shutdown with final residency; missing "
                + ", ".join(missing_gauges)
            )
        assert final_gauges is not None
        if shutdown.released_payload_bytes != final_gauges["payload_bytes.current"]:
            raise ConversationGateError(
                "runtime shutdown released payload bytes do not match final residency"
            )
        if shutdown.released_unit_count != final_gauges["units.current"]:
            raise ConversationGateError(
                "runtime shutdown released units do not match final residency"
            )
        _validate_shutdown_device_restoration(shutdown, device_restoration)
        runs.append(
            ConversationSeedReport(
                seed=seed,
                command=seeded_command,
                transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                physical_execution=physical_execution,
                shutdown=shutdown,
                device_restoration=device_restoration,
                discarded_warmup_sets=reports[:-1],
                measured_set=reports[-1],
            )
        )
    return ConversationGateReport(
        ok=True,
        minimum_decode_tokens_per_second=minimum_decode_tokens_per_second,
        minimum_tensor_parallel_islands=minimum_tensor_parallel_islands,
        require_thinking=require_thinking,
        warmup_conversation_sets=(
            len(runs[0].discarded_warmup_sets) if runs else warmup_conversation_sets
        ),
        maximum_conversation_sets=maximum_conversation_sets,
        package=package,
        runs=runs,
    )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or any(seed < 0 or seed > 0xFFFF_FFFF for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must contain one or more U32 values")
    return seeds


def _residency_delta(
    start: dict[str, int | float] | None,
    end: dict[str, int | float] | None,
) -> dict[str, int | float] | None:
    if end is None:
        return None
    if start is None:
        return dict(end)
    if start.keys() != end.keys():
        raise ConversationGateError(
            "resource residency counter schema changed within one resident session"
        )
    delta: dict[str, int | float] = {}
    for key, end_value in end.items():
        start_value = start[key]
        if end_value < start_value:
            raise ConversationGateError(
                f"cumulative resource residency counter {key!r} decreased from "
                f"{start_value} to {end_value}"
            )
        delta[key] = end_value - start_value
    return delta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the canonical warmup plus five-turn correctness/performance gate "
            "through nerve-runtime's normal resident chat mode."
        )
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(0,))
    parser.add_argument("--minimum-decode-tps", type=float, default=0.0)
    parser.add_argument("--minimum-tensor-parallel-islands", type=int, default=0)
    parser.add_argument("--require-thinking", action="store_true")
    parser.add_argument(
        "--warmup-conversation-sets",
        type=int,
        default=0,
        help=(
            "discard at least this many complete canonical conversations and "
            "continue in the same process until two consecutive sets are fully "
            "resident before measuring the latter"
        ),
    )
    parser.add_argument(
        "--maximum-conversation-sets",
        type=int,
        default=_MAXIMUM_RESIDENT_CONVERSATION_SETS,
        help=(
            "fail if adaptive full-residency warming cannot produce two "
            "consecutive zero-load conversations within this many sets"
        ),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--transcript-dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = run_conversation_gate(
            command,
            seeds=args.seeds,
            minimum_decode_tokens_per_second=args.minimum_decode_tps,
            minimum_tensor_parallel_islands=args.minimum_tensor_parallel_islands,
            require_thinking=args.require_thinking,
            warmup_conversation_sets=args.warmup_conversation_sets,
            maximum_conversation_sets=args.maximum_conversation_sets,
            transcript_dir=args.transcript_dir,
        )
    except ConversationGateError as error:
        print(f"conversation gate failed: {error}", file=sys.stderr)
        return 1

    encoded = json.dumps(asdict(report), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
