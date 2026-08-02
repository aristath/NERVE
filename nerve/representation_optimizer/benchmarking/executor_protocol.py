from __future__ import annotations

from hashlib import sha256

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.protocols import (
    BenchmarkMountRequest,
)
from nerve.representation_optimizer.contracts import (
    DEVICE_STATE_DIGEST_SCHEMA,
    canonical_json_bytes,
)


EXECUTOR_COMMAND_SCHEMA = "nerve.optimizer.executor_command.v3"
EXECUTOR_RESPONSE_SCHEMA = "nerve.optimizer.executor_response.v3"
ARTIFACT_DIGEST_PREFIX = "nerve.optimizer.artifact_sha256.v1:"


def request_id(kind: str, document: Json) -> str:
    return (
        f"{kind}-"
        f"{sha256(canonical_json_bytes(document)).hexdigest()[:32]}"
    )


def validated_response(
    response: Json,
    *,
    expected_request_id: str,
    expected_status: str,
) -> Json:
    if set(response) != {"schema", "request_id", "status", "payload"}:
        raise ModelCompileError(
            "resident executor response fields are invalid"
        )
    if (
        response["schema"] != EXECUTOR_RESPONSE_SCHEMA
        or response["request_id"] != expected_request_id
        or response["status"] != expected_status
        or not isinstance(response["payload"], dict)
    ):
        raise ModelCompileError(
            "resident executor response does not match its request"
        )
    return response


def validate_mount_payload(
    payload: Json,
    *,
    request: BenchmarkMountRequest,
    component_id: str,
    physical_node_id: str,
    logical_device_id: str,
    physical_device_id: str,
    candidate_id: str | None,
) -> None:
    expected = {
        "candidate_id": candidate_id,
        "component_id": component_id,
        "physical_node_id": physical_node_id,
        "logical_device_id": logical_device_id,
        "physical_device_id": physical_device_id,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ModelCompileError(
            "resident executor mounted different runtime conditions"
        )
    for field in (
        "package_id",
        "device_name",
        "mounted_state_digest",
    ):
        required_text(payload, field)
    for field in (
        "mount_duration_ns",
        "resident_parameter_bytes",
        "resident_transient_bytes",
    ):
        nonnegative_integer(payload.get(field), f"executor mount {field}")
    required_device_state_digest(payload, "mounted_state_digest")
    is_candidate = request.implementation["implementation_id"].startswith(
        "staged-representation:"
    )
    if is_candidate != (candidate_id is not None):
        raise ModelCompileError(
            "resident executor implementation role changed at mount"
        )


def validated_windows(value: object, useful_units: int) -> list[Json]:
    if not isinstance(value, list) or not value:
        raise ModelCompileError(
            "resident executor omitted throughput windows"
        )
    windows: list[Json] = []
    expected_start = 0
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ModelCompileError(
                "resident executor throughput window is not an object"
            )
        if set(raw) != {
            "index",
            "start_unit",
            "end_unit",
            "duration_ns",
        }:
            raise ModelCompileError(
                "resident executor throughput window fields are invalid"
            )
        if (
            raw["index"] != index
            or raw["start_unit"] != expected_start
            or not isinstance(raw["end_unit"], int)
            or isinstance(raw["end_unit"], bool)
            or raw["end_unit"] <= expected_start
        ):
            raise ModelCompileError(
                "resident executor throughput windows are not contiguous"
            )
        positive_integer(
            raw["duration_ns"],
            "executor throughput window duration_ns",
        )
        expected_start = raw["end_unit"]
        windows.append(dict(raw))
    if expected_start != useful_units:
        raise ModelCompileError(
            "resident executor throughput windows do not cover useful work"
        )
    return windows


def validate_executor_shutdown_payload(
    payload: Json,
    *,
    logical_by_physical: dict[str, str],
) -> None:
    physical_device_ids = sorted(logical_by_physical)
    if (
        set(payload)
        != {
            "released",
            "physical_device_ids",
            "pre_release_quiesce_duration_ns",
            "device_releases",
            "shutdown_duration_ns",
        }
        or payload["released"] is not True
        or payload["physical_device_ids"] != physical_device_ids
    ):
        raise ModelCompileError(
            "resident executor shutdown proof is invalid"
        )
    positive_integer(
        payload["pre_release_quiesce_duration_ns"],
        "executor pre-release quiesce duration",
    )
    positive_integer(
        payload["shutdown_duration_ns"],
        "executor shutdown duration",
    )
    releases = payload["device_releases"]
    if (
        not isinstance(releases, list)
        or len(releases) != len(physical_device_ids)
    ):
        raise ModelCompileError(
            "resident executor did not release every physical device"
        )
    for release, physical_device_id in zip(
        releases,
        physical_device_ids,
        strict=True,
    ):
        if (
            not isinstance(release, dict)
            or set(release)
            != {
                "physical_device_id",
                "logical_device_id",
                "released_buffer_count",
                "released_buffer_bytes",
                "quiesced",
                "device_context_destroyed",
                "release_duration_ns",
            }
            or release["physical_device_id"] != physical_device_id
            or release["logical_device_id"]
            != logical_by_physical[physical_device_id]
            or release["quiesced"] is not True
            or release["device_context_destroyed"] is not True
        ):
            raise ModelCompileError(
                "resident executor device shutdown proof is invalid"
            )
        for field in (
            "released_buffer_count",
            "released_buffer_bytes",
        ):
            nonnegative_integer(
                release[field],
                f"executor shutdown {field}",
            )
        positive_integer(
            release["release_duration_ns"],
            "executor device release duration",
        )


def required_object(document: Json, field: str) -> Json:
    value = document.get(field)
    if not isinstance(value, dict):
        raise ModelCompileError(f"executor {field} must be an object")
    return value


def required_text(document: Json, field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ModelCompileError(f"executor {field} must be non-empty text")
    return value


def required_digest(document: Json, field: str) -> str:
    value = required_text(document, field)
    hexadecimal = value.removeprefix(ARTIFACT_DIGEST_PREFIX)
    if (
        not value.startswith(ARTIFACT_DIGEST_PREFIX)
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ModelCompileError(f"executor {field} is not an artifact digest")
    return value


def required_device_state_digest(document: Json, field: str) -> str:
    value = required_text(document, field)
    prefix = f"{DEVICE_STATE_DIGEST_SCHEMA}:"
    hexadecimal = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(hexadecimal) != 64
        or any(
            character not in "0123456789abcdef"
            for character in hexadecimal
        )
    ):
        raise ModelCompileError(
            f"executor {field} is not a device-state digest"
        )
    return value


def nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelCompileError(f"{label} must be a non-negative integer")
    return value


def positive_integer(value: object, label: str) -> int:
    result = nonnegative_integer(value, label)
    if result == 0:
        raise ModelCompileError(f"{label} must be positive")
    return result
