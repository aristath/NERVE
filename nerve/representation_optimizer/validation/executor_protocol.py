from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    positive_integer,
    required_digest,
    required_device_state_digest,
    required_object,
    required_text,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    EXECUTOR_PROGRESS_SCHEMA,
)


VALIDATION_EXECUTOR_COMMAND_SCHEMA = (
    "nerve.optimizer.validation_executor_command.v4"
)
VALIDATION_EXECUTOR_RESPONSE_SCHEMA = (
    "nerve.optimizer.validation_executor_response.v4"
)

_PROGRESS_FIELDS = {
    "generation": {
        "phase",
        "turn_index",
        "generated_tokens",
        "elapsed_ns",
    },
    "turn_completed": {
        "phase",
        "turn_index",
        "generated_tokens",
        "elapsed_ns",
        "component_activations",
        "scheduler_steps",
    },
    "teacher_forced_turn_completed": {
        "phase",
        "turn_index",
        "generated_tokens",
        "elapsed_ns",
        "component_activations",
        "scheduler_steps",
    },
    "lifecycle_completed": {
        "phase",
        "elapsed_ns",
        "component_activations",
        "scheduler_steps",
    },
}


def validate_validation_progress(
    event: Json,
    *,
    expected_request_id: str,
    turn_count: int,
) -> None:
    if (
        event.get("schema") != EXECUTOR_PROGRESS_SCHEMA
        or event.get("request_id") != expected_request_id
        or isinstance(event.get("sequence"), bool)
        or not isinstance(event.get("sequence"), int)
        or event["sequence"] < 0
        or not isinstance(event.get("payload"), dict)
    ):
        raise ModelCompileError(
            "validation executor progress does not match its request"
        )
    payload = event["payload"]
    phase = payload.get("phase")
    expected_fields = _PROGRESS_FIELDS.get(phase)
    if expected_fields is None or set(payload) != expected_fields:
        raise ModelCompileError(
            "validation executor progress payload is invalid"
        )
    for field in (
        "elapsed_ns",
        "component_activations",
        "scheduler_steps",
    ):
        if field in payload:
            positive_integer(
                payload[field],
                f"validation progress {field}",
            )
    generated_tokens = payload.get("generated_tokens")
    if generated_tokens is not None and (
        isinstance(generated_tokens, bool)
        or not isinstance(generated_tokens, int)
        or generated_tokens < 0
    ):
        raise ModelCompileError(
            "validation progress generated token count is invalid"
        )
    turn_index = payload.get("turn_index")
    if turn_index is not None and (
        isinstance(turn_index, bool)
        or not isinstance(turn_index, int)
        or turn_index < 0
        or turn_index >= turn_count
    ):
        raise ModelCompileError(
            "validation progress turn index is invalid"
        )


def validated_validation_response(
    response: Json,
    *,
    expected_request_id: str,
    expected_status: str,
) -> Json:
    if set(response) != {
        "schema",
        "request_id",
        "status",
        "payload",
    }:
        raise ModelCompileError(
            "validation executor response fields are invalid"
        )
    if (
        response["schema"] != VALIDATION_EXECUTOR_RESPONSE_SCHEMA
        or response["request_id"] != expected_request_id
        or response["status"] != expected_status
        or not isinstance(response["payload"], dict)
    ):
        raise ModelCompileError(
            "validation executor response does not match its request"
        )
    return response


def validate_validation_mount_payload(
    payload: Json,
    *,
    candidate_id: str | None,
    physical_device_ids: tuple[str, ...],
) -> None:
    if (
        payload.get("candidate_id") != candidate_id
        or payload.get("physical_device_ids")
        != list(physical_device_ids)
    ):
        raise ModelCompileError(
            "validation executor mounted different role conditions"
        )
    required_text(payload, "package_id")
    required_device_state_digest(payload, "mounted_state_digest")
    positive_integer(
        payload.get("context_capacity"),
        "validation executor context capacity",
    )
    positive_integer(
        payload.get("mount_duration_ns"),
        "validation executor mount duration",
    )


def validate_validation_execution_payload(
    payload: Json,
    *,
    expected_step_unit: str,
    expected_turns: tuple[str, ...],
) -> None:
    expected = {
        "output_digest",
        "state_digest",
        "steps",
        "step_unit",
        "scheduler_steps",
        "elapsed_ns",
        "turns",
        "execution_counters",
    }
    if set(payload) != expected:
        raise ModelCompileError(
            "validation executor execution fields are invalid"
        )
    required_digest(payload, "output_digest")
    required_digest(payload, "state_digest")
    if payload.get("step_unit") != expected_step_unit:
        raise ModelCompileError(
            "validation executor reported a different step unit"
        )
    positive_integer(
        payload.get("steps"),
        "validation executor component activations",
    )
    positive_integer(
        payload.get("scheduler_steps"),
        "validation executor scheduler steps",
    )
    positive_integer(
        payload.get("elapsed_ns"),
        "validation executor elapsed time",
    )
    turns = payload.get("turns")
    if (
        not isinstance(turns, list)
        or len(turns) != len(expected_turns)
    ):
        raise ModelCompileError(
            "validation executor did not complete every requested turn"
        )
    for index, (turn, expected_user) in enumerate(
        zip(turns, expected_turns, strict=True)
    ):
        if (
            not isinstance(turn, dict)
            or turn.get("turn_index") != index
            or turn.get("user") != expected_user
            or not isinstance(turn.get("assistant"), str)
            or not isinstance(turn.get("generated_token_ids"), list)
            or not isinstance(
                turn.get("canonical_committed_token_ids"),
                list,
            )
            or turn.get("stop_reason")
            not in {"eos", "output_allowance", "fixture_completed"}
        ):
            raise ModelCompileError(
                "validation executor returned malformed conversation trace"
            )
        required_digest(turn, "state_digest")
    required_object(payload, "execution_counters")


def validate_validation_release_payload(
    payload: Json,
    *,
    mounted_state_digest: str,
    physical_device_ids: tuple[str, ...],
) -> None:
    if (
        payload.get("released") is not True
        or payload.get("mounted_state_digest")
        != mounted_state_digest
        or sorted(payload.get("released_device_ids", []))
        != sorted(
            "optimizer:device:" + str(index)
            for index in range(len(physical_device_ids))
        )
    ):
        raise ModelCompileError(
            "validation executor did not prove complete role release"
        )
    positive_integer(
        payload.get("reset_duration_ns"),
        "validation executor state reset duration",
    )
    positive_integer(
        payload.get("state_proof_duration_ns"),
        "validation executor state proof duration",
    )
    positive_integer(
        payload.get("release_duration_ns"),
        "validation executor release duration",
    )


def validate_validation_shutdown_payload(
    payload: Json,
    *,
    physical_device_ids: tuple[str, ...],
) -> None:
    if set(payload) != {
        "released",
        "physical_device_ids",
        "pre_release_quiesce_duration_ns",
        "role_release_duration_ns",
        "device_releases",
        "shutdown_duration_ns",
    }:
        raise ModelCompileError(
            "validation executor shutdown fields are invalid"
        )
    if (
        payload["released"] is not True
        or payload["physical_device_ids"] != list(physical_device_ids)
    ):
        raise ModelCompileError(
            "validation executor shut down a different device topology"
        )
    for field in (
        "pre_release_quiesce_duration_ns",
        "role_release_duration_ns",
        "shutdown_duration_ns",
    ):
        positive_integer(
            payload[field],
            f"validation executor {field}",
        )
    releases = payload["device_releases"]
    if (
        not isinstance(releases, list)
        or len(releases) != len(physical_device_ids)
    ):
        raise ModelCompileError(
            "validation executor did not release every physical device"
        )
    for index, (release, physical_device_id) in enumerate(
        zip(releases, physical_device_ids, strict=True)
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
            != f"optimizer:device:{index}"
            or release["quiesced"] is not True
            or release["device_context_destroyed"] is not True
        ):
            raise ModelCompileError(
                "validation executor device shutdown proof is invalid"
            )
        for field in (
            "released_buffer_count",
            "released_buffer_bytes",
        ):
            value = release[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ModelCompileError(
                    f"validation executor shutdown {field} is invalid"
                )
        positive_integer(
            release["release_duration_ns"],
            "validation executor device release duration",
        )
