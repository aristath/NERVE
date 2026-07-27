from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    positive_integer,
    required_digest,
    required_device_state_digest,
    required_object,
    required_text,
)


VALIDATION_EXECUTOR_COMMAND_SCHEMA = (
    "nerve.optimizer.validation_executor_command.v1"
)
VALIDATION_EXECUTOR_RESPONSE_SCHEMA = (
    "nerve.optimizer.validation_executor_response.v2"
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


def validate_validation_execution_payload(payload: Json) -> None:
    expected = {
        "output_digest",
        "state_digest",
        "steps",
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
    if not isinstance(turns, list) or not turns:
        raise ModelCompileError(
            "validation executor omitted conversation turns"
        )
    for index, turn in enumerate(turns):
        if (
            not isinstance(turn, dict)
            or turn.get("turn_index") != index
            or not isinstance(turn.get("user"), str)
            or not turn["user"]
            or not isinstance(turn.get("assistant"), str)
            or not isinstance(turn.get("generated_token_ids"), list)
            or not isinstance(
                turn.get("canonical_committed_token_ids"),
                list,
            )
        ):
            raise ModelCompileError(
                "validation executor returned malformed conversation trace"
            )
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
        payload.get("release_duration_ns"),
        "validation executor release duration",
    )
