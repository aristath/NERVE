from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    nonnegative_integer,
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
    "nerve.optimizer.validation_executor_command.v8"
)
VALIDATION_EXECUTOR_RESPONSE_SCHEMA = (
    "nerve.optimizer.validation_executor_response.v7"
)

_PROGRESS_FIELDS = {
    "generation": {
        "phase",
        "conversation_set_index",
        "turn_index",
        "generated_tokens",
        "token_id",
        "selected_logit_bits",
        "elapsed_ns",
    },
    "turn_completed": {
        "phase",
        "conversation_set_index",
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
    for field in ("token_id", "selected_logit_bits"):
        if field in payload:
            value = nonnegative_integer(
                payload[field],
                f"validation progress {field}",
            )
            if value > 0xFFFF_FFFF:
                raise ModelCompileError(
                    f"validation progress {field} exceeds u32"
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
    conversation_set_index = payload.get("conversation_set_index")
    if conversation_set_index is not None and (
        isinstance(conversation_set_index, bool)
        or not isinstance(conversation_set_index, int)
        or conversation_set_index < 0
        or conversation_set_index > 2
    ):
        raise ModelCompileError(
            "validation progress conversation set index is invalid"
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
    expected_conversation_set_policy: Json,
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
        "conversation_sets",
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
    _validate_execution_turns(payload.get("turns"), expected_turns)
    required_object(payload, "execution_counters")
    _validate_conversation_sets(
        payload.get("conversation_sets"),
        expected_turns=expected_turns,
        expected_policy=expected_conversation_set_policy,
    )


def _validate_execution_turns(
    value: object,
    expected_turns: tuple[str, ...],
) -> None:
    if not isinstance(value, list) or len(value) != len(expected_turns):
        raise ModelCompileError(
            "validation executor did not complete every requested turn"
        )
    for index, (turn, expected_user) in enumerate(
        zip(value, expected_turns, strict=True)
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
        positive_integer(
            turn.get("elapsed_ns"),
            "validation executor turn elapsed time",
        )
        positive_integer(
            turn.get("scheduler_steps"),
            "validation executor turn scheduler steps",
        )
        required_object(turn, "execution_counters")
        speculative = required_object(turn, "speculative")
        feedback = required_object(turn, "resident_feedback")
        transport = required_object(turn, "transport")
        for path, document, fields in (
            (
                "speculative",
                speculative,
                (
                    "cycle_count",
                    "rollback_cycle_count",
                    "proposed_draft_tokens",
                    "accepted_draft_tokens",
                    "emitted_tokens",
                    "draft_time_ns",
                    "target_verification_time_ns",
                    "draft_catch_up_time_ns",
                    "total_time_ns",
                ),
            ),
            (
                "resident_feedback",
                feedback,
                (
                    "window_count",
                    "planned_tick_count",
                    "submitted_tick_count",
                    "executed_tick_count",
                    "retained_tick_count",
                    "sampled_tick_count",
                    "discarded_tick_count",
                    "template_record_count",
                    "template_replay_count",
                    "asynchronous_submission_count",
                    "completion_poll_count",
                    "bounded_wait_count",
                    "bounded_wait_timeout_count",
                ),
            ),
            (
                "transport",
                transport,
                (
                    "published_packet_count",
                    "published_byte_count",
                    "received_packet_count",
                    "received_byte_count",
                    "direct_copy_count",
                    "direct_copy_byte_count",
                    "direct_receive_count",
                    "direct_receive_byte_count",
                ),
            ),
        ):
            if set(document) != set(fields):
                raise ModelCompileError(
                    f"validation executor turn {path} fields are invalid"
                )
            for field in fields:
                field_value = document[field]
                if (
                    isinstance(field_value, bool)
                    or not isinstance(field_value, int)
                    or field_value < 0
                ):
                    raise ModelCompileError(
                        f"validation executor turn {path}.{field} is invalid"
                    )
        if (
            speculative["accepted_draft_tokens"]
            > speculative["proposed_draft_tokens"]
        ):
            raise ModelCompileError(
                "validation executor accepted more draft tokens than proposed"
            )


def _validate_conversation_sets(
    value: object,
    *,
    expected_turns: tuple[str, ...],
    expected_policy: Json,
) -> None:
    if not isinstance(value, list):
        raise ModelCompileError(
            "validation executor conversation sets are invalid"
        )
    minimum = nonnegative_integer(
        expected_policy.get("minimum_sets"),
        "validation conversation minimum sets",
    )
    maximum = nonnegative_integer(
        expected_policy.get("maximum_sets"),
        "validation conversation maximum sets",
    )
    repeat_loading = expected_policy.get(
        "repeat_second_set_if_residency_loaded"
    )
    if (
        maximum < minimum
        or maximum > 3
        or not isinstance(repeat_loading, bool)
        or ((minimum == 0) != (maximum == 0))
        or (minimum == 0 and repeat_loading)
        or (repeat_loading and (minimum != 2 or maximum != 3))
    ):
        raise ModelCompileError(
            "validation conversation set policy is invalid"
        )
    if not (minimum <= len(value) <= maximum):
        raise ModelCompileError(
            "validation executor completed an invalid conversation set count"
        )
    for set_index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "set_index",
            "disposition",
            "residency_activity",
            "turns",
        }:
            raise ModelCompileError(
                "validation executor conversation set fields are invalid"
            )
        expected_disposition = (
            "measured"
            if set_index == len(value) - 1
            else "discarded_warmup"
        )
        if (
            item["set_index"] != set_index
            or item["disposition"] != expected_disposition
        ):
            raise ModelCompileError(
                "validation executor conversation set ordering is invalid"
            )
        activity = required_object(item, "residency_activity")
        if set(activity) != {
            "load_required_count",
            "physical_bytes_read",
            "reload_count",
            "uploaded_bytes",
        }:
            raise ModelCompileError(
                "validation executor conversation residency activity is invalid"
            )
        for field, field_value in activity.items():
            nonnegative_integer(
                field_value,
                f"validation conversation residency activity {field}",
            )
        _validate_execution_turns(item["turns"], expected_turns)
    if repeat_loading and len(value) == 2:
        second_activity = value[1]["residency_activity"]
        if any(second_activity[field] > 0 for field in second_activity):
            raise ModelCompileError(
                "validation executor measured a loading second conversation "
                "instead of running the required third set"
            )


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
        "engine_shutdown",
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
    _validate_engine_shutdown_payload(
        payload["engine_shutdown"],
        physical_device_ids=physical_device_ids,
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


def _validate_engine_shutdown_payload(
    payload: object,
    *,
    physical_device_ids: tuple[str, ...],
) -> None:
    fields = {
        "stream_count",
        "package_count",
        "scheduler_in_flight_activation_count",
        "physical_device_count",
        "acknowledged_device_count",
        "released_unit_count",
        "released_payload_bytes",
        "cancelled_load_count",
        "resource_teardowns",
        "complete",
        "errors",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ModelCompileError(
            "validation executor engine shutdown fields are invalid"
        )
    for field in fields - {"resource_teardowns", "complete", "errors"}:
        nonnegative_integer(
            payload[field],
            f"validation executor engine shutdown {field}",
        )
    if (
        payload["complete"] is not True
        or payload["errors"] != []
        or payload["scheduler_in_flight_activation_count"] != 0
        or payload["physical_device_count"]
        != payload["acknowledged_device_count"]
    ):
        raise ModelCompileError(
            "validation executor engine shutdown proof is incomplete"
        )
    teardowns = payload["resource_teardowns"]
    if (
        not isinstance(teardowns, list)
        or len(teardowns) != payload["package_count"]
    ):
        raise ModelCompileError(
            "validation executor engine shutdown package proof is invalid"
        )
    acknowledged_device_count = 0
    released_unit_count = 0
    released_payload_bytes = 0
    cancelled_load_count = 0
    for teardown in teardowns:
        counts = _validate_engine_resource_teardown(
            teardown,
            physical_device_ids=physical_device_ids,
        )
        acknowledged_device_count += counts[0]
        released_unit_count += counts[1]
        released_payload_bytes += counts[2]
        cancelled_load_count += counts[3]
    if (
        payload["acknowledged_device_count"]
        != acknowledged_device_count
        or payload["released_unit_count"] != released_unit_count
        or payload["released_payload_bytes"] != released_payload_bytes
        or payload["cancelled_load_count"] != cancelled_load_count
    ):
        raise ModelCompileError(
            "validation executor engine shutdown totals are invalid"
        )


def _validate_engine_resource_teardown(
    payload: object,
    *,
    physical_device_ids: tuple[str, ...],
) -> tuple[int, int, int, int]:
    fields = {
        "package_id",
        "execution_scope",
        "physical_device_count",
        "released_unit_count",
        "released_payload_bytes",
        "cancelled_load_count",
        "acknowledged_device_count",
        "complete",
        "devices",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ModelCompileError(
            "validation executor engine resource teardown fields are invalid"
        )
    if (
        not isinstance(payload["package_id"], str)
        or not payload["package_id"]
        or not isinstance(payload["execution_scope"], str)
        or not payload["execution_scope"]
    ):
        raise ModelCompileError(
            "validation executor engine resource identity is invalid"
        )
    for field in fields - {
        "package_id",
        "execution_scope",
        "complete",
        "devices",
    }:
        nonnegative_integer(
            payload[field],
            f"validation executor engine resource teardown {field}",
        )
    devices = payload["devices"]
    if (
        payload["complete"] is not True
        or payload["physical_device_count"]
        != payload["acknowledged_device_count"]
        or not isinstance(devices, list)
        or len(devices) != payload["physical_device_count"]
    ):
        raise ModelCompileError(
            "validation executor engine resource teardown is incomplete"
        )
    released_unit_count = 0
    released_payload_bytes = 0
    cancelled_load_count = 0
    seen_physical_devices: set[str] = set()
    for device in devices:
        counts = _validate_engine_resource_device_teardown(
            device,
            physical_device_ids=physical_device_ids,
        )
        physical_device_id = device["physical_device_id"]
        if physical_device_id in seen_physical_devices:
            raise ModelCompileError(
                "validation executor engine resource device proof is duplicated"
            )
        seen_physical_devices.add(physical_device_id)
        released_unit_count += counts[0]
        released_payload_bytes += counts[1]
        cancelled_load_count += counts[2]
    if (
        payload["released_unit_count"] != released_unit_count
        or payload["released_payload_bytes"] != released_payload_bytes
        or payload["cancelled_load_count"] != cancelled_load_count
    ):
        raise ModelCompileError(
            "validation executor engine resource totals are invalid"
        )
    return (
        payload["acknowledged_device_count"],
        released_unit_count,
        released_payload_bytes,
        cancelled_load_count,
    )


def _validate_engine_resource_device_teardown(
    payload: object,
    *,
    physical_device_ids: tuple[str, ...],
) -> tuple[int, int, int]:
    fields = {
        "store_id",
        "physical_device_id",
        "logical_device_ids",
        "released_unit_count",
        "released_payload_bytes",
        "cancelled_load_count",
        "remaining_unit_count",
        "remaining_payload_bytes",
        "acknowledged",
        "error",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ModelCompileError(
            "validation executor engine resource device fields are invalid"
        )
    logical_device_ids = payload["logical_device_ids"]
    if (
        not isinstance(payload["store_id"], str)
        or not payload["store_id"]
        or payload["physical_device_id"] not in physical_device_ids
        or not isinstance(logical_device_ids, list)
        or not logical_device_ids
        or any(
            not isinstance(device_id, str) or not device_id
            for device_id in logical_device_ids
        )
        or len(set(logical_device_ids)) != len(logical_device_ids)
    ):
        raise ModelCompileError(
            "validation executor engine resource device identity is invalid"
        )
    for field in fields - {
        "store_id",
        "physical_device_id",
        "logical_device_ids",
        "acknowledged",
        "error",
    }:
        nonnegative_integer(
            payload[field],
            f"validation executor engine resource device {field}",
        )
    if (
        payload["acknowledged"] is not True
        or payload["error"] is not None
        or payload["remaining_unit_count"] != 0
        or payload["remaining_payload_bytes"] != 0
    ):
        raise ModelCompileError(
            "validation executor engine resource device proof is incomplete"
        )
    return (
        payload["released_unit_count"],
        payload["released_payload_bytes"],
        payload["cancelled_load_count"],
    )
