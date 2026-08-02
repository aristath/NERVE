from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    EXECUTOR_PROGRESS_SCHEMA,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    device_state_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.validation.contracts import (
    ValidationResidencyEvent,
    ValidationRoleResult,
)
from nerve.representation_optimizer.validation.conversation_semantics import (
    VALIDATION_CONVERSATION_SCHEMA,
)
from nerve.representation_optimizer.validation.executor_protocol import (
    VALIDATION_EXECUTOR_RESPONSE_SCHEMA,
    validate_validation_execution_payload,
    validate_validation_progress,
    validate_validation_shutdown_payload,
)
from nerve.representation_optimizer.validation.planning import (
    create_validation_check,
)
from nerve.representation_optimizer.validation.protocols import (
    ValidationComparisonRequest,
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)
from nerve.representation_optimizer.validation.whole_model_executor import (
    ResidentWholeModelValidationBackend,
)


class FixtureWholeModelExecutor:
    def __init__(self) -> None:
        self.commands: list[Json] = []
        self.request_cancellation_callbacks: list[object | None] = []
        self.closed = False
        self.aborted = False

    def request(
        self,
        document: Json,
        *,
        cancel_requested=None,
        progress_received=None,
    ) -> Json:
        assert cancel_requested is None or not cancel_requested()
        self.request_cancellation_callbacks.append(cancel_requested)
        self.commands.append(document)
        if document["command"] == "mount":
            capacity = document["context_capacity"]
            payload = {
                "package_id": "package-fixture",
                "candidate_id": document["candidate_id"],
                "physical_device_ids": document[
                    "physical_device_ids"
                ],
                "context_capacity": (
                    capacity["activations"]
                    if capacity["kind"] == "declared"
                    else 77
                ),
                "mounted_state_digest": _device_digest(b"mounted"),
                "mount_duration_ns": 11,
            }
            status = "mounted"
        elif document["command"] == "execute":
            assert progress_received is not None
            progress_received(
                {
                    "schema": EXECUTOR_PROGRESS_SCHEMA,
                    "request_id": document["request_id"],
                    "sequence": 0,
                    "payload": {
                        "phase": "teacher_forced_turn_completed",
                        "turn_index": 0,
                        "generated_tokens": 0,
                        "elapsed_ns": 101,
                        "component_activations": 180,
                        "scheduler_steps": 8,
                    },
                }
            )
            turns = [
                {
                    "turn_index": index,
                    "user": user,
                    "assistant": f"answer {index}",
                    "generated_token_ids": [index + 1, 2],
                    "canonical_committed_token_ids": [
                        index + 1,
                        2,
                        3,
                    ],
                    "stop_reason": "fixture_completed",
                    "state_digest": _digest(
                        f"turn-state-{index}".encode()
                    ),
                    "component_activations": 180,
                    "scheduler_steps": 8,
                    "elapsed_ns": 101,
                    "execution_counters": {
                        "execution_quantum_dispatch_count": 32,
                    },
                    "speculative": {
                        "cycle_count": 0,
                        "rollback_cycle_count": 0,
                        "proposed_draft_tokens": 0,
                        "accepted_draft_tokens": 0,
                        "emitted_tokens": 0,
                        "draft_time_ns": 0,
                        "target_verification_time_ns": 0,
                        "draft_catch_up_time_ns": 0,
                        "total_time_ns": 0,
                    },
                    "resident_feedback": {
                        "window_count": 0,
                        "planned_tick_count": 0,
                        "submitted_tick_count": 0,
                        "executed_tick_count": 0,
                        "retained_tick_count": 0,
                        "sampled_tick_count": 0,
                        "discarded_tick_count": 0,
                        "template_record_count": 0,
                        "template_replay_count": 0,
                        "asynchronous_submission_count": 0,
                        "completion_poll_count": 0,
                        "bounded_wait_count": 0,
                        "bounded_wait_timeout_count": 0,
                    },
                    "transport": {
                        "published_packet_count": 0,
                        "published_byte_count": 0,
                        "received_packet_count": 0,
                        "received_byte_count": 0,
                        "direct_copy_count": 0,
                        "direct_copy_byte_count": 0,
                        "direct_receive_count": 0,
                        "direct_receive_byte_count": 0,
                    },
                }
                for index, user in enumerate(document["turns"])
            ]
            payload = {
                "output_digest": _digest(b"output"),
                "state_digest": _digest(b"state"),
                "steps": 360,
                "step_unit": document["step_unit"],
                "scheduler_steps": 16,
                "elapsed_ns": 202,
                "turns": turns,
                "execution_counters": {
                    "execution_quantum_dispatch_count": 64,
                },
            }
            status = "completed"
        elif document["command"] == "close":
            payload = {
                "released": True,
                "mounted_state_digest": _device_digest(b"mounted"),
                "released_device_ids": [
                    "optimizer:device:0",
                    "optimizer:device:1",
                ],
                "release_duration_ns": 7,
                "reset_duration_ns": 2,
                "state_proof_duration_ns": 3,
            }
            status = "released"
        elif document["command"] == "shutdown":
            physical_device_ids = [
                "vulkan-uuid:" + "1" * 32,
                "vulkan-uuid:" + "2" * 32,
            ]
            payload = {
                "released": True,
                "physical_device_ids": physical_device_ids,
                "pre_release_quiesce_duration_ns": 5,
                "role_release_duration_ns": 6,
                "engine_shutdown": _engine_shutdown_payload(
                    tuple(physical_device_ids)
                ),
                "device_releases": [
                    {
                        "physical_device_id": physical_device_id,
                        "logical_device_id": f"optimizer:device:{index}",
                        "released_buffer_count": 7 + index,
                        "released_buffer_bytes": 1024 * (index + 1),
                        "quiesced": True,
                        "device_context_destroyed": True,
                        "release_duration_ns": 8 + index,
                    }
                    for index, physical_device_id in enumerate(
                        physical_device_ids
                    )
                ],
                "shutdown_duration_ns": 10,
            }
            status = "shutdown_complete"
        else:
            raise AssertionError(
                f"unexpected command {document['command']!r}"
            )
        return {
            "schema": VALIDATION_EXECUTOR_RESPONSE_SCHEMA,
            "request_id": document["request_id"],
            "status": status,
            "payload": payload,
        }

    def close(self, *, cancel_requested=None) -> None:
        assert cancel_requested is None or not cancel_requested()
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


def test_generation_progress_identifies_the_observed_token() -> None:
    event = {
        "schema": EXECUTOR_PROGRESS_SCHEMA,
        "request_id": "execute",
        "sequence": 0,
        "payload": {
            "phase": "generation",
            "turn_index": 0,
            "generated_tokens": 1,
            "token_id": 128_799,
            "selected_logit_bits": 0x4120_0000,
            "elapsed_ns": 1,
        },
    }

    validate_validation_progress(
        event,
        expected_request_id="execute",
        turn_count=1,
    )
    del event["payload"]["token_id"]
    with pytest.raises(ModelCompileError, match="progress payload is invalid"):
        validate_validation_progress(
            event,
            expected_request_id="execute",
            turn_count=1,
        )

    event["payload"]["token_id"] = 128_799
    del event["payload"]["selected_logit_bits"]
    with pytest.raises(ModelCompileError, match="progress payload is invalid"):
        validate_validation_progress(
            event,
            expected_request_id="execute",
            turn_count=1,
        )


def test_validation_execution_requires_every_requested_turn() -> None:
    payload = {
        "output_digest": _digest(b"output"),
        "state_digest": _digest(b"state"),
        "steps": 1,
        "step_unit": "component_activations",
        "scheduler_steps": 1,
        "elapsed_ns": 1,
        "turns": [
            {
                "turn_index": 0,
                "user": "first",
                "assistant": "answer",
                "generated_token_ids": [1],
                "canonical_committed_token_ids": [1],
                "stop_reason": "eos",
                "state_digest": _digest(b"turn-state"),
            }
        ],
        "execution_counters": {"dispatches": 1},
    }

    with pytest.raises(
        ModelCompileError,
        match="complete every requested turn",
    ):
        validate_validation_execution_payload(
            payload,
            expected_step_unit="component_activations",
            expected_turns=("first", "second"),
        )


def test_validation_shutdown_requires_ordered_destroyed_device_proof() -> None:
    physical_device_ids = (
        "vulkan-uuid:" + "1" * 32,
        "vulkan-uuid:" + "2" * 32,
    )
    payload = {
        "released": True,
        "physical_device_ids": list(physical_device_ids),
        "pre_release_quiesce_duration_ns": 1,
        "role_release_duration_ns": 2,
        "engine_shutdown": _engine_shutdown_payload(
            physical_device_ids
        ),
        "device_releases": [
            {
                "physical_device_id": physical_device_id,
                "logical_device_id": f"optimizer:device:{index}",
                "released_buffer_count": 1,
                "released_buffer_bytes": 1024,
                "quiesced": True,
                "device_context_destroyed": True,
                "release_duration_ns": 3,
            }
            for index, physical_device_id in enumerate(
                reversed(physical_device_ids)
            )
        ],
        "shutdown_duration_ns": 4,
    }

    with pytest.raises(
        ModelCompileError,
        match="device shutdown proof is invalid",
    ):
        validate_validation_shutdown_payload(
            payload,
            physical_device_ids=physical_device_ids,
        )


def test_validation_shutdown_rejects_incomplete_engine_teardown() -> None:
    physical_device_ids = (
        "vulkan-uuid:" + "1" * 32,
        "vulkan-uuid:" + "2" * 32,
    )
    payload = {
        "released": True,
        "physical_device_ids": list(physical_device_ids),
        "pre_release_quiesce_duration_ns": 1,
        "role_release_duration_ns": 2,
        "engine_shutdown": _engine_shutdown_payload(
            physical_device_ids
        ),
        "device_releases": [
            {
                "physical_device_id": physical_device_id,
                "logical_device_id": f"optimizer:device:{index}",
                "released_buffer_count": 1,
                "released_buffer_bytes": 1024,
                "quiesced": True,
                "device_context_destroyed": True,
                "release_duration_ns": 3,
            }
            for index, physical_device_id in enumerate(
                physical_device_ids
            )
        ],
        "shutdown_duration_ns": 4,
    }
    payload["engine_shutdown"]["resource_teardowns"][0]["devices"][0][
        "remaining_payload_bytes"
    ] = 1

    with pytest.raises(
        ModelCompileError,
        match="resource device proof is incomplete",
    ):
        validate_validation_shutdown_payload(
            payload,
            physical_device_ids=physical_device_ids,
        )


def test_whole_model_validation_uses_fixture_sized_structural_replay_and_rotates_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = package / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "candidate"
    fixtures = candidate / "fixtures"
    fixtures.mkdir(parents=True)
    fixture_payload = (
        f'{{"schema":"{VALIDATION_CONVERSATION_SCHEMA}",'
        '"turns":["Who are you?","What is the capital of Greece?"],'
        '"teacher_forced_assistant_turns":["A model.","Athens."]}\n'
    ).encode()
    (fixtures / "conversation.json").write_bytes(fixture_payload)
    driver = tmp_path / "radeon_icd.json"
    driver.write_text("{}\n", encoding="utf-8")
    executor = FixtureWholeModelExecutor()
    captured_environment: dict[str, str] = {}
    factory_calls = 0
    candidate_id = "candidate_" + "a" * 32

    def loader(
        workspace_root: Path,
        requested_candidate_id: str,
        package_dir: Path,
    ) -> object:
        assert workspace_root == (tmp_path / "workspace").resolve()
        assert requested_candidate_id == candidate_id
        assert package_dir == package.resolve()
        return SimpleNamespace(path=candidate.resolve())

    def factory(
        command: tuple[str, ...],
        environment: dict[str, str],
    ) -> FixtureWholeModelExecutor:
        nonlocal factory_calls
        factory_calls += 1
        assert command == ("fixture-validation-executor",)
        captured_environment.update(environment)
        return executor

    check = create_validation_check(
        name="fixture alternative placement conversation",
        stage="full_local",
        kind="placement",
        product_performance=False,
        coverage=("alternative_placements",),
        execution_scope="whole_model",
        activation_batch_width=1,
        context_size=0,
        context_size_basis={"kind": "not_applicable"},
        state_size=0,
        boundary_mode="cross_device",
        input_artifact={
            "path": "fixtures/conversation.json",
            "digest": _digest(fixture_payload),
        },
        initial_state_artifact=None,
        controls={
            "execution": "ordinary",
            "execution_mode": "teacher_forced",
            "enable_thinking": True,
            "sampler": {"top_k": 1},
        },
        seeds=(17,),
        step_unit="component_activations",
        completion_condition="all_fixture_turns",
        minimum_steps=None,
        output_allowance=None,
        output_allowance_basis={"kind": "unlimited"},
        metrics=("token_exact_match",),
    )
    devices = (
        "vulkan-uuid:" + "1" * 32,
        "vulkan-uuid:" + "2" * 32,
    )
    conditions = {
        "devices": [
            {"device_id": device_id} for device_id in devices
        ],
        "placement": {
            "block_0": devices[0],
            "block_1": devices[1],
        },
        "controls": {
            "scheduler": "normal",
            "maximum_quantum_wait_ns": 9_000_000,
        },
        "environment": {"power_profile": "matched"},
        "idle_device_state_digest": _device_digest(b"idle"),
        "exclusive_residency": True,
    }
    plan_id = stable_contract_id(
        "validation_plan",
        str(tmp_path),
    )
    implementation = {
        "implementation_id": (
            f"staged-representation:{candidate_id}"
        )
    }
    mount_request = ValidationRoleMountRequest(
        plan_id=plan_id,
        candidate_id=candidate_id,
        stage="full_local",
        check=check,
        role="candidate",
        implementation=implementation,
        matched_conditions=conditions,
        matched_conditions_digest=contract_digest(conditions),
        seed=17,
        block_index=0,
    )
    execution_request = ValidationRoleExecutionRequest(
        plan_id=plan_id,
        candidate_id=candidate_id,
        check=check,
        role="candidate",
        implementation=implementation,
        matched_conditions=conditions,
        matched_conditions_digest=contract_digest(conditions),
        seed=17,
    )
    monkeypatch.setenv(
        "VK_ICD_FILENAMES",
        "/forbidden/nvidia.json",
    )
    backend = ResidentWholeModelValidationBackend(
        package_manifest=manifest,
        candidate_workspace=tmp_path / "workspace",
        trace_store=ExecutorArtifactStore(
            tmp_path / "traces",
            label="validation trace",
            create=True,
        ),
        executor_command=("fixture-validation-executor",),
        vulkan_driver_files=(driver,),
        executor_factory=factory,
        staged_candidate_loader=loader,  # type: ignore[arg-type]
        run_nonce="fixture",
    )
    cancel_stage = False

    with backend.validation_stage(
        "full_local",
        cancel_requested=lambda: cancel_stage,
    ):
        session = backend.open_session(mount_request)
        mount = ValidationResidencyEvent.from_json(
            session.mount_event
        ).to_json()
        result = ValidationRoleResult.from_json(
            session.execute(execution_request)
        ).to_json()
        unmount = ValidationResidencyEvent.from_json(
            session.close()
        ).to_json()
        assert executor.closed is False
        second_session = backend.open_session(mount_request)
        second_session.execute(execution_request)
        second_session.close()
        assert executor.closed is False
        # Once release begins, an expired optimization deadline must not turn
        # orderly accelerator teardown into a process kill.
        cancel_stage = True
    comparison = backend.compare_results(
        ValidationComparisonRequest(
            plan_id=plan_id,
            candidate_id=candidate_id,
            check=check,
            seed=17,
            behavioral_contract={"mode": "exact"},
        ),
        result,
        result,
    )

    mount_command = executor.commands[0]
    assert mount_command["component_placement"] == {
        "block_0": devices[1],
        "block_1": devices[0],
    }
    assert mount_command["enable_thinking"] is True
    assert mount_command["graph_operation"] == "none"
    assert mount_command["graph_target_component_id"] is None
    assert mount_command["context_capacity"] == {
        "kind": "fixture_exact",
    }
    assert mount_command["validation_turns"] == [
        "Who are you?",
        "What is the capital of Greece?",
    ]
    assert mount_command["teacher_forced_assistant_turns"] == [
        "A model.",
        "Athens.",
    ]
    assert mount_command["execution_mode"] == "teacher_forced"
    assert mount_command["speculative_draft_tokens"] == 0
    assert mount_command["sampler_config"] == {"top_k": 1}
    assert mount_command["residency_policy"] == "eager"
    assert executor.commands[1]["max_output_tokens"] is None
    assert executor.commands[1]["execution_mode"] == "teacher_forced"
    assert executor.commands[1]["step_unit"] == "component_activations"
    assert executor.commands[1][
        "teacher_forced_assistant_turns"
    ] == ["A model.", "Athens."]
    assert [
        command["command"] for command in executor.commands
    ] == [*(["mount", "execute", "close"] * 2), "shutdown"]
    assert executor.request_cancellation_callbacks == [None] * 7
    # Fixture completion is semantic: a short but complete conversation must
    # not be rejected because it did not cross an unrelated activation count.
    assert result["steps"] == 360
    assert result["horizon_completion"] == {
        "condition": "all_fixture_turns",
        "satisfied": True,
        "observed_steps": 360,
        "minimum_steps": None,
        "expected_turns": 2,
        "completed_turns": 2,
        "stop_reasons": ["fixture_completed", "fixture_completed"],
    }
    assert result["default_statistics"]["scheduler_steps"] == 16
    progress_ref = next(
        trace
        for trace in result["traces"]
        if trace["path"].endswith("/progress.jsonl")
    )
    progress_lines = [
        json.loads(line)
        for line in (
            tmp_path / "traces" / progress_ref["path"]
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert progress_lines[0]["payload"]["phase"] == (
        "teacher_forced_turn_completed"
    )
    assert progress_lines[-1] == {
        "schema": "nerve.optimizer.validation_progress_terminal.v1",
        "status": "completed",
    }
    assert comparison["metrics"][0]["error"] == 0.0
    assert mount["device_state_before_digest"] == _device_digest(b"idle")
    assert unmount["device_state_after_digest"] == _device_digest(b"idle")
    assert executor.closed is True
    assert executor.aborted is False
    assert factory_calls == 1
    assert captured_environment["VK_DRIVER_FILES"] == str(
        driver.resolve()
    )
    assert "VK_ICD_FILENAMES" not in captured_environment


def test_whole_model_validation_compares_free_running_traces_semantically(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = package / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    candidate = tmp_path / "candidate"
    fixtures = candidate / "fixtures"
    fixtures.mkdir(parents=True)
    fixture = {
        "schema": VALIDATION_CONVERSATION_SCHEMA,
        "turns": [
            "what is the capital of Greece?",
            "Which country did I ask about?",
        ],
        "teacher_forced_assistant_turns": [
            "Athens.",
            "Greece.",
        ],
        "semantic_expectations": {
            "require_thinking": True,
            "forbid_repeated_suffix": True,
            "turns": [
                {
                    "required_concepts": [
                        {
                            "name": "capital_city",
                            "any_terms": ["athens"],
                        }
                    ],
                    "conversation_memory": False,
                },
                {
                    "required_concepts": [
                        {
                            "name": "recalled_country",
                            "any_terms": ["greece"],
                        }
                    ],
                    "conversation_memory": True,
                },
            ],
        },
    }
    (fixtures / "conversation.json").write_text(
        json.dumps(fixture),
        encoding="utf-8",
    )
    candidate_id = "candidate_" + "b" * 32

    def loader(
        workspace_root: Path,
        requested_candidate_id: str,
        package_dir: Path,
    ) -> object:
        assert requested_candidate_id == candidate_id
        return SimpleNamespace(path=candidate.resolve())

    trace_store = ExecutorArtifactStore(
        tmp_path / "traces",
        label="validation trace",
        create=True,
    )
    reference_trace = {
        "turns": [
            {
                "user": fixture["turns"][0],
                "assistant": "reasoning</think>\n\nAthens is the capital.",
            },
            {
                "user": fixture["turns"][1],
                "assistant": "reasoning</think>\n\nYou asked about Greece.",
            },
        ]
    }
    candidate_trace = {
        "turns": [
            {
                "user": fixture["turns"][0],
                "assistant": "different reasoning</think>\n\nThe answer is Athens.",
            },
            {
                "user": fixture["turns"][1],
                "assistant": "different reasoning</think>\n\nThe country was Greece.",
            },
        ]
    }
    reference = trace_store.publish(
        "reference/conversation.json",
        json.dumps(reference_trace).encode(),
    )
    candidate_result = trace_store.publish(
        "candidate/conversation.json",
        json.dumps(candidate_trace).encode(),
    )
    backend = ResidentWholeModelValidationBackend(
        package_manifest=manifest,
        candidate_workspace=tmp_path / "workspace",
        trace_store=trace_store,
        executor_command=("unused",),
        vulkan_driver_files=(),
        executor_factory=lambda command, environment: (
            FixtureWholeModelExecutor()
        ),
        staged_candidate_loader=loader,  # type: ignore[arg-type]
        run_nonce="fixture",
    )

    comparison = backend.compare_results(
        ValidationComparisonRequest(
            plan_id="validation-plan",
            candidate_id=candidate_id,
            check={
                "input": {"path": "fixtures/conversation.json"},
                "comparison": {
                    "output_mode": "fixture_semantics",
                    "state_mode": "trajectory_local",
                },
                "metrics": [
                    "conversation_memory",
                    "semantic_consistency",
                ],
            },
            seed=1,
            behavioral_contract={"mode": "exact"},
        ),
        {"traces": [reference]},
        {"traces": [candidate_result]},
    )

    assert comparison["diagnostics"] == []
    assert all(
        metric["error"] == 0.0
        for metric in comparison["metrics"]
    )


def _digest(payload: bytes) -> str:
    return staged_artifact_digest(payload)


def _device_digest(payload: bytes) -> str:
    return device_state_digest({"fixture_state": payload.hex()})


def _engine_shutdown_payload(
    physical_device_ids: tuple[str, ...],
) -> Json:
    devices = [
        {
            "store_id": f"compiled-store-{index}",
            "physical_device_id": physical_device_id,
            "logical_device_ids": [f"optimizer:device:{index}"],
            "released_unit_count": index + 1,
            "released_payload_bytes": 1024 * (index + 1),
            "cancelled_load_count": 0,
            "remaining_unit_count": 0,
            "remaining_payload_bytes": 0,
            "acknowledged": True,
            "error": None,
        }
        for index, physical_device_id in enumerate(
            physical_device_ids
        )
    ]
    return {
        "stream_count": 1,
        "package_count": 1,
        "scheduler_in_flight_activation_count": 0,
        "physical_device_count": len(devices),
        "acknowledged_device_count": len(devices),
        "released_unit_count": sum(
            device["released_unit_count"] for device in devices
        ),
        "released_payload_bytes": sum(
            device["released_payload_bytes"] for device in devices
        ),
        "cancelled_load_count": 0,
        "resource_teardowns": [
            {
                "package_id": "package-fixture",
                "execution_scope": "whole_model",
                "physical_device_count": len(devices),
                "released_unit_count": sum(
                    device["released_unit_count"]
                    for device in devices
                ),
                "released_payload_bytes": sum(
                    device["released_payload_bytes"]
                    for device in devices
                ),
                "cancelled_load_count": 0,
                "acknowledged_device_count": len(devices),
                "complete": True,
                "devices": devices,
            }
        ],
        "complete": True,
        "errors": [],
    }
