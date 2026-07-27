from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nerve.compilation import Json
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.validation.contracts import (
    ValidationResidencyEvent,
    ValidationRoleResult,
)
from nerve.representation_optimizer.validation.executor_protocol import (
    VALIDATION_EXECUTOR_RESPONSE_SCHEMA,
)
from nerve.representation_optimizer.validation.planning import (
    create_validation_check,
)
from nerve.representation_optimizer.validation.protocols import (
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)
from nerve.representation_optimizer.validation.whole_model_executor import (
    ResidentWholeModelValidationBackend,
)


class FixtureWholeModelExecutor:
    def __init__(self) -> None:
        self.commands: list[Json] = []
        self.closed = False
        self.aborted = False

    def request(self, document: Json, *, cancel_requested=None) -> Json:
        assert cancel_requested is None or not cancel_requested()
        self.commands.append(document)
        if document["command"] == "mount":
            payload = {
                "package_id": "package-fixture",
                "candidate_id": document["candidate_id"],
                "physical_device_ids": document[
                    "physical_device_ids"
                ],
                "context_capacity": (
                    document["context_capacity"] or 131_072
                ),
                "mounted_state_digest": _digest(b"mounted"),
                "mount_duration_ns": 11,
            }
            status = "mounted"
        elif document["command"] == "execute":
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
                    "component_activations": 512,
                    "scheduler_steps": 8,
                    "elapsed_ns": 101,
                    "execution_counters": {
                        "execution_quantum_dispatch_count": 32,
                    },
                }
                for index, user in enumerate(document["turns"])
            ]
            payload = {
                "output_digest": _digest(b"output"),
                "state_digest": _digest(b"state"),
                "steps": 1_024,
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
                "mounted_state_digest": _digest(b"mounted"),
                "released_device_ids": [
                    "optimizer:device:0",
                    "optimizer:device:1",
                ],
                "release_duration_ns": 7,
            }
            status = "released"
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


def test_whole_model_validation_runs_normal_conversation_and_rotates_placement(
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
        b'{"schema":"nerve.optimizer.validation_conversation.v1",'
        b'"turns":["Who are you?","What is the capital of Greece?"],'
        b'"teacher_forced_assistant_turns":["A model.","Athens."]}\n'
    )
    (fixtures / "conversation.json").write_bytes(fixture_payload)
    driver = tmp_path / "radeon_icd.json"
    driver.write_text("{}\n", encoding="utf-8")
    executor = FixtureWholeModelExecutor()
    captured_environment: dict[str, str] = {}
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
        assert command == ("fixture-validation-executor",)
        captured_environment.update(environment)
        return executor

    check = create_validation_check(
        name="fixture alternative placement conversation",
        stage="full_local",
        kind="placement",
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
            "enable_thinking": True,
        },
        seeds=(17,),
        minimum_steps=512,
        output_allowance=65_536,
        output_allowance_basis={
            "kind": "declared_model_limit",
            "artifact": {
                "path": "fixtures/model_limits.json",
                "digest": _digest(b"limits"),
            },
            "json_pointer": "/max_output_tokens",
            "declared_limit": 65_536,
        },
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
        "idle_device_state_digest": _digest(b"idle"),
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
    comparison = backend.compare_results(
        {
            "check": check,
            "behavioral_contract": {"mode": "exact"},
        },
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
    assert mount_command["context_capacity"] is None
    assert executor.commands[1]["max_output_tokens"] == 65_536
    assert executor.commands[1]["execution_mode"] == "conversation"
    assert executor.commands[1][
        "teacher_forced_assistant_turns"
    ] == ["A model.", "Athens."]
    assert result["steps"] == 1_024
    assert result["default_statistics"]["scheduler_steps"] == 16
    assert comparison["metrics"][0]["error"] == 0.0
    assert mount["device_state_before_digest"] == _digest(b"idle")
    assert unmount["device_state_after_digest"] == _digest(b"idle")
    assert executor.closed is True
    assert executor.aborted is False
    assert captured_environment["VK_DRIVER_FILES"] == str(
        driver.resolve()
    )
    assert "VK_ICD_FILENAMES" not in captured_environment


def _digest(payload: bytes) -> str:
    return staged_artifact_digest(payload)
