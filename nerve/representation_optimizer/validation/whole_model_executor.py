from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
    StagedCandidateLoader,
    resolve_candidate_mount,
)
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    request_id,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    ExecutorFactory,
    ExecutorTransport,
)
from nerve.representation_optimizer.contracts import canonical_json_bytes
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_RESIDENCY_EVENT_SCHEMA,
    VALIDATION_ROLE_RESULT_SCHEMA,
    ValidationResidencyEvent,
    ValidationRoleResult,
    validation_residency_event_id,
    validation_role_result_id,
)
from nerve.representation_optimizer.validation.comparison import (
    compare_exact_role_results,
)
from nerve.representation_optimizer.validation.executor_protocol import (
    VALIDATION_EXECUTOR_COMMAND_SCHEMA,
    validate_validation_execution_payload,
    validate_validation_mount_payload,
    validate_validation_release_payload,
    validated_validation_response,
)
from nerve.representation_optimizer.validation.protocols import (
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)


class ResidentWholeModelValidationBackend:
    """Whole-model validation through normal resident chat transactions."""

    def __init__(
        self,
        *,
        package_manifest: Path,
        candidate_workspace: Path,
        trace_store: ExecutorArtifactStore,
        executor_command: tuple[str, ...],
        vulkan_driver_files: tuple[Path, ...],
        executor_factory: ExecutorFactory,
        staged_candidate_loader: StagedCandidateLoader,
        run_nonce: str,
    ) -> None:
        self.package_manifest = package_manifest.resolve()
        self.package_dir = self.package_manifest.parent
        self.candidate_workspace = candidate_workspace.resolve()
        self.trace_store = trace_store
        self.executor_command = executor_command
        self.vulkan_driver_files = tuple(
            path.resolve() for path in vulkan_driver_files
        )
        self.executor_factory = executor_factory
        self.staged_candidate_loader = staged_candidate_loader
        self.run_nonce = run_nonce

    def open_session(
        self,
        request: ValidationRoleMountRequest,
    ) -> ResidentWholeModelValidationSession:
        check = request.check
        if check["regime"]["execution_scope"] != "whole_model":
            raise ModelCompileError(
                "whole-model validation received a component check"
            )
        turns, teacher_forced_assistant_turns = (
            self._conversation_fixture(
            request.candidate_id,
            check["input"]["path"],
            )
        )
        physical_device_ids = _physical_device_ids(
            request.matched_conditions
        )
        placement = _role_placement(
            request.matched_conditions,
            check,
            physical_device_ids,
        )
        candidate_id, candidate_root = resolve_candidate_mount(
            implementation_id=request.implementation[
                "implementation_id"
            ],
            workspace_root=self.candidate_workspace,
            package_dir=self.package_dir,
            loader=self.staged_candidate_loader,
        )
        context_size = check["regime"]["context_size"]
        command = {
            "schema": VALIDATION_EXECUTOR_COMMAND_SCHEMA,
            "command": "mount",
            "request_id": request_id("validation-mount", request.to_json()),
            "package_manifest": str(self.package_manifest),
            "candidate_root": (
                None if candidate_root is None else str(candidate_root)
            ),
            "candidate_id": candidate_id,
            "physical_device_ids": list(physical_device_ids),
            "component_placement": placement,
            "context_capacity": (
                context_size if context_size > 0 else None
            ),
            "random_seed": request.seed,
            "enable_thinking": (
                check["controls"].get("enable_thinking") is True
            ),
            "graph_operation": check["controls"].get(
                "graph_operation",
                "none",
            ),
            "graph_target_component_id": check["controls"].get(
                "graph_target_component_id"
            ),
        }
        transport = self.executor_factory(
            self.executor_command,
            self._environment(),
        )
        started = time.monotonic_ns()
        try:
            response = validated_validation_response(
                transport.request(command),
                expected_request_id=command["request_id"],
                expected_status="mounted",
            )
            payload = response["payload"]
            validate_validation_mount_payload(
                payload,
                candidate_id=candidate_id,
                physical_device_ids=physical_device_ids,
            )
        except BaseException:
            transport.abort()
            raise
        return ResidentWholeModelValidationSession(
            backend=self,
            request=request,
            transport=transport,
            mount_payload=payload,
            mount_duration_ns=max(
                1,
                time.monotonic_ns() - started,
            ),
            turns=turns,
            teacher_forced_assistant_turns=(
                teacher_forced_assistant_turns
            ),
            physical_device_ids=physical_device_ids,
        )

    def compare_results(
        self,
        request: Json,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        if request["behavioral_contract"]["mode"] != "exact":
            raise ModelCompileError(
                "approximate whole-model validation requires a declared "
                "metric comparator"
            )
        return compare_exact_role_results(
            request,
            reference_result,
            candidate_result,
            divergence_diagnostic=(
                "candidate conversation output or resident state "
                "diverged from the exact implementation"
            ),
        )

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment["VK_DRIVER_FILES"] = os.pathsep.join(
            str(path) for path in self.vulkan_driver_files
        )
        environment.pop("VK_ICD_FILENAMES", None)
        return environment

    def _conversation_fixture(
        self,
        candidate_id: str,
        relative_path: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        candidate = self.staged_candidate_loader(
            self.candidate_workspace,
            candidate_id,
            self.package_dir,
        )
        store = ExecutorArtifactStore(
            candidate.path,
            label="validation fixture",
            create=False,
        )
        captured = bytearray()
        for chunk in store.iter_file(relative_path):
            captured.extend(chunk)
            if len(captured) > 1_048_576:
                raise ModelCompileError(
                    "conversation validation fixture exceeds 1 MiB"
                )
        try:
            document = json.loads(captured)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelCompileError(
                "conversation validation fixture is not JSON"
            ) from error
        if (
            not isinstance(document, dict)
            or document.get("schema")
            != "nerve.optimizer.validation_conversation.v1"
            or not isinstance(document.get("turns"), list)
            or not document["turns"]
            or any(
                not isinstance(turn, str) or not turn.strip()
                for turn in document["turns"]
            )
            or not isinstance(
                document.get("teacher_forced_assistant_turns"),
                list,
            )
            or len(document["teacher_forced_assistant_turns"])
            != len(document["turns"])
            or any(
                not isinstance(turn, str) or not turn.strip()
                for turn in document[
                    "teacher_forced_assistant_turns"
                ]
            )
        ):
            raise ModelCompileError(
                "conversation validation fixture is invalid"
            )
        return (
            tuple(document["turns"]),
            tuple(document["teacher_forced_assistant_turns"]),
        )


class ResidentWholeModelValidationSession:
    def __init__(
        self,
        *,
        backend: ResidentWholeModelValidationBackend,
        request: ValidationRoleMountRequest,
        transport: ExecutorTransport,
        mount_payload: Json,
        mount_duration_ns: int,
        turns: tuple[str, ...],
        teacher_forced_assistant_turns: tuple[str, ...],
        physical_device_ids: tuple[str, ...],
    ) -> None:
        self.backend = backend
        self.request = request
        self.transport = transport
        self.mount_payload = mount_payload
        self.mount_duration_ns = mount_duration_ns
        self.turns = turns
        self.teacher_forced_assistant_turns = (
            teacher_forced_assistant_turns
        )
        self.physical_device_ids = physical_device_ids
        self.session_nonce = uuid4().hex
        self.closed = False
        self._mount_event = self._event(
            action="mount",
            duration_ns=mount_duration_ns,
            before=request.matched_conditions[
                "idle_device_state_digest"
            ],
            after=mount_payload["mounted_state_digest"],
            released=False,
        )

    @property
    def mount_event(self) -> Json:
        return dict(self._mount_event)

    def execute(
        self,
        request: ValidationRoleExecutionRequest,
    ) -> Json:
        if self.closed:
            raise ModelCompileError(
                "whole-model validation session is closed"
            )
        _require_same_role_request(self.request, request)
        max_output_tokens = request.check["horizon"][
            "output_allowance"
        ]
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ModelCompileError(
                "whole-model validation requires a positive declared "
                "output allowance"
            )
        command = {
            "schema": VALIDATION_EXECUTOR_COMMAND_SCHEMA,
            "command": "execute",
            "request_id": request_id(
                "validation-execute",
                request.to_json(),
            ),
            "turns": list(self.turns),
            "teacher_forced_assistant_turns": list(
                self.teacher_forced_assistant_turns
            ),
            "execution_mode": request.check["controls"].get(
                "execution_mode",
                "conversation",
            ),
            "max_output_tokens": max_output_tokens,
        }
        started = time.monotonic_ns()
        response = validated_validation_response(
            self.transport.request(command),
            expected_request_id=command["request_id"],
            expected_status="completed",
        )
        report = response["payload"]
        validate_validation_execution_payload(report)
        host_execution_ns = max(
            1,
            time.monotonic_ns() - started,
        )
        trace_payloads = {
            "conversation": {"turns": report["turns"]},
            "state": {"state_digest": report["state_digest"]},
            "schedule": {
                "steps": report["steps"],
                "scheduler_steps": report["scheduler_steps"],
                "execution_counters": report[
                    "execution_counters"
                ],
            },
        }
        prefix = (
            f"traces/validation/{self.backend.run_nonce}/"
            f"{self.session_nonce}/{request.check['check_id']}/"
            f"{request.seed}/{request.role}"
        )
        traces = [
            self.backend.trace_store.publish(
                f"{prefix}/{name}.json",
                canonical_json_bytes(payload) + b"\n",
            )
            for name, payload in sorted(trace_payloads.items())
        ]
        document = {
            "schema": VALIDATION_ROLE_RESULT_SCHEMA,
            "result_id": "",
            "plan_id": request.plan_id,
            "check_id": request.check["check_id"],
            "stage": request.check["stage"],
            "seed": request.seed,
            "role": request.role,
            "implementation_id": request.implementation[
                "implementation_id"
            ],
            "status": "completed",
            "output_digest": report["output_digest"],
            "state_digest": report["state_digest"],
            "steps": report["steps"],
            "traces": traces,
            "default_statistics": {
                "execution_path": "resident_whole_model_chat",
                "host_execution_ns": host_execution_ns,
                "device_execution_ns": report["elapsed_ns"],
                "transport_bytes": (
                    len(canonical_json_bytes(command))
                    + len(canonical_json_bytes(response))
                    + 2
                ),
                "scheduler_steps": report["scheduler_steps"],
                "execution_counters": report[
                    "execution_counters"
                ],
            },
            "diagnostics": [],
        }
        document["result_id"] = validation_role_result_id(document)
        return ValidationRoleResult.from_json(document).to_json()

    def close(self) -> Json:
        if self.closed:
            raise ModelCompileError(
                "whole-model validation session closed twice"
            )
        self.closed = True
        command = {
            "schema": VALIDATION_EXECUTOR_COMMAND_SCHEMA,
            "command": "close",
            "request_id": request_id(
                "validation-close",
                {
                    "plan_id": self.request.plan_id,
                    "check_id": self.request.check["check_id"],
                    "seed": self.request.seed,
                    "role": self.request.role,
                    "block_index": self.request.block_index,
                },
            ),
        }
        started = time.monotonic_ns()
        try:
            response = validated_validation_response(
                self.transport.request(command),
                expected_request_id=command["request_id"],
                expected_status="released",
            )
            payload = response["payload"]
            validate_validation_release_payload(
                payload,
                mounted_state_digest=self.mount_payload[
                    "mounted_state_digest"
                ],
                physical_device_ids=self.physical_device_ids,
            )
            self.transport.close()
        except BaseException:
            self.transport.abort()
            raise
        return self._event(
            action="unmount",
            duration_ns=max(1, time.monotonic_ns() - started),
            before=self.mount_payload["mounted_state_digest"],
            after=self.request.matched_conditions[
                "idle_device_state_digest"
            ],
            released=True,
        )

    def _event(
        self,
        *,
        action: str,
        duration_ns: int,
        before: str,
        after: str,
        released: bool,
    ) -> Json:
        document = {
            "schema": VALIDATION_RESIDENCY_EVENT_SCHEMA,
            "event_id": "",
            "plan_id": self.request.plan_id,
            "stage": self.request.stage,
            "check_id": self.request.check["check_id"],
            "seed": self.request.seed,
            "role": self.request.role,
            "implementation_id": self.request.implementation[
                "implementation_id"
            ],
            "block_index": self.request.block_index,
            "action": action,
            "duration_ns": duration_ns,
            "device_state_before_digest": before,
            "device_state_after_digest": after,
            "released": released,
            "default_statistics": {
                "execution_path": "resident_whole_model_chat",
                "physical_device_ids": list(
                    self.physical_device_ids
                ),
                "context_capacity": self.mount_payload[
                    "context_capacity"
                ],
            },
        }
        document["event_id"] = validation_residency_event_id(
            document
        )
        return ValidationResidencyEvent.from_json(document).to_json()


def _require_same_role_request(
    mount: ValidationRoleMountRequest,
    execution: ValidationRoleExecutionRequest,
) -> None:
    if (
        execution.plan_id != mount.plan_id
        or execution.candidate_id != mount.candidate_id
        or execution.check["check_id"]
        != mount.check["check_id"]
        or execution.role != mount.role
        or execution.implementation["implementation_id"]
        != mount.implementation["implementation_id"]
        or execution.seed != mount.seed
        or execution.reset_to_initial_state is not True
    ):
        raise ModelCompileError(
            "whole-model validation changed its mounted role request"
        )


def _physical_device_ids(
    matched_conditions: Json,
) -> tuple[str, ...]:
    devices = matched_conditions.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ModelCompileError(
            "whole-model validation requires declared devices"
        )
    ids = tuple(
        str(device.get("device_id", ""))
        for device in devices
        if isinstance(device, dict)
    )
    if (
        len(ids) != len(devices)
        or any(
            not value.startswith("vulkan-uuid:")
            for value in ids
        )
        or len(set(ids)) != len(ids)
    ):
        raise ModelCompileError(
            "whole-model validation requires unique stable Vulkan devices"
        )
    return ids


def _role_placement(
    matched_conditions: Json,
    check: Json,
    physical_device_ids: tuple[str, ...],
) -> dict[str, str]:
    placement = matched_conditions.get("placement")
    if not isinstance(placement, dict):
        raise ModelCompileError(
            "whole-model validation requires explicit placement"
        )
    resolved = {
        str(component_id): str(device_id)
        for component_id, device_id in placement.items()
    }
    if any(
        not component_id
        or device_id not in physical_device_ids
        for component_id, device_id in resolved.items()
    ):
        raise ModelCompileError(
            "whole-model validation placement is invalid"
        )
    if (
        check["kind"] == "placement"
        and len(physical_device_ids) > 1
    ):
        successor = {
            device_id: physical_device_ids[
                (index + 1) % len(physical_device_ids)
            ]
            for index, device_id in enumerate(physical_device_ids)
        }
        resolved = {
            component_id: successor[device_id]
            for component_id, device_id in resolved.items()
        }
    return resolved
