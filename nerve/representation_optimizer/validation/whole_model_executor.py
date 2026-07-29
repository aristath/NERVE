from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from nerve.compilation import (
    Json,
    ModelCompileError,
    check_compile_cancelled,
)
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
from nerve.representation_optimizer.validation.conversation_semantics import (
    compare_semantic_conversations,
    validate_semantic_expectations,
)
from nerve.representation_optimizer.validation.executor_protocol import (
    VALIDATION_EXECUTOR_COMMAND_SCHEMA,
    validate_validation_execution_payload,
    validate_validation_mount_payload,
    validate_validation_progress,
    validate_validation_release_payload,
    validate_validation_shutdown_payload,
    validated_validation_response,
)
from nerve.representation_optimizer.validation.protocols import (
    ValidationComparisonRequest,
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)


class _ProgressJournal:
    """Live JSONL progress with atomic immutable final publication."""

    def __init__(
        self,
        store: ExecutorArtifactStore,
        relative_path: str,
    ) -> None:
        self.store = store
        self.relative_path = relative_path
        self._payload = bytearray()
        self._reference: Json | None = None
        self._partial = store.confined_path(
            f"{relative_path}.partial-{uuid4().hex}"
        )
        self._partial.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self._partial, flags, 0o644)
        self._stream = os.fdopen(descriptor, "wb")

    @property
    def byte_count(self) -> int:
        return len(self._payload)

    def append(self, event: Json) -> None:
        if self._reference is not None or self._stream.closed:
            raise ModelCompileError(
                "validation progress journal is already finalized"
            )
        line = canonical_json_bytes(event) + b"\n"
        self._stream.write(line)
        self._stream.flush()
        self._payload.extend(line)

    def finalize(self, *, status: str) -> Json:
        if self._reference is not None:
            return dict(self._reference)
        self.append(
            {
                "schema": "nerve.optimizer.validation_progress_terminal.v1",
                "status": status,
            }
        )
        os.fsync(self._stream.fileno())
        self._stream.close()
        reference = self.store.publish(
            self.relative_path,
            bytes(self._payload),
        )
        self._partial.unlink()
        self._reference = reference
        return dict(reference)


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
        self._active_stage: str | None = None
        self._transport: ExecutorTransport | None = None
        self._role_is_mounted = False
        self._stage_physical_device_ids: tuple[str, ...] | None = None
        self.partial_progress_refs: list[Json] = []

    @contextmanager
    def validation_stage(
        self,
        stage: str,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        if self._active_stage is not None:
            raise ModelCompileError(
                "whole-model validation stage ownership is not reentrant"
            )
        self._active_stage = stage
        try:
            yield
        finally:
            try:
                self._release_stage_executor()
            finally:
                self._active_stage = None

    def open_session(
        self,
        request: ValidationRoleMountRequest,
    ) -> ResidentWholeModelValidationSession:
        if self._active_stage != request.stage:
            raise ModelCompileError(
                "whole-model validation role requires an active matching "
                "validation stage"
            )
        if self._role_is_mounted:
            raise ModelCompileError(
                "whole-model validation cannot mount overlapping roles"
            )
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
        if self._stage_physical_device_ids is None:
            self._stage_physical_device_ids = physical_device_ids
        elif self._stage_physical_device_ids != physical_device_ids:
            raise ModelCompileError(
                "one whole-model validation stage cannot change its "
                "physical device topology"
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
        execution_mode = check["controls"].get(
            "execution_mode",
            "conversation",
        )
        speculative_draft_tokens = check["controls"].get(
            "speculative_draft_tokens",
            0,
        )
        if (
            isinstance(speculative_draft_tokens, bool)
            or not isinstance(speculative_draft_tokens, int)
            or speculative_draft_tokens < 0
        ):
            raise ModelCompileError(
                "whole-model validation speculative draft tokens must be "
                "a non-negative integer"
            )
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
                {
                    "kind": "declared",
                    "activations": context_size,
                }
                if context_size > 0
                else {"kind": "fixture_exact"}
            ),
            "validation_turns": list(turns),
            "teacher_forced_assistant_turns": list(
                teacher_forced_assistant_turns
            ),
            "execution_mode": execution_mode,
            "speculative_draft_tokens": speculative_draft_tokens,
            "random_seed": request.seed,
            "sampler_config": dict(
                check["controls"].get("sampler", {})
            ),
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
        transport = self._stage_executor()
        self._role_is_mounted = True
        started = time.monotonic_ns()
        try:
            # Accelerator commands are cancellation quanta. Refuse new work
            # at the boundary, but never SIGKILL an executor after a command
            # may have submitted Vulkan work.
            check_compile_cancelled(request.cancel_requested)
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
            self._abort_stage_executor()
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

    def role_released(self, transport: ExecutorTransport) -> None:
        if transport is not self._transport or not self._role_is_mounted:
            self._abort_stage_executor()
            raise ModelCompileError(
                "whole-model validation released an unowned role"
            )
        self._role_is_mounted = False

    def role_failed(self, transport: ExecutorTransport) -> None:
        if transport is self._transport:
            self._abort_stage_executor()

    def _stage_executor(self) -> ExecutorTransport:
        if self._active_stage is None:
            raise ModelCompileError(
                "whole-model validation executor has no active stage"
            )
        if self._transport is None:
            self._transport = self.executor_factory(
                self.executor_command,
                self._environment(),
            )
        return self._transport

    def _release_stage_executor(self) -> None:
        transport = self._transport
        if transport is None:
            return
        if self._role_is_mounted:
            self._abort_stage_executor()
            raise ModelCompileError(
                "whole-model validation stage ended with a mounted role"
            )
        physical_device_ids = self._stage_physical_device_ids
        if physical_device_ids is None:
            self._abort_stage_executor()
            raise ModelCompileError(
                "whole-model validation stage has no mounted device topology"
            )
        command = {
            "schema": VALIDATION_EXECUTOR_COMMAND_SCHEMA,
            "command": "shutdown",
            "request_id": request_id(
                "validation-shutdown",
                {
                    "run_nonce": self.run_nonce,
                    "stage": self._active_stage,
                    "physical_device_ids": list(physical_device_ids),
                },
            ),
        }
        try:
            response = validated_validation_response(
                transport.request(command),
                expected_request_id=command["request_id"],
                expected_status="shutdown_complete",
            )
            validate_validation_shutdown_payload(
                response["payload"],
                physical_device_ids=physical_device_ids,
            )
            # Cancellation stops new optimization work.  It must not turn an
            # acknowledged, serialized accelerator release into SIGKILL.
            transport.close()
        except BaseException:
            transport.abort()
            raise
        finally:
            self._transport = None
            self._stage_physical_device_ids = None

    def _abort_stage_executor(self) -> None:
        transport = self._transport
        self._transport = None
        self._role_is_mounted = False
        self._stage_physical_device_ids = None
        if transport is not None:
            transport.abort()

    def compare_results(
        self,
        request: ValidationComparisonRequest,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        if request.behavioral_contract["mode"] != "exact":
            raise ModelCompileError(
                "approximate whole-model validation requires a declared "
                "metric comparator"
            )
        comparison = request.check["comparison"]
        if comparison == {
            "output_mode": "fixture_semantics",
            "state_mode": "trajectory_local",
        }:
            fixture = self._conversation_fixture_document(
                request.candidate_id,
                request.check["input"]["path"],
            )
            return compare_semantic_conversations(
                request.to_json(),
                fixture,
                self._trace_document(
                    reference_result,
                    "conversation.json",
                ),
                self._trace_document(
                    candidate_result,
                    "conversation.json",
                ),
            )
        if comparison != {
            "output_mode": "exact_digest",
            "state_mode": "exact_digest",
        }:
            raise ModelCompileError(
                "whole-model validation received an unsupported comparison "
                "contract"
            )
        return compare_exact_role_results(
            request.to_json(),
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
        document = self._conversation_fixture_document(
            candidate_id,
            relative_path,
        )
        return (
            tuple(document["turns"]),
            tuple(document["teacher_forced_assistant_turns"]),
        )

    def _conversation_fixture_document(
        self,
        candidate_id: str,
        relative_path: str,
    ) -> Json:
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
        if document.get("semantic_expectations") is not None:
            validate_semantic_expectations(
                document,
                turn_count=len(document["turns"]),
            )
        return document

    def _trace_document(
        self,
        result: Json,
        filename: str,
    ) -> Json:
        matches = [
            trace["path"]
            for trace in result["traces"]
            if trace["path"].endswith(f"/{filename}")
        ]
        if len(matches) != 1:
            raise ModelCompileError(
                f"whole-model validation result does not contain one "
                f"{filename!r} trace"
            )
        captured = bytearray()
        for chunk in self.trace_store.iter_file(matches[0]):
            captured.extend(chunk)
            if len(captured) > 16 * 1024 * 1024:
                raise ModelCompileError(
                    "whole-model validation trace exceeds 16 MiB"
                )
        try:
            document = json.loads(captured)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelCompileError(
                "whole-model validation trace is not JSON"
            ) from error
        if not isinstance(document, dict):
            raise ModelCompileError(
                "whole-model validation trace is not an object"
            )
        return document


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
        execution_mode = request.check["controls"].get(
            "execution_mode",
            "conversation",
        )
        if execution_mode == "conversation" and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ModelCompileError(
                "free-running whole-model validation requires a positive "
                "declared output allowance"
            )
        if execution_mode != "conversation" and (
            max_output_tokens is not None
            and (
                isinstance(max_output_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or max_output_tokens <= 0
            )
        ):
            raise ModelCompileError(
                "whole-model validation output allowance is invalid"
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
            "execution_mode": execution_mode,
            "step_unit": request.check["horizon"]["unit"],
            "max_output_tokens": max_output_tokens,
        }
        started = time.monotonic_ns()
        prefix = (
            f"traces/validation/{self.backend.run_nonce}/"
            f"{self.session_nonce}/{request.check['check_id']}/"
            f"{request.seed}/{request.role}"
        )
        progress = _ProgressJournal(
            self.backend.trace_store,
            f"{prefix}/progress.jsonl",
        )

        def progress_received(event: Json) -> None:
            validate_validation_progress(
                event,
                expected_request_id=command["request_id"],
                turn_count=len(self.turns),
            )
            progress.append(event)

        try:
            check_compile_cancelled(self.request.cancel_requested)
            response = validated_validation_response(
                self.transport.request(
                    command,
                    progress_received=progress_received,
                ),
                expected_request_id=command["request_id"],
                expected_status="completed",
            )
        except BaseException:
            self.backend.partial_progress_refs.append(
                progress.finalize(status="failed")
            )
            self.backend.role_failed(self.transport)
            raise
        progress_ref = progress.finalize(status="completed")
        report = response["payload"]
        validate_validation_execution_payload(
            report,
            expected_step_unit=request.check["horizon"]["unit"],
            expected_turns=self.turns,
        )
        horizon = request.check["horizon"]
        completion_condition = horizon["completion_condition"]
        stop_reasons = [
            str(turn["stop_reason"])
            for turn in report["turns"]
        ]
        if completion_condition == "minimum_steps":
            horizon_completion = {
                "condition": completion_condition,
                "satisfied": report["steps"] >= horizon["minimum_steps"],
                "observed_steps": report["steps"],
                "minimum_steps": horizon["minimum_steps"],
                "expected_turns": None,
                "completed_turns": None,
                "stop_reasons": [],
            }
        elif completion_condition == "all_fixture_turns":
            if any(reason != "fixture_completed" for reason in stop_reasons):
                raise ModelCompileError(
                    "teacher-forced validation did not complete its declared "
                    "fixture turns"
                )
            horizon_completion = {
                "condition": completion_condition,
                "satisfied": len(report["turns"]) == len(self.turns),
                "observed_steps": report["steps"],
                "minimum_steps": None,
                "expected_turns": len(self.turns),
                "completed_turns": len(report["turns"]),
                "stop_reasons": stop_reasons,
            }
        elif (
            completion_condition
            == "semantic_stop_or_allowance_per_turn"
        ):
            if any(
                reason not in {"eos", "output_allowance"}
                for reason in stop_reasons
            ):
                raise ModelCompileError(
                    "free-running validation did not reach an ordinary "
                    "semantic stop or its declared output allowance"
                )
            horizon_completion = {
                "condition": completion_condition,
                "satisfied": len(report["turns"]) == len(self.turns),
                "observed_steps": report["steps"],
                "minimum_steps": None,
                "expected_turns": len(self.turns),
                "completed_turns": len(report["turns"]),
                "stop_reasons": stop_reasons,
            }
        else:
            raise ModelCompileError(
                "whole-model validation completion condition is unsupported"
            )
        host_execution_ns = max(
            1,
            time.monotonic_ns() - started,
        )
        trace_payloads = {
            "conversation": {"turns": report["turns"]},
            "state": {"state_digest": report["state_digest"]},
            "schedule": {
                "steps": report["steps"],
                "step_unit": report["step_unit"],
                "scheduler_steps": report["scheduler_steps"],
                "execution_counters": report[
                    "execution_counters"
                ],
                "turn_statistics": [
                    {
                        "turn_index": turn["turn_index"],
                        "generated_tokens": len(
                            turn["generated_token_ids"]
                        ),
                        "elapsed_ns": turn["elapsed_ns"],
                        "scheduler_steps": turn["scheduler_steps"],
                        "execution_counters": dict(
                            turn["execution_counters"]
                        ),
                        "speculative": dict(turn["speculative"]),
                        "resident_feedback": dict(
                            turn["resident_feedback"]
                        ),
                        "transport": dict(turn["transport"]),
                    }
                    for turn in report["turns"]
                ],
            },
        }
        traces = sorted(
            [
                *(
                    self.backend.trace_store.publish(
                        f"{prefix}/{name}.json",
                        canonical_json_bytes(payload) + b"\n",
                    )
                    for name, payload in sorted(
                        trace_payloads.items()
                    )
                ),
                progress_ref,
            ],
            key=lambda reference: reference["path"],
        )
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
            "horizon_completion": horizon_completion,
            "traces": traces,
            "default_statistics": {
                "execution_path": "resident_whole_model_chat",
                "host_execution_ns": host_execution_ns,
                "device_execution_ns": report["elapsed_ns"],
                "transport_bytes": (
                    len(canonical_json_bytes(command))
                    + len(canonical_json_bytes(response))
                    + progress.byte_count
                    + 2
                ),
                "scheduler_steps": report["scheduler_steps"],
                "execution_counters": report[
                    "execution_counters"
                ],
                "turn_statistics": trace_payloads["schedule"][
                    "turn_statistics"
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
            self.backend.role_released(self.transport)
        except BaseException:
            self.backend.role_failed(self.transport)
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
