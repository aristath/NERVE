from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
)
from nerve.representation_optimizer.benchmarking.executor_client import (
    ResidentExecutorClient,
    ResidentExecutorMountSpec,
    ResidentExecutorSession,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    ExecutorTransport,
)
from nerve.representation_optimizer.contracts import canonical_json_bytes
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
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
from nerve.representation_optimizer.validation.protocols import (
    ValidationComparisonRequest,
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)


class ResidentComponentValidationBackend:
    """Component validation through the same resident executor as benchmarks."""

    def __init__(
        self,
        *,
        executor_client: ResidentExecutorClient,
        trace_store: ExecutorArtifactStore,
        run_nonce: str,
    ) -> None:
        self.executor_client = executor_client
        self.trace_store = trace_store
        self.run_nonce = run_nonce
        self._stage_active = False
        self._stage_transport: ExecutorTransport | None = None
        self._stage_cancel_requested: Callable[[], bool] | None = None
        self._stage_physical_device_ids: set[str] = set()

    @contextmanager
    def validation_stage(
        self,
        stage: str,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        if self._stage_active:
            raise ModelCompileError(
                "component validation stage cannot be nested"
            )
        if not isinstance(stage, str) or not stage:
            raise ModelCompileError(
                "component validation stage must not be empty"
            )
        self._stage_active = True
        self._stage_cancel_requested = cancel_requested
        try:
            yield
        except BaseException:
            if self._stage_transport is not None:
                self._stage_transport.abort()
            raise
        else:
            if self._stage_transport is not None:
                self.executor_client.shutdown_transport(
                    self._stage_transport,
                    tuple(sorted(self._stage_physical_device_ids)),
                )
        finally:
            self._stage_transport = None
            self._stage_cancel_requested = None
            self._stage_physical_device_ids.clear()
            self._stage_active = False

    def open_session(
        self,
        request: ValidationRoleMountRequest,
    ) -> ResidentComponentValidationSession:
        check = request.check
        if check["regime"]["execution_scope"] != "component":
            raise ModelCompileError(
                "component validation received a whole-model check"
            )
        component_id = _required_text(
            check["controls"],
            "component_id",
        )
        physical_node_id = _required_text(
            check["controls"],
            "physical_node_id",
        )
        phase = _required_text(check["controls"], "phase")
        if phase not in {"decode", "prefill"}:
            raise ModelCompileError(
                "component validation check must declare one execution phase"
            )
        physical_device_id = _component_device(
            request.matched_conditions,
            component_id,
        )
        if self._stage_active:
            self._stage_physical_device_ids.add(physical_device_id)
        maximum_wait_ns = _positive_integer(
            request.matched_conditions.get("controls", {}).get(
                "maximum_quantum_wait_ns"
            ),
            "validation maximum quantum wait",
        )
        width = _positive_integer(
            check["regime"]["activation_batch_width"],
            "validation activation batch width",
        )
        mount_spec = ResidentExecutorMountSpec(
            implementation_id=request.implementation[
                "implementation_id"
            ],
            component_id=component_id,
            physical_node_id=physical_node_id,
            phase=phase,
            activation_batch_width=width,
            physical_device_id=physical_device_id,
            dynamic_state_capacity_activations=max(
                width,
                int(check["regime"]["state_size"]),
                1,
            ),
            maximum_quantum_wait_ns=maximum_wait_ns,
            request_identity=request.to_json(),
            capture_output_values=_optional_boolean(
                check["controls"],
                "capture_output_values",
            ),
            cancel_requested=request.cancel_requested,
        )
        if self._stage_active:
            if request.cancel_requested is not self._stage_cancel_requested:
                raise ModelCompileError(
                    "component validation request changed stage cancellation authority"
                )
            if self._stage_transport is None:
                self._stage_transport = self.executor_client.start_transport()
            executor_session = self.executor_client.open_on_transport(
                mount_spec,
                self._stage_transport,
            )
        else:
            executor_session = self.executor_client.open(mount_spec)
        return ResidentComponentValidationSession(
            backend=self,
            request=request,
            executor_session=executor_session,
        )

    def compare_results(
        self,
        request: ValidationComparisonRequest,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        if request.behavioral_contract["mode"] == "exact":
            return compare_exact_role_results(
                request.to_json(),
                reference_result,
                candidate_result,
                divergence_diagnostic=(
                    "candidate component output or transient state "
                    "diverged from the exact implementation"
                ),
            )
        return _compare_approximate_output_values(
            request.to_json(),
            _output_values(self.trace_store, reference_result),
            _output_values(self.trace_store, candidate_result),
        )

class ResidentComponentValidationSession:
    def __init__(
        self,
        *,
        backend: ResidentComponentValidationBackend,
        request: ValidationRoleMountRequest,
        executor_session: ResidentExecutorSession,
    ) -> None:
        self.backend = backend
        self.request = request
        self.executor_session = executor_session
        self.mount_payload = executor_session.mount_payload
        self.session_nonce = uuid4().hex
        self.closed = False
        self._mount_event = self._event(
            action="mount",
            duration_ns=executor_session.host_mount_ns,
            before=request.matched_conditions[
                "idle_device_state_digest"
            ],
            after=self.mount_payload["mounted_state_digest"],
            released=False,
        )

    @property
    def mount_event(self) -> Json:
        return dict(self._mount_event)

    def execute(self, request: ValidationRoleExecutionRequest) -> Json:
        if self.closed:
            raise ModelCompileError(
                "component validation session is closed"
            )
        if (
            request.plan_id != self.request.plan_id
            or request.candidate_id != self.request.candidate_id
            or request.check["check_id"]
            != self.request.check["check_id"]
            or request.role != self.request.role
            or request.implementation["implementation_id"]
            != self.request.implementation["implementation_id"]
            or request.seed != self.request.seed
            or request.reset_to_initial_state is not True
        ):
            raise ModelCompileError(
                "component validation changed its mounted role request"
            )
        horizon = request.check["horizon"]
        if horizon["completion_condition"] != "minimum_steps":
            raise ModelCompileError(
                "component validation requires a minimum-steps completion "
                "condition"
            )
        useful_units = _positive_integer(
            horizon["minimum_steps"],
            "component validation minimum steps",
        )
        if horizon["unit"] != "component_activations":
            raise ModelCompileError(
                "component validation requires a component_activations "
                "horizon"
            )
        execution = self.executor_session.execute(
            useful_units=useful_units,
            seed=request.seed,
            request_identity=request.to_json(),
        )
        report = execution.report
        trace_payloads = {
            "output": {
                "output_digest": report["output_digest"],
                "op": report["op"],
                **(
                    {
                        "output_values_f32_le_hex": report[
                            "output_values_f32_le_hex"
                        ]
                    }
                    if report.get("output_values_f32_le_hex") is not None
                    else {}
                ),
            },
            "state": {"state_digest": report["state_digest"]},
            "schedule": {
                "throughput_windows": report["throughput_windows"],
                "physical_dispatch_count": report[
                    "physical_dispatch_count"
                ],
                "queue_submission_count": report[
                    "queue_submission_count"
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
            "steps": useful_units,
            "horizon_completion": {
                "condition": "minimum_steps",
                "satisfied": True,
                "observed_steps": useful_units,
                "minimum_steps": useful_units,
                "expected_turns": None,
                "completed_turns": None,
                "stop_reasons": [],
            },
            "traces": traces,
            "default_statistics": {
                "execution_path": "resident_targeted_component",
                "host_execution_ns": execution.host_execution_ns,
                "device_execution_ns": report["execution_ns"],
                "transport_bytes": execution.transport_bytes,
                "physical_dispatch_count": report[
                    "physical_dispatch_count"
                ],
                "queue_submission_count": report[
                    "queue_submission_count"
                ],
                "synchronization_wait_count": report[
                    "synchronization_wait_count"
                ],
                "synchronization_wait_ns": report[
                    "synchronization_wait_ns"
                ],
                "queue_wait_ns": report["queue_wait_ns"],
            },
            "diagnostics": [],
        }
        document["result_id"] = validation_role_result_id(document)
        return ValidationRoleResult.from_json(document).to_json()

    def close(self) -> Json:
        if self.closed:
            raise ModelCompileError(
                "component validation session closed twice"
            )
        self.closed = True
        release = self.executor_session.close(
            request_identity={
                "plan_id": self.request.plan_id,
                "check_id": self.request.check["check_id"],
                "role": self.request.role,
                "seed": self.request.seed,
                "block_index": self.request.block_index,
            }
        )
        return self._event(
            action="unmount",
            duration_ns=release.host_release_ns,
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
                "execution_path": "resident_targeted_component",
                "physical_device_id": self.mount_payload[
                    "physical_device_id"
                ],
                "logical_device_id": self.mount_payload[
                    "logical_device_id"
                ],
                "resident_parameter_bytes": self.mount_payload[
                    "resident_parameter_bytes"
                ],
                "resident_transient_bytes": self.mount_payload[
                    "resident_transient_bytes"
                ],
                "resident_asset_pool_bytes": self.mount_payload[
                    "resident_asset_pool_bytes"
                ],
                "resident_asset_pool_buffers": self.mount_payload[
                    "resident_asset_pool_buffers"
                ],
                "resident_asset_pool_hits": self.mount_payload[
                    "resident_asset_pool_hits"
                ],
                "resident_asset_pool_misses": self.mount_payload[
                    "resident_asset_pool_misses"
                ],
            },
        }
        document["event_id"] = validation_residency_event_id(document)
        return ValidationResidencyEvent.from_json(document).to_json()


def _output_values(
    trace_store: ExecutorArtifactStore,
    result: Json,
) -> tuple[float, ...]:
    refs = [
        trace
        for trace in result["traces"]
        if str(trace["path"]).endswith("/output.json")
    ]
    if len(refs) != 1:
        raise ModelCompileError(
            "approximate component validation requires one output trace"
        )
    reference = refs[0]
    payload = b"".join(trace_store.iter_file(reference["path"]))
    if staged_artifact_digest(payload) != reference["digest"]:
        raise ModelCompileError(
            "approximate component output trace digest disagrees"
        )
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            "approximate component output trace is not valid JSON"
        ) from error
    encoded = document.get("output_values_f32_le_hex")
    if (
        not isinstance(encoded, str)
        or not encoded
        or len(encoded) % 8 != 0
    ):
        raise ModelCompileError(
            "approximate component output trace has no F32 values"
        )
    try:
        raw = bytes.fromhex(encoded)
    except ValueError as error:
        raise ModelCompileError(
            "approximate component output trace has invalid hex"
        ) from error
    values = tuple(value[0] for value in struct.iter_unpack("<f", raw))
    if not values or any(not math.isfinite(value) for value in values):
        raise ModelCompileError(
            "approximate component output trace contains non-finite values"
        )
    return values


def _compare_approximate_output_values(
    request: Json,
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
) -> Json:
    if len(reference) != len(candidate):
        raise ModelCompileError(
            "approximate component implementations emitted different output widths"
        )
    squared_reference = sum(value * value for value in reference)
    squared_error = sum(
        (candidate_value - reference_value) ** 2
        for reference_value, candidate_value in zip(
            reference,
            candidate,
            strict=True,
        )
    )
    normalized_rms_error = math.sqrt(
        squared_error / max(squared_reference, 1e-24)
    )
    top_count = min(32, len(reference))
    reference_top = {
        index
        for _, index in sorted(
            ((value, index) for index, value in enumerate(reference)),
            reverse=True,
        )[:top_count]
    }
    candidate_top = {
        index
        for _, index in sorted(
            ((value, index) for index, value in enumerate(candidate)),
            reverse=True,
        )[:top_count]
    }
    measured = {
        "normalized_rms_logit_error": (
            normalized_rms_error,
            "relative_rms",
        ),
        "top_32_mismatch_rate": (
            1.0 - len(reference_top & candidate_top) / top_count,
            "fraction",
        ),
        "top_1_mismatch_rate": (
            float(
                max(range(len(reference)), key=reference.__getitem__)
                != max(range(len(candidate)), key=candidate.__getitem__)
            ),
            "fraction",
        ),
    }
    metrics = []
    for name in request["check"]["metrics"]:
        try:
            error, unit = measured[name]
        except KeyError as failure:
            raise ModelCompileError(
                f"approximate component comparator cannot measure {name!r}"
            ) from failure
        metrics.append(
            {
                "name": name,
                "reference_value": 0.0,
                "candidate_value": error,
                "error": error,
                "unit": unit,
            }
        )
    return {"metrics": metrics, "diagnostics": []}


def _component_device(
    matched_conditions: Json,
    component_id: str,
) -> str:
    placement = matched_conditions.get("placement")
    if not isinstance(placement, dict):
        raise ModelCompileError(
            "component validation requires explicit placement"
        )
    device_id = _required_text(placement, component_id)
    devices = matched_conditions.get("devices")
    if (
        not isinstance(devices, list)
        or device_id
        not in {
            device.get("device_id")
            for device in devices
            if isinstance(device, dict)
        }
    ):
        raise ModelCompileError(
            "component validation placement references an undeclared device"
        )
    return device_id


def _required_text(document: Json, field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ModelCompileError(
            f"component validation {field} must be non-empty text"
        )
    return value


def _positive_integer(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ModelCompileError(f"{label} must be a positive integer")
    return value


def _optional_boolean(document: Json, field: str) -> bool:
    value = document.get(field, False)
    if not isinstance(value, bool):
        raise ModelCompileError(
            f"component validation control {field!r} must be boolean"
        )
    return value
