from __future__ import annotations

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
from nerve.representation_optimizer.validation.protocols import (
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
        executor_session = self.executor_client.open(
            ResidentExecutorMountSpec(
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
            )
        )
        return ResidentComponentValidationSession(
            backend=self,
            request=request,
            executor_session=executor_session,
        )

    def compare_results(
        self,
        request: Json,
        reference_result: Json,
        candidate_result: Json,
    ) -> Json:
        if request["behavioral_contract"]["mode"] != "exact":
            raise ModelCompileError(
                "approximate component validation requires a declared "
                "metric comparator"
            )
        return compare_exact_role_results(
            request,
            reference_result,
            candidate_result,
            divergence_diagnostic=(
                "candidate component output or transient state "
                "diverged from the exact implementation"
            ),
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
        useful_units = _positive_integer(
            request.check["horizon"]["minimum_steps"],
            "component validation minimum steps",
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
            },
        }
        document["event_id"] = validation_residency_event_id(document)
        return ValidationResidencyEvent.from_json(document).to_json()


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
