from __future__ import annotations

import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

from nerve.compilation import Json, ModelCompileError, check_compile_cancelled
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    StagedCandidateLoader,
    resolve_candidate_mount,
)
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    EXECUTOR_COMMAND_SCHEMA,
    positive_integer,
    request_id,
    required_digest,
    required_object,
    required_text,
    validated_response,
    validated_windows,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    ExecutorFactory,
    ExecutorTransport,
)
from nerve.representation_optimizer.contracts import canonical_json_bytes


@dataclass(frozen=True)
class ResidentExecutorMountSpec:
    implementation_id: str
    component_id: str
    physical_node_id: str
    phase: str
    activation_batch_width: int
    physical_device_id: str
    dynamic_state_capacity_activations: int
    maximum_quantum_wait_ns: int
    request_identity: Json
    cancel_requested: Callable[[], bool] | None = None


@dataclass(frozen=True)
class ResidentExecutorExecution:
    report: Json
    host_execution_ns: int
    transport_bytes: int


@dataclass(frozen=True)
class ResidentExecutorRelease:
    payload: Json
    host_release_ns: int


class ResidentExecutorClient:
    """Shared client for the ordinary resident targeted-component runtime."""

    def __init__(
        self,
        *,
        package_manifest: Path,
        candidate_workspace: Path,
        executor_command: tuple[str, ...],
        vulkan_driver_files: tuple[Path, ...],
        executor_factory: ExecutorFactory,
        staged_candidate_loader: StagedCandidateLoader,
    ) -> None:
        self.package_manifest = package_manifest
        self.package_dir = package_manifest.parent
        self.candidate_workspace = candidate_workspace
        self.executor_command = executor_command
        self.vulkan_driver_files = vulkan_driver_files
        self.executor_factory = executor_factory
        self.staged_candidate_loader = staged_candidate_loader

    def environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment["VK_DRIVER_FILES"] = os.pathsep.join(
            str(path) for path in self.vulkan_driver_files
        )
        environment.pop("VK_ICD_FILENAMES", None)
        return environment

    def open(
        self,
        spec: ResidentExecutorMountSpec,
    ) -> ResidentExecutorSession:
        if spec.phase not in {"decode", "prefill"}:
            raise ModelCompileError(
                "resident executor phase must be decode or prefill"
            )
        for value, label in (
            (spec.activation_batch_width, "activation batch width"),
            (
                spec.dynamic_state_capacity_activations,
                "dynamic-state capacity",
            ),
            (spec.maximum_quantum_wait_ns, "maximum quantum wait"),
        ):
            positive_integer(value, f"resident executor {label}")
        candidate_id, candidate_root = resolve_candidate_mount(
            implementation_id=spec.implementation_id,
            workspace_root=self.candidate_workspace,
            package_dir=self.package_dir,
            loader=self.staged_candidate_loader,
        )
        logical_device_id = (
            "optimizer:"
            + sha256(spec.physical_device_id.encode("utf-8")).hexdigest()[:16]
        )
        command = {
            "schema": EXECUTOR_COMMAND_SCHEMA,
            "command": "mount",
            "request_id": request_id("mount", spec.request_identity),
            "package_manifest": str(self.package_manifest),
            "candidate_root": (
                None if candidate_root is None else str(candidate_root)
            ),
            "candidate_id": candidate_id,
            "component_id": spec.component_id,
            "physical_node_id": spec.physical_node_id,
            "phase": spec.phase,
            "activation_batch_width": spec.activation_batch_width,
            "logical_device_id": logical_device_id,
            "physical_device_id": spec.physical_device_id,
            "dynamic_state_capacity_activations": (
                spec.dynamic_state_capacity_activations
            ),
            "maximum_quantum_wait_ns": spec.maximum_quantum_wait_ns,
        }
        transport = self.executor_factory(
            self.executor_command,
            self.environment(),
        )
        started = time.monotonic_ns()
        try:
            response = validated_response(
                transport.request(
                    command,
                    cancel_requested=spec.cancel_requested,
                ),
                expected_request_id=command["request_id"],
                expected_status="mounted",
            )
            payload = required_object(response, "payload")
            _validate_mount_payload(
                payload,
                spec=spec,
                candidate_id=candidate_id,
                logical_device_id=logical_device_id,
            )
            return ResidentExecutorSession(
                transport=transport,
                spec=spec,
                candidate_id=candidate_id,
                logical_device_id=logical_device_id,
                mount_payload=payload,
                host_mount_ns=max(1, time.monotonic_ns() - started),
            )
        except BaseException:
            transport.abort()
            raise


class ResidentExecutorSession:
    def __init__(
        self,
        *,
        transport: ExecutorTransport,
        spec: ResidentExecutorMountSpec,
        candidate_id: str | None,
        logical_device_id: str,
        mount_payload: Json,
        host_mount_ns: int,
    ) -> None:
        self.transport = transport
        self.spec = spec
        self.candidate_id = candidate_id
        self.logical_device_id = logical_device_id
        self.mount_payload = mount_payload
        self.host_mount_ns = host_mount_ns
        self.closed = False

    def execute(
        self,
        *,
        useful_units: int,
        seed: int,
        request_identity: Json,
    ) -> ResidentExecutorExecution:
        if self.closed:
            raise ModelCompileError("resident executor session is closed")
        positive_integer(useful_units, "resident executor useful units")
        command = {
            "schema": EXECUTOR_COMMAND_SCHEMA,
            "command": "execute",
            "request_id": request_id("execute", request_identity),
            "useful_units": useful_units,
            "seed": seed,
        }
        started = time.monotonic_ns()
        response = validated_response(
            self.transport.request(
                command,
                cancel_requested=self.spec.cancel_requested,
            ),
            expected_request_id=command["request_id"],
            expected_status="completed",
        )
        host_execution_ns = max(1, time.monotonic_ns() - started)
        report = required_object(response, "payload")
        _validate_execution_report(
            report,
            spec=self.spec,
            useful_units=useful_units,
        )
        return ResidentExecutorExecution(
            report=report,
            host_execution_ns=host_execution_ns,
            transport_bytes=(
                len(canonical_json_bytes(command))
                + len(canonical_json_bytes(response))
                + 2
            ),
        )

    def close(self, *, request_identity: Json) -> ResidentExecutorRelease:
        if self.closed:
            raise ModelCompileError("resident executor session closed twice")
        self.closed = True
        try:
            check_compile_cancelled(self.spec.cancel_requested)
        except BaseException:
            self.transport.abort()
            raise
        command = {
            "schema": EXECUTOR_COMMAND_SCHEMA,
            "command": "close",
            "request_id": request_id("close", request_identity),
        }
        started = time.monotonic_ns()
        try:
            response = validated_response(
                self.transport.request(
                    command,
                    cancel_requested=self.spec.cancel_requested,
                ),
                expected_request_id=command["request_id"],
                expected_status="released",
            )
            payload = required_object(response, "payload")
            if (
                payload.get("released") is not True
                or payload.get("mounted_state_digest")
                != self.mount_payload["mounted_state_digest"]
            ):
                raise ModelCompileError(
                    "resident executor did not prove release of its mounted state"
                )
            self.transport.close(
                cancel_requested=self.spec.cancel_requested,
            )
        except BaseException:
            self.transport.abort()
            raise
        return ResidentExecutorRelease(
            payload=payload,
            host_release_ns=max(1, time.monotonic_ns() - started),
        )


def _validate_mount_payload(
    payload: Json,
    *,
    spec: ResidentExecutorMountSpec,
    candidate_id: str | None,
    logical_device_id: str,
) -> None:
    expected = {
        "candidate_id": candidate_id,
        "component_id": spec.component_id,
        "physical_node_id": spec.physical_node_id,
        "logical_device_id": logical_device_id,
        "physical_device_id": spec.physical_device_id,
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ModelCompileError(
            "resident executor mounted different runtime conditions"
        )
    for field in ("package_id", "device_name"):
        required_text(payload, field)
    required_digest(payload, "mounted_state_digest")
    for field in (
        "mount_duration_ns",
        "resident_parameter_bytes",
        "resident_transient_bytes",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelCompileError(
                f"executor mount {field} must be a non-negative integer"
            )


def _validate_execution_report(
    report: Json,
    *,
    spec: ResidentExecutorMountSpec,
    useful_units: int,
) -> None:
    if (
        required_text(report, "component_id") != spec.component_id
        or required_text(report, "node_id") != spec.physical_node_id
        or required_text(report, "phase") != spec.phase
        or positive_integer(
            report.get("activation_batch_width"),
            "executor report activation_batch_width",
        )
        != spec.activation_batch_width
        or positive_integer(
            report.get("useful_units"),
            "executor report useful_units",
        )
        != useful_units
    ):
        raise ModelCompileError(
            "resident executor completed different targeted component work"
        )
    required_text(report, "op")
    required_digest(report, "output_digest")
    required_digest(report, "state_digest")
    positive_integer(report.get("execution_ns"), "executor report execution_ns")
    validated_windows(report.get("throughput_windows"), useful_units)
    for field in (
        "physical_dispatch_count",
        "queue_submission_count",
        "synchronization_wait_count",
    ):
        positive_integer(report.get(field), f"executor report {field}")
    for field in (
        "resident_parameter_bytes",
        "resident_transient_bytes",
        "synchronization_wait_ns",
        "queue_wait_ns",
    ):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelCompileError(
                f"executor report {field} must be a non-negative integer"
            )
