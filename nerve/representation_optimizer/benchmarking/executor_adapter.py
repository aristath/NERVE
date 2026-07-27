from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.executor_artifacts import (
    ExecutorArtifactStore,
    LazyExecutorArtifactStore,
    StagedCandidateLoader,
    default_staged_candidate_loader,
)
from nerve.representation_optimizer.benchmarking.executor_client import (
    ResidentExecutorClient,
    ResidentExecutorMountSpec,
    ResidentExecutorSession,
)
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_OBSERVATION_SCHEMA,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA,
    BenchmarkObservation,
    benchmark_observation_id,
    benchmark_residency_event_id,
)
from nerve.representation_optimizer.benchmarking.protocols import (
    BenchmarkExecutionRequest,
    BenchmarkMountRequest,
)
from nerve.representation_optimizer.benchmarking.executor_protocol import (
    nonnegative_integer,
    positive_integer,
    required_digest,
    required_device_state_digest,
    required_text,
    validated_windows,
)
from nerve.representation_optimizer.benchmarking.executor_transport import (
    ExecutorFactory,
    subprocess_executor,
)
from nerve.representation_optimizer.contracts import (
    canonical_json_bytes,
    contract_digest,
)


class ResidentComponentExecutionAdapter:
    """Production adapter for ordinary resident component execution."""

    def __init__(
        self,
        *,
        package_manifest: Path,
        candidate_workspace: Path,
        trace_root: Path,
        executor_command: tuple[str, ...],
        vulkan_driver_files: tuple[Path, ...],
        executor_factory: ExecutorFactory | None = None,
        staged_candidate_loader: StagedCandidateLoader | None = None,
    ) -> None:
        package_manifest = package_manifest.resolve()
        if not package_manifest.is_file():
            raise ModelCompileError(
                "resident execution package manifest is unavailable"
            )
        if not executor_command or any(not part for part in executor_command):
            raise ModelCompileError(
                "resident execution requires a non-empty executor command"
            )
        drivers = tuple(path.resolve() for path in vulkan_driver_files)
        if not drivers or any(not path.is_file() for path in drivers):
            raise ModelCompileError(
                "resident execution requires explicit existing AMD Vulkan "
                "driver manifests"
            )
        if trace_root.is_symlink():
            raise ModelCompileError(
                "resident execution trace root must not be a symlink"
            )
        trace_root = trace_root.resolve()
        self.package_manifest = package_manifest
        self.package_dir = package_manifest.parent
        self.candidate_workspace = candidate_workspace.resolve()
        self.trace_root = trace_root
        self.trace_store = LazyExecutorArtifactStore(
            trace_root,
            label="benchmark trace",
        )
        self.executor_command = tuple(executor_command)
        self.vulkan_driver_files = drivers
        self.executor_factory = executor_factory or subprocess_executor
        self.staged_candidate_loader = (
            staged_candidate_loader or default_staged_candidate_loader
        )
        self.executor_client = ResidentExecutorClient(
            package_manifest=self.package_manifest,
            candidate_workspace=self.candidate_workspace,
            executor_command=self.executor_command,
            vulkan_driver_files=self.vulkan_driver_files,
            executor_factory=self.executor_factory,
            staged_candidate_loader=self.staged_candidate_loader,
        )
        self.run_nonce = uuid4().hex

    def iter_fixture_artifact(
        self,
        relative_path: str,
        *,
        candidate_id: str,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        candidate = self.staged_candidate_loader(
            self.candidate_workspace,
            candidate_id,
            self.package_dir,
        )
        store = ExecutorArtifactStore(
            candidate.path,
            label="benchmark fixture",
            create=False,
        )
        yield from store.iter_file(relative_path, chunk_bytes=chunk_bytes)

    def open_session(
        self,
        request: BenchmarkMountRequest,
    ) -> ResidentComponentExecutionSession:
        return ResidentComponentExecutionSession.open(self, request)

    def iter_trace_artifact(
        self,
        relative_path: str,
        *,
        chunk_bytes: int = 8 * 1024 * 1024,
    ) -> Iterable[bytes]:
        yield from self.trace_store.iter_file(
            relative_path,
            chunk_bytes=chunk_bytes,
        )

    def write_trace(self, relative_path: str, payload: bytes) -> Json:
        return self.trace_store.publish(relative_path, payload)

    def executor_environment(self) -> dict[str, str]:
        return self.executor_client.environment()


class ResidentComponentExecutionSession:
    def __init__(
        self,
        *,
        adapter: ResidentComponentExecutionAdapter,
        request: BenchmarkMountRequest,
        executor_session: ResidentExecutorSession,
        mount_payload: Json,
        mount_duration_ns: int,
    ) -> None:
        self.adapter = adapter
        self.request = request
        self.executor_session = executor_session
        self.mount_payload = mount_payload
        self.mount_duration_ns = mount_duration_ns
        self.session_nonce = uuid4().hex
        self.closed = False
        self._mount_event = self._residency_event(
            action="mount",
            duration_ns=mount_duration_ns,
            before=request.matched_conditions["idle_device_state_digest"],
            after=required_device_state_digest(
                mount_payload,
                "mounted_state_digest",
            ),
            released=False,
        )

    @classmethod
    def open(
        cls,
        adapter: ResidentComponentExecutionAdapter,
        request: BenchmarkMountRequest,
    ) -> ResidentComponentExecutionSession:
        controls = request.workload["controls"]
        component_id = required_text(controls, "component_id")
        physical_node_id = required_text(controls, "physical_node_id")
        phase = required_text(controls, "phase")
        if controls.get("execution") != "ordinary":
            raise ModelCompileError(
                "resident component benchmark requires ordinary execution"
            )
        width = int(request.workload["regime"]["activation_batch_width"])
        if phase not in {"decode", "prefill"}:
            raise ModelCompileError(
                "resident component benchmark phase must be decode or prefill"
            )
        devices = request.matched_conditions.get("devices")
        if not isinstance(devices, list) or not devices:
            raise ModelCompileError(
                "resident component benchmark requires declared physical devices"
            )
        placement = request.matched_conditions.get("placement")
        if not isinstance(placement, dict):
            raise ModelCompileError(
                "resident component benchmark requires explicit placement"
            )
        physical_device_id = required_text(
            placement,
            component_id,
        )
        declared_device_ids = {
            required_text(device, "device_id")
            for device in devices
            if isinstance(device, dict)
        }
        if (
            len(declared_device_ids) != len(devices)
            or physical_device_id not in declared_device_ids
        ):
            raise ModelCompileError(
                "resident component placement references an undeclared device"
            )
        maximum_wait_ns = positive_integer(
            request.matched_conditions.get("controls", {}).get(
                "maximum_quantum_wait_ns"
            ),
            "matched_conditions.controls.maximum_quantum_wait_ns",
        )
        executor_session = adapter.executor_client.open(
            ResidentExecutorMountSpec(
                implementation_id=request.implementation["implementation_id"],
                component_id=component_id,
                physical_node_id=physical_node_id,
                phase=phase,
                activation_batch_width=width,
                physical_device_id=physical_device_id,
                dynamic_state_capacity_activations=max(width, 1),
                maximum_quantum_wait_ns=maximum_wait_ns,
                request_identity=request.to_json(),
                cancel_requested=request.cancel_requested,
            )
        )
        payload = executor_session.mount_payload
        if request.implementation["implementation_id"].startswith(
            "staged-representation:"
        ) != (payload["candidate_id"] is not None):
            executor_session.close(
                request_identity={
                    "plan_id": request.plan_id,
                    "block_index": request.block_index,
                    "reason": "implementation_identity_mismatch",
                }
            )
            raise ModelCompileError(
                "resident executor implementation role changed at mount"
            )
        return cls(
            adapter=adapter,
            request=request,
            executor_session=executor_session,
            mount_payload=payload,
            mount_duration_ns=executor_session.host_mount_ns,
        )

    @property
    def mount_event(self) -> Json:
        return dict(self._mount_event)

    def execute(self, request: BenchmarkExecutionRequest) -> Json:
        if self.closed:
            raise ModelCompileError("resident component execution session is closed")
        if (
            request.plan_id != self.request.plan_id
            or request.role != self.request.role
            or request.implementation_id
            != self.request.implementation["implementation_id"]
            or request.workload["workload_id"] != self.request.workload["workload_id"]
            or request.seed != self.request.seed
            or request.reset_to_initial_state is not True
        ):
            raise ModelCompileError(
                "resident component execution request changed its mounted trial"
            )
        useful_units = int(request.workload["useful_work"]["minimum_units"])
        execution = self.executor_session.execute(
            useful_units=useful_units,
            seed=request.seed,
            request_identity=request.to_json(),
        )
        return self._observation(
            request,
            execution.report,
            host_execution_ns=execution.host_execution_ns,
            transport_bytes=execution.transport_bytes,
        )

    def close(self) -> Json:
        if self.closed:
            raise ModelCompileError("resident component execution session closed twice")
        self.closed = True
        release = self.executor_session.close(
            request_identity={
                "plan_id": self.request.plan_id,
                "block_index": self.request.block_index,
            }
        )
        return self._residency_event(
            action="unmount",
            duration_ns=release.host_release_ns,
            before=self.mount_payload["mounted_state_digest"],
            after=self.request.matched_conditions["idle_device_state_digest"],
            released=True,
        )

    def _observation(
        self,
        request: BenchmarkExecutionRequest,
        report: Json,
        *,
        host_execution_ns: int,
        transport_bytes: int,
    ) -> Json:
        controls = request.workload["controls"]
        expected_phase = required_text(controls, "phase")
        expected_width = int(request.workload["regime"]["activation_batch_width"])
        if (
            required_text(report, "component_id")
            != required_text(controls, "component_id")
            or required_text(report, "node_id")
            != required_text(controls, "physical_node_id")
            or required_text(report, "phase") != expected_phase
            or positive_integer(
                report.get("activation_batch_width"),
                "executor report activation_batch_width",
            )
            != expected_width
        ):
            raise ModelCompileError(
                "resident executor completed different targeted component work"
            )
        useful_units = positive_integer(
            report.get("useful_units"),
            "executor report useful_units",
        )
        expected_units = int(request.workload["useful_work"]["minimum_units"])
        if useful_units != expected_units:
            raise ModelCompileError(
                "resident executor changed the requested useful work"
            )
        device_busy_ns = positive_integer(
            report.get("execution_ns"),
            "executor report execution_ns",
        )
        measurement_ns = max(host_execution_ns, device_busy_ns)
        windows = validated_windows(
            report.get("throughput_windows"),
            useful_units,
        )
        traces = self._write_execution_traces(request, report)
        permanent_bytes = nonnegative_integer(
            report.get("resident_parameter_bytes"),
            "executor report resident_parameter_bytes",
        )
        transient_bytes = nonnegative_integer(
            report.get("resident_transient_bytes"),
            "executor report resident_transient_bytes",
        )
        resident_bytes = permanent_bytes + transient_bytes
        queue_wait_ns = nonnegative_integer(
            report.get("queue_wait_ns"),
            "executor report queue_wait_ns",
        )
        synchronization_wait_ns = nonnegative_integer(
            report.get("synchronization_wait_ns"),
            "executor report synchronization_wait_ns",
        )
        if (
            queue_wait_ns > host_execution_ns
            or synchronization_wait_ns > host_execution_ns
        ):
            raise ModelCompileError(
                "executor wait counters exceed observed host execution time"
            )
        synchronization_count = positive_integer(
            report.get("synchronization_wait_count"),
            "executor report synchronization_wait_count",
        )
        physical_dispatch_count = positive_integer(
            report.get("physical_dispatch_count"),
            "executor report physical_dispatch_count",
        )
        queue_submission_count = positive_integer(
            report.get("queue_submission_count"),
            "executor report queue_submission_count",
        )
        document = {
            "schema": BENCHMARK_OBSERVATION_SCHEMA,
            "observation_id": "",
            "plan_id": request.plan_id,
            "implementation_id": request.implementation_id,
            "role": request.role,
            "workload_id": request.workload["workload_id"],
            "phase": request.phase,
            "seed": request.seed,
            "block_index": request.block_index,
            "pair_index": request.pair_index,
            "order_index": request.order_index,
            "matched_conditions_digest": request.matched_conditions_digest,
            "input_digest": request.workload["input"]["digest"],
            "initial_state_digest": (
                None
                if request.workload["initial_state"] is None
                else request.workload["initial_state"]["digest"]
            ),
            "controls_digest": contract_digest(request.workload["controls"]),
            "status": "completed",
            "stop_reason": request.workload["useful_work"]["completion_condition"],
            "timing": {
                "setup_ns": 0,
                "execution_ns": host_execution_ns,
                "teardown_ns": 0,
                "queue_wait_ns": queue_wait_ns,
            },
            "work": {
                "unit": request.workload["useful_work"]["unit"],
                "useful_units": useful_units,
                "speculative_units": 0,
                "cancelled_units": 0,
                "discarded_units": 0,
                "corrective_units": 0,
            },
            "memory": {
                "permanent_bytes": permanent_bytes,
                "peak_transient_bytes": transient_bytes,
                "resident_before_bytes": resident_bytes,
                "resident_peak_bytes": resident_bytes,
                "resident_after_bytes": resident_bytes,
            },
            "representation": {
                "conversion_bytes": 0,
                "conversion_ns": 0,
                "boundary_count": 0,
            },
            "device": {
                "measurement_ns": measurement_ns,
                "busy_ns": device_busy_ns,
                "utilization_ppm": round(device_busy_ns * 1_000_000 / measurement_ns),
            },
            "synchronization": {
                "operation_count": synchronization_count,
                "wait_ns": synchronization_wait_ns,
            },
            "transport": {
                "bytes": transport_bytes,
                "duration_ns": 0,
                "queue_wait_count": 0,
                "queue_wait_ns": 0,
                "timeout_count": 0,
            },
            "throughput_windows": windows,
            "traces": traces,
            "default_statistics": {
                "execution_path": "resident_targeted_component",
                "physical_dispatch_count": physical_dispatch_count,
                "queue_submission_count": queue_submission_count,
                "synchronization_wait_count": synchronization_count,
                "synchronization_wait_ns": synchronization_wait_ns,
                "queue_wait_ns": queue_wait_ns,
                "device_execution_ns": device_busy_ns,
                "host_execution_ns": host_execution_ns,
                "activation_batch_width": report["activation_batch_width"],
            },
            "diagnostics": [],
        }
        document["observation_id"] = benchmark_observation_id(document)
        return BenchmarkObservation.from_json(document).to_json()

    def _write_execution_traces(
        self,
        request: BenchmarkExecutionRequest,
        report: Json,
    ) -> Json:
        prefix = (
            f"traces/executor/{self.adapter.run_nonce}/"
            f"{self.session_nonce}/"
            f"{request.workload['workload_id']}/{request.role}/"
            f"{request.seed}/{request.block_index}/"
            f"{request.phase}/{request.order_index}/"
            f"{request.pair_index}"
        )
        payloads = {
            "distribution": {
                "output_digest": required_digest(
                    report,
                    "output_digest",
                ),
                "op": required_text(report, "op"),
            },
            "tokens": {
                "output_digest": report["output_digest"],
                "useful_units": report["useful_units"],
            },
            "state": {
                "state_digest": required_digest(
                    report,
                    "state_digest",
                ),
            },
            "random_draws": {
                "algorithm": "deterministic_fixture_counter",
                "seed": request.seed,
            },
            "schedule": {
                "throughput_windows": [
                    {
                        "index": window["index"],
                        "start_unit": window["start_unit"],
                        "end_unit": window["end_unit"],
                    }
                    for window in report["throughput_windows"]
                ],
                "physical_dispatch_count": report["physical_dispatch_count"],
                "queue_submission_count": report["queue_submission_count"],
                "synchronization_wait_count": report["synchronization_wait_count"],
            },
        }
        return {
            name: self.adapter.write_trace(
                f"{prefix}/{name}.json",
                canonical_json_bytes(payload) + b"\n",
            )
            for name, payload in payloads.items()
        }

    def _residency_event(
        self,
        *,
        action: str,
        duration_ns: int,
        before: str,
        after: str,
        released: bool,
    ) -> Json:
        permanent_bytes = nonnegative_integer(
            self.mount_payload.get("resident_parameter_bytes"),
            "executor mount resident_parameter_bytes",
        )
        transient_bytes = nonnegative_integer(
            self.mount_payload.get("resident_transient_bytes"),
            "executor mount resident_transient_bytes",
        )
        document = {
            "schema": BENCHMARK_RESIDENCY_EVENT_SCHEMA,
            "event_id": "",
            "plan_id": self.request.plan_id,
            "implementation_id": self.request.implementation["implementation_id"],
            "role": self.request.role,
            "workload_id": self.request.workload["workload_id"],
            "seed": self.request.seed,
            "block_index": self.request.block_index,
            "action": action,
            "duration_ns": duration_ns,
            "permanent_bytes": permanent_bytes,
            "peak_transient_bytes": transient_bytes,
            "matched_conditions_digest": (self.request.matched_conditions_digest),
            "device_state_before_digest": before,
            "device_state_after_digest": after,
            "released": released,
            "default_statistics": {
                "execution_path": "resident_targeted_component",
                "action": action,
                "physical_device_id": self.mount_payload["physical_device_id"],
                "logical_device_id": self.mount_payload["logical_device_id"],
            },
        }
        document["event_id"] = benchmark_residency_event_id(document)
        return document
