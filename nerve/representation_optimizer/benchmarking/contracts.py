from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import (
    BENCHMARK_RECORD_SCHEMA,
    ContractValidationError,
    DEVICE_STATE_DIGEST_SCHEMA,
    canonical_json_bytes,
    contract_digest,
    stable_contract_id,
)


BENCHMARK_WORKLOAD_SCHEMA = "nerve.optimizer.benchmark_workload.v1"
BENCHMARK_PLAN_SCHEMA = "nerve.optimizer.benchmark_plan.v5"
BENCHMARK_OBSERVATION_SCHEMA = "nerve.optimizer.benchmark_observation.v3"
BENCHMARK_RESIDENCY_EVENT_SCHEMA = "nerve.optimizer.benchmark_residency_event.v2"
BENCHMARK_RUN_SCHEMA = "nerve.optimizer.benchmark_run.v5"
BENCHMARK_EVIDENCE_INTEGRITY_SCHEMA = "nerve.optimizer.benchmark_evidence_integrity.v1"

_ARTIFACT_DIGEST_PREFIX = "nerve.optimizer.artifact_sha256.v1:"
_CONTRACT_DIGEST_PREFIX = "nerve.optimizer.canonical_json_sha256.v1:"
_ROLES = ("reference", "candidate")
_MOUNT_MODES = ("cold", "resident_reuse")
_PHASES = ("warmup", "measured")
_STATUSES = ("completed", "cancelled", "timeout", "failed")


class BenchmarkContractError(ContractValidationError):
    """A matched benchmark contract is malformed or internally inconsistent."""


@dataclass(frozen=True)
class BenchmarkWorkload:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BenchmarkWorkload:
        normalized = deepcopy(document)
        validate_benchmark_workload(normalized)
        return cls(normalized)

    @property
    def workload_id(self) -> str:
        return str(self._document["workload_id"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self._document["randomness"]["seeds"])

    @property
    def mount_mode(self) -> str:
        return str(self._document["regime"]["mount_mode"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class BenchmarkPlan:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BenchmarkPlan:
        normalized = deepcopy(document)
        validate_benchmark_plan(normalized)
        return cls(normalized)

    @property
    def plan_id(self) -> str:
        return str(self._document["plan_id"])

    @property
    def candidate_id(self) -> str:
        return str(self._document["candidate_id"])

    @property
    def workloads(self) -> tuple[BenchmarkWorkload, ...]:
        return tuple(
            BenchmarkWorkload.from_json(workload)
            for workload in self._document["workloads"]
        )

    @property
    def policy(self) -> Json:
        return deepcopy(self._document["policy"])

    @property
    def matched_conditions(self) -> Json:
        return deepcopy(self._document["matched_conditions"])

    def implementation(self, role: str) -> Json:
        if role not in _ROLES:
            raise BenchmarkContractError(f"unknown benchmark role {role!r}")
        return deepcopy(self._document["implementations"][role])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class BenchmarkObservation:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BenchmarkObservation:
        normalized = deepcopy(document)
        validate_benchmark_observation(normalized)
        return cls(normalized)

    @property
    def observation_id(self) -> str:
        return str(self._document["observation_id"])

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class BenchmarkResidencyEvent:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BenchmarkResidencyEvent:
        normalized = deepcopy(document)
        validate_benchmark_residency_event(normalized)
        return cls(normalized)

    def to_json(self) -> Json:
        return deepcopy(self._document)


@dataclass(frozen=True)
class BenchmarkRun:
    _document: Json

    @classmethod
    def from_json(cls, document: Json) -> BenchmarkRun:
        normalized = deepcopy(document)
        validate_benchmark_run(normalized)
        return cls(normalized)

    def to_json(self) -> Json:
        return deepcopy(self._document)


def benchmark_workload_id(document: Json) -> str:
    return _content_id("benchmark_workload", "workload_id", document)


def benchmark_plan_id(document: Json) -> str:
    return _content_id("benchmark_plan", "plan_id", document)


def benchmark_observation_id(document: Json) -> str:
    return _content_id("benchmark_observation", "observation_id", document)


def benchmark_residency_event_id(document: Json) -> str:
    return _content_id("benchmark_residency", "event_id", document)


def benchmark_run_id(document: Json) -> str:
    return _content_id("benchmark_run", "run_id", document)


def benchmark_record_id(document: Json) -> str:
    return _content_id("benchmark", "benchmark_id", document)


def validate_benchmark_workload(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "workload_id",
            "name",
            "regime",
            "input",
            "initial_state",
            "controls",
            "randomness",
            "useful_work",
        },
        "benchmark workload",
    )
    _schema(document, BENCHMARK_WORKLOAD_SCHEMA, "benchmark workload")
    _stable_id(document["workload_id"], "benchmark_workload", "workload_id")
    _text(document["name"], "name")
    regime = _object(document["regime"], "regime")
    _fields(
        regime,
        {
            "execution_phase",
            "activation_batch_width",
            "context_size",
            "state_size",
            "stream_count",
            "mount_mode",
            "boundary_mode",
        },
        "regime",
    )
    if regime["execution_phase"] not in {
        "prefill",
        "decode",
        "mixed",
        "component",
        "state_transition",
    }:
        raise BenchmarkContractError("regime.execution_phase is unsupported")
    _positive(regime["activation_batch_width"], "regime.activation_batch_width")
    _nonnegative(regime["context_size"], "regime.context_size")
    _nonnegative(regime["state_size"], "regime.state_size")
    _positive(regime["stream_count"], "regime.stream_count")
    if regime["mount_mode"] not in _MOUNT_MODES:
        raise BenchmarkContractError("regime.mount_mode is unsupported")
    if regime["boundary_mode"] not in {"local", "cross_device"}:
        raise BenchmarkContractError("regime.boundary_mode is unsupported")
    _fixture_ref(document["input"], "input")
    if document["initial_state"] is not None:
        _fixture_ref(document["initial_state"], "initial_state")
    _object(document["controls"], "controls")
    randomness = _object(document["randomness"], "randomness")
    _fields(
        randomness,
        {
            "algorithm",
            "seeds",
            "deterministic_replay_required",
            "permit_sampling_variance",
            "permit_numerical_nondeterminism",
            "permit_speculative_schedule_variance",
        },
        "randomness",
    )
    _text(randomness["algorithm"], "randomness.algorithm")
    seeds = _list(randomness["seeds"], "randomness.seeds")
    if (
        len(seeds) != 1
        or seeds != sorted(set(seeds))
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > 0xFFFF_FFFF
            for seed in seeds
        )
    ):
        raise BenchmarkContractError(
            "randomness.seeds must contain exactly one U32 seed"
        )
    for field in (
        "deterministic_replay_required",
        "permit_sampling_variance",
        "permit_numerical_nondeterminism",
        "permit_speculative_schedule_variance",
    ):
        _boolean(randomness[field], f"randomness.{field}")
    useful = _object(document["useful_work"], "useful_work")
    _fields(
        useful,
        {
            "unit",
            "minimum_units",
            "completion_condition",
            "output_allowance",
            "output_allowance_basis",
            "matched_work_policy",
            "sustained_window_count",
        },
        "useful_work",
    )
    _text(useful["unit"], "useful_work.unit")
    minimum = _positive(useful["minimum_units"], "useful_work.minimum_units")
    _text(useful["completion_condition"], "useful_work.completion_condition")
    allowance = useful["output_allowance"]
    basis = _object(
        useful["output_allowance_basis"],
        "useful_work.output_allowance_basis",
    )
    if allowance is None:
        _fields(
            basis,
            {"kind"},
            "useful_work.output_allowance_basis",
        )
        if basis["kind"] != "unlimited":
            raise BenchmarkContractError(
                "an unlimited workload requires an explicit unlimited basis"
            )
    else:
        if _positive(allowance, "useful_work.output_allowance") < minimum:
            raise BenchmarkContractError(
                "useful_work.output_allowance is below minimum useful work"
            )
        kind = basis.get("kind")
        if kind == "declared_model_limit":
            _fields(
                basis,
                {
                    "kind",
                    "artifact",
                    "json_pointer",
                    "declared_limit",
                },
                "useful_work.output_allowance_basis",
            )
            _fixture_ref(
                basis["artifact"],
                "useful_work.output_allowance_basis.artifact",
            )
            pointer = _text(
                basis["json_pointer"],
                "useful_work.output_allowance_basis.json_pointer",
            )
            if not pointer.startswith("/"):
                raise BenchmarkContractError(
                    "declared output limit JSON pointer must start with '/'"
                )
            for segment in pointer.split("/")[1:]:
                index = 0
                while index < len(segment):
                    if segment[index] == "~" and (
                        index + 1 == len(segment)
                        or segment[index + 1] not in {"0", "1"}
                    ):
                        raise BenchmarkContractError(
                            "declared output limit JSON pointer has an invalid escape"
                        )
                    index += 2 if segment[index] == "~" else 1
            if basis["declared_limit"] != allowance:
                raise BenchmarkContractError(
                    "declared output limit does not match output allowance"
                )
        elif kind == "validity_regime":
            _fields(
                basis,
                {
                    "kind",
                    "candidate_id",
                    "predicate",
                    "maximum_units",
                },
                "useful_work.output_allowance_basis",
            )
            _stable_id(
                basis["candidate_id"],
                "candidate",
                "useful_work.output_allowance_basis.candidate_id",
            )
            if not _object(
                basis["predicate"],
                "useful_work.output_allowance_basis.predicate",
            ):
                raise BenchmarkContractError(
                    "output validity regime predicate must not be empty"
                )
            if basis["maximum_units"] != allowance:
                raise BenchmarkContractError(
                    "validity-regime maximum does not match output allowance"
                )
        else:
            raise BenchmarkContractError(
                "bounded output allowance needs verifiable source or "
                "candidate-regime evidence"
            )
    if useful["matched_work_policy"] != "equal_useful_work":
        raise BenchmarkContractError(
            "matched benchmark currently requires equal useful work"
        )
    sustained_window_count = _positive(
        useful["sustained_window_count"],
        "useful_work.sustained_window_count",
    )
    if sustained_window_count > minimum:
        raise BenchmarkContractError(
            "useful_work sustained window count exceeds minimum useful work"
        )
    _content_identity(
        document,
        "benchmark_workload",
        "workload_id",
        "benchmark workload",
    )


def validate_benchmark_plan(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "plan_id",
            "candidate_id",
            "construction_record_digest",
            "source_contract_digests",
            "implementations",
            "matched_conditions",
            "matched_conditions_digest",
            "workloads",
            "policy",
        },
        "benchmark plan",
    )
    _schema(document, BENCHMARK_PLAN_SCHEMA, "benchmark plan")
    _stable_id(document["plan_id"], "benchmark_plan", "plan_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    _contract_digest(
        document["construction_record_digest"],
        "construction_record_digest",
    )
    digests = _sorted_strings(
        document["source_contract_digests"],
        "source_contract_digests",
        nonempty=True,
    )
    for index, digest in enumerate(digests):
        _contract_digest(digest, f"source_contract_digests[{index}]")
    implementations = _object(document["implementations"], "implementations")
    _fields(implementations, set(_ROLES), "implementations")
    for role in _ROLES:
        _implementation(implementations[role], f"implementations.{role}")
    if (
        implementations["reference"]["implementation_id"]
        == implementations["candidate"]["implementation_id"]
    ):
        raise BenchmarkContractError(
            "reference and candidate implementation ids must differ"
        )
    if (
        implementations["reference"]["contract_digest"] not in digests
        or implementations["candidate"]["contract_digest"]
        != document["construction_record_digest"]
    ):
        raise BenchmarkContractError(
            "benchmark implementations are not bound to the source and "
            "constructed candidate contracts"
        )
    conditions = _object(document["matched_conditions"], "matched_conditions")
    validate_matched_conditions(conditions)
    expected_conditions_digest = contract_digest(conditions)
    if document["matched_conditions_digest"] != expected_conditions_digest:
        raise BenchmarkContractError(
            "matched_conditions_digest does not match matched conditions"
        )
    workloads = _list(document["workloads"], "workloads")
    parsed = [BenchmarkWorkload.from_json(workload) for workload in workloads]
    workload_ids = [workload.workload_id for workload in parsed]
    if not workload_ids or workload_ids != sorted(set(workload_ids)):
        raise BenchmarkContractError(
            "benchmark plan workloads must be non-empty, sorted, and unique"
        )
    fixture_digests: dict[str, str] = {}
    for workload in workloads:
        useful = workload["useful_work"]
        basis = useful["output_allowance_basis"]
        references = [
            workload["input"],
            workload["initial_state"],
            basis.get("artifact"),
        ]
        for reference in references:
            if reference is None:
                continue
            previous = fixture_digests.setdefault(
                reference["path"],
                reference["digest"],
            )
            if previous != reference["digest"]:
                raise BenchmarkContractError(
                    "benchmark fixture path is bound to different bytes"
                )
        if (
            basis["kind"] == "validity_regime"
            and basis["candidate_id"] != document["candidate_id"]
        ):
            raise BenchmarkContractError(
                "workload validity regime cites another candidate"
            )
    policy = _object(document["policy"], "policy")
    _benchmark_policy(policy)
    _content_identity(
        document,
        "benchmark_plan",
        "plan_id",
        "benchmark plan",
    )


def validate_benchmark_observation(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "observation_id",
            "plan_id",
            "implementation_id",
            "role",
            "workload_id",
            "phase",
            "seed",
            "block_index",
            "pair_index",
            "order_index",
            "matched_conditions_digest",
            "input_digest",
            "initial_state_digest",
            "controls_digest",
            "status",
            "stop_reason",
            "timing",
            "work",
            "memory",
            "representation",
            "resource_loading",
            "device",
            "synchronization",
            "transport",
            "throughput_windows",
            "traces",
            "default_statistics",
            "diagnostics",
        },
        "benchmark observation",
    )
    _schema(document, BENCHMARK_OBSERVATION_SCHEMA, "benchmark observation")
    _stable_id(
        document["observation_id"],
        "benchmark_observation",
        "observation_id",
    )
    _stable_id(document["plan_id"], "benchmark_plan", "plan_id")
    _text(document["implementation_id"], "implementation_id")
    if document["role"] not in _ROLES:
        raise BenchmarkContractError("observation role is unsupported")
    _stable_id(document["workload_id"], "benchmark_workload", "workload_id")
    if document["phase"] not in _PHASES:
        raise BenchmarkContractError("observation phase is unsupported")
    _u32(document["seed"], "seed")
    _nonnegative(document["block_index"], "block_index")
    _nonnegative(document["pair_index"], "pair_index")
    if document["order_index"] not in {0, 1}:
        raise BenchmarkContractError("order_index must be zero or one")
    _contract_digest(
        document["matched_conditions_digest"],
        "matched_conditions_digest",
    )
    _artifact_digest(document["input_digest"], "input_digest")
    if document["initial_state_digest"] is not None:
        _artifact_digest(
            document["initial_state_digest"],
            "initial_state_digest",
        )
    _contract_digest(document["controls_digest"], "controls_digest")
    if document["status"] not in _STATUSES:
        raise BenchmarkContractError("observation status is unsupported")
    _text(document["stop_reason"], "stop_reason")
    timing = _object(document["timing"], "timing")
    _fields(
        timing,
        {
            "setup_ns",
            "execution_ns",
            "teardown_ns",
            "queue_wait_ns",
        },
        "timing",
    )
    for field, value in timing.items():
        _nonnegative(value, f"timing.{field}")
    if document["status"] == "completed" and timing["execution_ns"] == 0:
        raise BenchmarkContractError(
            "completed observation must have positive execution time"
        )
    if timing["queue_wait_ns"] > timing["execution_ns"]:
        raise BenchmarkContractError("timing.queue_wait_ns exceeds execution time")
    work = _object(document["work"], "work")
    _fields(
        work,
        {
            "unit",
            "useful_units",
            "speculative_units",
            "cancelled_units",
            "discarded_units",
            "corrective_units",
        },
        "work",
    )
    _text(work["unit"], "work.unit")
    for field in (
        "useful_units",
        "speculative_units",
        "cancelled_units",
        "discarded_units",
        "corrective_units",
    ):
        _nonnegative(work[field], f"work.{field}")
    memory = _object(document["memory"], "memory")
    _fields(
        memory,
        {
            "permanent_bytes",
            "peak_transient_bytes",
            "resident_before_bytes",
            "resident_peak_bytes",
            "resident_after_bytes",
        },
        "memory",
    )
    for field, value in memory.items():
        _nonnegative(value, f"memory.{field}")
    if memory["resident_peak_bytes"] < max(
        memory["resident_before_bytes"],
        memory["resident_after_bytes"],
    ):
        raise BenchmarkContractError(
            "memory.resident_peak_bytes is below a residency endpoint"
        )
    representation = _object(document["representation"], "representation")
    _fields(
        representation,
        {
            "conversion_bytes",
            "conversion_ns",
            "boundary_count",
        },
        "representation",
    )
    for field, value in representation.items():
        _nonnegative(value, f"representation.{field}")
    resource_loading = _object(
        document["resource_loading"],
        "resource_loading",
    )
    _fields(
        resource_loading,
        {
            "load_count",
            "reload_count",
            "physical_read_bytes",
            "resident_bytes_produced",
            "uploaded_bytes",
            "read_ns",
            "derivation_ns",
            "upload_ns",
            "blocking_ns",
        },
        "resource_loading",
    )
    for field, value in resource_loading.items():
        _nonnegative(value, f"resource_loading.{field}")
    device = _object(document["device"], "device")
    _fields(
        device,
        {"measurement_ns", "busy_ns", "utilization_ppm"},
        "device",
    )
    measured = _nonnegative(device["measurement_ns"], "device.measurement_ns")
    busy = _nonnegative(device["busy_ns"], "device.busy_ns")
    utilization = _nonnegative(
        device["utilization_ppm"],
        "device.utilization_ppm",
    )
    if busy > measured or utilization > 1_000_000:
        raise BenchmarkContractError("device utilization counters are invalid")
    expected_utilization = round(busy * 1_000_000 / measured) if measured else 0
    if utilization != expected_utilization:
        raise BenchmarkContractError(
            "device.utilization_ppm does not match busy and measurement time"
        )
    synchronization = _object(document["synchronization"], "synchronization")
    _fields(
        synchronization,
        {"operation_count", "wait_ns"},
        "synchronization",
    )
    _nonnegative(
        synchronization["operation_count"],
        "synchronization.operation_count",
    )
    _nonnegative(synchronization["wait_ns"], "synchronization.wait_ns")
    transport = _object(document["transport"], "transport")
    _fields(
        transport,
        {
            "bytes",
            "duration_ns",
            "queue_wait_count",
            "queue_wait_ns",
            "timeout_count",
        },
        "transport",
    )
    for field, value in transport.items():
        _nonnegative(value, f"transport.{field}")
    windows = _list(document["throughput_windows"], "throughput_windows")
    expected_start = 0
    for index, raw_window in enumerate(windows):
        window = _object(raw_window, f"throughput_windows[{index}]")
        _fields(
            window,
            {"index", "start_unit", "end_unit", "duration_ns"},
            f"throughput_windows[{index}]",
        )
        if window["index"] != index:
            raise BenchmarkContractError("throughput window indexes must be contiguous")
        if window["start_unit"] != expected_start:
            raise BenchmarkContractError(
                "throughput window useful-work ranges must be contiguous"
            )
        end = _positive(
            window["end_unit"],
            f"throughput_windows[{index}].end_unit",
        )
        if end <= window["start_unit"]:
            raise BenchmarkContractError(
                "throughput window must contain positive useful work"
            )
        _positive(
            window["duration_ns"],
            f"throughput_windows[{index}].duration_ns",
        )
        expected_start = end
    if windows and expected_start != work["useful_units"]:
        raise BenchmarkContractError(
            "throughput windows must exactly cover observed useful work"
        )
    if sum(window["duration_ns"] for window in windows) > timing["execution_ns"]:
        raise BenchmarkContractError(
            "throughput-window duration exceeds execution time"
        )
    traces = _object(document["traces"], "traces")
    _fields(
        traces,
        {
            "distribution",
            "tokens",
            "state",
            "random_draws",
            "schedule",
        },
        "traces",
    )
    trace_paths = []
    for field, artifact in traces.items():
        _artifact_ref(artifact, f"traces.{field}")
        if not artifact["path"].startswith("traces/"):
            raise BenchmarkContractError(f"traces.{field}.path must live below traces/")
        trace_paths.append(artifact["path"])
    if len(trace_paths) != len(set(trace_paths)):
        raise BenchmarkContractError("observation trace artifact paths must be unique")
    if not _object(document["default_statistics"], "default_statistics"):
        raise BenchmarkContractError(
            "normal runtime default statistics must not be empty"
        )
    _string_list(document["diagnostics"], "diagnostics")
    _content_identity(
        document,
        "benchmark_observation",
        "observation_id",
        "benchmark observation",
    )


def validate_benchmark_residency_event(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "event_id",
            "plan_id",
            "implementation_id",
            "role",
            "workload_id",
            "seed",
            "block_index",
            "action",
            "duration_ns",
            "permanent_bytes",
            "peak_transient_bytes",
            "matched_conditions_digest",
            "device_state_before_digest",
            "device_state_after_digest",
            "released",
            "default_statistics",
        },
        "benchmark residency event",
    )
    _schema(
        document,
        BENCHMARK_RESIDENCY_EVENT_SCHEMA,
        "benchmark residency event",
    )
    _stable_id(document["event_id"], "benchmark_residency", "event_id")
    _stable_id(document["plan_id"], "benchmark_plan", "plan_id")
    _text(document["implementation_id"], "implementation_id")
    if document["role"] not in _ROLES:
        raise BenchmarkContractError("residency event role is unsupported")
    _stable_id(document["workload_id"], "benchmark_workload", "workload_id")
    _u32(document["seed"], "seed")
    _nonnegative(document["block_index"], "block_index")
    if document["action"] not in {"mount", "unmount"}:
        raise BenchmarkContractError("residency event action is unsupported")
    _nonnegative(document["duration_ns"], "duration_ns")
    _nonnegative(document["permanent_bytes"], "permanent_bytes")
    _nonnegative(document["peak_transient_bytes"], "peak_transient_bytes")
    _contract_digest(
        document["matched_conditions_digest"],
        "matched_conditions_digest",
    )
    _device_state_digest(
        document["device_state_before_digest"],
        "device_state_before_digest",
    )
    _device_state_digest(
        document["device_state_after_digest"],
        "device_state_after_digest",
    )
    _boolean(document["released"], "released")
    if document["action"] == "mount" and document["released"]:
        raise BenchmarkContractError("mount event cannot report released=true")
    if document["action"] == "unmount" and not document["released"]:
        raise BenchmarkContractError("unmount event must release residency")
    if not _object(document["default_statistics"], "default_statistics"):
        raise BenchmarkContractError(
            "normal runtime default residency statistics must not be empty"
        )
    _content_identity(
        document,
        "benchmark_residency",
        "event_id",
        "benchmark residency event",
    )


def validate_benchmark_run(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "run_id",
            "plan_id",
            "status",
            "execution_order",
            "observations",
            "residency_events",
            "sampling_outcomes",
            "host_elapsed_ns",
            "diagnostics",
        },
        "benchmark run",
    )
    _schema(document, BENCHMARK_RUN_SCHEMA, "benchmark run")
    _stable_id(document["run_id"], "benchmark_run", "run_id")
    _stable_id(document["plan_id"], "benchmark_plan", "plan_id")
    if document["status"] not in _STATUSES:
        raise BenchmarkContractError("benchmark run status is unsupported")
    observations = [
        BenchmarkObservation.from_json(observation)
        for observation in _list(document["observations"], "observations")
    ]
    observation_ids = [observation.observation_id for observation in observations]
    trace_paths = [
        trace["path"]
        for observation in observations
        for trace in observation.to_json()["traces"].values()
    ]
    if len(trace_paths) != len(set(trace_paths)):
        raise BenchmarkContractError("benchmark run reuses a raw trace artifact path")
    order = _string_list(document["execution_order"], "execution_order")
    if (
        (document["status"] == "completed" and not observation_ids)
        or order != observation_ids
        or len(order) != len(set(order))
    ):
        raise BenchmarkContractError(
            "benchmark execution_order must exactly cover observations"
        )
    for observation in observations:
        if observation.to_json()["plan_id"] != document["plan_id"]:
            raise BenchmarkContractError(
                "benchmark observation belongs to another plan"
            )
    events = [
        BenchmarkResidencyEvent.from_json(event)
        for event in _list(document["residency_events"], "residency_events")
    ]
    if document["status"] == "completed" and not events:
        raise BenchmarkContractError("benchmark run must retain residency evidence")
    if any(event.to_json()["plan_id"] != document["plan_id"] for event in events):
        raise BenchmarkContractError(
            "benchmark residency event belongs to another plan"
        )
    outcomes = _list(document["sampling_outcomes"], "sampling_outcomes")
    outcome_ids = []
    for index, outcome in enumerate(outcomes):
        path = f"sampling_outcomes[{index}]"
        outcome = _object(outcome, path)
        _fields(
            outcome,
            {
                "workload_id",
                "warmup_groups",
                "measured_calls_per_role",
                "decision",
                "reasons",
                "termination",
            },
            path,
        )
        outcome_ids.append(
            _stable_id(
                outcome["workload_id"],
                "benchmark_workload",
                f"{path}.workload_id",
            )
        )
        groups = _list(outcome["warmup_groups"], f"{path}.warmup_groups")
        group_keys = []
        for group_index, group in enumerate(groups):
            group_path = f"{path}.warmup_groups[{group_index}]"
            group = _object(group, group_path)
            _fields(
                group,
                {
                    "role",
                    "seed",
                    "cycle_index",
                    "order_block_index",
                    "attempt_index",
                    "sample_count",
                    "maximum_shift_ppm",
                    "converged",
                    "observation_ids",
                },
                group_path,
            )
            if group["role"] not in _ROLES:
                raise BenchmarkContractError(f"{group_path}.role is unsupported")
            seed = _u32(group["seed"], f"{group_path}.seed")
            attempt_index = _nonnegative(
                group["attempt_index"],
                f"{group_path}.attempt_index",
            )
            cycle_index = group["cycle_index"]
            order_block_index = group["order_block_index"]
            if cycle_index is not None:
                _nonnegative(cycle_index, f"{group_path}.cycle_index")
            if order_block_index is not None:
                if order_block_index not in {0, 1}:
                    raise BenchmarkContractError(
                        f"{group_path}.order_block_index must be zero, one, or null"
                    )
            if (cycle_index is None) != (order_block_index is None):
                raise BenchmarkContractError(
                    f"{group_path} cycle and order block must both be null or set"
                )
            _positive(group["sample_count"], f"{group_path}.sample_count")
            _nonnegative(
                group["maximum_shift_ppm"],
                f"{group_path}.maximum_shift_ppm",
            )
            _boolean(group["converged"], f"{group_path}.converged")
            group_observation_ids = _string_list(
                group["observation_ids"],
                f"{group_path}.observation_ids",
            )
            if len(group_observation_ids) != group["sample_count"] or len(
                group_observation_ids
            ) != len(set(group_observation_ids)):
                raise BenchmarkContractError(
                    f"{group_path}.observation_ids must exactly cover the group"
                )
            for observation_index, observation_id in enumerate(group_observation_ids):
                _stable_id(
                    observation_id,
                    "benchmark_observation",
                    f"{group_path}.observation_ids[{observation_index}]",
                )
            group_keys.append(
                (
                    seed,
                    -1 if cycle_index is None else cycle_index,
                    -1 if order_block_index is None else order_block_index,
                    _ROLES.index(group["role"]),
                    attempt_index,
                )
            )
        if not groups or group_keys != sorted(set(group_keys)):
            raise BenchmarkContractError(
                f"{path}.warmup_groups must be non-empty, sorted, and unique"
            )
        _positive(
            outcome["measured_calls_per_role"],
            f"{path}.measured_calls_per_role",
        )
        _decision(outcome["decision"], f"{path}.decision")
        _string_list(outcome["reasons"], f"{path}.reasons")
        if outcome["termination"] not in {
            "fixed_sample_complete",
            "invalid",
        }:
            raise BenchmarkContractError(f"{path}.termination is unsupported")
    if (document["status"] == "completed" and not outcome_ids) or outcome_ids != sorted(
        set(outcome_ids)
    ):
        raise BenchmarkContractError(
            "benchmark sampling outcomes must be non-empty, sorted, and unique"
        )
    elapsed = _list(document["host_elapsed_ns"], "host_elapsed_ns")
    if len(elapsed) != len(observations):
        raise BenchmarkContractError(
            "host elapsed evidence must cover every observation"
        )
    for index, record in enumerate(elapsed):
        record = _object(record, f"host_elapsed_ns[{index}]")
        _fields(
            record,
            {"observation_id", "duration_ns"},
            f"host_elapsed_ns[{index}]",
        )
        if record["observation_id"] != observation_ids[index]:
            raise BenchmarkContractError(
                "host elapsed evidence is not in execution order"
            )
        _positive(record["duration_ns"], f"host_elapsed_ns[{index}].duration_ns")
    _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "completed" and any(
        observation.to_json()["status"] != "completed" for observation in observations
    ):
        raise BenchmarkContractError(
            "completed benchmark run contains incomplete observations"
        )
    _content_identity(
        document,
        "benchmark_run",
        "run_id",
        "benchmark run",
    )


def validate_benchmark_record(document: Json) -> None:
    canonical_json_bytes(document)
    _fields(
        document,
        {
            "schema",
            "benchmark_id",
            "candidate_id",
            "plan_digest",
            "run_digest",
            "construction_record_digest",
            "reference_implementation_id",
            "matched_conditions_digest",
            "workloads",
            "reproducibility",
            "resource_measurements",
            "raw_evidence",
            "decision",
            "decision_reasons",
        },
        "benchmark record",
    )
    _schema(document, BENCHMARK_RECORD_SCHEMA, "benchmark record")
    _stable_id(document["benchmark_id"], "benchmark", "benchmark_id")
    _stable_id(document["candidate_id"], "candidate", "candidate_id")
    for field in (
        "plan_digest",
        "run_digest",
        "construction_record_digest",
        "matched_conditions_digest",
    ):
        _contract_digest(document[field], field)
    _text(
        document["reference_implementation_id"],
        "reference_implementation_id",
    )
    summaries = _list(document["workloads"], "workloads")
    workload_ids = []
    for index, summary in enumerate(summaries):
        path = f"workloads[{index}]"
        summary = _object(summary, path)
        _fields(
            summary,
            {
                "workload_id",
                "decision",
                "reasons",
                "sample_count_per_role",
                "warmup",
                "reference",
                "candidate",
                "paired",
                "sustained",
            },
            path,
        )
        workload_ids.append(
            _stable_id(
                summary["workload_id"],
                "benchmark_workload",
                f"{path}.workload_id",
            )
        )
        _decision(summary["decision"], f"{path}.decision")
        _string_list(summary["reasons"], f"{path}.reasons")
        _positive(
            summary["sample_count_per_role"],
            f"{path}.sample_count_per_role",
        )
        warmup = _object(summary["warmup"], f"{path}.warmup")
        _fields(warmup, set(_ROLES), f"{path}.warmup")
        for role in _ROLES:
            record = _object(warmup[role], f"{path}.warmup.{role}")
            _fields(
                record,
                {"sample_count", "maximum_shift_ppm", "converged"},
                f"{path}.warmup.{role}",
            )
            _positive(
                record["sample_count"],
                f"{path}.warmup.{role}.sample_count",
            )
            _nonnegative(
                record["maximum_shift_ppm"],
                f"{path}.warmup.{role}.maximum_shift_ppm",
            )
            _boolean(record["converged"], f"{path}.warmup.{role}.converged")
        for role in _ROLES:
            _role_summary(summary[role], f"{path}.{role}")
        paired = _object(summary["paired"], f"{path}.paired")
        _fields(
            paired,
            {
                "speedup_ppm",
                "candidate_is_faster",
            },
            f"{path}.paired",
        )
        _integer(paired["speedup_ppm"], f"{path}.paired.speedup_ppm")
        _boolean(
            paired["candidate_is_faster"],
            f"{path}.paired.candidate_is_faster",
        )
        if paired["candidate_is_faster"] != (paired["speedup_ppm"] > 0):
            raise BenchmarkContractError(
                f"{path}.paired binary decision disagrees with measured speedup"
            )
        sustained = _object(summary["sustained"], f"{path}.sustained")
        _fields(
            sustained,
            {
                "reference_slope_ppm_per_window",
                "candidate_slope_ppm_per_window",
                "candidate_regression_ppm",
                "passed",
            },
            f"{path}.sustained",
        )
        _integer(
            sustained["reference_slope_ppm_per_window"],
            f"{path}.sustained.reference_slope_ppm_per_window",
        )
        _integer(
            sustained["candidate_slope_ppm_per_window"],
            f"{path}.sustained.candidate_slope_ppm_per_window",
        )
        _nonnegative(
            sustained["candidate_regression_ppm"],
            f"{path}.sustained.candidate_regression_ppm",
        )
        _boolean(sustained["passed"], f"{path}.sustained.passed")
    if not workload_ids or workload_ids != sorted(set(workload_ids)):
        raise BenchmarkContractError(
            "benchmark workload summaries must be non-empty, sorted, and unique"
        )
    reproducibility = _list(document["reproducibility"], "reproducibility")
    repro_keys = []
    for index, record in enumerate(reproducibility):
        path = f"reproducibility[{index}]"
        record = _object(record, path)
        _fields(
            record,
            {
                "workload_id",
                "role",
                "seed",
                "order_index",
                "classification",
                "observation_ids",
            },
            path,
        )
        _stable_id(
            record["workload_id"],
            "benchmark_workload",
            f"{path}.workload_id",
        )
        if record["role"] not in _ROLES:
            raise BenchmarkContractError(f"{path}.role is unsupported")
        _u32(record["seed"], f"{path}.seed")
        if record["order_index"] not in {0, 1}:
            raise BenchmarkContractError(f"{path}.order_index must be zero or one")
        if record["classification"] not in {
            "identical",
            "permitted_sampling_variance",
            "numerical_nondeterminism",
            "speculative_scheduling",
            "correctness_defect",
        }:
            raise BenchmarkContractError(f"{path}.classification is unsupported")
        ids = _string_list(record["observation_ids"], f"{path}.observation_ids")
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise BenchmarkContractError(
                f"{path}.observation_ids must contain unique repetitions"
            )
        for observation_index, observation_id in enumerate(ids):
            _stable_id(
                observation_id,
                "benchmark_observation",
                f"{path}.observation_ids[{observation_index}]",
            )
        repro_keys.append(
            (
                record["workload_id"],
                record["role"],
                record["seed"],
                record["order_index"],
            )
        )
    if repro_keys != sorted(set(repro_keys)):
        raise BenchmarkContractError("reproducibility groups must be sorted and unique")
    resources = _object(
        document["resource_measurements"],
        "resource_measurements",
    )
    _fields(
        resources,
        {"construction", "roles"},
        "resource_measurements",
    )
    construction = _object(
        resources["construction"],
        "resource_measurements.construction",
    )
    _fields(
        construction,
        {
            "construction_time_ns",
            "peak_temporary_bytes",
            "peak_staging_bytes",
            "final_permanent_bytes",
            "generated_artifact_bytes",
        },
        "resource_measurements.construction",
    )
    for field, value in construction.items():
        _nonnegative(value, f"resource_measurements.construction.{field}")
    roles = _object(resources["roles"], "resource_measurements.roles")
    _fields(roles, set(_ROLES), "resource_measurements.roles")
    for role in _ROLES:
        _resource_role(roles[role], f"resource_measurements.roles.{role}")
    raw = _object(document["raw_evidence"], "raw_evidence")
    _fields(
        raw,
        {
            "run_id",
            "observation_count",
            "residency_event_count",
            "host_elapsed_sample_count",
            "trace_artifact_count",
        },
        "raw_evidence",
    )
    _stable_id(raw["run_id"], "benchmark_run", "raw_evidence.run_id")
    for field in (
        "observation_count",
        "residency_event_count",
        "host_elapsed_sample_count",
        "trace_artifact_count",
    ):
        _positive(raw[field], f"raw_evidence.{field}")
    if raw["observation_count"] != raw["host_elapsed_sample_count"]:
        raise BenchmarkContractError(
            "raw evidence host elapsed count does not cover observations"
        )
    _decision(document["decision"], "decision")
    reasons = _string_list(document["decision_reasons"], "decision_reasons")
    if document["decision"] != "materially_faster" and not reasons:
        raise BenchmarkContractError("non-winning benchmark decision requires reasons")
    _content_identity(
        document,
        "benchmark",
        "benchmark_id",
        "benchmark record",
    )


def _role_summary(value: Any, path: str) -> None:
    summary = _object(value, path)
    _fields(
        summary,
        {
            "latency_ns",
            "throughput_per_second",
            "permanent_bytes",
            "peak_transient_bytes",
            "resident_before_bytes",
            "resident_peak_bytes",
            "resident_after_bytes",
            "conversion_bytes",
            "conversion_ns",
            "boundary_count",
            "resource_load_count",
            "resource_reload_count",
            "resource_physical_read_bytes",
            "resource_resident_bytes_produced",
            "resource_uploaded_bytes",
            "resource_read_ns",
            "resource_derivation_ns",
            "resource_upload_ns",
            "resource_blocking_ns",
            "utilization_ppm",
            "synchronization_wait_ns",
            "transport_bytes",
            "transport_ns",
            "queue_wait_ns",
            "timeout_count",
            "useful_units",
            "wasted_units",
        },
        path,
    )
    _distribution(summary["latency_ns"], f"{path}.latency_ns")
    _distribution(
        summary["throughput_per_second"],
        f"{path}.throughput_per_second",
    )
    for field in (
        "permanent_bytes",
        "peak_transient_bytes",
        "resident_before_bytes",
        "resident_peak_bytes",
        "resident_after_bytes",
        "conversion_bytes",
        "conversion_ns",
        "boundary_count",
        "resource_load_count",
        "resource_reload_count",
        "resource_physical_read_bytes",
        "resource_resident_bytes_produced",
        "resource_uploaded_bytes",
        "resource_read_ns",
        "resource_derivation_ns",
        "resource_upload_ns",
        "resource_blocking_ns",
        "utilization_ppm",
        "synchronization_wait_ns",
        "transport_bytes",
        "transport_ns",
        "queue_wait_ns",
        "timeout_count",
        "useful_units",
        "wasted_units",
    ):
        _nonnegative(summary[field], f"{path}.{field}")
    if summary["utilization_ppm"] > 1_000_000:
        raise BenchmarkContractError(f"{path}.utilization_ppm exceeds 100%")
    if summary["resident_peak_bytes"] < max(
        summary["resident_before_bytes"],
        summary["resident_after_bytes"],
    ):
        raise BenchmarkContractError(
            f"{path}.resident_peak_bytes is below a residency endpoint"
        )


def _distribution(value: Any, path: str) -> None:
    distribution = _object(value, path)
    _fields(
        distribution,
        {
            "sample_count",
            "minimum",
            "maximum",
            "median",
            "mean",
            "standard_deviation",
            "confidence_interval_low",
            "confidence_interval_high",
            "relative_ci_width_ppm",
        },
        path,
    )
    _positive(distribution["sample_count"], f"{path}.sample_count")
    for field in (
        "minimum",
        "maximum",
        "median",
        "mean",
        "standard_deviation",
        "confidence_interval_low",
        "confidence_interval_high",
        "relative_ci_width_ppm",
    ):
        _nonnegative(distribution[field], f"{path}.{field}")
    if not (
        distribution["minimum"] <= distribution["median"] <= distribution["maximum"]
    ):
        raise BenchmarkContractError(f"{path} median is outside its range")
    if (
        distribution["confidence_interval_low"]
        > distribution["confidence_interval_high"]
    ):
        raise BenchmarkContractError(f"{path} confidence interval is inverted")


def _resource_role(value: Any, path: str) -> None:
    record = _object(value, path)
    _fields(
        record,
        {
            "setup_ns",
            "teardown_ns",
            "host_elapsed_ns",
            "permanent_bytes",
            "peak_transient_bytes",
            "resident_before_bytes",
            "resident_peak_bytes",
            "resident_after_bytes",
            "conversion_bytes",
            "conversion_ns",
            "boundary_count",
            "resource_load_count",
            "resource_reload_count",
            "resource_physical_read_bytes",
            "resource_resident_bytes_produced",
            "resource_uploaded_bytes",
            "resource_read_ns",
            "resource_derivation_ns",
            "resource_upload_ns",
            "resource_blocking_ns",
            "device_measurement_ns",
            "device_busy_ns",
            "utilization_ppm",
            "synchronization_count",
            "synchronization_wait_ns",
            "transport_bytes",
            "transport_ns",
            "queue_wait_count",
            "queue_wait_ns",
            "timeout_count",
            "useful_units",
            "speculative_units",
            "cancelled_units",
            "discarded_units",
            "corrective_units",
        },
        path,
    )
    for field, value in record.items():
        _nonnegative(value, f"{path}.{field}")
    if record["device_busy_ns"] > record["device_measurement_ns"]:
        raise BenchmarkContractError(f"{path}.device_busy_ns exceeds measurement time")
    if record["resident_peak_bytes"] < max(
        record["resident_before_bytes"],
        record["resident_after_bytes"],
    ):
        raise BenchmarkContractError(
            f"{path}.resident_peak_bytes is below a residency endpoint"
        )
    expected = (
        round(record["device_busy_ns"] * 1_000_000 / record["device_measurement_ns"])
        if record["device_measurement_ns"]
        else 0
    )
    if record["utilization_ppm"] != expected:
        raise BenchmarkContractError(
            f"{path}.utilization_ppm does not match aggregate device time"
        )


def _decision(value: Any, path: str) -> str:
    decision = _text(value, path)
    if decision not in {
        "materially_faster",
        "performance_equivalent",
        "materially_slower",
        "not_materially_faster",
        "inconclusive",
        "invalid",
    }:
        raise BenchmarkContractError(f"{path} is unsupported")
    return decision


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(f"{path} must be an integer")
    return value


def _implementation(value: Any, path: str) -> None:
    implementation = _object(value, path)
    _fields(
        implementation,
        {
            "implementation_id",
            "contract_digest",
            "artifact_refs",
        },
        path,
    )
    _text(implementation["implementation_id"], f"{path}.implementation_id")
    _contract_digest(implementation["contract_digest"], f"{path}.contract_digest")
    refs = _list(implementation["artifact_refs"], f"{path}.artifact_refs")
    paths = []
    for index, ref in enumerate(refs):
        _artifact_ref(ref, f"{path}.artifact_refs[{index}]")
        paths.append(ref["path"])
    if not paths or paths != sorted(set(paths)):
        raise BenchmarkContractError(
            f"{path}.artifact_refs must be non-empty, sorted, and unique"
        )


def validate_matched_conditions(conditions: Json) -> None:
    _fields(
        conditions,
        {
            "devices",
            "placement",
            "controls",
            "environment",
            "capacity_reservation_digest",
            "residency_scope",
        },
        "matched_conditions",
    )
    devices = _list(conditions["devices"], "matched_conditions.devices")
    device_ids = []
    for index, raw_device in enumerate(devices):
        path = f"matched_conditions.devices[{index}]"
        device = _object(raw_device, path)
        _fields(
            device,
            {
                "device_id",
                "hardware_profile_digest",
                "capability_class",
                "api",
            },
            path,
        )
        device_ids.append(_text(device["device_id"], f"{path}.device_id"))
        _contract_digest(
            device["hardware_profile_digest"],
            f"{path}.hardware_profile_digest",
        )
        _text(device["capability_class"], f"{path}.capability_class")
        _text(device["api"], f"{path}.api")
    if not device_ids or device_ids != sorted(set(device_ids)):
        raise BenchmarkContractError(
            "matched condition devices must be non-empty, sorted, and unique"
        )
    placement = _object(
        conditions["placement"],
        "matched_conditions.placement",
    )
    controls = _object(
        conditions["controls"],
        "matched_conditions.controls",
    )
    environment = _object(
        conditions["environment"],
        "matched_conditions.environment",
    )
    if not placement or not controls or not environment:
        raise BenchmarkContractError(
            "matched placement, controls, and environment must be explicit"
        )
    if any(
        not isinstance(device_id, str) or device_id not in device_ids
        for device_id in placement.values()
    ):
        raise BenchmarkContractError(
            "matched placement references an undeclared device"
        )
    _device_state_digest(
        conditions["capacity_reservation_digest"],
        "matched_conditions.capacity_reservation_digest",
    )
    if conditions["residency_scope"] != "capacity_partition":
        raise BenchmarkContractError(
            "matched benchmark conditions require capacity-partition residency"
        )


def _benchmark_policy(policy: Json) -> None:
    _fields(
        policy,
        {
            "minimum_warmup_samples",
            "maximum_warmup_samples",
            "warmup_stability_window_samples",
            "measured_calls_per_role",
            "maximum_benchmark_duration_ns",
            "minimum_material_improvement_ppm",
            "maximum_material_regression_ppm",
            "maximum_sustained_regression_ppm",
        },
        "policy",
    )
    minimum_warmup = _positive(
        policy["minimum_warmup_samples"],
        "policy.minimum_warmup_samples",
    )
    maximum_warmup = _positive(
        policy["maximum_warmup_samples"],
        "policy.maximum_warmup_samples",
    )
    stability_window = _positive(
        policy["warmup_stability_window_samples"],
        "policy.warmup_stability_window_samples",
    )
    if maximum_warmup < minimum_warmup:
        raise BenchmarkContractError(
            "policy warmup bound must cover the minimum"
        )
    if (
        minimum_warmup != 1
        or maximum_warmup != 1
        or stability_window != 1
    ):
        raise BenchmarkContractError(
            "microbenchmark policy requires one fixed warmup per role"
        )
    measured_calls = _positive(
        policy["measured_calls_per_role"],
        "policy.measured_calls_per_role",
    )
    if measured_calls != 1:
        raise BenchmarkContractError(
            "microbenchmark policy requires exactly one measured call per role"
        )
    maximum_duration = _positive(
        policy["maximum_benchmark_duration_ns"],
        "policy.maximum_benchmark_duration_ns",
    )
    if maximum_duration > 60_000_000_000:
        raise BenchmarkContractError(
            "microbenchmark duration must not exceed one minute"
        )
    for field in (
        "minimum_material_improvement_ppm",
        "maximum_material_regression_ppm",
        "maximum_sustained_regression_ppm",
    ):
        value = _nonnegative(policy[field], f"policy.{field}")
        if value >= 1_000_000:
            raise BenchmarkContractError(f"policy.{field} must be below 1000000")


def _artifact_ref(value: Any, path: str) -> None:
    ref = _object(value, path)
    _fields(ref, {"path", "digest"}, path)
    _relative_path(ref["path"], f"{path}.path")
    _artifact_digest(ref["digest"], f"{path}.digest")


def _fixture_ref(value: Any, path: str) -> None:
    _artifact_ref(value, path)


def _relative_path(value: Any, path: str) -> str:
    text = _text(value, path)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise BenchmarkContractError(f"{path} must be a normalized relative path")
    return text


def _content_id(prefix: str, identity_field: str, document: Json) -> str:
    payload = deepcopy(document)
    payload.pop(identity_field, None)
    return stable_contract_id(prefix, payload)


def _content_identity(
    document: Json,
    prefix: str,
    identity_field: str,
    label: str,
) -> None:
    expected = _content_id(prefix, identity_field, document)
    if document[identity_field] != expected:
        raise BenchmarkContractError(
            f"{label} identity does not match canonical content"
        )


def _schema(document: Json, expected: str, path: str) -> None:
    if document["schema"] != expected:
        raise BenchmarkContractError(f"{path} schema is unsupported")


def _fields(record: Json, expected: set[str], path: str) -> None:
    if not isinstance(record, dict):
        raise BenchmarkContractError(f"{path} must be an object")
    actual = set(record)
    if actual != expected:
        raise BenchmarkContractError(
            f"{path} fields are invalid: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _object(value: Any, path: str) -> Json:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkContractError(f"{path} must be a list")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkContractError(f"{path} must be boolean")
    return value


def _nonnegative(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkContractError(f"{path} must be a non-negative integer")
    return value


def _positive(value: Any, path: str) -> int:
    result = _nonnegative(value, path)
    if result == 0:
        raise BenchmarkContractError(f"{path} must be positive")
    return result


def _u32(value: Any, path: str) -> int:
    result = _nonnegative(value, path)
    if result > 0xFFFF_FFFF:
        raise BenchmarkContractError(f"{path} must fit in U32")
    return result


def _string_list(value: Any, path: str) -> list[str]:
    values = _list(value, path)
    for index, item in enumerate(values):
        _text(item, f"{path}[{index}]")
    return values


def _sorted_strings(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _string_list(value, path)
    if nonempty and not values:
        raise BenchmarkContractError(f"{path} must not be empty")
    if values != sorted(set(values)):
        raise BenchmarkContractError(f"{path} must be sorted and unique")
    return values


def _stable_id(value: Any, prefix: str, path: str) -> str:
    identifier = _text(value, path)
    hexadecimal = identifier.removeprefix(f"{prefix}_")
    if (
        not identifier.startswith(f"{prefix}_")
        or len(identifier) != len(prefix) + 33
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise BenchmarkContractError(f"{path} is not a stable {prefix} id")
    return identifier


def _hex_digest(value: Any, prefix: str, path: str) -> str:
    digest = _text(value, path)
    hexadecimal = digest.removeprefix(prefix)
    if (
        not digest.startswith(prefix)
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise BenchmarkContractError(f"{path} is not a recognized digest")
    return digest


def _artifact_digest(value: Any, path: str) -> str:
    return _hex_digest(value, _ARTIFACT_DIGEST_PREFIX, path)


def _device_state_digest(value: Any, path: str) -> str:
    return _hex_digest(
        value,
        f"{DEVICE_STATE_DIGEST_SCHEMA}:",
        path,
    )


def _contract_digest(value: Any, path: str) -> str:
    return _hex_digest(value, _CONTRACT_DIGEST_PREFIX, path)


Validator = Callable[[Json], None]

BENCHMARK_SCHEMA_VALIDATORS: dict[str, Validator] = {
    BENCHMARK_WORKLOAD_SCHEMA: validate_benchmark_workload,
    BENCHMARK_PLAN_SCHEMA: validate_benchmark_plan,
    BENCHMARK_OBSERVATION_SCHEMA: validate_benchmark_observation,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA: validate_benchmark_residency_event,
    BENCHMARK_RUN_SCHEMA: validate_benchmark_run,
    BENCHMARK_RECORD_SCHEMA: validate_benchmark_record,
}
