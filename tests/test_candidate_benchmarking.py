from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import time

import pytest

from nerve.compilation import ModelCompileCancelled, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_OBSERVATION_SCHEMA,
    BENCHMARK_RESIDENCY_EVENT_SCHEMA,
    BenchmarkObservation,
    BenchmarkPlan,
    BenchmarkWorkload,
    benchmark_observation_id,
    benchmark_plan_id,
    benchmark_residency_event_id,
    benchmark_workload_id,
)
from nerve.representation_optimizer.benchmarking.orchestrator import (
    benchmark_candidate,
)
from nerve.representation_optimizer.benchmarking.planning import (
    build_benchmark_plan,
)
from nerve.representation_optimizer.benchmarking.runner import (
    execute_benchmark_plan,
)
from nerve.representation_optimizer.benchmarking.statistics import (
    summarize_benchmark,
)
from nerve.representation_optimizer.benchmarking.storage import (
    load_benchmark_evidence,
)
from nerve.representation_optimizer.contracts import (
    contract_digest,
    device_state_digest,
    stable_contract_id,
)
from nerve.representation_optimizer.lifecycle import CandidateState
from nerve.representation_optimizer.providers.source_artifacts import (
    PackageSourceArtifactResolver,
)
from nerve.representation_optimizer.staging.contracts import (
    staged_artifact_digest,
)
from nerve.representation_optimizer.staging.orchestrator import stage_candidate
from tests.test_candidate_staging import (
    CompletePhysicalOptimizer,
    CompleteRelowerer,
    CompleteSemanticConstructor,
    _package,
    _plan,
    _session_with_candidate,
)
from tests.test_representation_optimizer_contracts import (
    hardware_profile_contract,
)


@dataclass
class AdapterBehavior:
    candidate_duration_ns: int = 800_000
    candidate_duration_ns_by_execution_phase: dict[str, int] | None = None
    candidate_measured_durations_ns: tuple[int, ...] | None = None
    reference_duration_ns: int = 1_000_000
    order_biased_candidate: bool = False
    candidate_sustained_regression: bool = False
    candidate_warmup_drift: bool = False
    candidate_resident_warmup_failures: int = 0
    candidate_initial_warmup_penalty_samples: int = 0
    candidate_initial_noisy_measured_pairs: int = 0
    timeout_count: int = 0
    reproducibility_defect: bool = False
    speculative_schedule_variance: bool = False
    sampling_variance: bool = False
    numerical_variance: bool = False


class FixtureExecutionAdapter:
    def __init__(self, behavior: AdapterBehavior | None = None) -> None:
        self.behavior = behavior or AdapterBehavior()
        self.mount_requests = []
        self.execution_requests = []
        self.closed_sessions = 0
        self.fixture_candidate_ids: list[str] = []
        self.trace_artifacts: dict[str, bytes] = {}
        self.fixture_artifacts = {
            "fixtures/decode-input.bin": b"fixture input",
            "fixtures/decode-state.bin": b"fixture state",
            "fixtures/prefill-input.bin": b"fixture prefill input",
            "fixtures/prefill-state.bin": b"fixture prefill state",
            "fixtures/model-limits.json": b'{"max_output_tokens":65536}',
        }

    def iter_fixture_artifact(
        self,
        relative_path,
        *,
        candidate_id,
        chunk_bytes=8 * 1024 * 1024,
    ):
        self.fixture_candidate_ids.append(candidate_id)
        payload = self.fixture_artifacts[relative_path]
        for offset in range(0, len(payload), chunk_bytes):
            yield payload[offset : offset + chunk_bytes]

    def open_session(self, request):
        self.mount_requests.append(request)
        return FixtureExecutionSession(self, request)

    def iter_trace_artifact(self, relative_path, *, chunk_bytes=8 * 1024 * 1024):
        payload = self.trace_artifacts[relative_path]
        for offset in range(0, len(payload), chunk_bytes):
            yield payload[offset : offset + chunk_bytes]


class DeadlineAwareExecutionAdapter(FixtureExecutionAdapter):
    def open_session(self, request):
        self.mount_requests.append(request)
        return DeadlineAwareExecutionSession(self, request)


class DeadlineAwareFixtureAdapter(FixtureExecutionAdapter):
    def iter_fixture_artifact(
        self,
        relative_path,
        *,
        candidate_id,
        chunk_bytes=8 * 1024 * 1024,
    ):
        del relative_path, candidate_id, chunk_bytes
        while True:
            yield b""


class FixtureExecutionSession:
    def __init__(self, adapter: FixtureExecutionAdapter, request) -> None:
        self.adapter = adapter
        self.request = request
        self.closed = False
        behavior = adapter.behavior
        self.fail_resident_warmup = (
            request.role == "candidate"
            and request.workload["regime"]["mount_mode"] == "resident_reuse"
            and behavior.candidate_resident_warmup_failures > 0
        )
        if self.fail_resident_warmup:
            behavior.candidate_resident_warmup_failures -= 1
        self.mounted_state = device_state_digest(
            {
                "fixture_state": "mounted",
                "role": request.role,
                "block_index": request.block_index,
                "seed": request.seed,
            }
        )
        self._mount_event = self._event(
            action="mount",
            before=request.matched_conditions["capacity_reservation_digest"],
            after=self.mounted_state,
            released=False,
        )

    @property
    def mount_event(self):
        return dict(self._mount_event)

    def execute(self, request):
        if self.closed:
            raise RuntimeError("fixture execution session is closed")
        self.adapter.execution_requests.append(request)
        workload = request.workload
        useful_units = workload["useful_work"]["minimum_units"]
        duration = self._duration(request)
        window_count = workload["useful_work"]["sustained_window_count"]
        width = useful_units // max(1, window_count)
        windows = []
        start = 0
        window_weights = [
            (
                1.0 + 0.20 * index
                if (
                    self.adapter.behavior.candidate_sustained_regression
                    and request.role == "candidate"
                )
                else 1.0
            )
            for index in range(window_count)
        ]
        remaining_duration = duration
        for index in range(window_count):
            end = useful_units if index == window_count - 1 else start + width
            window_duration = (
                remaining_duration
                if index == window_count - 1
                else max(
                    1,
                    round(duration * window_weights[index] / sum(window_weights)),
                )
            )
            remaining_duration -= window_duration
            windows.append(
                {
                    "index": index,
                    "start_unit": start,
                    "end_unit": end,
                    "duration_ns": window_duration,
                }
            )
            start = end
        behavior = self.adapter.behavior
        token_label = f"tokens:{request.seed}:{request.role}"
        if behavior.reproducibility_defect and request.role == "candidate":
            token_label += f":{request.pair_index}"
        if behavior.sampling_variance and request.role == "candidate":
            token_label += f":sampling:{request.pair_index}"
        if behavior.numerical_variance and request.role == "candidate":
            token_label += f":numerical:{request.pair_index}"
        distribution_label = f"distribution:{request.seed}:{request.role}"
        state_label = f"state:{request.seed}:{request.role}"
        random_label = f"random:{request.seed}"
        schedule_label = f"schedule:{request.seed}:{request.role}:{request.order_index}"
        if behavior.sampling_variance and request.role == "candidate":
            random_label += f":{request.pair_index}"
        if behavior.numerical_variance and request.role == "candidate":
            distribution_label += f":{request.pair_index}"
            state_label += f":{request.pair_index}"
        if behavior.speculative_schedule_variance and request.role == "candidate":
            schedule_label += f":{request.pair_index}"
        trace_payloads = {
            "distribution": distribution_label.encode(),
            "tokens": token_label.encode(),
            "state": state_label.encode(),
            "random_draws": random_label.encode(),
            "schedule": schedule_label.encode(),
        }
        traces = {}
        for name, payload in trace_payloads.items():
            path = (
                f"traces/{workload['workload_id']}/{request.role}/"
                f"{request.seed}/{request.block_index}/"
                f"{request.phase}/{request.order_index}/"
                f"{request.pair_index}/{name}.bin"
            )
            self.adapter.trace_artifacts[path] = payload
            traces[name] = {
                "path": path,
                "digest": staged_artifact_digest(payload),
            }
        busy_ns = duration * 4 // 5
        document = {
            "schema": BENCHMARK_OBSERVATION_SCHEMA,
            "observation_id": "",
            "plan_id": request.plan_id,
            "implementation_id": request.implementation_id,
            "role": request.role,
            "workload_id": workload["workload_id"],
            "phase": request.phase,
            "seed": request.seed,
            "block_index": request.block_index,
            "pair_index": request.pair_index,
            "order_index": request.order_index,
            "matched_conditions_digest": request.matched_conditions_digest,
            "input_digest": workload["input"]["digest"],
            "initial_state_digest": (
                workload["initial_state"]["digest"]
                if workload["initial_state"] is not None
                else None
            ),
            "controls_digest": contract_digest(workload["controls"]),
            "status": "completed",
            "stop_reason": workload["useful_work"]["completion_condition"],
            "timing": {
                "setup_ns": 0,
                "execution_ns": duration,
                "teardown_ns": 0,
                "queue_wait_ns": 2_000,
            },
            "work": {
                "unit": workload["useful_work"]["unit"],
                "useful_units": useful_units,
                "speculative_units": 8 if request.role == "candidate" else 0,
                "cancelled_units": 0,
                "discarded_units": 2 if request.role == "candidate" else 0,
                "corrective_units": 1 if request.role == "candidate" else 0,
            },
            "memory": {
                "permanent_bytes": 128 if request.role == "candidate" else 256,
                "peak_transient_bytes": 64,
                "resident_before_bytes": 128,
                "resident_peak_bytes": 192,
                "resident_after_bytes": 128,
            },
            "representation": {
                "conversion_bytes": (16 if request.role == "candidate" else 0),
                "conversion_ns": 1_000 if request.role == "candidate" else 0,
                "boundary_count": 1 if request.role == "candidate" else 0,
            },
            "device": {
                "measurement_ns": duration,
                "busy_ns": busy_ns,
                "utilization_ppm": round(busy_ns * 1_000_000 / duration),
            },
            "synchronization": {
                "operation_count": 2,
                "wait_ns": 1_000,
            },
            "transport": {
                "bytes": (
                    32 if workload["regime"]["boundary_mode"] == "cross_device" else 0
                ),
                "duration_ns": 500,
                "queue_wait_count": 1,
                "queue_wait_ns": 500,
                "timeout_count": behavior.timeout_count,
            },
            "throughput_windows": windows,
            "traces": traces,
            "default_statistics": {
                "execution_path": "normal_fixture_runtime",
                "decode_tokens": useful_units,
                "discarded_ticks": 2,
                "bounded_wait_timeouts": behavior.timeout_count,
            },
            "diagnostics": [],
        }
        document["observation_id"] = benchmark_observation_id(document)
        return BenchmarkObservation.from_json(document).to_json()

    def close(self):
        if self.closed:
            raise RuntimeError("fixture session closed twice")
        self.closed = True
        self.adapter.closed_sessions += 1
        return self._event(
            action="unmount",
            before=self.mounted_state,
            after=self.request.matched_conditions["capacity_reservation_digest"],
            released=True,
        )

    def _duration(self, request) -> int:
        behavior = self.adapter.behavior
        duration = (
            (
                behavior.candidate_duration_ns_by_execution_phase or {}
            ).get(
                request.workload["regime"]["execution_phase"],
                behavior.candidate_duration_ns,
            )
            if request.role == "candidate"
            else behavior.reference_duration_ns
        )
        if (
            request.role == "candidate"
            and request.phase == "measured"
            and behavior.candidate_measured_durations_ns
        ):
            duration = behavior.candidate_measured_durations_ns[
                request.pair_index % len(behavior.candidate_measured_durations_ns)
            ]
        jitter = (-2, -1, 0, 1, 2)[request.pair_index % 5]
        duration += jitter * 1_000
        if behavior.order_biased_candidate and request.role == "candidate":
            duration = (
                round(duration * 0.60)
                if request.order_index == 0
                else round(duration * 1.40)
            )
        if (
            (behavior.candidate_warmup_drift or self.fail_resident_warmup)
            and request.role == "candidate"
            and request.phase == "warmup"
        ):
            duration = round(duration * (1.0 + 0.10 * request.pair_index))
        if (
            request.role == "candidate"
            and request.phase == "warmup"
            and request.pair_index < behavior.candidate_initial_warmup_penalty_samples
        ):
            duration = round(duration * 1.50)
        if (
            request.role == "candidate"
            and request.phase == "measured"
            and request.pair_index < behavior.candidate_initial_noisy_measured_pairs
        ):
            duration = round(duration * (1.30 if request.pair_index % 2 == 0 else 0.75))
        return duration

    def _event(self, *, action, before, after, released):
        request = self.request
        document = {
            "schema": BENCHMARK_RESIDENCY_EVENT_SCHEMA,
            "event_id": "",
            "plan_id": request.plan_id,
            "implementation_id": request.implementation["implementation_id"],
            "role": request.role,
            "workload_id": request.workload["workload_id"],
            "seed": request.seed,
            "block_index": request.block_index,
            "action": action,
            "duration_ns": 100_000,
            "permanent_bytes": 128 if request.role == "candidate" else 256,
            "peak_transient_bytes": 64,
            "matched_conditions_digest": request.matched_conditions_digest,
            "device_state_before_digest": before,
            "device_state_after_digest": after,
            "released": released,
            "default_statistics": {
                "execution_path": "normal_fixture_runtime",
                "action": action,
            },
        }
        document["event_id"] = benchmark_residency_event_id(document)
        return document


class DeadlineAwareExecutionSession(FixtureExecutionSession):
    def execute(self, request):
        del request
        while True:
            if (
                self.request.cancel_requested is not None
                and self.request.cancel_requested()
            ):
                raise ModelCompileCancelled("deadline reached")
            time.sleep(0.001)


def _fixture(
    tmp_path: Path,
    behavior: AdapterBehavior | None = None,
    *,
    randomness_overrides: dict[str, bool] | None = None,
):
    package_dir, source_session = _package(tmp_path)
    candidate_plan = _plan()
    if randomness_overrides:
        workloads = []
        for workload in candidate_plan.benchmark_workloads:
            document = workload.to_json()
            document["randomness"].update(randomness_overrides)
            document["workload_id"] = benchmark_workload_id(document)
            workloads.append(BenchmarkWorkload.from_json(document))
        candidate_plan = replace(
            candidate_plan,
            benchmark_workloads=tuple(workloads),
        )
    source_session = _session_with_candidate(source_session, candidate_plan)
    construction = stage_candidate(
        package_dir=package_dir,
        source_artifacts=PackageSourceArtifactResolver(package_dir),
        workspace_root=tmp_path / "candidate-workspace",
        plan=candidate_plan,
        session=source_session,
        semantic_constructor=CompleteSemanticConstructor([]),
        ordinary_relowerer=CompleteRelowerer([]),
        physical_optimizer=CompletePhysicalOptimizer([]),
    )
    statically_validated = construction.session.transition_candidate(
        candidate_plan.candidate_id,
        CandidateState.STATICALLY_VALIDATED,
        evidence_refs=("static-validation.json",),
        reason="fixture static validation passed",
    )
    prebenchmark_validated = statically_validated.transition_candidate(
        candidate_plan.candidate_id,
        CandidateState.PREBENCHMARK_VALIDATED,
        evidence_refs=("prebenchmark-validation.json",),
        reason="fixture proof and behavioral sanity passed",
    )
    profile = hardware_profile_contract()
    plan = build_benchmark_plan(
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        hardware_profiles=(profile,),
        reference_implementation_id="exact-reference",
        reference_contract_digest=candidate_plan.candidate.to_json()[
            "source_contract_digests"
        ][0],
        reference_artifact_refs=(
            {
                "path": "lowered/exact-reference.json",
                "digest": staged_artifact_digest(b"exact reference"),
            },
        ),
        matched_conditions={
            "devices": [
                {
                    "device_id": profile["hardware_identity"]["stable_device_id"],
                    "hardware_profile_digest": contract_digest(profile),
                    "capability_class": profile["capability_class"],
                    "api": profile["provenance"]["api"],
                }
            ],
            "placement": {"fixture_scope": "vulkan:fixture"},
            "controls": {"scheduler": "normal"},
            "environment": {"power_profile": "matched"},
            "capacity_reservation_digest": device_state_digest({"fixture_state": "capacity_available"}),
            "residency_scope": "capacity_partition",
        },
    )
    return (
        candidate_plan,
        construction,
        prebenchmark_validated,
        plan,
        FixtureExecutionAdapter(behavior),
    )


def test_matched_benchmark_promotes_only_measured_material_speedup(
    tmp_path: Path,
) -> None:
    _, construction, session, plan, adapter = _fixture(tmp_path)

    outcome = benchmark_candidate(
        plan=plan,
        construction_record=construction.record,
        session=session,
        adapter=adapter,
        workspace_root=tmp_path / "benchmark-workspace",
    )

    record = outcome.record.to_json()
    assert record["decision"] == "materially_faster"
    assert all(
        workload["paired"]["candidate_is_faster"]
        and workload["paired"]["speedup_ppm"] > 50_000
        for workload in record["workloads"]
    )
    assert record["resource_measurements"]["roles"]["candidate"]["discarded_units"] > 0
    assert record["resource_measurements"]["roles"]["candidate"]["conversion_bytes"] > 0
    assert (
        record["resource_measurements"]["roles"]["reference"]["permanent_bytes"] == 256
    )
    candidate_resources = record["resource_measurements"]["roles"]["candidate"]
    assert candidate_resources["setup_ns"] > 0
    assert candidate_resources["teardown_ns"] > 0
    assert candidate_resources["synchronization_count"] > 0
    assert candidate_resources["synchronization_wait_ns"] > 0
    assert candidate_resources["queue_wait_count"] > 0
    assert candidate_resources["queue_wait_ns"] > 0
    assert candidate_resources["transport_bytes"] > 0
    assert candidate_resources["boundary_count"] > 0
    assert candidate_resources["resident_peak_bytes"] >= max(
        candidate_resources["resident_before_bytes"],
        candidate_resources["resident_after_bytes"],
    )
    assert all(
        item["classification"] == "identical" for item in record["reproducibility"]
    )
    assert set(adapter.fixture_candidate_ids) == {plan.candidate_id}
    lifecycle = next(
        candidate
        for candidate in outcome.session.candidates
        if candidate.candidate_id == plan.candidate_id
    )
    assert lifecycle.state == CandidateState.BENCHMARKED
    loaded = load_benchmark_evidence(
        tmp_path / "benchmark-workspace",
        record["benchmark_id"],
    )
    assert loaded == (plan, outcome.run, outcome.record)
    assert (
        outcome.evidence_path / "fixtures/decode-input.bin"
    ).read_bytes() == b"fixture input"
    assert (
        outcome.evidence_path / "fixtures/model-limits.json"
    ).read_bytes() == b'{"max_output_tokens":65536}'
    assert adapter.closed_sessions == len(adapter.mount_requests)
    regimes = [workload.to_json()["regime"] for workload in plan.workloads]
    assert {regime["execution_phase"] for regime in regimes} == {
        "decode",
        "prefill",
    }
    assert {regime["mount_mode"] for regime in regimes} == {
        "cold",
        "resident_reuse",
    }
    assert any(regime["activation_batch_width"] > 1 for regime in regimes)
    assert any(regime["stream_count"] > 1 for regime in regimes)
    assert any(regime["boundary_mode"] == "cross_device" for regime in regimes)
    assert all(
        observation["default_statistics"]["execution_path"] == "normal_fixture_runtime"
        for observation in outcome.run.to_json()["observations"]
    )
    assert all(
        event["default_statistics"]["execution_path"] == "normal_fixture_runtime"
        for event in outcome.run.to_json()["residency_events"]
    )


def test_slower_candidate_is_not_materially_faster(tmp_path: Path) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_duration_ns=1_100_000),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "not_materially_faster"
    assert all(
        workload["decision"] == "materially_slower"
        and workload["paired"]["speedup_ppm"] < 0
        for workload in record["workloads"]
    )
    assert {
        outcome["termination"] for outcome in run.to_json()["sampling_outcomes"]
    } == {"fixed_sample_complete"}
    assert {
        outcome["measured_calls_per_role"]
        for outcome in run.to_json()["sampling_outcomes"]
    } == {plan.policy["measured_calls_per_role"]}


def test_small_measured_speedup_answers_binary_faster_question(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_duration_ns=970_000),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "materially_faster"
    assert all(
        workload["decision"] == "materially_faster"
        and any(
            "execution is faster" in reason
            for reason in workload["reasons"]
        )
        for workload in record["workloads"]
    )
    assert {
        outcome["termination"] for outcome in run.to_json()["sampling_outcomes"]
    } == {"fixed_sample_complete"}


def test_one_material_win_and_one_equivalent_regime_is_a_pareto_win(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(
            candidate_duration_ns_by_execution_phase={
                "decode": 800_000,
                "prefill": 1_000_000,
            }
        ),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "materially_faster"
    assert {
        workload["decision"] for workload in record["workloads"]
    } == {"materially_faster", "performance_equivalent"}
    assert {
        outcome["termination"] for outcome in run.to_json()["sampling_outcomes"]
    } == {"fixed_sample_complete"}
    assert all(
        outcome["measured_calls_per_role"]
        == plan.policy["measured_calls_per_role"]
        for outcome in run.to_json()["sampling_outcomes"]
    )


def test_binary_decision_uses_one_measured_call_per_role(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(
            candidate_measured_durations_ns=(300_000,),
        ),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "materially_faster"
    assert all(
        workload["decision"] == "materially_faster"
        and workload["paired"]["speedup_ppm"] > 0
        and workload["paired"]["candidate_is_faster"]
        and workload["sample_count_per_role"] == 1
        for workload in record["workloads"]
    )


def test_fixed_warmup_is_discarded_without_adaptive_sampling(tmp_path: Path) -> None:
    _, _, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_warmup_drift=True),
    )
    run = execute_benchmark_plan(plan, adapter).to_json()

    assert run["status"] == "completed"
    assert run["sampling_outcomes"]
    assert {
        observation["phase"] for observation in run["observations"]
    } == {"warmup", "measured"}
    assert all(
        group["sample_count"] == 1
        for outcome in run["sampling_outcomes"]
        for group in outcome["warmup_groups"]
    )
    assert adapter.closed_sessions == len(adapter.mount_requests)


def test_microbenchmark_never_retries_a_resident_block(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_resident_warmup_failures=1),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert run.to_json()["status"] == "completed"
    assert record["decision"] == "materially_faster"
    groups = [
        group
        for outcome in run.to_json()["sampling_outcomes"]
        for group in outcome["warmup_groups"]
    ]
    assert groups
    assert {group["attempt_index"] for group in groups} == {0}
    assert {group["sample_count"] for group in groups} == {1}
    # One resident and one cold workload: warmup plus measurement for both roles.
    assert len(adapter.execution_requests) == 8


def test_binary_screen_rejects_candidate_that_is_not_faster(tmp_path: Path) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_measured_durations_ns=(1_050_000,)),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "not_materially_faster"
    assert {
        outcome["measured_calls_per_role"]
        for outcome in run.to_json()["sampling_outcomes"]
    } == {plan.policy["measured_calls_per_role"]}
    assert {
        outcome["termination"] for outcome in run.to_json()["sampling_outcomes"]
    } == {"fixed_sample_complete"}


def test_one_fixed_cold_warmup_is_discarded(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_initial_warmup_penalty_samples=1),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    candidate_groups = [
        group
        for outcome in run.to_json()["sampling_outcomes"]
        for group in outcome["warmup_groups"]
        if group["role"] == "candidate"
    ]
    assert candidate_groups
    assert all(group["converged"] for group in candidate_groups)
    assert {group["sample_count"] for group in candidate_groups} == {1}
    assert record["decision"] == "materially_faster"


def test_microbenchmark_always_stops_after_one_measured_call(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(
            candidate_duration_ns=600_000,
            candidate_initial_noisy_measured_pairs=6,
        ),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    measured_counts = {
        outcome["measured_calls_per_role"]
        for outcome in run.to_json()["sampling_outcomes"]
    }
    assert measured_counts == {plan.policy["measured_calls_per_role"]}
    assert record["decision"] == "materially_faster"


def test_measured_calls_use_one_declared_order(
    tmp_path: Path,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    run = execute_benchmark_plan(plan, adapter).to_json()
    for workload_index, workload in enumerate(plan.to_json()["workloads"]):
        outcome = next(
            item
            for item in run["sampling_outcomes"]
            if item["workload_id"] == workload["workload_id"]
        )
        for seed_index, seed in enumerate(workload["randomness"]["seeds"]):
            references = sorted(
                (
                    observation
                    for observation in run["observations"]
                    if observation["workload_id"] == workload["workload_id"]
                    and observation["seed"] == seed
                    and observation["phase"] == "measured"
                    and observation["role"] == "reference"
                ),
                key=lambda observation: observation["pair_index"],
            )
            assert len(references) == outcome["measured_calls_per_role"]
            for observation in references:
                expected = (
                    (
                        "reference",
                        "candidate",
                    )
                    if (workload_index + seed_index) % 2 == 0
                    else (
                        "candidate",
                        "reference",
                    )
                )
                assert observation["order_index"] == expected.index("reference")


def test_binary_microbenchmark_does_not_multiply_requests_for_order_statistics(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(order_biased_candidate=True),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] in {
        "materially_faster",
        "not_materially_faster",
    }
    assert all(
        workload["sample_count_per_role"] == 1
        for workload in record["workloads"]
    )
    assert len(adapter.execution_requests) == 8


def test_sustained_throughput_regression_is_inconclusive(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(candidate_sustained_regression=True),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "inconclusive"
    assert not record["workloads"][0]["sustained"]["passed"]


def test_default_runtime_timeout_counter_invalidates_benchmark(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(timeout_count=1),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "invalid"
    assert record["resource_measurements"]["roles"]["candidate"]["timeout_count"] > 0


def test_microbenchmark_defers_trace_correctness_to_behavioral_validation(
    tmp_path: Path,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        AdapterBehavior(reproducibility_defect=True),
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["decision"] == "materially_faster"
    assert record["reproducibility"] == []


@pytest.mark.parametrize(
    ("behavior", "randomness_flag"),
    (
        (
            AdapterBehavior(speculative_schedule_variance=True),
            "permit_speculative_schedule_variance",
        ),
        (
            AdapterBehavior(sampling_variance=True),
            "permit_sampling_variance",
        ),
        (
            AdapterBehavior(numerical_variance=True),
            "permit_numerical_nondeterminism",
        ),
    ),
)
def test_microbenchmark_retains_raw_traces_without_extra_repetitions(
    tmp_path: Path,
    behavior: AdapterBehavior,
    randomness_flag: str,
) -> None:
    _, construction, _, plan, adapter = _fixture(
        tmp_path,
        behavior,
        randomness_overrides={randomness_flag: True},
    )
    run = execute_benchmark_plan(plan, adapter)
    record = summarize_benchmark(
        plan=plan,
        run=run,
        construction_record=construction.record,
    ).to_json()

    assert record["reproducibility"] == []
    assert record["raw_evidence"]["trace_artifact_count"] > 0
    assert record["decision"] == "materially_faster"


def test_microbenchmark_fails_when_one_minute_contract_is_exceeded(
    tmp_path: Path,
) -> None:
    _, _, _, plan, _ = _fixture(tmp_path)
    document = plan.to_json()
    document["policy"]["maximum_benchmark_duration_ns"] = 5_000_000
    document["plan_id"] = benchmark_plan_id(document)
    plan = BenchmarkPlan.from_json(document)
    adapter = DeadlineAwareExecutionAdapter()

    started = time.monotonic()
    run = execute_benchmark_plan(plan, adapter).to_json()
    elapsed = time.monotonic() - started

    assert run["status"] == "timeout"
    assert run["diagnostics"] == [
        "microbenchmark exceeded its one-minute wall-clock contract"
    ]
    assert run["observations"] == []
    assert elapsed < 1
    assert adapter.closed_sessions == len(adapter.mount_requests) == 1


def test_microbenchmark_deadline_also_bounds_fixture_verification(
    tmp_path: Path,
) -> None:
    _, _, _, plan, _ = _fixture(tmp_path)
    document = plan.to_json()
    document["policy"]["maximum_benchmark_duration_ns"] = 5_000_000
    document["plan_id"] = benchmark_plan_id(document)
    plan = BenchmarkPlan.from_json(document)
    adapter = DeadlineAwareFixtureAdapter()

    started = time.monotonic()
    run = execute_benchmark_plan(plan, adapter).to_json()

    assert run["status"] == "timeout"
    assert run["observations"] == []
    assert run["residency_events"] == []
    assert time.monotonic() - started < 1
    assert adapter.mount_requests == []


def test_external_cancellation_is_not_misreported_as_benchmark_timeout(
    tmp_path: Path,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)

    with pytest.raises(ModelCompileCancelled, match="cancelled"):
        execute_benchmark_plan(
            plan,
            adapter,
            cancel_requested=lambda: True,
        )


def test_fixture_bytes_are_verified_before_any_implementation_mount(
    tmp_path: Path,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    adapter.fixture_artifacts["fixtures/decode-input.bin"] = b"corrupt"

    with pytest.raises(ModelCompileError, match="fixture failed digest"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.mount_requests == []


def test_raw_trace_digest_is_verified_before_accepting_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)

    def corrupt_trace(self, relative_path, *, chunk_bytes=8 * 1024 * 1024):
        del self, relative_path, chunk_bytes
        yield b"corrupt"

    monkeypatch.setattr(
        FixtureExecutionAdapter,
        "iter_trace_artifact",
        corrupt_trace,
    )
    with pytest.raises(ModelCompileError, match="failed digest validation"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests)


def test_matched_pair_cannot_report_different_useful_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    original = FixtureExecutionSession.execute

    def mismatched_work(self, request):
        document = original(self, request)
        if request.role == "candidate":
            document["work"]["useful_units"] += 1
            document["throughput_windows"][-1]["end_unit"] += 1
            document["observation_id"] = benchmark_observation_id(document)
        return document

    monkeypatch.setattr(FixtureExecutionSession, "execute", mismatched_work)
    with pytest.raises(ModelCompileError, match="different useful work"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests)


def test_normal_execution_cannot_exceed_declared_output_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    original = FixtureExecutionSession.execute

    def overrun(self, request):
        document = original(self, request)
        allowance = request.workload["useful_work"]["output_allowance"]
        if request.role == "candidate" and allowance is not None:
            document["work"]["useful_units"] = allowance + 1
            document["throughput_windows"][-1]["end_unit"] = allowance + 1
            document["observation_id"] = benchmark_observation_id(document)
        return document

    monkeypatch.setattr(FixtureExecutionSession, "execute", overrun)
    with pytest.raises(ModelCompileError, match="exceeded.*output allowance"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests)


def test_mismatched_mount_evidence_still_closes_open_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    original = FixtureExecutionAdapter.open_session

    def mismatched_mount(self, request):
        session = original(self, request)
        session._mount_event["plan_id"] = stable_contract_id(
            "benchmark_plan",
            "wrong",
        )
        session._mount_event["event_id"] = benchmark_residency_event_id(
            session._mount_event
        )
        return session

    monkeypatch.setattr(
        FixtureExecutionAdapter,
        "open_session",
        mismatched_mount,
    )
    with pytest.raises(ModelCompileError, match="mismatched mount"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests) == 1


def test_mismatched_unmount_evidence_fails_after_releasing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    original = FixtureExecutionSession.close

    def mismatched_unmount(self):
        document = original(self)
        document["device_state_after_digest"] = device_state_digest(
            {"fixture_state": "capacity_unavailable"}
        )
        document["event_id"] = benchmark_residency_event_id(document)
        return document

    monkeypatch.setattr(
        FixtureExecutionSession,
        "close",
        mismatched_unmount,
    )
    with pytest.raises(ModelCompileError, match="did not release"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests) == 1


def test_observation_cannot_change_matched_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    original = FixtureExecutionSession.execute

    def mismatched_execute(self, request):
        document = original(self, request)
        document["input_digest"] = staged_artifact_digest(b"different")
        document["observation_id"] = benchmark_observation_id(document)
        return document

    monkeypatch.setattr(FixtureExecutionSession, "execute", mismatched_execute)
    with pytest.raises(ModelCompileError, match="matched trial conditions"):
        execute_benchmark_plan(plan, adapter)
    assert adapter.closed_sessions == len(adapter.mount_requests)


def test_corrupt_raw_evidence_is_rejected(tmp_path: Path) -> None:
    _, construction, session, plan, adapter = _fixture(tmp_path)
    workspace = tmp_path / "benchmark-workspace"
    outcome = benchmark_candidate(
        plan=plan,
        construction_record=construction.record,
        session=session,
        adapter=adapter,
        workspace_root=workspace,
    )
    raw = outcome.evidence_path / "raw_run.json"
    raw.write_bytes(b"corrupt")

    with pytest.raises(ModelCompileError, match="failed integrity"):
        load_benchmark_evidence(
            workspace,
            outcome.record.to_json()["benchmark_id"],
        )


def test_corrupt_raw_trace_evidence_is_rejected(tmp_path: Path) -> None:
    _, construction, session, plan, adapter = _fixture(tmp_path)
    workspace = tmp_path / "benchmark-workspace"
    outcome = benchmark_candidate(
        plan=plan,
        construction_record=construction.record,
        session=session,
        adapter=adapter,
        workspace_root=workspace,
    )
    first_trace = next(
        trace["path"]
        for observation in outcome.run.to_json()["observations"]
        for trace in observation["traces"].values()
    )
    (outcome.evidence_path / first_trace).write_bytes(b"corrupt")

    with pytest.raises(ModelCompileError, match="failed integrity"):
        load_benchmark_evidence(
            workspace,
            outcome.record.to_json()["benchmark_id"],
        )


def test_plan_identity_is_deterministic_and_supports_multiple_devices(
    tmp_path: Path,
) -> None:
    candidate_plan, construction, _, plan, _ = _fixture(tmp_path)
    profile = hardware_profile_contract()
    plan_document = plan.to_json()
    rebuilt = build_benchmark_plan(
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        hardware_profiles=(profile,),
        reference_implementation_id="exact-reference",
        reference_contract_digest=candidate_plan.candidate.to_json()[
            "source_contract_digests"
        ][0],
        reference_artifact_refs=plan_document["implementations"]["reference"][
            "artifact_refs"
        ],
        matched_conditions=plan_document["matched_conditions"],
    )
    assert rebuilt == plan

    second = deepcopy(profile)
    second["hardware_identity"]["stable_device_id"] = "vulkan:fixture-b"
    second["hardware_identity"]["name"] = "fixture GPU B"
    second["hardware_identity"]["physical_location"] = "fixture_slot_b"
    second["profile_id"] = stable_contract_id(
        "hardware_profile",
        [
            second["hardware_identity"],
            second["capability_class"],
            second["provenance"],
            second["identity_extensions"],
            second["measurements"],
        ],
    )
    devices = sorted(
        (
            {
                "device_id": hardware["hardware_identity"]["stable_device_id"],
                "hardware_profile_digest": contract_digest(hardware),
                "capability_class": hardware["capability_class"],
                "api": hardware["provenance"]["api"],
            }
            for hardware in (profile, second)
        ),
        key=lambda device: device["device_id"],
    )
    conditions = deepcopy(plan_document["matched_conditions"])
    conditions["devices"] = devices
    conditions["placement"] = {
        "fixture_scope": "vulkan:fixture",
        "cross_device_peer": "vulkan:fixture-b",
    }
    multi_device = build_benchmark_plan(
        candidate_plan=candidate_plan,
        construction_record=construction.record,
        hardware_profiles=(profile, second),
        reference_implementation_id="exact-reference",
        reference_contract_digest=candidate_plan.candidate.to_json()[
            "source_contract_digests"
        ][0],
        reference_artifact_refs=plan_document["implementations"]["reference"][
            "artifact_refs"
        ],
        matched_conditions=conditions,
    )
    assert len(multi_device.matched_conditions["devices"]) == 2
    assert multi_device.matched_conditions["placement"] == conditions["placement"]


def test_output_allowance_requires_verifiable_non_arbitrary_evidence(
    tmp_path: Path,
) -> None:
    _, _, _, plan, _ = _fixture(tmp_path)
    document = plan.to_json()
    workload = next(
        item
        for item in document["workloads"]
        if item["useful_work"]["output_allowance"] is not None
    )
    workload["useful_work"]["output_allowance_basis"]["declared_limit"] -= 1
    workload["workload_id"] = benchmark_workload_id(workload)
    document["plan_id"] = stable_contract_id("benchmark_plan", "wrong")

    with pytest.raises(
        ModelCompileError,
        match="declared output limit.*allowance",
    ):
        BenchmarkPlan.from_json(document)


def test_output_allowance_value_is_checked_against_immutable_evidence(
    tmp_path: Path,
) -> None:
    _, _, _, plan, adapter = _fixture(tmp_path)
    document = plan.to_json()
    workload = next(
        item
        for item in document["workloads"]
        if item["useful_work"]["output_allowance"] is not None
    )
    workload["useful_work"]["output_allowance"] = 65_000
    workload["useful_work"]["output_allowance_basis"]["declared_limit"] = 65_000
    workload["workload_id"] = benchmark_workload_id(workload)
    document["workloads"].sort(key=lambda item: item["workload_id"])
    document["plan_id"] = benchmark_plan_id(document)
    mismatched = BenchmarkPlan.from_json(document)

    with pytest.raises(ModelCompileError, match="does not match.*evidence"):
        execute_benchmark_plan(mismatched, adapter)
    assert adapter.mount_requests == []


def test_benchmark_requires_complete_prebenchmark_lifecycle(
    tmp_path: Path,
) -> None:
    _, construction, session, plan, adapter = _fixture(tmp_path)
    staged_session = replace(
        session,
        candidates=tuple(
            replace(candidate, state=CandidateState.STAGED, history=())
            for candidate in session.candidates
        ),
    )

    with pytest.raises(ModelCompileError, match="proof and prebenchmark"):
        benchmark_candidate(
            plan=plan,
            construction_record=construction.record,
            session=staged_session,
            adapter=adapter,
            workspace_root=tmp_path / "benchmark-workspace",
        )
