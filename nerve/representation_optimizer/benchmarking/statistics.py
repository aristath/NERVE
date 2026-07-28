from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_RECORD_SCHEMA,
    BenchmarkPlan,
    BenchmarkRun,
    benchmark_record_id,
    validate_benchmark_record,
)
from nerve.representation_optimizer.contracts import (
    CANDIDATE_CONSTRUCTION_SCHEMA,
    ContractDocument,
    contract_digest,
)


_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def summarize_benchmark(
    *,
    plan: BenchmarkPlan,
    run: BenchmarkRun,
    construction_record: ContractDocument,
) -> ContractDocument:
    plan_document = plan.to_json()
    run_document = run.to_json()
    construction = construction_record.to_json()
    if run_document["status"] != "completed":
        raise ModelCompileError(
            "only a completed matched benchmark run can be summarized"
        )
    if (
        run_document["plan_id"] != plan_document["plan_id"]
        or construction["schema"] != CANDIDATE_CONSTRUCTION_SCHEMA
        or construction["status"] != "completed"
        or construction["candidate_id"] != plan_document["candidate_id"]
        or construction_record.digest != plan_document["construction_record_digest"]
    ):
        raise ModelCompileError(
            "benchmark plan, run, and candidate construction do not match"
        )
    observations = run_document["observations"]
    sampling_by_workload = {
        outcome["workload_id"]: outcome for outcome in run_document["sampling_outcomes"]
    }
    summaries = []
    reproducibility = []
    decisions = []
    decision_reasons: list[str] = []
    for workload in plan_document["workloads"]:
        sampling = sampling_by_workload.get(workload["workload_id"])
        if sampling is None:
            raise ModelCompileError(
                "benchmark run has no sampling outcome for workload"
            )
        summary, groups = summarize_workload_samples(
            workload,
            observations,
            plan_document["policy"],
            sampling["warmup_groups"],
        )
        if (
            summary["decision"] != sampling["decision"]
            or summary["reasons"] != sampling["reasons"]
            or summary["sample_count_per_role"]
            != sampling["measured_calls_per_role"]
            * len(workload["randomness"]["seeds"])
        ):
            raise ModelCompileError(
                "sampling outcome disagrees with benchmark evidence"
            )
        summaries.append(summary)
        reproducibility.extend(groups)
        decisions.append(summary["decision"])
        decision_reasons.extend(
            f"{workload['workload_id']}: {reason}" for reason in summary["reasons"]
        )
    decision = _overall_decision(decisions)
    if decision == "not_materially_faster" and not decision_reasons:
        decision_reasons.append(
            "no matched regime established a material speedup"
        )
    record = {
        "schema": BENCHMARK_RECORD_SCHEMA,
        "benchmark_id": "",
        "candidate_id": plan_document["candidate_id"],
        "plan_digest": contract_digest(plan_document),
        "run_digest": contract_digest(run_document),
        "construction_record_digest": construction_record.digest,
        "reference_implementation_id": plan_document["implementations"]["reference"][
            "implementation_id"
        ],
        "matched_conditions_digest": plan_document["matched_conditions_digest"],
        "workloads": summaries,
        "reproducibility": sorted(
            reproducibility,
            key=lambda record: (
                record["workload_id"],
                record["role"],
                record["seed"],
                record["order_index"],
            ),
        ),
        "resource_measurements": _resource_measurements(
            construction,
            run_document,
        ),
        "raw_evidence": {
            "run_id": run_document["run_id"],
            "observation_count": len(observations),
            "residency_event_count": len(run_document["residency_events"]),
            "host_elapsed_sample_count": len(run_document["host_elapsed_ns"]),
            "trace_artifact_count": len(
                {
                    trace["path"]
                    for observation in observations
                    for trace in observation["traces"].values()
                }
            ),
        },
        "decision": decision,
        "decision_reasons": decision_reasons,
    }
    record["benchmark_id"] = benchmark_record_id(record)
    validate_benchmark_record(record)
    return ContractDocument.from_json(
        record,
        expected_schema=BENCHMARK_RECORD_SCHEMA,
    )


def summarize_workload_samples(
    workload: Json,
    observations: list[Json],
    policy: Json,
    warmup_groups: list[Json],
) -> tuple[Json, list[Json]]:
    selected = [
        observation
        for observation in observations
        if observation["workload_id"] == workload["workload_id"]
    ]
    warmup = {
        role: _warmup_summary(
            role,
            warmup_groups,
        )
        for role in ("reference", "candidate")
    }
    measured = {
        role: [
            observation
            for observation in selected
            if observation["phase"] == "measured" and observation["role"] == role
        ]
        for role in ("reference", "candidate")
    }
    paired = _paired_summary(measured)
    sustained = _sustained_summary(measured, policy)
    reproducibility = _reproducibility(workload, measured)
    reasons: list[str] = []
    decision = "materially_faster"
    if any(
        group["classification"] == "correctness_defect" for group in reproducibility
    ):
        decision = "invalid"
        reasons.append("fixed-seed repetitions exposed a correctness defect")
    if any(observation["transport"]["timeout_count"] > 0 for observation in selected):
        decision = "invalid"
        reasons.append("runtime reported transport or bounded-wait timeouts")
    if decision != "invalid" and not sustained["passed"]:
        decision = "inconclusive"
        reasons.append("candidate throughput degrades across sustained windows")
    if decision == "materially_faster":
        improvement_floor = policy["minimum_material_improvement_ppm"]
        regression_floor = -policy["maximum_material_regression_ppm"]
        speedup = paired["speedup_ppm"]
        if speedup < regression_floor:
            decision = "materially_slower"
            reasons.append(
                "paired execution is slower than the permitted regression floor"
            )
        elif speedup <= improvement_floor:
            decision = "performance_equivalent"
            reasons.append(
                "paired execution is not faster but remains within the "
                "permitted regression floor"
            )
        else:
            reasons.append(
                "paired execution is faster than the exact implementation"
            )
    role_summaries = {
        role: _role_summary(measured[role]) for role in ("reference", "candidate")
    }
    return (
        {
            "workload_id": workload["workload_id"],
            "decision": decision,
            "reasons": reasons,
            "sample_count_per_role": len(measured["reference"]),
            "warmup": warmup,
            "reference": role_summaries["reference"],
            "candidate": role_summaries["candidate"],
            "paired": paired,
            "sustained": sustained,
        },
        reproducibility,
    )


def warmup_group_summary(
    observations: list[Json],
    policy: Json,
) -> Json:
    window = policy["warmup_stability_window_samples"]
    samples = [_throughput(observation) for observation in observations]
    if len(samples) < 2 * window:
        maximum_shift = 2**63 - 1
    else:
        previous = statistics.median(samples[-2 * window : -window])
        current = statistics.median(samples[-window:])
        maximum_shift = (
            round(abs(current - previous) * 1_000_000 / previous)
            if previous
            else (0 if current == 0 else 2**63 - 1)
        )
    return {
        "sample_count": len(samples),
        "maximum_shift_ppm": maximum_shift,
        # Warmup is deliberately a fixed discarded execution, not an adaptive
        # statistical experiment. The shift remains diagnostic evidence only.
        "converged": len(samples) >= policy["minimum_warmup_samples"],
    }


def _warmup_summary(
    role: str,
    groups: list[Json],
) -> Json:
    selected = [group for group in groups if group["role"] == role]
    by_block: dict[tuple[int, int | None, int | None], list[Json]] = defaultdict(list)
    for group in selected:
        by_block[
            (
                group["seed"],
                group["cycle_index"],
                group["order_block_index"],
            )
        ].append(group)
    effective = [
        next(
            (group for group in attempts if group["converged"]),
            attempts[-1],
        )
        for attempts in by_block.values()
    ]
    return {
        "sample_count": sum(group["sample_count"] for group in selected),
        "maximum_shift_ppm": max(
            (group["maximum_shift_ppm"] for group in effective),
            default=2**63 - 1,
        ),
        "converged": bool(effective) and all(group["converged"] for group in effective),
    }


def _paired_summary(measured: dict[str, list[Json]]) -> Json:
    by_role = {
        role: {
            (observation["seed"], observation["pair_index"]): observation
            for observation in samples
        }
        for role, samples in measured.items()
    }
    if set(by_role["reference"]) != set(by_role["candidate"]):
        raise ModelCompileError(
            "matched benchmark lost a reference/candidate sample pair"
        )
    ratios = []
    for key in sorted(by_role["reference"]):
        reference = by_role["reference"][key]
        candidate = by_role["candidate"][key]
        if reference["work"]["useful_units"] != candidate["work"]["useful_units"]:
            raise ModelCompileError(
                "matched benchmark pair performed different useful work"
            )
        reference_rate = _throughput(reference)
        candidate_rate = _throughput(candidate)
        if reference_rate <= 0 or candidate_rate <= 0:
            raise ModelCompileError("matched benchmark throughput must be positive")
        ratios.append(candidate_rate / reference_rate)
    if not ratios:
        raise ModelCompileError("matched benchmark has no measured pair")
    ratio = math.exp(statistics.fmean(math.log(value) for value in ratios))
    speedup_ppm = round((ratio - 1.0) * 1_000_000)
    return {
        "speedup_ppm": speedup_ppm,
        "candidate_is_faster": speedup_ppm > 0,
    }


def _sustained_summary(
    measured: dict[str, list[Json]],
    policy: Json,
) -> Json:
    slopes = {
        role: [
            _throughput_slope(observation["throughput_windows"])
            for observation in observations
            if len(observation["throughput_windows"]) >= 2
        ]
        for role, observations in measured.items()
    }
    median = {
        role: (round(statistics.median(values)) if values else 0)
        for role, values in slopes.items()
    }
    reference_regression = max(0, -median["reference"])
    candidate_regression = max(0, -median["candidate"])
    limit = policy["maximum_sustained_regression_ppm"]
    passed = (
        candidate_regression <= limit
        and candidate_regression <= reference_regression + limit
    )
    return {
        "reference_slope_ppm_per_window": median["reference"],
        "candidate_slope_ppm_per_window": median["candidate"],
        "candidate_regression_ppm": candidate_regression,
        "passed": passed,
    }


def _reproducibility(
    workload: Json,
    measured: dict[str, list[Json]],
) -> list[Json]:
    records = []
    randomness = workload["randomness"]
    for role in ("reference", "candidate"):
        grouped: dict[tuple[int, int], list[Json]] = defaultdict(list)
        for observation in measured[role]:
            grouped[(observation["seed"], observation["order_index"])].append(
                observation
            )
        for (seed, order_index), observations in sorted(grouped.items()):
            if len(observations) < 2:
                continue
            traces = [observation["traces"] for observation in observations]
            if all(
                all(
                    trace[field]["digest"] == traces[0][field]["digest"]
                    for field in traces[0]
                )
                for trace in traces[1:]
            ):
                classification = "identical"
            else:
                semantic_fields = ("distribution", "tokens", "state")
                semantic_equal = all(
                    all(
                        trace[field]["digest"] == traces[0][field]["digest"]
                        for field in semantic_fields
                    )
                    for trace in traces[1:]
                )
                random_equal = all(
                    trace["random_draws"]["digest"]
                    == traces[0]["random_draws"]["digest"]
                    for trace in traces[1:]
                )
                schedule_equal = all(
                    trace["schedule"]["digest"] == traces[0]["schedule"]["digest"]
                    for trace in traces[1:]
                )
                if (
                    semantic_equal
                    and random_equal
                    and not schedule_equal
                    and randomness["permit_speculative_schedule_variance"]
                    and any(
                        observation["work"]["speculative_units"] > 0
                        for observation in observations
                    )
                ):
                    classification = "speculative_scheduling"
                elif not random_equal and randomness["permit_sampling_variance"]:
                    classification = "permitted_sampling_variance"
                elif (
                    not semantic_equal
                    and random_equal
                    and randomness["permit_numerical_nondeterminism"]
                ):
                    classification = "numerical_nondeterminism"
                else:
                    classification = "correctness_defect"
            records.append(
                {
                    "workload_id": workload["workload_id"],
                    "role": role,
                    "seed": seed,
                    "order_index": order_index,
                    "classification": classification,
                    "observation_ids": [
                        observation["observation_id"] for observation in observations
                    ],
                }
            )
    return records


def _role_summary(observations: list[Json]) -> Json:
    if not observations:
        raise ModelCompileError("benchmark role has no measured observations")
    measurement_ns = sum(
        observation["device"]["measurement_ns"] for observation in observations
    )
    busy_ns = sum(observation["device"]["busy_ns"] for observation in observations)
    return {
        "latency_ns": _distribution(
            observation["timing"]["execution_ns"] for observation in observations
        ),
        "throughput_per_second": _distribution(
            round(_throughput(observation)) for observation in observations
        ),
        "permanent_bytes": max(
            observation["memory"]["permanent_bytes"] for observation in observations
        ),
        "peak_transient_bytes": max(
            observation["memory"]["peak_transient_bytes"]
            for observation in observations
        ),
        "resident_before_bytes": max(
            observation["memory"]["resident_before_bytes"]
            for observation in observations
        ),
        "resident_peak_bytes": max(
            observation["memory"]["resident_peak_bytes"] for observation in observations
        ),
        "resident_after_bytes": max(
            observation["memory"]["resident_after_bytes"]
            for observation in observations
        ),
        "conversion_bytes": sum(
            observation["representation"]["conversion_bytes"]
            for observation in observations
        ),
        "conversion_ns": sum(
            observation["representation"]["conversion_ns"]
            for observation in observations
        ),
        "boundary_count": sum(
            observation["representation"]["boundary_count"]
            for observation in observations
        ),
        "utilization_ppm": (
            round(busy_ns * 1_000_000 / measurement_ns) if measurement_ns else 0
        ),
        "synchronization_wait_ns": sum(
            observation["synchronization"]["wait_ns"] for observation in observations
        ),
        "transport_bytes": sum(
            observation["transport"]["bytes"] for observation in observations
        ),
        "transport_ns": sum(
            observation["transport"]["duration_ns"] for observation in observations
        ),
        "queue_wait_ns": sum(
            observation["timing"]["queue_wait_ns"]
            + observation["transport"]["queue_wait_ns"]
            for observation in observations
        ),
        "timeout_count": sum(
            observation["transport"]["timeout_count"] for observation in observations
        ),
        "useful_units": sum(
            observation["work"]["useful_units"] for observation in observations
        ),
        "wasted_units": sum(
            observation["work"]["speculative_units"]
            + observation["work"]["cancelled_units"]
            + observation["work"]["discarded_units"]
            + observation["work"]["corrective_units"]
            for observation in observations
        ),
    }


def _resource_measurements(construction: Json, run: Json) -> Json:
    elapsed = {
        record["observation_id"]: record["duration_ns"]
        for record in run["host_elapsed_ns"]
    }
    roles = {}
    for role in ("reference", "candidate"):
        observations = [
            observation
            for observation in run["observations"]
            if observation["role"] == role
        ]
        mounts = [
            event
            for event in run["residency_events"]
            if event["role"] == role and event["action"] == "mount"
        ]
        unmounts = [
            event
            for event in run["residency_events"]
            if event["role"] == role and event["action"] == "unmount"
        ]
        measured_ns = sum(
            observation["device"]["measurement_ns"] for observation in observations
        )
        busy_ns = sum(observation["device"]["busy_ns"] for observation in observations)
        roles[role] = {
            "setup_ns": (
                sum(event["duration_ns"] for event in mounts)
                + sum(observation["timing"]["setup_ns"] for observation in observations)
            ),
            "teardown_ns": (
                sum(event["duration_ns"] for event in unmounts)
                + sum(
                    observation["timing"]["teardown_ns"] for observation in observations
                )
            ),
            "host_elapsed_ns": sum(
                elapsed[observation["observation_id"]] for observation in observations
            ),
            "permanent_bytes": max(
                (event["permanent_bytes"] for event in mounts),
                default=0,
            ),
            "peak_transient_bytes": max(
                [event["peak_transient_bytes"] for event in mounts]
                + [
                    observation["memory"]["peak_transient_bytes"]
                    for observation in observations
                ],
                default=0,
            ),
            "resident_before_bytes": max(
                (
                    observation["memory"]["resident_before_bytes"]
                    for observation in observations
                ),
                default=0,
            ),
            "resident_peak_bytes": max(
                (
                    observation["memory"]["resident_peak_bytes"]
                    for observation in observations
                ),
                default=0,
            ),
            "resident_after_bytes": max(
                (
                    observation["memory"]["resident_after_bytes"]
                    for observation in observations
                ),
                default=0,
            ),
            "conversion_bytes": sum(
                observation["representation"]["conversion_bytes"]
                for observation in observations
            ),
            "conversion_ns": sum(
                observation["representation"]["conversion_ns"]
                for observation in observations
            ),
            "boundary_count": sum(
                observation["representation"]["boundary_count"]
                for observation in observations
            ),
            "device_measurement_ns": measured_ns,
            "device_busy_ns": busy_ns,
            "utilization_ppm": (
                round(busy_ns * 1_000_000 / measured_ns) if measured_ns else 0
            ),
            "synchronization_count": sum(
                observation["synchronization"]["operation_count"]
                for observation in observations
            ),
            "synchronization_wait_ns": sum(
                observation["synchronization"]["wait_ns"]
                for observation in observations
            ),
            "transport_bytes": sum(
                observation["transport"]["bytes"] for observation in observations
            ),
            "transport_ns": sum(
                observation["transport"]["duration_ns"] for observation in observations
            ),
            "queue_wait_count": sum(
                observation["transport"]["queue_wait_count"]
                for observation in observations
            ),
            "queue_wait_ns": sum(
                observation["timing"]["queue_wait_ns"]
                + observation["transport"]["queue_wait_ns"]
                for observation in observations
            ),
            "timeout_count": sum(
                observation["transport"]["timeout_count"]
                for observation in observations
            ),
            "useful_units": sum(
                observation["work"]["useful_units"] for observation in observations
            ),
            "speculative_units": sum(
                observation["work"]["speculative_units"] for observation in observations
            ),
            "cancelled_units": sum(
                observation["work"]["cancelled_units"] for observation in observations
            ),
            "discarded_units": sum(
                observation["work"]["discarded_units"] for observation in observations
            ),
            "corrective_units": sum(
                observation["work"]["corrective_units"] for observation in observations
            ),
        }
    return {
        "construction": construction["resource_measurements"],
        "roles": roles,
    }


def _distribution(values: Iterable[int]) -> Json:
    samples = list(values)
    if not samples:
        raise ModelCompileError("benchmark distribution has no samples")
    count = len(samples)
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if count > 1 else 0.0
    critical = _T_CRITICAL_95.get(count - 1, 1.96)
    margin = critical * deviation / math.sqrt(count)
    low = max(0, round(mean - margin))
    high = max(low, round(mean + margin))
    return {
        "sample_count": count,
        "minimum": min(samples),
        "maximum": max(samples),
        "median": round(statistics.median(samples)),
        "mean": round(mean),
        "standard_deviation": round(deviation),
        "confidence_interval_low": low,
        "confidence_interval_high": high,
        "relative_ci_width_ppm": (
            round((high - low) * 1_000_000 / mean) if mean else 0
        ),
    }


def _throughput(observation: Json) -> float:
    duration = observation["timing"]["execution_ns"]
    return observation["work"]["useful_units"] * 1_000_000_000 / duration


def _throughput_slope(windows: list[Json]) -> int:
    if len(windows) < 2:
        return 0
    x_values = [float(window["index"]) for window in windows]
    y_values = [
        (
            (window["end_unit"] - window["start_unit"])
            * 1_000_000_000
            / window["duration_ns"]
        )
        for window in windows
    ]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0 or y_mean == 0:
        return 0
    slope = (
        sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
        )
        / denominator
    )
    return round(slope * 1_000_000 / y_mean)


def _overall_decision(decisions: list[str]) -> str:
    if "invalid" in decisions:
        return "invalid"
    if "materially_slower" in decisions:
        return "not_materially_faster"
    if "inconclusive" in decisions:
        return "inconclusive"
    if (
        "materially_faster" in decisions
        and all(
            decision in {"materially_faster", "performance_equivalent"}
            for decision in decisions
        )
    ):
        return "materially_faster"
    return "not_materially_faster"
