from __future__ import annotations

import math
import statistics
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

from nerve.compilation import Json
from nerve.representation_optimizer.contracts import stable_contract_id

from .contracts import (
    CALIBRATION_SUMMARY_SCHEMA,
    CalibrationContractError,
    validate_calibration_plan,
    validate_calibration_run,
    validate_calibration_summary,
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


@dataclass(frozen=True)
class _NormalizedSample:
    duration_ns: int
    throughput_per_second: int
    window_index: int | None


def summarize_calibration_run(profile: Json, plan: Json, run: Json) -> Json:
    validate_calibration_plan(plan)
    validate_calibration_run(run)
    if run["status"] != "completed":
        raise CalibrationContractError(
            "only a completed calibration run can produce a hardware profile"
        )
    if profile["profile_id"] != plan["hardware_profile_id"]:
        raise CalibrationContractError(
            "calibration plan does not target the supplied hardware profile"
        )
    if run["plan_id"] != plan["plan_id"]:
        raise CalibrationContractError("calibration run does not match its plan")
    if run["hardware_profile_id"] != profile["profile_id"]:
        raise CalibrationContractError(
            "calibration run does not match its hardware profile"
        )

    plan_by_id = {
        workload["workload_id"]: workload for workload in plan["workloads"]
    }
    run_by_id = {
        workload["workload_id"]: workload for workload in run["workloads"]
    }
    if set(plan_by_id) != set(run_by_id):
        missing = sorted(set(plan_by_id) - set(run_by_id))
        unexpected = sorted(set(run_by_id) - set(plan_by_id))
        raise CalibrationContractError(
            "calibration result workload set differs from its plan "
            f"(missing={missing}, unexpected={unexpected})"
        )

    workload_summaries: list[Json] = []
    measurements: list[Json] = []
    calibrated_processes: set[str] = set()
    unreliable_processes: set[str] = set()
    for workload_id in sorted(plan_by_id):
        workload = plan_by_id[workload_id]
        result = run_by_id[workload_id]
        if [artifact["name"] for artifact in result["artifacts"]] != [
            artifact["name"] for artifact in workload["artifacts"]
        ]:
            raise CalibrationContractError(
                f"calibration artifacts do not match workload {workload_id!r}"
            )
        summary = _summarize_workload(workload, result, plan["policy"])
        workload_summaries.append(summary)
        if summary["reliable"]:
            calibrated_processes.update(workload["process_names"])
            measurements.extend(
                _hardware_measurements(
                    workload,
                    result,
                    plan,
                    run,
                )
            )
        else:
            unreliable_processes.update(workload["process_names"])

    required_processes = sorted(
        {
            process
            for workload in plan["workloads"]
            for process in workload["process_names"]
        }
    )
    excluded_processes = sorted(
        exclusion["process_name"] for exclusion in plan["excluded_processes"]
    )
    missing_processes = sorted(
        set(required_processes) - calibrated_processes | unreliable_processes
    )
    calibrated_profile = _attach_measurements(profile, measurements)
    coverage = {
        "required_processes": required_processes,
        "calibrated_processes": sorted(calibrated_processes),
        "excluded_processes": excluded_processes,
        "missing_processes": missing_processes,
    }
    summary_id = stable_contract_id(
        "calibration_summary",
        plan["plan_id"],
        run["run_id"],
        calibrated_profile["profile_id"],
        workload_summaries,
        coverage,
    )
    document = {
        "schema": CALIBRATION_SUMMARY_SCHEMA,
        "summary_id": summary_id,
        "plan_id": plan["plan_id"],
        "run_id": run["run_id"],
        "hardware_profile": calibrated_profile,
        "workloads": workload_summaries,
        "coverage": coverage,
    }
    validate_calibration_summary(document)
    return document


def _summarize_workload(workload: Json, result: Json, policy: Json) -> Json:
    diagnostics: list[str] = []
    steady = _normalized_samples(workload, result, phase="steady")
    sustained = _normalized_samples(workload, result, phase="sustained")
    if len(steady) < policy["steady_iterations"]:
        diagnostics.append(
            f"steady sample count {len(steady)} is below "
            f"{policy['steady_iterations']}"
        )
    if len(sustained) < policy["sustained_window_count"]:
        diagnostics.append(
            f"sustained sample count {len(sustained)} is below "
            f"{policy['sustained_window_count']}"
        )
    steady_distribution = _distribution(
        steady,
        confidence_level_ppm=policy["confidence_level_ppm"],
    )
    sustained_distribution = (
        _distribution(
            sustained,
            confidence_level_ppm=policy["confidence_level_ppm"],
        )
        if sustained
        else None
    )
    if (
        steady_distribution["relative_ci_width_ppm"]
        > policy["maximum_relative_ci_width_ppm"]
    ):
        diagnostics.append(
            "steady confidence interval is too wide: "
            f"{steady_distribution['relative_ci_width_ppm']} ppm"
        )
    if result["validation"]["status"] != "passed":
        diagnostics.append("workload output validation did not pass")
    return {
        "workload_id": workload["workload_id"],
        "steady": steady_distribution,
        "sustained": sustained_distribution,
        "construction_duration_ns": result["construction_duration_ns"],
        "reliable": not diagnostics,
        "diagnostics": diagnostics,
    }


def _normalized_samples(
    workload: Json,
    result: Json,
    *,
    phase: str,
) -> list[_NormalizedSample]:
    useful_operations = max(
        workload["work"]["operations_per_iteration"],
        workload["work"]["items_per_iteration"],
        workload["work"]["bytes_read_per_iteration"]
        + workload["work"]["bytes_written_per_iteration"],
    )
    samples: list[_NormalizedSample] = []
    for sample in result["samples"]:
        if sample["phase"] != phase or not sample["valid"]:
            continue
        iterations = sample["iterations"]
        duration = sample["device_duration_ns"] or sample["duration_ns"]
        duration_per_iteration = max(1, _round_div(duration, iterations))
        throughput = _round_div(
            useful_operations * iterations * 1_000_000_000,
            duration,
        )
        samples.append(
            _NormalizedSample(
                duration_ns=duration_per_iteration,
                throughput_per_second=throughput,
                window_index=sample["window_index"],
            )
        )
    return samples


def _distribution(
    samples: list[_NormalizedSample],
    *,
    confidence_level_ppm: int,
) -> Json:
    if not samples:
        raise CalibrationContractError("calibration distribution has no valid samples")
    durations = [sample.duration_ns for sample in samples]
    throughputs = [sample.throughput_per_second for sample in samples]
    sample_count = len(durations)
    mean = statistics.fmean(durations)
    standard_deviation = statistics.stdev(durations) if sample_count > 1 else 0.0
    critical = _t_critical(
        sample_count - 1,
        confidence_level_ppm / 1_000_000,
    )
    margin = critical * standard_deviation / math.sqrt(sample_count)
    ci_low = max(0, round(mean - margin))
    ci_high = round(mean + margin)
    relative_ci_width = (
        round((ci_high - ci_low) * 1_000_000 / mean) if mean > 0 else 0
    )
    return {
        "sample_count": sample_count,
        "minimum_ns": min(durations),
        "maximum_ns": max(durations),
        "median_ns": round(statistics.median(durations)),
        "mean_ns": round(mean),
        "standard_deviation_ns": round(standard_deviation),
        "confidence_interval_low_ns": ci_low,
        "confidence_interval_high_ns": ci_high,
        "relative_ci_width_ppm": relative_ci_width,
        "throughput_per_second": round(statistics.median(throughputs)),
        "throughput_slope_ppm_per_window": _throughput_slope_ppm(samples),
    }


def _throughput_slope_ppm(samples: list[_NormalizedSample]) -> int:
    indexed = [
        (sample.window_index, sample.throughput_per_second)
        for sample in samples
        if sample.window_index is not None
    ]
    if len(indexed) < 2:
        return 0
    x_values = [float(index) for index, _ in indexed]
    y_values = [float(throughput) for _, throughput in indexed]
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0 or y_mean == 0:
        return 0
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator
    return round(slope * 1_000_000 / y_mean)


def _hardware_measurements(
    workload: Json,
    result: Json,
    plan: Json,
    run: Json,
) -> Iterable[Json]:
    base_regime = {
        **workload["regime"],
        "workload_id": workload["workload_id"],
        "executor": workload["executor"],
        "operation": workload["operation"],
        "calibration_plan_id": plan["plan_id"],
        "calibration_run_id": run["run_id"],
        "calibrator_fingerprint": plan["implementation"]["fingerprint"],
        **{
            f"work_{key}": str(value)
            for key, value in workload["work"].items()
        },
    }
    for phase in ("steady", "sustained"):
        samples = [
            sample
            for sample in result["samples"]
            if sample["phase"] == phase and sample["valid"]
        ]
        if not samples:
            continue
        yield {
            "name": f"{workload['workload_id']}.{phase}_duration_ns",
            "unit": "nanoseconds_per_iteration",
            "regime": {**base_regime, "phase": phase},
            "samples": [
                max(
                    1,
                    _round_div(
                        sample["device_duration_ns"] or sample["duration_ns"],
                        sample["iterations"],
                    ),
                )
                for sample in samples
            ],
        }
    yield {
        "name": f"{workload['workload_id']}.construction_duration_ns",
        "unit": "nanoseconds",
        "regime": {**base_regime, "phase": "cold"},
        "samples": [result["construction_duration_ns"]],
    }


def _attach_measurements(profile: Json, measurements: list[Json]) -> Json:
    calibrated = deepcopy(profile)
    calibrated["measurements"] = sorted(
        measurements,
        key=lambda measurement: measurement["name"],
    )
    calibrated["profile_id"] = stable_contract_id(
        "hardware_profile",
        [
            calibrated["hardware_identity"],
            calibrated["capability_class"],
            calibrated["provenance"],
            calibrated["identity_extensions"],
            calibrated["measurements"],
        ],
    )
    return calibrated


def _t_critical(
    degrees_of_freedom: int,
    confidence_level: float,
) -> float:
    if degrees_of_freedom <= 0:
        return 0.0
    if confidence_level == 0.95 and degrees_of_freedom <= 30:
        return _T_CRITICAL_95[degrees_of_freedom]
    probability = 0.5 + confidence_level / 2
    normal = statistics.NormalDist().inv_cdf(probability)
    # Cornish-Fisher expansion of Student's t quantiles. The retained terms
    # are accurate enough for calibration CIs while avoiding a scientific
    # runtime dependency in the compiler.
    degrees = float(degrees_of_freedom)
    z2 = normal * normal
    z3 = z2 * normal
    z5 = z3 * z2
    z7 = z5 * z2
    return (
        normal
        + (z3 + normal) / (4 * degrees)
        + (5 * z5 + 16 * z3 + 3 * normal) / (96 * degrees**2)
        + (3 * z7 + 19 * z5 + 17 * z3 - 15 * normal)
        / (384 * degrees**3)
    )


def _round_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise CalibrationContractError("calibration sample denominator must be positive")
    return (numerator + denominator // 2) // denominator
