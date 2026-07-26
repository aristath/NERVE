from __future__ import annotations

from typing import Any

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    HARDWARE_PROCESS_PROFILE_SCHEMA,
    canonical_json_bytes,
    stable_contract_id,
    validate_contract,
)


CALIBRATION_PLAN_SCHEMA = "nerve.optimizer.hardware_calibration_plan.v1"
CALIBRATION_RUN_SCHEMA = "nerve.optimizer.hardware_calibration_run.v1"
CALIBRATION_SUMMARY_SCHEMA = "nerve.optimizer.hardware_calibration_summary.v1"
CALIBRATION_MANIFEST_SCHEMA = "nerve.optimizer.hardware_calibration_manifest.v1"

EXECUTORS = frozenset(
    {
        "cpu",
        "vulkan_compute",
        "vulkan_graphics",
        "vulkan_ray",
        "vulkan_transfer",
        "vulkan_video",
    }
)
SAMPLE_PHASES = frozenset({"cold", "warmup", "steady", "sustained"})


class CalibrationContractError(ModelCompileError):
    """A calibration plan, raw run, or statistical summary is invalid."""


def validate_calibration_plan(document: Json) -> None:
    _object(document, "$")
    _fields(
        document,
        {
            "schema",
            "plan_id",
            "hardware_profile_id",
            "capability_class",
            "implementation",
            "policy",
            "workloads",
            "excluded_processes",
        },
        "$",
    )
    _schema(document, CALIBRATION_PLAN_SCHEMA)
    _stable_id(document["plan_id"], "calibration_plan", "plan_id")
    _stable_id(
        document["hardware_profile_id"],
        "hardware_profile",
        "hardware_profile_id",
    )
    _stable_id(
        document["capability_class"],
        "hardware_capability",
        "capability_class",
    )
    _implementation(document["implementation"], "implementation")
    policy = _object(document["policy"], "policy")
    _fields(
        policy,
        {
            "warmup_iterations",
            "steady_iterations",
            "minimum_sample_duration_ns",
            "sustained_window_duration_ms",
            "sustained_window_count",
            "confidence_level_ppm",
            "maximum_relative_ci_width_ppm",
        },
        "policy",
    )
    _positive_integer(policy["warmup_iterations"], "policy.warmup_iterations")
    steady_iterations = _positive_integer(
        policy["steady_iterations"],
        "policy.steady_iterations",
    )
    if steady_iterations < 5:
        raise CalibrationContractError(
            "policy.steady_iterations must contain at least five independent samples"
        )
    _positive_integer(
        policy["minimum_sample_duration_ns"],
        "policy.minimum_sample_duration_ns",
    )
    _positive_integer(
        policy["sustained_window_duration_ms"],
        "policy.sustained_window_duration_ms",
    )
    _positive_integer(
        policy["sustained_window_count"],
        "policy.sustained_window_count",
    )
    confidence = _positive_integer(
        policy["confidence_level_ppm"],
        "policy.confidence_level_ppm",
    )
    if confidence >= 1_000_000:
        raise CalibrationContractError(
            "policy.confidence_level_ppm must be below 1000000"
        )
    relative_ci = _positive_integer(
        policy["maximum_relative_ci_width_ppm"],
        "policy.maximum_relative_ci_width_ppm",
    )
    if relative_ci >= 1_000_000:
        raise CalibrationContractError(
            "policy.maximum_relative_ci_width_ppm must be below 1000000"
        )

    workloads = _list(document["workloads"], "workloads")
    if not workloads:
        raise CalibrationContractError("calibration plan has no workloads")
    workload_ids: list[str] = []
    covered_processes: list[str] = []
    for index, raw_workload in enumerate(workloads):
        path = f"workloads[{index}]"
        workload = _object(raw_workload, path)
        _fields(
            workload,
            {
                "workload_id",
                "process_names",
                "executor",
                "operation",
                "regime",
                "work",
                "artifacts",
                "validation",
            },
            path,
        )
        workload_id = _stable_id(
            workload["workload_id"],
            "calibration_workload",
            f"{path}.workload_id",
        )
        workload_ids.append(workload_id)
        process_names = _sorted_unique_strings(
            workload["process_names"],
            f"{path}.process_names",
            nonempty=True,
        )
        covered_processes.extend(process_names)
        if workload["executor"] not in EXECUTORS:
            raise CalibrationContractError(
                f"{path}.executor is unsupported: {workload['executor']!r}"
            )
        _nonempty_string(workload["operation"], f"{path}.operation")
        _string_map(workload["regime"], f"{path}.regime")
        work = _object(workload["work"], f"{path}.work")
        _fields(
            work,
            {
                "items_per_iteration",
                "operations_per_iteration",
                "bytes_read_per_iteration",
                "bytes_written_per_iteration",
            },
            f"{path}.work",
        )
        if not any(
            _nonnegative_integer(work[field], f"{path}.work.{field}") > 0
            for field in work
        ):
            raise CalibrationContractError(f"{path}.work declares no useful work")
        _artifact_declarations(workload["artifacts"], f"{path}.artifacts")
        validation = _object(workload["validation"], f"{path}.validation")
        _fields(
            validation,
            {"mode", "expected_digest", "maximum_error_ppm"},
            f"{path}.validation",
        )
        if validation["mode"] not in {"digest", "exact", "tolerance"}:
            raise CalibrationContractError(f"{path}.validation.mode is unsupported")
        expected_digest = validation["expected_digest"]
        if expected_digest is not None:
            _digest(expected_digest, f"{path}.validation.expected_digest")
        maximum_error = _nonnegative_integer(
            validation["maximum_error_ppm"],
            f"{path}.validation.maximum_error_ppm",
        )
        if validation["mode"] in {"digest", "exact"} and maximum_error != 0:
            raise CalibrationContractError(
                f"{path}.validation exact modes require zero error"
            )
        expected_id = stable_contract_id(
            "calibration_workload",
            process_names,
            workload["executor"],
            workload["operation"],
            workload["regime"],
            workload["work"],
            workload["artifacts"],
            workload["validation"],
        )
        if workload_id != expected_id:
            raise CalibrationContractError(
                f"{path}.workload_id does not match canonical workload identity"
            )
    _strictly_sorted_unique(workload_ids, "workloads")

    excluded = _list(document["excluded_processes"], "excluded_processes")
    excluded_names: list[str] = []
    for index, raw_exclusion in enumerate(excluded):
        path = f"excluded_processes[{index}]"
        exclusion = _object(raw_exclusion, path)
        _fields(exclusion, {"process_name", "reason"}, path)
        excluded_names.append(
            _nonempty_string(exclusion["process_name"], f"{path}.process_name")
        )
        if exclusion["reason"] not in {
            "unavailable",
            "not_programmable",
            "not_exposed_by_selected_api",
        }:
            raise CalibrationContractError(f"{path}.reason is unsupported")
    _strictly_sorted_unique(excluded_names, "excluded_processes")
    overlap = sorted(set(covered_processes) & set(excluded_names))
    if overlap:
        raise CalibrationContractError(
            f"processes cannot be both calibrated and excluded: {overlap}"
        )

    expected_plan_id = stable_contract_id(
        "calibration_plan",
        document["hardware_profile_id"],
        document["capability_class"],
        document["implementation"],
        policy,
        workloads,
        excluded,
    )
    if document["plan_id"] != expected_plan_id:
        raise CalibrationContractError(
            "plan_id does not match canonical calibration plan content"
        )


def validate_calibration_run(document: Json) -> None:
    _object(document, "$")
    _fields(
        document,
        {
            "schema",
            "run_id",
            "plan_id",
            "hardware_profile_id",
            "status",
            "started_at",
            "finished_at",
            "workloads",
            "diagnostics",
        },
        "$",
    )
    _schema(document, CALIBRATION_RUN_SCHEMA)
    _stable_id(document["run_id"], "calibration_run", "run_id")
    _stable_id(document["plan_id"], "calibration_plan", "plan_id")
    _stable_id(
        document["hardware_profile_id"],
        "hardware_profile",
        "hardware_profile_id",
    )
    if document["status"] not in {"completed", "failed", "cancelled"}:
        raise CalibrationContractError("calibration run status is unsupported")
    _timestamp(document["started_at"], "started_at")
    _timestamp(document["finished_at"], "finished_at")
    workloads = _list(document["workloads"], "workloads")
    workload_ids: list[str] = []
    for index, raw_result in enumerate(workloads):
        path = f"workloads[{index}]"
        result = _object(raw_result, path)
        _fields(
            result,
            {
                "workload_id",
                "status",
                "construction_duration_ns",
                "artifacts",
                "samples",
                "validation",
                "counters",
                "diagnostics",
            },
            path,
        )
        workload_ids.append(
            _stable_id(
                result["workload_id"],
                "calibration_workload",
                f"{path}.workload_id",
            )
        )
        if result["status"] not in {"completed", "failed", "cancelled"}:
            raise CalibrationContractError(f"{path}.status is unsupported")
        _nonnegative_integer(
            result["construction_duration_ns"],
            f"{path}.construction_duration_ns",
        )
        artifacts = _list(result["artifacts"], f"{path}.artifacts")
        artifact_names: list[str] = []
        for artifact_index, raw_artifact in enumerate(artifacts):
            artifact_path = f"{path}.artifacts[{artifact_index}]"
            artifact = _object(raw_artifact, artifact_path)
            _fields(
                artifact,
                {"name", "kind", "digest", "byte_length", "relative_path"},
                artifact_path,
            )
            artifact_names.append(
                _nonempty_string(artifact["name"], f"{artifact_path}.name")
            )
            _nonempty_string(artifact["kind"], f"{artifact_path}.kind")
            _digest(artifact["digest"], f"{artifact_path}.digest")
            _positive_integer(
                artifact["byte_length"],
                f"{artifact_path}.byte_length",
            )
            relative_path = _nonempty_string(
                artifact["relative_path"],
                f"{artifact_path}.relative_path",
            )
            parts = relative_path.replace("\\", "/").split("/")
            if relative_path.startswith(("/", "\\")) or ".." in parts:
                raise CalibrationContractError(
                    f"{artifact_path}.relative_path is unsafe"
                )
        _strictly_sorted_unique(artifact_names, f"{path}.artifacts")
        samples = _list(result["samples"], f"{path}.samples")
        sample_indices: list[int] = []
        for sample_index, raw_sample in enumerate(samples):
            sample_path = f"{path}.samples[{sample_index}]"
            sample = _object(raw_sample, sample_path)
            _fields(
                sample,
                {
                    "sample_index",
                    "phase",
                    "duration_ns",
                    "device_duration_ns",
                    "iterations",
                    "window_index",
                    "thermal_millidegrees_celsius",
                    "valid",
                },
                sample_path,
            )
            sample_indices.append(
                _nonnegative_integer(
                    sample["sample_index"],
                    f"{sample_path}.sample_index",
                )
            )
            if sample["phase"] not in SAMPLE_PHASES:
                raise CalibrationContractError(f"{sample_path}.phase is unsupported")
            _positive_integer(sample["duration_ns"], f"{sample_path}.duration_ns")
            if sample["device_duration_ns"] is not None:
                _positive_integer(
                    sample["device_duration_ns"],
                    f"{sample_path}.device_duration_ns",
                )
            _positive_integer(sample["iterations"], f"{sample_path}.iterations")
            if sample["window_index"] is not None:
                _nonnegative_integer(
                    sample["window_index"],
                    f"{sample_path}.window_index",
                )
            if sample["thermal_millidegrees_celsius"] is not None:
                _nonnegative_integer(
                    sample["thermal_millidegrees_celsius"],
                    f"{sample_path}.thermal_millidegrees_celsius",
                )
            if not isinstance(sample["valid"], bool):
                raise CalibrationContractError(f"{sample_path}.valid must be boolean")
        if sample_indices != list(range(len(sample_indices))):
            raise CalibrationContractError(
                f"{path}.sample_index values must be contiguous from zero"
            )
        validation = _object(result["validation"], f"{path}.validation")
        _fields(
            validation,
            {"status", "observed_digest", "maximum_error_ppm"},
            f"{path}.validation",
        )
        if validation["status"] not in {"passed", "failed", "not_run"}:
            raise CalibrationContractError(f"{path}.validation.status is unsupported")
        if validation["observed_digest"] is not None:
            _digest(
                validation["observed_digest"],
                f"{path}.validation.observed_digest",
            )
        _nonnegative_integer(
            validation["maximum_error_ppm"],
            f"{path}.validation.maximum_error_ppm",
        )
        _integer_map(result["counters"], f"{path}.counters")
        _string_list(result["diagnostics"], f"{path}.diagnostics")
        if result["status"] == "completed":
            if validation["status"] != "passed":
                raise CalibrationContractError(
                    f"{path} completed without passing validation"
                )
            if not samples:
                raise CalibrationContractError(
                    f"{path} completed without raw measurement samples"
                )
    _strictly_sorted_unique(workload_ids, "workloads")
    _string_list(document["diagnostics"], "diagnostics")
    if document["status"] == "completed" and any(
        result["status"] != "completed" for result in workloads
    ):
        raise CalibrationContractError(
            "completed calibration run contains incomplete workloads"
        )
    expected_run_id = stable_contract_id(
        "calibration_run",
        document["plan_id"],
        document["hardware_profile_id"],
        document["started_at"],
    )
    if document["run_id"] != expected_run_id:
        raise CalibrationContractError(
            "run_id does not match canonical calibration run identity"
        )


def validate_calibration_summary(document: Json) -> None:
    _object(document, "$")
    _fields(
        document,
        {
            "schema",
            "summary_id",
            "plan_id",
            "run_id",
            "hardware_profile",
            "workloads",
            "coverage",
        },
        "$",
    )
    _schema(document, CALIBRATION_SUMMARY_SCHEMA)
    _stable_id(document["summary_id"], "calibration_summary", "summary_id")
    _stable_id(document["plan_id"], "calibration_plan", "plan_id")
    _stable_id(document["run_id"], "calibration_run", "run_id")
    profile = _object(document["hardware_profile"], "hardware_profile")
    validate_contract(profile, expected_schema=HARDWARE_PROCESS_PROFILE_SCHEMA)
    workloads = _list(document["workloads"], "workloads")
    workload_ids: list[str] = []
    for index, raw_summary in enumerate(workloads):
        path = f"workloads[{index}]"
        summary = _object(raw_summary, path)
        _fields(
            summary,
            {
                "workload_id",
                "steady",
                "sustained",
                "construction_duration_ns",
                "reliable",
                "diagnostics",
            },
            path,
        )
        workload_ids.append(
            _stable_id(
                summary["workload_id"],
                "calibration_workload",
                f"{path}.workload_id",
            )
        )
        _distribution(summary["steady"], f"{path}.steady")
        if summary["sustained"] is not None:
            _distribution(summary["sustained"], f"{path}.sustained")
        _nonnegative_integer(
            summary["construction_duration_ns"],
            f"{path}.construction_duration_ns",
        )
        if not isinstance(summary["reliable"], bool):
            raise CalibrationContractError(f"{path}.reliable must be boolean")
        _string_list(summary["diagnostics"], f"{path}.diagnostics")
    _strictly_sorted_unique(workload_ids, "workloads")
    coverage = _object(document["coverage"], "coverage")
    _fields(
        coverage,
        {
            "required_processes",
            "calibrated_processes",
            "excluded_processes",
            "missing_processes",
        },
        "coverage",
    )
    for field in coverage:
        _sorted_unique_strings(coverage[field], f"coverage.{field}")
    if document["coverage"]["missing_processes"]:
        raise CalibrationContractError(
            "calibration summary has uncovered hardware processes"
        )
    expected_summary_id = stable_contract_id(
        "calibration_summary",
        document["plan_id"],
        document["run_id"],
        profile["profile_id"],
        workloads,
        coverage,
    )
    if document["summary_id"] != expected_summary_id:
        raise CalibrationContractError(
            "summary_id does not match canonical calibration summary content"
        )


def _distribution(value: Any, path: str) -> None:
    distribution = _object(value, path)
    _fields(
        distribution,
        {
            "sample_count",
            "minimum_ns",
            "maximum_ns",
            "median_ns",
            "mean_ns",
            "standard_deviation_ns",
            "confidence_interval_low_ns",
            "confidence_interval_high_ns",
            "relative_ci_width_ppm",
            "throughput_per_second",
            "throughput_slope_ppm_per_window",
        },
        path,
    )
    _positive_integer(distribution["sample_count"], f"{path}.sample_count")
    for field in (
        "minimum_ns",
        "maximum_ns",
        "median_ns",
        "mean_ns",
        "standard_deviation_ns",
        "confidence_interval_low_ns",
        "confidence_interval_high_ns",
        "relative_ci_width_ppm",
        "throughput_per_second",
    ):
        _nonnegative_integer(distribution[field], f"{path}.{field}")
    slope = distribution["throughput_slope_ppm_per_window"]
    if not isinstance(slope, int) or isinstance(slope, bool):
        raise CalibrationContractError(
            f"{path}.throughput_slope_ppm_per_window must be an integer"
        )
    if distribution["minimum_ns"] > distribution["maximum_ns"]:
        raise CalibrationContractError(f"{path} minimum exceeds maximum")
    if (
        distribution["confidence_interval_low_ns"]
        > distribution["confidence_interval_high_ns"]
    ):
        raise CalibrationContractError(f"{path} confidence interval is inverted")


def _artifact_declarations(value: Any, path: str) -> None:
    artifacts = _list(value, path)
    identities: list[str] = []
    for index, raw_artifact in enumerate(artifacts):
        artifact_path = f"{path}[{index}]"
        artifact = _object(raw_artifact, artifact_path)
        _fields(artifact, {"name", "kind", "digest"}, artifact_path)
        identities.append(_nonempty_string(artifact["name"], f"{artifact_path}.name"))
        _nonempty_string(artifact["kind"], f"{artifact_path}.kind")
        if artifact["digest"] is not None:
            _digest(artifact["digest"], f"{artifact_path}.digest")
    _strictly_sorted_unique(identities, path)


def _implementation(value: Any, path: str) -> None:
    implementation = _object(value, path)
    _fields(implementation, {"name", "version", "fingerprint"}, path)
    _nonempty_string(implementation["name"], f"{path}.name")
    _nonempty_string(implementation["version"], f"{path}.version")
    _digest(implementation["fingerprint"], f"{path}.fingerprint")


def _fields(document: Json, required: set[str], path: str) -> None:
    actual = set(document)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise CalibrationContractError(f"{path} is missing fields {missing}")
    if unknown:
        raise CalibrationContractError(f"{path} has unknown fields {unknown}")


def _schema(document: Json, expected: str) -> None:
    if document.get("schema") != expected:
        raise CalibrationContractError(
            f"expected calibration schema {expected!r}, found {document.get('schema')!r}"
        )


def _object(value: Any, path: str) -> Json:
    if not isinstance(value, dict):
        raise CalibrationContractError(f"{path} must be an object")
    canonical_json_bytes(value)
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise CalibrationContractError(f"{path} must be a list")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationContractError(f"{path} must be a non-empty string")
    return value


def _string_list(value: Any, path: str) -> list[str]:
    values = _list(value, path)
    for index, item in enumerate(values):
        _nonempty_string(item, f"{path}[{index}]")
    return values


def _sorted_unique_strings(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> list[str]:
    values = _string_list(value, path)
    if nonempty and not values:
        raise CalibrationContractError(f"{path} must not be empty")
    _strictly_sorted_unique(values, path)
    return values


def _strictly_sorted_unique(values: list[Any], path: str) -> None:
    if values != sorted(set(values)):
        raise CalibrationContractError(f"{path} must be unique and sorted")


def _string_map(value: Any, path: str) -> Json:
    mapping = _object(value, path)
    for key, item in mapping.items():
        _nonempty_string(key, f"{path} key")
        _nonempty_string(item, f"{path}.{key}")
    return mapping


def _integer_map(value: Any, path: str) -> Json:
    mapping = _object(value, path)
    for key, item in mapping.items():
        _nonempty_string(key, f"{path} key")
        _nonnegative_integer(item, f"{path}.{key}")
    return mapping


def _nonnegative_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CalibrationContractError(f"{path} must be a non-negative integer")
    return value


def _positive_integer(value: Any, path: str) -> int:
    result = _nonnegative_integer(value, path)
    if result == 0:
        raise CalibrationContractError(f"{path} must be positive")
    return result


def _stable_id(value: Any, prefix: str, path: str) -> str:
    result = _nonempty_string(value, path)
    expected_prefix = f"{prefix}_"
    suffix = result.removeprefix(expected_prefix)
    if (
        not result.startswith(expected_prefix)
        or len(suffix) != 32
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise CalibrationContractError(
            f"{path} must be a stable {prefix!r} identity"
        )
    return result


def _digest(value: Any, path: str) -> str:
    result = _nonempty_string(value, path)
    schema, separator, digest = result.rpartition(":")
    if (
        not separator
        or not schema
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CalibrationContractError(f"{path} must be a versioned SHA-256 digest")
    return result


def _timestamp(value: Any, path: str) -> str:
    result = _nonempty_string(value, path)
    if not result.endswith("Z") or "T" not in result:
        raise CalibrationContractError(f"{path} must be an RFC 3339 UTC timestamp")
    return result
