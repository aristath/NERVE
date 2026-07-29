from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Callable

from nerve.compilation import (
    Json,
    ModelCompileCancelled,
    ModelCompileError,
)
from nerve.representation_optimizer.staging.contracts import (
    STAGED_ARTIFACT_DIGEST_SCHEMA,
)
from nerve.representation_optimizer.validation.contracts import (
    VALIDATION_OBSERVATION_SCHEMA,
    VALIDATION_RUN_SCHEMA,
    BehavioralErrorContract,
    ValidationObservation,
    ValidationPlan,
    ValidationResidencyEvent,
    ValidationRoleResult,
    ValidationRun,
    validation_observation_id,
    validation_run_id,
)
from nerve.representation_optimizer.validation.protocols import (
    BehavioralValidationAdapter,
    ValidationComparisonRequest,
    ValidationRoleExecutionRequest,
    ValidationRoleMountRequest,
)


def execute_validation_stage(
    plan: ValidationPlan,
    *,
    stage: str,
    adapter: BehavioralValidationAdapter,
    cancel_requested: Callable[[], bool] | None = None,
) -> ValidationRun:
    return _execute_validation_stage(
        plan,
        stage=stage,
        adapter=adapter,
        cancel_requested=cancel_requested,
    )


def _execute_validation_stage(
    plan: ValidationPlan,
    *,
    stage: str,
    adapter: BehavioralValidationAdapter,
    cancel_requested: Callable[[], bool] | None = None,
) -> ValidationRun:
    checks = plan.checks_for_stage(stage)
    if not checks:
        raise ModelCompileError(
            f"validation plan has no checks for stage {stage!r}"
        )
    document = plan.to_json()
    _verify_fixture_artifacts(
        checks,
        adapter,
        candidate_id=plan.candidate_id,
    )
    observations: list[Json] = []
    events: list[Json] = []
    elapsed: list[Json] = []
    diagnostics: list[str] = []
    trace_paths: set[str] = set()
    status = "completed"
    block_index = 0
    pairs_by_scope = {
        execution_scope: tuple(
            (check, seed)
            for check in checks
            if check["regime"]["execution_scope"] == execution_scope
            for seed in check["seeds"]
        )
        for execution_scope in ("component", "whole_model")
    }
    pairs = tuple(
        pair
        for execution_scope in ("component", "whole_model")
        for pair in pairs_by_scope[execution_scope]
    )
    results: dict[tuple[str, int], dict[str, Json]] = {
        (check["check_id"], seed): {}
        for check, seed in pairs
    }
    role_elapsed_ns: dict[tuple[str, int], int] = {
        key: 0 for key in results
    }
    try:
        for execution_scope in ("component", "whole_model"):
            scoped_pairs = pairs_by_scope[execution_scope]
            if not scoped_pairs:
                continue
            with adapter.validation_stage(
                stage,
                execution_scope=execution_scope,
                cancel_requested=cancel_requested,
            ):
                for check, seed in scoped_pairs:
                    key = (check["check_id"], seed)
                    for role in ("reference", "candidate"):
                        _checkpoint(cancel_requested)
                        implementation = plan.implementation(role)
                        result, role_events, duration_ns = (
                            _execute_validation_role(
                                plan=plan,
                                plan_document=document,
                                stage=stage,
                                check=check,
                                seed=seed,
                                role=role,
                                implementation=implementation,
                                block_index=block_index,
                                adapter=adapter,
                                trace_paths=trace_paths,
                                cancel_requested=cancel_requested,
                            )
                        )
                        results[key][role] = result
                        events.extend(role_events)
                        role_elapsed_ns[key] += duration_ns
                        block_index += 1
                    comparison = adapter.compare_results(
                        ValidationComparisonRequest(
                            plan_id=plan.plan_id,
                            candidate_id=plan.candidate_id,
                            check=check,
                            seed=seed,
                            behavioral_contract=(
                                plan.behavioral_contract
                            ),
                        ),
                        results[key]["reference"],
                        results[key]["candidate"],
                    )
                    observation_document = _paired_observation(
                        plan=plan,
                        check=check,
                        seed=seed,
                        results=results[key],
                        comparison=comparison,
                    )
                    observation = ValidationObservation.from_json(
                        observation_document
                    )
                    observations.append(observation_document)
                    elapsed.append(
                        {
                            "observation_id": observation.observation_id,
                            "duration_ns": max(1, role_elapsed_ns[key]),
                        }
                    )
                    rejection = _behavioral_rejection(
                        plan,
                        check,
                        observation_document,
                    )
                    if rejection is not None:
                        status = "failed"
                        diagnostics.append(rejection)
                        break
            if status != "completed":
                break
    except ModelCompileCancelled as error:
        status = "cancelled"
        diagnostics.append(str(error))
    run_document = {
        "schema": VALIDATION_RUN_SCHEMA,
        "run_id": "",
        "plan_id": plan.plan_id,
        "stage": stage,
        "status": status,
        "execution_order": [
            observation["observation_id"]
            for observation in observations
        ],
        "observations": observations,
        "residency_events": events,
        "host_elapsed_ns": elapsed,
        "diagnostics": diagnostics,
    }
    run_document["run_id"] = validation_run_id(run_document)
    return ValidationRun.from_json(run_document)


def _execute_validation_role(
    *,
    plan: ValidationPlan,
    plan_document: Json,
    stage: str,
    check: Json,
    seed: int,
    role: str,
    implementation: Json,
    block_index: int,
    adapter: BehavioralValidationAdapter,
    trace_paths: set[str],
    cancel_requested: Callable[[], bool] | None,
) -> tuple[Json, tuple[Json, Json], int]:
    started = time.monotonic_ns()
    mount_request = ValidationRoleMountRequest(
        plan_id=plan.plan_id,
        candidate_id=plan.candidate_id,
        stage=stage,
        check=check,
        role=role,
        implementation=implementation,
        matched_conditions=dict(
            plan_document["matched_conditions"]
        ),
        matched_conditions_digest=str(
            plan_document["matched_conditions_digest"]
        ),
        seed=seed,
        block_index=block_index,
        cancel_requested=cancel_requested,
    )
    session = adapter.open_session(mount_request)
    mount: ValidationResidencyEvent | None = None
    try:
        mount = ValidationResidencyEvent.from_json(
            session.mount_event
        )
        _validate_mount(plan, mount_request, mount)
        execution_request = ValidationRoleExecutionRequest(
            plan_id=plan.plan_id,
            candidate_id=plan.candidate_id,
            check=check,
            role=role,
            implementation=implementation,
            matched_conditions=dict(
                plan_document["matched_conditions"]
            ),
            matched_conditions_digest=str(
                plan_document["matched_conditions_digest"]
            ),
            seed=seed,
        )
        result = ValidationRoleResult.from_json(
            session.execute(execution_request)
        )
        observed_paths = _validate_role_result(
            execution_request,
            result,
            adapter,
        )
        if trace_paths & observed_paths:
            raise ModelCompileError(
                "validation role results reused a raw trace path"
            )
        trace_paths.update(observed_paths)
    finally:
        if mount is not None:
            unmount = ValidationResidencyEvent.from_json(
                session.close()
            )
            _validate_unmount(
                plan,
                mount_request,
                mount,
                unmount,
            )
    return (
        result.to_json(),
        (mount.to_json(), unmount.to_json()),
        max(1, time.monotonic_ns() - started),
    )


def _behavioral_rejection(
    plan: ValidationPlan,
    check: Json,
    observation: Json,
) -> str | None:
    if observation["status"] != "completed":
        return (
            f"validation check {check['check_id']} seed "
            f"{observation['seed']} failed during execution"
        )
    completion_condition = check["horizon"]["completion_condition"]
    if any(
        observation[role]["horizon_completion"]["condition"]
        != completion_condition
        or not observation[role]["horizon_completion"]["satisfied"]
        for role in ("reference", "candidate")
    ):
        return (
            f"validation check {check['check_id']} did not satisfy its "
            "declared horizon completion condition"
        )
    observed_metrics = {
        metric["name"]: metric
        for metric in observation["metrics"]
    }
    if sorted(observed_metrics) != check["metrics"]:
        return (
            f"validation check {check['check_id']} did not report every "
            "declared metric"
        )
    behavioral = plan.behavioral_contract
    if behavioral["mode"] == "exact":
        comparison = check["comparison"]
        digest_fields = []
        if comparison["output_mode"] == "exact_digest":
            digest_fields.append("output_digest")
        if comparison["state_mode"] == "exact_digest":
            digest_fields.append("state_digest")
        for field in digest_fields:
            if (
                observation["reference"][field]
                != observation["candidate"][field]
            ):
                return (
                    f"exact candidate diverged in {field} during "
                    f"{check['check_id']}"
                )
        for metric in observed_metrics.values():
            if (
                metric["reference_value"] != metric["candidate_value"]
                or metric["error"] != 0
            ):
                return (
                    f"exact candidate diverged in metric "
                    f"{metric['name']!r} during {check['check_id']}"
                )
        return None
    limits = BehavioralErrorContract.from_json(
        behavioral["error_contract"]
    ).metric_limits
    for name, metric in observed_metrics.items():
        if metric["error"] > limits[name]:
            return (
                f"approximate candidate exceeded {name!r} error contract: "
                f"{metric['error']} > {limits[name]}"
            )
    return None


def _validate_role_result(
    request: ValidationRoleExecutionRequest,
    result: ValidationRoleResult,
    adapter: BehavioralValidationAdapter,
) -> set[str]:
    document = result.to_json()
    if (
        document["plan_id"] != request.plan_id
        or document["check_id"] != request.check["check_id"]
        or document["stage"] != request.check["stage"]
        or document["seed"] != request.seed
        or document["role"] != request.role
        or document["implementation_id"]
        != request.implementation["implementation_id"]
    ):
        raise ModelCompileError(
            "validation adapter returned a role result for another request"
        )
    completion = document["horizon_completion"]
    horizon = request.check["horizon"]
    if completion["condition"] != horizon["completion_condition"] or (
        completion["condition"] == "minimum_steps"
        and completion["minimum_steps"] != horizon["minimum_steps"]
    ):
        raise ModelCompileError(
            "validation adapter returned completion evidence for another "
            "horizon contract"
        )
    paths: set[str] = set()
    for trace in document["traces"]:
        path = str(trace["path"])
        if path in paths:
            raise ModelCompileError(
                "validation role result reused a trace path"
            )
        _verify_streamed_artifact(
            path,
            str(trace["digest"]),
            adapter.iter_trace_artifact,
        )
        paths.add(path)
    return paths


def _paired_observation(
    *,
    plan: ValidationPlan,
    check: Json,
    seed: int,
    results: dict[str, Json],
    comparison: Json,
) -> Json:
    if set(comparison) != {"metrics", "diagnostics"}:
        raise ModelCompileError(
            "validation result comparison fields are invalid"
        )
    if not isinstance(comparison["metrics"], list) or not isinstance(
        comparison["diagnostics"],
        list,
    ):
        raise ModelCompileError(
            "validation result comparison payload is invalid"
        )
    status = (
        "completed"
        if all(
            results[role]["status"] == "completed"
            for role in ("reference", "candidate")
        )
        else "failed"
    )
    diagnostics = [
        *(
            f"{role}: {diagnostic}"
            for role in ("reference", "candidate")
            for diagnostic in results[role]["diagnostics"]
        ),
        *comparison["diagnostics"],
    ]
    if status == "failed" and not diagnostics:
        diagnostics.append("one validation role failed without diagnostics")
    document = {
        "schema": VALIDATION_OBSERVATION_SCHEMA,
        "observation_id": "",
        "plan_id": plan.plan_id,
        "check_id": check["check_id"],
        "stage": check["stage"],
        "seed": seed,
        "status": status,
        "reference": _observation_role(results["reference"]),
        "candidate": _observation_role(results["candidate"]),
        "metrics": comparison["metrics"],
        "traces": {
            role: results[role]["traces"]
            for role in ("reference", "candidate")
        },
        "execution_statistics": {
            role: results[role]["default_statistics"]
            for role in ("reference", "candidate")
        },
        "diagnostics": diagnostics,
    }
    document["observation_id"] = validation_observation_id(document)
    return ValidationObservation.from_json(document).to_json()


def _observation_role(result: Json) -> Json:
    return {
        "implementation_id": result["implementation_id"],
        "output_digest": result["output_digest"],
        "state_digest": result["state_digest"],
        "steps": result["steps"],
        "horizon_completion": result["horizon_completion"],
    }


def _verify_fixture_artifacts(
    checks: tuple[Json, ...],
    adapter: BehavioralValidationAdapter,
    *,
    candidate_id: str,
) -> None:
    fixtures: dict[str, str] = {}
    limits: list[tuple[str, str, int]] = []
    for check in checks:
        for field in ("input", "initial_state"):
            reference = check[field]
            if reference is None:
                continue
            path = str(reference["path"])
            digest = str(reference["digest"])
            previous = fixtures.setdefault(path, digest)
            if previous != digest:
                raise ModelCompileError(
                    "validation fixture path is bound to multiple digests"
                )
        for basis in (
            check["regime"]["context_size_basis"],
            check["horizon"]["output_allowance_basis"],
        ):
            if basis["kind"] != "declared_model_limit":
                continue
            reference = basis["artifact"]
            path = str(reference["path"])
            digest = str(reference["digest"])
            previous = fixtures.setdefault(path, digest)
            if previous != digest:
                raise ModelCompileError(
                    "validation limit evidence path is bound to "
                    "multiple digests"
                )
            limits.append(
                (
                    path,
                    str(basis["json_pointer"]),
                    int(basis["declared_limit"]),
                )
            )
    for path, digest in sorted(fixtures.items()):
        _verify_streamed_artifact(
            path,
            digest,
            lambda relative_path, *, chunk_bytes=8 * 1024 * 1024: (
                adapter.iter_fixture_artifact(
                    relative_path,
                    candidate_id=candidate_id,
                    chunk_bytes=chunk_bytes,
                )
            ),
        )
    for path, pointer, expected in limits:
        _verify_declared_limit(
            adapter,
            candidate_id=candidate_id,
            relative_path=path,
            json_pointer=pointer,
            expected_limit=expected,
        )


def _verify_streamed_artifact(
    relative_path: str,
    expected_digest: str,
    reader,
) -> None:
    path = PurePosixPath(relative_path)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != relative_path
    ):
        raise ModelCompileError(
            f"validation artifact path is unsafe: {relative_path!r}"
        )
    digest = sha256()
    chunks = 0
    try:
        for chunk in reader(relative_path):
            if not isinstance(chunk, bytes) or not chunk:
                raise ModelCompileError(
                    "validation artifact readers must yield non-empty bytes"
                )
            digest.update(chunk)
            chunks += 1
    except (KeyError, OSError) as error:
        raise ModelCompileError(
            f"validation artifact is unavailable: {relative_path!r}"
        ) from error
    if chunks == 0:
        raise ModelCompileError(
            f"validation artifact is empty: {relative_path!r}"
        )
    observed = (
        f"{STAGED_ARTIFACT_DIGEST_SCHEMA}:{digest.hexdigest()}"
    )
    if observed != expected_digest:
        raise ModelCompileError(
            f"validation artifact digest mismatch: {relative_path!r}"
        )


def _verify_declared_limit(
    adapter: BehavioralValidationAdapter,
    *,
    candidate_id: str,
    relative_path: str,
    json_pointer: str,
    expected_limit: int,
) -> None:
    captured = bytearray()
    for chunk in adapter.iter_fixture_artifact(
        relative_path,
        candidate_id=candidate_id,
    ):
        if not isinstance(chunk, bytes) or not chunk:
            raise ModelCompileError(
                "validation limit evidence yielded invalid bytes"
            )
        captured.extend(chunk)
        if len(captured) > 1_048_576:
            raise ModelCompileError(
                "validation limit evidence exceeds 1 MiB"
            )
    try:
        value = json.loads(captured)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"validation limit evidence is not JSON: {relative_path!r}"
        ) from error
    try:
        for escaped in json_pointer.removeprefix("/").split("/"):
            segment = escaped.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict):
                value = value[segment]
            elif isinstance(value, list):
                value = value[int(segment)]
            else:
                raise KeyError(segment)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ModelCompileError(
            "validation declared-limit JSON pointer is unresolved: "
            f"{json_pointer!r}"
        ) from error
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected_limit
    ):
        raise ModelCompileError(
            "validation declared limit does not match immutable evidence"
        )


def _validate_mount(
    plan: ValidationPlan,
    request: ValidationRoleMountRequest,
    event: ValidationResidencyEvent,
) -> None:
    document = event.to_json()
    if (
        document["plan_id"] != plan.plan_id
        or document["stage"] != request.stage
        or document["check_id"] != request.check["check_id"]
        or document["seed"] != request.seed
        or document["role"] != request.role
        or document["implementation_id"]
        != request.implementation["implementation_id"]
        or document["block_index"] != request.block_index
        or document["action"] != "mount"
        or document["released"]
        or document["device_state_before_digest"]
        != request.matched_conditions["idle_device_state_digest"]
    ):
        raise ModelCompileError(
            "validation adapter returned invalid mount evidence"
        )


def _validate_unmount(
    plan: ValidationPlan,
    request: ValidationRoleMountRequest,
    mount: ValidationResidencyEvent,
    unmount: ValidationResidencyEvent,
) -> None:
    mounted = mount.to_json()
    released = unmount.to_json()
    if (
        released["plan_id"] != plan.plan_id
        or released["stage"] != request.stage
        or released["check_id"] != request.check["check_id"]
        or released["seed"] != request.seed
        or released["role"] != request.role
        or released["implementation_id"]
        != request.implementation["implementation_id"]
        or released["block_index"] != request.block_index
        or released["action"] != "unmount"
        or not released["released"]
        or released["device_state_before_digest"]
        != mounted["device_state_after_digest"]
        or released["device_state_after_digest"]
        != request.matched_conditions["idle_device_state_digest"]
    ):
        raise ModelCompileError(
            "validation adapter did not prove complete residency release"
        )


def _checkpoint(
    cancel_requested: Callable[[], bool] | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ModelCompileCancelled("candidate validation was cancelled")
