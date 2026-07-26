from __future__ import annotations

import json
import time
from hashlib import sha256
from typing import Callable

from nerve.compilation import Json, ModelCompileCancelled, ModelCompileError
from nerve.representation_optimizer.benchmarking.contracts import (
    BENCHMARK_RUN_SCHEMA,
    BenchmarkObservation,
    BenchmarkPlan,
    BenchmarkResidencyEvent,
    BenchmarkRun,
    benchmark_run_id,
)
from nerve.representation_optimizer.benchmarking.protocols import (
    BenchmarkExecutionRequest,
    BenchmarkMountRequest,
    NormalExecutionAdapter,
    NormalExecutionSession,
)
from nerve.representation_optimizer.contracts import contract_digest


def execute_benchmark_plan(
    plan: BenchmarkPlan,
    adapter: NormalExecutionAdapter,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> BenchmarkRun:
    document = plan.to_json()
    policy = document["policy"]
    observations: list[Json] = []
    events: list[Json] = []
    elapsed: list[Json] = []
    diagnostics: list[str] = []
    trace_paths: set[str] = set()
    run_status = "completed"
    block_index = 0

    _verify_fixture_artifacts(plan, adapter)

    def execute(
        session: NormalExecutionSession,
        *,
        role: str,
        workload: Json,
        phase: str,
        seed: int,
        pair_index: int,
        order_index: int,
    ) -> bool:
        nonlocal run_status
        _checkpoint(cancel_requested)
        request = BenchmarkExecutionRequest(
            plan_id=document["plan_id"],
            role=role,
            implementation_id=document["implementations"][role][
                "implementation_id"
            ],
            workload=workload,
            matched_conditions=document["matched_conditions"],
            matched_conditions_digest=document["matched_conditions_digest"],
            phase=phase,
            seed=seed,
            pair_index=pair_index,
            order_index=order_index,
        )
        started = time.monotonic_ns()
        raw = session.execute(request)
        host_elapsed = max(1, time.monotonic_ns() - started)
        observation = BenchmarkObservation.from_json(raw)
        observed_trace_paths = _validate_observation(
            observation,
            request,
            adapter,
        )
        if trace_paths & observed_trace_paths:
            raise ModelCompileError(
                "benchmark observations reused a raw trace artifact path"
            )
        trace_paths.update(observed_trace_paths)
        observation_document = observation.to_json()
        observations.append(observation_document)
        elapsed.append(
            {
                "observation_id": observation.observation_id,
                "duration_ns": host_elapsed,
            }
        )
        if observation_document["status"] != "completed":
            run_status = observation_document["status"]
            diagnostics.append(
                f"{role} {workload['workload_id']} {phase} "
                f"{pair_index} ended as {run_status}: "
                f"{observation_document['stop_reason']}"
            )
            return False
        return True

    try:
        for workload_index, workload in enumerate(document["workloads"]):
            for seed_index, seed in enumerate(workload["randomness"]["seeds"]):
                if workload["regime"]["mount_mode"] == "resident_reuse":
                    roles = _role_order(seed_index)
                    for order_index, role in enumerate(roles):
                        _checkpoint(cancel_requested)
                        session, mount = _open_session(
                            plan,
                            adapter,
                            workload=workload,
                            role=role,
                            seed=seed,
                            block_index=block_index,
                        )
                        block_index += 1
                        events.append(mount.to_json())
                        keep_running = True
                        unmount: BenchmarkResidencyEvent | None = None
                        try:
                            for warmup_index in range(policy["warmup_samples"]):
                                keep_running = execute(
                                    session,
                                    role=role,
                                    workload=workload,
                                    phase="warmup",
                                    seed=seed,
                                    pair_index=warmup_index,
                                    order_index=order_index,
                                )
                                if not keep_running:
                                    break
                            if keep_running:
                                for pair_index in range(
                                    policy["measured_pairs_per_seed"]
                                ):
                                    keep_running = execute(
                                        session,
                                        role=role,
                                        workload=workload,
                                        phase="measured",
                                        seed=seed,
                                        pair_index=pair_index,
                                        order_index=order_index,
                                    )
                                    if not keep_running:
                                        break
                        finally:
                            unmount = BenchmarkResidencyEvent.from_json(
                                session.close()
                            )
                            _validate_unmount(plan, mount, unmount)
                            events.append(unmount.to_json())
                        if not keep_running:
                            break
                    if run_status != "completed":
                        break
                else:
                    for phase, count in (
                        ("warmup", policy["warmup_samples"]),
                        ("measured", policy["measured_pairs_per_seed"]),
                    ):
                        for pair_index in range(count):
                            roles = _role_order(
                                workload_index + seed_index + pair_index
                            )
                            for order_index, role in enumerate(roles):
                                _checkpoint(cancel_requested)
                                session, mount = _open_session(
                                    plan,
                                    adapter,
                                    workload=workload,
                                    role=role,
                                    seed=seed,
                                    block_index=block_index,
                                )
                                block_index += 1
                                events.append(mount.to_json())
                                keep_running = True
                                try:
                                    keep_running = execute(
                                        session,
                                        role=role,
                                        workload=workload,
                                        phase=phase,
                                        seed=seed,
                                        pair_index=pair_index,
                                        order_index=order_index,
                                    )
                                finally:
                                    unmount = BenchmarkResidencyEvent.from_json(
                                        session.close()
                                    )
                                    _validate_unmount(plan, mount, unmount)
                                    events.append(unmount.to_json())
                                if not keep_running:
                                    break
                            if run_status != "completed":
                                break
                        if run_status != "completed":
                            break
                    if run_status != "completed":
                        break
            if run_status != "completed":
                break
    except ModelCompileCancelled as error:
        if not observations:
            raise
        run_status = "cancelled"
        diagnostics.append(str(error))

    run_document = {
        "schema": BENCHMARK_RUN_SCHEMA,
        "run_id": "",
        "plan_id": document["plan_id"],
        "status": run_status,
        "execution_order": [
            observation["observation_id"] for observation in observations
        ],
        "observations": observations,
        "residency_events": events,
        "host_elapsed_ns": elapsed,
        "diagnostics": diagnostics,
    }
    run_document["run_id"] = benchmark_run_id(run_document)
    run = BenchmarkRun.from_json(run_document)
    if run_status == "completed":
        validate_complete_run_against_plan(plan, run)
    return run


def validate_complete_run_against_plan(
    plan: BenchmarkPlan,
    run: BenchmarkRun,
) -> None:
    plan_document = plan.to_json()
    run_document = run.to_json()
    if (
        run_document["status"] != "completed"
        or run_document["plan_id"] != plan_document["plan_id"]
    ):
        raise ModelCompileError("completed benchmark run does not match its plan")
    expected_trials, expected_residency = _expected_execution(plan_document)
    observed_trials = [
        (
            observation["workload_id"],
            observation["role"],
            observation["seed"],
            observation["phase"],
            observation["pair_index"],
            observation["order_index"],
        )
        for observation in run_document["observations"]
    ]
    if observed_trials != expected_trials:
        raise ModelCompileError(
            "completed benchmark execution order does not exactly match its plan"
        )
    observed = {
        trial[:5]: observation
        for trial, observation in zip(
            observed_trials,
            run_document["observations"],
            strict=True,
        )
    }
    for key in sorted(
        {
            (
                observation["workload_id"],
                observation["seed"],
                observation["phase"],
                observation["pair_index"],
            )
            for observation in run_document["observations"]
        }
    ):
        reference = observed[(key[0], "reference", *key[1:])]
        candidate = observed[(key[0], "candidate", *key[1:])]
        if (
            reference["work"]["unit"] != candidate["work"]["unit"]
            or reference["work"]["useful_units"]
            != candidate["work"]["useful_units"]
        ):
            raise ModelCompileError(
                "matched benchmark pair performed different useful work"
            )
    expected_idle = plan_document["matched_conditions"][
        "idle_device_state_digest"
    ]
    mounts: dict[tuple[str, int], Json] = {}
    unmounts: dict[tuple[str, int], Json] = {}
    residency_sequence = []
    for event in run_document["residency_events"]:
        residency_sequence.append(
            (
                event["action"],
                event["workload_id"],
                event["role"],
                event["seed"],
                event["block_index"],
            )
        )
        key = (event["role"], event["block_index"])
        destination = mounts if event["action"] == "mount" else unmounts
        if key in destination:
            raise ModelCompileError(
                "benchmark run duplicates a residency lifecycle event"
            )
        destination[key] = event
        if event["matched_conditions_digest"] != plan_document[
            "matched_conditions_digest"
        ]:
            raise ModelCompileError(
                "benchmark residency event changed matched conditions"
            )
        if event["implementation_id"] != plan_document["implementations"][
            event["role"]
        ]["implementation_id"]:
            raise ModelCompileError(
                "benchmark residency event names the wrong implementation"
            )
    if residency_sequence != expected_residency:
        raise ModelCompileError(
            "completed benchmark residency order does not match its plan"
        )
    if set(mounts) != set(unmounts):
        raise ModelCompileError(
            "benchmark run did not pair every mount with an unmount"
        )
    for key, mount in mounts.items():
        unmount = unmounts[key]
        if (
            mount["device_state_before_digest"] != expected_idle
            or unmount["device_state_after_digest"] != expected_idle
            or mount["device_state_after_digest"]
            != unmount["device_state_before_digest"]
        ):
            raise ModelCompileError(
                "benchmark device residency did not return to matched idle state"
            )


def _expected_execution(
    plan_document: Json,
) -> tuple[list[tuple[str, str, int, str, int, int]], list[tuple[str, str, str, int, int]]]:
    trials = []
    residency = []
    policy = plan_document["policy"]
    block_index = 0
    for workload_index, workload in enumerate(plan_document["workloads"]):
        workload_id = workload["workload_id"]
        for seed_index, seed in enumerate(workload["randomness"]["seeds"]):
            if workload["regime"]["mount_mode"] == "resident_reuse":
                for order_index, role in enumerate(_role_order(seed_index)):
                    residency.append(
                        ("mount", workload_id, role, seed, block_index)
                    )
                    for phase, count in (
                        ("warmup", policy["warmup_samples"]),
                        ("measured", policy["measured_pairs_per_seed"]),
                    ):
                        trials.extend(
                            (
                                workload_id,
                                role,
                                seed,
                                phase,
                                pair_index,
                                order_index,
                            )
                            for pair_index in range(count)
                        )
                    residency.append(
                        ("unmount", workload_id, role, seed, block_index)
                    )
                    block_index += 1
            else:
                for phase, count in (
                    ("warmup", policy["warmup_samples"]),
                    ("measured", policy["measured_pairs_per_seed"]),
                ):
                    for pair_index in range(count):
                        roles = _role_order(
                            workload_index + seed_index + pair_index
                        )
                        for order_index, role in enumerate(roles):
                            residency.append(
                                (
                                    "mount",
                                    workload_id,
                                    role,
                                    seed,
                                    block_index,
                                )
                            )
                            trials.append(
                                (
                                    workload_id,
                                    role,
                                    seed,
                                    phase,
                                    pair_index,
                                    order_index,
                                )
                            )
                            residency.append(
                                (
                                    "unmount",
                                    workload_id,
                                    role,
                                    seed,
                                    block_index,
                                )
                            )
                            block_index += 1
    return trials, residency


def _open_session(
    plan: BenchmarkPlan,
    adapter: NormalExecutionAdapter,
    *,
    workload: Json,
    role: str,
    seed: int,
    block_index: int,
) -> tuple[NormalExecutionSession, BenchmarkResidencyEvent]:
    document = plan.to_json()
    request = BenchmarkMountRequest(
        plan_id=document["plan_id"],
        role=role,
        implementation=document["implementations"][role],
        workload=workload,
        matched_conditions=document["matched_conditions"],
        matched_conditions_digest=document["matched_conditions_digest"],
        seed=seed,
        block_index=block_index,
    )
    session = adapter.open_session(request)
    try:
        mount = BenchmarkResidencyEvent.from_json(session.mount_event)
        _validate_mount(plan, mount, request)
        return session, mount
    except BaseException:
        try:
            released = BenchmarkResidencyEvent.from_json(session.close())
            _validate_emergency_release(plan, released)
        except BaseException as cleanup_error:
            raise ModelCompileError(
                "failed to prove device release after an invalid mount"
            ) from cleanup_error
        raise


def _validate_mount(
    plan: BenchmarkPlan,
    event: BenchmarkResidencyEvent,
    request: BenchmarkMountRequest,
) -> None:
    document = event.to_json()
    expected_idle = plan.matched_conditions["idle_device_state_digest"]
    if (
        document["action"] != "mount"
        or document["plan_id"] != request.plan_id
        or document["role"] != request.role
        or document["implementation_id"]
        != request.implementation["implementation_id"]
        or document["workload_id"] != request.workload["workload_id"]
        or document["seed"] != request.seed
        or document["block_index"] != request.block_index
        or document["matched_conditions_digest"]
        != request.matched_conditions_digest
        or document["device_state_before_digest"] != expected_idle
    ):
        raise ModelCompileError(
            "normal execution adapter returned a mismatched mount event"
        )


def _validate_unmount(
    plan: BenchmarkPlan,
    mount: BenchmarkResidencyEvent,
    unmount: BenchmarkResidencyEvent,
) -> None:
    mounted = mount.to_json()
    released = unmount.to_json()
    expected_idle = plan.matched_conditions["idle_device_state_digest"]
    if (
        released["action"] != "unmount"
        or any(
            released[field] != mounted[field]
            for field in (
                "plan_id",
                "implementation_id",
                "role",
                "workload_id",
                "seed",
                "block_index",
                "matched_conditions_digest",
            )
        )
        or released["device_state_before_digest"]
        != mounted["device_state_after_digest"]
        or released["device_state_after_digest"] != expected_idle
    ):
        raise ModelCompileError(
            "normal execution adapter did not release matched residency"
        )


def _validate_emergency_release(
    plan: BenchmarkPlan,
    event: BenchmarkResidencyEvent,
) -> None:
    released = event.to_json()
    conditions = plan.matched_conditions
    if (
        released["action"] != "unmount"
        or not released["released"]
        or released["matched_conditions_digest"]
        != contract_digest(conditions)
        or released["device_state_after_digest"]
        != conditions["idle_device_state_digest"]
    ):
        raise ModelCompileError(
            "normal execution adapter did not prove emergency device release"
        )


def _validate_observation(
    observation: BenchmarkObservation,
    request: BenchmarkExecutionRequest,
    adapter: NormalExecutionAdapter,
) -> set[str]:
    document = observation.to_json()
    workload = request.workload
    initial_state = workload["initial_state"]
    expected = {
        "plan_id": request.plan_id,
        "implementation_id": request.implementation_id,
        "role": request.role,
        "workload_id": workload["workload_id"],
        "phase": request.phase,
        "seed": request.seed,
        "pair_index": request.pair_index,
        "order_index": request.order_index,
        "matched_conditions_digest": request.matched_conditions_digest,
        "input_digest": workload["input"]["digest"],
        "initial_state_digest": (
            initial_state["digest"] if initial_state is not None else None
        ),
        "controls_digest": contract_digest(workload["controls"]),
    }
    if any(document[field] != value for field, value in expected.items()):
        raise ModelCompileError(
            "normal execution observation changed matched trial conditions"
        )
    if document["work"]["unit"] != workload["useful_work"]["unit"]:
        raise ModelCompileError(
            "normal execution observation changed useful-work units"
        )
    if (
        document["status"] == "completed"
        and document["work"]["useful_units"]
        < workload["useful_work"]["minimum_units"]
    ):
        raise ModelCompileError(
            "normal execution stopped before minimum useful work"
        )
    allowance = workload["useful_work"]["output_allowance"]
    if (
        allowance is not None
        and document["work"]["useful_units"] > allowance
    ):
        raise ModelCompileError(
            "normal execution exceeded the declared output allowance"
        )
    required_windows = workload["useful_work"]["sustained_window_count"]
    if (
        document["status"] == "completed"
        and request.phase == "measured"
        and len(document["throughput_windows"]) < required_windows
    ):
        raise ModelCompileError(
            "normal execution omitted required sustained throughput windows"
        )
    trace_paths = set()
    for name, trace in document["traces"].items():
        digest = sha256()
        for chunk in adapter.iter_trace_artifact(trace["path"]):
            if not isinstance(chunk, bytes):
                raise ModelCompileError(
                    f"normal execution trace {name!r} yielded a non-byte chunk"
                )
            digest.update(chunk)
        actual = (
            "nerve.optimizer.artifact_sha256.v1:"
            f"{digest.hexdigest()}"
        )
        if actual != trace["digest"]:
            raise ModelCompileError(
                f"normal execution trace {name!r} failed digest validation"
            )
        trace_paths.add(trace["path"])
    return trace_paths


def _verify_fixture_artifacts(
    plan: BenchmarkPlan,
    adapter: NormalExecutionAdapter,
) -> None:
    artifacts: dict[str, str] = {}
    limit_evidence: dict[str, list[Json]] = {}
    for workload in plan.to_json()["workloads"]:
        for reference in (workload["input"], workload["initial_state"]):
            if reference is None:
                continue
            prior = artifacts.setdefault(reference["path"], reference["digest"])
            if prior != reference["digest"]:
                raise ModelCompileError(
                    "benchmark workloads bind one fixture path to different bytes"
                )
        basis = workload["useful_work"]["output_allowance_basis"]
        if basis["kind"] == "declared_model_limit":
            reference = basis["artifact"]
            prior = artifacts.setdefault(
                reference["path"],
                reference["digest"],
            )
            if prior != reference["digest"]:
                raise ModelCompileError(
                    "benchmark workloads bind limit evidence to different bytes"
                )
            limit_evidence.setdefault(reference["path"], []).append(
                {
                    "json_pointer": basis["json_pointer"],
                    "declared_limit": basis["declared_limit"],
                }
            )
    for relative_path, expected_digest in sorted(artifacts.items()):
        digest = sha256()
        captured = bytearray()
        for chunk in adapter.iter_fixture_artifact(relative_path):
            if not isinstance(chunk, bytes):
                raise ModelCompileError(
                    "benchmark fixture source yielded a non-byte chunk"
                )
            digest.update(chunk)
            if relative_path in limit_evidence:
                captured.extend(chunk)
                if len(captured) > 1_048_576:
                    raise ModelCompileError(
                        "benchmark limit-evidence artifact exceeds 1 MiB"
                    )
        actual = (
            "nerve.optimizer.artifact_sha256.v1:"
            f"{digest.hexdigest()}"
        )
        if actual != expected_digest:
            raise ModelCompileError(
                f"benchmark fixture failed digest validation: {relative_path!r}"
            )
        for claim in limit_evidence.get(relative_path, ()):
            _validate_declared_limit(
                relative_path=relative_path,
                payload=bytes(captured),
                json_pointer=claim["json_pointer"],
                expected_limit=claim["declared_limit"],
            )


def _validate_declared_limit(
    *,
    relative_path: str,
    payload: bytes,
    json_pointer: str,
    expected_limit: int,
) -> None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCompileError(
            f"benchmark limit evidence is not JSON: {relative_path!r}"
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
            f"benchmark output-limit JSON pointer is unresolved: {json_pointer!r}"
        ) from error
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected_limit
    ):
        raise ModelCompileError(
            "benchmark output allowance does not match its immutable evidence"
        )


def _role_order(index: int) -> tuple[str, str]:
    return (
        ("reference", "candidate")
        if index % 2 == 0
        else ("candidate", "reference")
    )


def _checkpoint(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ModelCompileCancelled("matched candidate benchmark cancelled")
