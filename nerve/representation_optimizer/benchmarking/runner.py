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
from nerve.representation_optimizer.benchmarking.statistics import (
    summarize_workload_samples,
    warmup_group_summary,
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
    sampling_outcomes: list[Json] = []
    diagnostics: list[str] = []
    trace_paths: set[str] = set()
    run_status = "completed"
    block_index = 0
    benchmark_started_ns = time.monotonic_ns()
    maximum_duration_ns = policy["maximum_benchmark_duration_ns"]
    deadline_reached = False

    def benchmark_stop_requested() -> bool:
        nonlocal deadline_reached
        if cancel_requested is not None and cancel_requested():
            return True
        if time.monotonic_ns() - benchmark_started_ns >= maximum_duration_ns:
            deadline_reached = True
            return True
        return False

    def record_timeout() -> None:
        nonlocal run_status
        run_status = "timeout"
        message = "microbenchmark exceeded its one-minute wall-clock contract"
        if message not in diagnostics:
            diagnostics.append(message)

    def execute(
        session: NormalExecutionSession,
        *,
        role: str,
        workload: Json,
        phase: str,
        seed: int,
        block_index: int,
        pair_index: int,
        order_index: int,
    ) -> Json | None:
        nonlocal run_status
        _checkpoint(benchmark_stop_requested)
        request = BenchmarkExecutionRequest(
            plan_id=document["plan_id"],
            role=role,
            implementation_id=document["implementations"][role]["implementation_id"],
            workload=workload,
            matched_conditions=document["matched_conditions"],
            matched_conditions_digest=document["matched_conditions_digest"],
            phase=phase,
            seed=seed,
            block_index=block_index,
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
            return None
        if benchmark_stop_requested():
            if deadline_reached:
                record_timeout()
                return None
            _checkpoint(benchmark_stop_requested)
        return observation_document

    def open_session(
        *,
        workload: Json,
        role: str,
        seed: int,
    ) -> tuple[NormalExecutionSession, BenchmarkResidencyEvent, int]:
        nonlocal block_index
        _checkpoint(benchmark_stop_requested)
        current_block = block_index
        session, mount = _open_session(
            plan,
            adapter,
            workload=workload,
            role=role,
            seed=seed,
            block_index=current_block,
            cancel_requested=benchmark_stop_requested,
        )
        block_index += 1
        events.append(mount.to_json())
        return session, mount, current_block

    def close_session(
        session: NormalExecutionSession,
        mount: BenchmarkResidencyEvent,
    ) -> None:
        unmount = BenchmarkResidencyEvent.from_json(session.close())
        _validate_unmount(plan, mount, unmount)
        events.append(unmount.to_json())

    def resident_block(
        *,
        workload: Json,
        role: str,
        seed: int,
        order_index: int,
    ) -> Json:
        session, mount, current_block = open_session(
            workload=workload,
            role=role,
            seed=seed,
        )
        warmup_observations: list[Json] = []
        try:
            for warmup_index in range(policy["maximum_warmup_samples"]):
                observation = execute(
                    session,
                    role=role,
                    workload=workload,
                    phase="warmup",
                    seed=seed,
                    block_index=current_block,
                    pair_index=warmup_index,
                    order_index=order_index,
                )
                if observation is None:
                    break
                warmup_observations.append(observation)
                if warmup_group_summary(
                    warmup_observations,
                    policy,
                )["converged"]:
                    break
            summary = warmup_group_summary(warmup_observations, policy)
            if run_status == "completed" and summary["converged"]:
                execute(
                    session,
                    role=role,
                    workload=workload,
                    phase="measured",
                    seed=seed,
                    block_index=current_block,
                    pair_index=0,
                    order_index=order_index,
                )
        finally:
            close_session(session, mount)
        return {
            "role": role,
            "seed": seed,
            "cycle_index": None,
            "order_block_index": None,
            "attempt_index": 0,
            **summary,
            "observation_ids": [
                observation["observation_id"] for observation in warmup_observations
            ],
        }

    def cold_observation(
        *,
        workload: Json,
        role: str,
        phase: str,
        seed: int,
        pair_index: int,
        order_index: int,
    ) -> Json | None:
        session, mount, current_block = open_session(
            workload=workload,
            role=role,
            seed=seed,
        )
        try:
            return execute(
                session,
                role=role,
                workload=workload,
                phase=phase,
                seed=seed,
                block_index=current_block,
                pair_index=pair_index,
                order_index=order_index,
            )
        finally:
            close_session(session, mount)

    try:
        _verify_fixture_artifacts(
            plan,
            adapter,
            cancel_requested=benchmark_stop_requested,
        )
        for workload_index, workload in enumerate(document["workloads"]):
            warmup_groups: list[Json] = []
            seeds = workload["randomness"]["seeds"]
            if workload["regime"]["mount_mode"] == "cold":
                for seed_index, seed in enumerate(seeds):
                    by_role: dict[str, list[Json]] = {
                        "reference": [],
                        "candidate": [],
                    }
                    for warmup_index in range(policy["maximum_warmup_samples"]):
                        roles = _role_order(workload_index + seed_index + warmup_index)
                        for order_index, role in enumerate(roles):
                            observation = cold_observation(
                                workload=workload,
                                role=role,
                                phase="warmup",
                                seed=seed,
                                pair_index=warmup_index,
                                order_index=order_index,
                            )
                            if observation is None:
                                break
                            by_role[role].append(observation)
                        if run_status != "completed":
                            break
                        if all(
                            warmup_group_summary(by_role[role], policy)["converged"]
                            for role in ("reference", "candidate")
                        ):
                            break
                    for role in ("reference", "candidate"):
                        summary = warmup_group_summary(by_role[role], policy)
                        warmup_groups.append(
                            {
                                "role": role,
                                "seed": seed,
                                "cycle_index": None,
                                "order_block_index": None,
                                "attempt_index": 0,
                                **summary,
                                "observation_ids": [
                                    observation["observation_id"]
                                    for observation in by_role[role]
                                ],
                            }
                        )
                    if run_status != "completed":
                        break
                if any(not group["converged"] for group in warmup_groups):
                    run_status = "failed"
                    diagnostics.append(
                        f"{workload['workload_id']} cold warmup did not converge "
                        "within the declared bound"
                    )
            if run_status != "completed":
                break

            for seed_index, seed in enumerate(seeds):
                roles = _role_order(workload_index + seed_index)
                for order_index, role in enumerate(roles):
                    if workload["regime"]["mount_mode"] == "resident_reuse":
                        group = resident_block(
                            workload=workload,
                            role=role,
                            seed=seed,
                            order_index=order_index,
                        )
                        warmup_groups.append(group)
                        if run_status == "completed" and not group["converged"]:
                            run_status = "failed"
                            diagnostics.append(
                                f"{role} {workload['workload_id']} fixed warmup failed"
                            )
                    else:
                        cold_observation(
                            workload=workload,
                            role=role,
                            phase="measured",
                            seed=seed,
                            pair_index=0,
                            order_index=order_index,
                        )
                    if run_status != "completed":
                        break
                if run_status != "completed":
                    break
            if run_status != "completed":
                break
            summary, _ = summarize_workload_samples(
                workload,
                observations,
                policy,
                warmup_groups,
            )
            termination = (
                "invalid" if summary["decision"] == "invalid"
                else "fixed_sample_complete"
            )
            sampling_outcomes.append(
                {
                    "workload_id": workload["workload_id"],
                    "warmup_groups": sorted(
                        warmup_groups,
                        key=lambda group: (
                            group["seed"],
                            -1
                            if group["cycle_index"] is None
                            else group["cycle_index"],
                            -1
                            if group["order_block_index"] is None
                            else group["order_block_index"],
                            ("reference", "candidate").index(group["role"]),
                            group["attempt_index"],
                        ),
                    ),
                    "measured_calls_per_role": policy["measured_calls_per_role"],
                    "decision": summary["decision"],
                    "reasons": summary["reasons"],
                    "termination": termination,
                }
            )
    except ModelCompileCancelled as error:
        if deadline_reached:
            record_timeout()
        elif not observations:
            raise
        else:
            run_status = "cancelled"
            diagnostics.append(str(error))

    if run_status == "completed" and benchmark_stop_requested():
        if deadline_reached:
            record_timeout()
        else:
            _checkpoint(benchmark_stop_requested)

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
        "sampling_outcomes": sampling_outcomes,
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
    policy = plan_document["policy"]
    observations = run_document["observations"]
    observation_by_id = {
        observation["observation_id"]: observation for observation in observations
    }
    workloads = {
        workload["workload_id"]: (index, workload)
        for index, workload in enumerate(plan_document["workloads"])
    }
    outcomes = {
        outcome["workload_id"]: outcome for outcome in run_document["sampling_outcomes"]
    }
    if set(outcomes) != set(workloads):
        raise ModelCompileError(
            "completed benchmark sampling outcomes do not exactly cover its plan"
        )

    expected_reservation = plan_document["matched_conditions"]["capacity_reservation_digest"]
    events = run_document["residency_events"]
    if len(events) % 2:
        raise ModelCompileError(
            "completed benchmark has an incomplete residency lifecycle"
        )
    mounts: dict[int, Json] = {}
    for expected_block, event_index in enumerate(range(0, len(events), 2)):
        mount = events[event_index]
        unmount = events[event_index + 1]
        if (
            mount["action"] != "mount"
            or unmount["action"] != "unmount"
            or mount["block_index"] != expected_block
            or unmount["block_index"] != expected_block
            or any(
                mount[field] != unmount[field]
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
        ):
            raise ModelCompileError(
                "completed benchmark residency order is not paired and contiguous"
            )
        if (
            mount["matched_conditions_digest"]
            != plan_document["matched_conditions_digest"]
            or mount["implementation_id"]
            != plan_document["implementations"][mount["role"]]["implementation_id"]
        ):
            raise ModelCompileError(
                "benchmark residency changed its implementation or conditions"
            )
        if (
            mount["device_state_before_digest"] != expected_reservation
            or unmount["device_state_after_digest"] != expected_reservation
            or mount["device_state_after_digest"]
            != unmount["device_state_before_digest"]
        ):
            raise ModelCompileError(
                "benchmark device residency did not restore the matched capacity reservation"
            )
        mounts[expected_block] = mount

    by_block: dict[int, list[Json]] = {}
    previous_block = -1
    for observation in observations:
        block = observation["block_index"]
        if block < previous_block:
            raise ModelCompileError(
                "benchmark observations are not in residency-block order"
            )
        previous_block = block
        mount = mounts.get(block)
        if mount is None or any(
            observation[field] != mount[field]
            for field in ("role", "workload_id", "seed", "block_index")
        ):
            raise ModelCompileError(
                "benchmark observation does not belong to its residency block"
            )
        if (
            observation["implementation_id"] != mount["implementation_id"]
            or observation["matched_conditions_digest"]
            != mount["matched_conditions_digest"]
        ):
            raise ModelCompileError(
                "benchmark observation changed its mounted implementation"
            )
        by_block.setdefault(block, []).append(observation)
    if set(by_block) != set(mounts):
        raise ModelCompileError(
            "benchmark residency blocks do not exactly cover observations"
        )
    for block, block_observations in by_block.items():
        phases = [observation["phase"] for observation in block_observations]
        if phases != sorted(phases, key=("warmup", "measured").index):
            raise ModelCompileError("benchmark block measured before completing warmup")
        workload = workloads[mounts[block]["workload_id"]][1]
        if workload["regime"]["mount_mode"] == "cold" and len(block_observations) != 1:
            raise ModelCompileError(
                "cold benchmark residency block executed more than one trial"
            )

    warmup_ids: list[str] = []
    for workload_id, (workload_index, workload) in workloads.items():
        outcome = outcomes[workload_id]
        if (
            outcome["measured_calls_per_role"]
            != policy["measured_calls_per_role"]
            or outcome["measured_calls_per_role"] != 1
        ):
            raise ModelCompileError(
                "microbenchmark did not use exactly one measured call per role"
            )
        expected_group_keys = {
            (seed, role)
            for seed in workload["randomness"]["seeds"]
            for role in ("reference", "candidate")
        }
        observed_group_keys = {
            (group["seed"], group["role"])
            for group in outcome["warmup_groups"]
        }
        if observed_group_keys != expected_group_keys:
            raise ModelCompileError(
                "fixed warmups do not cover every benchmark role"
            )
        for group in outcome["warmup_groups"]:
            if (
                group["cycle_index"] is not None
                or group["order_block_index"] is not None
                or group["attempt_index"] != 0
                or not group["converged"]
                or group["sample_count"] != 1
            ):
                raise ModelCompileError(
                    "microbenchmark warmup is not exactly one discarded call"
                )
            selected = []
            for observation_id in group["observation_ids"]:
                observation = observation_by_id.get(observation_id)
                if (
                    observation is None
                    or observation["phase"] != "warmup"
                    or observation["workload_id"] != workload_id
                    or observation["role"] != group["role"]
                    or observation["seed"] != group["seed"]
                ):
                    raise ModelCompileError(
                        "fixed warmup group cites a mismatched observation"
                    )
                selected.append(observation)
                warmup_ids.append(observation_id)
            computed = warmup_group_summary(selected, policy)
            if any(
                group[field] != computed[field]
                for field in ("sample_count", "maximum_shift_ppm", "converged")
            ):
                raise ModelCompileError(
                    "fixed warmup outcome disagrees with raw observations"
                )
            if workload["regime"]["mount_mode"] == "resident_reuse":
                block = selected[0]["block_index"]
                measured_in_block = [
                    observation
                    for observation in by_block[block]
                    if observation["phase"] == "measured"
                ]
                if (
                    len(measured_in_block) != 1
                    or measured_in_block[0]["role"] != group["role"]
                    or measured_in_block[0]["pair_index"] != 0
                ):
                    raise ModelCompileError(
                        "resident benchmark did not measure immediately after warmup"
                    )
        selected_workload = [
            observation
            for observation in observations
            if observation["workload_id"] == workload_id
        ]
        for seed_index, seed in enumerate(workload["randomness"]["seeds"]):
            measured = [
                observation
                for observation in selected_workload
                if observation["phase"] == "measured" and observation["seed"] == seed
            ]
            if (
                len(measured) != 2
                or {observation["role"] for observation in measured}
                != {"reference", "candidate"}
                or {observation["pair_index"] for observation in measured} != {0}
            ):
                raise ModelCompileError(
                    "microbenchmark did not collect one matched measured call"
                )
            expected_roles = _role_order(workload_index + seed_index)
            by_role = {observation["role"]: observation for observation in measured}
            if any(
                by_role[role]["order_index"] != expected_roles.index(role)
                for role in expected_roles
            ):
                raise ModelCompileError(
                    "microbenchmark measured calls changed their declared order"
                )
            reference = by_role["reference"]
            candidate = by_role["candidate"]
            if (
                reference["work"]["unit"] != candidate["work"]["unit"]
                or reference["work"]["useful_units"]
                != candidate["work"]["useful_units"]
            ):
                raise ModelCompileError(
                    "matched benchmark pair performed different useful work"
                )
        summary, _ = summarize_workload_samples(
            workload,
            observations,
            policy,
            outcome["warmup_groups"],
        )
        if (
            outcome["decision"] != summary["decision"]
            or outcome["reasons"] != summary["reasons"]
        ):
            raise ModelCompileError(
                "microbenchmark termination disagrees with its evidence"
            )
        expected_termination = (
            "invalid"
            if summary["decision"] == "invalid"
            else "fixed_sample_complete"
        )
        if outcome["termination"] != expected_termination:
            raise ModelCompileError(
                "microbenchmark stopped before its binary decision"
            )
    observed_warmup_ids = [
        observation["observation_id"]
        for observation in observations
        if observation["phase"] == "warmup"
    ]
    if len(warmup_ids) != len(set(warmup_ids)) or set(warmup_ids) != set(
        observed_warmup_ids
    ):
        raise ModelCompileError(
            "fixed warmup groups do not exactly partition warmup evidence"
        )


def _open_session(
    plan: BenchmarkPlan,
    adapter: NormalExecutionAdapter,
    *,
    workload: Json,
    role: str,
    seed: int,
    block_index: int,
    cancel_requested: Callable[[], bool] | None,
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
        cancel_requested=cancel_requested,
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
    expected_reservation = plan.matched_conditions["capacity_reservation_digest"]
    if (
        document["action"] != "mount"
        or document["plan_id"] != request.plan_id
        or document["role"] != request.role
        or document["implementation_id"] != request.implementation["implementation_id"]
        or document["workload_id"] != request.workload["workload_id"]
        or document["seed"] != request.seed
        or document["block_index"] != request.block_index
        or document["matched_conditions_digest"] != request.matched_conditions_digest
        or document["device_state_before_digest"] != expected_reservation
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
    expected_reservation = plan.matched_conditions["capacity_reservation_digest"]
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
        or released["device_state_after_digest"] != expected_reservation
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
        or released["matched_conditions_digest"] != contract_digest(conditions)
        or released["device_state_after_digest"]
        != conditions["capacity_reservation_digest"]
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
        "block_index": request.block_index,
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
        and document["work"]["useful_units"] < workload["useful_work"]["minimum_units"]
    ):
        raise ModelCompileError("normal execution stopped before minimum useful work")
    allowance = workload["useful_work"]["output_allowance"]
    if allowance is not None and document["work"]["useful_units"] > allowance:
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
        actual = f"nerve.optimizer.artifact_sha256.v1:{digest.hexdigest()}"
        if actual != trace["digest"]:
            raise ModelCompileError(
                f"normal execution trace {name!r} failed digest validation"
            )
        trace_paths.add(trace["path"])
    return trace_paths


def _verify_fixture_artifacts(
    plan: BenchmarkPlan,
    adapter: NormalExecutionAdapter,
    *,
    cancel_requested: Callable[[], bool] | None = None,
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
        _checkpoint(cancel_requested)
        digest = sha256()
        captured = bytearray()
        for chunk in adapter.iter_fixture_artifact(
            relative_path,
            candidate_id=plan.candidate_id,
        ):
            _checkpoint(cancel_requested)
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
        actual = f"nerve.optimizer.artifact_sha256.v1:{digest.hexdigest()}"
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
    if isinstance(value, bool) or not isinstance(value, int) or value != expected_limit:
        raise ModelCompileError(
            "benchmark output allowance does not match its immutable evidence"
        )


def _role_order(index: int) -> tuple[str, str]:
    return ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")


def _checkpoint(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ModelCompileCancelled("matched candidate benchmark cancelled")
