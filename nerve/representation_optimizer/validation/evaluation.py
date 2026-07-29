from __future__ import annotations

from nerve.compilation import Json, ModelCompileError
from nerve.representation_optimizer.contracts import (
    PREBENCHMARK_RECORD_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    ContractDocument,
    contract_digest,
)
from nerve.representation_optimizer.validation.contracts import (
    PREBENCHMARK_RECORD_SCHEMA as PREBENCHMARK_SCHEMA,
    VALIDATION_FUNNEL_STAGE_NAMES,
    ProofResult,
    ValidationPlan,
    ValidationRun,
    prebenchmark_record_id,
    validation_record_id,
)


def build_prebenchmark_record(
    *,
    plan: ValidationPlan,
    static_validation: Json,
    proof_results: tuple[ProofResult, ...],
    sanity_run: ValidationRun | None,
    failure_reason: str | None,
) -> ContractDocument:
    if PREBENCHMARK_RECORD_SCHEMA != PREBENCHMARK_SCHEMA:
        raise RuntimeError("prebenchmark schema registry is inconsistent")
    behavioral = plan.behavioral_contract
    has_proof_obligations = bool(behavioral["proof_obligations"])
    proof_status = (
        "not_applicable"
        if not has_proof_obligations
        else (
            "passed"
            if proof_results
            and all(result.status == "proven" for result in proof_results)
            else "failed"
        )
    )
    static_status = str(static_validation["status"])
    sanity_status = (
        "not_run"
        if sanity_run is None
        else (
            "passed"
            if sanity_run.status == "completed"
            else "failed"
        )
    )
    status = (
        "passed"
        if (
            static_status == "passed"
            and proof_status in {"passed", "not_applicable"}
            and sanity_status == "passed"
        )
        else "failed"
    )
    stages = [
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[0],
            static_status,
            evidence_digests=(),
            metrics={
                "artifact_count": static_validation["artifact_count"]
            },
            reason=(
                None
                if static_status == "passed"
                else failure_reason or "static validation failed"
            ),
        ),
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[1],
            proof_status,
            evidence_digests=tuple(
                contract_digest(result.to_json())
                for result in proof_results
            ),
            metrics={
                "obligation_count": len(proof_results),
                "proven_count": sum(
                    result.status == "proven"
                    for result in proof_results
                ),
            },
            reason=(
                "candidate declares no algebraic proof obligations"
                if proof_status == "not_applicable"
                else (
                    None
                    if proof_status == "passed"
                    else failure_reason
                    or "one or more exact proof obligations were not proven"
                )
            ),
        ),
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[2],
            sanity_status,
            evidence_digests=(
                ()
                if sanity_run is None
                else (contract_digest(sanity_run.to_json()),)
            ),
            metrics=(
                {}
                if sanity_run is None
                else summarize_validation_run(sanity_run)
            ),
            reason=(
                None
                if sanity_status == "passed"
                else (
                    failure_reason
                    or (
                        "sanity validation was not run because an earlier "
                        "gate failed"
                        if sanity_status == "not_run"
                        else "sanity validation exceeded its behavioral contract"
                    )
                )
            ),
        ),
    ]
    plan_document = plan.to_json()
    document = {
        "schema": PREBENCHMARK_RECORD_SCHEMA,
        "prebenchmark_id": "",
        "candidate_id": plan.candidate_id,
        "source_contract_digests": list(
            plan_document["source_contract_digests"]
        ),
        "validation_plan_digest": contract_digest(plan_document),
        "construction_record_digest": plan_document[
            "construction_record_digest"
        ],
        "static_validation": dict(static_validation),
        "proof_results": [
            result.to_json()
            for result in sorted(
                proof_results,
                key=lambda result: result.to_json()["obligation"],
            )
        ],
        "sanity_run_digest": (
            None
            if sanity_run is None
            else contract_digest(sanity_run.to_json())
        ),
        "stages": stages,
        "counterexamples": _collected_counterexamples(
            plan,
            (() if sanity_run is None else (sanity_run,)),
            initial=tuple(plan_document["counterexamples"]),
        ),
        "status": status,
    }
    document["prebenchmark_id"] = prebenchmark_record_id(document)
    return ContractDocument.from_json(
        document,
        expected_schema=PREBENCHMARK_RECORD_SCHEMA,
    )


def build_validation_record(
    *,
    plan: ValidationPlan,
    prebenchmark_record: ContractDocument,
    benchmark_record: ContractDocument,
    runs: tuple[ValidationRun, ...],
    product_performance: Json,
    failure_reason: str | None,
) -> ContractDocument:
    prebenchmark = prebenchmark_record.to_json()
    benchmark = benchmark_record.to_json()
    run_by_stage = {
        run.to_json()["stage"]: run for run in runs
    }
    if (
        prebenchmark["candidate_id"] != plan.candidate_id
        or benchmark["candidate_id"] != plan.candidate_id
    ):
        raise ModelCompileError(
            "validation evidence belongs to different candidates"
        )
    benchmark_status = (
        "passed"
        if benchmark["decision"] == "materially_faster"
        else "failed"
    )
    local = run_by_stage.get("full_local")
    local_status = (
        "not_run"
        if local is None
        else ("passed" if local.status == "completed" else "failed")
    )
    whole = run_by_stage.get("whole_model")
    whole_status = (
        "not_run"
        if whole is None
        else ("passed" if whole.status == "completed" else "failed")
    )
    product_status = str(product_performance["status"])
    status = (
        "passed"
        if (
            prebenchmark["status"] == "passed"
            and benchmark_status == "passed"
            and local_status == "passed"
            and whole_status == "passed"
            and product_status == "passed"
        )
        else "failed"
    )
    stages = [
        *prebenchmark["stages"],
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[3],
            benchmark_status,
            evidence_digests=(benchmark_record.digest,),
            metrics={
                "decision": benchmark["decision"],
                "workload_count": len(benchmark["workloads"]),
            },
            reason=(
                None
                if benchmark_status == "passed"
                else failure_reason
                or "candidate was not materially faster under matched conditions"
            ),
        ),
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[4],
            local_status,
            evidence_digests=(
                ()
                if local is None
                else (contract_digest(local.to_json()),)
            ),
            metrics=(
                {} if local is None else summarize_validation_run(local)
            ),
            reason=(
                None
                if local_status == "passed"
                else failure_reason
                or (
                    "full local validation was skipped because the "
                    "performance gate failed"
                    if local_status == "not_run"
                    else "full local behavior exceeded its error contract"
                )
            ),
        ),
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[5],
            whole_status,
            evidence_digests=(
                ()
                if whole is None
                else (contract_digest(whole.to_json()),)
            ),
            metrics=(
                {} if whole is None else summarize_validation_run(whole)
            ),
            reason=(
                None
                if whole_status == "passed"
                else failure_reason
                or (
                    "whole-model validation was skipped because an earlier "
                    "gate failed"
                    if whole_status == "not_run"
                    else "whole-model free-running behavior exceeded its "
                    "error contract"
                )
            ),
        ),
        _stage(
            VALIDATION_FUNNEL_STAGE_NAMES[6],
            product_status,
            evidence_digests=(
                ()
                if whole is None
                else (contract_digest(whole.to_json()),)
            ),
            metrics=dict(product_performance["metrics"]),
            reason=product_performance["reason"],
        ),
    ]
    plan_document = plan.to_json()
    document = {
        "schema": VALIDATION_RECORD_SCHEMA,
        "validation_id": "",
        "candidate_id": plan.candidate_id,
        "source_contract_digests": list(
            plan_document["source_contract_digests"]
        ),
        "behavioral_contract": plan.behavioral_contract,
        "validation_plan_digest": contract_digest(plan_document),
        "construction_record_digest": plan_document[
            "construction_record_digest"
        ],
        "prebenchmark_record_digest": prebenchmark_record.digest,
        "benchmark_record_digest": benchmark_record.digest,
        "runs": [
            {
                "stage": run.to_json()["stage"],
                "run_digest": contract_digest(run.to_json()),
            }
            for run in runs
        ],
        "stages": stages,
        "counterexamples": _collected_counterexamples(
            plan,
            runs,
            initial=tuple(prebenchmark["counterexamples"]),
        ),
        "status": status,
    }
    document["validation_id"] = validation_record_id(document)
    return ContractDocument.from_json(
        document,
        expected_schema=VALIDATION_RECORD_SCHEMA,
    )


def summarize_validation_run(run: ValidationRun) -> Json:
    document = run.to_json()
    observations = document["observations"]
    errors: dict[str, float] = {}
    seeds: set[int] = set()
    minimum_steps: int | None = None
    for observation in observations:
        seeds.add(observation["seed"])
        minimum_steps = min(
            (
                observation["reference"]["steps"],
                observation["candidate"]["steps"],
                (
                    minimum_steps
                    if minimum_steps is not None
                    else observation["reference"]["steps"]
                ),
            )
        )
        for metric in observation["metrics"]:
            errors[metric["name"]] = max(
                errors.get(metric["name"], 0.0),
                float(metric["error"]),
            )
    return {
        "observation_count": len(observations),
        "fixed_seed_count": len(seeds),
        "minimum_executed_steps": minimum_steps or 0,
        "maximum_observed_error": dict(sorted(errors.items())),
        "host_elapsed_ns": sum(
            timing["duration_ns"]
            for timing in document["host_elapsed_ns"]
        ),
    }


def _stage(
    name: str,
    status: str,
    *,
    evidence_digests: tuple[str, ...],
    metrics: Json,
    reason: str | None,
) -> Json:
    return {
        "name": name,
        "status": status,
        "evidence_digests": sorted(set(evidence_digests)),
        "metrics": metrics,
        "artifacts": [],
        "reason": reason,
    }


def _collected_counterexamples(
    plan: ValidationPlan,
    runs: tuple[ValidationRun, ...],
    *,
    initial: tuple[Json, ...],
) -> list[Json]:
    by_path = {
        str(reference["path"]): dict(reference)
        for reference in initial
    }
    checks = {
        check["check_id"]: check
        for stage in ("sanity", "full_local", "whole_model")
        for check in plan.checks_for_stage(stage)
    }
    for run in runs:
        document = run.to_json()
        if document["status"] == "completed" or not document["observations"]:
            continue
        failed_observation = document["observations"][-1]
        check = checks[failed_observation["check_id"]]
        reference = check["input"]
        by_path[str(reference["path"])] = dict(reference)
    return [by_path[path] for path in sorted(by_path)]
